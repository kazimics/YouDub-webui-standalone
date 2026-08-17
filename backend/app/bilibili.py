from __future__ import annotations

import base64
import math
import re
import time
from http.cookiejar import MozillaCookieJar
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests

from . import runtime_security

PREUPLOAD_URL = "https://member.bilibili.com/preupload"
COVER_UP_URL = "https://member.bilibili.com/x/vu/web/cover/up"
DRAFT_URL = "https://member.bilibili.com/x/vupre/web/draft/add"

CHUNK_RETRY = 3
CHUNK_RETRY_DELAY = 3.0
REQUEST_TIMEOUT = 30
UPLOAD_TIMEOUT = 120

BILIBILI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_TAG = "搬运"
MAX_TITLE_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 2000
MAX_TAGS = 10
COVER_ASPECT_RATIO = 16.0 / 10.0

ERROR_MESSAGES: dict[int, str] = {
    -101: "未登录或登录已过期，请重新导出 B 站 Cookie（SESSDATA）",
    -111: "CSRF 校验失败，bili_jct 与 SESSDATA 不匹配，请重新导出 B 站 Cookie",
    -400: "请求参数错误",
    21070: "稿件正在审核中，无法编辑",
    53019: "标题过长（最多 80 字）",
}


class BilibiliError(Exception):
    """B 站接口错误，携带业务错误码。"""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> "BilibiliError":
        code = data.get("code", data.get("OK", -1))
        raw = data.get("message") or data.get("msg") or ""
        message = ERROR_MESSAGES.get(int(code), raw) or "B 站接口返回错误"
        return cls(int(code), message)


def parse_cookie_file(path: Path) -> tuple[str, str]:
    """解析 B 站 Cookie 文件，返回 (SESSDATA, bili_jct)。

    优先按 Netscape（Mozilla）格式解析，失败时回退到 k=v 逐行解析。
    """
    sessdata: str | None = None
    bili_jct: str | None = None
    try:
        jar = MozillaCookieJar()
        jar.load(str(path), ignore_discard=True, ignore_expires=True)
        for cookie in jar:
            if cookie.domain and "bilibili.com" not in cookie.domain:
                continue
            if cookie.name == "SESSDATA":
                sessdata = cookie.value
            elif cookie.name == "bili_jct":
                bili_jct = cookie.value
    except Exception:
        pass

    if not (sessdata and bili_jct):
        values: dict[str, str] = {}
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                values[parts[5]] = parts[6]
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
        sessdata = sessdata or values.get("SESSDATA")
        bili_jct = bili_jct or values.get("bili_jct")

    if not sessdata or not bili_jct:
        missing = []
        if not sessdata:
            missing.append("SESSDATA")
        if not bili_jct:
            missing.append("bili_jct")
        raise BilibiliError(-1, f"B 站 Cookie 中缺少 {', '.join(missing)}，请重新导出")
    return unquote(sessdata), bili_jct


def read_bilibili_credentials(path: Path) -> tuple[str, str]:
    """读取并校验 B 站 Cookie 文件，返回 (SESSDATA, bili_jct)。"""
    runtime_security.secure_existing_file(path, required=False)
    if not Path(path).is_file():
        raise BilibiliError(-1, "尚未配置 B 站 Cookie，请先在“设置”中粘贴导出的 B 站 Cookie")
    return parse_cookie_file(Path(path))


def build_session(sessdata: str, bili_jct: str = "") -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": BILIBILI_USER_AGENT,
            "Referer": "https://member.bilibili.com/",
            "Origin": "https://member.bilibili.com",
        }
    )
    session.cookies.set("SESSDATA", sessdata, domain=".bilibili.com", path="/")
    if bili_jct:
        session.cookies.set("bili_jct", bili_jct, domain=".bilibili.com", path="/")
    return session


def prepare_upload(session: requests.Session, filename: str, size: int) -> dict[str, Any]:
    params = {
        "name": filename,
        "size": size,
        "r": "upos",
        "profile": "ugcupos/bup",
        "ssl": 1,
        "version": "2.10.4.0",
        "build": "100000",
    }
    response = session.get(PREUPLOAD_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if data.get("OK") != 1:
        raise BilibiliError.from_response(data)
    return data


def _chunk_etag(response: requests.Response) -> str:
    return response.headers.get("ETag", "etag").strip('"')


def upload_video(
    session: requests.Session,
    video_path: Path,
    auth: str,
    endpoint: str,
    upos_uri: str,
    chunk_size: int,
    biz_id: int,
) -> tuple[str, int]:
    """上传视频文件并合并分片，返回 (B 站服务器端无后缀文件名, biz_id)。

    filename 必须取自 preupload 返回的 upos_uri（服务器端随机文件名），
    不能使用本地文件名，否则生成的草稿会因找不到视频而无效。
    """
    filename = video_path.name
    total_size = video_path.stat().st_size
    url = f"https:{endpoint}/{upos_uri.replace('upos://', '')}"
    headers = {"X-Upos-Auth": auth}

    init_response = session.post(f"{url}?uploads&output=json", headers=headers, timeout=REQUEST_TIMEOUT)
    init_response.raise_for_status()
    upload_id = init_response.json()["upload_id"]

    chunks = max(1, math.ceil(total_size / chunk_size))
    parts: list[dict[str, Any]] = []
    with video_path.open("rb") as file:
        for chunk_index in range(chunks):
            data = file.read(chunk_size)
            offset = chunk_index * chunk_size
            params = {
                "partNumber": chunk_index + 1,
                "uploadId": upload_id,
                "chunk": chunk_index,
                "chunks": chunks,
                "size": len(data),
                "start": offset,
                "end": offset + len(data),
                "total": total_size,
            }
            etag = None
            for attempt in range(1, CHUNK_RETRY + 1):
                try:
                    response = session.put(
                        url,
                        params=params,
                        data=data,
                        headers=headers,
                        timeout=UPLOAD_TIMEOUT,
                    )
                except requests.RequestException:
                    response = None
                if response is not None and response.status_code == 200:
                    etag = _chunk_etag(response)
                    if etag:
                        break
                if attempt < CHUNK_RETRY:
                    time.sleep(CHUNK_RETRY_DELAY)
            if not etag:
                raise BilibiliError(-1, f"视频分片 {chunk_index + 1}/{chunks} 上传失败")
            parts.append({"partNumber": chunk_index + 1, "eTag": etag})

    finalize_params = {
        "name": filename,
        "uploadId": upload_id,
        "biz_id": biz_id,
        "output": "json",
        "profile": "ugcupos/bup",
    }
    finalize_response = session.post(
        url,
        params=finalize_params,
        json={"parts": parts},
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    finalize_response.raise_for_status()
    result = finalize_response.json()
    if result.get("OK") != 1:
        raise BilibiliError.from_response(result)
    bili_filename = Path(upos_uri.replace('upos://', '')).stem
    return bili_filename, int(biz_id)


def upload_cover(session: requests.Session, csrf: str, image_path: Path) -> str:
    """上传封面（裁为 16:10），返回 B 站 CDN 封面 URL。"""
    from PIL import Image

    with Image.open(image_path) as image:
        xsize, ysize = image.size
        if xsize / ysize > COVER_ASPECT_RATIO:
            delta = xsize - ysize * COVER_ASPECT_RATIO
            region = image.crop((delta / 2, 0, xsize - delta / 2, ysize))
        else:
            delta = ysize - xsize / COVER_ASPECT_RATIO
            region = image.crop((0, delta / 2, xsize, ysize - delta / 2))
        buffered = BytesIO()
        region.save(buffered, format="JPEG")

    data = b"data:image/jpeg;base64," + base64.b64encode(buffered.getvalue())
    response = session.post(COVER_UP_URL, data={"cover": data, "csrf": csrf}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 0 or not result.get("data"):
        raise BilibiliError.from_response(result)
    return result["data"]["url"]


def normalize_tags(tag: str) -> str:
    tags = [part.strip() for part in tag.replace("，", ",").split(",") if part.strip()]
    if not tags:
        tags = [DEFAULT_TAG]
    return ",".join(tags[:MAX_TAGS])


def build_draft_payload(
    *,
    csrf: str,
    title: str,
    tid: int,
    tag: str,
    description: str,
    filename: str,
    cid: int,
    cover: str,
    source: str,
) -> dict[str, Any]:
    """构建 B 站草稿投稿 payload（转载，copyright=2）。"""
    return {
        "videos": [
            {
                "filename": filename,
                "title": title[:MAX_TITLE_LENGTH],
                "desc": "",
                "cid": cid,
                "is_4k": False,
                "is_8k": False,
                "is_hdr": False,
            }
        ],
        "cover": cover,
        "cover43": "",
        "title": title[:MAX_TITLE_LENGTH],
        "copyright": 2,
        "tid": int(tid),
        "tag": normalize_tags(tag),
        "human_type2": 0,
        "topic_id": 0,
        "mission_id": 0,
        "topic_name": "",
        "topic_from": "",
        "desc_format_id": 9999,
        "desc": description[:MAX_DESCRIPTION_LENGTH],
        "recreate": 0,
        "dynamic": "",
        "interactive": 0,
        "act_reserve_create": 0,
        "no_disturbance": 0,
        "no_reprint": 1,
        "subtitle": {"open": 0, "lan": ""},
        "dolby": 0,
        "lossless_music": 0,
        "up_selection_reply": False,
        "up_close_reply": False,
        "up_close_danmu": False,
        "web_os": 3,
        "csrf": csrf,
        "source": source,
    }


def create_draft(session: requests.Session, payload: dict[str, Any]) -> int:
    """提交草稿，返回 draft_id（新版接口不返回时返回 0）。"""
    cookies = getattr(session, "cookies", None)
    csrf = (cookies.get("bili_jct") if cookies is not None else None) or payload.get("csrf") or ""
    params = {"csrf": csrf, "t": int(time.time() * 1000)}
    response = session.post(DRAFT_URL, json=payload, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 0:
        raise BilibiliError.from_response(result)
    data = result.get("data") or {}
    return int(data.get("draft_id") or 0)

QR_GENERATE_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
QR_POLL_URL = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
QR_EXPIRE_SECONDS = 180

QR_LOGIN_COOKIES = ("SESSDATA", "bili_jct", "DedeUserID", "DedeUserID__ckMd5")

# 以 qrcode_key 为键暂存登录会话（含创建时间），轮询复用同一会话；过期后清理。
_qr_sessions: dict[str, tuple[requests.Session, float]] = {}


def _qr_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": BILIBILI_USER_AGENT,
            "Referer": "https://www.bilibili.com/",
        }
    )
    return session


def generate_qr_code() -> tuple[str, str]:
    """调用 B 站二维码生成接口，返回 (登录链接 url, qrcode_key)。"""
    session = _qr_session()
    response = session.get(QR_GENERATE_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0 or not data.get("data"):
        raise BilibiliError.from_response(data)
    url = data["data"].get("url") or ""
    qrcode_key = data["data"].get("qrcode_key") or ""
    if not url or not qrcode_key:
        raise BilibiliError(-1, "B 站未返回二维码数据")
    now = time.monotonic()
    for stale_key in [
        key
        for key, (_, created_at) in _qr_sessions.items()
        if now - created_at > QR_EXPIRE_SECONDS
    ]:
        _qr_sessions.pop(stale_key, None)
    _qr_sessions[qrcode_key] = (session, now)
    return url, qrcode_key


def render_qr_data_uri(content: str) -> str:
    """把二维码内容渲染为 base64 PNG data URI（qrcode 延迟导入）。"""
    import qrcode

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(content)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_login_cookies(session: requests.Session) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for name in QR_LOGIN_COOKIES:
        value = session.cookies.get(name)
        if value:
            cookies[name] = value
    return cookies


def poll_qr_login(qrcode_key: str) -> dict[str, Any]:
    """轮询扫码状态，成功时返回含 SESSDATA/bili_jct 的 cookie 字典。"""
    entry = _qr_sessions.get(qrcode_key)
    session = entry[0] if entry is not None else _qr_session()
    response = session.get(
        QR_POLL_URL, params={"qrcode_key": qrcode_key}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        raise BilibiliError.from_response(data)
    inner = data.get("data") or {}
    status_code = inner.get("code")
    if status_code == 0:
        _qr_sessions.pop(qrcode_key, None)
        cookies = _extract_login_cookies(session)
        if not cookies.get("SESSDATA") or not cookies.get("bili_jct"):
            raise BilibiliError(-1, "扫码登录成功但未获取到完整 Cookie，请重试")
        return {"status": "success", **cookies}
    if status_code == 86090:
        return {"status": "scanned"}
    if status_code == 86038:
        _qr_sessions.pop(qrcode_key, None)
        return {"status": "expired"}
    return {"status": "pending"}


def format_cookie_file(cookies: dict[str, str]) -> str:
    """把扫码登录得到的 cookie 格式化为 Netscape 格式文本。"""
    lines = ["# Netscape HTTP Cookie File", ""]
    for name in QR_LOGIN_COOKIES:
        value = cookies.get(name)
        if value:
            lines.append(f".bilibili.com\tTRUE\t/\tTRUE\t0\t{name}\t{value}")
    return "\n".join(lines) + "\n"

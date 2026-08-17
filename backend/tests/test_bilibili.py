from __future__ import annotations

from pathlib import Path

import pytest
import requests

from backend.app import bilibili, bilibili_uploader, database, main
from backend.app.bilibili import BilibiliError
from backend.tests.test_settings_and_api import authenticated_client, configure_tmp_runtime


class FakeResponse:
    def __init__(self, json_data=None, status_code=200, headers=None, text=""):
        self._json = json_data
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json


def test_parse_netscape_cookie_file(tmp_path):
    cookie_file = tmp_path / "bilibili.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n"
        ".bilibili.com\tTRUE\t/\tTRUE\t0\tSESSDATA\tabc%2B123\n"
        ".bilibili.com\tTRUE\t/\tTRUE\t0\tbili_jct\tcsrf_token\n",
        encoding="utf-8",
    )
    sessdata, bili_jct = bilibili.parse_cookie_file(cookie_file)
    assert sessdata == "abc+123"
    assert bili_jct == "csrf_token"


def test_parse_cookie_kv_fallback(tmp_path):
    cookie_file = tmp_path / "bilibili.txt"
    cookie_file.write_text(
        "SESSDATA=plain-value\nbili_jct=token2\n",
        encoding="utf-8",
    )
    sessdata, bili_jct = bilibili.parse_cookie_file(cookie_file)
    assert sessdata == "plain-value"
    assert bili_jct == "token2"


def test_parse_cookie_missing_fields(tmp_path):
    cookie_file = tmp_path / "bilibili.txt"
    cookie_file.write_text("SESSDATA=only-one\n", encoding="utf-8")
    with pytest.raises(BilibiliError) as exc:
        bilibili.parse_cookie_file(cookie_file)
    assert "bili_jct" in str(exc.value)


def test_read_bilibili_credentials_missing_file(tmp_path):
    with pytest.raises(BilibiliError) as exc:
        bilibili.read_bilibili_credentials(tmp_path / "nope.txt")
    assert "尚未配置" in str(exc.value)


def test_prepare_upload_rejects_ok_not_one():
    session = _session_with_get(
        FakeResponse(json_data={"OK": -101, "message": "Not logged in"})
    )
    with pytest.raises(BilibiliError) as exc:
        bilibili.prepare_upload(session, "a.mp4", 10)
    assert exc.value.code == -101


def test_prepare_upload_returns_data():
    data = {"OK": 1, "auth": "auth-token", "endpoint": "//upos.example", "upos_uri": "upos://bucket/x", "biz_id": 99}
    session = _session_with_get(FakeResponse(json_data=data))
    assert bilibili.prepare_upload(session, "a.mp4", 10) == data


def test_upload_video_chunks_and_merges(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x" * 10)
    session = MagicSession()
    session.put_responses = [
        FakeResponse(status_code=200, headers={"ETag": "\"etag-1\""})
    ] * 3
    session.post_responses = [
        FakeResponse(json_data={"upload_id": "upload-1"}),
        FakeResponse(json_data={"OK": 1}),
    ]
    result = bilibili.upload_video(
        session, video, auth="auth", endpoint="//upos.example", upos_uri="upos://ugcever/n260817abc123.mp4",
        chunk_size=4, biz_id=7,
    )
    assert result == ("n260817abc123", 7)
    assert session.put_calls == 3
    assert session.post_calls == 2
    finalize_body = session.post_bodies[-1]["json"]
    assert finalize_body["parts"][0]["eTag"] == "etag-1"


def test_upload_video_retries_chunk(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"y" * 4)
    sleeps = []
    monkeypatch.setattr(bilibili.time, "sleep", sleeps.append)
    session = MagicSession()
    fail = FakeResponse(status_code=500)
    ok = FakeResponse(status_code=200, headers={"ETag": "etag-2"})
    session.put_responses = [fail, fail, ok]
    session.post_responses = [
        FakeResponse(json_data={"upload_id": "upload-1"}),
        FakeResponse(json_data={"OK": 1}),
    ]
    result = bilibili.upload_video(
        session, video, auth="auth", endpoint="//upos.example", upos_uri="upos://ugcever/n260817abc123.mp4",
        chunk_size=4, biz_id=7,
    )
    assert result == ("n260817abc123", 7)
    assert session.put_calls == 3
    assert sleeps == [3.0, 3.0]


def test_upload_video_fails_after_retries(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"z" * 4)
    session = MagicSession()
    session.put_responses = [FakeResponse(status_code=500)] * 3
    session.post_responses = [FakeResponse(json_data={"upload_id": "upload-1"})]
    with pytest.raises(BilibiliError) as exc:
        bilibili.upload_video(
            session, video, auth="auth", endpoint="//upos.example", upos_uri="upos://ugcever/n260817abc123.mp4",
            chunk_size=4, biz_id=7,
        )
    assert "上传失败" in str(exc.value)


def test_upload_cover_crops_and_returns_url(tmp_path):
    from PIL import Image

    image_path = tmp_path / "cover.png"
    Image.new("RGB", (160, 80), (10, 20, 30)).save(image_path)
    session = MagicSession()
    session.post_responses = [FakeResponse(json_data={"code": 0, "data": {"url": "//i0.hdslb.com/cover.jpg"}})]
    url = bilibili.upload_cover(session, "csrf-token", image_path)
    assert url == "//i0.hdslb.com/cover.jpg"
    body = session.post_bodies[0]["data"]
    assert body["csrf"] == "csrf-token"
    assert body["cover"].startswith(b"data:image/jpeg;base64,")


def test_normalize_tags_default_and_limit():
    assert bilibili.normalize_tags("") == "搬运"
    assert bilibili.normalize_tags("科技，生活") == "科技,生活"
    tags = bilibili.normalize_tags(",".join(f"t{i}" for i in range(12)))
    assert len(tags.split(",")) == 10


def test_build_draft_payload_truncates_and_sets_source():
    payload = bilibili.build_draft_payload(
        csrf="csrf-token",
        title="长" * 200,
        tid=171,
        tag="",
        description="d" * 5000,
        filename="clip",
        cid=42,
        cover="//cover.jpg",
        source="https://example.com/video",
    )
    assert len(payload["title"]) == 80
    assert len(payload["desc"]) == 2000
    assert payload["copyright"] == 2
    assert payload["source"] == "https://example.com/video"
    assert payload["csrf"] == "csrf-token"
    assert payload["videos"][0]["cid"] == 42
    assert payload["videos"][0]["filename"] == "clip"


def test_create_draft_returns_draft_id():
    session = MagicSession()
    session.post_responses = [FakeResponse(json_data={"code": 0, "data": {"draft_id": 888}})]
    assert bilibili.create_draft(session, {}) == 888


def test_create_draft_raises_business_error():
    session = MagicSession()
    session.post_responses = [FakeResponse(json_data={"code": -101, "message": "expired"})]
    with pytest.raises(BilibiliError) as exc:
        bilibili.create_draft(session, {})
    assert "未登录" in str(exc.value)


def _session_with_get(response):
    session = MagicSession()
    session.get_responses = [response]
    return session


class MagicSession:
    def __init__(self):
        self.get_responses = []
        self.put_responses = []
        self.post_responses = []
        self.put_calls = 0
        self.post_calls = 0
        self.get_calls = 0
        self.put_bodies = []
        self.post_bodies = []
        self.get_bodies = []

    def get(self, url, **kwargs):
        self.get_calls += 1
        self.get_bodies.append(kwargs)
        return self.get_responses.pop(0)

    def put(self, url, **kwargs):
        self.put_calls += 1
        self.put_bodies.append(kwargs)
        return self.put_responses.pop(0)

    def post(self, url, **kwargs):
        self.post_calls += 1
        self.post_bodies.append(kwargs)
        return self.post_responses.pop(0)


def _make_succeeded_task(tmp_path, with_final=True):
    task_id = database.create_task("https://example.com/bili-draft", task_id="bilidraft")
    database.update_task(
        task_id,
        status="succeeded",
        title="Original title",
        translated_title="Translated title",
        translated_description="Desc line",
    )
    final_path = tmp_path / "final.mp4"
    if with_final:
        final_path.write_bytes(b"video-data")
        database.update_task(task_id, final_video_path=str(final_path))
    return task_id


def test_bilibili_draft_api_404(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = authenticated_client()
    response = client.post("/api/tasks/missing/bilibili/draft", json={"title": "x"})
    assert response.status_code == 404


def test_bilibili_draft_api_409_when_not_succeeded(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task("https://example.com/bili-queued", task_id="biliq")
    database.update_task(task_id, status="queued")
    client = authenticated_client()
    response = client.post(f"/api/tasks/{task_id}/bilibili/draft", json={"title": "x"})
    assert response.status_code == 409


def test_bilibili_draft_api_409_when_no_final_video(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = _make_succeeded_task(tmp_path, with_final=False)
    client = authenticated_client()
    response = client.post(f"/api/tasks/{task_id}/bilibili/draft", json={"title": "x"})
    assert response.status_code == 409


def test_bilibili_draft_api_422_when_title_empty(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = _make_succeeded_task(tmp_path)
    database.update_task(task_id, title="", translated_title="")
    client = authenticated_client()
    response = client.post(f"/api/tasks/{task_id}/bilibili/draft", json={"title": "   "})
    assert response.status_code == 422


def test_bilibili_draft_api_502_when_cookie_missing(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = _make_succeeded_task(tmp_path)
    client = authenticated_client()
    response = client.post(f"/api/tasks/{task_id}/bilibili/draft", json={"title": "x"})
    assert response.status_code == 502
    assert "尚未配置" in response.json()["detail"]


def test_bilibili_draft_api_success(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = _make_succeeded_task(tmp_path)
    session_path = tmp_path / "session"
    media_dir = session_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_path = media_dir / "thumb.jpg"
    thumbnail_path.write_bytes(b"img")
    database.update_task(task_id, session_path=str(session_path), thumbnail_path=str(thumbnail_path))
    cookie_file = tmp_path / "cookies" / "bilibili.txt"
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.write_text("SESSDATA=sess\nbili_jct=csrf\n", encoding="utf-8")
    monkeypatch.setattr(
        bilibili_uploader.bilibili,
        "prepare_upload",
        lambda *a, **k: {
            "auth": "auth",
            "endpoint": "//upos.example",
            "upos_uri": "upos://bucket/x",
            "chunk_size": 1024,
            "biz_id": 9,
        },
    )
    monkeypatch.setattr(bilibili_uploader.bilibili, "upload_video", lambda *a, **k: ("final", 42))
    monkeypatch.setattr(bilibili_uploader.bilibili, "upload_cover", lambda *a, **k: "//cover.jpg")
    monkeypatch.setattr(bilibili_uploader.bilibili, "create_draft", lambda *a, **k: 999)
    client = authenticated_client()
    response = client.post(
        f"/api/tasks/{task_id}/bilibili/draft",
        json={"title": "自定义标题", "tid": 138, "tag": "科技,生活", "description": "desc"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["draft_id"] == 999
    assert body["title"] == "自定义标题"
    assert body["cover"] == "//cover.jpg"


def test_bilibili_cookie_api_roundtrip(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    client = authenticated_client()
    response = client.get("/api/cookies/bilibili")
    assert response.status_code == 200
    assert response.json()["exists"] is False

    response = client.post(
        "/api/cookies/bilibili",
        json={"content": "SESSDATA=abc\nbili_jct=def\n"},
    )
    assert response.status_code == 200
    assert response.json()["exists"] is True
    assert Path(main.BILIBILI_COOKIE_PATH).is_file()

    response = client.post("/api/cookies/bilibili", json={"content": ""})
    assert response.status_code == 200
    assert response.json()["exists"] is False

class CookieSession(MagicSession):
    """带 cookie jar 的假会话，用于扫码登录成功场景。"""

    def __init__(self, cookies=None):
        super().__init__()
        self.cookies = requests.cookies.RequestsCookieJar()
        for name, value in (cookies or {}).items():
            self.cookies.set(name, value, domain=".bilibili.com", path="/")


def _clear_qr_sessions():
    bilibili._qr_sessions.clear()


def test_generate_qr_code(monkeypatch):
    session = MagicSession()
    session.get_responses = [
        FakeResponse(
            json_data={
                "code": 0,
                "data": {
                    "url": "https://passport.bilibili.com/h5-app/passport/login/scan?navhide=1&qrcode_key=key123&from=",
                    "qrcode_key": "key123",
                },
            }
        )
    ]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    url, qrcode_key = bilibili.generate_qr_code()
    assert qrcode_key == "key123"
    assert "qrcode_key=key123" in url
    assert bilibili._qr_sessions["key123"][0] is session
    _clear_qr_sessions()


def test_generate_qr_code_raises_when_missing_data(monkeypatch):
    session = MagicSession()
    session.get_responses = [FakeResponse(json_data={"code": -101, "message": "expired"})]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    with pytest.raises(BilibiliError) as exc:
        bilibili.generate_qr_code()
    assert exc.value.code == -101


def test_render_qr_data_uri():
    data_uri = bilibili.render_qr_data_uri("https://example.com/scan?key=x")
    assert data_uri.startswith("data:image/png;base64,")
    import base64
    payload = data_uri.split(",", 1)[1]
    assert base64.b64decode(payload)[:8] == b"\x89PNG\r\n\x1a\n"


def test_poll_qr_login_pending(monkeypatch):
    session = MagicSession()
    session.get_responses = [FakeResponse(json_data={"code": 0, "data": {"code": 86101}})]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    assert bilibili.poll_qr_login("key") == {"status": "pending"}


def test_poll_qr_login_scanned(monkeypatch):
    session = MagicSession()
    session.get_responses = [FakeResponse(json_data={"code": 0, "data": {"code": 86090}})]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    assert bilibili.poll_qr_login("key") == {"status": "scanned"}


def test_poll_qr_login_expired(monkeypatch):
    session = MagicSession()
    session.get_responses = [FakeResponse(json_data={"code": 0, "data": {"code": 86038}})]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    assert bilibili.poll_qr_login("key") == {"status": "expired"}


def test_poll_qr_login_success_extracts_cookies(monkeypatch):
    session = CookieSession(
        cookies={"SESSDATA": "sess%2Bdata", "bili_jct": "jct", "DedeUserID": "123"}
    )
    session.get_responses = [FakeResponse(json_data={"code": 0, "data": {"code": 0}})]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    result = bilibili.poll_qr_login("key")
    assert result["status"] == "success"
    assert result["SESSDATA"] == "sess%2Bdata"
    assert result["bili_jct"] == "jct"
    assert result["DedeUserID"] == "123"
    assert "key" not in bilibili._qr_sessions


def test_poll_qr_login_success_missing_cookie_raises(monkeypatch):
    session = CookieSession(cookies={})
    session.get_responses = [FakeResponse(json_data={"code": 0, "data": {"code": 0}})]
    monkeypatch.setattr(bilibili, "_qr_session", lambda: session)
    with pytest.raises(BilibiliError) as exc:
        bilibili.poll_qr_login("key")
    assert "完整 Cookie" in str(exc.value)


def test_format_cookie_file_netscape():
    text = bilibili.format_cookie_file(
        {
            "SESSDATA": "s%2Bd",
            "bili_jct": "jct",
            "DedeUserID": "123",
            "DedeUserID__ckMd5": "md5",
        }
    )
    assert text.startswith("# Netscape HTTP Cookie File")
    assert ".bilibili.com\tTRUE\t/\tTRUE\t0\tSESSDATA\ts%2Bd" in text
    assert ".bilibili.com\tTRUE\t/\tTRUE\t0\tbili_jct\tjct" in text


def test_bilibili_qr_api_generate(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main.bilibili,
        "generate_qr_code",
        lambda: ("https://passport.bilibili.com/h5-app/passport/login/scan?navhide=1&qrcode_key=key1&from=", "key1"),
    )
    monkeypatch.setattr(main.bilibili, "render_qr_data_uri", lambda url: "data:image/png;base64,AAAA")
    client = authenticated_client()
    response = client.get("/api/cookies/bilibili/qr")
    assert response.status_code == 200
    body = response.json()
    assert body["qrcode_key"] == "key1"
    assert body["qr_image"] == "data:image/png;base64,AAAA"
    assert body["expires_in"] == 180


def test_bilibili_qr_api_poll_success_saves_cookie(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main.bilibili,
        "poll_qr_login",
        lambda key: {
            "status": "success",
            "SESSDATA": "s%2Bdata",
            "bili_jct": "jct",
            "DedeUserID": "1",
            "DedeUserID__ckMd5": "m",
        },
    )
    client = authenticated_client()
    response = client.post("/api/cookies/bilibili/qr/poll", json={"qrcode_key": "key1"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["exists"] is True
    cookie_file = Path(main.BILIBILI_COOKIE_PATH)
    assert cookie_file.is_file()
    content = cookie_file.read_text(encoding="utf-8")
    assert "SESSDATA" in content and "bili_jct" in content


def test_bilibili_qr_api_poll_pending(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(main.bilibili, "poll_qr_login", lambda key: {"status": "pending"})
    client = authenticated_client()
    response = client.post("/api/cookies/bilibili/qr/poll", json={"qrcode_key": "key1"})
    assert response.status_code == 200
    assert response.json() == {"status": "pending"}


def test_bilibili_qr_api_poll_expired(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(main.bilibili, "poll_qr_login", lambda key: {"status": "expired"})
    client = authenticated_client()
    response = client.post("/api/cookies/bilibili/qr/poll", json={"qrcode_key": "key1"})
    assert response.status_code == 200
    assert response.json() == {"status": "expired"}


def test_bilibili_qr_api_poll_error(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)

    def boom(key):
        raise BilibiliError(-1, "扫码登录成功但未获取到完整 Cookie，请重试")

    monkeypatch.setattr(main.bilibili, "poll_qr_login", boom)
    client = authenticated_client()
    response = client.post("/api/cookies/bilibili/qr/poll", json={"qrcode_key": "key1"})
    assert response.status_code == 502
    assert "完整 Cookie" in response.json()["detail"]


def test_bilibili_draft_stage_inserted_per_enabled_setting(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    enabled_id = database.create_task(
        "https://example.com/bili-stage-on", task_id="bili-stage-on"
    )
    disabled_id = database.create_task(
        "https://example.com/bili-stage-off",
        task_id="bili-stage-off",
        bilibili_draft_enabled=False,
    )

    enabled = database.get_task(enabled_id)
    disabled = database.get_task(disabled_id)
    assert enabled["bilibili_draft_enabled"] is True
    assert disabled["bilibili_draft_enabled"] is False
    enabled_stages = {stage["name"]: stage for stage in enabled["stages"]}
    disabled_stages = {stage["name"]: stage for stage in disabled["stages"]}
    assert enabled_stages["bilibili_draft"]["status"] == "pending"
    assert disabled_stages["bilibili_draft"]["status"] == "skipped"


def test_bilibili_auto_upload_succeeded_stage_and_keeps_task_succeeded(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = _make_succeeded_task(tmp_path)
    monkeypatch.setattr(
        bilibili_uploader,
        "submit_bilibili_draft",
        lambda *a, **k: {"draft_id": 123, "aid": 0, "title": "Auto", "cover": ""},
    )

    thread = bilibili_uploader.submit_bilibili_draft_async(task_id)
    thread.join(timeout=10)

    task = database.get_task(task_id)
    assert task["status"] == "succeeded"
    stage = {entry["name"]: entry for entry in task["stages"]}["bilibili_draft"]
    assert stage["status"] == "succeeded"
    assert stage["progress"] == 100
    assert "Draft created: 123" in stage["last_message"]
    log = database.log_path(task_id).read_text(encoding="utf-8")
    assert "[bilibili_draft] Draft created" in log


def test_bilibili_auto_upload_failure_keeps_task_succeeded(monkeypatch, tmp_path):
    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = _make_succeeded_task(tmp_path)
    monkeypatch.setattr(
        bilibili_uploader,
        "submit_bilibili_draft",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("upload boom")),
    )

    thread = bilibili_uploader.submit_bilibili_draft_async(task_id)
    thread.join(timeout=10)

    task = database.get_task(task_id)
    assert task["status"] == "succeeded"
    stage = {entry["name"]: entry for entry in task["stages"]}["bilibili_draft"]
    assert stage["status"] == "failed"
    assert stage["error_message"] == "upload boom"
    log = database.log_path(task_id).read_text(encoding="utf-8")
    assert "[bilibili_draft] Auto-upload failed: upload boom" in log

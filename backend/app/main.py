from __future__ import annotations

import logging
import os
import shutil
import stat
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import requests

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, SecretStr

from . import auth, bilibili, database, runtime_security, worker
from .adapters.local_subtitles import parse_srt, uploaded_subtitle_dir
from .adapters.local_video import remove_upload, uploaded_video_dir
from .adapters.openai_client import validate_openai_base_url
from .adapters.openai_translate import list_models as list_openai_models
from .config import BILIBILI_COOKIE_PATH, WORKFOLDER, YOUTUBE_COOKIE_PATH, ensure_runtime_dirs
from .pipeline import run_task
from .runtime_checks import validate_runtime_device
from .sanitize import sanitize_text
from .sources import detect_source
from .subtitle_style import (
    DEFAULT_CHINESE_FONT,
    DEFAULT_CHINESE_FONT_SIZE,
    DEFAULT_ENGLISH_FONT,
    DEFAULT_ENGLISH_FONT_SIZE,
    normalize_subtitle_style,
)
from .stage_reset import remove_stage_artifacts
from .stages import STAGE_NAMES
from .youtube import LOCAL_UPLOAD_DIRECTIONS, is_local_upload_url, validate_video_url

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv", ".wmv"}
ALLOWED_SUBTITLE_SUFFIXES = {".srt"}
LOCAL_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_LOCAL_UPLOAD_BYTES = int(os.getenv("LOCAL_UPLOAD_MAX_BYTES", str(4 * 1024 * 1024 * 1024)))
MAX_LOCAL_SUBTITLE_BYTES = int(os.getenv("LOCAL_SUBTITLE_MAX_BYTES", str(20 * 1024 * 1024)))

logger = logging.getLogger(__name__)

TaskListStatus = Literal["all", "queued", "running", "paused", "succeeded", "failed"]
TaskListExecutionMode = Literal["all", "auto", "manual"]
TaskListSort = Literal[
    "created_desc",
    "created_asc",
    "started_desc",
    "started_asc",
    "completed_desc",
    "completed_asc",
    "status_asc",
    "status_desc",
    "title_asc",
    "title_desc",
]


def mask_secret(value: str) -> str:
    if not value:
        return ""
    return "********"


class TaskCreate(BaseModel):
    url: str
    execution_mode: str = "auto"
    dubbing_enabled: bool = True
    bilibili_draft_enabled: bool = True
    subtitle_zh_font: str = DEFAULT_CHINESE_FONT
    subtitle_en_font: str = DEFAULT_ENGLISH_FONT
    subtitle_zh_font_size: int = DEFAULT_CHINESE_FONT_SIZE
    subtitle_en_font_size: int = DEFAULT_ENGLISH_FONT_SIZE


class ContinueTaskRequest(BaseModel):
    execution_mode: str | None = None


class YouTubeCookieUpdate(BaseModel):
    content: str


class BilibiliCookieUpdate(BaseModel):
    content: str


class BilibiliQrPollRequest(BaseModel):
    qrcode_key: str


class BilibiliDraftRequest(BaseModel):
    title: str = ""
    tid: int = 171
    tag: str = ""
    description: str = ""


class OpenAISettingsUpdate(BaseModel):
    base_url: str
    api_key: str = ""
    clear_api_key: bool = False
    model: str
    translate_concurrency: str = ""


class OpenAIModelsRequest(BaseModel):
    base_url: str = ""
    api_key: str = ""


class YtdlpSettingsUpdate(BaseModel):
    proxy_port: str = ""


class LoginRequest(BaseModel):
    password: SecretStr


def normalize_proxy_port(value: str) -> str:
    proxy_port = value.strip()
    if not proxy_port:
        return ""
    if not proxy_port.isdigit():
        raise HTTPException(status_code=422, detail="Proxy port must be numeric.")
    port = int(proxy_port)
    if port < 1 or port > 65535:
        raise HTTPException(status_code=422, detail="Proxy port must be between 1 and 65535.")
    return str(port)


def normalize_translate_concurrency(value: str) -> str:
    concurrency = value.strip()
    if not concurrency:
        return ""
    if not all("0" <= char <= "9" for char in concurrency):
        raise HTTPException(status_code=422, detail="Translate concurrency must be numeric.")
    workers = int(concurrency)
    if workers < 1 or workers > 200:
        raise HTTPException(
            status_code=422, detail="Translate concurrency must be between 1 and 200."
        )
    return concurrency


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_dirs()
    auth.validate_auth_configuration()
    database.init_db()
    database.delete_expired_auth_sessions(database.now_iso())
    database.backfill_titles_from_metadata()
    database.fail_stale_active_tasks()
    worker.start(run_task)
    yield


app = FastAPI(
    title="YouDub API",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.exception_handler(RequestValidationError)
async def redact_login_validation_error(
    request: Request, exc: RequestValidationError
) -> Response:
    if request.url.path == "/api/auth/login":
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials."},
            headers={"Cache-Control": "no-store"},
        )
    return await request_validation_exception_handler(request, exc)


DEFAULT_CORS_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|"
    r"127\.0\.0\.1|"
    r"\[::1\]"
    r"):3000$"
)


def cors_origins() -> list[str]:
    defaults = ["http://localhost:3000", "http://127.0.0.1:3000"]
    configured = os.getenv("CORS_ALLOW_ORIGINS", "")
    extra = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if "*" in extra:
        raise RuntimeError("CORS_ALLOW_ORIGINS cannot contain '*' when credentials are enabled.")
    return [*defaults, *extra]


def cors_origin_regex() -> str:
    configured = os.getenv("CORS_ALLOW_ORIGIN_REGEX", "").strip()
    return configured or DEFAULT_CORS_ORIGIN_REGEX


_cors_origins = cors_origins()
_cors_origin_regex = cors_origin_regex()

app.add_middleware(
    auth.AuthMiddleware,
    allowed_origins=_cors_origins,
    allowed_origin_regex=_cors_origin_regex,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _clear_replaced_login_cookie(
    response: Response,
    old_token: str,
    settings: auth.AuthSettings | None = None,
) -> None:
    if not old_token:
        return
    if settings is not None:
        auth.clear_session_cookie(response, settings)
        return
    response.delete_cookie(
        key=auth.SESSION_COOKIE_NAME,
        path=auth.SESSION_COOKIE_PATH,
        httponly=True,
    )


@app.post("/api/auth/login")
def login(payload: LoginRequest, request: Request) -> JSONResponse:
    old_token = request.cookies.get(auth.SESSION_COOKIE_NAME, "")
    auth.revoke_session_token(old_token)
    client_host = request.client.host if request.client else "unknown"

    try:
        settings = auth.validate_auth_configuration()
    except auth.AuthConfigurationError:
        response = JSONResponse(
            status_code=503,
            content={"detail": "Authentication is not configured."},
            headers={"Cache-Control": "no-store"},
        )
        _clear_replaced_login_cookie(response, old_token)
        return response

    rate_limit = auth.reserve_login_attempt(client_host)
    if not rate_limit.allowed:
        response = JSONResponse(
            status_code=429,
            content={"detail": "Too many login attempts."},
            headers={
                "Cache-Control": "no-store",
                "Retry-After": str(rate_limit.retry_after_seconds),
            },
        )
        _clear_replaced_login_cookie(response, old_token, settings)
        return response

    try:
        password_matches = auth.verify_password(
            payload.password.get_secret_value(), settings
        )
    except auth.AuthConfigurationError:
        auth.clear_login_attempts(client_host)
        response = JSONResponse(
            status_code=503,
            content={"detail": "Authentication is not configured."},
            headers={"Cache-Control": "no-store"},
        )
        _clear_replaced_login_cookie(response, old_token, settings)
        return response

    if not password_matches:
        response = JSONResponse(
            status_code=401,
            content={"detail": "Invalid credentials."},
            headers={"Cache-Control": "no-store"},
        )
        _clear_replaced_login_cookie(response, old_token, settings)
        return response

    auth.clear_login_attempts(client_host)
    token, session = auth.create_session(settings)
    response = JSONResponse(
        content={
            "authenticated": True,
            "csrf_token": session.csrf_token,
            "expires_at": session.expires_at,
        },
        headers={"Cache-Control": "no-store"},
    )
    auth.set_session_cookie(response, token, settings, session.expires_at)
    return response


@app.get("/api/auth/session")
def auth_session(request: Request) -> JSONResponse:
    session: auth.AuthenticatedSession = request.state.auth_session
    return JSONResponse(
        content={
            "authenticated": True,
            "csrf_token": session.csrf_token,
            "expires_at": session.expires_at,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/auth/logout", status_code=204)
def logout(request: Request) -> Response:
    session: auth.AuthenticatedSession = request.state.auth_session
    settings: auth.AuthSettings = request.state.auth_settings
    database.delete_auth_session(session.token_hash)
    response = Response(status_code=204, headers={"Cache-Control": "no-store"})
    auth.clear_session_cookie(response, settings)
    return response


def _ensure_runtime_ready() -> None:
    try:
        validate_runtime_device()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def normalize_execution_mode(value: str) -> str:
    try:
        return database.normalize_execution_mode(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/tasks", status_code=201)
def create_task(payload: TaskCreate) -> dict:
    try:
        validated_url = validate_video_url(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        subtitle_style = normalize_subtitle_style(
            payload.subtitle_zh_font,
            payload.subtitle_en_font,
            payload.subtitle_zh_font_size,
            payload.subtitle_en_font_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    matching_id = database.find_task_by_video_id(
        validated_url.video_id,
        dubbing_enabled=payload.dubbing_enabled,
        subtitle_style=subtitle_style,
    )
    if matching_id:
        return database.get_task(matching_id)

    existing_id = database.find_task_by_video_id(validated_url.video_id)
    if existing_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "A task for this video already exists with different output settings. "
                "Delete the existing task before creating it with the new settings."
            ),
        )

    _ensure_runtime_ready()
    task_id = database.create_task(
        validated_url.url,
        task_id=validated_url.video_id,
        execution_mode=normalize_execution_mode(payload.execution_mode),
        dubbing_enabled=payload.dubbing_enabled,
        bilibili_draft_enabled=payload.bilibili_draft_enabled,
        subtitle_style=subtitle_style,
    )
    worker.enqueue(task_id)
    return database.get_task(task_id)


def _clean_upload_filename(filename: str | None) -> str:
    original = Path(filename or "").name.strip()
    if not original:
        raise HTTPException(status_code=422, detail="Video filename is required.")
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="Unsupported video file type.")
    safe_stem = sanitize_text(Path(original).stem) or "video"
    return f"{safe_stem}{suffix}"


def _clean_subtitle_filename(filename: str | None) -> str:
    original = Path(filename or "").name.strip()
    if not original:
        raise HTTPException(status_code=422, detail="Subtitle filename is required.")
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_SUBTITLE_SUFFIXES:
        raise HTTPException(status_code=422, detail="Only .srt subtitle files are supported.")
    safe_stem = sanitize_text(Path(original).stem) or "subtitles"
    return f"{safe_stem}{suffix}"


def _save_uploaded_file(file: UploadFile, destination: Path, *, max_bytes: int, too_large_detail: str) -> int:
    total = 0
    created = False
    try:
        with runtime_security.open_private_binary_exclusive(destination) as handle:
            created = True
            while True:
                chunk = file.file.read(LOCAL_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=too_large_detail)
                handle.write(chunk)
    except Exception:
        if created:
            runtime_security.remove_private_file(destination, missing_ok=True)
        raise
    if total == 0:
        runtime_security.remove_private_file(destination, missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    return total


def _validate_uploaded_srt(path: Path) -> None:
    try:
        parse_srt(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid SRT subtitle file encoding.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid SRT subtitle file: {exc}") from exc


def _rollback_local_upload(task_id: str) -> None:
    try:
        remove_upload(WORKFOLDER, task_id)
    except Exception:
        logger.exception("Failed to remove local upload files for task %s", task_id)
    try:
        database.delete_task(task_id)
    except Exception:
        logger.exception("Failed to remove local upload database record for task %s", task_id)


@app.post("/api/tasks/upload", status_code=201)
def upload_local_video(
    direction: str = Form("en-zh"),
    file: UploadFile = File(...),
    subtitle_file: UploadFile | None = File(None),
    execution_mode: str = Form("auto"),
    dubbing_enabled: bool = Form(True),
    bilibili_draft_enabled: bool = Form(True),
    subtitle_zh_font: str = Form(DEFAULT_CHINESE_FONT),
    subtitle_en_font: str = Form(DEFAULT_ENGLISH_FONT),
    subtitle_zh_font_size: int = Form(DEFAULT_CHINESE_FONT_SIZE),
    subtitle_en_font_size: int = Form(DEFAULT_ENGLISH_FONT_SIZE),
) -> dict:
    if direction not in LOCAL_UPLOAD_DIRECTIONS:
        raise HTTPException(status_code=422, detail="Unsupported local video direction.")

    original_name = Path(file.filename or "").name.strip()
    stored_name = _clean_upload_filename(original_name)
    stored_subtitle_name = None
    if subtitle_file is not None:
        stored_subtitle_name = _clean_subtitle_filename(subtitle_file.filename)
    normalized_execution_mode = normalize_execution_mode(execution_mode)
    try:
        subtitle_style = normalize_subtitle_style(
            subtitle_zh_font,
            subtitle_en_font,
            subtitle_zh_font_size,
            subtitle_en_font_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _ensure_runtime_ready()

    task_id = str(uuid.uuid4())
    try:
        _save_uploaded_file(
            file,
            uploaded_video_dir(WORKFOLDER, task_id) / stored_name,
            max_bytes=MAX_LOCAL_UPLOAD_BYTES,
            too_large_detail="Uploaded video is too large.",
        )
        if subtitle_file is not None and stored_subtitle_name is not None:
            subtitle_path = uploaded_subtitle_dir(WORKFOLDER, task_id) / stored_subtitle_name
            _save_uploaded_file(
                subtitle_file,
                subtitle_path,
                max_bytes=MAX_LOCAL_SUBTITLE_BYTES,
                too_large_detail="Uploaded subtitle is too large.",
            )
            _validate_uploaded_srt(subtitle_path)

        url = f"local://upload/{task_id}?direction={direction}&filename={quote(original_name)}"
        database.create_task(
            url,
            task_id=task_id,
            execution_mode=normalized_execution_mode,
            dubbing_enabled=dubbing_enabled,
            bilibili_draft_enabled=bilibili_draft_enabled,
            subtitle_style=subtitle_style,
        )
        database.update_task(task_id, title=Path(original_name).stem)
        task = database.get_task(task_id)
        if task is None:
            raise RuntimeError(f"Local upload task {task_id} was not persisted.")
        worker.enqueue(task_id)
        return task
    except Exception:
        _rollback_local_upload(task_id)
        raise


@app.get("/api/tasks/current")
def current_task() -> dict | None:
    return database.get_current_task()


@app.get("/api/tasks")
def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: str = Query("", max_length=200),
    status: TaskListStatus = "all",
    execution_mode: TaskListExecutionMode = "all",
    sort: TaskListSort = "created_desc",
) -> dict:
    return database.list_tasks_page(
        page=page,
        page_size=page_size,
        query=q,
        status=status,
        execution_mode=execution_mode,
        sort=sort,
    )


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


def _is_inside_workfolder(path: Path) -> bool:
    workfolder = WORKFOLDER.resolve()
    try:
        path.resolve().relative_to(workfolder)
    except ValueError:
        return False
    return True


def _make_writable_and_retry(function, path: str, _exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _remove_tree(path: Path) -> None:
    last_error: OSError | None = None
    for delay in (0.0, 0.1, 0.3):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path, onerror=_make_writable_and_retry)
            return
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _remove_file(path: Path) -> None:
    last_error: OSError | None = None
    for delay in (0.0, 0.1, 0.3):
        if delay:
            time.sleep(delay)
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                path.chmod(stat.S_IWRITE)
            except OSError:
                pass
    if last_error is not None:
        raise last_error


def _purge_task(task: dict, *, tolerate_file_errors: bool = False) -> None:
    cleanup_errors: list[tuple[str, OSError]] = []
    session_path = task.get("session_path")
    if session_path:
        session_dir = Path(session_path)
        if session_dir.exists() and _is_inside_workfolder(session_dir):
            try:
                _remove_tree(session_dir)
            except OSError as exc:
                cleanup_errors.append((str(session_dir), exc))
    log_file = database.log_path(task["id"])
    if log_file.exists():
        try:
            _remove_file(log_file)
        except OSError as exc:
            cleanup_errors.append((str(log_file), exc))
    if cleanup_errors and not tolerate_file_errors:
        raise cleanup_errors[0][1]
    for path, exc in cleanup_errors:
        logger.warning("Could not remove task artifact %s: %s", path, exc)
    database.delete_task(task["id"])


@app.delete("/api/tasks/{task_id}", status_code=204)
def delete_task(task_id: str) -> Response:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a running task.")
    _purge_task(task, tolerate_file_errors=True)
    if is_local_upload_url(task["url"]):
        try:
            remove_upload(WORKFOLDER, task["id"])
        except OSError as exc:
            logger.warning("Could not remove local upload for task %s: %s", task["id"], exc)
    return Response(status_code=204)


@app.post("/api/tasks/{task_id}/rerun")
def rerun_task(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot rerun a running task.")

    _ensure_runtime_ready()
    url = task["url"]
    execution_mode = task.get("execution_mode") or database.DEFAULT_EXECUTION_MODE
    dubbing_enabled = bool(task.get("dubbing_enabled", True))
    bilibili_draft_enabled = bool(task.get("bilibili_draft_enabled", True))
    subtitle_style = normalize_subtitle_style(
        task["subtitle_zh_font"],
        task["subtitle_en_font"],
        task["subtitle_zh_font_size"],
        task["subtitle_en_font_size"],
    )
    _purge_task(task)
    new_id = database.create_task(
        url,
        task_id=task_id,
        execution_mode=execution_mode,
        dubbing_enabled=dubbing_enabled,
        bilibili_draft_enabled=bilibili_draft_enabled,
        subtitle_style=subtitle_style,
    )
    worker.enqueue(new_id)
    return database.get_task(new_id)


@app.post("/api/tasks/{task_id}/stages/{stage_name}/redo")
def redo_stage(task_id: str, stage_name: str) -> dict:
    if stage_name not in STAGE_NAMES:
        raise HTTPException(status_code=404, detail="Stage not found.")
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if (task.get("execution_mode") or database.DEFAULT_EXECUTION_MODE) != "manual":
        raise HTTPException(status_code=409, detail="Only manual tasks support per-stage redo.")
    if task["status"] in {"running", "queued"}:
        raise HTTPException(status_code=409, detail="Task is already running or queued.")
    stage = next((item for item in task["stages"] if item["name"] == stage_name), None)
    if not stage:
        raise HTTPException(status_code=404, detail="Stage not found.")
    if stage["status"] not in {"succeeded", "failed"}:
        raise HTTPException(status_code=409, detail="Only completed or failed stages can be redone.")
    _ensure_runtime_ready()
    session_path = task.get("session_path")
    if session_path:
        remove_stage_artifacts(Path(session_path), stage_name, detect_source(task["url"]))
    database.reset_stages_from(task_id, stage_name)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@app.post("/api/tasks/{task_id}/continue")
def continue_task(task_id: str, payload: ContinueTaskRequest | None = None) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] != "paused":
        raise HTTPException(status_code=409, detail="Only paused tasks can be continued.")
    if (task.get("execution_mode") or database.DEFAULT_EXECUTION_MODE) != "manual":
        raise HTTPException(status_code=409, detail="Only manual tasks can be continued step by step.")
    if payload and payload.execution_mode is not None:
        database.update_task(task_id, execution_mode=normalize_execution_mode(payload.execution_mode))
    _ensure_runtime_ready()
    database.queue_task_for_continue(task_id)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str) -> dict:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] != "failed":
        raise HTTPException(status_code=409, detail="Only failed tasks can be resumed.")
    _ensure_runtime_ready()
    database.reset_failed_for_resume(task_id)
    worker.enqueue(task_id)
    return database.get_task(task_id)


@app.get("/api/tasks/{task_id}/log", response_class=PlainTextResponse)
def task_log(task_id: str) -> str:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    path = database.log_path(task_id)
    return path.read_text(encoding="utf-8") if path.exists() else ""


@app.get("/api/tasks/{task_id}/artifact/final-video")
def final_video(task_id: str, download: bool = False) -> FileResponse:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    final_path = task.get("final_video_path")
    if not final_path or not Path(final_path).exists():
        raise HTTPException(status_code=404, detail="Final video is not available.")
    name = Path(final_path).name
    if download:
        return FileResponse(final_path, media_type="video/mp4", filename=name)
    headers = {"Content-Disposition": f'inline; filename="{name}"'}
    return FileResponse(final_path, media_type="video/mp4", headers=headers)


@app.get("/api/tasks/{task_id}/artifact/thumbnail")
def task_thumbnail(task_id: str) -> FileResponse:
    task = database.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    thumbnail_path = task.get("thumbnail_path")
    session_path = task.get("session_path")
    if not thumbnail_path or not session_path:
        raise HTTPException(status_code=404, detail="Thumbnail is not available.")

    path = Path(thumbnail_path).resolve()
    media_dir = (Path(session_path) / "media").resolve()
    try:
        path.relative_to(media_dir)
    except ValueError:
        raise HTTPException(status_code=404, detail="Thumbnail is not available.") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail is not available.")
    return FileResponse(path)


@app.post("/api/tasks/{task_id}/bilibili/draft")
def create_bilibili_draft(task_id: str, payload: BilibiliDraftRequest) -> dict:
    from .bilibili_uploader import (
        BILIBILI_DRAFT_STAGE,
        BilibiliDraftError,
        submit_bilibili_draft,
    )

    try:
        result = submit_bilibili_draft(
            task_id,
            title=payload.title,
            tid=payload.tid,
            tag=payload.tag,
            description=payload.description,
        )
    except BilibiliDraftError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except bilibili.BilibiliError as exc:
        try:
            database.update_stage(
                task_id,
                BILIBILI_DRAFT_STAGE,
                status="failed",
                completed_at=database.now_iso(),
                last_message="Failed",
                error_message=str(exc),
            )
        except Exception:
            logger.exception("Failed to mark bilibili_draft stage failed for task %s", task_id)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except requests.RequestException as exc:
        try:
            database.update_stage(
                task_id,
                BILIBILI_DRAFT_STAGE,
                status="failed",
                completed_at=database.now_iso(),
                last_message="Failed",
                error_message=f"Bilibili request failed: {exc}",
            )
        except Exception:
            logger.exception("Failed to mark bilibili_draft stage failed for task %s", task_id)
        raise HTTPException(status_code=502, detail=f"Bilibili request failed: {exc}") from exc
    database.update_stage(
        task_id,
        BILIBILI_DRAFT_STAGE,
        status="succeeded",
        progress=100,
        started_at=database.now_iso(),
        completed_at=database.now_iso(),
        last_message=f"Draft created: {result['draft_id']}",
        error_message=None,
    )
    return result


@app.get("/api/cookies/youtube")
def get_youtube_cookie() -> dict:
    metadata = runtime_security.private_file_stat(YOUTUBE_COOKIE_PATH)
    exists = metadata is not None
    size = metadata.st_size if metadata else 0
    updated_at = metadata.st_mtime if metadata else None
    return {"exists": exists, "size": size, "updated_at": updated_at, "content": ""}


@app.post("/api/cookies/youtube")
def save_youtube_cookie(payload: YouTubeCookieUpdate) -> dict:
    content = payload.content.strip()
    if content:
        runtime_security.atomic_write_private_text(YOUTUBE_COOKIE_PATH, content + "\n")
    else:
        runtime_security.remove_private_file(YOUTUBE_COOKIE_PATH, missing_ok=True)
    return get_youtube_cookie()


@app.get("/api/cookies/bilibili")
def get_bilibili_cookie() -> dict:
    metadata = runtime_security.private_file_stat(BILIBILI_COOKIE_PATH)
    exists = metadata is not None
    size = metadata.st_size if metadata else 0
    updated_at = metadata.st_mtime if metadata else None
    return {"exists": exists, "size": size, "updated_at": updated_at, "content": ""}


@app.post("/api/cookies/bilibili")
def save_bilibili_cookie(payload: BilibiliCookieUpdate) -> dict:
    content = payload.content.strip()
    if content:
        runtime_security.atomic_write_private_text(BILIBILI_COOKIE_PATH, content + "\n")
    else:
        runtime_security.remove_private_file(BILIBILI_COOKIE_PATH, missing_ok=True)
    return get_bilibili_cookie()


@app.get("/api/cookies/bilibili/qr")
def generate_bilibili_qr() -> dict:
    """生成 B 站扫码登录二维码，返回图片 data URI 与 qrcode_key。"""
    try:
        url, qrcode_key = bilibili.generate_qr_code()
        qr_image = bilibili.render_qr_data_uri(url)
    except bilibili.BilibiliError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Bilibili request failed: {exc}") from exc
    return {
        "qrcode_key": qrcode_key,
        "qr_image": qr_image,
        "expires_in": bilibili.QR_EXPIRE_SECONDS,
    }


@app.post("/api/cookies/bilibili/qr/poll")
def poll_bilibili_qr(payload: BilibiliQrPollRequest) -> dict:
    """轮询扫码状态；成功后自动写入 B 站 Cookie 文件。"""
    try:
        result = bilibili.poll_qr_login(payload.qrcode_key)
    except bilibili.BilibiliError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Bilibili request failed: {exc}") from exc
    if result.get("status") == "success":
        content = bilibili.format_cookie_file(result)
        runtime_security.atomic_write_private_text(BILIBILI_COOKIE_PATH, content)
        return {"status": "success", **get_bilibili_cookie()}
    return result


@app.get("/api/settings/openai")
def get_openai_settings() -> dict:
    settings = database.get_openai_settings()
    return {
        "base_url": settings["base_url"],
        "api_key": mask_secret(settings["api_key"]),
        "has_api_key": bool(settings["api_key"]),
        "model": settings["model"],
        "translate_concurrency": settings["translate_concurrency"],
    }


@app.post("/api/settings/openai")
def save_openai_settings(payload: OpenAISettingsUpdate) -> dict:
    try:
        database.save_openai_settings(
            payload.base_url,
            payload.api_key,
            payload.model,
            normalize_translate_concurrency(payload.translate_concurrency),
            clear_api_key=payload.clear_api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_openai_settings()


@app.post("/api/settings/openai/models")
def get_openai_models(payload: OpenAIModelsRequest) -> dict:
    settings = database.get_openai_settings()
    try:
        saved_base_url = validate_openai_base_url(settings["base_url"])
        requested_base_url = payload.base_url.strip()
        base_url = (
            validate_openai_base_url(requested_base_url)
            if requested_base_url
            else saved_base_url
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    requested_api_key = payload.api_key.strip()
    if requested_base_url and base_url != saved_base_url and not requested_api_key:
        raise HTTPException(
            status_code=422,
            detail="An API key is required when testing a different OpenAI base URL.",
        )
    api_key = requested_api_key or settings["api_key"]
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is not configured.")
    try:
        models = list_openai_models(base_url=base_url, api_key=api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch models from the OpenAI-compatible API.",
        ) from exc
    return {"models": models}


@app.get("/api/settings/ytdlp")
def get_ytdlp_settings() -> dict:
    return database.get_ytdlp_settings()


@app.post("/api/settings/ytdlp")
def save_ytdlp_settings(payload: YtdlpSettingsUpdate) -> dict:
    database.save_ytdlp_settings(normalize_proxy_port(payload.proxy_port))
    return get_ytdlp_settings()

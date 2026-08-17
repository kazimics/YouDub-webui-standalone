"""Shared Bilibili draft upload logic for the manual endpoint and automatic uploads."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

import requests

from . import bilibili, database, runtime_security

logger = logging.getLogger(__name__)

BILIBILI_DRAFT_STAGE = "bilibili_draft"

# Guards against uploading the same task twice at the same time (manual +
# automatic, or repeated clicks). One lock per task id; `_draft_upload_guard`
# protects the dict itself.
_draft_upload_locks: dict[str, threading.Lock] = {}
_draft_upload_guard = threading.Lock()


def _draft_upload_lock(task_id: str) -> threading.Lock:
    with _draft_upload_guard:
        lock = _draft_upload_locks.get(task_id)
        if lock is None:
            lock = threading.Lock()
            _draft_upload_locks[task_id] = lock
        return lock


def try_acquire_draft_upload(task_id: str) -> bool:
    """Try to mark this task as uploading; returns False if already uploading."""
    return _draft_upload_lock(task_id).acquire(blocking=False)


def release_draft_upload(task_id: str) -> None:
    _draft_upload_lock(task_id).release()


class BilibiliDraftError(Exception):
    """Raised for invalid upload requests (maps to an HTTP error code)."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _append_log(task_id: str, message: str) -> None:
    path = database.log_path(task_id)
    timestamp = database.now_iso()
    with runtime_security.open_private_append_text(path) as handle:
        for line in message.rstrip().splitlines() or [""]:
            handle.write(f"[{timestamp}] {line}\n")


def _upload_cover(session: requests.Session, csrf: str, task: dict) -> str:
    thumbnail_path = task.get("thumbnail_path")
    session_path = task.get("session_path")
    if not thumbnail_path or not session_path:
        return ""
    path = Path(thumbnail_path).resolve()
    media_dir = (Path(session_path) / "media").resolve()
    try:
        path.relative_to(media_dir)
    except ValueError:
        return ""
    if not path.is_file():
        return ""
    return bilibili.upload_cover(session, csrf, path)


def submit_bilibili_draft(
    task_id: str,
    *,
    title: str = "",
    tid: int = 171,
    tag: str = "",
    description: str = "",
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Upload the final video of a succeeded task as a Bilibili draft.

    Raises BilibiliDraftError for invalid requests, BilibiliError or
    requests.RequestException for upload failures.
    """
    task = database.get_task(task_id)
    if not task:
        raise BilibiliDraftError("Task not found.", status_code=404)
    if task.get("status") != "succeeded":
        raise BilibiliDraftError(
            "Only succeeded tasks can be uploaded to Bilibili.", status_code=409
        )
    final_path = task.get("final_video_path")
    if not final_path or not Path(final_path).is_file():
        raise BilibiliDraftError("Final video is not available.", status_code=409)
    resolved_title = (
        title.strip() or (task.get("translated_title") or task.get("title") or "").strip()
    ).strip()
    if not resolved_title:
        raise BilibiliDraftError("Bilibili draft title is required.", status_code=422)
    resolved_description = (description or task.get("translated_description") or "").strip()

    from . import main

    sessdata, csrf = bilibili.read_bilibili_credentials(main.BILIBILI_COOKIE_PATH)
    session = bilibili.build_session(sessdata, csrf)
    video_path = Path(final_path)
    pre = bilibili.prepare_upload(session, video_path.name, video_path.stat().st_size)
    if progress_callback is not None:
        progress_callback(0, "准备上传")
    filename, cid = bilibili.upload_video(
        session,
        video_path,
        pre["auth"],
        pre["endpoint"],
        pre["upos_uri"],
        pre["chunk_size"],
        pre["biz_id"],
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(100, "正在创建草稿")
    cover = _upload_cover(session, csrf, task)
    draft_payload = bilibili.build_draft_payload(
        csrf=csrf,
        title=resolved_title,
        tid=tid,
        tag=tag,
        description=resolved_description,
        filename=filename,
        cid=cid,
        cover=cover,
        source=task.get("url") or "",
    )
    draft_id = bilibili.create_draft(session, draft_payload)
    return {"draft_id": draft_id, "aid": 0, "title": resolved_title, "cover": cover}


def _update_stage(task_id: str, **fields: Any) -> None:
    try:
        database.update_stage(task_id, BILIBILI_DRAFT_STAGE, **fields)
    except Exception:
        logger.exception("Failed to update bilibili_draft stage for task %s", task_id)


def submit_bilibili_draft_async(task_id: str) -> threading.Thread | None:
    """Start a background daemon thread that uploads the draft for a succeeded task.

    Returns None when this task is already uploading (manual upload in
    progress or a previous auto-upload thread still running).
    """
    if not try_acquire_draft_upload(task_id):
        _append_log(
            task_id,
            "[bilibili_draft] Auto-upload skipped: a draft upload is already in progress",
        )
        return None
    thread = threading.Thread(
        target=_auto_upload_worker, args=(task_id,), daemon=True
    )
    thread.start()
    return thread


def _auto_upload_worker(task_id: str) -> None:
    now = database.now_iso()
    try:
        database.ensure_bilibili_draft_stage(task_id)
        _update_stage(
            task_id,
            status="running",
            started_at=now,
            progress=0,
            last_message="正在上传 B 站草稿",
            error_message=None,
        )
        _append_log(task_id, "[bilibili_draft] Auto-uploading draft to Bilibili...")
        result = submit_bilibili_draft(
            task_id,
            progress_callback=lambda progress, message: _update_stage(
                task_id,
                progress=progress,
                last_message=message,
            ),
        )
        _update_stage(
            task_id,
            status="succeeded",
            progress=100,
            completed_at=database.now_iso(),
            last_message=f"Draft created: {result['draft_id']}",
            error_message=None,
        )
        _append_log(
            task_id,
            f"[bilibili_draft] Draft created (id={result['draft_id']}, title={result['title']})",
        )
    except Exception as exc:
        logger.exception("Bilibili draft auto-upload failed for task %s", task_id)
        error_message = str(exc).strip() or type(exc).__name__
        _update_stage(
            task_id,
            status="failed",
            completed_at=database.now_iso(),
            last_message="Failed",
            error_message=error_message,
        )
        _append_log(task_id, f"[bilibili_draft] Auto-upload failed: {error_message}")
    finally:
        release_draft_upload(task_id)

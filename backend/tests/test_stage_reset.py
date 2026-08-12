from __future__ import annotations

from pathlib import Path

from backend.app.sources import detect_source
from backend.app.stage_reset import collect_artifact_paths, remove_stage_artifacts
from backend.app.stages import STAGE_NAMES


def test_collect_artifact_paths_from_translate(tmp_path):
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "asr_fixed.json").write_text("{}", encoding="utf-8")
    (metadata / "translation.zh.json").write_text("{}", encoding="utf-8")
    (metadata / "translation_preprocess.json").write_text("{}", encoding="utf-8")
    (metadata / "subtitles.zh.srt").write_text("", encoding="utf-8")
    (metadata / "subtitles.bilingual.ass").write_text("", encoding="utf-8")
    (session / "segments" / "vocals").mkdir(parents=True)
    (session / "segments" / "tts").mkdir(parents=True)

    source = detect_source("https://www.youtube.com/watch?v=abcdefghijk")
    paths = collect_artifact_paths(session, "translate", source)
    relative = {path.relative_to(session).as_posix() for path in paths}

    assert "metadata/translation_preprocess.json" in relative
    assert "metadata/translation.zh.json" in relative
    assert "metadata/subtitles.bilingual.ass" in relative
    assert "metadata/asr_fixed.json" not in relative
    assert "segments/vocals" in relative
    assert "segments/tts" in relative


def test_remove_stage_artifacts_keeps_upstream(tmp_path):
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    asr_fixed = metadata / "asr_fixed.json"
    translation = metadata / "translation.zh.json"
    asr_fixed.write_text("{}", encoding="utf-8")
    translation.write_text("{}", encoding="utf-8")

    source = detect_source("https://www.youtube.com/watch?v=abcdefghijk")
    remove_stage_artifacts(session, "translate", source)

    assert asr_fixed.exists()
    assert not translation.exists()


def test_reset_stages_from_only_resets_downstream(monkeypatch, tmp_path):
    from backend.app import database
    from backend.tests.test_settings_and_api import configure_tmp_runtime

    configure_tmp_runtime(monkeypatch, tmp_path)
    task_id = database.create_task(
        "https://www.youtube.com/watch?v=redostages1",
        task_id="redostages1",
        execution_mode="manual",
    )
    for stage in STAGE_NAMES:
        database.update_stage(task_id, stage, status="succeeded", completed_at=database.now_iso())
    database.update_task(task_id, status="succeeded", current_stage="merge_video")

    database.reset_stages_from(task_id, "translate")

    stages = {stage["name"]: stage for stage in database.get_task(task_id)["stages"]}
    assert stages["asr_fix"]["status"] == "succeeded"
    assert stages["translate"]["status"] == "pending"
    assert stages["merge_video"]["status"] == "pending"
    task = database.get_task(task_id)
    assert task["status"] == "queued"
    assert task["current_stage"] == "translate"
    assert task["final_video_path"] is None

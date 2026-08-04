from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backend.app.adapters import ffmpeg


def test_video_orientation_uses_height_greater_than_width(monkeypatch):
    def fake_run(cmd, capture_output=False, text=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="720,1280\n", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    assert ffmpeg.get_video_orientation(Path("video.mp4")) == "portrait"


def test_video_orientation_defaults_to_landscape_when_probe_fails(monkeypatch):
    def fake_run(cmd, capture_output=False, text=False, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ffprobe failed")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    assert ffmpeg.get_video_orientation(Path("video.mp4")) == "landscape"


def test_subtitle_styles_match_backend_orientation_rules():
    portrait = ffmpeg.subtitle_style_for_orientation("portrait", "Noto Sans CJK SC", "zh")
    landscape = ffmpeg.subtitle_style_for_orientation("landscape", "Noto Sans CJK SC", "zh")

    assert "FontSize=12" in portrait
    assert "MarginV=70" in portrait
    assert "FontSize=24" in landscape
    assert "MarginV=5" in landscape


def test_subtitle_styles_use_smaller_size_for_english():
    portrait_en = ffmpeg.subtitle_style_for_orientation("portrait", "Arial", "en")
    landscape_en = ffmpeg.subtitle_style_for_orientation("landscape", "Arial", "en")

    assert "FontSize=9" in portrait_en
    assert "FontSize=18" in landscape_en


def test_subtitle_filter_picks_chinese_font_for_zh_srt(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg, "get_video_orientation", lambda _: "landscape")
    sub_zh = tmp_path / "subtitles.zh.srt"
    sub_zh.write_text("", encoding="utf-8")
    assert "FontName=Noto Sans CJK SC" in ffmpeg.subtitle_filter(tmp_path / "v.mp4", sub_zh, tmp_path)
    sub_en = tmp_path / "subtitles.en.srt"
    sub_en.write_text("", encoding="utf-8")
    assert "FontName=Arial" in ffmpeg.subtitle_filter(tmp_path / "v.mp4", sub_en, tmp_path)


def test_merge_video_burns_portrait_subtitles(monkeypatch, tmp_path):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    timings = metadata_dir / "timings.json"
    timings.write_text(
        json.dumps(
            {
                "translation": [
                    {
                        "start_time": 0,
                        "end_time": 1200,
                        "actual_start_time": 0,
                        "actual_end_time": 1200,
                        "zh": "你好",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    cwd_values: list[Path | None] = []

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        commands.append(cmd)
        cwd_values.append(kwargs.get("cwd"))
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="720,1280\n", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": "1.0"},
                    }
                ),
                stderr="",
            )
        output = Path(cmd[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"media")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    final_video = ffmpeg.merge_video(
        tmp_path / "video.mp4",
        tmp_path / "dubbing.wav",
        tmp_path / "bgm.wav",
        timings,
        session,
    )

    assert final_video == session / "media" / "video_final.mp4"
    ffmpeg_commands = [cmd for cmd in commands if Path(cmd[0]).name.startswith("ffmpeg")]
    assert len(ffmpeg_commands) == 2
    final_command = ffmpeg_commands[-1]
    filter_arg = final_command[final_command.index("-vf") + 1]
    assert filter_arg.startswith("subtitles=filename='metadata/subtitles.zh.srt'")
    assert "FontSize=12" in filter_arg
    assert "MarginV=70" in filter_arg
    assert "-c:s" not in final_command
    assert cwd_values[commands.index(final_command)] == session.resolve()


def test_merge_video_uses_absolute_media_paths_when_cwd_is_session(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    session = Path("workfolder") / "uploader" / "title__videoid"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    timings = metadata_dir / "timings.json"
    timings.write_text(
        json.dumps(
            {
                "translation": [
                    {
                        "start_time": 0,
                        "end_time": 1200,
                        "actual_start_time": 0,
                        "actual_end_time": 1200,
                        "zh": "你好",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    cwd_values: list[Path | None] = []

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        commands.append(cmd)
        cwd_values.append(kwargs.get("cwd"))
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="720,1280\n", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": "1.0"},
                    }
                ),
                stderr="",
            )
        output = Path(cmd[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"media")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    ffmpeg.merge_video(
        session / "media" / "video_source.mp4",
        session / "tmp" / "audio_dubbing.wav",
        session / "media" / "audio_bgm.wav",
        timings,
        session,
    )

    ffmpeg_commands = [cmd for cmd in commands if Path(cmd[0]).name.startswith("ffmpeg")]
    mix_command = ffmpeg_commands[0]
    final_command = ffmpeg_commands[-1]
    assert Path(mix_command[mix_command.index("-i") + 1]).is_absolute()
    assert Path(mix_command[mix_command.index("-i", mix_command.index("-i") + 1) + 1]).is_absolute()
    assert Path(mix_command[-1]).is_absolute()
    assert Path(final_command[final_command.index("-i") + 1]).is_absolute()
    assert Path(final_command[final_command.index("-i", final_command.index("-i") + 1) + 1]).is_absolute()
    assert Path(final_command[-1]).is_absolute()
    assert cwd_values[commands.index(final_command)] == session.resolve()


def test_merge_video_with_original_audio_burns_subtitles_and_keeps_source_audio(
    monkeypatch,
    tmp_path,
):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    translation = metadata_dir / "translation.zh.json"
    translation.write_text(
        json.dumps(
            {
                "translation": [
                    {"start_time": 0, "end_time": 1200, "dst_lang": "zh", "dst": "你好"}
                ]
            }
        ),
        encoding="utf-8",
    )
    video = session / "media" / "video_source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source")
    commands: list[list[str]] = []

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        commands.append(cmd)
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="1920,1080\n", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": "1.0"},
                    }
                ),
                stderr="",
            )
        Path(cmd[-1]).write_bytes(b"rendered")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    output = ffmpeg.merge_video_with_original_audio(video, translation, session)

    render_command = next(cmd for cmd in commands if Path(cmd[0]).name.startswith("ffmpeg"))
    assert output == session / "media" / "video_final.mp4"
    assert output.read_bytes() == b"rendered"
    assert render_command[render_command.index("-map") + 1] == "0:v:0"
    assert "0:a:0" in render_command
    assert render_command[render_command.index("-c:a") + 1] == "aac"
    assert render_command[render_command.index("-b:a") + 1] == "192k"
    assert any(
        arg.startswith("subtitles=filename='metadata/subtitles.zh.srt'")
        for arg in render_command
    )
    assert render_command[-1].endswith("video_final.part.mp4")
    assert not (session / "media" / "video_final.part.mp4").exists()


def test_merge_video_retries_native_encoder_crash_with_safe_x264_settings(
    monkeypatch,
    tmp_path,
):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    translation = metadata_dir / "translation.zh.json"
    translation.write_text(
        json.dumps(
            {"translation": [{"start_time": 0, "end_time": 1200, "dst": "你好"}]}
        ),
        encoding="utf-8",
    )
    video = session / "media" / "video_source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source")
    render_commands: list[list[str]] = []

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="1920,1080\n", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": "1.2"},
                    }
                ),
                stderr="",
            )

        render_commands.append(cmd)
        part_file = Path(cmd[-1])
        if len(render_commands) == 1:
            part_file.write_bytes(b"partial")
            raise subprocess.CalledProcessError(0xC000008F, cmd)
        assert not part_file.exists()
        part_file.write_bytes(b"rendered")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    output = ffmpeg.merge_video_with_original_audio(video, translation, session)

    assert len(render_commands) == 2
    normal_command, retry_command = render_commands
    assert normal_command[normal_command.index("-preset") + 1] == "fast"
    assert "-x264-params" not in normal_command
    assert retry_command[retry_command.index("-preset") + 1] == "ultrafast"
    assert retry_command[retry_command.index("-threads") + 1] == "1"
    assert retry_command[retry_command.index("-filter_threads") + 1] == "1"
    assert retry_command[retry_command.index("-filter_complex_threads") + 1] == "1"
    assert retry_command[retry_command.index("-x264-params") + 1] == "threads=1"
    assert retry_command[retry_command.index("-pix_fmt") + 1] == "yuv420p"
    assert output.read_bytes() == b"rendered"
    assert not (session / "media" / "video_final.part.mp4").exists()


def test_verified_output_does_not_retry_normal_ffmpeg_error(monkeypatch, tmp_path):
    part_file = tmp_path / "video.part.mp4"
    final_file = tmp_path / "video.mp4"
    command = ["ffmpeg", "normal", str(part_file)]
    retry_command = ["ffmpeg", "safe", str(part_file)]
    calls: list[list[str]] = []

    def fake_run(cmd, check=False, cwd=None):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        ffmpeg._run_verified_output(
            command,
            part_file,
            final_file,
            {"video", "audio"},
            native_crash_retry_command=retry_command,
        )

    assert calls == [command]
    assert not part_file.exists()
    assert not final_file.exists()


def test_merge_video_rejects_and_removes_zero_duration_partial_output(monkeypatch, tmp_path):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    translation = metadata_dir / "translation.zh.json"
    translation.write_text(
        json.dumps(
            {"translation": [{"start_time": 0, "end_time": 1000, "dst": "你好"}]}
        ),
        encoding="utf-8",
    )
    video = session / "media" / "video_source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source")

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="1920,1080\n", stderr="")
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": "0.0"},
                    }
                ),
                stderr="",
            )
        Path(cmd[-1]).write_bytes(b"broken")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    try:
        ffmpeg.merge_video_with_original_audio(video, translation, session)
    except RuntimeError as exc:
        assert "output validation failed" in str(exc)
    else:
        raise AssertionError("invalid FFmpeg output must fail validation")

    assert not (session / "media" / "video_final.part.mp4").exists()
    assert not (session / "media" / "video_final.mp4").exists()


def test_merge_video_rejects_nonzero_but_severely_truncated_output(monkeypatch, tmp_path):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    translation = metadata_dir / "translation.zh.json"
    translation.write_text(
        json.dumps(
            {"translation": [{"start_time": 0, "end_time": 10_000, "dst": "你好"}]}
        ),
        encoding="utf-8",
    )
    video = session / "media" / "video_source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source")

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="1920,1080\n", stderr="")
            duration = "10.0" if Path(cmd[-1]).name == "video_source.mp4" else "0.2"
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": duration},
                    }
                ),
                stderr="",
            )
        Path(cmd[-1]).write_bytes(b"truncated")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="output validation failed"):
        ffmpeg.merge_video_with_original_audio(video, translation, session)

    assert not (session / "media" / "video_final.part.mp4").exists()
    assert not (session / "media" / "video_final.mp4").exists()


def test_merge_video_replaces_zero_duration_existing_final(monkeypatch, tmp_path):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    translation = metadata_dir / "translation.zh.json"
    translation.write_text(
        json.dumps(
            {"translation": [{"start_time": 0, "end_time": 1000, "dst": "你好"}]}
        ),
        encoding="utf-8",
    )
    media_dir = session / "media"
    media_dir.mkdir(parents=True)
    video = media_dir / "video_source.mp4"
    final_video = media_dir / "video_final.mp4"
    video.write_bytes(b"source")
    final_video.write_bytes(b"old-zero-duration")
    render_count = 0

    def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
        nonlocal render_count
        if Path(cmd[0]).name.startswith("ffprobe"):
            if "stream=width,height" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="1920,1080\n", stderr="")
            duration = "0.0" if Path(cmd[-1]).name == "video_final.mp4" else "2.0"
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                        "format": {"duration": duration},
                    }
                ),
                stderr="",
            )
        render_count += 1
        Path(cmd[-1]).write_bytes(b"new-valid-video")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    output = ffmpeg.merge_video_with_original_audio(video, translation, session)

    assert render_count == 1
    assert output == final_video
    assert output.read_bytes() == b"new-valid-video"


def test_split_subtitle_text_breaks_on_punctuation_and_keeps_protected():
    out = ffmpeg.split_subtitle_text("我们今天讨论一下宇宙的边界，那是一个神秘话题；不过别担心，我会详细解释。")
    assert len(out) >= 3
    assert all(len(s) >= 2 for s in out)
    protected = ffmpeg.split_subtitle_text("他说《三体，黑暗森林》是经典，必读。")
    assert any("《三体，黑暗森林》" in s for s in protected)


def test_write_srt_splits_long_sentence_into_multiple_entries(tmp_path):
    session = tmp_path / "session"
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True)
    timings = metadata_dir / "timings.json"
    timings.write_text(
        json.dumps(
            {
                "translation": [
                    {
                        "start_time": 0,
                        "end_time": 6000,
                        "actual_start_time": 0,
                        "actual_end_time": 6000,
                        "zh": "我们今天讨论宇宙的边界，那是一个神秘话题；不过别担心，我会详细解释",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    srt = ffmpeg.write_srt(timings, session)
    content = srt.read_text(encoding="utf-8")
    blocks = [b for b in content.strip().split("\n\n") if b.strip()]
    assert len(blocks) >= 3
    assert all("-->" in b for b in blocks)


def test_probe_video_size_uses_configured_ffprobe(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(cmd, capture_output=False, text=False, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="1920,1080\n", stderr="")

    monkeypatch.setenv("FFPROBE_PATH", "/opt/bin/ffprobe")
    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)

    assert ffmpeg.probe_video_size(Path("video.mp4")) == (1920, 1080)
    assert commands[0][0] == "/opt/bin/ffprobe"

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from pathlib import Path

from ..config import ffmpeg_binary, ffprobe_binary
from ..subtitle_style import DEFAULT_SUBTITLE_STYLE, SubtitleStyle

SUBTITLE_PUNCTUATION = {"，", ",", "；", ";", "：", ":", "。", "?", "？", "!", "！", "、"}
SUBTITLE_PROTECTED_PAIRS = {"《": "》", "（": "）", "【": "】", "「": "」", "『": "』"}
SUBTITLE_CLOSING_QUOTES = {'"', "'", "」", "』", "》", "）", "】", "\u201d", "\u2019", "]"}
SUBTITLE_MIN_FRAGMENT_LEN = 5
SUBTITLE_MIN_DURATION_MS = 200
SUBTITLE_TAIL_BUFFER_MS = 100
SUBTITLE_DURATION_FLOOR_MS = 600
MEDIA_MIN_DURATION_SECONDS = 0.05
MEDIA_MAX_DURATION_TOLERANCE_SECONDS = 2.0
MEDIA_DURATION_TOLERANCE_RATIO = 0.05
NATIVE_PROCESS_CRASH_THRESHOLD = 0xC0000000


SUBTITLE_FONTS = {
    "zh": "Noto Sans CJK SC",
    "en": "Arial",
    "bilingual": "Noto Sans CJK SC",
}

SUBTITLE_FONT_SIZES = {
    "zh": {"portrait": 12, "landscape": 24},
    "en": {"portrait": 9, "landscape": 18},
    "bilingual": {"portrait": 9, "landscape": 18},
}


def _subtitle_style(font: str, size: int, margin_v: int) -> str:
    return (
        f"FontName={font},"
        f"FontSize={size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Alignment=2,"
        f"MarginV={margin_v}"
    )


def _srt_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1000
    millis = ms - seconds * 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _ass_time(ms: int) -> str:
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    seconds = ms // 1000
    centiseconds = (ms - seconds * 1000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _split_protected(text: str) -> list[str]:
    segments: list[str] = []
    buf: list[str] = []
    inside = None
    for ch in text:
        if inside is None and ch in SUBTITLE_PROTECTED_PAIRS:
            inside = SUBTITLE_PROTECTED_PAIRS[ch]
            buf.append(ch)
            continue
        if inside is not None and ch == inside:
            inside = None
            buf.append(ch)
            continue
        if inside is None and ch in SUBTITLE_PUNCTUATION:
            chunk = "".join(buf).strip()
            if chunk:
                segments.append(chunk)
            buf.clear()
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return segments


def _attach_closing_quotes(segments: list[str]) -> list[str]:
    fixed: list[str] = []
    for seg in segments:
        if seg and seg[0] in SUBTITLE_CLOSING_QUOTES and fixed:
            fixed[-1] = f"{fixed[-1]}{seg}".strip()
            continue
        fixed.append(seg.strip())
    return fixed


def _merge_short_fragments(segments: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        if len(cur.strip()) < SUBTITLE_MIN_FRAGMENT_LEN and i + 1 < len(segments):
            segments[i + 1] = f"{cur}{segments[i + 1]}".strip()
            i += 1
            continue
        merged.append(cur)
        i += 1
    return merged


def _strip_trailing_punct(segments: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in segments:
        text = item.strip()
        if not text:
            continue
        if text.endswith(("，", ",", "。")):
            text = text[:-1]
        cleaned.append(re.sub(r"\s+", " ", text).strip())
    return cleaned


def split_subtitle_text(text: str) -> list[str]:
    original = (text or "").strip()
    if not original:
        return []
    segments = _split_protected(original)
    if not segments:
        return [original]
    segments = _attach_closing_quotes(segments)
    segments = _merge_short_fragments(segments)
    cleaned = _strip_trailing_punct(segments)
    return cleaned or [original]


def _allocate_durations(fragments: list[str], total_duration: int) -> list[int]:
    if len(fragments) == 1:
        return [total_duration]
    weights = [max(1, len(f.replace(" ", ""))) for f in fragments]
    total_weight = sum(weights)
    durations: list[int] = []
    allocated = 0
    for i, weight in enumerate(weights[:-1]):
        share = round(total_duration * weight / total_weight)
        if total_duration >= SUBTITLE_DURATION_FLOOR_MS:
            ceiling = total_duration - allocated - SUBTITLE_TAIL_BUFFER_MS
            share = max(SUBTITLE_MIN_DURATION_MS, min(share, ceiling))
        else:
            share = max(int(SUBTITLE_MIN_DURATION_MS / 2), share)
        durations.append(share)
        allocated += share
    durations.append(max(SUBTITLE_TAIL_BUFFER_MS, total_duration - allocated))
    return durations


def _segment_times(item: dict) -> tuple[int, int]:
    start = int(item.get("actual_start_time", item["start_time"]))
    end = int(item.get("actual_end_time", item["end_time"]))
    return start, end


def _dst_lang(translation: list[dict]) -> str:
    for item in translation:
        lang = item.get("dst_lang")
        if lang:
            return lang
    return "zh"


def _dst_text(item: dict) -> str:
    return item.get("dst") or item.get("zh") or ""


def _clean_subtitle_line(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _bilingual_text(item: dict) -> str:
    src = _clean_subtitle_line(str(item.get("src") or ""))
    dst = _clean_subtitle_line(str(_dst_text(item)))
    if not src or not dst or src == dst:
        return ""

    src_lang = item.get("src_lang")
    dst_lang = item.get("dst_lang")
    if src_lang == "zh":
        zh, en = src, dst
    elif dst_lang == "zh":
        zh, en = dst, src
    else:
        zh, en = dst, src
    return f"{zh}\n{en}"


def write_srt(translation_file: Path, session: Path) -> Path:
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    translation = data["translation"]
    dst_lang = _dst_lang(translation)
    has_bilingual_items = any(_bilingual_text(item) for item in translation)
    subtitle_kind = "bilingual" if has_bilingual_items else dst_lang
    output_file = session / "metadata" / f"subtitles.{subtitle_kind}.srt"
    lines: list[str] = []
    idx = 1
    for item in translation:
        start, end = _segment_times(item)
        if end <= start:
            continue
        bilingual = _bilingual_text(item)
        if bilingual:
            lines.extend([str(idx), f"{_srt_time(start)} --> {_srt_time(end)}", bilingual, ""])
            idx += 1
            continue
        fragments = split_subtitle_text(_dst_text(item))
        if not fragments:
            continue
        cursor = start
        for fragment, duration in zip(fragments, _allocate_durations(fragments, end - start)):
            lines.extend([str(idx), f"{_srt_time(cursor)} --> {_srt_time(cursor + duration)}", fragment, ""])
            cursor += duration
            idx += 1
    output_file.write_text("\n".join(lines), encoding="utf-8")
    return output_file


def _oriented_font_size(size: int, orientation: str) -> int:
    return size if orientation == "landscape" else max(6, round(size / 2))


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def _ass_char_width(char: str, font_size: int) -> float:
    if char.isspace():
        return font_size * 0.35
    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return float(font_size)
    return font_size * 0.62


def _wrap_ass_lines(text: str, font_size: int, max_width: float) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    lines: list[str] = []
    current: list[str] = []
    current_width = 0.0
    for char in normalized:
        char_width = _ass_char_width(char, font_size)
        if current and current_width + char_width > max_width:
            break_at = "".join(current).rfind(" ")
            if break_at > 0:
                lines.append("".join(current[:break_at]).rstrip())
                current = current[break_at + 1 :]
                current_width = sum(_ass_char_width(item, font_size) for item in current)
            else:
                lines.append("".join(current).rstrip())
                current = []
                current_width = 0.0
            if char.isspace():
                continue
        current.append(char)
        current_width += char_width
    if current:
        lines.append("".join(current).rstrip())
    return lines


def _balanced_chunks(lines: list[str], count: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    cursor = 0
    for index in range(count):
        remaining = len(lines) - cursor
        pages_left = count - index
        take = (remaining + pages_left - 1) // pages_left
        chunk = lines[cursor : cursor + take]
        if not chunk and lines:
            chunk = [lines[-1]]
        chunks.append(chunk)
        cursor += take
    return chunks


def _bilingual_ass_pages(
    zh: str,
    en: str,
    zh_size: int,
    en_size: int,
    max_width: float,
    play_res_y: int,
) -> list[tuple[str, str]]:
    zh_lines = _wrap_ass_lines(zh, zh_size, max_width)
    en_lines = _wrap_ass_lines(en, en_size, max_width)
    available_height = play_res_y * 0.45
    one_pair_height = zh_size + en_size
    zh_lines_per_page = 1
    en_lines_per_page = 1
    if one_pair_height + zh_size <= available_height:
        zh_lines_per_page = 2
    if (
        one_pair_height
        + (zh_size if zh_lines_per_page > 1 else 0)
        + en_size
        <= available_height
    ):
        en_lines_per_page = 2
    page_count = max(
        1,
        (len(zh_lines) + zh_lines_per_page - 1) // zh_lines_per_page,
        (len(en_lines) + en_lines_per_page - 1) // en_lines_per_page,
    )
    zh_chunks = _balanced_chunks(zh_lines, page_count)
    en_chunks = _balanced_chunks(en_lines, page_count)
    return [
        (
            r"\N".join(_ass_escape(line) for line in zh_chunk),
            r"\N".join(_ass_escape(line) for line in en_chunk),
        )
        for zh_chunk, en_chunk in zip(zh_chunks, en_chunks)
    ]


def write_bilingual_ass(
    translation_file: Path,
    session: Path,
    style: SubtitleStyle,
    orientation: str,
) -> Path:
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    translation = data["translation"]
    output_file = session / "metadata" / "subtitles.bilingual.ass"
    margin_v = 70 if orientation == "portrait" else 5
    zh_size = _oriented_font_size(style.chinese_font_size, orientation)
    en_size = _oriented_font_size(style.english_font_size, orientation)
    play_res_x, play_res_y = ((216, 384) if orientation == "portrait" else (384, 216))
    max_text_width = play_res_x - 36
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Chinese,{style.chinese_font},{zh_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,{margin_v},1"
        ),
        (
            f"Style: English,{style.english_font},{en_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for item in translation:
        start, end = _segment_times(item)
        bilingual = _bilingual_text(item)
        if end <= start or not bilingual:
            continue
        zh, en = bilingual.split("\n", 1)
        pages = _bilingual_ass_pages(
            zh,
            en,
            zh_size,
            en_size,
            max_text_width,
            play_res_y,
        )
        durations = _allocate_durations(
            [f"{zh_page}{en_page}" for zh_page, en_page in pages],
            end - start,
        )
        cursor = start
        for (zh_page, en_page), duration in zip(pages, durations):
            text = rf"{{\rChinese}}{zh_page}\N{{\rEnglish}}{en_page}"
            lines.append(
                f"Dialogue: 0,{_ass_time(cursor)},{_ass_time(cursor + duration)},"
                f"Chinese,,0,0,0,,{text}"
            )
            cursor += duration
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file


def write_subtitles(
    translation_file: Path,
    session: Path,
    style: SubtitleStyle,
    orientation: str,
) -> Path:
    data = json.loads(translation_file.read_text(encoding="utf-8"))
    if any(_bilingual_text(item) for item in data["translation"]):
        return write_bilingual_ass(translation_file, session, style, orientation)
    return write_srt(translation_file, session)


def probe_video_size(video_file: Path) -> tuple[int, int] | None:
    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    parts = lines[0].split(",", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def get_video_orientation(video_file: Path) -> str:
    size = probe_video_size(video_file)
    if size is None:
        return "landscape"
    width, height = size
    return "portrait" if height > width else "landscape"


def subtitle_style_for_orientation(orientation: str, font: str, lang: str = "zh") -> str:
    sizes = SUBTITLE_FONT_SIZES.get(lang, SUBTITLE_FONT_SIZES["zh"])
    margin_v = 70 if orientation == "portrait" else 5
    return _subtitle_style(font, size=sizes[orientation], margin_v=margin_v)


def _subtitle_filter_path(subtitle_file: Path, session: Path) -> str:
    try:
        return subtitle_file.resolve().relative_to(session.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Subtitle file must be inside the session directory.") from exc


def subtitle_filter(
    video_file: Path,
    subtitle_file: Path,
    session: Path,
    style: SubtitleStyle = DEFAULT_SUBTITLE_STYLE,
) -> str:
    lang = subtitle_file.stem.rsplit(".", 1)[-1]
    sub_path = _subtitle_filter_path(subtitle_file, session)
    if subtitle_file.suffix.lower() == ".ass":
        return f"subtitles=filename='{sub_path}'"
    orientation = get_video_orientation(video_file)
    if lang == "zh":
        font = style.chinese_font
        size = _oriented_font_size(style.chinese_font_size, orientation)
    else:
        font = style.english_font
        size = _oriented_font_size(style.english_font_size, orientation)
    margin_v = 70 if orientation == "portrait" else 5
    force_style = _subtitle_style(font, size, margin_v)
    return f"subtitles=filename='{sub_path}':force_style='{force_style}'"


def _probe_media(media_file: Path) -> tuple[set[str], float]:
    if not media_file.is_file() or media_file.stat().st_size <= 0:
        return set(), 0.0
    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            str(media_file),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set(), 0.0
    try:
        data = json.loads(result.stdout)
        stream_types = {
            str(stream.get("codec_type") or "").strip()
            for stream in data.get("streams", [])
            if str(stream.get("codec_type") or "").strip()
        }
        duration = float(data.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError, json.JSONDecodeError):
        return set(), 0.0
    return stream_types, duration


def _minimum_output_duration(reference_duration: float) -> float:
    if reference_duration <= 0:
        return MEDIA_MIN_DURATION_SECONDS
    tolerance = min(
        MEDIA_MAX_DURATION_TOLERANCE_SECONDS,
        reference_duration * MEDIA_DURATION_TOLERANCE_RATIO,
    )
    return max(MEDIA_MIN_DURATION_SECONDS, reference_duration - tolerance)


def _is_valid_media(
    media_file: Path,
    expected_streams: set[str],
    minimum_duration: float = MEDIA_MIN_DURATION_SECONDS,
) -> bool:
    stream_types, duration = _probe_media(media_file)
    return duration >= minimum_duration and expected_streams.issubset(stream_types)


def _is_native_process_crash(returncode: int) -> bool:
    return returncode < 0 or returncode >= NATIVE_PROCESS_CRASH_THRESHOLD


def _safe_x264_command(command: list[str]) -> list[str]:
    safe_command = list(command)
    safe_command[1:1] = [
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ]
    if "-preset" in safe_command:
        preset_index = safe_command.index("-preset") + 1
        safe_command[preset_index] = "ultrafast"

    output_index = len(safe_command) - 1
    if "-x264-params" not in safe_command:
        safe_command[output_index:output_index] = ["-x264-params", "threads=1"]
        output_index += 2
    if "-pix_fmt" not in safe_command:
        safe_command[output_index:output_index] = ["-pix_fmt", "yuv420p"]
    return safe_command


def _run_verified_output(
    command: list[str],
    part_file: Path,
    final_file: Path,
    expected_streams: set[str],
    *,
    minimum_duration: float = MEDIA_MIN_DURATION_SECONDS,
    cwd: Path | None = None,
    native_crash_retry_command: list[str] | None = None,
) -> None:
    part_file.unlink(missing_ok=True)
    try:
        try:
            subprocess.run(command, check=True, cwd=cwd)
        except subprocess.CalledProcessError as exc:
            if (
                native_crash_retry_command is None
                or not _is_native_process_crash(exc.returncode)
            ):
                raise
            part_file.unlink(missing_ok=True)
            subprocess.run(native_crash_retry_command, check=True, cwd=cwd)
        if not _is_valid_media(part_file, expected_streams, minimum_duration):
            expected = ", ".join(sorted(expected_streams))
            raise RuntimeError(
                f"FFmpeg output validation failed for {part_file} "
                f"(expected streams: {expected}, minimum duration: {minimum_duration:.3f}s)."
            )
        part_file.replace(final_file)
    except Exception:
        part_file.unlink(missing_ok=True)
        raise


def _reuse_valid_final(final_video: Path, minimum_duration: float) -> bool:
    if _is_valid_media(final_video, {"video", "audio"}, minimum_duration):
        return True
    final_video.unlink(missing_ok=True)
    return False


def merge_video(
    video_file: Path,
    dubbing_file: Path,
    bgm_file: Path,
    timings_file: Path,
    session: Path,
    subtitle_style: SubtitleStyle = DEFAULT_SUBTITLE_STYLE,
) -> Path:
    tmp_dir = session / "tmp"
    media_dir = session / "media"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    final_video = media_dir / "video_final.mp4"
    session_dir = session.resolve()
    video_input = video_file.resolve()
    dubbing_input = dubbing_file.resolve()
    bgm_input = bgm_file.resolve()
    _, video_duration = _probe_media(video_input)
    minimum_video_duration = _minimum_output_duration(video_duration)
    if _reuse_valid_final(final_video, minimum_video_duration):
        return final_video

    orientation = get_video_orientation(video_input)
    subtitles = write_subtitles(timings_file, session, subtitle_style, orientation)
    mixed_audio = tmp_dir / "audio_mixed.m4a"
    mixed_audio_part = tmp_dir / "audio_mixed.part.m4a"
    mixed_audio_output = mixed_audio.resolve()
    mixed_audio_part_output = mixed_audio_part.resolve()
    final_video_output = final_video.resolve()
    final_video_part = media_dir / "video_final.part.mp4"
    final_video_part_output = final_video_part.resolve()
    _, bgm_duration = _probe_media(bgm_input)
    _run_verified_output(
        [
            ffmpeg_binary(),
            "-y",
            "-i",
            str(dubbing_input),
            "-i",
            str(bgm_input),
            "-filter_complex",
            "[0:a]volume=1.0[a0];[1:a]volume=0.30[a1];[a0][a1]amix=inputs=2:duration=longest:normalize=0[aout]",
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            str(mixed_audio_part_output),
        ],
        mixed_audio_part_output,
        mixed_audio_output,
        {"audio"},
        minimum_duration=_minimum_output_duration(bgm_duration),
    )
    render_command = [
        ffmpeg_binary(),
        "-y",
        "-i",
        str(video_input),
        "-i",
        str(mixed_audio_output),
        "-vf",
        subtitle_filter(video_input, subtitles, session_dir, subtitle_style),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(final_video_part_output),
    ]
    _run_verified_output(
        render_command,
        final_video_part_output,
        final_video_output,
        {"video", "audio"},
        minimum_duration=minimum_video_duration,
        cwd=session_dir,
        native_crash_retry_command=_safe_x264_command(render_command),
    )
    return final_video


def merge_video_with_original_audio(
    video_file: Path,
    translation_file: Path,
    session: Path,
    subtitle_style: SubtitleStyle = DEFAULT_SUBTITLE_STYLE,
) -> Path:
    media_dir = session / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    final_video = media_dir / "video_final.mp4"
    session_dir = session.resolve()
    video_input = video_file.resolve()
    _, video_duration = _probe_media(video_input)
    minimum_video_duration = _minimum_output_duration(video_duration)
    if _reuse_valid_final(final_video, minimum_video_duration):
        return final_video

    orientation = get_video_orientation(video_input)
    subtitles = write_subtitles(translation_file, session, subtitle_style, orientation)
    final_video_output = final_video.resolve()
    final_video_part = media_dir / "video_final.part.mp4"
    final_video_part_output = final_video_part.resolve()
    render_command = [
        ffmpeg_binary(),
        "-y",
        "-i",
        str(video_input),
        "-vf",
        subtitle_filter(video_input, subtitles, session_dir, subtitle_style),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(final_video_part_output),
    ]
    _run_verified_output(
        render_command,
        final_video_part_output,
        final_video_output,
        {"video", "audio"},
        minimum_duration=minimum_video_duration,
        cwd=session_dir,
        native_crash_retry_command=_safe_x264_command(render_command),
    )
    return final_video

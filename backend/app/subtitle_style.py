from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CHINESE_FONT_OPTIONS = (
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
)
ENGLISH_FONT_OPTIONS = (
    "Arial",
    "Segoe UI",
    "Calibri",
    "Times New Roman",
)
DEFAULT_CHINESE_FONT = CHINESE_FONT_OPTIONS[0]
DEFAULT_ENGLISH_FONT = ENGLISH_FONT_OPTIONS[0]
DEFAULT_CHINESE_FONT_SIZE = 12
DEFAULT_ENGLISH_FONT_SIZE = 9
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 36


@dataclass(frozen=True)
class SubtitleStyle:
    chinese_font: str = DEFAULT_CHINESE_FONT
    english_font: str = DEFAULT_ENGLISH_FONT
    chinese_font_size: int = DEFAULT_CHINESE_FONT_SIZE
    english_font_size: int = DEFAULT_ENGLISH_FONT_SIZE


DEFAULT_SUBTITLE_STYLE = SubtitleStyle()


def normalize_subtitle_style(
    chinese_font: str,
    english_font: str,
    chinese_font_size: int,
    english_font_size: int,
) -> SubtitleStyle:
    chinese_font = chinese_font.strip()
    english_font = english_font.strip()
    if chinese_font not in CHINESE_FONT_OPTIONS:
        raise ValueError("Unsupported Chinese subtitle font.")
    if english_font not in ENGLISH_FONT_OPTIONS:
        raise ValueError("Unsupported English subtitle font.")
    for label, value in (
        ("Chinese", chinese_font_size),
        ("English", english_font_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} subtitle font size must be an integer.")
        if value < MIN_FONT_SIZE or value > MAX_FONT_SIZE:
            raise ValueError(
                f"{label} subtitle font size must be between {MIN_FONT_SIZE} and {MAX_FONT_SIZE}."
            )
    return SubtitleStyle(
        chinese_font=chinese_font,
        english_font=english_font,
        chinese_font_size=chinese_font_size,
        english_font_size=english_font_size,
    )


def subtitle_style_from_task(task: dict[str, Any]) -> SubtitleStyle:
    return normalize_subtitle_style(
        str(task.get("subtitle_zh_font") or DEFAULT_CHINESE_FONT),
        str(task.get("subtitle_en_font") or DEFAULT_ENGLISH_FONT),
        int(task.get("subtitle_zh_font_size") or DEFAULT_CHINESE_FONT_SIZE),
        int(task.get("subtitle_en_font_size") or DEFAULT_ENGLISH_FONT_SIZE),
    )

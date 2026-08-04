from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, StrictStr, ValidationError, field_validator

from ..sources import SourceConfig
from ._translate_prompts import PREPROCESS_PROMPT, TRANSLATE_RULES
from .openai_client import normalize_openai_base_url

log = logging.getLogger(__name__)

API_SETTING_KEYS = ("base_url", "api_key", "model")
PREPROCESS_RETRY = 2
TRANSLATE_RETRY = 3
DESCRIPTION_LIMIT = 500
PREPROCESS_DESCRIPTION_LIMIT = 5000
DEFAULT_CONCURRENCY = 50
MAX_CHUNK_CONCURRENCY = 4
TRANSLATE_CHUNK_MAX_SENTENCES = 20
TRANSLATE_CHUNK_MAX_CHARS = 3000
TRANSLATE_CONTEXT_SENTENCES = 3


class HotwordItem(BaseModel):
    src: str
    dst: str


class CorrectionItem(BaseModel):
    wrong: str
    correct: str


class PreprocessResponse(BaseModel):
    translated_title: str = ""
    translated_description: str = ""
    summary: str = ""
    hotwords: list[HotwordItem] = Field(default_factory=list)
    corrections: list[CorrectionItem] = Field(default_factory=list)


class ChunkTranslationItem(BaseModel):
    id: int
    dst: StrictStr

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("translation id must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isascii() and stripped.isdigit():
                return int(stripped)
        raise ValueError("translation id must be an integer or numeric string")


class ChunkTranslationResponse(BaseModel):
    translations: list[ChunkTranslationItem]


class ChunkResponseError(RuntimeError):
    pass


def list_models(*, base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))
    response = client.models.list()
    seen: set[str] = set()
    models: list[str] = []
    for item in response.data:
        model_id = getattr(item, "id", "")
        if model_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_id)
    return models


def _client(base_url: str, api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    return OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise json.JSONDecodeError(f"no JSON object found; raw[:300]={raw[:300]!r}", raw, 0)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg}; len={len(raw)}; raw[:300]={raw[:300]!r}; raw[-200:]={raw[-200:]!r}",
            raw,
            exc.pos,
        ) from None


def _call_json(client: OpenAI, model: str, system: str, user: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    raw = response.choices[0].message.content or "{}"
    return _extract_json(raw)


def _format_terms(items: list, fmt: str, empty: str) -> str:
    if not items:
        return empty
    return "\n".join(fmt.format(**item.model_dump()) for item in items)


def _meta_view(
    meta: dict[str, Any],
    *,
    description_limit: int = DESCRIPTION_LIMIT,
) -> dict[str, str]:
    description = (meta.get("description") or "").strip()
    if len(description) > description_limit:
        description = description[:description_limit] + "..."
    return {
        "title": str(meta.get("title") or "").strip() or "(unknown)",
        "uploader": str(meta.get("uploader") or "").strip() or "(unknown)",
        "description": description or "(none)",
    }


def preprocess(
    full_text: str,
    meta: dict[str, Any],
    source: SourceConfig,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> PreprocessResponse:
    user = PREPROCESS_PROMPT.format(
        src_language_name=source.asr_language_name,
        dst_language_name=source.target_language_name,
        full_text=full_text,
        **_meta_view(meta, description_limit=PREPROCESS_DESCRIPTION_LIMIT),
    )
    client = _client(base_url, api_key)
    last_error: Exception | None = None
    for attempt in range(PREPROCESS_RETRY + 1):
        try:
            data = _call_json(client, model, "You output strict JSON only.", user)
            return PreprocessResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning("preprocess attempt %d failed: %s", attempt + 1, exc)
    log.error("preprocess gave up, returning empty: %s", last_error)
    return PreprocessResponse()


def _translate_system(source: SourceConfig, meta: dict[str, Any], pre: PreprocessResponse) -> str:
    rules = TRANSLATE_RULES[source.target_language]
    return rules.format(
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )


def _post_process(text: str, target_language: str) -> str:
    cleaned = text.strip()
    if target_language == "zh":
        cleaned = cleaned.replace("——", "，")
    return cleaned


def _translation_chunks(texts: list[str]) -> list[list[tuple[int, str]]]:
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for item in enumerate(texts):
        item_chars = len(item[1])
        if current and (
            len(current) >= TRANSLATE_CHUNK_MAX_SENTENCES
            or current_chars + item_chars > TRANSLATE_CHUNK_MAX_CHARS
        ):
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        chunks.append(current)
    return chunks


def _chunk_user_payload(
    chunk: list[tuple[int, str]],
    all_items: list[tuple[int, str]],
) -> str:
    first_id = chunk[0][0]
    last_id = chunk[-1][0]

    def view(items: list[tuple[int, str]]) -> list[dict[str, Any]]:
        return [{"id": item_id, "src": text} for item_id, text in items]

    payload = {
        "context_before": view(
            all_items[max(0, first_id - TRANSLATE_CONTEXT_SENTENCES):first_id]
        ),
        "items": view(chunk),
        "context_after": view(
            all_items[last_id + 1:last_id + 1 + TRANSLATE_CONTEXT_SENTENCES]
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _translate_chunk(
    chunk: list[tuple[int, str]],
    all_items: list[tuple[int, str]],
    target_language: str,
    client: OpenAI,
    model: str,
    system: str,
) -> list[str]:
    user = _chunk_user_payload(chunk, all_items)
    expected_ids = [item_id for item_id, _ in chunk]
    last_error: Exception | None = None
    for attempt in range(TRANSLATE_RETRY):
        try:
            data = _call_json(client, model, system, user)
        except json.JSONDecodeError as exc:
            last_error = exc
        except Exception as exc:
            raise RuntimeError(
                f"translation API request failed for chunk "
                f"{expected_ids[0]}-{expected_ids[-1]}: {exc}"
            ) from exc
        else:
            try:
                response = ChunkTranslationResponse.model_validate(data)
                actual_ids = [item.id for item in response.translations]
                if actual_ids != expected_ids:
                    raise ValueError(
                        f"translation IDs must exactly match {expected_ids}, got {actual_ids}"
                    )
                translations = [
                    _post_process(item.dst, target_language) for item in response.translations
                ]
                if any(not dst for dst in translations):
                    raise ValueError("translation response contains an empty dst")
                return translations
            except (ValidationError, ValueError) as exc:
                last_error = exc
        log.warning(
            "translate chunk %d-%d response attempt %d failed: %s",
            expected_ids[0],
            expected_ids[-1],
            attempt + 1,
            last_error,
        )
    raise ChunkResponseError(
        f"translate chunk {expected_ids[0]}-{expected_ids[-1]} failed "
        f"after {TRANSLATE_RETRY} attempts: {last_error}"
    )


def _translate_chunk_resilient(
    chunk: list[tuple[int, str]],
    all_items: list[tuple[int, str]],
    target_language: str,
    client: OpenAI,
    model: str,
    system: str,
) -> list[str]:
    try:
        return _translate_chunk(
            chunk, all_items, target_language, client, model, system,
        )
    except ChunkResponseError:
        if len(chunk) == 1:
            raise
        midpoint = len(chunk) // 2
        log.warning(
            "Splitting failed translation chunk %d-%d at %d",
            chunk[0][0],
            chunk[-1][0],
            chunk[midpoint][0],
        )
        return _translate_chunk_resilient(
            chunk[:midpoint], all_items, target_language, client, model, system,
        ) + _translate_chunk_resilient(
            chunk[midpoint:], all_items, target_language, client, model, system,
        )


def translate_batch(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[str]:
    if not texts:
        return []
    system = _translate_system(source, meta, pre)
    client = _client(base_url, api_key)
    all_items = list(enumerate(texts))
    chunks = _translation_chunks(texts)
    effective_concurrency = min(max(1, concurrency), MAX_CHUNK_CONCURRENCY, len(chunks))
    log.info(
        "translate_batch: %d sentences in %d chunks, configured_concurrency=%d, "
        "effective_concurrency=%d",
        len(texts), len(chunks), concurrency, effective_concurrency,
    )
    with ThreadPoolExecutor(max_workers=effective_concurrency) as pool:
        translated_chunks = list(pool.map(
            lambda chunk: _translate_chunk_resilient(
                chunk,
                all_items,
                source.target_language,
                client,
                model,
                system,
            ),
            chunks,
        ))
    return [dst for chunk in translated_chunks for dst in chunk]


def _read_meta(session: Path) -> dict[str, Any]:
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        return {}
    return json.loads(info_file.read_text(encoding="utf-8"))


def _speaker(utt: dict[str, Any]) -> str:
    additions = utt.get("additions") or {}
    if isinstance(additions, dict):
        return str(additions.get("speaker") or "1")
    return "1"


def _full_text(data: dict[str, Any], texts: list[str]) -> str:
    raw = data.get("result", {}).get("text") or ""
    if raw.strip():
        return raw
    return " ".join(texts)


def preprocess_artifact_path(session: Path) -> Path:
    return session / "metadata" / "translation_preprocess.json"


def write_preprocess_artifact(session: Path, pre: PreprocessResponse) -> Path:
    path = preprocess_artifact_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pre.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_preprocess_artifact(session: Path) -> PreprocessResponse | None:
    path = preprocess_artifact_path(session)
    if not path.exists():
        return None
    return PreprocessResponse.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _concurrency_from(settings: dict[str, str]) -> int:
    raw = str(settings.get("translate_concurrency") or "").strip()
    if not raw or not all("0" <= char <= "9" for char in raw):
        return DEFAULT_CONCURRENCY
    concurrency = int(raw)
    if concurrency < 1 or concurrency > 200:
        return DEFAULT_CONCURRENCY
    return concurrency


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    if output_file.exists():
        return output_file

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    full_text = _full_text(data, texts)
    meta = _read_meta(session)

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre = load_preprocess_artifact(session)
    if pre is None:
        pre = preprocess(full_text, meta, source, **api)
        write_preprocess_artifact(session, pre)
        log.info("Wrote translation preprocess artifact to %s", preprocess_artifact_path(session))
    else:
        log.info("Reusing translation preprocess artifact from %s", preprocess_artifact_path(session))
    dst_list = translate_batch(
        texts, source, meta, pre, **api, concurrency=_concurrency_from(settings)
    )

    translation = [
        {
            "src": text,
            "dst": dst,
            "src_lang": source.asr_language,
            "dst_lang": source.target_language,
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "speaker": _speaker(utt),
        }
        for text, dst, utt in zip(texts, dst_list, utterances)
    ]
    output_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file

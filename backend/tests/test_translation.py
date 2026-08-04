from __future__ import annotations

import json

import pytest

from backend.app.adapters import openai_translate
from backend.app.adapters.openai_translate import (
    HotwordItem,
    PreprocessResponse,
    CorrectionItem,
)
from backend.app.sources import detect_source


YT_SOURCE = detect_source("https://www.youtube.com/watch?v=abcdefghijk")
BB_SOURCE = detect_source("https://www.bilibili.com/video/BV1xx411c7mD")


def _write_asr(path, n: int, full_text: str | None = None) -> None:
    utterances = [
        {"text": f"S{i}.", "start_time": i * 1000, "end_time": (i + 1) * 1000}
        for i in range(n)
    ]
    payload = {"result": {"utterances": utterances, "text": full_text or " ".join(u["text"] for u in utterances)}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _settings() -> dict[str, str]:
    return {"base_url": "https://example.com/v1", "api_key": "sk-test", "model": "model-x"}


def _stub_preprocess(monkeypatch, response: PreprocessResponse | None = None):
    seen: list[dict] = []

    def fake(full_text, meta, source, **kw):
        seen.append({"full_text": full_text, "meta": meta, "source": source, **kw})
        return response or PreprocessResponse()

    monkeypatch.setattr(openai_translate, "preprocess", fake)
    return seen


def _stub_translate_batch(monkeypatch, transform):
    seen: list[dict] = []

    def fake(texts, source, meta, pre, **kw):
        seen.append({"texts": list(texts), "source": source, "meta": meta, "pre": pre, **kw})
        return [transform(t) for t in texts]

    monkeypatch.setattr(openai_translate, "translate_batch", fake)
    return seen


def _chunk_response(user: str, transform=lambda text: text) -> dict:
    payload = json.loads(user)
    return {
        "translations": [
            {"id": item["id"], "dst": transform(item["src"])}
            for item in payload["items"]
        ]
    }


def test_translate_asr_writes_preprocess_artifact(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr_fixed.json"
    _write_asr(asr_file, 1)

    pre = PreprocessResponse(
        translated_title="寓言 5 深度解析",
        translated_description="本期视频介绍 Fable 5。",
        summary="Video recap",
        hotwords=[HotwordItem(src="Fable 5", dst="Fable 5")],
        corrections=[CorrectionItem(wrong="java script", correct="JavaScript")],
    )
    monkeypatch.setattr(openai_translate, "preprocess", lambda *a, **kw: pre)
    _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    artifact = metadata / "translation_preprocess.json"
    assert artifact.exists()
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["translated_title"] == "寓言 5 深度解析"
    assert saved["translated_description"] == "本期视频介绍 Fable 5。"
    assert saved["summary"] == "Video recap"
    assert saved["hotwords"][0]["src"] == "Fable 5"
    assert saved["corrections"][0]["correct"] == "JavaScript"


def test_translate_asr_reuses_preprocess_artifact_without_calling_api(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr_fixed.json"
    _write_asr(asr_file, 1)
    (metadata / "translation_preprocess.json").write_text(
        json.dumps(
            {
                "summary": "cached",
                "hotwords": [{"src": "GPU", "dst": "GPU"}],
                "corrections": [],
            }
        ),
        encoding="utf-8",
    )

    def fail_preprocess(*args, **kwargs):
        raise AssertionError("preprocess should not run when artifact exists")

    monkeypatch.setattr(openai_translate, "preprocess", fail_preprocess)
    seen = _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert len(seen) == 1
    assert seen[0]["pre"].summary == "cached"
    assert seen[0]["pre"].translated_title == ""
    assert seen[0]["pre"].translated_description == ""


def test_translate_asr_writes_schema_with_speaker_and_lang(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 2)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    items = json.loads(out.read_text(encoding="utf-8"))["translation"]
    assert [i["dst"] for i in items] == ["zh:S0.", "zh:S1."]
    assert {i["src_lang"] for i in items} == {"en"}
    assert {i["dst_lang"] for i in items} == {"zh"}
    assert {i["speaker"] for i in items} == {"1"}
    assert items[0]["start_time"] == 0


def test_translate_asr_output_filename_uses_target_lang(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1)

    _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda _t: "x")

    out = openai_translate.translate_asr(asr_file, tmp_path, _settings(), BB_SOURCE)
    assert out.name == "translation.en.json"


def test_translate_asr_passes_meta_and_full_text_to_preprocess(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 1, full_text="hello world")
    (metadata / "ytdlp_info.json").write_text(
        json.dumps({"title": "T", "uploader": "U", "description": "D"}),
        encoding="utf-8",
    )

    seen = _stub_preprocess(monkeypatch)
    _stub_translate_batch(monkeypatch, lambda t: t)

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert seen[0]["full_text"] == "hello world"
    assert seen[0]["meta"] == {"title": "T", "uploader": "U", "description": "D"}


def test_translate_asr_invokes_translate_batch_with_all_texts_at_once(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    asr_file = metadata / "asr.json"
    _write_asr(asr_file, 5)

    _stub_preprocess(monkeypatch, PreprocessResponse(hotwords=[HotwordItem(src="x", dst="y")]))
    seen = _stub_translate_batch(monkeypatch, lambda t: f"zh:{t}")

    openai_translate.translate_asr(asr_file, tmp_path, _settings(), YT_SOURCE)
    assert len(seen) == 1
    assert seen[0]["texts"] == ["S0.", "S1.", "S2.", "S3.", "S4."]
    assert seen[0]["pre"].hotwords[0].src == "x"


def test_translate_batch_replaces_em_dash_for_zh_target(monkeypatch):
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: _chunk_response(user, lambda _text: "你好——世界"),
    )
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["Hello world."], YT_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m",
    )
    assert out == ["你好，世界"]


def test_translate_batch_does_not_replace_em_dash_for_en_target(monkeypatch):
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: _chunk_response(
            user, lambda _text: "He said—wait—and left."
        ),
    )
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["他说——等等——就走了。"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m",
    )
    assert out == ["He said—wait—and left."]


def test_translate_batch_uses_shared_system_prompt(monkeypatch):
    captured: list[str] = []
    lock = __import__("threading").Lock()

    def fake_call_json(client, model, system, user):
        with lock:
            captured.append(system)
        return _chunk_response(user, lambda text: f"dst:{text}")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = [f"s{i}" for i in range(45)]
    out = openai_translate.translate_batch(
        texts, BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=4,
    )
    assert out == [f"dst:s{i}" for i in range(45)]
    assert len(captured) == 3
    assert len(set(captured)) == 1, "system prompt must be identical across calls for prompt cache"


def test_translation_chunks_enforce_sentence_and_character_limits():
    by_sentence = openai_translate._translation_chunks(["x"] * 41)
    assert [len(chunk) for chunk in by_sentence] == [20, 20, 1]

    by_characters = openai_translate._translation_chunks(["x" * 1000] * 4)
    assert [len(chunk) for chunk in by_characters] == [3, 1]
    assert all(
        sum(len(text) for _, text in chunk) <= 3000
        for chunk in by_characters
    )

    oversized_singleton = openai_translate._translation_chunks(["x" * 3001])
    assert [len(chunk) for chunk in oversized_singleton] == [1]
    assert oversized_singleton[0][0][1] == "x" * 3001


def test_translate_batch_uses_global_ids_and_adjacent_context(monkeypatch):
    requests: list[dict] = []

    def fake_call_json(client, model, system, user):
        payload = json.loads(user)
        requests.append(payload)
        return _chunk_response(user, lambda text: f"dst:{text}")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = [f"s{i}" for i in range(25)]
    out = openai_translate.translate_batch(
        texts, BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=1,
    )

    assert out == [f"dst:s{i}" for i in range(25)]
    assert len(requests) == 2
    assert [item["id"] for item in requests[0]["items"]] == list(range(20))
    assert [item["id"] for item in requests[0]["context_after"]] == [20, 21, 22]
    assert [item["id"] for item in requests[1]["context_before"]] == [17, 18, 19]
    assert [item["id"] for item in requests[1]["items"]] == list(range(20, 25))


def test_translate_batch_retries_when_response_ids_do_not_match(monkeypatch):
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        response = _chunk_response(user)
        if calls["n"] == 1:
            response["translations"].reverse()
        return response

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["one", "two"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=1,
    )
    assert out == ["one", "two"]
    assert calls["n"] == 2


def test_translate_batch_accepts_numeric_string_ids_and_ignores_extra_fields(monkeypatch):
    def fake_call_json(client, model, system, user):
        payload = json.loads(user)
        return {
            "translations": [
                {"id": str(item["id"]), "dst": f"dst:{item['src']}", "note": "ignored"}
                for item in payload["items"]
            ],
            "usage": {"output_tokens": 10},
        }

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    assert openai_translate.translate_batch(
        ["one", "two"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=1,
    ) == ["dst:one", "dst:two"]


def test_translate_batch_does_not_bisect_transport_errors(monkeypatch):
    calls = {"n": 0}

    def fake_call_json(client, model, system, user):
        calls["n"] += 1
        raise ConnectionError("provider unavailable")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    with pytest.raises(RuntimeError, match="translation API request failed"):
        openai_translate.translate_batch(
            ["one", "two", "three", "four"],
            BB_SOURCE,
            {},
            PreprocessResponse(),
            base_url="u",
            api_key="k",
            model="m",
            concurrency=1,
        )
    assert calls["n"] == 1


def test_translate_batch_bisects_a_chunk_after_retries(monkeypatch):
    request_sizes: list[int] = []

    def fake_call_json(client, model, system, user):
        payload = json.loads(user)
        request_sizes.append(len(payload["items"]))
        if len(payload["items"]) > 1:
            return {"translations": []}
        return _chunk_response(user, lambda text: f"dst:{text}")

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    out = openai_translate.translate_batch(
        ["a", "b", "c", "d"], BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=1,
    )
    assert out == ["dst:a", "dst:b", "dst:c", "dst:d"]
    assert request_sizes.count(4) == 3
    assert request_sizes.count(2) == 6
    assert request_sizes.count(1) == 4


def test_translate_batch_caps_effective_chunk_concurrency(monkeypatch):
    seen_workers: list[int] = []

    class RecordingPool:
        def __init__(self, max_workers):
            seen_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, items):
            return map(function, items)

    monkeypatch.setattr(openai_translate, "ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(
        openai_translate,
        "_call_json",
        lambda client, model, system, user: _chunk_response(user),
    )
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    texts = [f"s{i}" for i in range(81)]
    assert openai_translate.translate_batch(
        texts, BB_SOURCE, {}, PreprocessResponse(),
        base_url="u", api_key="k", model="m", concurrency=50,
    ) == texts
    assert seen_workers == [4]


@pytest.mark.parametrize("value", ["abc", "1.5", "0", "-1", "201", ""])
def test_concurrency_from_bad_saved_values_falls_back_to_default(value):
    assert openai_translate._concurrency_from({"translate_concurrency": value}) == 50


def test_preprocess_returns_empty_when_repeatedly_invalid(monkeypatch):
    def fake_call_json(client, model, system, user):
        return {"summary": 123, "hotwords": "bad"}

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    pre = openai_translate.preprocess(
        "text", {"title": "t"}, YT_SOURCE,
        base_url="u", api_key="k", model="m",
    )
    assert pre.summary == ""
    assert pre.hotwords == []
    assert pre.corrections == []


def test_preprocess_requests_and_parses_translated_metadata(monkeypatch):
    captured: dict[str, str] = {}

    def fake_call_json(client, model, system, user):
        captured["user"] = user
        return {
            "translated_title": "Translated title",
            "translated_description": "Translated description",
            "summary": "Summary",
            "hotwords": [],
            "corrections": [],
        }

    monkeypatch.setattr(openai_translate, "_call_json", fake_call_json)
    monkeypatch.setattr(openai_translate, "_client", lambda *a, **kw: object())

    pre = openai_translate.preprocess(
        "text",
        {"title": "Original", "description": "D" * 600},
        YT_SOURCE,
        base_url="u",
        api_key="k",
        model="m",
    )

    assert pre.translated_title == "Translated title"
    assert pre.translated_description == "Translated description"
    assert '"translated_title"' in captured["user"]
    assert '"translated_description"' in captured["user"]
    assert "D" * 600 in captured["user"]


def test_translate_system_prompt_contains_meta_summary_hotwords(monkeypatch):
    pre = PreprocessResponse(
        summary="Recap of the talk.",
        hotwords=[HotwordItem(src="LEGO", dst="乐高")],
    )
    meta = {"title": "Demo", "uploader": "Alice", "description": "Long description"}
    system = openai_translate._translate_system(YT_SOURCE, meta, pre)
    assert "Demo" in system
    assert "Alice" in system
    assert "Long description" in system
    assert "Recap of the talk." in system
    assert "LEGO -> 乐高" in system

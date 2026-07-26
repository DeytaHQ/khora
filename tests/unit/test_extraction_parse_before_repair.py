"""#1563 Part A: parse-before-repair - the extraction storm fix.

The old order ran ``_repair_json`` on every response BEFORE parsing. Its
string-blind ``//``-comment strip amputated any string containing a URL,
corrupting schema-constrained (guaranteed-valid) JSON into "Unterminated
string" parse failures, which a string-match classifier then mislabelled as
truncation and bisected: ~2.2x LLM-call amplification on URL-rich corpora
(three full-tier benchmark ingests died of this).

Contract pinned here:
- valid JSON parses byte-identically regardless of URLs / ``//`` in strings,
- repairs (trailing commas) still fire, but only as a parse-failure fallback,
- genuine truncation is classified by finish_reason only (pre-parse),
- a parse failure on non-truncated output raises (tenacity retry), never
  fabricates a truncated_response,
- terminal unparseable output carries an error key (no silent empty success).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from khora.extraction.extractors.llm import (
    LLMEntityExtractor,
    _parse_llm_json,
    _repair_json,
    _strip_json_fences,
)

URL_PAYLOADS = [
    # single-line: the historical amputation produced "Unterminated string"
    '{"entities": [{"name": "logfire alert", "entity_type": "EVENT", "description": "see https://logfire.dev/alerts/d802 for details"}], "relationships": []}',
    # pretty-printed: the historical amputation produced "Invalid control character"
    '{\n  "entities": [\n    {\n      "name": "cdn",\n      "entity_type": "TECHNOLOGY",\n      "description": "served from //cdn.example.com assets"\n    }\n  ],\n  "relationships": []\n}',
    # bare // mid-string
    '{"entities": [{"name": "a//b path", "entity_type": "CONCEPT", "description": "path a//b"}], "relationships": []}',
]


@pytest.mark.parametrize("payload", URL_PAYLOADS)
def test_url_bearing_valid_json_parses_intact(payload: str) -> None:
    data = _parse_llm_json(payload)
    assert data == json.loads(payload)  # byte-identical semantics, nothing amputated


def test_fenced_url_json_parses() -> None:
    payload = '```json\n{"entities": [], "relationships": [], "note": "https://example.com/x"}\n```'
    assert _parse_llm_json(payload)["note"] == "https://example.com/x"


def test_trailing_comma_still_repaired_via_fallback() -> None:
    assert _parse_llm_json('{"a": [1, 2,],}') == {"a": [1, 2]}


def test_literal_control_char_in_string_accepted() -> None:
    # json_object-mode models sometimes emit raw newlines inside strings.
    payload = '{"description": "line one\nline two"}'
    assert _parse_llm_json(payload)["description"] == "line one\nline two"


def test_nul_stripped_before_parse() -> None:
    payload = '{"name": "abc\x00def"}'
    assert _parse_llm_json(payload)["name"] == "abcdef"  # asyncpg rejects NUL at storage


def test_repair_json_no_longer_strips_comments() -> None:
    # Deliberate: comment-stripping was the corruption mechanism. A genuinely
    # comment-bearing response now fails parse loudly instead.
    s = '{"url": "https://x.example/y"}'
    assert _repair_json(s) == s
    with pytest.raises(json.JSONDecodeError):
        _parse_llm_json('{"a": 1} // comment')


def test_strip_json_fences_is_pure() -> None:
    assert _strip_json_fences('```json\n{"a": "https://x"}\n```') == '{"a": "https://x"}'


# --------------------------------------------------------------------------- #
# Batch classification: truncation by finish_reason only
# --------------------------------------------------------------------------- #


def _batch_response(content: str, finish_reason: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    response.choices = [choice]
    response.model = "test-model"
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return response


@pytest.mark.asyncio
async def test_genuine_truncation_still_classified_by_finish_reason(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
    truncated = '{"sections": [{"entities": [{"name": "x", "entity_'  # cut mid-JSON

    async def fake_acompletion(*args, **kwargs):
        return _batch_response(truncated, "length")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    results = await extractor.extract_multi(
        ["text one", "text two"], batch_size=5, tiered_extraction=False, entity_types=["PERSON", "EVENT"]
    )
    assert all(r.metadata.get("error") == "truncated_response" for r in results)


@pytest.mark.asyncio
async def test_url_valid_json_no_longer_misclassified_as_truncation(monkeypatch) -> None:
    """The regression's cost signature: URL-laden VALID output must produce
    zero truncation classifications and exactly one LLM call."""
    extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=1)
    payload = (
        '{"sections": [{"entities": [{"name": "alert", "entity_type": "EVENT", '
        '"description": "https://logfire.dev/a/1"}], "relationships": []},'
        '{"entities": [], "relationships": []}]}'
    )
    calls = 0

    async def fake_acompletion(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _batch_response(payload, "stop")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    results = await extractor.extract_multi(
        ["text one", "text two"], batch_size=5, tiered_extraction=False, entity_types=["PERSON", "EVENT"]
    )
    assert calls == 1  # no bisection, no retry
    assert all(r.metadata.get("error") is None for r in results)
    assert results[0].entities and results[0].entities[0].name == "alert"


@pytest.mark.asyncio
async def test_corrupt_nontruncated_output_raises_for_retry(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model="gpt-4o-mini", max_retries=2)
    seen = 0

    async def fake_acompletion(*args, **kwargs):
        nonlocal seen
        seen += 1
        if seen == 1:
            return _batch_response('{"sections": [{"broken', "stop")  # corrupt, NOT truncated
        return _batch_response('{"sections": [{"entities": [], "relationships": []}]}', "stop")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    results = await extractor.extract_multi(
        ["only text"], batch_size=5, tiered_extraction=False, entity_types=["PERSON", "EVENT"]
    )
    assert seen == 2  # retried the call instead of fabricating truncated_response
    assert results[0].metadata.get("error") is None


def test_terminal_unparseable_response_carries_error_key() -> None:
    extractor = LLMEntityExtractor(model="gpt-4o-mini")
    result = extractor._extract_json_from_text("no json here at all")
    assert result.metadata.get("error") == "unparseable_response"
    assert not result.entities

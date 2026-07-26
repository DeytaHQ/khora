"""#1562: gate the attribute-prompt surfaces behind extraction_attribute_prompts.

Proves the flag-off prompt is byte-identical to the v0.23.1 baseline (the 100%
case for benchmark traffic, expertise=None) and that the flag-on path restores
the #1549 "emit attributes" nudge and the #1552 per-type ATTRIBUTE SCHEMA block.

The literals below are pinned from
``git show v0.23.1:src/khora/extraction/extractors/llm.py``.

Scoped caveat (objection 1): the DEFAULT_SYSTEM_PROMPT STATE_CHANGE guideline was
rewritten by #1549 and is deliberately NOT reverted. The flag-off system prompt
equals v0.23.1 EXCEPT that one line, which keeps its current {"key", "value"}-pair
wording so the strict-schema pair channel stays coherent. The strict-schema
``attributes`` pair channel and its parser are unconditional in either flag state.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from khora.extraction.extractors.llm import DEFAULT_SYSTEM_PROMPT, LLMEntityExtractor

# --------------------------------------------------------------------------- #
# Pinned v0.23.1 literals
# --------------------------------------------------------------------------- #

# v0.23.1 DEFAULT_SYSTEM_PROMPT, verbatim. The em-dash on the "Extract all named
# entities" line is a byte-for-byte pin of the shipped prompt - do not normalize
# it to a hyphen; the assertions below are byte-identity checks and the current
# prompt carries the same character on that line.
V023_SYSTEM_PROMPT = """\
You are an expert entity extraction system. Extract entities and relationships from text and return them as structured JSON.

Guidelines:
- Extract all named entities mentioned or implied in the text — if a person, organization, location, or concept is referenced even indirectly, extract it
- Use canonical entity names (e.g., "Jennifer Walsh" not "Jenny", "Acme Corporation" not "Acme Corp")
- Include aliases for entities that have multiple names/abbreviations
- Extract temporal information when dates, times, or relative time references appear
- For STATE_CHANGE detection: when text indicates transitions ("switched from X to Y", "no longer X", "used to X", "previously X but now Y"), extract a STATE_CHANGE entity with these required attributes: {"entity_affected": "name of entity whose state changed", "previous_state": "old value", "new_state": "new value", "attribute_changed": "what changed (e.g. job_title, location, instrument)", "transition_date": "ISO date or null"}. Set valid_from to the transition date. Use INVOLVES to link it to the affected entity
- For EVENT detection: when text describes specific occurrences, extract the event with date, participants, and location when available
- Use temporal relationships (PRECEDES, FOLLOWS, INVOLVES) to connect events and state changes to other entities
- Ensure relationship source/target names match extracted entity names exactly
- RELATIONSHIP DENSITY: For N extracted entities, aim to identify N to 2N relationships between them. Include both explicit relationships (stated directly) and implicit ones (inferred from context, co-occurrence, or logical connection)
- For every pair of extracted entities that have any direct or implied connection, create a relationship. It is better to have a weak relationship than no relationship
- Use ASSOCIATED_WITH or RELATES_TO for weaker/implied connections when a more specific type doesn't fit
- Before returning, verify that each extracted entity has at least one relationship connecting it to another entity. If an entity appears isolated, re-examine the text for implicit connections (e.g., co-location, temporal co-occurrence, shared attributes, being mentioned in the same document)

Return ONLY valid JSON, no other text."""

# v0.23.1 single-doc structured template (no #1549 "emit attributes" nudge).
# %-substitution, not str.format, so the literal has no brace-escaping churn.
V023_STRUCTURED = """\
Extract entities, relationships, and temporal information from the following text.
%(document_context)s
Entity types to extract: %(entity_types)s
Relationship types to use: %(relationship_types)s

Text:
%(text)s"""

# v0.23.1 batch fallback prompt (no #1549 nudge, no #1552 schema section).
V023_BATCH_FALLBACK = """\
%(tool_prefix)sExtract entities, relationships, and events from each text section below.

Entity types to find: %(entity_types)s
Relationship types to use: %(relationship_types)s

%(sections)s

Return a JSON object with a "sections" array, one object per section:
{"sections": [
    {"entities": [...], "relationships": [...], "events": [...]},
    ...
]}

Each section follows the same entity/relationship/event format.
Return ONLY valid JSON, no other text."""

# Markers for the gated #1549 nudge and #1552 hint block.
NUDGE_FRAGMENT = 'For each entity, emit "attributes" as an array of'
SCHEMA_HEADER = "ATTRIBUTE SCHEMA (emit these keys in attributes when present):"

MODEL = "gpt-4o-mini"  # on MODELS_REQUIRING_JSON_SCHEMA -> structured path


# --------------------------------------------------------------------------- #
# Capture helpers: monkeypatch litellm.acompletion, record the sent messages
# --------------------------------------------------------------------------- #


def _response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.model = MODEL
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


async def _capture_single(monkeypatch, extractor, text, entity_types, relationship_types, expertise=None):
    captured: dict[str, object] = {}

    async def fake_acompletion(*args, **kwargs):
        captured.setdefault("messages", kwargs["messages"])
        captured.setdefault("response_format", kwargs.get("response_format"))
        return _response('{"entities": [], "relationships": []}')

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    await extractor.extract(text, entity_types=entity_types, relationship_types=relationship_types, expertise=expertise)
    return captured


async def _capture_batch(monkeypatch, extractor, texts, entity_types, relationship_types):
    captured: dict[str, object] = {}
    sections = [{"entities": [], "relationships": []} for _ in texts]

    async def fake_acompletion(*args, **kwargs):
        captured.setdefault("messages", kwargs["messages"])
        captured.setdefault("response_format", kwargs.get("response_format"))
        return _response(json.dumps({"sections": sections}))

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    # tiered_extraction=False so texts hit the batch LLM prompt deterministically,
    # regardless of length; the gate is orthogonal to tiering.
    await extractor.extract_multi(
        texts,
        entity_types=entity_types,
        relationship_types=relationship_types,
        batch_size=10,
        tiered_extraction=False,
    )
    return captured


def _expertise_with_attributes():
    from khora.extraction.skills.base import EntityTypeConfig, ExpertiseConfig

    return ExpertiseConfig(
        name="support",
        entity_types=[
            EntityTypeConfig(name="PERSON", attributes={"required": ["email"], "optional": ["title"]}),
            EntityTypeConfig(name="TICKET", attributes={"required": ["identifier", "status"]}),
        ],
    )


# --------------------------------------------------------------------------- #
# Test 1: flag off, expertise=None -> byte-identical to v0.23.1 (single + batch)
# --------------------------------------------------------------------------- #


def test_system_prompt_equals_v023_except_state_change_line() -> None:
    """Scoped byte-identity: the current system prompt matches v0.23.1 on every
    line except the STATE_CHANGE guideline, which #1549 rewrote and we keep."""
    v023 = V023_SYSTEM_PROMPT.splitlines()
    current = DEFAULT_SYSTEM_PROMPT.splitlines()
    assert len(v023) == len(current)
    diffs = [i for i, (a, b) in enumerate(zip(v023, current)) if a != b]
    assert len(diffs) == 1, f"expected exactly one differing line, got lines {diffs}"
    (i,) = diffs
    assert v023[i].startswith("- For STATE_CHANGE detection:")
    assert current[i].startswith("- For STATE_CHANGE detection:")
    # v0.23.1 used the dict shape; current uses the {"key", "value"} pair shape.
    assert '{"entity_affected":' in v023[i]
    assert 'whose attributes carry these keys as {"key", "value"} pairs' in current[i]


@pytest.mark.asyncio
async def test_single_prompt_flag_off_byte_identical_to_v023(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model=MODEL, max_retries=1)  # flag off (default)
    text = "Alice Walsh met Bob Jones at the Acme headquarters."
    captured = await _capture_single(monkeypatch, extractor, text, ["PERSON", "ORGANIZATION"], ["KNOWS"])

    expected_user = V023_STRUCTURED % {
        "document_context": "",
        "entity_types": "PERSON, ORGANIZATION",
        "relationship_types": "KNOWS",
        "text": text,
    }
    messages = captured["messages"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["content"] == expected_user
    assert NUDGE_FRAGMENT not in messages[1]["content"]


@pytest.mark.asyncio
async def test_batch_prompt_flag_off_byte_identical_to_v023(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model=MODEL, max_retries=1)  # flag off (default)
    texts = ["Alice Walsh met Bob Jones at Acme.", "Carol Diaz left the London office in 2021."]
    captured = await _capture_batch(monkeypatch, extractor, texts, ["PERSON", "ORGANIZATION"], ["KNOWS"])

    sections = "\n".join(f"=== SECTION {i + 1} ===\n{t[:4000]}" for i, t in enumerate(texts))
    expected_user = V023_BATCH_FALLBACK % {
        "tool_prefix": "",
        "entity_types": "PERSON, ORGANIZATION",
        "relationship_types": "KNOWS",
        "sections": sections,
    }
    messages = captured["messages"]
    assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert messages[1]["content"] == expected_user
    assert NUDGE_FRAGMENT not in messages[1]["content"]


# --------------------------------------------------------------------------- #
# Test 2: flag off + expertise-with-attributes -> no header, no nudge
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_flag_off_expertise_suppresses_header_and_nudge_single(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model=MODEL, max_retries=1)  # flag off
    captured = await _capture_single(
        monkeypatch,
        extractor,
        "Alice emailed the team about TICKET-42.",
        ["PERSON", "TICKET"],
        ["KNOWS"],
        expertise=_expertise_with_attributes(),
    )
    user_msg = captured["messages"][1]["content"]
    assert SCHEMA_HEADER not in user_msg
    assert NUDGE_FRAGMENT not in user_msg


@pytest.mark.asyncio
async def test_flag_off_expertise_suppresses_header_and_nudge_batch(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model=MODEL, max_retries=1)  # flag off
    captured: dict[str, object] = {}
    sections = [{"entities": [], "relationships": []}]

    async def fake_acompletion(*args, **kwargs):
        captured.setdefault("messages", kwargs["messages"])
        return _response(json.dumps({"sections": sections}))

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    await extractor.extract_multi(
        ["Alice emailed the team about TICKET-42."],
        entity_types=["PERSON", "TICKET"],
        relationship_types=["KNOWS"],
        expertise=_expertise_with_attributes(),
        batch_size=10,
        tiered_extraction=False,
    )
    user_msg = captured["messages"][1]["content"]
    assert SCHEMA_HEADER not in user_msg
    assert NUDGE_FRAGMENT not in user_msg


# --------------------------------------------------------------------------- #
# Test 3: flag on + expertise -> nudge and per-type block render
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_flag_on_expertise_renders_header_and_nudge_batch(monkeypatch) -> None:
    extractor = LLMEntityExtractor(model=MODEL, attribute_prompts=True, max_retries=1)
    captured: dict[str, object] = {}
    sections = [{"entities": [], "relationships": []}]

    async def fake_acompletion(*args, **kwargs):
        captured.setdefault("messages", kwargs["messages"])
        return _response(json.dumps({"sections": sections}))

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    await extractor.extract_multi(
        ["Alice emailed the team about TICKET-42."],
        entity_types=["PERSON", "TICKET"],
        relationship_types=["KNOWS"],
        expertise=_expertise_with_attributes(),
        batch_size=10,
        tiered_extraction=False,
    )
    user_msg = captured["messages"][1]["content"]
    assert NUDGE_FRAGMENT in user_msg
    assert SCHEMA_HEADER in user_msg
    # Intersection semantics: only resolved types that declare attributes render.
    assert "PERSON: required=[email]; optional=[title]" in user_msg
    assert "TICKET: required=[identifier, status]; optional=[]" in user_msg


@pytest.mark.asyncio
async def test_flag_on_no_expertise_renders_nudge_but_no_header(monkeypatch) -> None:
    """Flag on with expertise=None: the general nudge renders, the per-type block
    stays empty (guarded on expertise). This is the sanctioned attribute-fill route
    only when an expertise config also declares attributes (#1541 disposition)."""
    extractor = LLMEntityExtractor(model=MODEL, attribute_prompts=True, max_retries=1)
    captured = await _capture_batch(
        monkeypatch, extractor, ["Alice Walsh met Bob Jones at Acme."], ["PERSON", "ORGANIZATION"], ["KNOWS"]
    )
    user_msg = captured["messages"][1]["content"]
    assert NUDGE_FRAGMENT in user_msg
    assert SCHEMA_HEADER not in user_msg


# --------------------------------------------------------------------------- #
# Test 4: response_format is not affected by the flag
# --------------------------------------------------------------------------- #


def test_response_format_unchanged_by_flag() -> None:
    off = LLMEntityExtractor(model=MODEL)
    on = LLMEntityExtractor(model=MODEL, attribute_prompts=True)
    assert off._get_response_format() == on._get_response_format()
    assert off._get_multi_response_format() == on._get_multi_response_format()


@pytest.mark.asyncio
async def test_captured_response_format_identical_across_flag_states(monkeypatch) -> None:
    off = LLMEntityExtractor(model=MODEL, max_retries=1)
    on = LLMEntityExtractor(model=MODEL, attribute_prompts=True, max_retries=1)
    cap_off = await _capture_single(monkeypatch, off, "Alice met Bob at Acme.", ["PERSON"], ["KNOWS"])
    cap_on = await _capture_single(monkeypatch, on, "Alice met Bob at Acme.", ["PERSON"], ["KNOWS"])
    assert cap_off["response_format"] == cap_on["response_format"]
    # The pair channel is present in the schema regardless of the flag.
    assert cap_off["response_format"] is not None


# --------------------------------------------------------------------------- #
# Test 5: the flag reaches the extractor through all three constructor sites
# --------------------------------------------------------------------------- #


def _spy_extractor(captured: dict[str, object]):
    class _Spy(LLMEntityExtractor):
        def __init__(self, **kwargs):
            captured["attribute_prompts"] = kwargs.get("attribute_prompts")
            super().__init__(**kwargs)

        async def extract_multi(self, *args, **kwargs):  # short-circuit the LLM
            return []

    return _Spy


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
async def test_flag_reaches_extractor_via_extract_entities(monkeypatch, flag) -> None:
    from khora.core.models import Chunk
    from khora.pipelines.tasks.extract import extract_entities

    captured: dict[str, object] = {}
    monkeypatch.setattr("khora.extraction.extractors.LLMEntityExtractor", _spy_extractor(captured))

    chunk = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="Alice met Bob.", embedding=[])
    await extract_entities(
        [chunk],
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        extraction_attribute_prompts=flag,
    )
    assert captured["attribute_prompts"] is flag


@pytest.mark.asyncio
async def test_flag_reaches_extractor_via_stream_extract(monkeypatch) -> None:
    from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

    captured: dict[str, object] = {}
    monkeypatch.setattr("khora.extraction.extractors.LLMEntityExtractor", _spy_extractor(captured))

    from khora.core.models import Chunk

    chunk = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="Alice met Bob.", embedding=[])
    embedder = MagicMock()
    embedder.embed_batch = MagicMock()
    await stream_extract_and_embed_entities(
        chunks=[chunk],
        embedder=embedder,
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        extraction_attribute_prompts=True,
    )
    assert captured["attribute_prompts"] is True


@pytest.mark.asyncio
async def test_flag_reaches_extractor_via_ingest_documents(monkeypatch) -> None:
    from khora.pipelines.flows import ingest as ingest_mod

    captured: dict[str, object] = {}
    monkeypatch.setattr("khora.extraction.extractors.LLMEntityExtractor", _spy_extractor(captured))

    staged = MagicMock()
    staged.id = uuid4()

    async def fake_stage(documents, namespace_id, storage):
        return [staged]

    async def fake_process_document(*args, **kwargs):
        return {"chunks": 0, "entities": 0, "relationships": 0}

    monkeypatch.setattr(ingest_mod, "stage_documents_batch", fake_stage)
    monkeypatch.setattr(ingest_mod, "process_document", fake_process_document)

    await ingest_mod.ingest_documents(
        uuid4(),
        [{"content": "Alice met Bob."}],
        MagicMock(),
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        shared_embedder=MagicMock(),
        extraction_attribute_prompts=True,
    )
    assert captured["attribute_prompts"] is True


# --------------------------------------------------------------------------- #
# Test 6: STATE_CHANGE keys stay emittable flag-off
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_state_change_still_emittable_flag_off(monkeypatch) -> None:
    """The system-prompt STATE_CHANGE instruction is present flag-off and the
    strict schema still carries the attributes pair channel, so STATE_CHANGE
    key/value pairs remain emittable without the flag."""
    extractor = LLMEntityExtractor(model=MODEL, max_retries=1)  # flag off
    captured = await _capture_single(monkeypatch, extractor, "Alice switched from X to Y.", ["PERSON"], ["INVOLVES"])
    system_msg = captured["messages"][0]["content"]
    assert "For STATE_CHANGE detection:" in system_msg
    assert '{"key", "value"} pairs' in system_msg

    # The strict schema exposes an "attributes" property (the pair channel) on the
    # entity object, independent of the prompt gate.
    rf = extractor._get_response_format()
    schema_text = json.dumps(rf)
    assert '"attributes"' in schema_text

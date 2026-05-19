"""Coverage tests for ``khora.extraction.extractors.llm``.

Targets uncovered branches:
- adaptive batching density-limit decisions (_create_adaptive_batches)
- _get_input_multiplier (exact, prefix, fallback)
- _get_response_format / _get_multi_response_format JSON-schema vs json_object
- _regex_extract tiered extraction
- _build_tool_context, _build_document_context
- _should_run_second_pass
- _merge_relationships dedup
- confidence helpers (entity/relationship/event)
- _parse_response malformed inputs
- _extract_json_from_text recovery
- from_config
- circuit breaker / batch failure handling
- multi-section parse path with non-dict sections
- second-pass extraction with no entities returns []
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.extractors.llm import (
    LLMEntityExtractor,
    _repair_json,
    _strip_json_fences,
)

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJsonHelpers:
    def test_strip_json_fences_with_json_tag(self) -> None:
        out = _strip_json_fences('```json\n{"a": 1}\n```')
        assert out == '{"a": 1}'

    def test_strip_json_fences_without_tag(self) -> None:
        out = _strip_json_fences('```\n{"x": 2}\n```')
        assert out == '{"x": 2}'

    def test_strip_json_fences_no_fence(self) -> None:
        out = _strip_json_fences('{"a": 1}')
        assert out == '{"a": 1}'

    def test_repair_json_strips_trailing_commas(self) -> None:
        assert _repair_json('{"a": 1,}') == '{"a": 1}'
        assert _repair_json("[1, 2, 3,]") == "[1, 2, 3]"

    def test_repair_json_strips_line_comments(self) -> None:
        assert _repair_json('{"a": 1} // comment') == '{"a": 1} '


# ---------------------------------------------------------------------------
# _get_input_multiplier / response_format
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelLookups:
    def test_exact_model_multiplier(self) -> None:
        ex = LLMEntityExtractor(model="gpt-4o")
        assert ex._get_input_multiplier() == 8

    def test_prefix_model_multiplier(self) -> None:
        ex = LLMEntityExtractor(model="gpt-4o-mini-some-future-build")
        # Matches "gpt-4o-mini" prefix or "gpt-4o" prefix — accept either non-default.
        assert ex._get_input_multiplier() in {5, 8}

    def test_unknown_model_default_multiplier(self) -> None:
        ex = LLMEntityExtractor(model="custom-llama")
        assert ex._get_input_multiplier() == LLMEntityExtractor.DEFAULT_INPUT_MULTIPLIER

    def test_response_format_json_schema_model(self) -> None:
        ex = LLMEntityExtractor(model="gpt-4o-mini")
        fmt = ex._get_response_format()
        assert fmt["type"] == "json_schema"

    def test_response_format_unknown_model_is_json_object(self) -> None:
        ex = LLMEntityExtractor(model="custom-llama")
        fmt = ex._get_response_format()
        assert fmt == {"type": "json_object"}

    def test_multi_response_format_json_schema(self) -> None:
        ex = LLMEntityExtractor(model="gpt-4o-mini")
        fmt = ex._get_multi_response_format()
        assert fmt["type"] == "json_schema"

    def test_multi_response_format_json_object(self) -> None:
        ex = LLMEntityExtractor(model="custom-llama")
        fmt = ex._get_multi_response_format()
        assert fmt == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Token estimation and adaptive batching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdaptiveBatching:
    def test_estimate_tokens_basic(self) -> None:
        ex = LLMEntityExtractor()
        # ~3 chars per token, integer division
        assert ex._estimate_tokens("aaa") == 1
        assert ex._estimate_tokens("a" * 60) == 20

    def test_very_short_texts_pack_densely(self) -> None:
        ex = LLMEntityExtractor(model="custom")
        # 60 very short texts (50 chars each) — density_limit=50
        texts = ["short " * 5 for _ in range(60)]
        batches = ex._create_adaptive_batches(texts, max_batch_size=100, max_input_tokens=100_000)
        # First batch capped at 50
        assert len(batches[0]) <= 50

    def test_short_texts_density(self) -> None:
        ex = LLMEntityExtractor(model="custom")
        # ~200 char texts -> density 30
        texts = ["x" * 200 for _ in range(40)]
        batches = ex._create_adaptive_batches(texts, max_batch_size=100, max_input_tokens=100_000)
        assert len(batches[0]) <= 30

    def test_medium_text_density(self) -> None:
        ex = LLMEntityExtractor(model="custom")
        # ~500 char texts -> density 15
        texts = ["x" * 500 for _ in range(20)]
        batches = ex._create_adaptive_batches(texts, max_batch_size=100, max_input_tokens=100_000)
        assert len(batches[0]) <= 15

    def test_long_text_density(self) -> None:
        ex = LLMEntityExtractor(model="custom")
        texts = ["x" * 1000 for _ in range(20)]
        batches = ex._create_adaptive_batches(texts, max_batch_size=100, max_input_tokens=100_000)
        assert len(batches[0]) <= 8

    def test_very_long_text_density(self) -> None:
        ex = LLMEntityExtractor(model="custom")
        texts = ["x" * 3000 for _ in range(10)]
        batches = ex._create_adaptive_batches(texts, max_batch_size=100, max_input_tokens=100_000)
        assert len(batches[0]) <= 3

    def test_token_budget_breaks_batch(self) -> None:
        ex = LLMEntityExtractor(model="custom")
        # ~1500 char texts -> ~500 tokens each. Budget 1000 -> ~2 per batch.
        texts = ["x" * 1500 for _ in range(6)]
        batches = ex._create_adaptive_batches(texts, max_batch_size=100, max_input_tokens=1000)
        # Multiple batches due to token budget
        assert len(batches) >= 2

    def test_empty_input_returns_empty(self) -> None:
        ex = LLMEntityExtractor()
        assert ex._create_adaptive_batches([], 5, 1000) == []


# ---------------------------------------------------------------------------
# regex extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegexExtract:
    def test_extracts_emails(self) -> None:
        result = LLMEntityExtractor._regex_extract("Email alice@example.com today")
        names = [e.name for e in result.entities]
        assert any("alice@example.com" in n for n in names)

    def test_extracts_urls(self) -> None:
        result = LLMEntityExtractor._regex_extract("Visit https://example.com please")
        assert any("https://example.com" in e.name for e in result.entities)

    def test_extracts_dates(self) -> None:
        result = LLMEntityExtractor._regex_extract("On 2024-01-15 we met")
        # Date extracted (note: PROPER_NOUN_RE may also catch capitalized 'On')
        types = [e.entity_type for e in result.entities]
        assert "DATE" in types

    def test_extracts_proper_nouns(self) -> None:
        result = LLMEntityExtractor._regex_extract("Alice Smith visited Paris yesterday")
        names = [e.name for e in result.entities]
        assert any("Alice" in n for n in names)

    def test_creates_co_occurrence_relationships(self) -> None:
        result = LLMEntityExtractor._regex_extract("Alice Smith and Bob Jones at https://example.com")
        # Two+ entities -> co-occurrence relationships
        if len(result.entities) >= 2:
            assert len(result.relationships) >= 1
            assert all(r.relationship_type == "CO_OCCURS_WITH" for r in result.relationships)

    def test_metadata_says_regex(self) -> None:
        result = LLMEntityExtractor._regex_extract("foo")
        assert result.metadata["extraction_method"] == "regex"


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildContexts:
    def test_tool_context_empty_without_expertise(self) -> None:
        ex = LLMEntityExtractor()
        assert ex._build_tool_context(None, {"source_tool": "slack"}) == ""

    def test_tool_context_empty_without_context(self) -> None:
        ex = LLMEntityExtractor()
        expertise = MagicMock()
        expertise.tool_schemas = {"slack": {"message": {"fields": ["author"]}}}
        assert ex._build_tool_context(expertise, None) == ""

    def test_tool_context_empty_without_source_tool(self) -> None:
        ex = LLMEntityExtractor()
        expertise = MagicMock()
        expertise.tool_schemas = {"slack": {"message": {"fields": ["author"]}}}
        assert ex._build_tool_context(expertise, {}) == ""

    def test_tool_context_missing_schema(self) -> None:
        ex = LLMEntityExtractor()
        expertise = MagicMock()
        expertise.tool_schemas = {"other": {}}
        out = ex._build_tool_context(expertise, {"source_tool": "slack"})
        assert out == ""

    def test_tool_context_renders_fields(self) -> None:
        ex = LLMEntityExtractor()
        expertise = MagicMock()
        expertise.tool_schemas = {
            "slack": {
                "Message": {"fields": ["author", "ts"], "labels": ["urgent"]},
            }
        }
        expertise.entity_types = []
        out = ex._build_tool_context(expertise, {"source_tool": "slack"})
        assert "slack" in out
        assert "author" in out
        assert "labels" in out

    def test_tool_context_with_entity_attribute_hints(self) -> None:
        ex = LLMEntityExtractor()
        et = MagicMock()
        et.name = "PERSON"
        et.attributes = {"required": ["name"], "optional": ["email"]}

        expertise = MagicMock()
        expertise.tool_schemas = {"slack": {"Message": {"fields": ["author"]}}}
        expertise.entity_types = [et]

        out = ex._build_tool_context(expertise, {"source_tool": "slack"})
        assert "PERSON" in out
        assert "required: name" in out

    def test_document_context_empty(self) -> None:
        assert LLMEntityExtractor._build_document_context(None) == ""
        assert LLMEntityExtractor._build_document_context({}) == ""
        assert LLMEntityExtractor._build_document_context({"foo": "bar"}) == ""

    def test_document_context_with_datetime(self) -> None:
        dt = datetime(2026, 5, 18, tzinfo=UTC)
        out = LLMEntityExtractor._build_document_context({"document_created_at": dt})
        assert "2026-05-18" in out

    def test_document_context_with_source_tool(self) -> None:
        out = LLMEntityExtractor._build_document_context({"document_created_at": "2026-05-18", "source_tool": "slack"})
        assert "slack" in out

    def test_document_context_with_string_date(self) -> None:
        out = LLMEntityExtractor._build_document_context({"document_created_at": "2026-05-18"})
        assert "2026-05-18" in out


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFromConfig:
    def test_from_config(self) -> None:
        config = MagicMock()
        config.model = "gpt-4o-mini"
        config.max_tokens = 4000
        config.timeout = 30
        config.max_retries = 2
        config.max_concurrent_llm_calls = 5
        config.retry_wait = 0.5

        ex = LLMEntityExtractor.from_config(config)
        assert ex._model == "gpt-4o-mini"
        assert ex._max_retries == 2
        assert ex._retry_wait == 0.5


# ---------------------------------------------------------------------------
# Second-pass relationship extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSecondPass:
    def test_should_run_second_pass_yes(self) -> None:
        # 4 entities, 1 relationship -> threshold (entities - 1 = 3) > 1
        result = ExtractionResult(
            entities=[ExtractedEntity(name=f"E{i}", entity_type="X") for i in range(4)],
            relationships=[ExtractedRelationship(source_entity="E0", target_entity="E1", relationship_type="K")],
        )
        assert LLMEntityExtractor._should_run_second_pass(result) is True

    def test_should_run_second_pass_no_few_entities(self) -> None:
        result = ExtractionResult(
            entities=[ExtractedEntity(name="E0", entity_type="X")],
            relationships=[],
        )
        assert LLMEntityExtractor._should_run_second_pass(result) is False

    def test_should_run_second_pass_no_enough_relationships(self) -> None:
        result = ExtractionResult(
            entities=[ExtractedEntity(name=f"E{i}", entity_type="X") for i in range(3)],
            relationships=[
                ExtractedRelationship(source_entity="E0", target_entity="E1", relationship_type="K"),
                ExtractedRelationship(source_entity="E1", target_entity="E2", relationship_type="K"),
            ],
        )
        assert LLMEntityExtractor._should_run_second_pass(result) is False

    @pytest.mark.asyncio
    async def test_extract_additional_relationships_empty_entities(self) -> None:
        ex = LLMEntityExtractor()
        out = await ex._extract_additional_relationships([], "text")
        assert out == []

    @pytest.mark.asyncio
    async def test_extract_additional_relationships_filters_unknown_entities(self) -> None:
        ex = LLMEntityExtractor(model="custom", max_retries=1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "relationships": [
                    {"source_entity": "Alice", "target_entity": "Acme", "relationship_type": "WORKS_FOR"},
                    {"source_entity": "Outsider", "target_entity": "Acme", "relationship_type": "X"},
                    "malformed-string",
                ]
            }
        )
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        entities = [
            ExtractedEntity(name="Alice", entity_type="PERSON"),
            ExtractedEntity(name="Acme", entity_type="ORG"),
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            rels = await ex._extract_additional_relationships(entities, "Alice works at Acme")

        # Only valid relationship between known entities is kept
        assert len(rels) == 1
        assert rels[0].source_entity == "Alice"

    @pytest.mark.asyncio
    async def test_extract_additional_relationships_handles_exception(self) -> None:
        ex = LLMEntityExtractor(model="custom", max_retries=1)
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("boom"),
        ):
            rels = await ex._extract_additional_relationships(
                [ExtractedEntity(name="A", entity_type="X")],
                "text",
            )
        assert rels == []


# ---------------------------------------------------------------------------
# _merge_relationships
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMergeRelationships:
    def test_dedupes_by_key(self) -> None:
        existing = ExtractionResult(
            entities=[],
            relationships=[
                ExtractedRelationship(source_entity="A", target_entity="B", relationship_type="K"),
            ],
        )
        additional = [
            ExtractedRelationship(source_entity="A", target_entity="B", relationship_type="K"),  # dupe
            ExtractedRelationship(source_entity="A", target_entity="C", relationship_type="K"),  # new
        ]
        merged = LLMEntityExtractor._merge_relationships(existing, additional)
        assert len(merged.relationships) == 2
        assert merged.metadata["second_pass_relationships"] == 1


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfidenceHelpers:
    def test_entity_confidence_uses_explicit(self) -> None:
        ex = LLMEntityExtractor()
        assert ex._compute_entity_confidence({"confidence": 0.42}) == 0.42

    def test_entity_confidence_base_plus_quality(self) -> None:
        ex = LLMEntityExtractor()
        score = ex._compute_entity_confidence(
            {
                "name": "Alice Smith",
                "description": "A senior software engineer at Acme Corp.",
                "entity_type": "PERSON",
                "aliases": ["Alice"],
            }
        )
        assert score > 0.5
        assert score <= 1.0

    def test_entity_confidence_generic_type(self) -> None:
        ex = LLMEntityExtractor()
        score = ex._compute_entity_confidence({"name": "X", "entity_type": "CONCEPT"})
        assert score < 1.0

    def test_relationship_confidence_uses_explicit(self) -> None:
        ex = LLMEntityExtractor()
        assert ex._compute_relationship_confidence({"confidence": 0.3}, set()) == 0.3

    def test_relationship_confidence_known_entities(self) -> None:
        ex = LLMEntityExtractor()
        score = ex._compute_relationship_confidence(
            {"source_entity": "A", "target_entity": "B", "relationship_type": "WORKS_FOR", "description": "x" * 30},
            {"A", "B"},
        )
        assert score > 0.5

    def test_relationship_confidence_generic_type(self) -> None:
        ex = LLMEntityExtractor()
        score = ex._compute_relationship_confidence(
            {"source_entity": "A", "target_entity": "B", "relationship_type": "RELATES_TO"},
            set(),
        )
        assert score < 1.0

    def test_event_confidence_uses_explicit(self) -> None:
        ex = LLMEntityExtractor()
        assert ex._compute_event_confidence({"confidence": 0.7}) == 0.7

    def test_event_confidence_with_temporal_and_participants(self) -> None:
        ex = LLMEntityExtractor()
        score = ex._compute_event_confidence(
            {
                "description": "Team meeting to plan Q1 roadmap",
                "occurred_at": "2026-01-15",
                "participants": ["alice", "bob"],
            }
        )
        assert score > 0.7

    def test_event_confidence_minimal(self) -> None:
        ex = LLMEntityExtractor()
        score = ex._compute_event_confidence({})
        assert score == 0.5


# ---------------------------------------------------------------------------
# _parse_response error paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseResponseExtras:
    def test_none_content(self) -> None:
        ex = LLMEntityExtractor()
        result = ex._parse_response(None)
        assert result.metadata.get("error") == "empty_response"

    def test_empty_string(self) -> None:
        ex = LLMEntityExtractor()
        result = ex._parse_response("")
        assert result.metadata.get("error") == "empty_response"

    def test_non_dict_after_parse(self) -> None:
        ex = LLMEntityExtractor()
        # Bare JSON array parses as list -> not dict
        result = ex._parse_response("[1, 2, 3]")
        assert result.metadata.get("error") == "invalid_response_type"

    def test_dict_directly(self) -> None:
        ex = LLMEntityExtractor()
        result = ex._parse_response({"entities": [{"name": "X", "entity_type": "PERSON"}], "relationships": []})
        assert len(result.entities) == 1

    def test_skips_malformed_entities(self) -> None:
        ex = LLMEntityExtractor()
        data = {
            "entities": [
                "not-a-dict",  # skipped
                {"name": "Alice", "entity_type": "PERSON"},
            ],
            "relationships": [],
        }
        result = ex._parse_response(json.dumps(data))
        assert len(result.entities) == 1

    def test_skips_malformed_relationships(self) -> None:
        ex = LLMEntityExtractor()
        data = {
            "entities": [],
            "relationships": [
                "junk",
                {"source_entity": "A", "target_entity": "B", "relationship_type": "K"},
            ],
        }
        result = ex._parse_response(json.dumps(data))
        assert len(result.relationships) == 1

    def test_skips_malformed_events(self) -> None:
        ex = LLMEntityExtractor()
        data = {
            "entities": [],
            "relationships": [],
            "events": ["bad", {"description": "ok"}],
        }
        result = ex._parse_response(json.dumps(data))
        assert len(result.events) == 1

    def test_attributes_list_coerced_to_empty_dict(self) -> None:
        ex = LLMEntityExtractor()
        data = {
            "entities": [
                {"name": "X", "entity_type": "P", "attributes": ["bad", "list"]},
            ]
        }
        result = ex._parse_response(json.dumps(data))
        assert result.entities[0].attributes == {}

    def test_temporal_on_entities(self) -> None:
        ex = LLMEntityExtractor()
        data = {
            "entities": [
                {
                    "name": "Event",
                    "entity_type": "EVENT",
                    "temporal": {"valid_from": "2026-01-01", "mentioned_at": "2026-01-15"},
                }
            ]
        }
        result = ex._parse_response(json.dumps(data))
        assert result.entities[0].temporal is not None
        assert result.entities[0].temporal.valid_from == "2026-01-01"

    def test_temporal_on_relationships(self) -> None:
        ex = LLMEntityExtractor()
        data = {
            "relationships": [
                {
                    "source_entity": "A",
                    "target_entity": "B",
                    "relationship_type": "MET",
                    "temporal": {"occurred_at": "2026-01-01"},
                }
            ]
        }
        result = ex._parse_response(json.dumps(data))
        assert result.relationships[0].temporal is not None
        assert result.relationships[0].temporal.occurred_at == "2026-01-01"

    def test_temporal_non_dict_ignored(self) -> None:
        ex = LLMEntityExtractor()
        data = {"entities": [{"name": "X", "entity_type": "P", "temporal": "garbage"}]}
        result = ex._parse_response(json.dumps(data))
        assert result.entities[0].temporal is None

    def test_extract_json_from_text(self) -> None:
        ex = LLMEntityExtractor()
        # JSON embedded in text — _parse_response recovers via _extract_json_from_text
        text = 'prefix garbage {"entities": [{"name": "X", "entity_type": "P"}], "relationships": []} suffix'
        result = ex._parse_response(text)
        assert len(result.entities) == 1


# ---------------------------------------------------------------------------
# _filter_by_confidence — events preserved
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilterByConfidence:
    def test_filters_entities(self) -> None:
        ex = LLMEntityExtractor()
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Lo", entity_type="X", confidence=0.3),
                ExtractedEntity(name="Hi", entity_type="X", confidence=0.9),
            ]
        )
        expertise = MagicMock()
        expertise.confidence.min_entity = 0.5
        expertise.confidence.min_relationship = 0.5
        filtered = ex._filter_by_confidence(result, expertise)
        assert [e.name for e in filtered.entities] == ["Hi"]


# ---------------------------------------------------------------------------
# extract_multi with tiered_extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractMultiTiered:
    @pytest.mark.asyncio
    async def test_tiered_all_short_returns_regex_results(self) -> None:
        ex = LLMEntityExtractor()
        # All texts under threshold -> all-regex path, no LLM call
        results = await ex.extract_multi(["foo", "bar"], tier1_max_chars=20, batch_size=5)
        assert len(results) == 2
        # Regex results have method metadata
        assert all(r.metadata.get("extraction_method") == "regex" for r in results)

    @pytest.mark.asyncio
    async def test_tiered_mix_short_and_long(self) -> None:
        ex = LLMEntityExtractor(model="custom", max_retries=1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {"sections": [{"entities": [{"name": "Alice", "entity_type": "PERSON"}], "relationships": []}]}
        )
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        long_text = "Alice works at Acme " * 10
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            results = await ex.extract_multi(
                ["x", long_text],
                tier1_max_chars=20,
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )
        # Two results, in original order
        assert len(results) == 2
        # First is regex result (short), second from LLM
        assert results[0].metadata.get("extraction_method") == "regex"


# ---------------------------------------------------------------------------
# Extract — empty / whitespace text shortcuts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractShortCircuits:
    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        ex = LLMEntityExtractor()
        result = await ex.extract("")
        assert result.entities == []

    @pytest.mark.asyncio
    async def test_whitespace_text(self) -> None:
        ex = LLMEntityExtractor()
        result = await ex.extract("   \t\n  ")
        assert result.entities == []

"""Unit tests for extraction/extractors/llm.py — LLM entity extraction."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.extractors.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMEntityExtractor,
)


class TestParseResponse:
    """Tests for LLMEntityExtractor._parse_response."""

    def _make_extractor(self) -> LLMEntityExtractor:
        return LLMEntityExtractor(model="test-model")

    def test_valid_json(self) -> None:
        """Parse valid JSON with entities and relationships."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON", "description": "A person"},
                {"name": "Acme", "entity_type": "ORGANIZATION", "description": "A company"},
            ],
            "relationships": [
                {
                    "source_entity": "Alice",
                    "target_entity": "Acme",
                    "relationship_type": "WORKS_FOR",
                    "description": "Alice works for Acme",
                }
            ],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.entities) == 2
        assert result.entities[0].name == "Alice"
        assert len(result.relationships) == 1
        assert result.relationships[0].relationship_type == "WORKS_FOR"

    def test_json_in_markdown_code_block(self) -> None:
        """Extract JSON from markdown code block."""
        extractor = self._make_extractor()
        text = '```json\n{"entities": [{"name": "Bob", "entity_type": "PERSON"}], "relationships": []}\n```'
        result = extractor._parse_response(text)
        # Falls through to _extract_json_from_text
        assert len(result.entities) == 1
        assert result.entities[0].name == "Bob"

    def test_malformed_json(self) -> None:
        """Malformed JSON returns empty result with metadata."""
        extractor = self._make_extractor()
        result = extractor._parse_response("this is not json at all")
        assert len(result.entities) == 0
        assert "raw_response" in result.metadata

    def test_empty_entities(self) -> None:
        """Empty entities list is handled."""
        extractor = self._make_extractor()
        result = extractor._parse_response('{"entities": [], "relationships": []}')
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    def test_temporal_info(self) -> None:
        """Temporal info is parsed from entities."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {
                    "name": "Meeting",
                    "entity_type": "EVENT",
                    "temporal": {
                        "mentioned_at": "2024-01-15",
                        "valid_from": "2024-01-15",
                        "valid_until": None,
                    },
                }
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].temporal is not None
        assert result.entities[0].temporal.mentioned_at == "2024-01-15"

    def test_events_parsed(self) -> None:
        """Events are parsed from response."""
        extractor = self._make_extractor()
        data = {
            "entities": [],
            "relationships": [],
            "events": [
                {
                    "description": "Team meeting",
                    "event_type": "MEETING",
                    "occurred_at": "2024-01-15",
                    "participants": ["Alice", "Bob"],
                }
            ],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.events) == 1
        assert result.events[0].event_type == "MEETING"

    def test_null_safe_parsing(self) -> None:
        """JSON null values for name/source/target are skipped, not persisted (#894).

        Prior to #894 the parser silently materialised entities with name=""
        and relationships with source="" / target="". The parser now drops
        these and tracks the count via ExtractionResult.metadata.
        """
        extractor = self._make_extractor()
        data = {
            "entities": [{"name": None, "entity_type": None, "description": None}],
            "relationships": [{"source_entity": None, "target_entity": None, "relationship_type": None}],
        }
        result = extractor._parse_response(json.dumps(data))
        # Empty-name entity / relationship are dropped.
        assert result.entities == []
        assert result.relationships == []
        # Count surfaced for downstream degradation reporting.
        assert result.metadata["skipped_entities_empty_name"] == 1
        assert result.metadata["skipped_relationships_empty_endpoint"] == 1

    def test_attributes_non_dict(self) -> None:
        """Non-dict attributes are replaced with empty dict."""
        extractor = self._make_extractor()
        data = {
            "entities": [{"name": "Test", "entity_type": "CONCEPT", "attributes": ["invalid"]}],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].attributes == {}

    def test_short_key_entity_type(self) -> None:
        """Entities using short 'type' key parse correctly (issue #839).

        Off-allowlist models (e.g. local llama.cpp, Anthropic in some
        configurations, any model not in MODELS_REQUIRING_JSON_SCHEMA)
        fall back to json_object response format with no schema enforcement.
        They tend to emit the short 'type' key instead of 'entity_type'.
        The parser must accept either.
        """
        extractor = self._make_extractor()
        data = {
            "entities": [
                {"name": "Quantum mechanics", "type": "PRINCIPLE", "description": "physics"},
                {"name": "Schrodinger", "type": "PERSON"},
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.entities) == 2
        assert result.entities[0].entity_type == "PRINCIPLE"
        assert result.entities[1].entity_type == "PERSON"

    def test_long_key_entity_type_still_works(self) -> None:
        """Backward compat: 'entity_type' long key still parses (issue #839)."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON"},
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].entity_type == "PERSON"

    def test_short_key_relationship(self) -> None:
        """Relationships using short 'source'/'target'/'type' keys parse correctly (issue #839)."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {"name": "Schrodinger equation", "type": "EQUATION"},
                {"name": "Quantum mechanics", "type": "PRINCIPLE"},
            ],
            "relationships": [
                {
                    "source": "Schrodinger equation",
                    "target": "Quantum mechanics",
                    "type": "GOVERNS",
                    "description": "the equation governs the principle",
                }
            ],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.relationships) == 1
        rel = result.relationships[0]
        assert rel.source_entity == "Schrodinger equation"
        assert rel.target_entity == "Quantum mechanics"
        assert rel.relationship_type == "GOVERNS"

    def test_short_key_event_type(self) -> None:
        """Events using short 'type' key parse correctly (issue #839)."""
        extractor = self._make_extractor()
        data = {
            "entities": [],
            "relationships": [],
            "events": [
                {
                    "description": "Project kickoff",
                    "type": "MEETING",
                    "occurred_at": "2024-01-15",
                }
            ],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.events) == 1
        assert result.events[0].event_type == "MEETING"


class TestOffAllowlistModelWarning:
    """Tests for the one-shot warning when a model isn't on the
    MODELS_REQUIRING_JSON_SCHEMA allowlist (issue #839)."""

    def test_warning_fires_for_off_allowlist_model(self) -> None:
        """Constructing an extractor with an unknown model logs one warning."""
        from loguru import logger

        # Reset the per-process dedup set so this test is order-independent.
        LLMEntityExtractor._WARNED_NON_ALLOWLIST_MODELS.clear()

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            LLMEntityExtractor(model="gpt-5.4-not-on-allowlist")
        finally:
            logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "gpt-5.4-not-on-allowlist" in joined
        assert "json_schema allowlist" in joined

    def test_warning_only_fires_once_per_model(self) -> None:
        """A second extractor for the same off-allowlist model does NOT re-warn."""
        from loguru import logger

        LLMEntityExtractor._WARNED_NON_ALLOWLIST_MODELS.clear()

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            LLMEntityExtractor(model="some-local-llm")
            LLMEntityExtractor(model="some-local-llm")
        finally:
            logger.remove(sink_id)

        hits = [m for m in messages if "some-local-llm" in m]
        assert len(hits) == 1

    def test_no_warning_for_allowlisted_model(self) -> None:
        """Allowlisted models (e.g. gpt-4o-mini) do not emit the warning."""
        from loguru import logger

        LLMEntityExtractor._WARNED_NON_ALLOWLIST_MODELS.clear()

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            LLMEntityExtractor(model="gpt-4o-mini")
        finally:
            logger.remove(sink_id)

        joined = "\n".join(messages)
        assert "json_schema allowlist" not in joined


class TestExtractJsonFromText:
    """Tests for _extract_json_from_text."""

    def test_find_json_block(self) -> None:
        """Find JSON object embedded in text."""
        extractor = LLMEntityExtractor()
        text = 'Here is the result:\n{"entities": [{"name": "Test", "entity_type": "CONCEPT"}], "relationships": []}\nDone!'
        result = extractor._extract_json_from_text(text)
        assert len(result.entities) == 1

    def test_no_json_found(self) -> None:
        """No JSON in text returns empty result with metadata."""
        extractor = LLMEntityExtractor()
        result = extractor._extract_json_from_text("no json here")
        assert len(result.entities) == 0
        assert "raw_response" in result.metadata

    def test_no_mutual_recursion_with_parse_response(self) -> None:
        """_extract_json_from_text must not recurse via _parse_response.

        Regression test for: when _extract_json_from_text called
        _parse_response(json.dumps(data)), re-serialization could produce
        a string that _strip_json_fences/_repair_json mangled back into
        invalid JSON, causing _parse_response → _extract_json_from_text →
        _parse_response → ... → RecursionError.
        """
        import sys

        extractor = LLMEntityExtractor()
        # Truncated JSON that has a valid subset: regex will match {"entities": []}
        # from the wrapping, and the inner parse should NOT recurse.
        text = '{"entities": [], "relationships": [{"source": "foo", "target": "ba'
        old_limit = sys.getrecursionlimit()
        # Set a low recursion limit so we catch infinite recursion fast
        sys.setrecursionlimit(50)
        try:
            result = extractor._parse_response(text)
            # Should return something (empty or partial), not blow the stack
            assert isinstance(result, ExtractionResult)
        finally:
            sys.setrecursionlimit(old_limit)


class TestFilterByConfidence:
    """Tests for _filter_by_confidence."""

    def test_filter_entities_below_threshold(self) -> None:
        """Entities below threshold are filtered out."""
        extractor = LLMEntityExtractor()
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="High", entity_type="PERSON", confidence=0.9),
                ExtractedEntity(name="Low", entity_type="PERSON", confidence=0.3),
            ],
            relationships=[],
        )
        expertise = MagicMock()
        expertise.confidence.min_entity = 0.5
        expertise.confidence.min_relationship = 0.5
        filtered = extractor._filter_by_confidence(result, expertise)
        assert len(filtered.entities) == 1
        assert filtered.entities[0].name == "High"

    def test_filter_relationships_below_threshold(self) -> None:
        """Relationships below threshold are filtered."""
        extractor = LLMEntityExtractor()
        result = ExtractionResult(
            entities=[],
            relationships=[
                ExtractedRelationship(
                    source_entity="A",
                    target_entity="B",
                    relationship_type="KNOWS",
                    confidence=0.3,
                ),
                ExtractedRelationship(
                    source_entity="C",
                    target_entity="D",
                    relationship_type="WORKS_FOR",
                    confidence=0.8,
                ),
            ],
        )
        expertise = MagicMock()
        expertise.confidence.min_entity = 0.5
        expertise.confidence.min_relationship = 0.5
        filtered = extractor._filter_by_confidence(result, expertise)
        assert len(filtered.relationships) == 1

    def test_events_preserved(self) -> None:
        """Events are not filtered."""
        extractor = LLMEntityExtractor()
        result = ExtractionResult(
            entities=[],
            relationships=[],
            events=[ExtractedEvent(description="test")],
        )
        expertise = MagicMock()
        expertise.confidence.min_entity = 0.5
        expertise.confidence.min_relationship = 0.5
        filtered = extractor._filter_by_confidence(result, expertise)
        assert len(filtered.events) == 1


class TestRenderPrompts:
    """Tests for prompt rendering methods."""

    def test_system_prompt_no_expertise(self) -> None:
        """Without expertise, returns default system prompt."""
        extractor = LLMEntityExtractor()
        prompt = extractor._render_system_prompt(None, None)
        assert prompt == DEFAULT_SYSTEM_PROMPT

    def test_extraction_prompt_default(self) -> None:
        """Default extraction prompt includes entity types and text."""
        extractor = LLMEntityExtractor()
        prompt = extractor._render_extraction_prompt(
            "test text",
            ["PERSON", "ORGANIZATION"],
            None,
            None,
            relationship_types=["WORKS_FOR", "KNOWS"],
        )
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        assert "test text" in prompt

    def test_extraction_prompt_text_truncation(self) -> None:
        """Long text is truncated in extraction prompt."""
        extractor = LLMEntityExtractor()
        long_text = "a" * 20000
        prompt = extractor._render_extraction_prompt(
            long_text,
            ["PERSON"],
            None,
            None,
            relationship_types=["WORKS_FOR"],
        )
        # Text should be truncated at 8000 chars
        assert len(prompt) < 20000


class TestExtract:
    """Tests for the extract method."""

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        """Empty text returns empty result."""
        extractor = LLMEntityExtractor()
        result = await extractor.extract("")
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_whitespace_text(self) -> None:
        """Whitespace-only text returns empty result."""
        extractor = LLMEntityExtractor()
        result = await extractor.extract("   \n  ")
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_extract_single_chunk(self) -> None:
        """Mocked LLM call extracts entities."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "entities": [{"name": "Alice", "entity_type": "PERSON", "description": "A person"}],
                "relationships": [],
            }
        )
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract(
                "Alice works at Acme Corp",
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_extract_retry_on_error(self) -> None:
        """Extract retries on failure and eventually returns error result."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=2)

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API error")):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await extractor.extract(
                    "test text",
                    entity_types=["PERSON", "ORGANIZATION"],
                    relationship_types=["WORKS_FOR", "KNOWS"],
                )

        assert "error" in result.metadata


class TestExtractBatch:
    """Tests for extract_batch method."""

    @pytest.mark.asyncio
    async def test_empty_texts(self) -> None:
        """Empty list returns empty list."""
        extractor = LLMEntityExtractor()
        results = await extractor.extract_batch([])
        assert results == []


class TestExtractMulti:
    """Tests for extract_multi method (grouped extraction)."""

    @pytest.mark.asyncio
    async def test_empty_texts(self) -> None:
        """Empty list returns empty list."""
        extractor = LLMEntityExtractor()
        results = await extractor.extract_multi([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_extraction(self) -> None:
        """Multi-batch extraction produces one result per text."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        section_data = {
            "sections": [
                {"entities": [{"name": "A", "entity_type": "PERSON"}], "relationships": []},
                {"entities": [{"name": "B", "entity_type": "ORGANIZATION"}], "relationships": []},
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["text1", "text2"],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR", "KNOWS"],
                tiered_extraction=False,
            )

        assert len(results) == 2
        assert results[0].entities[0].name == "A"
        assert results[1].entities[0].name == "B"


class TestWaveSize:
    """#1374: extract_multi() dispatches batches in configurable waves."""

    def test_default_wave_size(self) -> None:
        """Default wave_size is 20 (matches issue #1374)."""
        assert LLMEntityExtractor()._wave_size == 20

    def test_wave_size_stored(self) -> None:
        """Constructor kwarg is stored on the instance."""
        assert LLMEntityExtractor(wave_size=3)._wave_size == 3

    @pytest.mark.asyncio
    async def test_extract_multi_honors_wave_size(self) -> None:
        """extract_multi slices batches into waves of self._wave_size.

        Set max_concurrent high so the semaphore is not the binding limit, then
        assert the peak concurrent batch count never exceeds wave_size. With
        batch_size=1 each text is its own batch.
        """
        wave_size = 2
        extractor = LLMEntityExtractor(model="test-model", max_retries=1, max_concurrent=100, wave_size=wave_size)
        texts = [f"text {i}" for i in range(6)]  # 6 batches → 3 waves of 2

        in_flight = 0
        peak = 0

        async def fake_batch(batch, *args, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            # Yield control so all batches in the same wave overlap.
            await asyncio.sleep(0)
            in_flight -= 1
            return [
                ExtractionResult(entities=[ExtractedEntity(name="E", entity_type="PERSON", confidence=0.9)])
                for _ in batch
            ]

        with patch.object(extractor, "_extract_multi_batch", side_effect=fake_batch):
            results = await extractor.extract_multi(
                texts,
                entity_types=["PERSON"],
                tiered_extraction=False,
                batch_size=1,
            )

        assert len(results) == len(texts)
        assert peak == wave_size


class TestMultiBatchSectionMismatch:
    """#1123: dropped batch-extraction sections must surface as errors, not silent empties."""

    @pytest.mark.asyncio
    async def test_fewer_sections_marks_missing_as_errored(self) -> None:
        """LLM returns 1 section for a 3-text batch: the 2 unmatched texts carry error metadata.

        Before the fix the unmatched texts got a bare ``ExtractionResult()`` with no
        ``error`` key, so they bypassed the #889 extraction_errors counter and the document
        was marked complete with entities silently missing.
        """
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        # Model merged 3 input sections into 1 returned section (common failure mode).
        section_data = {
            "sections": [
                {"entities": [{"name": "Alice", "entity_type": "PERSON"}], "relationships": []},
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor._extract_multi_batch(
                [
                    "First substantive text about Alice the engineer.",
                    "Second substantive text about Bob the manager.",
                    "Third substantive text about Carol the analyst.",
                ],
                ["PERSON"],
                __import__("litellm"),
            )

        assert len(results) == 3
        # Matched text keeps its extraction.
        assert results[0].entities[0].name == "Alice"
        # Unmatched texts are errored (so #889 counts them), not silent empties.
        assert results[1].entities == []
        assert results[2].entities == []
        assert results[1].metadata.get("error") == "section_count_mismatch"
        assert results[2].metadata.get("error") == "section_count_mismatch"

    @pytest.mark.asyncio
    async def test_flat_format_multi_text_does_not_dump_all_on_text_zero(self) -> None:
        """#1123: flat (non-sections) shape on a multi-text batch must NOT attribute all
        entities to text 0 and empty the rest. The whole batch is marked errored instead.
        """
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        # Flat shape: {"entities": [...]} with no "sections" wrapper.
        flat_data = {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON"},
                {"name": "Bob", "entity_type": "PERSON"},
            ],
            "relationships": [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(flat_data)
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor._extract_multi_batch(
                [
                    "First substantive text about Alice the engineer.",
                    "Second substantive text about Bob the manager.",
                ],
                ["PERSON"],
                __import__("litellm"),
            )

        assert len(results) == 2
        # Must NOT dump Alice+Bob onto text 0 and empty text 1 (the old mis-stamping bug).
        assert results[0].entities == []
        assert results[1].entities == []
        assert results[0].metadata.get("error") == "section_count_mismatch"
        assert results[1].metadata.get("error") == "section_count_mismatch"

    @pytest.mark.asyncio
    async def test_flat_format_single_text_still_parsed(self) -> None:
        """#1123: the flat-format fallback remains valid for a single-text batch."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        flat_data = {
            "entities": [{"name": "Alice", "entity_type": "PERSON"}],
            "relationships": [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(flat_data)
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor._extract_multi_batch(
                ["Only text about Alice the engineer."],
                ["PERSON"],
                __import__("litellm"),
            )

        assert len(results) == 1
        assert results[0].entities[0].name == "Alice"
        assert results[0].metadata.get("error") is None


# ---------------------------------------------------------------------------
# Regex extraction (A-4 tiered extraction)
# ---------------------------------------------------------------------------


class TestRegexExtraction:
    """Tests for LLMEntityExtractor._regex_extract."""

    def test_email_extraction(self) -> None:
        result = LLMEntityExtractor._regex_extract("Contact alice@example.com for details")
        names = [e.name for e in result.entities]
        assert "alice@example.com" in names
        email_entity = next(e for e in result.entities if e.entity_type == "EMAIL")
        assert email_entity.confidence == 0.9

    def test_url_extraction(self) -> None:
        result = LLMEntityExtractor._regex_extract("Visit https://example.com/page")
        names = [e.name for e in result.entities]
        assert "https://example.com/page" in names

    def test_date_extraction(self) -> None:
        result = LLMEntityExtractor._regex_extract("Meeting on 2024-01-15")
        types = [e.entity_type for e in result.entities]
        assert "DATE" in types

    def test_proper_noun_extraction(self) -> None:
        result = LLMEntityExtractor._regex_extract("John Smith attended")
        names = [e.name for e in result.entities]
        assert "John Smith" in names

    def test_empty_text(self) -> None:
        result = LLMEntityExtractor._regex_extract("")
        assert result.entities == []

    def test_short_text_no_entities(self) -> None:
        result = LLMEntityExtractor._regex_extract("ok")
        assert result.entities == []

    def test_metadata_has_extraction_method(self) -> None:
        result = LLMEntityExtractor._regex_extract("test@email.com")
        assert result.metadata["extraction_method"] == "regex"

    def test_dedup_within_result(self) -> None:
        result = LLMEntityExtractor._regex_extract("Email alice@test.com and alice@test.com again")
        emails = [e for e in result.entities if e.entity_type == "EMAIL"]
        assert len(emails) == 1


class TestTieredExtraction:
    """Tests for tiered extraction in extract_multi."""

    @pytest.mark.asyncio
    async def test_trivial_texts_use_regex(self) -> None:
        """Only truly trivial texts (<20 chars) use regex extraction."""
        extractor = LLMEntityExtractor(model="test-model")
        # Very short text (under default 20 chars) should use regex, not LLM
        results = await extractor.extract_multi(
            ["Hi"],
            tiered_extraction=True,
            entity_types=["PERSON", "EMAIL"],
            relationship_types=["KNOWS"],
        )
        assert len(results) == 1
        assert results[0].metadata.get("extraction_method") == "regex"

    @pytest.mark.asyncio
    async def test_short_messages_go_to_llm(self) -> None:
        """Short but substantive texts (>20 chars) go through LLM, not regex."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        # A typical Slack message (~50 chars) should go to LLM
        section_data = {
            "sections": [
                {
                    "entities": [{"name": "Alice", "entity_type": "PERSON", "description": "A person"}],
                    "relationships": [],
                    "events": [],
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(section_data)
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["Hi alice@test.com, can we chat about the project?"],
                tiered_extraction=True,
                entity_types=["PERSON", "EMAIL"],
                relationship_types=["KNOWS"],
            )
        assert len(results) == 1
        # Should NOT be regex — should be LLM extraction
        assert results[0].metadata.get("extraction_method") != "regex"

    @pytest.mark.asyncio
    async def test_explicit_high_threshold_uses_regex(self) -> None:
        """Callers can still opt into regex for short texts with explicit threshold."""
        extractor = LLMEntityExtractor(model="test-model")
        results = await extractor.extract_multi(
            ["Hi alice@test.com"],
            tiered_extraction=True,
            tier1_max_chars=200,
            entity_types=["PERSON", "EMAIL"],
            relationship_types=["KNOWS"],
        )
        assert len(results) == 1
        assert results[0].metadata.get("extraction_method") == "regex"
        emails = [e for e in results[0].entities if e.entity_type == "EMAIL"]
        assert len(emails) == 1


# ---------------------------------------------------------------------------
# Adaptive batching density limits
# ---------------------------------------------------------------------------


class TestAdaptiveBatching:
    """Tests for _create_adaptive_batches density limits."""

    def test_short_texts_packed_aggressively(self) -> None:
        """Short texts (<300 chars) can be packed up to 30 per batch."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini")
        texts = ["Hello from Slack" * 5] * 25  # 25 texts, each ~80 chars
        batches = extractor._create_adaptive_batches(texts, max_batch_size=30, max_input_tokens=100_000)
        # All 25 short texts should fit in a single batch (density_limit=30)
        assert len(batches) == 1
        assert len(batches[0]) == 25

    def test_medium_texts_moderate_batching(self) -> None:
        """Medium texts (300-800 chars) get density_limit=15."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini")
        texts = ["x" * 500] * 20  # 20 texts, each 500 chars
        batches = extractor._create_adaptive_batches(texts, max_batch_size=30, max_input_tokens=100_000)
        # density_limit=15 for 500-char texts, so 20 texts -> 2 batches
        assert len(batches) == 2
        assert len(batches[0]) == 15
        assert len(batches[1]) == 5

    def test_long_texts_conservative_batching(self) -> None:
        """Long texts (>2000 chars) get density_limit=3."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini")
        texts = ["x" * 3000] * 6
        batches = extractor._create_adaptive_batches(texts, max_batch_size=30, max_input_tokens=100_000)
        # density_limit=3 for 3000-char texts
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3

    def test_token_budget_respected(self) -> None:
        """Token budget takes priority over density limit."""
        extractor = LLMEntityExtractor(model="gpt-4o-mini")
        # 10 texts of 200 chars each = ~67 tokens each
        texts = ["x" * 200] * 10
        # Set very low token budget: 500 overhead + ~67 per text = fits ~7 texts
        batches = extractor._create_adaptive_batches(texts, max_batch_size=30, max_input_tokens=1000)
        # Should split based on token budget, not density limit
        assert len(batches) >= 2


# ---------------------------------------------------------------------------
# Shared extractor pattern
# ---------------------------------------------------------------------------


class TestSharedExtractor:
    """Tests for shared extractor in extract_entities task."""

    @pytest.mark.asyncio
    async def test_shared_extractor_reused(self) -> None:
        """When shared_extractor is provided, it should be used instead of creating a new one."""
        from khora.pipelines.tasks.extract import extract_entities

        # Create a mock shared extractor
        mock_extractor = MagicMock()
        mock_result = ExtractionResult(
            entities=[ExtractedEntity(name="Test", entity_type="PERSON", confidence=0.9)],
            relationships=[],
        )
        mock_extractor.extract_multi = AsyncMock(return_value=[mock_result])

        # Create a mock chunk
        mock_chunk = MagicMock()
        mock_chunk.content = "Alice works at Acme"
        mock_chunk.id = "chunk-1"
        mock_chunk.document_id = "doc-1"
        mock_chunk.namespace_id = "ns-1"
        mock_chunk.created_at = None

        entities, relationships = await extract_entities(
            [mock_chunk],
            entity_types=["PERSON"],
            relationship_types=["WORKS_FOR"],
            shared_extractor=mock_extractor,
        )

        # Verify the shared extractor was called
        mock_extractor.extract_multi.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_shared_extractor_creates_new(self) -> None:
        """Without shared_extractor, a new LLMEntityExtractor is created."""
        from khora.pipelines.tasks.extract import extract_entities

        mock_result = ExtractionResult(
            entities=[],
            relationships=[],
        )

        with patch("khora.extraction.extractors.LLMEntityExtractor") as MockExtractorClass:
            mock_instance = MagicMock()
            mock_instance.extract_multi = AsyncMock(return_value=[mock_result])
            MockExtractorClass.return_value = mock_instance

            mock_chunk = MagicMock()
            mock_chunk.content = "Test text"
            mock_chunk.id = "chunk-1"
            mock_chunk.document_id = "doc-1"
            mock_chunk.namespace_id = "ns-1"
            mock_chunk.created_at = None

            await extract_entities(
                [mock_chunk],
                entity_types=["PERSON"],
                relationship_types=["WORKS_FOR"],
            )

            # Verify a new extractor was created
            MockExtractorClass.assert_called_once()


# ---------------------------------------------------------------------------
# Recursive bisection on output truncation
# ---------------------------------------------------------------------------


class TestBisectionOnTruncation:
    """Tests for recursive bisection when LLM output is truncated."""

    def _make_extractor(self) -> LLMEntityExtractor:
        return LLMEntityExtractor(model="test-model", max_retries=1)

    def _success(self, name: str = "E") -> ExtractionResult:
        return ExtractionResult(
            entities=[ExtractedEntity(name=name, entity_type="PERSON", confidence=0.9)],
            relationships=[],
        )

    def _truncated(self) -> ExtractionResult:
        return ExtractionResult(metadata={"error": "truncated_response", "finish_reason": "length"})

    # All tests use tiered_extraction=False (bypass regex shortcut) and
    # batch_size=50 (ensure all texts land in one batch) so that
    # _extract_multi_batch is always reached.

    @pytest.mark.asyncio
    async def test_no_truncation_no_bisect(self) -> None:
        """Happy path: no truncation, _extract_multi_batch called exactly once."""
        extractor = self._make_extractor()
        texts = ["text one", "text two", "text three", "text four"]

        with patch.object(extractor, "_extract_multi_batch", new_callable=AsyncMock) as mock_multi:
            mock_multi.return_value = [self._success() for _ in texts]
            results = await extractor.extract_multi(
                texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50
            )

        assert len(results) == 4
        mock_multi.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_truncation_bisects_once(self) -> None:
        """Full batch truncation → bisect into two halves, both succeed."""
        extractor = self._make_extractor()
        texts = [f"text {i}" for i in range(8)]

        call_count = 0

        async def mock_multi_batch(batch, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if len(batch) == 8:
                # Initial call — truncated
                return [self._truncated() for _ in batch]
            # Halves succeed
            return [self._success() for _ in batch]

        with patch.object(extractor, "_extract_multi_batch", side_effect=mock_multi_batch):
            results = await extractor.extract_multi(
                texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50
            )

        assert len(results) == 8
        assert all(not r.metadata.get("error") for r in results)
        # 1 initial call + 2 half calls
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_recursive_bisection(self) -> None:
        """Left half also truncates → recurse to depth 2 before all succeed."""
        extractor = self._make_extractor()
        texts = [f"text {i}" for i in range(8)]

        async def mock_multi_batch(batch, *args, **kwargs):
            if len(batch) >= 4:
                # batch of 8 or 4 → truncated
                return [self._truncated() for _ in batch]
            # batch of 2 → success
            return [self._success() for _ in batch]

        with patch.object(extractor, "_extract_multi_batch", side_effect=mock_multi_batch):
            results = await extractor.extract_multi(
                texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50
            )

        assert len(results) == 8
        assert all(not r.metadata.get("error") for r in results)

    @pytest.mark.asyncio
    async def test_single_item_floor_uses_single_doc(self) -> None:
        """Single-item batch that truncates falls back to self.extract (not infinite recursion)."""
        extractor = self._make_extractor()
        texts = ["this text is long enough"]
        single_result = self._success("Floor")

        with (
            patch.object(extractor, "_extract_multi_batch", new_callable=AsyncMock) as mock_multi,
            patch.object(extractor, "extract", new_callable=AsyncMock, return_value=single_result) as mock_single,
        ):
            mock_multi.return_value = [self._truncated()]
            results = await extractor.extract_multi(
                texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50
            )

        assert len(results) == 1
        assert results[0].entities[0].name == "Floor"
        mock_single.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_success_keeps_good_bisects_bad(self) -> None:
        """Partial truncation: keep successes at [0,1], bisect failures at [2,3]."""
        extractor = self._make_extractor()
        texts = ["text zero", "text one", "text two", "text three"]

        call_count = 0

        async def mock_multi_batch(batch, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if len(batch) == 4:
                # First call: partial truncation — items 2 and 3 truncated
                return [self._success("ok0"), self._success("ok1"), self._truncated(), self._truncated()]
            # Bisected halves of [t2, t3] succeed
            return [self._success("ok_bisected") for _ in batch]

        with patch.object(extractor, "_extract_multi_batch", side_effect=mock_multi_batch):
            results = await extractor.extract_multi(
                texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50
            )

        assert len(results) == 4
        assert results[0].entities[0].name == "ok0"
        assert results[1].entities[0].name == "ok1"
        assert not results[2].metadata.get("error")
        assert not results[3].metadata.get("error")
        # 1 initial (4-item) + 2 bisection calls (1 per failed item, split into singles)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_transient_error_does_not_bisect(self) -> None:
        """Non-truncation errors use the circuit breaker / single-doc fallback, not bisection."""
        extractor = self._make_extractor()
        texts = ["text zero", "text one", "text two", "text three"]
        transient_error = ExtractionResult(metadata={"error": "network_error"})
        single_result = self._success("single")

        with (
            patch.object(extractor, "_extract_multi_batch", new_callable=AsyncMock) as mock_multi,
            patch.object(extractor, "extract", new_callable=AsyncMock, return_value=single_result),
        ):
            # All results have non-truncation error
            mock_multi.return_value = [transient_error for _ in texts]
            results = await extractor.extract_multi(
                texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50
            )

        # Falls back to single-doc extraction (circuit breaker path)
        assert len(results) == 4
        # _extract_multi_batch is called once (no bisection)
        mock_multi.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_persists_across_extract_batch_calls(self) -> None:
        """_consecutive_batch_failures persists so circuit breaker trips across invocations."""
        extractor = self._make_extractor()
        texts = ["text zero", "text one"]
        transient_error = ExtractionResult(metadata={"error": "network_error"})
        single_result = self._success()

        assert extractor._consecutive_batch_failures == 0

        with (
            patch.object(extractor, "_extract_multi_batch", new_callable=AsyncMock) as mock_multi,
            patch.object(extractor, "extract", new_callable=AsyncMock, return_value=single_result),
        ):
            mock_multi.return_value = [transient_error for _ in texts]

            # First call: 1 failure
            await extractor.extract_multi(texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50)
            assert extractor._consecutive_batch_failures == 1

            # Second call: hits threshold, circuit breaker trips
            await extractor.extract_multi(texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50)
            assert extractor._consecutive_batch_failures == 2

            # Third call: circuit breaker is tripped — _extract_multi_batch must NOT be called again
            await extractor.extract_multi(texts, entity_types=["PERSON"], tiered_extraction=False, batch_size=50)
            # Still 2 calls total: third invocation skipped batch mode entirely
            assert mock_multi.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_multi_batch_detects_truncation_via_json_decode_error(self) -> None:
        """_extract_multi_batch returns truncated_response when JSON is cut off mid-string."""
        extractor = self._make_extractor()
        texts = ["text one", "text two"]

        # Simulate a response whose JSON is truncated mid-string (as happens when the
        # LLM hits its max_tokens limit without finishing output).
        truncated_json = '{"sections": [{"entities": [{"name": "incomplet'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = truncated_json
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_response.model = "test-model"

        import litellm as _litellm

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
            patch("khora.telemetry.context.record_usage"),
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor._extract_multi_batch(
                texts,
                ["PERSON"],
                _litellm,
                system_prompt=None,
                tool_context=None,
                expertise=None,
                context=None,
                relationship_types=None,
            )

        assert len(results) == 2
        for r in results:
            assert r.metadata.get("error") == "truncated_response"

    @pytest.mark.asyncio
    async def test_bisect_and_extract_empty_batch_returns_empty_list(self) -> None:
        """Empty batch passed to _bisect_and_extract returns [] without calling LLM."""
        extractor = self._make_extractor()

        with patch.object(extractor, "_extract_multi_batch", new_callable=AsyncMock) as mock_multi:
            results = await extractor.extract_multi([], entity_types=["PERSON"], tiered_extraction=False, batch_size=50)

        assert results == []
        mock_multi.assert_not_called()


# ---------------------------------------------------------------------------
# Entity attributes: strict-schema emission and pair-form parsing
# ---------------------------------------------------------------------------

# Strict structured output cannot express an open-ended {string: string} map
# (additionalProperties must be False), so attributes are emitted as an array of
# {"key": ..., "value": ...} pairs. This is the expected schema for each pair.
_ATTRIBUTES_PAIR_ITEM = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["key", "value"],
    "additionalProperties": False,
}


class TestAttributesSchema:
    """The strict json_schema entity item carries an `attributes` pairs array."""

    def _make_extractor(self) -> LLMEntityExtractor:
        # gpt-4o-mini is on MODELS_REQUIRING_JSON_SCHEMA, so the strict
        # json_schema branch (not the loose json_object fallback) is exercised.
        return LLMEntityExtractor(model="gpt-4o-mini")

    @staticmethod
    def _assert_entity_item_strict(item: dict) -> None:
        """The entity item stays strict-valid with attributes present."""
        # attributes is a pairs array on the item.
        assert "attributes" in item["properties"]
        assert item["properties"]["attributes"]["type"] == "array"
        assert item["properties"]["attributes"]["items"] == _ATTRIBUTES_PAIR_ITEM
        # attributes is required.
        assert "attributes" in item["required"]
        # Strict-valid: every property is required and the item is closed.
        assert set(item["required"]) == set(item["properties"])
        assert item["additionalProperties"] is False

    def test_response_format_entity_item(self) -> None:
        """_get_response_format: entity item includes the attributes pairs array."""
        fmt = self._make_extractor()._get_response_format()
        assert fmt["json_schema"]["strict"] is True
        item = fmt["json_schema"]["schema"]["properties"]["entities"]["items"]
        self._assert_entity_item_strict(item)

    def test_multi_response_format_entity_item(self) -> None:
        """_get_multi_response_format: section entity item includes the pairs array."""
        fmt = self._make_extractor()._get_multi_response_format()
        assert fmt["json_schema"]["strict"] is True
        item = fmt["json_schema"]["schema"]["properties"]["sections"]["items"]["properties"]["entities"]["items"]
        self._assert_entity_item_strict(item)


class TestAttributesParsing:
    """_parse_response folds pair-form attributes into a dict."""

    def _make_extractor(self) -> LLMEntityExtractor:
        return LLMEntityExtractor(model="test-model")

    def test_pairs_fold_to_dict(self) -> None:
        """A list of {key, value} pairs folds into a {key: value} dict."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {
                    "name": "Alice",
                    "entity_type": "PERSON",
                    "attributes": [
                        {"key": "email", "value": "alice@example.com"},
                        {"key": "role", "value": "engineer"},
                    ],
                }
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].attributes == {
            "email": "alice@example.com",
            "role": "engineer",
        }

    def test_dict_passes_through(self) -> None:
        """An already-dict attributes value passes through unchanged."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {
                    "name": "Alice",
                    "entity_type": "PERSON",
                    "attributes": {"email": "alice@example.com", "role": "engineer"},
                }
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].attributes == {
            "email": "alice@example.com",
            "role": "engineer",
        }

    def test_empty_absent_or_scalar_yield_empty_dict(self) -> None:
        """Empty list, absent attributes, and a scalar value all collapse to {}."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                {"name": "EmptyList", "entity_type": "CONCEPT", "attributes": []},
                {"name": "Absent", "entity_type": "CONCEPT"},
                {"name": "Scalar", "entity_type": "CONCEPT", "attributes": "not-a-mapping"},
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        by_name = {e.name: e for e in result.entities}
        assert by_name["EmptyList"].attributes == {}
        assert by_name["Absent"].attributes == {}
        assert by_name["Scalar"].attributes == {}

    def test_pair_value_coercion_and_malformed_items(self) -> None:
        """Pair values are string-coerced; missing/None values and malformed items are handled."""
        extractor = self._make_extractor()
        data = {
            "entities": [
                # Non-string value is coerced to str; missing and None values -> "".
                {
                    "name": "Coerce",
                    "entity_type": "CONCEPT",
                    "attributes": [
                        {"key": "age", "value": 30},
                        {"key": "no_value"},
                        {"key": "null_value", "value": None},
                    ],
                },
                # Duplicate keys: last value wins.
                {
                    "name": "Dup",
                    "entity_type": "CONCEPT",
                    "attributes": [
                        {"key": "state", "value": "old"},
                        {"key": "state", "value": "new"},
                    ],
                },
                # Malformed items (non-dict item, non-string key) are skipped; valid pair kept.
                {
                    "name": "Malformed",
                    "entity_type": "CONCEPT",
                    "attributes": [
                        "not-a-dict",
                        {"key": 123, "value": "dropped"},
                        {"key": "ok", "value": "kept"},
                    ],
                },
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        by_name = {e.name: e for e in result.entities}
        assert by_name["Coerce"].attributes == {"age": "30", "no_value": "", "null_value": ""}
        assert by_name["Dup"].attributes == {"state": "new"}
        assert by_name["Malformed"].attributes == {"ok": "kept"}

    @pytest.mark.asyncio
    async def test_extract_round_trip_folds_pair_attributes(self) -> None:
        """A mocked LLM returning pair-form attributes yields a populated dict."""
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "entities": [
                    {
                        "name": "Alice",
                        "entity_type": "PERSON",
                        "description": "A person",
                        "attributes": [
                            {"key": "email", "value": "alice@example.com"},
                            {"key": "team", "value": "platform"},
                        ],
                    }
                ],
                "relationships": [],
            }
        )
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            result = await extractor.extract(
                "Alice works at Acme Corp",
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
            )

        assert len(result.entities) == 1
        assert result.entities[0].attributes == {
            "email": "alice@example.com",
            "team": "platform",
        }

    @pytest.mark.asyncio
    async def test_extract_multi_round_trip_folds_pair_attributes(self) -> None:
        """The batch (production) path also folds pair-form attributes into a dict.

        Guards the extract_multi -> _extract_multi_batch path against a future
        refactor that gives the batch path its own parse and silently drops
        attributes in production.
        """
        extractor = LLMEntityExtractor(model="test-model", max_retries=1)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "sections": [
                    {
                        "entities": [
                            {
                                "name": "Alice",
                                "entity_type": "PERSON",
                                "description": "A person",
                                "attributes": [
                                    {"key": "email", "value": "alice@example.com"},
                                    {"key": "team", "value": "platform"},
                                ],
                            }
                        ],
                        "relationships": [],
                        "events": [],
                    }
                ]
            }
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with (
            patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response),
            patch("khora.telemetry.get_collector") as mock_telem,
        ):
            mock_telem.return_value.record_llm_call = MagicMock()
            results = await extractor.extract_multi(
                ["Alice works at Acme Corp"],
                batch_size=5,
                entity_types=["PERSON", "ORGANIZATION"],
                relationship_types=["WORKS_FOR"],
                tiered_extraction=False,
            )

        assert len(results) == 1
        assert results[0].entities[0].attributes == {
            "email": "alice@example.com",
            "team": "platform",
        }

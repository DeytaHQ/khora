"""Unit tests for extraction/extractors/llm.py — LLM entity extraction."""

from __future__ import annotations

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
        """JSON null values for name/type are handled (the staged bugfix)."""
        extractor = self._make_extractor()
        data = {
            "entities": [{"name": None, "entity_type": None, "description": None}],
            "relationships": [{"source_entity": None, "target_entity": None, "relationship_type": None}],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].name == ""
        assert result.entities[0].entity_type == "CONCEPT"
        assert result.relationships[0].source_entity == ""
        assert result.relationships[0].relationship_type == "RELATES_TO"

    def test_attributes_non_dict(self) -> None:
        """Non-dict attributes are replaced with empty dict."""
        extractor = self._make_extractor()
        data = {
            "entities": [{"name": "Test", "entity_type": "CONCEPT", "attributes": ["invalid"]}],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert result.entities[0].attributes == {}


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
    async def test_short_texts_use_regex(self) -> None:
        extractor = LLMEntityExtractor(model="test-model")
        # Very short text should use regex, not LLM
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

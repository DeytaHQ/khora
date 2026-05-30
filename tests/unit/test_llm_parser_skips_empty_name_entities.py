"""Regression tests for #894 — LLM parser must skip empty-name entities.

The previous parser called ``ExtractedEntity(name=e.get("name") or "")`` with no
skip-guard, so the LLM occasionally produced entities with empty / whitespace /
missing names that then got persisted as rows with empty names. This file
checks that the parser skips all three shapes and surfaces a count + a
forward-compatible degradation entry on ExtractionResult.metadata.
"""

from __future__ import annotations

import json

from khora.extraction.extractors.llm import LLMEntityExtractor


def _make_extractor() -> LLMEntityExtractor:
    return LLMEntityExtractor(model="test-model")


class TestLLMParserSkipsEmptyNameEntities:
    def test_empty_string_null_and_missing_name_are_all_skipped(self) -> None:
        """All three empty-name shapes must be skipped — no rows inserted."""
        extractor = _make_extractor()
        data = {
            "entities": [
                # 1) explicit empty string
                {"name": "", "entity_type": "PERSON", "description": "blank"},
                # 2) explicit JSON null
                {"name": None, "entity_type": "PERSON", "description": "null"},
                # 3) name key entirely absent
                {"entity_type": "PERSON", "description": "missing key"},
                # control: a real entity that must survive
                {"name": "Alice", "entity_type": "PERSON", "description": "ok"},
            ],
            "relationships": [],
        }

        result = extractor._parse_response(json.dumps(data))

        # Only the control survives.
        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"

        # Counter is wired through ExtractionResult.metadata.
        assert result.metadata.get("skipped_entities_empty_name") == 3

        # Forward-compatible degradation entry per ADR-001.
        degradations = result.metadata.get("degradations", [])
        assert len(degradations) == 1
        assert degradations[0]["component"] == "llm.entity_parser"
        assert degradations[0]["reason"] == "empty_name"
        assert degradations[0]["skipped_entities"] == 3
        assert degradations[0]["skipped_relationships"] == 0

    def test_whitespace_only_name_is_skipped(self) -> None:
        """A whitespace-only name (" ", "\\t\\n") must be treated as empty."""
        extractor = _make_extractor()
        data = {
            "entities": [
                {"name": "   ", "entity_type": "PERSON"},
                {"name": "\t\n", "entity_type": "PERSON"},
                {"name": "Bob", "entity_type": "PERSON"},
            ],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert [e.name for e in result.entities] == ["Bob"]
        assert result.metadata.get("skipped_entities_empty_name") == 2

    def test_relationships_with_empty_endpoint_are_skipped(self) -> None:
        """Relationships missing source_entity or target_entity must be skipped."""
        extractor = _make_extractor()
        data = {
            "entities": [
                {"name": "Alice", "entity_type": "PERSON"},
                {"name": "Acme", "entity_type": "ORG"},
            ],
            "relationships": [
                # missing source
                {"target_entity": "Acme", "relationship_type": "WORKS_FOR"},
                # missing target
                {"source_entity": "Alice", "relationship_type": "WORKS_FOR"},
                # empty source
                {"source_entity": "", "target_entity": "Acme", "relationship_type": "WORKS_FOR"},
                # whitespace target
                {"source_entity": "Alice", "target_entity": "   ", "relationship_type": "WORKS_FOR"},
                # valid control
                {
                    "source_entity": "Alice",
                    "target_entity": "Acme",
                    "relationship_type": "WORKS_FOR",
                },
            ],
        }
        result = extractor._parse_response(json.dumps(data))

        assert len(result.relationships) == 1
        assert result.relationships[0].source_entity == "Alice"
        assert result.relationships[0].target_entity == "Acme"

        assert result.metadata.get("skipped_relationships_empty_endpoint") == 4
        degradations = result.metadata.get("degradations", [])
        assert len(degradations) == 1
        assert degradations[0]["skipped_relationships"] == 4

    def test_clean_payload_has_no_degradation_entry(self) -> None:
        """If nothing was skipped, no degradation entry should be added."""
        extractor = _make_extractor()
        data = {
            "entities": [{"name": "Alice", "entity_type": "PERSON"}],
            "relationships": [],
        }
        result = extractor._parse_response(json.dumps(data))
        assert len(result.entities) == 1
        assert "degradations" not in result.metadata
        assert "skipped_entities_empty_name" not in result.metadata

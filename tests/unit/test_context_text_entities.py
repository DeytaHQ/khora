"""Tests for the ``format_entity_section`` and ``format_relationship_section`` helpers.

Verifies that the entity-section formatter renders entities (name, type,
description) and de-duplicates by ID, and that the relationship-section
formatter renders source/type/target/description tuples. Callers compose
these alongside ``chunk.content`` to build an LLM context string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models.recall import (
    RecallChunk,
    RecallEntity,
    RecallRelationship,
    RecallResult,
)
from khora.core.recall_context import format_entity_section, format_relationship_section


def _mk_entity(
    *,
    name: str,
    entity_type: str = "CONCEPT",
    description: str = "",
    entity_id=None,
    score: float = 0.9,
) -> RecallEntity:
    return RecallEntity(
        id=entity_id or uuid4(),
        name=name,
        entity_type=entity_type,
        description=description,
        score=score,
        attributes={},
        mention_count=1,
        source_document_ids=[],
        source_chunk_ids=[],
    )


def _mk_rel(
    *,
    source_id,
    target_id,
    relationship_type: str,
    description: str = "",
    score: float = 0.9,
) -> RecallRelationship:
    return RecallRelationship(
        id=uuid4(),
        source_entity_id=source_id,
        target_entity_id=target_id,
        relationship_type=relationship_type,
        description=description,
        score=score,
        valid_from=None,
        valid_until=None,
        source_document_ids=[],
    )


@pytest.mark.unit
class TestGetContextTextIncludesEntities:
    """Verify that entity data appears when callers compose a context string."""

    def test_chunks_and_entities(self) -> None:
        """When both chunks and entities are present, context_text has both."""
        entity = _mk_entity(
            name="TestEntity",
            entity_type="CONCEPT",
            description="A test entity",
        )

        entity_section = format_entity_section([entity])
        context_text = "First paragraph." + entity_section

        assert "First paragraph." in context_text
        assert "TestEntity" in context_text
        assert "CONCEPT" in context_text
        assert "A test entity" in context_text
        assert "--- Entities ---" in context_text

    def test_entity_with_empty_description(self) -> None:
        """Entity with no description still renders correctly."""
        entity = _mk_entity(
            name="NoDescEntity",
            entity_type="PERSON",
            description="",
        )

        section = format_entity_section([entity])

        assert "NoDescEntity" in section
        assert "PERSON" in section

    def test_chunks_only_no_entities(self) -> None:
        """When there are no entities, only chunk content appears."""
        section = format_entity_section([])
        assert section == ""

    def test_entities_only_no_chunks(self) -> None:
        """When there are entities but no chunks, entities still appear."""
        entity = _mk_entity(
            name="OnlyEntity",
            entity_type="EVENT",
            description="Solo entity",
        )
        section = format_entity_section([entity])
        assert "OnlyEntity" in section

    def test_duplicate_entities_deduplicated(self) -> None:
        """Same entity ID appearing twice should be deduplicated."""
        entity_id = uuid4()
        entity = _mk_entity(
            entity_id=entity_id,
            name="DupEntity",
            entity_type="CONCEPT",
            description="Duplicated",
        )

        # Pass same entity twice
        section = format_entity_section([entity, entity])
        # Count occurrences of the entity name
        assert section.count("DupEntity") == 1


@pytest.mark.unit
class TestContextTextEntityRegressions:
    """Regression tests for entity display edge cases."""

    def test_dedup_different_objects_same_id(self) -> None:
        """Two RecallEntity objects with the same ID (different Python objects)
        should be deduplicated."""
        eid = uuid4()
        e1 = _mk_entity(entity_id=eid, name="Entity1", entity_type="CONCEPT", description="First")
        e2 = _mk_entity(entity_id=eid, name="Entity1", entity_type="CONCEPT", description="Second")

        section = format_entity_section([e1, e2])
        assert section.count("Entity1") == 1

    def test_many_entities_no_truncation(self) -> None:
        """All entities should appear even if there are many."""
        entities = [
            _mk_entity(
                name=f"Entity{i}",
                entity_type="CONCEPT",
                description=f"Description {i}",
                score=0.9 - i * 0.01,
            )
            for i in range(20)
        ]

        section = format_entity_section(entities)

        for i in range(20):
            assert f"Entity{i}" in section

    def test_entity_special_characters(self) -> None:
        """Entities with special characters render correctly."""
        entity = _mk_entity(
            name='Entity "With Quotes" & <Brackets>',
            entity_type="CONCEPT",
            description="Has special chars: <>&\"'",
        )

        section = format_entity_section([entity])

        assert 'Entity "With Quotes" & <Brackets>' in section

    def test_format_entity_section_helper_directly(self) -> None:
        """Test the format_entity_section helper function directly."""
        entity = _mk_entity(
            name="DirectTest",
            entity_type="PERSON",
            description="Direct test entity",
        )

        result = format_entity_section([entity])

        assert "--- Entities ---" in result
        assert "DirectTest" in result
        assert "PERSON" in result
        assert "Direct test entity" in result

    def test_chunk_format_unchanged_with_entities(self) -> None:
        """Adding entities doesn't change how chunks are formatted."""
        entity = _mk_entity(name="TestEntity", entity_type="CONCEPT", description="Test")

        entity_section = format_entity_section([entity])
        context_text = "First paragraph." + entity_section

        # Chunks should be before entities
        assert "First paragraph." in context_text.split("--- Entities ---")[0]
        # Chunk content is unchanged
        assert "First paragraph." in context_text


@pytest.mark.unit
class TestRelationshipFormatting:
    """format_relationship_section and RecallResult relationship support."""

    def test_format_relationship_section_basic(self) -> None:
        """Arrow format with description: '- Alice --FOUNDED--> Acme Corp: Founded the company'."""
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded the company",
        )

        section = format_relationship_section(
            [rel],
            {alice_id: "Alice", acme_id: "Acme Corp"},
        )

        assert section.startswith("\n\n--- Relationships ---\n\n")
        assert "- Alice --FOUNDED--> Acme Corp: Founded the company" in section

    def test_format_relationship_section_no_description(self) -> None:
        """No trailing colon when description is empty."""
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="WORKS_AT",
            description="",
        )

        section = format_relationship_section(
            [rel],
            {alice_id: "Alice", acme_id: "Acme Corp"},
        )

        assert "- Alice --WORKS_AT--> Acme Corp" in section
        # No trailing colon
        assert "- Alice --WORKS_AT--> Acme Corp:" not in section

    def test_format_relationship_section_empty(self) -> None:
        """Returns empty string for empty list."""
        assert format_relationship_section([], {}) == ""

    def test_format_relationship_section_dedup(self) -> None:
        """Duplicate IDs collapsed (same relationship twice should appear once)."""
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded it",
        )

        section = format_relationship_section(
            [rel, rel],
            {alice_id: "Alice", acme_id: "Acme Corp"},
        )

        rel_lines = [line.strip() for line in section.strip().splitlines() if line.strip().startswith("- ")]
        assert len(rel_lines) == 1

    def test_format_relationship_section_uuid_fallback(self) -> None:
        """UUID fallback when entity names are empty."""
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="KNOWS",
            description="",
        )

        # Empty lookup -> both fall back to str(UUID)
        section_no_names = format_relationship_section([rel], {})
        assert str(alice_id) in section_no_names
        assert str(acme_id) in section_no_names

    def test_format_relationship_section_partial_names(self) -> None:
        """Partial names: set source but not target -> target falls back to UUID."""
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="KNOWS",
            description="",
        )

        section = format_relationship_section([rel], {alice_id: "Alice"})
        assert "Alice" in section
        assert str(acme_id) in section

    def test_context_text_with_entities_and_relationships(self) -> None:
        """Full context_text has chunk content + Entities + Relationships sections."""
        alice_id = uuid4()
        acme_id = uuid4()

        entity_alice = _mk_entity(
            entity_id=alice_id,
            name="Alice",
            entity_type="PERSON",
            description="Founder",
        )
        entity_acme = _mk_entity(
            entity_id=acme_id,
            name="Acme Corp",
            entity_type="ORGANIZATION",
            description="A company",
        )
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded the company",
        )

        text = "Alice founded Acme Corp."
        text += format_entity_section([entity_alice, entity_acme])
        text += format_relationship_section(
            [rel],
            {alice_id: "Alice", acme_id: "Acme Corp"},
        )

        assert "Alice founded Acme Corp." in text
        assert "--- Entities ---" in text
        assert "--- Relationships ---" in text

        # Verify ordering: chunk content, then entities, then relationships
        ent_pos = text.index("--- Entities ---")
        rel_pos = text.index("--- Relationships ---")
        assert ent_pos < rel_pos

        # Verify relationship content
        rel_section = text.split("--- Relationships ---")[1]
        assert "- Alice --FOUNDED--> Acme Corp: Founded the company" in rel_section

    def test_context_text_relationships_only(self) -> None:
        """Relationships without entities still works."""
        alice_id = uuid4()
        acme_id = uuid4()
        rel = _mk_rel(
            source_id=alice_id,
            target_id=acme_id,
            relationship_type="WORKS_AT",
            description="Employee",
        )

        # No entities, just relationships
        text = format_entity_section([])
        text += format_relationship_section(
            [rel],
            {alice_id: "Alice", acme_id: "Acme Corp"},
        )

        assert "--- Entities ---" not in text
        assert "--- Relationships ---" in text
        assert "- Alice --WORKS_AT--> Acme Corp: Employee" in text

    def test_backward_compat_no_relationships(self) -> None:
        """RecallResult without relationships kwarg works (default empty list)."""
        ns_id = uuid4()
        chunk_id = uuid4()
        doc_id = uuid4()

        result = RecallResult(
            query="test query",
            namespace_id=ns_id,
            documents=[],
            chunks=[
                RecallChunk(
                    id=chunk_id,
                    document_id=doc_id,
                    content="Some content.",
                    score=0.9,
                    created_at=datetime.now(UTC),
                )
            ],
            entities=[],
            relationships=[],
        )

        assert result.relationships == []
        assert result.chunks[0].content == "Some content."

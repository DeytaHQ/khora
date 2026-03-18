"""Tests for entity and relationship inclusion in context_text.

DYT-524: Verifies that RecallResult.context_text includes an entity section
when entities are present in the recall result.

DYT-563: Verifies relationship formatting in context_text and RecallResult
relationship support.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.core.models.document import Chunk, ChunkMetadata
from khora.core.models.entity import Entity, Relationship
from khora.memory_lake import RecallResult
from khora.query.engine import format_entity_section, format_relationship_section


@pytest.mark.unit
class TestGetContextTextIncludesEntities:
    """DYT-524: verify that entity data appears in RecallResult.context_text."""

    def test_chunks_and_entities(self) -> None:
        """When both chunks and entities are present, context_text has both."""
        ns_id = uuid4()
        chunk = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="First paragraph.",
            metadata=ChunkMetadata(),
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="TestEntity",
            entity_type="CONCEPT",
            description="A test entity",
        )

        entity_section = format_entity_section([(entity, 0.85)])
        context_text = chunk.content + entity_section

        assert "First paragraph." in context_text
        assert "TestEntity" in context_text
        assert "CONCEPT" in context_text
        assert "A test entity" in context_text
        assert "--- Entities ---" in context_text

    def test_entity_with_empty_description(self) -> None:
        """Entity with no description still renders correctly."""
        ns_id = uuid4()
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="NoDescEntity",
            entity_type="PERSON",
            description="",
        )

        section = format_entity_section([(entity, 0.5)])

        assert "NoDescEntity" in section
        assert "PERSON" in section

    def test_chunks_only_no_entities(self) -> None:
        """When there are no entities, only chunk content appears."""
        section = format_entity_section([])
        assert section == ""

    def test_entities_only_no_chunks(self) -> None:
        """When there are entities but no chunks, entities still appear."""
        ns_id = uuid4()
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="OnlyEntity",
            entity_type="EVENT",
            description="Solo entity",
        )
        section = format_entity_section([(entity, 0.9)])
        assert "OnlyEntity" in section

    def test_duplicate_entities_deduplicated(self) -> None:
        """Same entity ID appearing twice should be deduplicated."""
        ns_id = uuid4()
        entity_id = uuid4()
        entity = Entity(
            id=entity_id,
            namespace_id=ns_id,
            name="DupEntity",
            entity_type="CONCEPT",
            description="Duplicated",
        )

        # Pass same entity twice with different scores
        section = format_entity_section([(entity, 0.9), (entity, 0.7)])
        # Count occurrences of the entity name
        assert section.count("DupEntity") == 1


@pytest.mark.unit
class TestContextTextEntityRegressions:
    """Regression tests for entity display edge cases."""

    def test_dedup_different_objects_same_id(self) -> None:
        """Two Entity objects with the same ID (different Python objects)
        should be deduplicated."""
        ns_id = uuid4()
        eid = uuid4()
        e1 = Entity(id=eid, namespace_id=ns_id, name="Entity1", entity_type="CONCEPT", description="First")
        e2 = Entity(id=eid, namespace_id=ns_id, name="Entity1", entity_type="CONCEPT", description="Second")

        section = format_entity_section([(e1, 0.9), (e2, 0.7)])
        assert section.count("Entity1") == 1

    def test_many_entities_no_truncation(self) -> None:
        """All entities should appear even if there are many."""
        ns_id = uuid4()
        entities = [
            (
                Entity(
                    id=uuid4(),
                    namespace_id=ns_id,
                    name=f"Entity{i}",
                    entity_type="CONCEPT",
                    description=f"Description {i}",
                ),
                0.9 - i * 0.01,
            )
            for i in range(20)
        ]

        section = format_entity_section(entities)

        for i in range(20):
            assert f"Entity{i}" in section

    def test_entity_special_characters(self) -> None:
        """Entities with special characters render correctly."""
        ns_id = uuid4()
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name='Entity "With Quotes" & <Brackets>',
            entity_type="CONCEPT",
            description="Has special chars: <>&\"'",
        )

        section = format_entity_section([(entity, 0.8)])

        assert 'Entity "With Quotes" & <Brackets>' in section

    def test_format_entity_section_helper_directly(self) -> None:
        """Test the format_entity_section helper function directly."""
        ns_id = uuid4()
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="DirectTest",
            entity_type="PERSON",
            description="Direct test entity",
        )

        result = format_entity_section([(entity, 0.85)])

        assert "--- Entities ---" in result
        assert "DirectTest" in result
        assert "PERSON" in result
        assert "Direct test entity" in result

    def test_chunk_format_unchanged_with_entities(self) -> None:
        """Adding entities doesn't change how chunks are formatted."""
        ns_id = uuid4()

        chunk1 = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="First paragraph.",
            metadata=ChunkMetadata(),
        )
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="TestEntity",
            entity_type="CONCEPT",
            description="Test",
        )

        entity_section = format_entity_section([(entity, 0.85)])
        context_text = chunk1.content + entity_section

        # Chunks should be before entities
        assert "First paragraph." in context_text.split("--- Entities ---")[0]
        # Chunk content is unchanged
        assert "First paragraph." in context_text


@pytest.mark.unit
class TestRelationshipFormatting:
    """DYT-563: format_relationship_section and RecallResult relationship support."""

    def test_format_relationship_section_basic(self) -> None:
        """Arrow format with description: '- Alice --FOUNDED--> Acme Corp: Founded the company'."""
        ns_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="FOUNDED",
            description="Founded the company",
            source_entity_name="Alice",
            target_entity_name="Acme Corp",
        )

        section = format_relationship_section([(rel, 0.9)])

        assert section.startswith("\n\n--- Relationships ---\n\n")
        assert "- Alice --FOUNDED--> Acme Corp: Founded the company" in section

    def test_format_relationship_section_no_description(self) -> None:
        """No trailing colon when description is empty."""
        ns_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_AT",
            description="",
            source_entity_name="Alice",
            target_entity_name="Acme Corp",
        )

        section = format_relationship_section([(rel, 0.8)])

        assert "- Alice --WORKS_AT--> Acme Corp" in section
        # No trailing colon
        assert "- Alice --WORKS_AT--> Acme Corp:" not in section

    def test_format_relationship_section_empty(self) -> None:
        """Returns empty string for empty list."""
        assert format_relationship_section([]) == ""

    def test_format_relationship_section_dedup(self) -> None:
        """Duplicate IDs collapsed (same relationship twice should appear once)."""
        ns_id = uuid4()
        rel_id = uuid4()
        rel = Relationship(
            id=rel_id,
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="FOUNDED",
            description="Founded it",
            source_entity_name="Alice",
            target_entity_name="Acme Corp",
        )

        section = format_relationship_section([(rel, 0.9), (rel, 0.5)])

        rel_lines = [line.strip() for line in section.strip().splitlines() if line.strip().startswith("- ")]
        assert len(rel_lines) == 1

    def test_format_relationship_section_uuid_fallback(self) -> None:
        """UUID fallback when entity names are empty."""
        ns_id = uuid4()
        alice_id = uuid4()
        acme_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="KNOWS",
            description="",
        )

        # No names set -> both fall back to str(UUID)
        section_no_names = format_relationship_section([(rel, 0.7)])
        assert str(alice_id) in section_no_names
        assert str(acme_id) in section_no_names

    def test_format_relationship_section_partial_names(self) -> None:
        """Partial names: set source but not target -> target falls back to UUID."""
        ns_id = uuid4()
        acme_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=acme_id,
            relationship_type="KNOWS",
            description="",
            source_entity_name="Alice",
        )

        section = format_relationship_section([(rel, 0.7)])
        assert "Alice" in section
        assert str(acme_id) in section

    def test_context_text_with_entities_and_relationships(self) -> None:
        """Full context_text has chunk content + Entities + Relationships sections."""
        ns_id = uuid4()
        alice_id = uuid4()
        acme_id = uuid4()

        chunk = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Alice founded Acme Corp.",
            metadata=ChunkMetadata(),
        )
        entity_alice = Entity(
            id=alice_id,
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            description="Founder",
        )
        entity_acme = Entity(
            id=acme_id,
            namespace_id=ns_id,
            name="Acme Corp",
            entity_type="ORGANIZATION",
            description="A company",
        )
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded the company",
            source_entity_name="Alice",
            target_entity_name="Acme Corp",
        )

        text = chunk.content
        text += format_entity_section([(entity_alice, 0.85), (entity_acme, 0.7)])
        text += format_relationship_section([(rel, 0.9)])

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
        ns_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_AT",
            description="Employee",
            source_entity_name="Alice",
            target_entity_name="Acme Corp",
        )

        # No entities, just relationships
        text = format_entity_section([])
        text += format_relationship_section([(rel, 0.8)])

        assert "--- Entities ---" not in text
        assert "--- Relationships ---" in text
        assert "- Alice --WORKS_AT--> Acme Corp: Employee" in text

    def test_backward_compat_no_relationships(self) -> None:
        """RecallResult without relationships kwarg works (default empty list)."""
        ns_id = uuid4()
        chunk = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Some content.",
            metadata=ChunkMetadata(),
        )

        # Construct without passing relationships — should default to []
        result = RecallResult(
            query="test query",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[],
            context_text="Some content.",
        )

        assert result.relationships == []
        assert result.context_text == "Some content."

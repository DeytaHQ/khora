"""Tests proving DYT-524: context_text should include entity information.

These tests verify that QueryResult.get_context_text() includes entity data
when entities are present. Currently, context_text is built only from chunk
content, ignoring entities entirely.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from khora.core.models.document import Chunk, ChunkMetadata
from khora.core.models.entity import Entity, Relationship
from khora.memory_lake import RecallResult
from khora.query.engine import QueryResult, format_entity_section, format_relationship_section


@pytest.mark.unit
class TestGetContextTextIncludesEntities:
    """QueryResult.get_context_text should render entities when present."""

    def test_chunks_and_entities(self) -> None:
        """context_text contains both chunk content and entity info."""
        ns_id = uuid4()
        chunk = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Alice founded Acme Corp in 2020.",
            metadata=ChunkMetadata(),
        )
        entity_alice = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            description="Founder of Acme Corp",
        )
        entity_acme = Entity(
            namespace_id=ns_id,
            name="Acme Corp",
            entity_type="ORGANIZATION",
            description="A technology company founded in 2020",
        )

        result = QueryResult(
            chunks=[(chunk, 0.9)],
            entities=[(entity_alice, 0.85), (entity_acme, 0.7)],
        )
        text = result.get_context_text()

        # Chunk content must be present
        assert "Alice founded Acme Corp in 2020." in text

        # Entity separator must be present
        assert "--- Entities ---" in text

        # Entity names, types, and descriptions must be present
        assert "Alice" in text.split("--- Entities ---")[1]
        assert "PERSON" in text.split("--- Entities ---")[1]
        assert "Founder of Acme Corp" in text.split("--- Entities ---")[1]
        assert "Acme Corp" in text.split("--- Entities ---")[1]
        assert "ORGANIZATION" in text.split("--- Entities ---")[1]

    def test_entity_with_empty_description(self) -> None:
        """Entity with empty description renders without ': ' suffix."""
        ns_id = uuid4()
        chunk = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Some content.",
            metadata=ChunkMetadata(),
        )
        entity = Entity(
            namespace_id=ns_id,
            name="UnknownEntity",
            entity_type="CONCEPT",
            description="",
        )

        result = QueryResult(
            chunks=[(chunk, 0.9)],
            entities=[(entity, 0.5)],
        )
        text = result.get_context_text()

        assert "--- Entities ---" in text
        entity_section = text.split("--- Entities ---")[1]
        # Should have the name and type
        assert "UnknownEntity" in entity_section
        assert "CONCEPT" in entity_section
        # Should NOT have a trailing ": " with no description
        assert "UnknownEntity (CONCEPT):" not in entity_section

    def test_chunks_only_no_entities(self) -> None:
        """When there are no entities, context_text has no entity section (backward compat)."""
        chunk1 = MagicMock()
        chunk1.content = "first chunk"
        chunk2 = MagicMock()
        chunk2.content = "second chunk"

        result = QueryResult(
            chunks=[(chunk1, 0.9), (chunk2, 0.5)],
            entities=[],
        )
        text = result.get_context_text(max_chunks=2)

        assert "first chunk" in text
        assert "second chunk" in text
        # No entity section when entities list is empty
        assert "--- Entities ---" not in text

    def test_entities_only_no_chunks(self) -> None:
        """When there are only entities and no chunks, context_text still renders entities."""
        ns_id = uuid4()
        entity = Entity(
            namespace_id=ns_id,
            name="SoloEntity",
            entity_type="EVENT",
            description="An important event",
        )

        result = QueryResult(
            chunks=[],
            entities=[(entity, 0.8)],
        )
        text = result.get_context_text()

        assert "--- Entities ---" in text
        assert "SoloEntity" in text
        assert "EVENT" in text
        assert "An important event" in text

    def test_duplicate_entities_deduplicated(self) -> None:
        """Same entity appearing with different scores should appear only once."""
        ns_id = uuid4()
        entity = Entity(
            id=uuid4(),
            namespace_id=ns_id,
            name="DuplicateEntity",
            entity_type="PERSON",
            description="Appears twice",
        )

        result = QueryResult(
            chunks=[],
            entities=[(entity, 0.9), (entity, 0.5)],
        )
        text = result.get_context_text()

        assert "--- Entities ---" in text
        entity_section = text.split("--- Entities ---")[1]
        # The entity name should appear exactly once in the entity lines
        entity_lines = [line.strip() for line in entity_section.strip().splitlines() if line.strip().startswith("- ")]
        names = [line for line in entity_lines if "DuplicateEntity" in line]
        assert len(names) == 1, f"Expected 1 occurrence, got {len(names)}: {names}"


@pytest.mark.unit
class TestContextTextEntityRegressions:
    """Regression tests for DYT-524 edge cases."""

    def test_dedup_different_objects_same_id(self) -> None:
        """Two distinct Entity objects sharing the same id should deduplicate."""
        ns_id = uuid4()
        shared_id = uuid4()
        entity_a = Entity(
            id=shared_id,
            namespace_id=ns_id,
            name="SharedEntity",
            entity_type="PERSON",
            description="First instance",
        )
        entity_b = Entity(
            id=shared_id,
            namespace_id=ns_id,
            name="SharedEntity",
            entity_type="PERSON",
            description="Second instance",
        )
        assert entity_a is not entity_b

        result = QueryResult(
            chunks=[],
            entities=[(entity_a, 0.9), (entity_b, 0.4)],
        )
        text = result.get_context_text()
        entity_section = text.split("--- Entities ---")[1]
        entity_lines = [line.strip() for line in entity_section.strip().splitlines() if line.strip().startswith("- ")]
        assert len(entity_lines) == 1
        # Should keep the first occurrence
        assert "First instance" in entity_lines[0]

    def test_many_entities_no_truncation(self) -> None:
        """15+ entities should all render without silent truncation."""
        ns_id = uuid4()
        entities = [
            (
                Entity(
                    namespace_id=ns_id,
                    name=f"Entity_{i}",
                    entity_type="CONCEPT",
                    description=f"Description for entity {i}",
                ),
                0.9 - i * 0.01,
            )
            for i in range(20)
        ]

        result = QueryResult(chunks=[], entities=entities)
        text = result.get_context_text()

        entity_section = text.split("--- Entities ---")[1]
        entity_lines = [line.strip() for line in entity_section.strip().splitlines() if line.strip().startswith("- ")]
        assert len(entity_lines) == 20
        for i in range(20):
            assert f"Entity_{i}" in entity_section

    def test_entity_special_characters(self) -> None:
        """Entities with quotes, newlines, and unicode render without errors."""
        ns_id = uuid4()
        entities = [
            (
                Entity(
                    namespace_id=ns_id,
                    name='O\'Brien "The Great"',
                    entity_type="PERSON",
                    description="Has 'quotes' and \"double quotes\"",
                ),
                0.9,
            ),
            (
                Entity(
                    namespace_id=ns_id,
                    name="Muller",
                    entity_type="PERSON",
                    description="Beschreibung auf Deutsch",
                ),
                0.8,
            ),
            (
                Entity(
                    namespace_id=ns_id,
                    name="Tokyo Tower",
                    entity_type="LOCATION",
                    description="Famous landmark",
                ),
                0.7,
            ),
        ]

        result = QueryResult(chunks=[], entities=entities)
        text = result.get_context_text()

        assert "--- Entities ---" in text
        assert 'O\'Brien "The Great"' in text
        assert "Muller" in text
        assert "Tokyo Tower" in text
        assert "Famous landmark" in text

    def test_format_entity_section_helper_directly(self) -> None:
        """format_entity_section works correctly in isolation (used by VectorCypher)."""
        ns_id = uuid4()
        entities = [
            (
                Entity(
                    namespace_id=ns_id,
                    name="AlphaEntity",
                    entity_type="ORGANIZATION",
                    description="First org",
                ),
                0.9,
            ),
            (
                Entity(
                    namespace_id=ns_id,
                    name="BetaEntity",
                    entity_type="EVENT",
                    description="A big event",
                ),
                0.7,
            ),
        ]

        section = format_entity_section(entities)

        assert section.startswith("\n\n--- Entities ---\n\n")
        assert "- AlphaEntity (ORGANIZATION): First org" in section
        assert "- BetaEntity (EVENT): A big event" in section

        # Empty list returns empty string
        assert format_entity_section([]) == ""

    def test_chunk_format_unchanged_with_entities(self) -> None:
        """Chunk grouping by title is preserved when entities are also present."""
        ns_id = uuid4()
        doc_id = uuid4()
        meta1 = MagicMock()
        meta1.title = "My Document"
        meta1.custom = {}
        chunk1 = Chunk(
            namespace_id=ns_id,
            document_id=doc_id,
            content="First paragraph.",
            metadata=meta1,
        )
        chunk2 = Chunk(
            namespace_id=ns_id,
            document_id=doc_id,
            content="Second paragraph.",
            metadata=meta1,
        )
        meta2 = MagicMock()
        meta2.title = "Other Document"
        meta2.custom = {}
        chunk3 = Chunk(
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Other doc content.",
            metadata=meta2,
        )
        entity = Entity(
            namespace_id=ns_id,
            name="TestEntity",
            entity_type="CONCEPT",
            description="Test desc",
        )

        result = QueryResult(
            chunks=[(chunk1, 0.9), (chunk2, 0.8), (chunk3, 0.7)],
            entities=[(entity, 0.6)],
        )
        text = result.get_context_text()

        # Chunks grouped by title
        assert "--- From: My Document ---" in text
        assert "--- From: Other Document ---" in text
        assert "First paragraph." in text
        assert "Second paragraph." in text
        assert "Other doc content." in text

        # Both chunks from same doc appear under one header
        my_doc_section = text.split("--- From: My Document ---")[1].split("---")[0]
        assert "First paragraph." in my_doc_section
        assert "Second paragraph." in my_doc_section

        # Entity section comes after chunk sections
        assert "--- Entities ---" in text
        parts = text.split("--- Entities ---")
        # Chunks should be before entities
        assert "First paragraph." in parts[0]
        assert "TestEntity" in parts[1]


@pytest.mark.unit
class TestRelationshipFormatting:
    """DYT-563: format_relationship_section and RecallResult relationship support."""

    def test_format_relationship_section_basic(self) -> None:
        """Arrow format with description: '- Alice --FOUNDED--> Acme Corp: Founded the company'."""
        ns_id = uuid4()
        alice_id = uuid4()
        acme_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded the company",
        )
        entity_names = {alice_id: "Alice", acme_id: "Acme Corp"}

        section = format_relationship_section([(rel, 0.9)], entity_names=entity_names)

        assert section.startswith("\n\n--- Relationships ---\n\n")
        assert "- Alice --FOUNDED--> Acme Corp: Founded the company" in section

    def test_format_relationship_section_no_description(self) -> None:
        """No trailing colon when description is empty."""
        ns_id = uuid4()
        alice_id = uuid4()
        acme_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="WORKS_AT",
            description="",
        )
        entity_names = {alice_id: "Alice", acme_id: "Acme Corp"}

        section = format_relationship_section([(rel, 0.8)], entity_names=entity_names)

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
        alice_id = uuid4()
        acme_id = uuid4()
        rel = Relationship(
            id=rel_id,
            namespace_id=ns_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="FOUNDED",
            description="Founded it",
        )
        entity_names = {alice_id: "Alice", acme_id: "Acme Corp"}

        section = format_relationship_section([(rel, 0.9), (rel, 0.5)], entity_names=entity_names)

        rel_lines = [line.strip() for line in section.strip().splitlines() if line.strip().startswith("- ")]
        assert len(rel_lines) == 1

    def test_format_relationship_section_entity_name_lookup(self) -> None:
        """UUID fallback when entity_names dict is None or missing entries."""
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

        # entity_names=None -> both fall back to str(UUID)
        section_no_names = format_relationship_section([(rel, 0.7)], entity_names=None)
        assert str(alice_id) in section_no_names
        assert str(acme_id) in section_no_names

        # entity_names with only one entry -> missing one falls back to str(UUID)
        section_partial = format_relationship_section([(rel, 0.7)], entity_names={alice_id: "Alice"})
        assert "Alice" in section_partial
        assert str(acme_id) in section_partial

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
        )

        # Build context_text manually (matching how VectorCypher would build it)
        from khora.query.engine import format_entity_section, format_relationship_section

        entity_names = {alice_id: "Alice", acme_id: "Acme Corp"}
        text = chunk.content
        text += format_entity_section([(entity_alice, 0.85), (entity_acme, 0.7)])
        text += format_relationship_section([(rel, 0.9)], entity_names=entity_names)

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
        alice_id = uuid4()
        acme_id = uuid4()
        rel = Relationship(
            namespace_id=ns_id,
            source_entity_id=alice_id,
            target_entity_id=acme_id,
            relationship_type="WORKS_AT",
            description="Employee",
        )
        entity_names = {alice_id: "Alice", acme_id: "Acme Corp"}

        # No entities, just relationships
        text = format_entity_section([])
        text += format_relationship_section([(rel, 0.8)], entity_names=entity_names)

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

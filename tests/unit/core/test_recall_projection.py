"""Unit tests for ``khora.core.recall_projection`` (#1480 shared seam 3).

The chronicle and vectorcypher engines both delegate their entity /
relationship / doc-stub projection to these helpers, so the byte-parity
contract - especially the entity ``source_document_ids`` fallback that differs
between the two engines and the fixed doc-stub append order - lives here.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity, Relationship
from khora.core.models.document import DocumentSource
from khora.core.recall_projection import (
    project_document_stubs,
    project_entities,
    project_relationships,
)


@pytest.mark.unit
class TestProjectEntities:
    def test_basic_projection_preserves_fields_and_order(self) -> None:
        e1 = Entity(name="Alice", entity_type="PERSON", description="d1", mention_count=3)
        e2 = Entity(name="Bob", entity_type="PERSON", description="")
        out = project_entities([(e1, 0.9), (e2, 0.5)])

        assert [r.name for r in out] == ["Alice", "Bob"]
        assert out[0].score == 0.9
        assert out[0].mention_count == 3
        assert out[1].description == ""  # None/empty description coerced to ""

    def test_fallback_off_keeps_empty_source_document_ids(self) -> None:
        """Chronicle path: no fallback, so an entity with only source_documents
        keeps an EMPTY source_document_ids (byte-parity with its prior inline)."""
        doc_id = uuid4()
        e = Entity(name="Carol", source_documents={doc_id: DocumentSource(id=uuid4())})
        out = project_entities([(e, 0.7)], source_document_ids_fallback=False)
        assert out[0].source_document_ids == []

    def test_fallback_on_uses_source_documents_keys(self) -> None:
        """VectorCypher path: fallback fills source_document_ids from the map."""
        doc_id = uuid4()
        e = Entity(name="Dave", source_documents={doc_id: DocumentSource(id=uuid4())})
        out = project_entities([(e, 0.7)], source_document_ids_fallback=True)
        assert out[0].source_document_ids == [doc_id]

    def test_fallback_on_prefers_explicit_ids_over_map(self) -> None:
        """When the flat id list is populated, the fallback does NOT override it."""
        explicit = uuid4()
        mapped = uuid4()
        e = Entity(
            name="Eve",
            source_document_ids=[explicit],
            source_documents={mapped: DocumentSource(id=uuid4())},
        )
        out = project_entities([(e, 0.7)], source_document_ids_fallback=True)
        assert out[0].source_document_ids == [explicit]


@pytest.mark.unit
class TestProjectRelationships:
    def test_basic_projection(self) -> None:
        src, tgt = uuid4(), uuid4()
        rel = Relationship(
            source_entity_id=src,
            target_entity_id=tgt,
            relationship_type="OWNS",
            description="d",
        )
        out = project_relationships([(rel, 0.8)])
        assert len(out) == 1
        assert out[0].relationship_type == "OWNS"
        assert out[0].source_entity_id == src
        assert out[0].target_entity_id == tgt
        assert out[0].score == 0.8

    def test_empty(self) -> None:
        assert project_relationships([]) == []


@pytest.mark.unit
class TestProjectDocumentStubs:
    def test_chunk_docs_first_then_entity_then_relationship(self) -> None:
        """Append order is chunks, then entity-only docs, then relationship-only
        docs - the documented byte-parity contract."""
        chunk_doc = uuid4()
        entity_doc = uuid4()
        rel_doc = uuid4()

        chunk = Chunk(document_id=chunk_doc, content="c")
        entity = Entity(name="X", source_document_ids=[entity_doc])
        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            source_document_ids=[rel_doc],
        )
        recall_entities = project_entities([(entity, 1.0)])
        recall_rels = project_relationships([(rel, 1.0)])

        docs = project_document_stubs([(chunk, 1.0)], recall_entities, recall_rels)
        assert [d.id for d in docs] == [chunk_doc, entity_doc, rel_doc]

    def test_dedup_across_surfaces(self) -> None:
        """A doc referenced by chunk + entity + rel appears once (chunk wins)."""
        shared = uuid4()
        chunk = Chunk(document_id=shared, content="c")
        entity = Entity(name="X", source_document_ids=[shared])
        rel = Relationship(
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            source_document_ids=[shared],
        )
        docs = project_document_stubs(
            [(chunk, 1.0)],
            project_entities([(entity, 1.0)]),
            project_relationships([(rel, 1.0)]),
        )
        assert [d.id for d in docs] == [shared]

    def test_none_relationships_is_noop(self) -> None:
        """Chronicle passes None for relationships; the third loop must no-op."""
        chunk_doc = uuid4()
        entity_doc = uuid4()
        chunk = Chunk(document_id=chunk_doc, content="c")
        entity = Entity(name="X", source_document_ids=[entity_doc])
        docs = project_document_stubs([(chunk, 1.0)], project_entities([(entity, 1.0)]), None)
        assert [d.id for d in docs] == [chunk_doc, entity_doc]

    def test_chunk_doc_projection_carries_source_document_fields(self) -> None:
        """A chunk carrying a source_document surfaces its title/source/type."""
        doc_id = uuid4()
        src = DocumentSource(id=uuid4(), source_type="email", title="T", source="S")
        chunk = Chunk(document_id=doc_id, content="c", source_document=src)
        docs = project_document_stubs([(chunk, 1.0)], [], None)
        assert docs[0].source_type == "email"
        assert docs[0].title == "T"
        assert docs[0].source == "S"

    def test_chunk_without_source_document_defaults_to_library(self) -> None:
        chunk = Chunk(document_id=uuid4(), content="c")
        docs = project_document_stubs([(chunk, 1.0)], [], None)
        assert docs[0].source_type == "library"
        assert docs[0].title is None

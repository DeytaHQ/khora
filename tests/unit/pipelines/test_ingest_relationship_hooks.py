"""Chronicle ingest flow dispatches relationship hooks from canonical results (#1320).

``process_document``'s ``_store_relationships`` historically fired one
``relationship.created`` per *submitted* relationship using ``rel.id`` - the
submitted id, not the canonical persisted edge id - and never emitted
``relationship.updated`` on a dedup-merge. ``create_relationships_batch`` now
returns ``list[(relationship, is_new)]`` with the in-place id synced to the
stored edge (mirroring ``upsert_entities_batch``), so the dispatch splits
created/updated and carries the canonical id.

These drive ``process_document`` directly with stubbed storage + mocked
extraction (the harness style of ``test_ingest_temporal_chunks``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Entity, Relationship
from khora.core.models.event import EventType


def _make_storage_mock() -> MagicMock:
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.update_document = AsyncMock()
    storage.create_chunks_batch = AsyncMock()
    # Echo the input entities as all-new so the name+type -> id remap resolves.
    storage.upsert_entities_batch = AsyncMock(side_effect=lambda ns, ents, **kw: [(e, True) for e in ents])
    storage.update_entity_embeddings_batch = AsyncMock()
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    storage.get_entity_by_name = AsyncMock(return_value=None)
    storage.dispatch_hook = AsyncMock()
    return storage


def _make_chunk(ns_id, doc_id, content="Acme acquired Beta in 2025.") -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.now(UTC),
    )


def _make_document_mock(doc_id, ns_id, content):
    doc = MagicMock()
    doc.id = doc_id
    doc.namespace_id = ns_id
    doc.content = content
    doc.metadata = {}
    doc.title = ""
    doc.created_at = datetime.now(UTC)
    doc.mark_processing = MagicMock()
    doc.mark_completed = MagicMock()
    doc.mark_failed = MagicMock()
    doc.status = "pending"
    return doc


def _make_entity(ns_id, name, entity_type, chunk_id) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        description=f"{name} desc",
        source_chunk_ids=[chunk_id],
        confidence=0.9,
    )


def _events_of(storage, event_type):
    return [c.args[0] for c in storage.dispatch_hook.await_args_list if c.args[0].event_type == event_type]


def _fake_embedder():
    """An embedder whose embed_batch echoes a fixed-width vector per input."""
    emb = MagicMock()
    emb.model_name = "fake-embed"
    emb.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    return emb


@pytest.mark.unit
class TestIngestRelationshipHooks:
    @pytest.mark.asyncio
    async def test_relationship_created_uses_canonical_stored_id(self) -> None:
        """A genuine create fires relationship.created with the canonical id the
        backend returns, not the submitted rel.id."""
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        chunk = _make_chunk(ns_id, doc_id)
        acme = _make_entity(ns_id, "Acme", "ORGANIZATION", chunk.id)
        beta = _make_entity(ns_id, "Beta", "ORGANIZATION", chunk.id)
        rel = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=acme.id,
            target_entity_id=beta.id,
            relationship_type="ACQUIRED",
            confidence=0.9,
        )
        canonical_id = uuid4()

        def _create(relationships, **_kw):
            relationships[0].id = canonical_id
            return [(relationships[0], True)]

        storage = _make_storage_mock()
        storage.create_relationships_batch = AsyncMock(side_effect=lambda rels, **kw: _create(rels, **kw))

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([acme, beta], [rel]))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                _make_document_mock(doc_id, ns_id, chunk.content),
                storage,
                entity_types=["ORGANIZATION"],
                relationship_types=["ACQUIRED"],
            )

        created = _events_of(storage, EventType.RELATIONSHIP_CREATED)
        updated = _events_of(storage, EventType.RELATIONSHIP_UPDATED)
        assert len(created) == 1
        assert updated == []
        assert created[0].resource_type == "relationship"
        assert created[0].resource_id == canonical_id
        assert created[0].data["relationship_type"] == "ACQUIRED"

    @pytest.mark.asyncio
    async def test_dedup_merge_emits_relationship_updated_not_created(self) -> None:
        """A dedup-merge (is_new=False) fires relationship.updated with the
        canonical stored id, never a spurious relationship.created."""
        from khora.pipelines.flows.ingest import process_document

        ns_id, doc_id = uuid4(), uuid4()
        chunk = _make_chunk(ns_id, doc_id)
        acme = _make_entity(ns_id, "Acme", "ORGANIZATION", chunk.id)
        beta = _make_entity(ns_id, "Beta", "ORGANIZATION", chunk.id)
        rel = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=acme.id,
            target_entity_id=beta.id,
            relationship_type="ACQUIRED",
            confidence=0.9,
        )
        canonical_id = uuid4()

        def _merge(relationships, **_kw):
            relationships[0].id = canonical_id
            return [(relationships[0], False)]

        storage = _make_storage_mock()
        storage.create_relationships_batch = AsyncMock(side_effect=lambda rels, **kw: _merge(rels, **kw))

        with (
            patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=[chunk])),
            patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([acme, beta], [rel]))),
            patch("khora.extraction.embedders.LiteLLMEmbedder", return_value=_fake_embedder()),
        ):
            await process_document(
                _make_document_mock(doc_id, ns_id, chunk.content),
                storage,
                entity_types=["ORGANIZATION"],
                relationship_types=["ACQUIRED"],
            )

        created = _events_of(storage, EventType.RELATIONSHIP_CREATED)
        updated = _events_of(storage, EventType.RELATIONSHIP_UPDATED)
        assert created == []
        assert len(updated) == 1
        assert updated[0].resource_id == canonical_id

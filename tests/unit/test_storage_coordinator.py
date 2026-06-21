"""Unit tests for storage/coordinator.py — StorageCoordinator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document, Entity, MemoryEvent, Relationship
from khora.core.models.document import DocumentStatus
from khora.storage.coordinator import ReplaceResult, StorageCoordinator, StorageHealth


class TestStorageHealth:
    """Tests for StorageHealth dataclass."""

    def test_healthy_when_relational_and_vector(self) -> None:
        """is_healthy requires relational and vector."""
        h = StorageHealth(relational=True, vector=True, graph=False, event_store=False)
        assert h.is_healthy is True

    def test_unhealthy_without_relational(self) -> None:
        """Missing relational makes it unhealthy."""
        h = StorageHealth(relational=False, vector=True)
        assert h.is_healthy is False

    def test_unhealthy_without_vector(self) -> None:
        """Missing vector makes it unhealthy."""
        h = StorageHealth(relational=True, vector=False)
        assert h.is_healthy is False

    def test_summary(self) -> None:
        """summary returns dict of all backends."""
        h = StorageHealth(relational=True, vector=True, graph=True, event_store=False)
        summary = h.summary
        assert summary == {
            "relational": True,
            "vector": True,
            "graph": True,
            "event_store": False,
        }


class TestStorageCoordinatorLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """Connect calls connect on all backends."""
        rel = MagicMock()
        rel.connect = AsyncMock()
        vec = MagicMock()
        vec.connect = AsyncMock()
        graph = MagicMock()
        graph.connect = AsyncMock()

        coord = StorageCoordinator(relational=rel, vector=vec, graph=graph)
        await coord.connect()

        rel.connect.assert_awaited_once()
        vec.connect.assert_awaited_once()
        graph.connect.assert_awaited_once()
        assert coord._connected is True

    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        """Second connect call is a no-op."""
        rel = MagicMock()
        rel.connect = AsyncMock()
        coord = StorageCoordinator(relational=rel)
        await coord.connect()
        await coord.connect()
        rel.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Disconnect calls disconnect on all backends in reverse order."""
        rel = MagicMock()
        rel.connect = AsyncMock()
        rel.disconnect = AsyncMock()
        vec = MagicMock()
        vec.connect = AsyncMock()
        vec.disconnect = AsyncMock()

        coord = StorageCoordinator(relational=rel, vector=vec)
        await coord.connect()
        await coord.disconnect()

        rel.disconnect.assert_awaited_once()
        vec.disconnect.assert_awaited_once()
        assert coord._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        """Disconnect is no-op when not connected."""
        rel = MagicMock()
        rel.disconnect = AsyncMock()
        coord = StorageCoordinator(relational=rel)
        await coord.disconnect()
        rel.disconnect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Health check queries all backends."""
        rel = MagicMock()
        rel.is_healthy = AsyncMock(return_value=True)
        vec = MagicMock()
        vec.is_healthy = AsyncMock(return_value=True)
        graph = MagicMock()
        graph.is_healthy = AsyncMock(return_value=False)

        coord = StorageCoordinator(relational=rel, vector=vec, graph=graph)
        health = await coord.health_check()

        assert health.relational is True
        assert health.vector is True
        assert health.graph is False


class TestDocumentOps:
    """Tests for document operations (delegated to relational)."""

    @pytest.mark.asyncio
    async def test_create_document(self) -> None:
        """create_document delegates to relational."""
        doc = MagicMock(spec=Document)
        doc.namespace_id = uuid4()
        rel = MagicMock()
        rel.create_document = AsyncMock(return_value=doc)

        coord = StorageCoordinator(relational=rel)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            result = await coord.create_document(doc)

        assert result is doc
        rel.create_document.assert_awaited_once_with(doc)

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        """get_document delegates to relational."""
        doc_id = uuid4()
        ns_id = uuid4()
        rel = MagicMock()
        rel.get_document = AsyncMock(return_value=None)
        coord = StorageCoordinator(relational=rel)
        await coord.get_document(doc_id, namespace_id=ns_id)
        rel.get_document.assert_awaited_once_with(doc_id, namespace_id=ns_id)

    @pytest.mark.asyncio
    async def test_update_document(self) -> None:
        """update_document delegates to relational."""
        doc = MagicMock(spec=Document)
        rel = MagicMock()
        rel.update_document = AsyncMock(return_value=doc)
        coord = StorageCoordinator(relational=rel)
        await coord.update_document(doc)
        rel.update_document.assert_awaited_once_with(doc)

    @pytest.mark.asyncio
    async def test_delete_document(self) -> None:
        """delete_document deletes chunks first, then document."""
        doc_id = uuid4()
        ns_id = uuid4()
        rel = MagicMock()
        rel.delete_document = AsyncMock(return_value=True)
        vec = MagicMock()
        vec.delete_chunks_by_document = AsyncMock()

        coord = StorageCoordinator(relational=rel, vector=vec)
        result = await coord.delete_document(doc_id, namespace_id=ns_id)

        vec.delete_chunks_by_document.assert_awaited_once_with(doc_id, namespace_id=ns_id)
        rel.delete_document.assert_awaited_once_with(doc_id, namespace_id=ns_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_count_documents(self) -> None:
        """count_documents delegates to relational."""
        ns_id = uuid4()
        rel = MagicMock()
        rel.count_documents = AsyncMock(return_value=42)
        coord = StorageCoordinator(relational=rel)
        result = await coord.count_documents(ns_id)
        assert result == 42
        rel.count_documents.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_count_documents_missing_relational(self) -> None:
        """count_documents without relational raises RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.count_documents(uuid4())

    @pytest.mark.asyncio
    async def test_get_last_activity_at(self) -> None:
        """get_last_activity_at delegates to relational."""
        from datetime import UTC, datetime

        ns_id = uuid4()
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        rel = MagicMock()
        rel.get_last_activity_at = AsyncMock(return_value=ts)
        coord = StorageCoordinator(relational=rel)
        result = await coord.get_last_activity_at(ns_id)
        assert result == ts
        rel.get_last_activity_at.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_get_last_activity_at_none(self) -> None:
        """get_last_activity_at returns None when no documents exist."""
        ns_id = uuid4()
        rel = MagicMock()
        rel.get_last_activity_at = AsyncMock(return_value=None)
        coord = StorageCoordinator(relational=rel)
        result = await coord.get_last_activity_at(ns_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_last_activity_at_missing_relational(self) -> None:
        """get_last_activity_at without relational raises RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.get_last_activity_at(uuid4())

    @pytest.mark.asyncio
    async def test_missing_relational(self) -> None:
        """Operations without relational raise RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.create_document(MagicMock())

    @pytest.mark.asyncio
    async def test_get_document_by_external_id_delegates(self) -> None:
        """get_document_by_external_id delegates to relational."""
        ns_id = uuid4()
        doc = MagicMock(spec=Document)
        rel = MagicMock()
        rel.get_document_by_external_id = AsyncMock(return_value=doc)
        coord = StorageCoordinator(relational=rel)

        result = await coord.get_document_by_external_id("ext-1", namespace_id=ns_id)

        assert result is doc
        rel.get_document_by_external_id.assert_awaited_once_with("ext-1", namespace_id=ns_id)

    @pytest.mark.asyncio
    async def test_get_document_by_external_id_missing_relational(self) -> None:
        """get_document_by_external_id without relational raises RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.get_document_by_external_id("ext-1", namespace_id=uuid4())

    @pytest.mark.asyncio
    async def test_get_document_by_external_id_none_short_circuits(self) -> None:
        """Passing external_id=None still delegates to the backend (which guards)."""
        # The coordinator wrapper forwards unconditionally; backend-level guards
        # return None without a DB roundtrip. Verify delegation and return value.
        rel = MagicMock()
        rel.get_document_by_external_id = AsyncMock(return_value=None)
        coord = StorageCoordinator(relational=rel)

        result = await coord.get_document_by_external_id(None, namespace_id=uuid4())

        assert result is None
        rel.get_document_by_external_id.assert_awaited_once()


class TestChunkOps:
    """Tests for chunk operations (delegated to vector)."""

    @pytest.mark.asyncio
    async def test_create_chunks_batch(self) -> None:
        """create_chunks_batch delegates to vector."""
        chunks = [MagicMock(spec=Chunk, namespace_id=uuid4())]
        vec = MagicMock()
        vec.create_chunks_batch = AsyncMock(return_value=chunks)

        coord = StorageCoordinator(vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            result = await coord.create_chunks_batch(chunks)

        assert result == chunks

    @pytest.mark.asyncio
    async def test_search_similar_chunks(self) -> None:
        """search_similar_chunks delegates to vector."""
        ns_id = uuid4()
        embedding = [0.1, 0.2, 0.3]
        vec = MagicMock()
        vec.search_similar = AsyncMock(return_value=[])

        coord = StorageCoordinator(vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            result = await coord.search_similar_chunks(ns_id, embedding)

        assert result == []

    @pytest.mark.asyncio
    async def test_missing_vector(self) -> None:
        """Operations without vector raise RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Vector backend not configured"):
            await coord.create_chunk(MagicMock())


class TestEntityOps:
    """Tests for entity operations (cross-backend)."""

    @pytest.mark.asyncio
    async def test_create_entity_graph_and_vector(self) -> None:
        """create_entity stores in both graph and vector."""
        entity = MagicMock(spec=Entity, namespace_id=uuid4())
        graph = MagicMock()
        graph.create_entity = AsyncMock(return_value=entity)
        vec = MagicMock()
        vec.create_entity = AsyncMock()

        coord = StorageCoordinator(graph=graph, vector=vec)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            await coord.create_entity(entity)

        graph.create_entity.assert_awaited_once()
        vec.create_entity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_entity_parallel(self) -> None:
        """update_entity runs graph and vector in parallel."""
        ns_id = uuid4()
        entity = MagicMock(spec=Entity)
        entity.namespace_id = ns_id
        graph = MagicMock()
        graph.update_entity = AsyncMock(return_value=entity)
        vec = MagicMock()
        vec.update_entity = AsyncMock()

        coord = StorageCoordinator(graph=graph, vector=vec)
        await coord.update_entity(entity, namespace_id=ns_id)

        graph.update_entity.assert_awaited_once()
        vec.update_entity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_entity_removes_graph_and_vector(self) -> None:
        """delete_entity drops both the graph node and the pgvector mirror (#928)."""
        ns_id = uuid4()
        entity_id = uuid4()
        graph = MagicMock()
        graph.delete_entity = AsyncMock(return_value=True)
        vec = MagicMock()
        vec.delete_entities_batch = AsyncMock(return_value=1)

        coord = StorageCoordinator(graph=graph, vector=vec)
        result = await coord.delete_entity(entity_id, namespace_id=ns_id)

        assert result is True
        graph.delete_entity.assert_awaited_once_with(entity_id, namespace_id=ns_id)
        vec.delete_entities_batch.assert_awaited_once_with([entity_id], namespace_id=ns_id)

    @pytest.mark.asyncio
    async def test_delete_entity_skips_vector_on_unified_backend(self) -> None:
        """delete_entity does not double-delete on a unified backend (#928)."""
        ns_id = uuid4()
        entity_id = uuid4()
        graph = MagicMock()
        graph.delete_entity = AsyncMock(return_value=True)
        vec = MagicMock()
        vec.delete_entities_batch = AsyncMock(return_value=1)

        coord = StorageCoordinator(graph=graph, vector=vec)
        coord._is_unified_backend = True
        result = await coord.delete_entity(entity_id, namespace_id=ns_id)

        assert result is True
        graph.delete_entity.assert_awaited_once_with(entity_id, namespace_id=ns_id)
        vec.delete_entities_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_entity_by_name(self) -> None:
        """get_entity_by_name delegates to graph."""
        ns_id = uuid4()
        graph = MagicMock()
        graph.get_entity_by_name = AsyncMock(return_value=None)
        coord = StorageCoordinator(graph=graph)
        await coord.get_entity_by_name(ns_id, "test", "PERSON")
        graph.get_entity_by_name.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_entities_no_graph(self) -> None:
        """list_entities without graph returns empty list."""
        coord = StorageCoordinator()
        result = await coord.list_entities(uuid4())
        assert result == []

    @pytest.mark.asyncio
    async def test_upsert_entities_batch_empty(self) -> None:
        """Empty entities list returns empty."""
        coord = StorageCoordinator()
        result = await coord.upsert_entities_batch(uuid4(), [])
        assert result == []


class TestCountEntities:
    """Tests for count_entities ownership-based routing (#878).

    Entities are owned by the graph backend; the vector backend holds a
    denormalized mirror that may not implement ``count_entities`` (e.g.
    sqlite_lance / SurrealDB). count_entities must therefore prefer the
    graph backend and only fall back to vector when no graph exists.
    """

    @pytest.mark.asyncio
    async def test_count_entities_prefers_graph(self) -> None:
        """count_entities uses graph (owner) even when vector also has it (pg shape)."""
        ns_id = uuid4()
        vec = MagicMock()
        vec.count_entities = AsyncMock(return_value=42)
        graph = MagicMock()
        graph.count_entities = AsyncMock(return_value=99)
        coord = StorageCoordinator(vector=vec, graph=graph)
        result = await coord.count_entities(ns_id)
        assert result == 99
        graph.count_entities.assert_awaited_once_with(ns_id)
        vec.count_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_count_entities_sqlite_lance_shape(self) -> None:
        """count_entities counts from graph when the vector adapter lacks the method.

        This is the sqlite_lance / SurrealDB topology: the vector adapter has
        no ``count_entities``. Before #878 this raised AttributeError; now it
        routes to the graph backend that owns entities.
        """
        ns_id = uuid4()
        # spec= so the mock does NOT auto-create a count_entities attribute.
        vec = MagicMock(spec=["count_chunks"])
        graph = MagicMock()
        graph.count_entities = AsyncMock(return_value=7)
        coord = StorageCoordinator(vector=vec, graph=graph)
        result = await coord.count_entities(ns_id)
        assert result == 7
        graph.count_entities.assert_awaited_once_with(ns_id)
        assert not hasattr(vec, "count_entities")

    @pytest.mark.asyncio
    async def test_count_entities_falls_back_to_vector_when_no_graph(self) -> None:
        """count_entities uses vector when no graph backend configured (PG-only chronicle)."""
        ns_id = uuid4()
        vec = MagicMock()
        vec.count_entities = AsyncMock(return_value=42)
        coord = StorageCoordinator(vector=vec)
        result = await coord.count_entities(ns_id)
        assert result == 42
        vec.count_entities.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_count_entities_no_backends_returns_zero(self) -> None:
        """count_entities returns 0 when neither backend is configured."""
        coord = StorageCoordinator()
        assert await coord.count_entities(uuid4()) == 0

    @pytest.mark.asyncio
    async def test_count_entities_vector_raises(self) -> None:
        """count_entities propagates when vector raises (engine catches)."""
        ns_id = uuid4()
        vec = MagicMock()
        vec.count_entities = AsyncMock(side_effect=RuntimeError("db down"))
        coord = StorageCoordinator(vector=vec)
        with pytest.raises(RuntimeError, match="db down"):
            await coord.count_entities(ns_id)

    @pytest.mark.asyncio
    async def test_count_entities_no_backends(self) -> None:
        """count_entities returns 0 when neither backend is available."""
        coord = StorageCoordinator()
        result = await coord.count_entities(uuid4())
        assert result == 0


class TestRelationshipOps:
    """Tests for relationship operations."""

    @pytest.mark.asyncio
    async def test_create_relationship(self) -> None:
        """create_relationship delegates to graph."""
        rel = MagicMock(spec=Relationship, namespace_id=uuid4())
        graph = MagicMock()
        graph.create_relationship = AsyncMock(return_value=rel)

        coord = StorageCoordinator(graph=graph)
        with patch("khora.telemetry.get_collector") as mock_telem:
            mock_telem.return_value.record_storage_op = MagicMock()
            await coord.create_relationship(rel)

        graph.create_relationship.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_relationships_batch_empty(self) -> None:
        """Empty relationships list returns an empty result list (#1320)."""
        coord = StorageCoordinator()
        results = await coord.create_relationships_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_entity_relationships(self) -> None:
        """get_entity_relationships delegates to graph."""
        entity_id = uuid4()
        ns_id = uuid4()
        graph = MagicMock()
        graph.get_entity_relationships = AsyncMock(return_value=[])
        coord = StorageCoordinator(graph=graph)
        result = await coord.get_entity_relationships(entity_id, namespace_id=ns_id)
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_graph(self) -> None:
        """create_relationship without graph raises RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Graph backend not configured"):
            await coord.create_relationship(MagicMock())


class TestCountRelationships:
    """Tests for count_relationships delegation."""

    @pytest.mark.asyncio
    async def test_count_relationships_delegates_to_graph(self) -> None:
        """count_relationships delegates to graph backend."""
        ns_id = uuid4()
        graph = MagicMock()
        graph.count_relationships = AsyncMock(return_value=42)
        coord = StorageCoordinator(graph=graph)
        result = await coord.count_relationships(ns_id)
        assert result == 42
        graph.count_relationships.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_count_relationships_no_graph_returns_zero(self) -> None:
        """count_relationships returns 0 when no graph backend configured."""
        coord = StorageCoordinator()
        result = await coord.count_relationships(uuid4())
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_relationships_propagates_not_implemented(self) -> None:
        """count_relationships propagates NotImplementedError from graph."""
        ns_id = uuid4()
        graph = MagicMock()
        graph.count_relationships = AsyncMock(side_effect=NotImplementedError)
        coord = StorageCoordinator(graph=graph)
        with pytest.raises(NotImplementedError):
            await coord.count_relationships(ns_id)


class TestGraphOps:
    """Tests for graph traversal operations."""

    @pytest.mark.asyncio
    async def test_get_neighborhood_no_graph(self) -> None:
        """get_neighborhood without graph returns empty structure."""
        coord = StorageCoordinator()
        result = await coord.get_neighborhood(uuid4(), namespace_id=uuid4())
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_find_paths_no_graph(self) -> None:
        """find_paths without graph returns empty list."""
        coord = StorageCoordinator()
        result = await coord.find_paths(uuid4(), uuid4(), namespace_id=uuid4())
        assert result == []


class TestEventOps:
    """Tests for event operations."""

    @pytest.mark.asyncio
    async def test_append_event(self) -> None:
        """append_event delegates to event store."""
        event = MagicMock(spec=MemoryEvent)
        es = MagicMock()
        es.append_event = AsyncMock(return_value=event)
        coord = StorageCoordinator(event_store=es)
        await coord.append_event(event)
        es.append_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_events(self) -> None:
        """get_events delegates to event store."""
        ns_id = uuid4()
        es = MagicMock()
        es.get_events = AsyncMock(return_value=[])
        coord = StorageCoordinator(event_store=es)
        await coord.get_events(ns_id)
        es.get_events.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_event_store(self) -> None:
        """Operations without event store raise RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Event store not configured"):
            await coord.append_event(MagicMock())


class TestBatchOps:
    """Tests for batch operations."""

    @pytest.mark.asyncio
    async def test_get_entities_batch_empty(self) -> None:
        """Empty entity_ids returns empty dict."""
        coord = StorageCoordinator()
        result = await coord.get_entities_batch([], namespace_id=uuid4())
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_documents_batch_empty(self) -> None:
        """Empty document_ids returns empty dict."""
        coord = StorageCoordinator()
        result = await coord.get_documents_batch([], namespace_id=uuid4())
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_neighborhoods_batch_empty(self) -> None:
        """Empty entity_ids returns empty dict."""
        coord = StorageCoordinator()
        result = await coord.get_neighborhoods_batch([], namespace_id=uuid4())
        assert result == {}

    @pytest.mark.asyncio
    async def test_update_entity_embeddings_batch_fallback(self) -> None:
        """Fallback to individual updates when batch not supported."""
        vec = MagicMock(spec=[])  # No upsert_entities_batch method
        vec.update_entity_embedding = AsyncMock()
        coord = StorageCoordinator(vector=vec)

        ns_id = uuid4()
        entity_id = uuid4()
        updates = [(entity_id, [0.1, 0.2], "model")]
        count = await coord.update_entity_embeddings_batch(updates, namespace_id=ns_id)
        assert count == 1
        vec.update_entity_embedding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_entity_embeddings_batch_no_vector(self) -> None:
        """No vector backend returns 0."""
        coord = StorageCoordinator()
        count = await coord.update_entity_embeddings_batch([], namespace_id=uuid4())
        assert count == 0


class TestDocumentSourcesBatch:
    """Tests for get_document_sources_batch."""

    @pytest.mark.asyncio
    async def test_get_document_sources_batch(self) -> None:
        """get_document_sources_batch delegates to relational backend."""
        from khora.core.models.document import DocumentSource

        doc_id = uuid4()
        ns_id = uuid4()
        src = DocumentSource(id=doc_id, title="Test Doc")

        rel = MagicMock()
        rel.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        coord = StorageCoordinator(relational=rel)
        result = await coord.get_document_sources_batch([doc_id], namespace_id=ns_id)

        assert result == {doc_id: src}
        rel.get_document_sources_batch.assert_awaited_once_with([doc_id], namespace_id=ns_id)

    @pytest.mark.asyncio
    async def test_get_document_sources_batch_empty(self) -> None:
        """Empty list returns empty dict without hitting backend."""
        rel = MagicMock()
        rel.get_document_sources_batch = AsyncMock()

        coord = StorageCoordinator(relational=rel)
        result = await coord.get_document_sources_batch([], namespace_id=uuid4())

        assert result == {}
        rel.get_document_sources_batch.assert_not_awaited()


class _FakeTxnSession:
    """Stand-in for an AsyncSession owned by TransactionContext in tests."""

    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False
        self.closed = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True

    async def close(self) -> None:
        self.closed = True


def _make_coordinator_with_fake_txn(
    *,
    relational: MagicMock,
    vector: MagicMock,
    graph: MagicMock,
) -> tuple[StorageCoordinator, _FakeTxnSession]:
    session = _FakeTxnSession()
    # The coordinator resolves a session factory from the first SQL backend
    # that exposes ``_session_factory``.  Hand it ours so ``transaction()``
    # yields a TransactionContext wrapping our fake session.
    relational._session_factory = lambda: session  # type: ignore[attr-defined]
    coord = StorageCoordinator(relational=relational, vector=vector, graph=graph)
    return coord, session


class TestReplaceDocumentExtraction:
    """Unit tests for StorageCoordinator.replace_document_extraction."""

    @pytest.mark.asyncio
    async def test_happy_path_mixed_retire_survive_net_new(self) -> None:
        """Mixed old state: one orphan entity retires, one survivor remaps, one net-new is upserted.

        Relationship counterpart: one sole-sourced orphan retires, one survivor
        remaps, one net-new is created.
        """
        namespace_id = uuid4()
        old_doc_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="new body")

        survivor_entity_id = uuid4()
        orphan_entity_id = uuid4()
        new_entity_survivor = Entity(
            namespace_id=namespace_id,
            name="alice",
            entity_type="PERSON",
            source_document_ids=[new_doc.id],
        )
        new_entity_net_new = Entity(
            namespace_id=namespace_id,
            name="carol",
            entity_type="PERSON",
            source_document_ids=[new_doc.id],
        )

        survivor_rel_id = uuid4()
        orphan_rel_id = uuid4()
        new_rel_survivor = Relationship(
            namespace_id=namespace_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="KNOWS",
            source_document_ids=[new_doc.id],
        )
        # Align survivor rel identity with the prefetched old relationship
        survivor_rel_src = new_rel_survivor.source_entity_id
        survivor_rel_tgt = new_rel_survivor.target_entity_id

        new_rel_net_new = Relationship(
            namespace_id=namespace_id,
            source_entity_id=uuid4(),
            target_entity_id=uuid4(),
            relationship_type="WORKS_WITH",
            source_document_ids=[new_doc.id],
        )

        new_chunks = [Chunk(namespace_id=namespace_id, document_id=new_doc.id, content=f"chunk-{i}") for i in range(3)]

        rel_backend = MagicMock()
        rel_backend.update_document = AsyncMock(return_value=new_doc)

        vec_backend = MagicMock()
        vec_backend.delete_chunks_by_document = AsyncMock(return_value=7)
        vec_backend.create_chunks_batch = AsyncMock(return_value=new_chunks)
        # Strip optional vector-side entity writer so the coordinator's
        # parallel gather path uses the graph-only branch below.
        del vec_backend.upsert_entities_batch

        graph_backend = MagicMock()
        # Prefetch: one survivor, one orphan, each for entities + relationships.
        graph_backend.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [
                    {
                        "id": str(survivor_entity_id),
                        "name": "alice",
                        "entity_type": "PERSON",
                        "namespace_id": str(namespace_id),
                        "source_document_count": 2,
                    },
                    {
                        "id": str(orphan_entity_id),
                        "name": "bob",
                        "entity_type": "PERSON",
                        "namespace_id": str(namespace_id),
                        "source_document_count": 1,
                    },
                ],
                [
                    {
                        "id": str(survivor_rel_id),
                        "source_entity_id": str(survivor_rel_src),
                        "target_entity_id": str(survivor_rel_tgt),
                        "relationship_type": "KNOWS",
                        "source_document_count": 2,
                    },
                    {
                        "id": str(orphan_rel_id),
                        "source_entity_id": str(uuid4()),
                        "target_entity_id": str(uuid4()),
                        "relationship_type": "CITES",
                        "source_document_count": 1,
                    },
                ],
            )
        )
        graph_backend.retire_orphaned_entities_batch = AsyncMock(return_value=1)
        graph_backend.retire_orphaned_relationships_batch = AsyncMock(return_value=1)
        graph_backend.remap_source_document_ids_batch = AsyncMock(return_value=None)
        # upsert_entities_batch returns (entity, is_new) tuples
        graph_backend.upsert_entities_batch = AsyncMock(return_value=[(new_entity_net_new, True)])
        # #1320: returns (relationship, is_new) per persisted edge; the
        # coordinator counts via len(). Echo the net-new rel it is handed.
        graph_backend.create_relationships_batch = AsyncMock(side_effect=lambda rels, **kw: [(r, True) for r in rels])

        coord, session = _make_coordinator_with_fake_txn(
            relational=rel_backend, vector=vec_backend, graph=graph_backend
        )

        result = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=old_doc_id,
            new_document=new_doc,
            new_chunks=new_chunks,
            new_entities=[new_entity_survivor, new_entity_net_new],
            new_relationships=[new_rel_survivor, new_rel_net_new],
        )

        # PG transaction committed once
        assert session.committed is True
        assert session.rolled_back is False

        # PG ops were issued against the txn session
        rel_backend.update_document.assert_any_await(new_doc, session=session)
        vec_backend.delete_chunks_by_document.assert_awaited_with(
            old_doc_id, namespace_id=namespace_id, session=session
        )
        vec_backend.create_chunks_batch.assert_awaited_with(new_chunks, session=session)

        # Retirement: orphan entity and orphan relationship only
        retire_ents_call = graph_backend.retire_orphaned_entities_batch.await_args.args[0]
        assert len(retire_ents_call) == 1
        assert retire_ents_call[0]["current_id"] == str(orphan_entity_id)
        assert retire_ents_call[0]["namespace_id"] == str(namespace_id)

        retire_rels_call = graph_backend.retire_orphaned_relationships_batch.await_args.args[0]
        assert len(retire_rels_call) == 1
        assert retire_rels_call[0]["relationship_id"] == orphan_rel_id
        assert retire_rels_call[0]["old_doc_id"] == old_doc_id

        # Remap: survivor entity + survivor relationship
        remap_kwargs = graph_backend.remap_source_document_ids_batch.await_args.kwargs
        assert len(remap_kwargs["entity_survivors"]) == 1
        assert remap_kwargs["entity_survivors"][0]["entity_id"] == str(survivor_entity_id)
        assert remap_kwargs["entity_survivors"][0]["old_doc_id"] == str(old_doc_id)
        assert remap_kwargs["entity_survivors"][0]["new_doc_id"] == str(new_doc.id)
        assert len(remap_kwargs["relationship_survivors"]) == 1
        assert remap_kwargs["relationship_survivors"][0]["relationship_id"] == str(survivor_rel_id)

        # Upsert is called with net-new entities ONLY (survivor excluded)
        upsert_args = graph_backend.upsert_entities_batch.await_args.args
        assert upsert_args[0] == namespace_id
        assert len(upsert_args[1]) == 1
        assert upsert_args[1][0].name == "carol"

        # create_relationships_batch is called with net-new rel ONLY
        create_rel_args = graph_backend.create_relationships_batch.await_args.args
        assert len(create_rel_args[0]) == 1
        assert create_rel_args[0][0].relationship_type == "WORKS_WITH"

        # Result counts line up
        assert isinstance(result, ReplaceResult)
        assert result.document_id == new_doc.id
        assert result.chunks_deleted == 7
        assert result.chunks_created == 3
        assert result.entities_retired == 1
        assert result.entities_created == 1  # carol
        assert result.entities_updated == 1  # alice (survivor)
        assert result.relationships_retired == 1
        assert result.relationships_created == 1

        # Document was marked COMPLETED
        assert new_doc.status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_graph_failure_after_pg_commit_keeps_completed_and_reraises(self) -> None:
        """Fix for #887 (refined by #884): when PG commits and the graph phase
        fails, the document stays COMPLETED (data is fully written), a WARNING
        is logged, and a typed exception wraps the original so the caller can
        record the divergence.
        """
        from khora.exceptions import GraphMirrorFailedAfterPGCommitError

        namespace_id = uuid4()
        old_doc_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="new body")
        new_chunks = [Chunk(namespace_id=namespace_id, document_id=new_doc.id)]

        rel_backend = MagicMock()
        rel_backend.update_document = AsyncMock(return_value=new_doc)

        vec_backend = MagicMock()
        vec_backend.delete_chunks_by_document = AsyncMock(return_value=0)
        vec_backend.create_chunks_batch = AsyncMock(return_value=new_chunks)

        graph_backend = MagicMock()
        graph_backend.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [
                    {
                        "id": str(uuid4()),
                        "name": "bob",
                        "entity_type": "PERSON",
                        "namespace_id": str(namespace_id),
                        "source_document_count": 1,
                    }
                ],
                [],
            )
        )
        boom = RuntimeError("neo4j down")
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=boom)
        graph_backend.retire_orphaned_relationships_batch = AsyncMock()
        graph_backend.remap_source_document_ids_batch = AsyncMock()
        graph_backend.upsert_entities_batch = AsyncMock(return_value=[])
        graph_backend.create_relationships_batch = AsyncMock(return_value=0)

        coord, session = _make_coordinator_with_fake_txn(
            relational=rel_backend, vector=vec_backend, graph=graph_backend
        )

        # Capture loguru WARNING by patching the module-bound logger.
        with patch("khora.storage.coordinator.logger") as mock_logger:
            with pytest.raises(GraphMirrorFailedAfterPGCommitError) as exc_info:
                await coord.replace_document_extraction(
                    namespace_id=namespace_id,
                    old_document_id=old_doc_id,
                    new_document=new_doc,
                    new_chunks=new_chunks,
                    new_entities=[],
                    new_relationships=[],
                )

        # The original exception is preserved via __cause__ (raise ... from).
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "neo4j down"
        assert exc_info.value.document_id == new_doc.id
        assert exc_info.value.namespace_id == namespace_id
        assert exc_info.value.original_exception_type == "RuntimeError"

        # PG is committed even though the graph step raised.
        assert session.committed is True
        assert session.rolled_back is False
        # #887: document stays COMPLETED. The fully-written chunks + entity
        # counts must not be contradicted by a FAILED status.
        assert new_doc.status == DocumentStatus.COMPLETED
        assert new_doc.error_message is None
        assert new_doc.chunk_count == len(new_chunks)
        assert new_doc.entity_count == 0
        # update_document was called exactly once (in-tx status stamp) -
        # the buggy post-tx mark_failed write has been removed.
        assert rel_backend.update_document.await_count == 1
        # A WARNING was logged about the graph-phase failure (#887).
        mock_logger.warning.assert_called_once()
        warn_args = mock_logger.warning.call_args
        assert "#887" in warn_args.args[0]

    @pytest.mark.asyncio
    async def test_pg_failure_rolls_back_and_does_not_mark_failed(self) -> None:
        """Fix for #887: if the PG transaction raises, the rollback reverts
        the in-tx status stamp; the coordinator no longer issues an
        out-of-band ``mark_failed`` update against the rolled-back row.
        """
        namespace_id = uuid4()
        old_doc_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="new body")
        # Caller-set pre-replace status (typical: PROCESSING from the
        # ingest pipeline).
        new_doc.mark_processing()

        rel_backend = MagicMock()
        pg_error = RuntimeError("pg-constraint-violation")
        rel_backend.update_document = AsyncMock(side_effect=pg_error)

        vec_backend = MagicMock()
        vec_backend.delete_chunks_by_document = AsyncMock(return_value=0)
        vec_backend.create_chunks_batch = AsyncMock()

        graph_backend = MagicMock()
        graph_backend.fetch_document_extraction_state = AsyncMock(return_value=([], []))
        graph_backend.retire_orphaned_entities_batch = AsyncMock()
        graph_backend.retire_orphaned_relationships_batch = AsyncMock()
        graph_backend.remap_source_document_ids_batch = AsyncMock()
        graph_backend.upsert_entities_batch = AsyncMock()
        graph_backend.create_relationships_batch = AsyncMock()

        coord, session = _make_coordinator_with_fake_txn(
            relational=rel_backend, vector=vec_backend, graph=graph_backend
        )

        with pytest.raises(RuntimeError, match="pg-constraint-violation"):
            await coord.replace_document_extraction(
                namespace_id=namespace_id,
                old_document_id=old_doc_id,
                new_document=new_doc,
                new_chunks=[],
                new_entities=[],
                new_relationships=[],
            )

        assert session.committed is False
        assert session.rolled_back is True
        # Graph work was never issued.
        graph_backend.retire_orphaned_entities_batch.assert_not_awaited()
        graph_backend.retire_orphaned_relationships_batch.assert_not_awaited()
        graph_backend.remap_source_document_ids_batch.assert_not_awaited()
        graph_backend.upsert_entities_batch.assert_not_awaited()
        # In the rolled-back-tx case, the only update_document call is the
        # in-tx attempt that raised.  The buggy post-tx mark_failed update
        # has been removed (#887), so no second write happens.
        assert rel_backend.update_document.await_count == 1
        # Document was NOT marked FAILED post-tx; the PG row keeps its
        # pre-replace status because the SQL tx rolled back.
        assert new_doc.status != DocumentStatus.FAILED
        assert new_doc.error_message is None

    @pytest.mark.asyncio
    async def test_relationship_type_sanitization_classifies_survivor_correctly(self) -> None:
        """Mixed-case / punctuated rel types on new_relationships must sanitize to match Cypher storage.

        Regression guard: the Neo4j backend stores the sanitized (upper-case,
        alphanumerics+underscore) label, so the coordinator's survivor/orphan
        filter must sanitize the Python-side value or the relationship will
        be misclassified as both orphan (retire) and net-new (create).
        """
        namespace_id = uuid4()
        old_doc_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="b")
        new_chunks: list[Chunk] = []

        src_id = uuid4()
        tgt_id = uuid4()
        rel_id = uuid4()

        # Neo4j stores "KNOWS" (upper); caller passes "Knows" (mixed case).
        new_rel = Relationship(
            namespace_id=namespace_id,
            source_entity_id=src_id,
            target_entity_id=tgt_id,
            relationship_type="Knows",  # sanitizes to "KNOWS"
            source_document_ids=[new_doc.id],
        )

        rel_backend = MagicMock()
        rel_backend.update_document = AsyncMock(return_value=new_doc)

        vec_backend = MagicMock()
        vec_backend.delete_chunks_by_document = AsyncMock(return_value=0)
        vec_backend.create_chunks_batch = AsyncMock(return_value=new_chunks)
        del vec_backend.upsert_entities_batch

        graph_backend = MagicMock()
        graph_backend.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [],
                [
                    {
                        "id": str(rel_id),
                        "source_entity_id": str(src_id),
                        "target_entity_id": str(tgt_id),
                        "relationship_type": "KNOWS",  # sanitized form from Cypher
                        "source_document_count": 2,
                    }
                ],
            )
        )
        graph_backend.retire_orphaned_entities_batch = AsyncMock(return_value=0)
        graph_backend.retire_orphaned_relationships_batch = AsyncMock(return_value=0)
        graph_backend.remap_source_document_ids_batch = AsyncMock()
        graph_backend.upsert_entities_batch = AsyncMock(return_value=[])
        graph_backend.create_relationships_batch = AsyncMock(return_value=0)

        coord, _ = _make_coordinator_with_fake_txn(relational=rel_backend, vector=vec_backend, graph=graph_backend)

        result = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=old_doc_id,
            new_document=new_doc,
            new_chunks=new_chunks,
            new_entities=[],
            new_relationships=[new_rel],
        )

        # Classified as SURVIVOR → remap called, not retire, not create.
        graph_backend.retire_orphaned_relationships_batch.assert_not_awaited()
        graph_backend.create_relationships_batch.assert_not_awaited()
        remap_kwargs = graph_backend.remap_source_document_ids_batch.await_args.kwargs
        assert len(remap_kwargs["relationship_survivors"]) == 1
        assert remap_kwargs["relationship_survivors"][0]["relationship_id"] == str(rel_id)
        assert result.relationships_retired == 0
        assert result.relationships_created == 0

    @pytest.mark.asyncio
    async def test_no_prefetch_state_skips_retire_and_remap(self) -> None:
        """When there is no old graph state, only PG work + net-new upsert run."""
        namespace_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="fresh")
        new_chunks = [Chunk(namespace_id=namespace_id, document_id=new_doc.id)]

        rel_backend = MagicMock()
        rel_backend.update_document = AsyncMock(return_value=new_doc)

        vec_backend = MagicMock()
        vec_backend.delete_chunks_by_document = AsyncMock(return_value=0)
        vec_backend.create_chunks_batch = AsyncMock(return_value=new_chunks)
        del vec_backend.upsert_entities_batch

        graph_backend = MagicMock()
        graph_backend.fetch_document_extraction_state = AsyncMock(return_value=([], []))
        graph_backend.retire_orphaned_entities_batch = AsyncMock(return_value=0)
        graph_backend.retire_orphaned_relationships_batch = AsyncMock(return_value=0)
        graph_backend.remap_source_document_ids_batch = AsyncMock()
        entity = Entity(
            namespace_id=namespace_id,
            name="dana",
            entity_type="PERSON",
            source_document_ids=[new_doc.id],
        )
        graph_backend.upsert_entities_batch = AsyncMock(return_value=[(entity, True)])
        graph_backend.create_relationships_batch = AsyncMock(return_value=0)

        coord, session = _make_coordinator_with_fake_txn(
            relational=rel_backend, vector=vec_backend, graph=graph_backend
        )

        result = await coord.replace_document_extraction(
            namespace_id=namespace_id,
            old_document_id=uuid4(),
            new_document=new_doc,
            new_chunks=new_chunks,
            new_entities=[entity],
            new_relationships=[],
        )

        assert session.committed is True
        # Empty retirement rows → no call issued
        graph_backend.retire_orphaned_entities_batch.assert_not_awaited()
        graph_backend.retire_orphaned_relationships_batch.assert_not_awaited()
        graph_backend.remap_source_document_ids_batch.assert_not_awaited()
        assert result.entities_created == 1
        assert result.entities_updated == 0
        assert result.entities_retired == 0


class TestReplaceDocumentPartialFailureObservability:
    """Regression for issue #884: graph-mirror failure after PG commit must
    increment ``khora.storage.replace_document.partial_failure`` exactly once,
    leave PG state durable (chunks + COMPLETED), and surface as a typed
    ``GraphMirrorFailedAfterPGCommitError`` so the caller can record the
    divergence on its user-facing result (ADR-001 degradation convention).
    """

    @pytest.mark.asyncio
    async def test_graph_failure_after_pg_commit_increments_counter_and_raises_typed(
        self,
    ) -> None:
        """When the graph-mirror phase of ``replace_document_extraction`` raises
        AFTER the PG transaction commits, the coordinator must:

        1. Leave PG durable: chunks created, document marked COMPLETED.
        2. Increment ``khora.storage.replace_document.partial_failure`` exactly
           once with the documented metric name and unit.
        3. Wrap the underlying exception in ``GraphMirrorFailedAfterPGCommitError``
           so the caller can distinguish "graph mirror partial" from
           "full rollback" and record a degradation on the user-facing result.
        """
        from khora.exceptions import GraphMirrorFailedAfterPGCommitError

        namespace_id = uuid4()
        old_doc_id = uuid4()
        new_doc = Document(namespace_id=namespace_id, content="new body")
        new_chunks = [
            Chunk(namespace_id=namespace_id, document_id=new_doc.id, content="c1"),
            Chunk(namespace_id=namespace_id, document_id=new_doc.id, content="c2"),
        ]

        rel_backend = MagicMock()
        rel_backend.update_document = AsyncMock(return_value=new_doc)

        vec_backend = MagicMock()
        vec_backend.delete_chunks_by_document = AsyncMock(return_value=3)
        vec_backend.create_chunks_batch = AsyncMock(return_value=new_chunks)
        # Strip the optional vector-side entity writer so the coordinator's
        # upsert_entities_batch routes through the graph-only branch and the
        # net-new entity below tries the graph adapter.
        del vec_backend.upsert_entities_batch

        graph_backend = MagicMock()
        graph_backend.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [
                    {
                        "id": str(uuid4()),
                        "name": "alice",
                        "entity_type": "PERSON",
                        "namespace_id": str(namespace_id),
                        "source_document_count": 1,
                    }
                ],
                [],
            )
        )
        # First post-PG graph op blows up.
        boom = RuntimeError("neo4j unreachable")
        graph_backend.retire_orphaned_entities_batch = AsyncMock(side_effect=boom)
        graph_backend.retire_orphaned_relationships_batch = AsyncMock()
        graph_backend.remap_source_document_ids_batch = AsyncMock()
        graph_backend.upsert_entities_batch = AsyncMock(return_value=[])
        graph_backend.create_relationships_batch = AsyncMock(return_value=0)

        coord, session = _make_coordinator_with_fake_txn(
            relational=rel_backend, vector=vec_backend, graph=graph_backend
        )

        counter_mock = MagicMock()
        with patch("khora.storage.coordinator.metric_counter", return_value=counter_mock) as metric_counter_patched:
            with pytest.raises(GraphMirrorFailedAfterPGCommitError) as exc_info:
                await coord.replace_document_extraction(
                    namespace_id=namespace_id,
                    old_document_id=old_doc_id,
                    new_document=new_doc,
                    new_chunks=new_chunks,
                    new_entities=[],
                    new_relationships=[],
                )

        # --- Assert PG state is durable (caller can see PG committed) ------
        assert session.committed is True
        assert session.rolled_back is False
        # Doc row is COMPLETED with chunk count stamped in-tx.
        assert new_doc.status == DocumentStatus.COMPLETED
        assert new_doc.chunk_count == len(new_chunks)
        assert new_doc.error_message is None
        # update_document was issued exactly once (the in-tx status stamp).
        assert rel_backend.update_document.await_count == 1
        # Chunks were inserted within the PG transaction.
        vec_backend.create_chunks_batch.assert_awaited_once()

        # --- Assert partial_failure counter incremented exactly once -------
        # The contract for this metric: name + unit must match the
        # docs/telemetry-contract.json entry (issue #884).
        partial_failure_calls = [
            c
            for c in metric_counter_patched.call_args_list
            if c.args and c.args[0] == "khora.storage.replace_document.partial_failure"
        ]
        assert len(partial_failure_calls) == 1, (
            f"expected exactly one partial_failure counter creation, got "
            f"{len(partial_failure_calls)}: {metric_counter_patched.call_args_list}"
        )
        assert partial_failure_calls[0].kwargs["unit"] == "1"
        # The counter object's .add was called exactly once with delta=1.
        counter_mock.add.assert_called_once_with(1)

        # --- Assert the failure propagates to the caller observably -------
        # The typed exception preserves the original via __cause__ and
        # surfaces the document/namespace identity so the caller can record
        # an ADR-001 degradation on the user-facing result.
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "neo4j unreachable"
        assert exc_info.value.document_id == new_doc.id
        assert exc_info.value.namespace_id == namespace_id
        assert exc_info.value.original_exception_type == "RuntimeError"

        # --- Assert no data divergence wrt to what the caller sees --------
        # PG-side state is fully consistent (caller sees "PG is good"):
        # chunks present, status COMPLETED, no FAILED stamp.
        assert new_doc.status != DocumentStatus.FAILED
        # Graph-side state is partial (caller sees "graph is partial"):
        # the first graph op raised; the later ones (remap, upsert,
        # create_relationships) were never reached.
        graph_backend.retire_orphaned_entities_batch.assert_awaited_once()
        graph_backend.remap_source_document_ids_batch.assert_not_awaited()
        graph_backend.upsert_entities_batch.assert_not_awaited()
        graph_backend.create_relationships_batch.assert_not_awaited()


class TestUpsertEntitiesBatchOrdering:
    """Regression for issue #868: vector-first ordering + partial-failure metric."""

    @pytest.mark.asyncio
    async def test_vector_runs_before_graph_and_graph_failure_is_observable(self) -> None:
        """When graph raises after vector succeeds, the graph exception
        propagates AND the partial_failure counter increments exactly once.

        Locks in the vector-first ordering and the observability hook the
        fix for #868 introduced. The previous gather-based implementation
        could commit graph nodes before vector raised, leaving orphan
        graph rows the read path could not see.
        """
        ns_id = uuid4()
        entity = Entity(namespace_id=ns_id, name="alice", entity_type="PERSON")

        call_order: list[str] = []

        async def vector_upsert(*args, **kwargs):
            call_order.append("vector")
            return [(entity, True)]

        async def graph_upsert(*args, **kwargs):
            call_order.append("graph")
            raise RuntimeError("neo4j boom")

        vec = MagicMock()
        vec.upsert_entities_batch = AsyncMock(side_effect=vector_upsert)
        graph = MagicMock()
        graph.upsert_entities_batch = AsyncMock(side_effect=graph_upsert)
        # Force the dual-backend (non-unified) branch.
        # __post_init__ probes ``_conn`` on both backends; leaving them as
        # MagicMock attributes makes the probe yield distinct objects, so
        # _is_unified_backend stays False.

        coord = StorageCoordinator(vector=vec, graph=graph)
        assert coord._is_unified_backend is False  # sanity

        counter_mock = MagicMock()
        with patch("khora.storage.coordinator.metric_counter", return_value=counter_mock) as metric_counter_patched:
            with pytest.raises(RuntimeError, match="neo4j boom"):
                await coord.upsert_entities_batch(ns_id, [entity])

        # 1. Vector ran before graph.
        assert call_order == ["vector", "graph"], call_order
        # 2. Both backends were called once with the same entity batch.
        vec.upsert_entities_batch.assert_awaited_once()
        graph.upsert_entities_batch.assert_awaited_once()
        # 3. The partial_failure counter was created with the documented
        #    name and incremented exactly once.
        metric_counter_patched.assert_called_once()
        assert metric_counter_patched.call_args.args[0] == "khora.storage.upsert_entities_batch.partial_failure"
        counter_mock.add.assert_called_once_with(1)


class TestCreateEntityOrdering:
    """Regression for issue #1138: single-entity create dual-write must mirror
    the #868 batch ordering - vector first, then graph, with a partial-failure
    metric when graph raises after vector committed (instead of asyncio.gather
    with no partial-failure handling)."""

    @pytest.mark.asyncio
    async def test_vector_runs_before_graph_and_graph_failure_is_observable(self) -> None:
        ns_id = uuid4()
        entity = Entity(namespace_id=ns_id, name="alice", entity_type="PERSON")

        call_order: list[str] = []

        async def vector_create(*args, **kwargs):
            call_order.append("vector")
            return entity

        async def graph_create(*args, **kwargs):
            call_order.append("graph")
            raise RuntimeError("neo4j boom")

        vec = MagicMock()
        vec.create_entity = AsyncMock(side_effect=vector_create)
        graph = MagicMock()
        graph.create_entity = AsyncMock(side_effect=graph_create)

        coord = StorageCoordinator(vector=vec, graph=graph)
        assert coord._is_unified_backend is False  # sanity

        counter_mock = MagicMock()
        with patch("khora.storage.coordinator.metric_counter", return_value=counter_mock) as metric_counter_patched:
            with patch("khora.telemetry.get_collector") as mock_telem:
                mock_telem.return_value.record_storage_op = MagicMock()
                with pytest.raises(RuntimeError, match="neo4j boom"):
                    await coord.create_entity(entity)

        # Vector committed before graph; the graph exception is not masked.
        assert call_order == ["vector", "graph"], call_order
        vec.create_entity.assert_awaited_once()
        graph.create_entity.assert_awaited_once()
        # Partial failure was recorded (not silently swallowed).
        partial_failure_calls = [
            c
            for c in metric_counter_patched.call_args_list
            if c.args and c.args[0] == "khora.storage.create_entity.partial_failure"
        ]
        assert len(partial_failure_calls) == 1, metric_counter_patched.call_args_list
        counter_mock.add.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_no_partial_failure_metric_on_happy_path(self) -> None:
        ns_id = uuid4()
        entity = Entity(namespace_id=ns_id, name="alice", entity_type="PERSON")
        vec = MagicMock()
        vec.create_entity = AsyncMock(return_value=entity)
        graph = MagicMock()
        graph.create_entity = AsyncMock(return_value=entity)

        coord = StorageCoordinator(vector=vec, graph=graph)
        counter_mock = MagicMock()
        with patch("khora.storage.coordinator.metric_counter", return_value=counter_mock) as metric_counter_patched:
            with patch("khora.telemetry.get_collector") as mock_telem:
                mock_telem.return_value.record_storage_op = MagicMock()
                out = await coord.create_entity(entity)

        assert out is entity
        vec.create_entity.assert_awaited_once()
        graph.create_entity.assert_awaited_once()
        partial_failure_calls = [
            c
            for c in metric_counter_patched.call_args_list
            if c.args and c.args[0] == "khora.storage.create_entity.partial_failure"
        ]
        assert partial_failure_calls == []


class TestUpdateEntityOrdering:
    """Regression for issue #1138: single-entity update dual-write must mirror
    the #868 batch ordering - vector first, then graph, with a partial-failure
    metric when graph raises after vector committed."""

    @pytest.mark.asyncio
    async def test_vector_runs_before_graph_and_graph_failure_is_observable(self) -> None:
        ns_id = uuid4()
        entity = Entity(namespace_id=ns_id, name="alice", entity_type="PERSON")

        call_order: list[str] = []

        async def vector_update(*args, **kwargs):
            call_order.append("vector")
            return entity

        async def graph_update(*args, **kwargs):
            call_order.append("graph")
            raise RuntimeError("neo4j boom")

        vec = MagicMock()
        vec.update_entity = AsyncMock(side_effect=vector_update)
        graph = MagicMock()
        graph.update_entity = AsyncMock(side_effect=graph_update)

        coord = StorageCoordinator(vector=vec, graph=graph)
        assert coord._is_unified_backend is False  # sanity

        counter_mock = MagicMock()
        with patch("khora.storage.coordinator.metric_counter", return_value=counter_mock) as metric_counter_patched:
            with pytest.raises(RuntimeError, match="neo4j boom"):
                await coord.update_entity(entity, namespace_id=ns_id)

        assert call_order == ["vector", "graph"], call_order
        vec.update_entity.assert_awaited_once()
        graph.update_entity.assert_awaited_once()
        partial_failure_calls = [
            c
            for c in metric_counter_patched.call_args_list
            if c.args and c.args[0] == "khora.storage.update_entity.partial_failure"
        ]
        assert len(partial_failure_calls) == 1, metric_counter_patched.call_args_list
        counter_mock.add.assert_called_once_with(1)

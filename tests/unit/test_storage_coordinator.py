"""Unit tests for storage/coordinator.py — StorageCoordinator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document, Entity, MemoryEvent, Relationship
from khora.storage.coordinator import StorageCoordinator, StorageHealth


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
        rel = MagicMock()
        rel.get_document = AsyncMock(return_value=None)
        coord = StorageCoordinator(relational=rel)
        await coord.get_document(doc_id)
        rel.get_document.assert_awaited_once_with(doc_id)

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
        rel = MagicMock()
        rel.delete_document = AsyncMock(return_value=True)
        vec = MagicMock()
        vec.delete_chunks_by_document = AsyncMock()

        coord = StorageCoordinator(relational=rel, vector=vec)
        result = await coord.delete_document(doc_id)

        vec.delete_chunks_by_document.assert_awaited_once_with(doc_id)
        rel.delete_document.assert_awaited_once_with(doc_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_missing_relational(self) -> None:
        """Operations without relational raise RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Relational backend not configured"):
            await coord.create_document(MagicMock())


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
        entity = MagicMock(spec=Entity)
        graph = MagicMock()
        graph.update_entity = AsyncMock(return_value=entity)
        vec = MagicMock()
        vec.update_entity = AsyncMock()

        coord = StorageCoordinator(graph=graph, vector=vec)
        await coord.update_entity(entity)

        graph.update_entity.assert_awaited_once()
        vec.update_entity.assert_awaited_once()

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
        """Empty relationships list returns 0."""
        coord = StorageCoordinator()
        count = await coord.create_relationships_batch([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_entity_relationships(self) -> None:
        """get_entity_relationships delegates to graph."""
        entity_id = uuid4()
        graph = MagicMock()
        graph.get_entity_relationships = AsyncMock(return_value=[])
        coord = StorageCoordinator(graph=graph)
        result = await coord.get_entity_relationships(entity_id)
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_graph(self) -> None:
        """create_relationship without graph raises RuntimeError."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="Graph backend not configured"):
            await coord.create_relationship(MagicMock())


class TestGraphOps:
    """Tests for graph traversal operations."""

    @pytest.mark.asyncio
    async def test_get_neighborhood_no_graph(self) -> None:
        """get_neighborhood without graph returns empty structure."""
        coord = StorageCoordinator()
        result = await coord.get_neighborhood(uuid4())
        assert result == {"entities": [], "relationships": []}

    @pytest.mark.asyncio
    async def test_find_paths_no_graph(self) -> None:
        """find_paths without graph returns empty list."""
        coord = StorageCoordinator()
        result = await coord.find_paths(uuid4(), uuid4(), uuid4())
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
        result = await coord.get_entities_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_documents_batch_empty(self) -> None:
        """Empty document_ids returns empty dict."""
        coord = StorageCoordinator()
        result = await coord.get_documents_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_neighborhoods_batch_empty(self) -> None:
        """Empty entity_ids returns empty dict."""
        coord = StorageCoordinator()
        result = await coord.get_neighborhoods_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_update_entity_embeddings_batch_fallback(self) -> None:
        """Fallback to individual updates when batch not supported."""
        vec = MagicMock(spec=[])  # No upsert_entities_batch method
        vec.update_entity_embedding = AsyncMock()
        coord = StorageCoordinator(vector=vec)

        entity_id = uuid4()
        updates = [(entity_id, [0.1, 0.2], "model")]
        count = await coord.update_entity_embeddings_batch(updates)
        assert count == 1
        vec.update_entity_embedding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_entity_embeddings_batch_no_vector(self) -> None:
        """No vector backend returns 0."""
        coord = StorageCoordinator()
        count = await coord.update_entity_embeddings_batch([])
        assert count == 0


class TestDocumentSourcesBatch:
    """Tests for get_document_sources_batch (DYT-506)."""

    @pytest.mark.asyncio
    async def test_get_document_sources_batch(self) -> None:
        """get_document_sources_batch delegates to relational backend."""
        from khora.core.models.document import DocumentSource

        doc_id = uuid4()
        src = DocumentSource(id=doc_id, title="Test Doc")

        rel = MagicMock()
        rel.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        coord = StorageCoordinator(relational=rel)
        result = await coord.get_document_sources_batch([doc_id])

        assert result == {doc_id: src}
        rel.get_document_sources_batch.assert_awaited_once_with([doc_id])

    @pytest.mark.asyncio
    async def test_get_document_sources_batch_empty(self) -> None:
        """Empty list returns empty dict without hitting backend."""
        rel = MagicMock()
        rel.get_document_sources_batch = AsyncMock()

        coord = StorageCoordinator(relational=rel)
        result = await coord.get_document_sources_batch([])

        assert result == {}
        rel.get_document_sources_batch.assert_not_awaited()

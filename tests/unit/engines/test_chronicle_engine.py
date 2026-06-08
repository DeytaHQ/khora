"""Unit tests for the Chronicle engine."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.chronicle.engine import ChronicleEngine


def _ent(entity_id, source_document_ids):
    return SimpleNamespace(id=entity_id, source_document_ids=list(source_document_ids))


@pytest.mark.unit
class TestChronicleEngineForget:
    """Tests for ChronicleEngine.forget() and its vector-anchored cascade (#923).

    Cleanup is anchored on ``source_document_ids`` refcounting read off the
    vector store (pgvector). The Neo4j graph backend is mirrored opportunistically.
    """

    @pytest.fixture
    def connected_engine(self) -> ChronicleEngine:
        """Mock-connected ChronicleEngine. The vector backend lists empty
        entities/relationships by default; the graph backend exposes the
        Neo4j-style batch helpers."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        config.storage.backend = "pgvector"

        engine = ChronicleEngine(config)
        engine._storage = AsyncMock()
        # Vector backend is the authoritative refcount store on pg stacks.
        # Spec it to the pgvector method surface (no ``*_batch`` strip helpers)
        # so the cascade dispatches to the real pgvector method names.
        engine._storage.vector = MagicMock(
            spec=[
                "list_entities",
                "list_relationships",
                "delete_entities_batch",
                "delete_relationships_batch",
                "remove_document_from_entity_sources",
                "remove_document_from_relationship_sources",
            ]
        )
        engine._storage.vector.list_entities = AsyncMock(return_value=[])
        engine._storage.vector.list_relationships = AsyncMock(return_value=[])
        engine._storage.vector.delete_entities_batch = AsyncMock()
        engine._storage.vector.delete_relationships_batch = AsyncMock()
        engine._storage.vector.remove_document_from_entity_sources = AsyncMock()
        engine._storage.vector.remove_document_from_relationship_sources = AsyncMock()
        # Graph backend exposes the Neo4j-style batch helpers (mirror).
        engine._storage.graph = MagicMock()
        engine._storage.graph.list_relationships = AsyncMock(return_value=[])
        engine._storage.graph.delete_entities_batch = AsyncMock()
        engine._storage.graph.delete_relationships_batch = AsyncMock()
        engine._storage.graph.remove_document_from_entity_sources_batch = AsyncMock()
        engine._storage.graph.remove_document_from_relationship_sources_batch = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_forget_cascade_deletes_orphan_entity(self, connected_engine: ChronicleEngine) -> None:
        """Orphan entity (sole source = forgotten doc) is hard-deleted in both backends."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_ent_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_entities = AsyncMock(return_value=[_ent(orphan_ent_id, [doc_id])])

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._storage.vector.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_updates_survivor_entity_sources(self, connected_engine: ChronicleEngine) -> None:
        """Survivor entity has doc_id stripped from source_document_ids; not deleted."""
        doc_id = uuid4()
        namespace_id = uuid4()
        survivor_ent_id = uuid4()
        other_doc = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_entities = AsyncMock(
            return_value=[_ent(survivor_ent_id, [doc_id, other_doc])]
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.remove_document_from_entity_sources.assert_awaited_once_with(
            [survivor_ent_id], doc_id
        )
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_awaited_once_with(
            [survivor_ent_id], doc_id, namespace_id
        )
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_deletes_orphan_relationship(self, connected_engine: ChronicleEngine) -> None:
        """Orphan relationship is hard-deleted from both backends."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_rel_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_relationships = AsyncMock(return_value=[_ent(orphan_rel_id, [doc_id])])

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.delete_relationships_batch.assert_awaited_once_with(
            [orphan_rel_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.delete_relationships_batch.assert_awaited_once_with(
            [orphan_rel_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_updates_survivor_relationship_sources(
        self, connected_engine: ChronicleEngine
    ) -> None:
        """Survivor relationship has doc_id stripped, not deleted."""
        doc_id = uuid4()
        namespace_id = uuid4()
        survivor_rel_id = uuid4()
        other_doc = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.vector.list_relationships = AsyncMock(
            return_value=[_ent(survivor_rel_id, [doc_id, other_doc])]
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.vector.remove_document_from_relationship_sources.assert_awaited_once_with(
            [survivor_rel_id], doc_id
        )
        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_awaited_once_with(
            [survivor_rel_id], doc_id, namespace_id
        )
        connected_engine._storage.vector.delete_relationships_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_zero_extraction_skips_backend_calls(self, connected_engine: ChronicleEngine) -> None:
        """Document with no extracted entities/relationships: cascade lists empty
        sets and issues no delete/strip calls; document deletion still happens."""
        doc_id = uuid4()
        namespace_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()
        connected_engine._storage.vector.delete_relationships_batch.assert_not_called()
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()
        connected_engine._storage.graph.delete_relationships_batch.assert_not_called()
        connected_engine._storage.delete_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_cascade_cleans_when_graph_lacks_fetch_state(self, connected_engine: ChronicleEngine) -> None:
        """#923 regression: graph-less / non-Neo4j chronicle stacks (no
        ``fetch_document_extraction_state``) must STILL clean orphan entities
        from the vector store, not silently no-op."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_ent_id = uuid4()

        # Graph backend without any cleanup helpers (graph-less PG-only stack).
        connected_engine._storage.graph = None
        connected_engine._storage.vector.list_entities = AsyncMock(return_value=[_ent(orphan_ent_id, [doc_id])])

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        # The vector store still cleaned the orphan - no silent no-op.
        connected_engine._storage.vector.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.delete_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_namespace_mismatch_skips_cascade(self, connected_engine: ChronicleEngine) -> None:
        """When the caller-supplied namespace does not match the document's,
        forget short-circuits to False BEFORE the cascade runs."""
        doc_id = uuid4()
        namespace_id = uuid4()

        connected_engine._storage.get_document = AsyncMock(return_value=None)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is False
        connected_engine._storage.get_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)
        connected_engine._storage.vector.list_entities.assert_not_called()
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()
        connected_engine._storage.delete_document.assert_not_called()

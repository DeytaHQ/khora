"""Unit tests for the Chronicle engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.engines.chronicle.engine import ChronicleEngine


@pytest.mark.unit
class TestChronicleEngineForget:
    """Tests for ChronicleEngine.forget() and its cascade into entities/relationships."""

    @pytest.fixture
    def connected_engine(self) -> ChronicleEngine:
        """Mock-connected ChronicleEngine. Graph backend exposes
        ``fetch_document_extraction_state`` returning an empty pair by default."""
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
        engine._storage.graph.fetch_document_extraction_state = AsyncMock(return_value=([], []))
        return engine

    @pytest.mark.asyncio
    async def test_forget_cascade_deletes_orphan_entity(self, connected_engine: ChronicleEngine) -> None:
        """Orphan entity is hard-deleted in both backends."""
        doc_id = uuid4()
        namespace_id = uuid4()
        orphan_ent_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.graph.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [{"id": str(orphan_ent_id), "source_document_count": 1}],
                [],
            )
        )

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._storage.graph.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.vector.delete_entities_batch.assert_awaited_once_with(
            [orphan_ent_id], namespace_id=namespace_id
        )
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_updates_survivor_entity_sources(self, connected_engine: ChronicleEngine) -> None:
        """Survivor entity has doc_id stripped from source_document_ids; not deleted."""
        doc_id = uuid4()
        namespace_id = uuid4()
        survivor_ent_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.graph.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [{"id": str(survivor_ent_id), "source_document_count": 4}],
                [],
            )
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_awaited_once_with(
            [survivor_ent_id], doc_id, namespace_id
        )
        connected_engine._storage.vector.remove_document_from_entity_sources.assert_awaited_once_with(
            [survivor_ent_id], doc_id
        )
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()

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
        connected_engine._storage.graph.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [],
                [{"id": str(orphan_rel_id), "source_document_count": 1}],
            )
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.graph.delete_relationships_batch.assert_awaited_once_with(
            [orphan_rel_id], namespace_id=namespace_id
        )
        connected_engine._storage.vector.delete_relationships_batch.assert_awaited_once_with(
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

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)
        connected_engine._storage.graph.fetch_document_extraction_state = AsyncMock(
            return_value=(
                [],
                [{"id": str(survivor_rel_id), "source_document_count": 2}],
            )
        )

        await connected_engine.forget(doc_id, namespace_id)

        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_awaited_once_with(
            [survivor_rel_id], doc_id
        )
        connected_engine._storage.vector.remove_document_from_relationship_sources.assert_awaited_once_with(
            [survivor_rel_id], doc_id
        )
        connected_engine._storage.graph.delete_relationships_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forget_cascade_zero_extraction_skips_backend_calls(self, connected_engine: ChronicleEngine) -> None:
        """Document with no extracted entities/relationships: cascade is a no-op
        but document deletion still happens."""
        doc_id = uuid4()
        namespace_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()
        connected_engine._storage.graph.delete_relationships_batch.assert_not_called()
        connected_engine._storage.graph.remove_document_from_entity_sources_batch.assert_not_called()
        connected_engine._storage.graph.remove_document_from_relationship_sources_batch.assert_not_called()
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()
        connected_engine._storage.vector.delete_relationships_batch.assert_not_called()
        connected_engine._storage.delete_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_cascade_no_op_when_graph_lacks_fetch_state(self, connected_engine: ChronicleEngine) -> None:
        """Pgvector-only chronicle deployments (no Neo4j) — graph backend does
        not expose ``fetch_document_extraction_state``. Cascade short-circuits
        cleanly without raising and without calling any delete helpers."""
        doc_id = uuid4()
        namespace_id = uuid4()

        # Replace the graph mock with a plain MagicMock that does NOT have
        # fetch_document_extraction_state. ``getattr(..., None)`` must return None.
        graph_without_fetch = MagicMock(spec=[])
        connected_engine._storage.graph = graph_without_fetch

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        # Nothing on graph or vector was called for the cascade.
        connected_engine._storage.vector.delete_entities_batch.assert_not_called()
        connected_engine._storage.vector.delete_relationships_batch.assert_not_called()
        # Document deletion still happens.
        connected_engine._storage.delete_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)

    @pytest.mark.asyncio
    async def test_forget_namespace_mismatch_skips_cascade(self, connected_engine: ChronicleEngine) -> None:
        """When the caller-supplied namespace does not match the document's,
        forget short-circuits to False BEFORE the cascade runs.

        Security: namespace mismatch is now enforced at the SQL layer —
        ``storage.get_document(doc_id, namespace_id=wrong_ns)`` just returns
        ``None`` and the engine bails."""
        doc_id = uuid4()
        namespace_id = uuid4()

        connected_engine._storage.get_document = AsyncMock(return_value=None)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is False
        connected_engine._storage.get_document.assert_awaited_once_with(doc_id, namespace_id=namespace_id)
        connected_engine._storage.graph.fetch_document_extraction_state.assert_not_called()
        connected_engine._storage.graph.delete_entities_batch.assert_not_called()
        connected_engine._storage.delete_document.assert_not_called()

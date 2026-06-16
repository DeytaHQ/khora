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
        # #1237: the Neo4j graph backend now sweeps malformed orphan edges that
        # list_relationships can't deserialize. Default to "nothing swept".
        engine._storage.graph.delete_malformed_orphan_relationships = AsyncMock(return_value=0)
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


@pytest.mark.unit
class TestChronicleEngineRememberDedupScope:
    """Checksum dedup is scoped by caller-supplied external_id / session_id (#1139)."""

    @pytest.fixture
    def connected_engine(self) -> ChronicleEngine:
        """Mock-connected ChronicleEngine for the remember() dedup path."""
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
        return engine

    async def _remember(self, engine: ChronicleEngine, namespace_id, **kwargs):
        """Drive remember() with the post-dedup pipeline stubbed out."""
        from unittest.mock import patch

        with (
            patch(
                "khora.pipelines.flows.ingest.process_document",
                new=AsyncMock(return_value={"chunks": 1, "entities": 0, "relationships": 0, "chunk_ids": []}),
            ),
            patch.object(engine, "_events_enabled", return_value=False),
            patch.object(engine, "_facts_enabled", return_value=False),
        ):
            return await engine.remember(
                "yes",
                namespace_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
                **kwargs,
            )

    @pytest.mark.asyncio
    async def test_remember_same_content_new_session_creates_document(self, connected_engine: ChronicleEngine) -> None:
        """#1139: a checksum hit stored under a different session_id is NOT a duplicate."""
        namespace_id = uuid4()
        session_b = uuid4()

        existing_doc = MagicMock()
        existing_doc.id = uuid4()
        existing_doc.external_id = None
        existing_doc.session_id = uuid4()
        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        created_doc = MagicMock()
        created_doc.id = uuid4()
        connected_engine._storage.create_document = AsyncMock(return_value=created_doc)

        result = await self._remember(connected_engine, namespace_id, metadata={"session_id": str(session_b)})

        assert result.metadata.get("duplicate") is None
        assert result.document_id == created_doc.id
        connected_engine._storage.create_document.assert_called_once()
        assert connected_engine._storage.create_document.call_args[0][0].session_id == session_b

    @pytest.mark.asyncio
    async def test_remember_same_content_new_external_id_creates_document(
        self, connected_engine: ChronicleEngine
    ) -> None:
        """#1139: a checksum hit stored under a different external_id is NOT a duplicate."""
        namespace_id = uuid4()

        existing_doc = MagicMock()
        existing_doc.id = uuid4()
        existing_doc.external_id = "ext-a"
        existing_doc.session_id = None
        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        created_doc = MagicMock()
        created_doc.id = uuid4()
        connected_engine._storage.create_document = AsyncMock(return_value=created_doc)

        result = await self._remember(connected_engine, namespace_id, external_id="ext-b")

        assert result.metadata.get("duplicate") is None
        assert result.document_id == created_doc.id
        connected_engine._storage.create_document.assert_called_once()
        assert connected_engine._storage.create_document.call_args[0][0].external_id == "ext-b"

    @pytest.mark.asyncio
    async def test_remember_same_content_same_external_id_still_dedups(self, connected_engine: ChronicleEngine) -> None:
        """#1139: a checksum hit with a matching external_id is still a duplicate."""
        namespace_id = uuid4()

        existing_doc = MagicMock()
        existing_doc.id = uuid4()
        existing_doc.status = "completed"
        existing_doc.chunk_count = 1
        existing_doc.entity_count = 0
        existing_doc.relationship_count = 0
        existing_doc.external_id = "ext-1"
        existing_doc.session_id = None
        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        result = await self._remember(connected_engine, namespace_id, external_id="ext-1")

        assert result.metadata.get("duplicate") is True
        assert result.document_id == existing_doc.id
        connected_engine._storage.create_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_remember_same_content_no_identity_still_dedups(self, connected_engine: ChronicleEngine) -> None:
        """#1139: callers that supply neither external_id nor session_id keep checksum-only dedup."""
        namespace_id = uuid4()

        existing_doc = MagicMock()
        existing_doc.id = uuid4()
        existing_doc.status = "completed"
        existing_doc.chunk_count = 1
        existing_doc.entity_count = 0
        existing_doc.relationship_count = 0
        existing_doc.external_id = None
        existing_doc.session_id = uuid4()
        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        result = await self._remember(connected_engine, namespace_id)

        assert result.metadata.get("duplicate") is True
        connected_engine._storage.create_document.assert_not_called()

"""Tests for DYT-1953: count_documents, get_last_activity_at, get_document_stats, and engine stats().

Covers:
- Engine stats() methods returning last_activity_at for all 4 engines
- Engine stats() fallback behavior when storage methods raise
- SurrealDB relational adapter: count_documents, get_last_activity_at, get_document_stats
- StorageCoordinator: get_document_stats delegation (not on coordinator yet, tested at backend level)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.memory_lake import Stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_khora_config() -> MagicMock:
    """Create a mock KhoraConfig sufficient for engine construction."""
    config = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/test"
    config.get_neo4j_url.return_value = None
    config.get_neo4j_user.return_value = None
    config.get_neo4j_password.return_value = None
    config.get_neo4j_database.return_value = None
    config.get_graph_config.return_value = None
    config.get_vector_config.return_value = None
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.embedding_dimension = 1536
    config.llm.model = "gpt-4o-mini"
    config.llm.embedding_model = "text-embedding-3-small"
    config.llm.embedding_dimension = 1536
    config.llm.timeout = 30
    config.llm.max_retries = 3
    config.llm.extraction_model = None
    config.llm.max_concurrent_llm_calls = 5
    config.pipeline.chunking_strategy = "recursive"
    config.pipeline.chunk_size = 1000
    config.pipeline.chunk_overlap = 200
    config.pipeline.extract_entities = True
    config.telemetry_database_url = None
    config.telemetry_service_name = "test"
    return config


def _mock_storage(
    *,
    doc_count: int = 5,
    chunk_count: int = 20,
    entity_count: int = 10,
    relationship_count: int = 8,
    last_activity_at: datetime | None = None,
) -> AsyncMock:
    """Create a mock StorageCoordinator with count methods.

    Engines call get_document_stats() which returns (doc_count, last_activity_at).
    """
    storage = AsyncMock()
    storage.get_document_stats = AsyncMock(return_value=(doc_count, last_activity_at))
    storage.count_chunks = AsyncMock(return_value=chunk_count)
    storage.count_entities = AsyncMock(return_value=entity_count)
    storage.count_relationships = AsyncMock(return_value=relationship_count)
    return storage


# ===========================================================================
# Engine stats() tests
# ===========================================================================


@pytest.mark.unit
class TestChronicleEngineStats:
    """Tests for ChronicleEngine.stats() with last_activity_at."""

    @pytest.mark.asyncio
    async def test_stats_returns_last_activity_at(self) -> None:
        """stats() populates last_activity_at from storage."""
        from khora.engines.chronicle.engine import ChronicleEngine

        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        config = _mock_khora_config()
        engine = ChronicleEngine(config)
        engine._connected = True
        engine._storage = _mock_storage(
            doc_count=3,
            chunk_count=10,
            entity_count=5,
            relationship_count=2,
            last_activity_at=ts,
        )

        ns_id = uuid4()
        result = await engine.stats(ns_id)

        assert isinstance(result, Stats)
        assert result.documents == 3
        assert result.chunks == 10
        assert result.entities == 5
        assert result.relationships == 2
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_last_activity_at_none_when_empty(self) -> None:
        """stats() returns None for last_activity_at when namespace is empty."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _mock_khora_config()
        engine = ChronicleEngine(config)
        engine._connected = True
        engine._storage = _mock_storage(
            doc_count=0,
            chunk_count=0,
            entity_count=0,
            relationship_count=0,
            last_activity_at=None,
        )

        result = await engine.stats(uuid4())
        assert result.last_activity_at is None
        assert result.documents == 0

    @pytest.mark.asyncio
    async def test_stats_fallback_on_attribute_error(self) -> None:
        """stats() gracefully falls back when storage methods raise AttributeError."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _mock_khora_config()
        engine = ChronicleEngine(config)
        engine._connected = True

        storage = AsyncMock()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        storage.count_chunks = AsyncMock(side_effect=AttributeError)
        storage.count_entities = AsyncMock(side_effect=AttributeError)
        storage.count_relationships = AsyncMock(side_effect=AttributeError)
        engine._storage = storage

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.chunks == 0
        assert result.entities == 0
        assert result.relationships == 0
        assert result.last_activity_at is None

    @pytest.mark.asyncio
    async def test_stats_fallback_on_not_implemented(self) -> None:
        """stats() gracefully falls back when storage methods raise NotImplementedError."""
        from khora.engines.chronicle.engine import ChronicleEngine

        config = _mock_khora_config()
        engine = ChronicleEngine(config)
        engine._connected = True

        storage = AsyncMock()
        storage.get_document_stats = AsyncMock(side_effect=NotImplementedError)
        storage.count_chunks = AsyncMock(side_effect=NotImplementedError)
        storage.count_entities = AsyncMock(side_effect=NotImplementedError)
        storage.count_relationships = AsyncMock(side_effect=NotImplementedError)
        engine._storage = storage

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.last_activity_at is None


@pytest.mark.unit
class TestGraphRAGEngineStats:
    """Tests for GraphRAGEngine.stats() with last_activity_at."""

    @pytest.mark.asyncio
    async def test_stats_returns_last_activity_at(self) -> None:
        """stats() populates last_activity_at from storage."""
        from khora.engines.graphrag.engine import GraphRAGEngine

        ts = datetime(2026, 3, 15, 8, 30, 0, tzinfo=UTC)
        config = _mock_khora_config()
        engine = GraphRAGEngine(config)
        engine._connected = True
        engine._storage = _mock_storage(
            doc_count=7,
            chunk_count=35,
            entity_count=12,
            relationship_count=9,
            last_activity_at=ts,
        )

        result = await engine.stats(uuid4())

        assert result.documents == 7
        assert result.chunks == 35
        assert result.entities == 12
        assert result.relationships == 9
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_fallback_on_error(self) -> None:
        """stats() returns zeros and None when all storage methods fail."""
        from khora.engines.graphrag.engine import GraphRAGEngine

        config = _mock_khora_config()
        engine = GraphRAGEngine(config)
        engine._connected = True

        storage = AsyncMock()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        storage.count_chunks = AsyncMock(side_effect=NotImplementedError)
        storage.count_entities = AsyncMock(side_effect=AttributeError)
        storage.count_relationships = AsyncMock(side_effect=NotImplementedError)
        engine._storage = storage

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.chunks == 0
        assert result.entities == 0
        assert result.relationships == 0
        assert result.last_activity_at is None


@pytest.mark.unit
class TestSkeletonEngineStats:
    """Tests for SkeletonConstructionEngine.stats() with last_activity_at."""

    @pytest.mark.asyncio
    async def test_stats_returns_last_activity_at(self) -> None:
        """stats() populates last_activity_at from storage."""
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        ts = datetime(2026, 2, 20, 16, 45, 0, tzinfo=UTC)
        config = _mock_khora_config()
        engine = SkeletonConstructionEngine(config, backend="pgvector")
        engine._connected = True
        engine._storage = _mock_storage(
            doc_count=2,
            chunk_count=8,
            entity_count=4,
            relationship_count=1,
            last_activity_at=ts,
        )

        result = await engine.stats(uuid4())

        assert result.documents == 2
        assert result.chunks == 8
        assert result.entities == 4
        assert result.relationships == 1
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_fallback_all_errors(self) -> None:
        """stats() returns zeros and None when all storage methods fail."""
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        config = _mock_khora_config()
        engine = SkeletonConstructionEngine(config, backend="pgvector")
        engine._connected = True

        storage = AsyncMock()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        storage.count_chunks = AsyncMock(side_effect=NotImplementedError)
        storage.count_entities = AsyncMock(side_effect=AttributeError)
        storage.count_relationships = AsyncMock(side_effect=NotImplementedError)
        engine._storage = storage

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.chunks == 0
        assert result.entities == 0
        assert result.relationships == 0
        assert result.last_activity_at is None


@pytest.mark.unit
class TestVectorCypherEngineStats:
    """Tests for VectorCypherEngine.stats() with last_activity_at."""

    def _make_engine(self) -> VectorCypherEngine:  # noqa: F821
        from khora.engines.vectorcypher.engine import VectorCypherEngine

        config = _mock_khora_config()
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        return VectorCypherEngine(config)

    @pytest.mark.asyncio
    async def test_stats_routes_chunks_through_storage(self) -> None:
        """stats() gets chunk count from storage.count_chunks (DYT-2116)."""
        ts = datetime(2026, 1, 10, 9, 0, 0, tzinfo=UTC)
        engine = self._make_engine()
        engine._connected = True
        engine._storage = _mock_storage(
            doc_count=4,
            chunk_count=15,
            entity_count=6,
            relationship_count=3,
            last_activity_at=ts,
        )

        result = await engine.stats(uuid4())

        assert result.documents == 4
        assert result.chunks == 15
        assert result.entities == 6
        assert result.relationships == 3
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_fallback_on_error(self) -> None:
        """stats() returns zeros and None when storage methods fail."""
        engine = self._make_engine()
        engine._connected = True

        storage = AsyncMock()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        storage.count_chunks = AsyncMock(return_value=0)
        storage.count_entities = AsyncMock(side_effect=NotImplementedError)
        storage.count_relationships = AsyncMock(side_effect=AttributeError)
        engine._storage = storage

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.chunks == 0
        assert result.entities == 0
        assert result.relationships == 0
        assert result.last_activity_at is None

    @pytest.mark.asyncio
    async def test_stats_partial_failure(self) -> None:
        """stats() returns partial results when only some methods fail."""
        ts = datetime(2026, 4, 1, 0, 0, 0, tzinfo=UTC)
        engine = self._make_engine()
        engine._connected = True

        storage = AsyncMock()
        storage.get_document_stats = AsyncMock(return_value=(10, ts))
        storage.count_chunks = AsyncMock(return_value=50)
        storage.count_entities = AsyncMock(side_effect=AttributeError)
        storage.count_relationships = AsyncMock(return_value=5)
        engine._storage = storage

        result = await engine.stats(uuid4())

        assert result.documents == 10
        assert result.chunks == 50
        assert result.entities == 0  # fallback
        assert result.relationships == 5
        assert result.last_activity_at == ts


# ===========================================================================
# SurrealDB Backend Tests
# ===========================================================================


def _make_mock_conn(**query_returns: object) -> MagicMock:
    """Create a mock SurrealDBConnection with sensible defaults."""
    conn = MagicMock()
    conn.connected = True
    conn.connect = AsyncMock()
    conn.disconnect = AsyncMock()
    conn.is_healthy = AsyncMock(return_value=True)
    conn.query = AsyncMock(return_value=query_returns.get("query", []))
    conn.query_one = AsyncMock(return_value=query_returns.get("query_one", None))
    conn.execute = AsyncMock(return_value=query_returns.get("execute", None))
    return conn


@pytest.mark.unit
class TestSurrealDBCountDocuments:
    """Tests for SurrealDBRelationalAdapter.count_documents()."""

    @pytest.mark.asyncio
    async def test_count_documents(self) -> None:
        """count_documents returns count from SurrealDB query."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 42})
        adapter = SurrealDBRelationalAdapter(conn)

        ns_id = uuid4()
        result = await adapter.count_documents(ns_id)

        assert result == 42
        conn.query_one.assert_awaited_once()
        call_args = conn.query_one.call_args
        assert str(ns_id) in str(call_args)

    @pytest.mark.asyncio
    async def test_count_documents_empty_namespace(self) -> None:
        """count_documents returns 0 when no documents exist."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.count_documents(uuid4())
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_documents_null_cnt(self) -> None:
        """count_documents returns 0 when cnt is None."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": None})
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.count_documents(uuid4())
        assert result == 0


@pytest.mark.unit
class TestSurrealDBGetLastActivityAt:
    """Tests for SurrealDBRelationalAdapter.get_last_activity_at()."""

    @pytest.mark.asyncio
    async def test_get_last_activity_at(self) -> None:
        """get_last_activity_at returns the latest timestamp."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"latest": ts})
        adapter = SurrealDBRelationalAdapter(conn)

        ns_id = uuid4()
        result = await adapter.get_last_activity_at(ns_id)

        assert result == ts

    @pytest.mark.asyncio
    async def test_get_last_activity_at_empty_namespace(self) -> None:
        """get_last_activity_at returns None when namespace is empty."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_last_activity_at(uuid4())
        assert result is None


@pytest.mark.unit
class TestSurrealDBGetDocumentStats:
    """Tests for SurrealDBRelationalAdapter.get_document_stats()."""

    @pytest.mark.asyncio
    async def test_get_document_stats(self) -> None:
        """get_document_stats returns (count, last_activity_at) tuple."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 15, "latest": ts})
        adapter = SurrealDBRelationalAdapter(conn)

        ns_id = uuid4()
        count, last_activity = await adapter.get_document_stats(ns_id)

        assert count == 15
        assert last_activity == ts

    @pytest.mark.asyncio
    async def test_get_document_stats_empty_namespace(self) -> None:
        """get_document_stats returns (0, None) for empty namespace."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        count, last_activity = await adapter.get_document_stats(uuid4())

        assert count == 0
        assert last_activity is None

    @pytest.mark.asyncio
    async def test_get_document_stats_null_cnt(self) -> None:
        """get_document_stats handles null cnt gracefully."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": None, "latest": None})
        adapter = SurrealDBRelationalAdapter(conn)

        count, last_activity = await adapter.get_document_stats(uuid4())

        assert count == 0
        assert last_activity is None


# ===========================================================================
# Base backend get_document_stats default implementation
# ===========================================================================


@pytest.mark.unit
class TestBaseBackendGetDocumentStats:
    """Tests for RelationalBackend.get_document_stats() default implementation."""

    @pytest.mark.asyncio
    async def test_default_delegates_to_individual_methods(self) -> None:
        """Default get_document_stats calls count_documents + get_last_activity_at."""
        from khora.storage.backends.base import RelationalBackendProtocol

        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)

        # Create a concrete subclass with mocked abstract methods
        class FakeRelational(RelationalBackendProtocol):
            async def connect(self): ...
            async def disconnect(self): ...
            async def is_healthy(self) -> bool:
                return True

            async def create_namespace(self, ns): ...
            async def get_namespace(self, ns_id): ...
            async def resolve_namespace(self, ns_id): ...
            async def deactivate_namespace(self, ns_id): ...
            async def update_namespace(self, ns): ...
            async def list_namespaces(self, **kw): ...
            async def create_namespace_version(self, **kw): ...
            async def create_document(self, doc): ...
            async def get_document(self, doc_id): ...
            async def list_documents(self, ns_id, **kw): ...
            async def update_document(self, doc): ...
            async def delete_document(self, doc_id): ...
            async def get_document_by_checksum(self, ns_id, checksum): ...
            async def get_document_by_external_id(self, ns_id, external_id): ...
            async def get_sync_checkpoint(self, ns_id, source): ...
            async def set_sync_checkpoint(self, ns_id, source, checkpoint): ...

            async def count_documents(self, namespace_id):
                return 42

            async def get_last_activity_at(self, namespace_id):
                return ts

        backend = FakeRelational()
        count, last_activity = await backend.get_document_stats(uuid4())

        assert count == 42
        assert last_activity == ts

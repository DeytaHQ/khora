"""Unit tests for engine stats() methods — last_activity_at and fallback behavior."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from loguru import logger as loguru_logger

from khora.khora import Stats


def _make_mock_storage(
    *,
    doc_count: int = 5,
    chunk_count: int = 20,
    entity_count: int = 10,
    relationship_count: int = 8,
    last_activity_at: datetime | None = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC),
) -> AsyncMock:
    """Create a mock StorageCoordinator with count methods."""
    storage = AsyncMock()
    storage.count_documents = AsyncMock(return_value=doc_count)
    storage.count_chunks = AsyncMock(return_value=chunk_count)
    storage.count_entities = AsyncMock(return_value=entity_count)
    storage.count_relationships = AsyncMock(return_value=relationship_count)
    storage.get_last_activity_at = AsyncMock(return_value=last_activity_at)
    storage.get_document_stats = AsyncMock(return_value=(doc_count, last_activity_at))
    return storage


# =========================================================================
# Chronicle
# =========================================================================


class TestChronicleStats:
    """Tests for ChronicleEngine.stats()."""

    def _make_engine(self, storage: AsyncMock) -> object:
        from khora.engines.chronicle.engine import ChronicleEngine

        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"

        engine = ChronicleEngine(config)
        engine._connected = True
        engine._storage = storage
        return engine

    @pytest.mark.asyncio
    async def test_stats_includes_last_activity_at(self) -> None:
        """stats() returns last_activity_at from storage."""
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        storage = _make_mock_storage(last_activity_at=ts)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert isinstance(result, Stats)
        assert result.documents == 5
        assert result.chunks == 20
        assert result.entities == 10
        assert result.relationships == 8
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_last_activity_at_none(self) -> None:
        """stats() returns None last_activity_at for empty namespace."""
        storage = _make_mock_storage(last_activity_at=None)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.last_activity_at is None

    @pytest.mark.asyncio
    async def test_stats_get_document_stats_fallback(self) -> None:
        """stats() handles AttributeError on get_document_stats gracefully."""
        storage = _make_mock_storage()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.last_activity_at is None

    @pytest.mark.asyncio
    async def test_stats_get_document_stats_not_implemented(self) -> None:
        """stats() handles NotImplementedError on get_document_stats."""
        storage = _make_mock_storage()
        storage.get_document_stats = AsyncMock(side_effect=NotImplementedError)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.last_activity_at is None


# =========================================================================
# Skeleton
# =========================================================================


class TestSkeletonStats:
    """Tests for SkeletonConstructionEngine.stats()."""

    def _make_engine(self, storage: AsyncMock) -> object:
        from khora.engines.skeleton.engine import SkeletonConstructionEngine

        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"

        engine = SkeletonConstructionEngine(config)
        engine._connected = True
        engine._storage = storage
        engine._temporal_store = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_stats_includes_last_activity_at(self) -> None:
        """stats() returns last_activity_at from storage."""
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        storage = _make_mock_storage(last_activity_at=ts)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert isinstance(result, Stats)
        assert result.documents == 5
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_get_document_stats_fallback(self) -> None:
        """stats() defaults doc_count to 0 on get_document_stats failure."""
        storage = _make_mock_storage()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.last_activity_at is None
        # list_documents should NOT be called as fallback
        storage.list_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stats_count_chunks_fallback(self) -> None:
        """stats() handles count_chunks failure gracefully."""
        storage = _make_mock_storage()
        storage.count_chunks = AsyncMock(side_effect=NotImplementedError)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.chunks == 0
        assert result.documents == 5


# =========================================================================
# VectorCypher
# =========================================================================


class TestVectorCypherStats:
    """Tests for VectorCypherEngine.stats()."""

    def _make_engine(self, storage: AsyncMock, *, dual_nodes: AsyncMock | None = None) -> object:
        from khora.engines.vectorcypher.engine import VectorCypherEngine

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
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = storage
        engine._temporal_store = AsyncMock()
        engine._temporal_store.count_chunks = AsyncMock(return_value=20)
        engine._dual_nodes = dual_nodes
        engine._neo4j_driver = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_stats_includes_last_activity_at(self) -> None:
        """stats() returns last_activity_at from storage."""
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        storage = _make_mock_storage(last_activity_at=ts)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert isinstance(result, Stats)
        assert result.documents == 5
        assert result.last_activity_at == ts

    @pytest.mark.asyncio
    async def test_stats_uses_storage_for_chunks(self) -> None:
        """stats() uses storage.count_chunks for chunk count."""
        storage = _make_mock_storage(chunk_count=42)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.chunks == 42
        storage.count_chunks.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stats_get_document_stats_fallback(self) -> None:
        """stats() defaults doc_count to 0 on get_document_stats failure."""
        storage = _make_mock_storage()
        storage.get_document_stats = AsyncMock(side_effect=AttributeError)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.documents == 0
        assert result.last_activity_at is None
        storage.list_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stats_relationship_count_fallback(self) -> None:
        """stats() handles NotImplementedError on count_relationships."""
        storage = _make_mock_storage()
        storage.count_relationships = AsyncMock(side_effect=NotImplementedError)
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.relationships == 0
        assert result.documents == 5

    @pytest.mark.asyncio
    async def test_stats_unexpected_error_degrades_to_zero(self) -> None:
        """stats() degrades unexpected errors to 0 without breaking other counts."""
        storage = _make_mock_storage()
        storage.count_relationships = AsyncMock(side_effect=RuntimeError("pool exhausted"))
        engine = self._make_engine(storage)

        result = await engine.stats(uuid4())

        assert result.relationships == 0
        assert result.documents == 5
        assert result.entities == 10


# =========================================================================
# Stats dataclass
# =========================================================================


class TestStatsDataclass:
    """Tests for the Stats dataclass itself."""

    def test_last_activity_at_default_none(self) -> None:
        """last_activity_at defaults to None."""
        stats = Stats(documents=1, chunks=2, entities=3, relationships=4)
        assert stats.last_activity_at is None

    def test_last_activity_at_set(self) -> None:
        """last_activity_at can be set explicitly."""
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        stats = Stats(documents=1, chunks=2, entities=3, relationships=4, last_activity_at=ts)
        assert stats.last_activity_at == ts

    def test_stats_frozen(self) -> None:
        """Stats is frozen (immutable)."""
        stats = Stats(documents=1, chunks=2, entities=3, relationships=4)
        with pytest.raises(AttributeError):
            stats.documents = 10  # type: ignore[misc]

    def test_metadata_default_empty(self) -> None:
        """metadata defaults to an empty dict."""
        stats = Stats(documents=1, chunks=2, entities=3, relationships=4)
        assert stats.metadata == {}


# =========================================================================
# ADR-001: stats() counter failures are surfaced, not silently zeroed (#878)
# =========================================================================


class TestStatsCounterFailureObservability:
    """A counter that raises must NOT be indistinguishable from a count of 0.

    Per ADR-001: log at WARNING, append an ErrorRecord to Stats.metadata
    ['errors'], bump khora.stats.counter_failed_total{engine, counter}, and
    do NOT raise.
    """

    @pytest.mark.asyncio
    async def test_vectorcypher_counter_failure_surfaced(self) -> None:
        storage = _make_mock_storage()
        storage.count_entities = AsyncMock(side_effect=AttributeError("no count_entities"))
        engine = TestVectorCypherStats()._make_engine(storage)

        with patch("khora.engines._stats._STATS_COUNTER_FAILED") as mock_counter:
            result = await engine.stats(uuid4())

        # (a) does not raise; int stays 0
        assert result.entities == 0
        # other counters still ran
        assert result.chunks == 20
        assert result.relationships == 8
        # (c) ErrorRecord attached
        errors = result.metadata["errors"]
        assert len(errors) == 1
        assert errors[0]["component"] == "vectorcypher.stats.count_entities"
        assert errors[0]["reason"] == "counter_unavailable"
        assert errors[0]["exception"] == "AttributeError"
        # (d) metric incremented with engine + counter labels
        assert mock_counter.add.called
        args, kwargs = mock_counter.add.call_args_list[0]
        assert args[0] == 1
        assert kwargs["attributes"] == {"engine": "vectorcypher", "counter": "entities"}

    @pytest.mark.asyncio
    async def test_vectorcypher_counter_failure_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        storage = _make_mock_storage()
        storage.count_entities = AsyncMock(side_effect=RuntimeError("pool exhausted"))
        engine = TestVectorCypherStats()._make_engine(storage)

        # Bridge loguru -> stdlib so caplog sees the WARNING.
        sink_id = loguru_logger.add(
            lambda m: logging.getLogger("khora.engines._stats").log(m.record["level"].no, m.record["message"]),
            level="WARNING",
            format="{message}",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="khora.engines._stats"):
                result = await engine.stats(uuid4())
        finally:
            loguru_logger.remove(sink_id)

        assert result.entities == 0
        assert any("count_entities failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_chronicle_counter_failure_surfaced(self) -> None:
        storage = _make_mock_storage()
        storage.count_relationships = AsyncMock(side_effect=NotImplementedError)
        engine = TestChronicleStats()._make_engine(storage)

        with patch("khora.engines._stats._STATS_COUNTER_FAILED") as mock_counter:
            result = await engine.stats(uuid4())

        assert result.relationships == 0
        assert result.entities == 10
        errors = result.metadata["errors"]
        assert errors[0]["component"] == "chronicle.stats.count_relationships"
        assert errors[0]["exception"] == "NotImplementedError"
        assert mock_counter.add.called
        _, kwargs = mock_counter.add.call_args_list[0]
        assert kwargs["attributes"] == {"engine": "chronicle", "counter": "relationships"}

    @pytest.mark.asyncio
    async def test_all_counters_succeed_no_errors_key(self) -> None:
        """Happy path: metadata has no 'errors' key when every counter succeeds."""
        storage = _make_mock_storage()
        engine = TestVectorCypherStats()._make_engine(storage)
        result = await engine.stats(uuid4())
        assert "errors" not in result.metadata

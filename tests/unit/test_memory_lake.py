"""Unit tests for memory_lake.py — MemoryLake primary API."""

from __future__ import annotations

import warnings
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.memory_lake import BatchResult, MemoryLake, RecallResult, RememberResult, Stats

from .helpers import RESOLVE_ROW_ID as _RESOLVE_ROW_ID
from .helpers import make_lake as _make_lake
from .helpers import mock_config as _mock_config
from .helpers import mock_engine as _mock_engine

# ---------------------------------------------------------------------------
# RememberResult / RecallResult dataclass tests
# ---------------------------------------------------------------------------


class TestRememberResult:
    """Tests for RememberResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        r = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=5,
            entities_extracted=3,
            relationships_created=2,
        )
        assert r.chunks_created == 5
        assert r.entities_extracted == 3
        assert r.relationships_created == 2
        assert r.metadata == {}

    def test_custom_metadata(self) -> None:
        """Custom metadata can be set."""
        r = RememberResult(
            document_id=uuid4(),
            namespace_id=uuid4(),
            chunks_created=0,
            entities_extracted=0,
            relationships_created=0,
            metadata={"duplicate": True},
        )
        assert r.metadata["duplicate"] is True


class TestRecallResult:
    """Tests for RecallResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        from khora.core.models.document import Chunk
        from khora.core.models.entity import Entity

        ns_id = uuid4()
        chunk = Chunk(namespace_id=ns_id, document_id=uuid4(), content="hello")
        entity = Entity(namespace_id=ns_id, name="Alice", entity_type="PERSON")
        r = RecallResult(
            query="test query",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[(entity, 0.8)],
            context_text="some text",
        )
        assert r.query == "test query"
        assert r.namespace_id == ns_id
        assert len(r.chunks) == 1
        assert len(r.entities) == 1
        assert r.context_text == "some text"

    def test_default_metadata(self) -> None:
        """Default metadata is empty dict."""
        r = RecallResult(
            query="q",
            namespace_id=uuid4(),
            chunks=[],
            entities=[],
            context_text="",
        )
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# MemoryLake initialization
# ---------------------------------------------------------------------------


class TestMemoryLakeInit:
    """Tests for MemoryLake initialization."""

    def test_init_default(self) -> None:
        """Default init loads config from env."""
        lake = _make_lake()
        assert lake._connected is False
        assert lake._engine is None

    def test_init_with_config(self) -> None:
        """Init with explicit config skips load_config."""
        from khora.config import KhoraConfig

        # Create a real KhoraConfig (not a mock) to trigger the isinstance check
        cfg = KhoraConfig(database_url="postgresql://test")
        lake = MemoryLake(cfg)

        assert lake._config is cfg
        assert lake._config.database_url == "postgresql://test"

    def test_init_with_storage_config(self) -> None:
        """Init with explicit storage_config uses it directly."""
        storage_cfg = MagicMock()
        with patch("khora.memory_lake.load_config", return_value=_mock_config()):
            lake = MemoryLake(storage_config=storage_cfg)
        assert lake._storage_config is storage_cfg

    def test_not_connected_properties_raise(self) -> None:
        """Accessing storage before connect raises."""
        lake = _make_lake()

        with pytest.raises(RuntimeError, match="not connected"):
            _ = lake.storage

    def test_connected_properties_return(self) -> None:
        """Accessing storage after connect succeeds."""
        lake = _make_lake(connected=True)
        assert lake.storage is lake._engine._storage


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    """Tests for connect() and disconnect() lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """connect() creates engine and sets flag."""
        lake = _make_lake()

        mock_engine = _mock_engine()

        with patch("khora.engines.create_engine", return_value=mock_engine):
            await lake.connect()

        assert lake._connected is True
        assert lake._engine is mock_engine
        mock_engine.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        """Calling connect() when already connected is a no-op."""
        lake = _make_lake(connected=True)
        original_engine = lake._engine

        await lake.connect()

        assert lake._engine is original_engine

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """disconnect() tears down all components."""
        lake = _make_lake(connected=True)

        await lake.disconnect()

        assert lake._connected is False
        assert lake._engine is None

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self) -> None:
        """Calling disconnect() when not connected is a no-op."""
        lake = _make_lake()
        await lake.disconnect()  # Should not raise
        assert lake._connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """async with MemoryLake() connects and disconnects."""
        lake = _make_lake()
        lake.connect = AsyncMock()
        lake.disconnect = AsyncMock()

        async with lake as ctx:
            assert ctx is lake
            lake.connect.assert_awaited_once()

        lake.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# _resolve_namespace
# ---------------------------------------------------------------------------


class TestResolveNamespace:
    """Tests for _resolve_namespace helper.

    _resolve_namespace now performs a DB lookup via storage.resolve_namespace()
    to map a stable namespace_id to the active version's row-level id.
    """

    @pytest.mark.asyncio
    async def test_uuid_calls_resolve(self) -> None:
        """UUID is forwarded to storage.resolve_namespace()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        row_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=row_id)

        result = await lake._resolve_namespace(ns_id)
        assert result == row_id
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_uuid_string_parsed_and_resolved(self) -> None:
        """UUID string is parsed then forwarded to storage.resolve_namespace()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        row_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(return_value=row_id)

        result = await lake._resolve_namespace(str(ns_id))
        assert result == row_id
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(ns_id)

    @pytest.mark.asyncio
    async def test_invalid_string_raises_value_error(self) -> None:
        """Non-UUID string raises ValueError before DB lookup."""
        lake = _make_lake(connected=True)
        with pytest.raises(ValueError, match="Invalid namespace"):
            await lake._resolve_namespace("not-a-uuid")

    @pytest.mark.asyncio
    async def test_no_active_version_raises(self) -> None:
        """ValueError from storage.resolve_namespace propagates."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(
            side_effect=ValueError(f"No active namespace version found for namespace_id={ns_id}")
        )

        with pytest.raises(ValueError, match="No active namespace version"):
            await lake._resolve_namespace(ns_id)


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


class TestRemember:
    """Tests for remember()."""

    @pytest.mark.asyncio
    async def test_remember_delegates_to_engine(self) -> None:
        """remember() delegates to engine.remember()."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=ns_id,
            chunks_created=3,
            entities_extracted=2,
            relationships_created=1,
        )
        lake._engine.remember = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember(
                "test content",
                namespace=ns_id,
                title="Test",
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert result == mock_result
        assert result.llm_usage == []
        lake._engine.remember.assert_awaited_once()


# ---------------------------------------------------------------------------
# remember_batch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


class TestRecall:
    """Tests for recall()."""

    @pytest.mark.asyncio
    async def test_recall_delegates_to_engine(self) -> None:
        """recall() delegates to engine.recall() and returns result."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="search query",
            namespace_id=ns_id,
            chunks=[("chunk", 0.9)],
            entities=[("entity", 0.8)],
            context_text="found content",
            metadata={"mode": "HYBRID"},
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("search query", namespace=ns_id)

        assert isinstance(result, RecallResult)
        assert result.query == "search query"
        lake._engine.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_passes_search_mode(self) -> None:
        """recall() passes mode to engine."""
        from khora.query.engine import SearchMode

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test", namespace=ns_id, mode=SearchMode.VECTOR)

        call_kwargs = lake._engine.recall.call_args
        assert call_kwargs.kwargs.get("mode") == SearchMode.VECTOR


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    """Tests for forget()."""

    @pytest.mark.asyncio
    async def test_forget_delegates_to_engine(self) -> None:
        """forget() delegates to engine.forget() with resolved namespace."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        ns_id = uuid4()

        lake._engine.forget = AsyncMock(return_value=True)

        result = await lake.forget(doc_id, namespace=ns_id)
        assert result is True
        lake._engine.forget.assert_awaited_once_with(doc_id, _RESOLVE_ROW_ID)


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


class TestEntityOperations:
    """Tests for entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_entity(self) -> None:
        """get_entity delegates to engine."""
        lake = _make_lake(connected=True)
        entity_id = uuid4()
        mock_entity = MagicMock()

        lake._engine.get_entity = AsyncMock(return_value=mock_entity)

        result = await lake.get_entity(entity_id)
        assert result is mock_entity
        lake._engine.get_entity.assert_awaited_once_with(entity_id)

    @pytest.mark.asyncio
    async def test_list_entities(self) -> None:
        """list_entities delegates to engine with resolved namespace."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_entities = [MagicMock(), MagicMock()]
        lake._engine.list_entities = AsyncMock(return_value=mock_entities)

        result = await lake.list_entities(namespace=ns_id, entity_type="PERSON", limit=50)
        assert result == mock_entities
        lake._engine.list_entities.assert_awaited_once_with(_RESOLVE_ROW_ID, entity_type="PERSON", limit=50)

    @pytest.mark.asyncio
    async def test_find_related_entities(self) -> None:
        """find_related_entities delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        entity_id = uuid4()

        mock_related = [(MagicMock(), 0.8)]
        lake._engine.find_related_entities = AsyncMock(return_value=mock_related)

        result = await lake.find_related_entities(entity_id, namespace=ns_id, max_depth=3)
        assert result == mock_related


# ---------------------------------------------------------------------------
# Namespace management
# ---------------------------------------------------------------------------


class TestNamespaceManagement:
    """Tests for namespace operations."""

    @pytest.mark.asyncio
    async def test_create_namespace(self) -> None:
        """create_namespace delegates to engine."""
        lake = _make_lake(connected=True)

        mock_ns = MagicMock()
        lake._engine.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.create_namespace()
        assert result is mock_ns
        lake._engine.create_namespace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_namespace(self) -> None:
        """get_namespace delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        mock_ns = MagicMock()

        lake._engine.get_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.get_namespace(ns_id)
        assert result is mock_ns

    @pytest.mark.asyncio
    async def test_get_namespace_by_stable_id(self) -> None:
        """get_namespace_by_stable_id resolves stable id then delegates to engine."""
        lake = _make_lake(connected=True)
        stable_id = uuid4()
        mock_ns = MagicMock()

        lake._engine.get_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.get_namespace_by_stable_id(stable_id)
        assert result is mock_ns
        # Should have resolved the stable id first
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(stable_id)
        # Should pass the resolved row-level id to get_namespace
        lake._engine.get_namespace.assert_awaited_once_with(_RESOLVE_ROW_ID)

    @pytest.mark.asyncio
    async def test_get_namespace_by_stable_id_not_found(self) -> None:
        """get_namespace_by_stable_id raises ValueError when no active version exists."""
        lake = _make_lake(connected=True)
        stable_id = uuid4()
        lake._engine._storage.resolve_namespace = AsyncMock(
            side_effect=ValueError(f"No active namespace version found for namespace_id={stable_id}")
        )

        with pytest.raises(ValueError, match="No active namespace version"):
            await lake.get_namespace_by_stable_id(stable_id)

    @pytest.mark.asyncio
    async def test_get_namespace_by_stable_id_resolved_but_none(self) -> None:
        """get_namespace_by_stable_id returns None when resolved namespace not in engine."""
        lake = _make_lake(connected=True)
        stable_id = uuid4()

        lake._engine.get_namespace = AsyncMock(return_value=None)

        result = await lake.get_namespace_by_stable_id(stable_id)
        assert result is None
        lake._engine._storage.resolve_namespace.assert_awaited_once_with(stable_id)
        lake._engine.get_namespace.assert_awaited_once_with(_RESOLVE_ROW_ID)

    @pytest.mark.asyncio
    async def test_create_namespace_returns_namespace_id(self) -> None:
        """create_namespace returns object with distinct namespace_id."""
        from khora.core.models.tenancy import MemoryNamespace

        lake = _make_lake(connected=True)
        row_id = uuid4()
        stable_id = uuid4()
        mock_ns = MemoryNamespace(id=row_id, namespace_id=stable_id)
        lake._engine.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.create_namespace()
        assert result.namespace_id == stable_id
        assert result.id == row_id
        assert result.namespace_id != result.id  # namespace_id is independently generated


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health_check."""

    @pytest.mark.asyncio
    async def test_disconnected(self) -> None:
        """Health check when disconnected."""
        lake = _make_lake()
        result = await lake.health_check()
        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_healthy(self) -> None:
        """Health check delegates to engine."""
        lake = _make_lake(connected=True)
        lake._engine.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "storage": {"relational": True, "vector": True},
            }
        )

        result = await lake.health_check()
        assert result["status"] == "healthy"


# ---------------------------------------------------------------------------
# New API: Simplified Constructor
# ---------------------------------------------------------------------------


class TestSimplifiedConstructor:
    """Tests for the simplified MemoryLake constructor."""

    def test_init_with_database_url_string(self) -> None:
        """Init with database URL string creates config."""
        with patch("khora.memory_lake.load_config") as mock_load:
            lake = MemoryLake("postgresql://localhost/mydb")
            mock_load.assert_not_called()

        assert lake._config.database_url == "postgresql://localhost/mydb"

    def test_init_with_database_url_and_graph_url(self) -> None:
        """Init with both database and graph URLs."""
        with patch("khora.memory_lake.load_config"):
            lake = MemoryLake(
                "postgresql://localhost/mydb",
                graph_url="bolt://localhost:7687",
            )

        assert lake._config.database_url == "postgresql://localhost/mydb"
        assert lake._config.neo4j_url == "bolt://localhost:7687"

    def test_init_with_custom_embedding_model(self) -> None:
        """Init with custom embedding model."""
        with patch("khora.memory_lake.load_config"):
            lake = MemoryLake(
                "postgresql://localhost/mydb",
                embedding_model="text-embedding-3-large",
            )

        assert lake._config.llm.embedding_model == "text-embedding-3-large"

    def test_init_with_khora_config(self) -> None:
        """Init with full KhoraConfig object."""
        from khora.config import KhoraConfig

        # Create a real KhoraConfig (not a mock) to trigger the isinstance check
        cfg = KhoraConfig(database_url="postgresql://test")
        lake = MemoryLake(cfg)

        assert lake._config is cfg
        assert lake._config.database_url == "postgresql://test"

    def test_init_with_none_loads_from_env(self) -> None:
        """Init with None loads config from env/file."""
        with patch("khora.memory_lake.load_config", return_value=_mock_config()) as mock_load:
            lake = MemoryLake()
            mock_load.assert_called_once()

        assert lake._config is not None

    def test_init_none_with_graph_override(self) -> None:
        """Init with None but graph_url override."""
        mock_cfg = _mock_config()
        mock_cfg.neo4j_url = None
        with patch("khora.memory_lake.load_config", return_value=mock_cfg):
            lake = MemoryLake(graph_url="bolt://custom:7687")

        assert lake._config.neo4j_url == "bolt://custom:7687"

    def test_init_with_engine_parameter(self) -> None:
        """Init with explicit engine parameter."""
        with patch("khora.memory_lake.load_config", return_value=_mock_config()):
            lake = MemoryLake(engine="graphrag")

        assert lake._engine_name == "graphrag"


# ---------------------------------------------------------------------------
# New API: BatchResult and Stats dataclasses
# ---------------------------------------------------------------------------


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        r = BatchResult(
            total=10,
            processed=8,
            skipped=1,
            failed=1,
            chunks=50,
            entities=20,
            relationships=15,
        )
        assert r.total == 10
        assert r.processed == 8
        assert r.skipped == 1
        assert r.failed == 1
        assert r.chunks == 50
        assert r.entities == 20
        assert r.relationships == 15


class TestStats:
    """Tests for Stats dataclass."""

    def test_fields(self) -> None:
        """All fields are accessible."""
        s = Stats(
            documents=100,
            chunks=500,
            entities=200,
            relationships=150,
        )
        assert s.documents == 100
        assert s.chunks == 500
        assert s.entities == 200
        assert s.relationships == 150

    def test_last_activity_at_default_none(self) -> None:
        """last_activity_at defaults to None for backward compatibility."""
        s = Stats(documents=1, chunks=2, entities=3, relationships=4)
        assert s.last_activity_at is None

    def test_last_activity_at_with_value(self) -> None:
        """last_activity_at accepts a datetime value."""
        from datetime import UTC, datetime

        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        s = Stats(
            documents=1,
            chunks=2,
            entities=3,
            relationships=4,
            last_activity_at=ts,
        )
        assert s.last_activity_at == ts

    def test_frozen(self) -> None:
        """Stats is immutable."""
        s = Stats(documents=1, chunks=2, entities=3, relationships=4)
        with pytest.raises(AttributeError):
            s.last_activity_at = datetime.now()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# New API: Storage Property (stable API)
# ---------------------------------------------------------------------------


class TestStorageProperty:
    """Tests for the storage property (promoted to stable API)."""

    def test_storage_no_deprecation_warning(self) -> None:
        """Accessing storage property does NOT emit DeprecationWarning."""
        lake = _make_lake(connected=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = lake.storage
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0

    def test_storage_returns_coordinator(self) -> None:
        """storage property returns the engine's storage coordinator."""
        lake = _make_lake(connected=True)
        assert lake.storage is lake._engine._storage


# ---------------------------------------------------------------------------
# New API: Raw flag in recall
# ---------------------------------------------------------------------------


class TestRecallRawMode:
    """Tests for raw mode in recall()."""

    @pytest.mark.asyncio
    async def test_raw_mode_passed_to_engine(self) -> None:
        """raw=True is passed to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test query", namespace=ns_id, raw=True)

        call_kwargs = lake._engine.recall.call_args
        assert call_kwargs.kwargs.get("raw") is True


# ---------------------------------------------------------------------------
# New API: Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Tests for convenience methods (get_document, list_documents, etc.)."""

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        """get_document delegates to engine."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        mock_doc = MagicMock()

        lake._engine.get_document = AsyncMock(return_value=mock_doc)

        result = await lake.get_document(doc_id)
        assert result is mock_doc
        lake._engine.get_document.assert_awaited_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_list_documents(self) -> None:
        """list_documents delegates to engine with resolved namespace."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_docs = [MagicMock(), MagicMock()]
        lake._engine.list_documents = AsyncMock(return_value=mock_docs)

        result = await lake.list_documents(namespace=ns_id, limit=50)
        assert result == mock_docs
        lake._engine.list_documents.assert_awaited_once_with(_RESOLVE_ROW_ID, limit=50)

    @pytest.mark.asyncio
    async def test_search_entities(self) -> None:
        """search_entities delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_entities = [MagicMock()]
        lake._engine.search_entities = AsyncMock(return_value=mock_entities)

        result = await lake.search_entities("test query", namespace=ns_id, limit=5)

        assert len(result) == 1
        lake._engine.search_entities.assert_awaited_once()


# ---------------------------------------------------------------------------
# New API: Enhanced remember_batch
# ---------------------------------------------------------------------------


class TestEnhancedRememberBatch:
    """Tests for enhanced remember_batch() with BatchResult."""

    @pytest.mark.asyncio
    async def test_empty_batch_returns_batch_result(self) -> None:
        """Empty batch returns BatchResult with zeros."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=0,
                processed=0,
                skipped=0,
                failed=0,
                chunks=0,
                entities=0,
                relationships=0,
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(
                [],
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert isinstance(result, BatchResult)
        assert result.total == 0
        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_batch_returns_batch_result(self) -> None:
        """remember_batch() returns BatchResult with aggregated stats."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.remember_batch = AsyncMock(
            return_value=BatchResult(
                total=3,
                processed=2,
                skipped=1,
                failed=0,
                chunks=10,
                entities=5,
                relationships=5,
            )
        )

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch(
                [
                    {"content": "Doc 1"},
                    {"content": "Doc 2"},
                    {"content": "Doc 3"},
                ],
                namespace=ns_id,
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.processed == 2
        assert result.skipped == 1
        assert result.relationships == 5


# ---------------------------------------------------------------------------
# Engine Registry Tests
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    """Tests for engine registry functions."""

    def test_list_engines(self) -> None:
        """list_engines returns available engines."""
        from khora.engines import list_engines

        engines = list_engines()
        assert "graphrag" in engines

    def test_register_engine(self) -> None:
        """register_engine adds new engine to registry."""
        from khora.engines import list_engines, register_engine

        register_engine("test_engine", "test.module", "TestEngine")
        engines = list_engines()
        assert "test_engine" in engines

    def test_create_engine_unknown_raises(self) -> None:
        """create_engine raises for unknown engine."""
        from khora.engines import create_engine

        with pytest.raises(ValueError, match="Unknown engine"):
            create_engine("nonexistent", _mock_config())


# ---------------------------------------------------------------------------
# include_sources feature (DYT-506)
# ---------------------------------------------------------------------------


class TestIncludeSources:
    """Tests for include_sources parameter on read methods."""

    @pytest.mark.asyncio
    async def test_recall_include_sources_false(self) -> None:
        """Default include_sources=False does not call get_document_sources_batch."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id)

        assert isinstance(result, RecallResult)
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_include_sources_true(self) -> None:
        """include_sources=True populates source_document on chunks and source_documents on entities."""
        from khora.core.models.document import Chunk, DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id_1 = uuid4()
        doc_id_2 = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id_1, content="hello")
        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id_1, doc_id_2],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[(entity, 0.8)],
            context_text="hello",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        src_1 = DocumentSource(id=doc_id_1, title="Doc 1")
        src_2 = DocumentSource(id=doc_id_2, title="Doc 2")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id_1: src_1, doc_id_2: src_2})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        # Chunk should have source_document populated
        assert result.chunks[0][0].source_document is src_1

        # Entity should have source_documents populated
        assert result.entities[0][0].source_documents == {doc_id_1: src_1, doc_id_2: src_2}

        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_entities_include_sources(self) -> None:
        """list_entities with include_sources=True populates source_documents on entities."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Bob",
            entity_type="PERSON",
            source_document_ids=[doc_id],
        )
        lake._engine.list_entities = AsyncMock(return_value=[entity])

        src = DocumentSource(id=doc_id, title="Source Doc")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.list_entities(namespace=ns_id, include_sources=True)

        assert len(result) == 1
        assert result[0].source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_entities_include_sources(self) -> None:
        """search_entities with include_sources=True populates source_documents."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Acme Corp",
            entity_type="ORGANIZATION",
            source_document_ids=[doc_id],
        )
        lake._engine.search_entities = AsyncMock(return_value=[entity])

        src = DocumentSource(id=doc_id, title="Report")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.search_entities("acme", namespace=ns_id, include_sources=True)

        assert len(result) == 1
        assert result[0].source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_find_related_entities_include_sources(self) -> None:
        """find_related_entities with include_sources=True populates source_documents."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        entity_id = uuid4()
        doc_id = uuid4()

        related = Entity(
            namespace_id=ns_id,
            name="Related Entity",
            entity_type="CONCEPT",
            source_document_ids=[doc_id],
        )
        lake._engine.find_related_entities = AsyncMock(return_value=[(related, 0.75)])

        src = DocumentSource(id=doc_id, title="Origin")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.find_related_entities(entity_id, namespace=ns_id, include_sources=True)

        assert len(result) == 1
        assert result[0][0].source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_include_sources_empty_results(self) -> None:
        """Empty chunks/entities with include_sources=True does not crash or fetch."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()

        mock_result = RecallResult(
            query="nothing",
            namespace_id=ns_id,
            chunks=[],
            entities=[],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("nothing", namespace=ns_id, include_sources=True)

        assert result.chunks == []
        assert result.entities == []
        # No doc IDs to fetch, so get_document_sources_batch should not be called
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_entity_include_sources(self) -> None:
        """get_entity with include_sources=True populates source_documents."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id],
        )
        lake._engine.get_entity = AsyncMock(return_value=entity)

        src = DocumentSource(id=doc_id, title="Source Doc")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id: src})

        result = await lake.get_entity(entity.id, include_sources=True)

        assert result is not None
        assert result.source_documents == {doc_id: src}
        lake._engine._storage.get_document_sources_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_entity_include_sources_false(self) -> None:
        """Default include_sources=False does not call get_document_sources_batch."""
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Bob",
            entity_type="PERSON",
        )
        lake._engine.get_entity = AsyncMock(return_value=entity)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        result = await lake.get_entity(entity.id)

        assert result is not None
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_entity_include_sources_not_found(self) -> None:
        """get_entity returns None when entity not found, even with include_sources=True."""
        lake = _make_lake(connected=True)
        lake._engine.get_entity = AsyncMock(return_value=None)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        result = await lake.get_entity(uuid4(), include_sources=True)

        assert result is None
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deleted_document_skipped_on_entities(self) -> None:
        """Entity with partially-deleted source docs only gets found sources."""
        from khora.core.models.document import DocumentSource
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id_1 = uuid4()
        doc_id_2 = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            source_document_ids=[doc_id_1, doc_id_2],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[(entity, 0.8)],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        # Only doc_id_1 is returned; doc_id_2 was deleted
        src_1 = DocumentSource(id=doc_id_1, title="Doc 1")
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={doc_id_1: src_1})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        assert result.entities[0][0].source_documents == {doc_id_1: src_1}
        assert doc_id_2 not in result.entities[0][0].source_documents

    @pytest.mark.asyncio
    async def test_chunk_with_missing_document(self) -> None:
        """Chunk whose document_id is not in sources gets source_document=None."""
        from khora.core.models.document import Chunk

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id, content="orphan chunk")

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[],
            context_text="orphan chunk",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)

        # get_document_sources_batch returns empty dict (document was deleted)
        lake._engine._storage.get_document_sources_batch = AsyncMock(return_value={})

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        assert result.chunks[0][0].source_document is None

    @pytest.mark.asyncio
    async def test_storage_exception_propagation(self) -> None:
        """RuntimeError from get_document_sources_batch propagates to caller."""
        from khora.core.models.document import Chunk

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(namespace_id=ns_id, document_id=doc_id, content="test")

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[(chunk, 0.9)],
            entities=[],
            context_text="test",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock(side_effect=RuntimeError("DB error"))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            with pytest.raises(RuntimeError, match="DB error"):
                await lake.recall("test", namespace=ns_id, include_sources=True)

    @pytest.mark.asyncio
    async def test_entity_empty_source_document_ids(self) -> None:
        """Entity with empty source_document_ids skips fetch and gets source_documents=None."""
        from khora.core.models.entity import Entity

        lake = _make_lake(connected=True)
        ns_id = uuid4()

        entity = Entity(
            namespace_id=ns_id,
            name="Lonely",
            entity_type="CONCEPT",
            source_document_ids=[],
        )

        mock_result = RecallResult(
            query="test",
            namespace_id=ns_id,
            chunks=[],
            entities=[(entity, 0.7)],
            context_text="",
        )
        lake._engine.recall = AsyncMock(return_value=mock_result)
        lake._engine._storage.get_document_sources_batch = AsyncMock()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("test", namespace=ns_id, include_sources=True)

        # No doc IDs to fetch, so get_document_sources_batch should NOT be called
        lake._engine._storage.get_document_sources_batch.assert_not_awaited()
        assert result.entities[0][0].source_documents is None

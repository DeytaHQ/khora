"""Unit tests for memory_lake.py — MemoryLake primary API."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.memory_lake import BatchResult, MemoryLake, RecallResult, RememberResult, Stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config() -> MagicMock:
    """Create a mock KhoraConfig with all required methods."""
    mock_config = MagicMock()
    mock_config.get_postgresql_url.return_value = "postgresql://test"
    mock_config.get_graph_config.return_value = None
    mock_config.get_vector_config.return_value = None
    mock_config.get_neo4j_url.return_value = None
    mock_config.get_neo4j_user.return_value = None
    mock_config.get_neo4j_password.return_value = None
    mock_config.get_neo4j_database.return_value = None
    mock_config.storage.embedding_dimension = 1536
    mock_config.llm.model = "gpt-4o-mini"
    mock_config.llm.embedding_model = "text-embedding-3-small"
    mock_config.llm.embedding_dimension = 1536
    mock_config.llm.extraction_model = None
    mock_config.llm.timeout = 30
    mock_config.llm.max_retries = 3
    mock_config.telemetry_database_url = None
    mock_config.telemetry_service_name = "khora-test"
    return mock_config


def _mock_engine() -> MagicMock:
    """Create a mock engine with all required methods."""
    mock_eng = MagicMock()

    # Storage and embedder
    mock_eng._storage = MagicMock()
    mock_eng._embedder = MagicMock()

    # Default namespace ID
    mock_eng._default_namespace_id = None

    # Lifecycle
    mock_eng.connect = AsyncMock()
    mock_eng.disconnect = AsyncMock()
    mock_eng.health_check = AsyncMock(return_value={"status": "healthy"})

    # Core operations
    mock_eng.remember = AsyncMock()
    mock_eng.recall = AsyncMock()
    mock_eng.forget = AsyncMock()
    mock_eng.remember_batch = AsyncMock()

    # Namespace operations
    mock_eng.get_or_create_default_namespace = AsyncMock(return_value=uuid4())
    mock_eng.create_namespace = AsyncMock()
    mock_eng.get_namespace = AsyncMock()
    mock_eng.ensure_namespace = AsyncMock()

    # Entity operations
    mock_eng.get_entity = AsyncMock()
    mock_eng.list_entities = AsyncMock(return_value=[])
    mock_eng.find_related_entities = AsyncMock(return_value=[])

    # Document operations
    mock_eng.get_document = AsyncMock()
    mock_eng.list_documents = AsyncMock(return_value=[])
    mock_eng.search_entities = AsyncMock(return_value=[])

    # Stats
    mock_eng.stats = AsyncMock()

    return mock_eng


def _make_lake(*, connected: bool = False) -> MemoryLake:
    """Create a MemoryLake with mocked config, optionally pre-connected."""
    with patch("khora.memory_lake.load_config", return_value=_mock_config()):
        lake = MemoryLake()

    if connected:
        lake._connected = True
        lake._engine = _mock_engine()

    return lake


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
        ns_id = uuid4()
        r = RecallResult(
            query="test query",
            namespace_id=ns_id,
            chunks=[("chunk1", 0.9)],
            entities=[("entity1", 0.8)],
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
    """Tests for _resolve_namespace helper."""

    @pytest.mark.asyncio
    async def test_uuid_passthrough(self) -> None:
        """UUID passes through directly."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        result = await lake._resolve_namespace(ns_id)
        assert result == ns_id

    @pytest.mark.asyncio
    async def test_uuid_string_passthrough(self) -> None:
        """UUID string is parsed and returned."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        result = await lake._resolve_namespace(str(ns_id))
        assert result == ns_id

    @pytest.mark.asyncio
    async def test_none_calls_get_or_create_default(self) -> None:
        """None resolves via get_or_create_default_namespace on engine."""
        lake = _make_lake(connected=True)
        default_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=default_id)

        result = await lake._resolve_namespace(None)
        assert result == default_id

    @pytest.mark.asyncio
    async def test_slug_lookup(self) -> None:
        """Non-UUID string looks up namespace by slug (globally unique)."""
        lake = _make_lake(connected=True)

        found_ns = MagicMock()
        found_ns.id = uuid4()

        lake._engine._storage.get_namespace_by_slug = AsyncMock(return_value=found_ns)

        result = await lake._resolve_namespace("my-namespace")
        assert result == found_ns.id
        lake._engine._storage.get_namespace_by_slug.assert_awaited_once_with("my-namespace")

    @pytest.mark.asyncio
    async def test_slug_not_found_raises(self) -> None:
        """Non-UUID string that doesn't exist raises ValueError."""
        lake = _make_lake(connected=True)

        lake._engine._storage.get_namespace_by_slug = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Namespace not found"):
            await lake._resolve_namespace("nonexistent")


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
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

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
            result = await lake.remember("test content", title="Test")

        assert result is mock_result
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
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

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
            result = await lake.recall("search query")

        assert isinstance(result, RecallResult)
        assert result.query == "search query"
        lake._engine.recall.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recall_passes_search_mode(self) -> None:
        """recall() passes mode to engine."""
        from khora.query.engine import SearchMode

        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

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
            await lake.recall("test", mode=SearchMode.VECTOR)

        call_kwargs = lake._engine.recall.call_args
        assert call_kwargs.kwargs.get("mode") == SearchMode.VECTOR


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    """Tests for forget()."""

    @pytest.mark.asyncio
    async def test_forget_delegates_to_engine(self) -> None:
        """forget() delegates to engine.forget()."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()

        lake._engine.forget = AsyncMock(return_value=True)

        result = await lake.forget(doc_id)
        assert result is True
        lake._engine.forget.assert_awaited_once_with(doc_id, None)

    @pytest.mark.asyncio
    async def test_forget_with_namespace(self) -> None:
        """forget() resolves namespace and passes to engine."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        ns_id = uuid4()

        lake._engine.forget = AsyncMock(return_value=False)

        result = await lake.forget(doc_id, namespace=ns_id)
        assert result is False
        lake._engine.forget.assert_awaited_once_with(doc_id, ns_id)


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
        """list_entities delegates to engine with filters."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        mock_entities = [MagicMock(), MagicMock()]
        lake._engine.list_entities = AsyncMock(return_value=mock_entities)

        result = await lake.list_entities(entity_type="PERSON", limit=50)
        assert result == mock_entities
        lake._engine.list_entities.assert_awaited_once_with(ns_id, entity_type="PERSON", limit=50)

    @pytest.mark.asyncio
    async def test_find_related_entities(self) -> None:
        """find_related_entities delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
        entity_id = uuid4()

        mock_related = [(MagicMock(), 0.8)]
        lake._engine.find_related_entities = AsyncMock(return_value=mock_related)

        result = await lake.find_related_entities(entity_id, max_depth=3)
        assert result == mock_related


# ---------------------------------------------------------------------------
# Namespace management
# ---------------------------------------------------------------------------


class TestNamespaceManagement:
    """Tests for namespace operations."""

    @pytest.mark.asyncio
    async def test_create_namespace(self) -> None:
        """create_namespace delegates to engine without workspace_id."""
        lake = _make_lake(connected=True)

        mock_ns = MagicMock()
        lake._engine.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.create_namespace("test-ns", description="Test")
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
    async def test_get_or_create_default_namespace(self) -> None:
        """get_or_create_default_namespace delegates to engine."""
        lake = _make_lake(connected=True)
        default_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=default_id)

        result = await lake.get_or_create_default_namespace()
        assert result == default_id


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
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

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
            await lake.recall("test query", raw=True)

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
        """list_documents delegates to engine with namespace."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        mock_docs = [MagicMock(), MagicMock()]
        lake._engine.list_documents = AsyncMock(return_value=mock_docs)

        result = await lake.list_documents(limit=50)
        assert result == mock_docs
        lake._engine.list_documents.assert_awaited_once_with(ns_id, limit=50)

    @pytest.mark.asyncio
    async def test_search_entities(self) -> None:
        """search_entities delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)

        mock_entities = [MagicMock()]
        lake._engine.search_entities = AsyncMock(return_value=mock_entities)

        result = await lake.search_entities("test query", limit=5)

        assert len(result) == 1
        lake._engine.search_entities.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ensure_namespace(self) -> None:
        """ensure_namespace delegates to engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.ensure_namespace = AsyncMock(return_value=ns_id)

        result = await lake.ensure_namespace("my-namespace", description="Test")
        assert result == ns_id
        lake._engine.ensure_namespace.assert_awaited_once()


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
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
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
            result = await lake.remember_batch([])

        assert isinstance(result, BatchResult)
        assert result.total == 0
        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_batch_returns_batch_result(self) -> None:
        """remember_batch() returns BatchResult with aggregated stats."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._engine.get_or_create_default_namespace = AsyncMock(return_value=ns_id)
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
                ]
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

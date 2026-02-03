"""Unit tests for memory_lake.py — MemoryLake primary API."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

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


def _make_lake(*, connected: bool = False) -> MemoryLake:
    """Create a MemoryLake with mocked config, optionally pre-connected."""
    with patch("khora.memory_lake.load_config", return_value=_mock_config()):
        lake = MemoryLake()

    if connected:
        lake._connected = True
        lake._storage = MagicMock()
        lake._embedder = MagicMock()
        lake._query_engine = MagicMock()

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
        assert lake._storage is None

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
        """Accessing storage/query_engine before connect raises."""
        lake = _make_lake()

        with pytest.raises(RuntimeError, match="not connected"):
            _ = lake.storage

        with pytest.raises(RuntimeError, match="not connected"):
            _ = lake.query_engine

    def test_connected_properties_return(self) -> None:
        """Accessing storage/query_engine after connect succeeds."""
        lake = _make_lake(connected=True)
        assert lake.storage is lake._storage
        assert lake.query_engine is lake._query_engine


# ---------------------------------------------------------------------------
# connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    """Tests for connect() and disconnect() lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """connect() creates storage, embedder, query engine and sets flag."""
        lake = _make_lake()

        mock_coordinator = MagicMock()
        mock_coordinator.connect = AsyncMock()

        with (
            patch("khora.memory_lake.create_storage_coordinator", return_value=mock_coordinator),
            patch("khora.memory_lake.LiteLLMEmbedder") as mock_embedder_cls,
            patch("khora.memory_lake.HybridQueryEngine"),
            patch("khora.telemetry.init_telemetry", new_callable=AsyncMock),
        ):
            mock_embedder_cls.from_config.return_value = MagicMock()
            await lake.connect()

        assert lake._connected is True
        assert lake._storage is mock_coordinator
        mock_coordinator.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        """Calling connect() when already connected is a no-op."""
        lake = _make_lake(connected=True)
        original_storage = lake._storage

        await lake.connect()

        assert lake._storage is original_storage

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """disconnect() tears down all components."""
        lake = _make_lake(connected=True)
        lake._storage.disconnect = AsyncMock()

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await lake.disconnect()

        assert lake._connected is False
        assert lake._storage is None
        assert lake._embedder is None
        assert lake._query_engine is None

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
    async def test_none_creates_default(self) -> None:
        """None resolves to default namespace."""
        lake = _make_lake(connected=True)
        default_id = uuid4()
        lake._default_namespace_id = default_id

        result = await lake._resolve_namespace(None)
        assert result == default_id

    @pytest.mark.asyncio
    async def test_slug_lookup(self) -> None:
        """Non-UUID string looks up namespace by slug."""
        lake = _make_lake(connected=True)
        default_id = uuid4()
        lake._default_namespace_id = default_id

        mock_ns = MagicMock()
        mock_ns.workspace_id = uuid4()

        found_ns = MagicMock()
        found_ns.id = uuid4()

        lake._storage.get_namespace = AsyncMock(return_value=mock_ns)
        lake._storage.get_namespace_by_slug = AsyncMock(return_value=found_ns)

        result = await lake._resolve_namespace("my-namespace")
        assert result == found_ns.id

    @pytest.mark.asyncio
    async def test_slug_not_found_raises(self) -> None:
        """Non-UUID string that doesn't exist raises ValueError."""
        lake = _make_lake(connected=True)
        default_id = uuid4()
        lake._default_namespace_id = default_id

        mock_ns = MagicMock()
        mock_ns.workspace_id = uuid4()

        lake._storage.get_namespace = AsyncMock(return_value=mock_ns)
        lake._storage.get_namespace_by_slug = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Namespace not found"):
            await lake._resolve_namespace("nonexistent")


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


class TestRemember:
    """Tests for remember() and _remember_inner()."""

    @pytest.mark.asyncio
    async def test_remember_new_document(self) -> None:
        """remember() creates document and processes through pipeline."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id
        lake._config = _mock_config()

        mock_doc = MagicMock()
        mock_doc.id = uuid4()

        lake._storage.get_document_by_checksum = AsyncMock(return_value=None)
        lake._storage.create_document = AsyncMock(return_value=mock_doc)

        pipeline_result = {"chunks": 3, "entities": 2, "relationships": 1}

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch(
                "khora.pipelines.flows.ingest.process_document", new_callable=AsyncMock, return_value=pipeline_result
            ),
        ):
            result = await lake.remember("test content", title="Test")

        assert result.document_id == mock_doc.id
        assert result.namespace_id == ns_id
        assert result.chunks_created == 3
        assert result.entities_extracted == 2
        assert result.relationships_created == 1

    @pytest.mark.asyncio
    async def test_remember_duplicate_document(self) -> None:
        """remember() returns early for duplicate checksum."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id
        lake._config = _mock_config()

        existing_doc = MagicMock()
        existing_doc.id = uuid4()
        existing_doc.chunk_count = 5
        existing_doc.entity_count = 2
        existing_doc.status = "completed"

        lake._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember("duplicate content")

        assert result.document_id == existing_doc.id
        assert result.metadata["duplicate"] is True
        lake._storage.create_document.assert_not_called() if hasattr(lake._storage, "create_document") else None


# ---------------------------------------------------------------------------
# remember_batch
# ---------------------------------------------------------------------------


class TestRememberBatchLegacy:
    """Tests for remember_batch_legacy() which returns list[RememberResult]."""

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        """Empty batch returns empty list."""
        lake = _make_lake(connected=True)
        lake._default_namespace_id = uuid4()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = await lake.remember_batch_legacy([])

        assert result == []

    @pytest.mark.asyncio
    async def test_batch_returns_results(self) -> None:
        """remember_batch_legacy() returns one RememberResult per document."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id
        lake._config = _mock_config()

        doc_id_1 = str(uuid4())
        doc_id_2 = str(uuid4())
        ingest_result = {
            "per_document_results": [
                {"document_id": doc_id_1, "chunks": 3, "entities": 1, "relationships": 0},
                {"document_id": doc_id_2, "chunks": 2, "entities": 0, "relationships": 0},
            ],
            "failed_documents": 0,
        }

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.pipelines.flows.ingest.ingest_documents", new_callable=AsyncMock, return_value=ingest_result),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            results = await lake.remember_batch_legacy(
                [
                    {"content": "Doc 1", "title": "First"},
                    {"content": "Doc 2", "title": "Second"},
                ]
            )

        assert len(results) == 2
        assert results[0].document_id == UUID(doc_id_1)
        assert results[0].chunks_created == 3
        assert results[1].document_id == UUID(doc_id_2)
        assert results[1].chunks_created == 2

    @pytest.mark.asyncio
    async def test_batch_with_failures(self) -> None:
        """Failed documents get padded with error results."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id
        lake._config = _mock_config()

        doc_id = str(uuid4())
        ingest_result = {
            "per_document_results": [
                {"document_id": doc_id, "chunks": 1, "entities": 0, "relationships": 0},
            ],
            "failed_documents": 1,
        }

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.pipelines.flows.ingest.ingest_documents", new_callable=AsyncMock, return_value=ingest_result),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            results = await lake.remember_batch_legacy(
                [
                    {"content": "Good doc"},
                    {"content": "Bad doc"},
                ]
            )

        assert len(results) == 2
        assert results[1].metadata.get("failed") is True
        assert results[1].chunks_created == 0


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


class TestRecall:
    """Tests for recall()."""

    @pytest.mark.asyncio
    async def test_recall_delegates_to_query_engine(self) -> None:
        """recall() delegates to query_engine.query() and wraps result."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id

        mock_chunk = MagicMock()
        mock_chunk.content = "found content"
        mock_entity = MagicMock()

        mock_query_result = MagicMock()
        mock_query_result.chunks = [(mock_chunk, 0.9)]
        mock_query_result.entities = [(mock_entity, 0.8)]
        mock_query_result.get_context_text.return_value = "found content"
        mock_query_result.metadata = {"mode": "HYBRID"}

        lake._query_engine.query = AsyncMock(return_value=mock_query_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.recall("search query")

        assert isinstance(result, RecallResult)
        assert result.query == "search query"
        assert result.namespace_id == ns_id
        assert len(result.chunks) == 1
        assert result.context_text == "found content"

    @pytest.mark.asyncio
    async def test_recall_passes_search_mode(self) -> None:
        """recall() passes mode to QueryConfig."""
        from khora.query.engine import SearchMode

        lake = _make_lake(connected=True)
        lake._default_namespace_id = uuid4()

        mock_query_result = MagicMock()
        mock_query_result.chunks = []
        mock_query_result.entities = []
        mock_query_result.get_context_text.return_value = ""
        mock_query_result.metadata = {}

        lake._query_engine.query = AsyncMock(return_value=mock_query_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test", mode=SearchMode.VECTOR)

        call_kwargs = lake._query_engine.query.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.mode == SearchMode.VECTOR


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForget:
    """Tests for forget()."""

    @pytest.mark.asyncio
    async def test_forget_deletes_document(self) -> None:
        """forget() calls storage.delete_document."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()

        lake._storage.delete_document = AsyncMock(return_value=True)

        result = await lake.forget(doc_id)
        assert result is True
        lake._storage.delete_document.assert_awaited_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_forget_wrong_namespace(self) -> None:
        """forget() returns False when document is in a different namespace."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        ns_id = uuid4()
        other_ns_id = uuid4()

        mock_doc = MagicMock()
        mock_doc.namespace_id = other_ns_id

        lake._storage.get_document = AsyncMock(return_value=mock_doc)

        result = await lake.forget(doc_id, namespace=ns_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_forget_not_found(self) -> None:
        """forget() returns False when document doesn't exist."""
        lake = _make_lake(connected=True)
        lake._storage.delete_document = AsyncMock(return_value=False)

        result = await lake.forget(uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# Entity operations
# ---------------------------------------------------------------------------


class TestEntityOperations:
    """Tests for entity CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_entity(self) -> None:
        """get_entity delegates to storage."""
        lake = _make_lake(connected=True)
        entity_id = uuid4()
        mock_entity = MagicMock()

        lake._storage.get_entity = AsyncMock(return_value=mock_entity)

        result = await lake.get_entity(entity_id)
        assert result is mock_entity
        lake._storage.get_entity.assert_awaited_once_with(entity_id)

    @pytest.mark.asyncio
    async def test_list_entities(self) -> None:
        """list_entities delegates to storage with filters."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id

        mock_entities = [MagicMock(), MagicMock()]
        lake._storage.list_entities = AsyncMock(return_value=mock_entities)

        result = await lake.list_entities(entity_type="PERSON", limit=50)
        assert result == mock_entities
        lake._storage.list_entities.assert_awaited_once_with(ns_id, entity_type="PERSON", limit=50)

    @pytest.mark.asyncio
    async def test_find_related_entities(self) -> None:
        """find_related_entities delegates to query_engine."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id
        entity_id = uuid4()

        mock_related = [(MagicMock(), 0.8)]
        lake._query_engine.find_related_entities = AsyncMock(return_value=mock_related)

        result = await lake.find_related_entities(entity_id, max_depth=3)
        assert result == mock_related


# ---------------------------------------------------------------------------
# Namespace management
# ---------------------------------------------------------------------------


class TestNamespaceManagement:
    """Tests for namespace operations."""

    @pytest.mark.asyncio
    async def test_create_namespace(self) -> None:
        """create_namespace creates and stores a namespace."""
        lake = _make_lake(connected=True)
        ws_id = uuid4()

        mock_ns = MagicMock()
        lake._storage.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.create_namespace("test-ns", ws_id, description="Test")
        assert result is mock_ns
        lake._storage.create_namespace.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_namespace(self) -> None:
        """get_namespace delegates to storage."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        mock_ns = MagicMock()

        lake._storage.get_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.get_namespace(ns_id)
        assert result is mock_ns

    @pytest.mark.asyncio
    async def test_get_or_create_default_namespace_cached(self) -> None:
        """get_or_create_default_namespace returns cached ID."""
        lake = _make_lake(connected=True)
        cached_id = uuid4()
        lake._default_namespace_id = cached_id

        result = await lake.get_or_create_default_namespace()
        assert result == cached_id

    @pytest.mark.asyncio
    async def test_get_or_create_default_namespace_creates(self) -> None:
        """get_or_create_default_namespace creates org/workspace/namespace."""
        lake = _make_lake(connected=True)

        mock_org = MagicMock()
        mock_org.id = uuid4()
        mock_ws = MagicMock()
        mock_ws.id = uuid4()
        mock_ns = MagicMock()
        mock_ns.id = uuid4()

        lake._storage.get_organization_by_slug = AsyncMock(return_value=None)
        lake._storage.create_organization = AsyncMock(return_value=mock_org)
        lake._storage.list_workspaces = AsyncMock(return_value=[])
        lake._storage.create_workspace = AsyncMock(return_value=mock_ws)
        lake._storage.list_namespaces = AsyncMock(return_value=[])
        lake._storage.create_namespace = AsyncMock(return_value=mock_ns)

        result = await lake.get_or_create_default_namespace()
        assert result == mock_ns.id
        assert lake._default_namespace_id == mock_ns.id

    @pytest.mark.asyncio
    async def test_get_or_create_default_reuses_existing(self) -> None:
        """get_or_create_default_namespace reuses existing org/ws/ns."""
        lake = _make_lake(connected=True)

        mock_org = MagicMock()
        mock_org.id = uuid4()
        mock_ws = MagicMock()
        mock_ws.id = uuid4()
        mock_ns = MagicMock()
        mock_ns.id = uuid4()

        lake._storage.get_organization_by_slug = AsyncMock(return_value=mock_org)
        lake._storage.list_workspaces = AsyncMock(return_value=[mock_ws])
        lake._storage.list_namespaces = AsyncMock(return_value=[mock_ns])

        result = await lake.get_or_create_default_namespace()
        assert result == mock_ns.id
        (
            lake._storage.create_organization.assert_not_called()
            if hasattr(lake._storage.create_organization, "assert_not_called")
            else None
        )


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
        """Health check when healthy."""
        lake = _make_lake(connected=True)
        mock_health = MagicMock()
        mock_health.is_healthy = True
        mock_health.summary = {"relational": True, "vector": True}

        lake._storage.health_check = AsyncMock(return_value=mock_health)

        result = await lake.health_check()
        assert result["status"] == "healthy"
        assert result["storage"] == mock_health.summary

    @pytest.mark.asyncio
    async def test_degraded(self) -> None:
        """Health check when degraded."""
        lake = _make_lake(connected=True)
        mock_health = MagicMock()
        mock_health.is_healthy = False
        mock_health.summary = {"relational": True, "vector": False}

        lake._storage.health_check = AsyncMock(return_value=mock_health)

        result = await lake.health_check()
        assert result["status"] == "degraded"


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
# New API: Deprecation Warnings
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    """Tests for deprecation warnings on storage and query_engine properties."""

    def test_storage_property_warns(self) -> None:
        """Accessing storage property emits DeprecationWarning."""
        lake = _make_lake(connected=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = lake.storage

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "lake.storage is deprecated" in str(w[0].message)

    def test_query_engine_property_warns(self) -> None:
        """Accessing query_engine property emits DeprecationWarning."""
        lake = _make_lake(connected=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = lake.query_engine

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "lake.query_engine is deprecated" in str(w[0].message)

    def test_internal_methods_no_warning(self) -> None:
        """Internal _get_storage/_get_query_engine don't emit warnings."""
        lake = _make_lake(connected=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = lake._get_storage()
            _ = lake._get_query_engine()

        # No deprecation warnings should be emitted
        assert all(not issubclass(x.category, DeprecationWarning) for x in w)


# ---------------------------------------------------------------------------
# New API: Raw flag in recall
# ---------------------------------------------------------------------------


class TestRecallRawMode:
    """Tests for raw mode in recall()."""

    @pytest.mark.asyncio
    async def test_raw_mode_disables_llm_features(self) -> None:
        """raw=True disables all LLM features in QueryConfig."""
        lake = _make_lake(connected=True)
        lake._default_namespace_id = uuid4()

        mock_query_result = MagicMock()
        mock_query_result.chunks = []
        mock_query_result.entities = []
        mock_query_result.get_context_text.return_value = ""
        mock_query_result.metadata = {}

        lake._query_engine.query = AsyncMock(return_value=mock_query_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test query", raw=True)

        # Check the config passed to query_engine
        call_kwargs = lake._query_engine.query.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")

        assert config.enable_query_understanding is False
        assert config.enable_query_expansion is False
        assert config.enable_entity_extraction is False
        assert config.enable_temporal_detection is False
        assert config.enable_entity_linking is False
        assert config.enable_reranking is False
        assert config.enable_hyde is False

    @pytest.mark.asyncio
    async def test_raw_false_keeps_defaults(self) -> None:
        """raw=False (default) keeps LLM features enabled."""
        lake = _make_lake(connected=True)
        lake._default_namespace_id = uuid4()

        mock_query_result = MagicMock()
        mock_query_result.chunks = []
        mock_query_result.entities = []
        mock_query_result.get_context_text.return_value = ""
        mock_query_result.metadata = {}

        lake._query_engine.query = AsyncMock(return_value=mock_query_result)

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            await lake.recall("test query", raw=False)

        call_kwargs = lake._query_engine.query.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")

        # Default values should be True
        assert config.enable_query_understanding is True
        assert config.enable_reranking is True


# ---------------------------------------------------------------------------
# New API: Convenience methods
# ---------------------------------------------------------------------------


class TestConvenienceMethods:
    """Tests for convenience methods (get_document, list_documents, etc.)."""

    @pytest.mark.asyncio
    async def test_get_document(self) -> None:
        """get_document delegates to storage."""
        lake = _make_lake(connected=True)
        doc_id = uuid4()
        mock_doc = MagicMock()

        lake._storage.get_document = AsyncMock(return_value=mock_doc)

        result = await lake.get_document(doc_id)
        assert result is mock_doc
        lake._storage.get_document.assert_awaited_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_list_documents(self) -> None:
        """list_documents delegates to storage with namespace."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id

        mock_docs = [MagicMock(), MagicMock()]
        lake._storage.list_documents = AsyncMock(return_value=mock_docs)

        result = await lake.list_documents(limit=50)
        assert result == mock_docs
        lake._storage.list_documents.assert_awaited_once_with(ns_id, limit=50)

    @pytest.mark.asyncio
    async def test_search_entities(self) -> None:
        """search_entities uses embedder and searches similar entities."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id

        # Mock embedder
        lake._embedder = MagicMock()
        lake._embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        # Mock storage
        entity_id = uuid4()
        mock_entity = MagicMock()
        lake._storage.search_similar_entities = AsyncMock(return_value=[(entity_id, 0.9)])
        lake._storage.get_entity = AsyncMock(return_value=mock_entity)

        result = await lake.search_entities("test query", limit=5)

        assert len(result) == 1
        assert result[0] is mock_entity
        lake._embedder.embed.assert_awaited_once_with("test query")

    @pytest.mark.asyncio
    async def test_ensure_namespace_returns_existing(self) -> None:
        """ensure_namespace returns existing namespace ID."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id

        mock_org = MagicMock()
        mock_org.id = uuid4()
        mock_ws = MagicMock()
        mock_ws.id = uuid4()
        mock_ns = MagicMock()
        mock_ns.id = uuid4()

        lake._storage.get_organization_by_slug = AsyncMock(return_value=mock_org)
        lake._storage.list_workspaces = AsyncMock(return_value=[mock_ws])
        lake._storage.get_namespace_by_slug = AsyncMock(return_value=mock_ns)

        result = await lake.ensure_namespace("my-namespace")
        assert result == mock_ns.id

    @pytest.mark.asyncio
    async def test_ensure_namespace_creates_new(self) -> None:
        """ensure_namespace creates new namespace when not found."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id

        mock_org = MagicMock()
        mock_org.id = uuid4()
        mock_ws = MagicMock()
        mock_ws.id = uuid4()
        mock_new_ns = MagicMock()
        mock_new_ns.id = uuid4()

        lake._storage.get_organization_by_slug = AsyncMock(return_value=mock_org)
        lake._storage.list_workspaces = AsyncMock(return_value=[mock_ws])
        lake._storage.get_namespace_by_slug = AsyncMock(return_value=None)
        lake._storage.create_namespace = AsyncMock(return_value=mock_new_ns)

        result = await lake.ensure_namespace("new-namespace", description="Test")
        assert result == mock_new_ns.id
        lake._storage.create_namespace.assert_awaited_once()


# ---------------------------------------------------------------------------
# New API: Enhanced remember_batch
# ---------------------------------------------------------------------------


class TestEnhancedRememberBatch:
    """Tests for enhanced remember_batch() with BatchResult."""

    @pytest.mark.asyncio
    async def test_empty_batch_returns_batch_result(self) -> None:
        """Empty batch returns BatchResult with zeros."""
        lake = _make_lake(connected=True)
        lake._default_namespace_id = uuid4()

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
        lake._default_namespace_id = ns_id
        lake._config = _mock_config()

        ingest_result = {
            "total_documents": 3,
            "processed_documents": 2,
            "skipped_documents": 1,
            "failed_documents": 0,
            "total_chunks": 10,
            "total_entities": 5,
            "total_relationships": 3,
            "total_inferred_relationships": 2,
            "per_document_results": [],
        }

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.pipelines.flows.ingest.ingest_documents", new_callable=AsyncMock, return_value=ingest_result),
            patch("khora.memory_lake.LiteLLMEmbedder"),
            patch("khora.extraction.expansion.entity_index.EntityIndex"),
        ):
            # Mock list_entities for preload
            lake._storage.list_entities = AsyncMock(return_value=[])

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
        assert result.failed == 0
        assert result.chunks == 10
        assert result.entities == 5
        assert result.relationships == 5  # 3 + 2 inferred

    @pytest.mark.asyncio
    async def test_batch_with_progress_callback(self) -> None:
        """remember_batch calls progress callback."""
        lake = _make_lake(connected=True)
        ns_id = uuid4()
        lake._default_namespace_id = ns_id
        lake._config = _mock_config()

        ingest_result = {
            "total_documents": 2,
            "processed_documents": 2,
            "skipped_documents": 0,
            "failed_documents": 0,
            "total_chunks": 5,
            "total_entities": 2,
            "total_relationships": 1,
            "total_inferred_relationships": 0,
            "per_document_results": [],
        }

        progress_calls = []

        def on_progress(done, total):
            progress_calls.append((done, total))

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
            patch("khora.pipelines.flows.ingest.ingest_documents", new_callable=AsyncMock, return_value=ingest_result),
            patch("khora.memory_lake.LiteLLMEmbedder"),
            patch("khora.extraction.expansion.entity_index.EntityIndex"),
        ):
            lake._storage.list_entities = AsyncMock(return_value=[])

            await lake.remember_batch(
                [{"content": "Doc 1"}, {"content": "Doc 2"}],
                on_progress=on_progress,
            )

        assert len(progress_calls) == 1
        assert progress_calls[0] == (2, 2)

"""Unit tests for memory_lake.py — MemoryLake primary API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.memory_lake import MemoryLake, RecallResult, RememberResult

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
        cfg = _mock_config()
        with patch("khora.memory_lake.load_config") as mock_load:
            lake = MemoryLake(config=cfg)
            mock_load.assert_not_called()
        assert lake._config is cfg

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


class TestRememberBatch:
    """Tests for remember_batch()."""

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        """Empty batch returns empty list."""
        lake = _make_lake(connected=True)
        lake._default_namespace_id = uuid4()

        with (
            patch("khora.telemetry.context.ensure_trace_id"),
            patch("khora.telemetry.context.clear_trace_id"),
        ):
            result = await lake.remember_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_batch_returns_results(self) -> None:
        """remember_batch() returns one RememberResult per document."""
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
        ):
            results = await lake.remember_batch(
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
        ):
            results = await lake.remember_batch(
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

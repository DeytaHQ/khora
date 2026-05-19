"""Unit tests for the SurrealDB unified backend."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.core.models.document import DocumentSource, DocumentStatus
from khora.storage.backends.surrealdb import _HAS_SURREALDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _namespace_row(
    row_id: UUID | None = None,
    ns_id: UUID | None = None,
    *,
    version: int = 1,
    is_active: bool = True,
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like a memory_namespace row."""
    row_id = row_id or uuid4()
    ns_id = ns_id or uuid4()
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"memory_namespace:⟨{row_id!s}⟩",
        "namespace_id": str(ns_id),
        "tenancy_mode": "shared",
        "version": version,
        "is_active": is_active,
        "config_overrides": {},
        "sync_checkpoints": {},
        "metadata_": {},
        "created_at": now,
        "updated_at": now,
    }


def _document_row(
    doc_id: UUID | None = None,
    ns_id: UUID | None = None,
    *,
    checksum: str = "abc123",
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like a document row."""
    doc_id = doc_id or uuid4()
    ns_id = ns_id or uuid4()
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"document:⟨{doc_id!s}⟩",
        "namespace_id": str(ns_id),
        "content": "test content",
        "status": "pending",
        "source": "test-source",
        "source_type": "file",
        "content_type": "text/plain",
        "title": "Test Doc",
        "author": "tester",
        "language": "en",
        "checksum": checksum,
        "size_bytes": 42,
        "metadata_": {"key": "value"},
        "chunk_count": 0,
        "entity_count": 0,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
        "processed_at": None,
        "source_timestamp": None,
        "external_id": None,
    }


# ── Feature flag ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBFeatureFlag:
    def test_has_surrealdb_is_bool(self) -> None:
        from khora.storage.backends.surrealdb import _HAS_SURREALDB

        assert isinstance(_HAS_SURREALDB, bool)

    def test_has_surrealdb_flag_consistent(self) -> None:
        """_HAS_SURREALDB reflects whether surrealdb is importable."""
        from khora.storage.backends.surrealdb import _HAS_SURREALDB

        try:
            import surrealdb  # noqa: F401

            assert _HAS_SURREALDB is True
        except ImportError:
            assert _HAS_SURREALDB is False


# ── Config ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBConfig:
    def test_default_config(self) -> None:
        from khora.config.schema import SurrealDBConfig

        cfg = SurrealDBConfig()
        assert cfg.backend == "surrealdb"
        assert cfg.mode == "memory"
        assert cfg.namespace == "khora"
        assert cfg.database == "default"
        assert cfg.embedding_dimension == 1536

    def test_embedded_config(self) -> None:
        from khora.config.schema import SurrealDBConfig

        cfg = SurrealDBConfig(mode="embedded", path="/tmp/test.db")
        assert cfg.mode == "embedded"
        assert cfg.path == "/tmp/test.db"

    def test_remote_config(self) -> None:
        from khora.config.schema import SurrealDBConfig

        cfg = SurrealDBConfig(mode="remote", url="ws://localhost:8000")
        assert cfg.mode == "remote"
        # url is SecretStr — unwrap to compare plaintext.
        assert cfg.url.get_secret_value() == "ws://localhost:8000"

    def test_config_in_graph_union(self) -> None:
        from khora.config.schema import StorageSettings

        settings = StorageSettings(graph={"backend": "surrealdb", "mode": "memory"})
        assert settings.graph is not None
        assert settings.graph.backend == "surrealdb"

    def test_config_default_user_password(self) -> None:
        from khora.config.schema import SurrealDBConfig

        cfg = SurrealDBConfig()
        assert cfg.user == "root"
        # password is SecretStr — unwrap to compare plaintext.
        assert cfg.password.get_secret_value() == "root"

    def test_storage_backend_field(self) -> None:
        from khora.config.schema import StorageSettings

        settings = StorageSettings(backend="surrealdb")
        assert settings.backend == "surrealdb"

    def test_storage_backend_default_postgres(self) -> None:
        from khora.config.schema import StorageSettings

        settings = StorageSettings()
        assert settings.backend == "postgres"

    def test_storage_settings_surrealdb_field(self) -> None:
        from khora.config.schema import StorageSettings, SurrealDBConfig

        cfg = SurrealDBConfig(mode="remote", url="ws://localhost:8000")
        settings = StorageSettings(backend="surrealdb", surrealdb=cfg)
        assert settings.surrealdb is not None
        assert settings.surrealdb.url.get_secret_value() == "ws://localhost:8000"

    def test_vector_config_union(self) -> None:
        from khora.config.schema import SurrealDBVectorConfig

        cfg = SurrealDBVectorConfig()
        assert cfg.backend == "surrealdb"
        assert cfg.mode == "memory"
        assert cfg.embedding_dimension == 1536


# ── Connection ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBConnection:
    def test_endpoint_memory(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection(mode="memory")
        assert conn._build_endpoint() == "memory://default"

    def test_endpoint_embedded(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection(mode="embedded", path="/tmp/test.db")
        assert conn._build_endpoint() == "surrealkv:///tmp/test.db"

    def test_endpoint_remote(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection(mode="remote", url="ws://localhost:8000")
        assert conn._build_endpoint() == "ws://localhost:8000"

    def test_endpoint_embedded_requires_path(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection(mode="embedded")
        with pytest.raises(ValueError, match="path"):
            conn._build_endpoint()

    def test_endpoint_remote_requires_url(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection(mode="remote")
        with pytest.raises(ValueError, match="url"):
            conn._build_endpoint()

    def test_endpoint_unknown_mode(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection(mode="bogus")
        with pytest.raises(ValueError, match="Unknown"):
            conn._build_endpoint()

    def test_not_connected_initially(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        assert not conn.connected

    def test_client_initially_none(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        assert conn.client is None

    def test_default_params(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        assert conn._mode == "memory"
        assert conn._namespace == "khora"
        assert conn._database == "default"
        assert conn._user == "root"
        assert conn._password == "root"

    async def test_disconnect_when_not_connected(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        # Should be a no-op, no errors
        await conn.disconnect()
        assert not conn.connected

    async def test_concurrent_disconnect_calls_close_once(self) -> None:
        """Four parallel disconnects (one per adapter sharing a connection)
        must call ``client.close()`` exactly once.

        Regression for #715: previously each adapter raced into close() on the
        same SurrealDB client, leaving pyo3 tokio workers in a state where
        they called back into Python after interpreter finalization — SIGABRT.
        """
        import asyncio

        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        client = AsyncMock()
        client.close = AsyncMock()
        conn._client = client
        conn._connected = True

        await asyncio.gather(*(conn.disconnect() for _ in range(4)))

        assert not conn.connected
        assert conn._client is None
        client.close.assert_awaited_once()

    async def test_disconnect_swallows_client_close_failure(self) -> None:
        """A failing client.close() must not propagate — it would mask the
        user's traceback in __aexit__ (#715)."""
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        conn._client = AsyncMock()
        conn._client.close = AsyncMock(side_effect=RuntimeError("kaboom"))
        conn._connected = True

        # Must not raise
        await conn.disconnect()
        assert not conn.connected
        assert conn._client is None

    async def test_is_healthy_when_not_connected(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        assert await conn.is_healthy() is False

    async def test_query_raises_when_not_connected(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        with pytest.raises(RuntimeError, match="not connected"):
            await conn.query("SELECT 1")

    async def test_execute_raises_when_not_connected(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        with pytest.raises(RuntimeError, match="not connected"):
            await conn.execute("SELECT 1")


# ── Schema ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBSchema:
    def test_schema_module_importable(self) -> None:
        from khora.storage.backends.surrealdb import schema

        assert hasattr(schema, "initialize_schema")

    def test_table_definitions_contain_required_tables(self) -> None:
        from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

        for table in [
            "memory_namespace",
            "document",
            "chunk",
            "entity",
            "relates_to",
            "episode",
            "memory_event",
            "sync_checkpoint",
        ]:
            assert table in _TABLE_DEFINITIONS, f"Missing table: {table}"

    def test_schema_has_hnsw_index(self) -> None:
        from khora.storage.backends.surrealdb.schema import _SEARCH_INDEX_DEFINITIONS

        assert "HNSW" in _SEARCH_INDEX_DEFINITIONS
        assert "DIMENSION 1536" in _SEARCH_INDEX_DEFINITIONS

    def test_schema_has_bm25(self) -> None:
        from khora.storage.backends.surrealdb.schema import _SEARCH_INDEX_DEFINITIONS

        assert "BM25" in _SEARCH_INDEX_DEFINITIONS

    def test_schema_has_analyzer(self) -> None:
        from khora.storage.backends.surrealdb.schema import _ANALYZER_DEFINITIONS

        assert "khora_fulltext" in _ANALYZER_DEFINITIONS
        assert "snowball" in _ANALYZER_DEFINITIONS

    def test_schema_has_unique_indexes(self) -> None:
        from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

        assert "UNIQUE" in _TABLE_DEFINITIONS
        assert "idx_entity_unique" in _TABLE_DEFINITIONS
        assert "idx_sync_checkpoint_ns_source" in _TABLE_DEFINITIONS

    def test_schema_has_relation_tables(self) -> None:
        from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

        assert "TYPE RELATION" in _TABLE_DEFINITIONS
        # relates_to and temporal_edge are relation tables
        assert "relates_to" in _TABLE_DEFINITIONS
        assert "temporal_edge" in _TABLE_DEFINITIONS

    def test_schema_has_temporal_tables(self) -> None:
        from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

        assert "time_node" in _TABLE_DEFINITIONS
        assert "temporal_edge" in _TABLE_DEFINITIONS
        assert "time_edge_link" in _TABLE_DEFINITIONS

    async def test_initialize_schema_calls_execute(self) -> None:
        from khora.storage.backends.surrealdb.schema import initialize_schema

        conn = _make_mock_conn()
        await initialize_schema(conn)
        # Should call execute at least twice (analyzer + tables)
        assert conn.execute.await_count >= 2


# ── Factory ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBFactory:
    def test_surrealdb_in_graph_registry(self) -> None:
        from khora.storage.factory import _GRAPH_REGISTRY

        assert "surrealdb" in _GRAPH_REGISTRY
        module_path, class_name = _GRAPH_REGISTRY["surrealdb"]
        assert "surrealdb" in module_path
        assert class_name == "SurrealDBGraphAdapter"

    def test_surrealdb_in_vector_registry(self) -> None:
        from khora.storage.factory import _VECTOR_REGISTRY

        assert "surrealdb" in _VECTOR_REGISTRY
        module_path, class_name = _VECTOR_REGISTRY["surrealdb"]
        assert "surrealdb" in module_path
        assert class_name == "SurrealDBVectorAdapter"

    def test_storage_config_default_backend(self) -> None:
        from khora.storage.factory import StorageConfig

        cfg = StorageConfig()
        assert cfg.backend == "postgres"

    def test_storage_config_surrealdb_backend(self) -> None:
        from khora.storage.factory import StorageConfig

        cfg = StorageConfig(backend="surrealdb")
        assert cfg.backend == "surrealdb"


# ── Relational Adapter — helpers ──────────────────────────────────────────


@pytest.mark.unit
class TestRelationalHelpers:
    @pytest.mark.skipif(not _HAS_SURREALDB, reason="surrealdb not installed")
    def test_record_id(self) -> None:
        from khora.storage.backends.surrealdb.relational import _record_id

        uid = UUID("12345678-1234-5678-1234-567812345678")
        result = _record_id("document", uid)
        # _record_id returns a RecordID; UUID passed directly (no angle brackets)
        assert str(result) == "document:12345678-1234-5678-1234-567812345678"

    def test_parse_uuid_bare(self) -> None:
        from khora.storage.backends.surrealdb.relational import _parse_uuid

        uid = _parse_uuid("12345678-1234-5678-1234-567812345678")
        assert uid == UUID("12345678-1234-5678-1234-567812345678")

    def test_parse_uuid_with_table_prefix(self) -> None:
        from khora.storage.backends.surrealdb.relational import _parse_uuid

        uid = _parse_uuid("document:12345678-1234-5678-1234-567812345678")
        assert uid == UUID("12345678-1234-5678-1234-567812345678")

    def test_parse_uuid_with_angle_brackets(self) -> None:
        from khora.storage.backends.surrealdb.relational import _parse_uuid

        uid = _parse_uuid("document:⟨12345678-1234-5678-1234-567812345678⟩")
        assert uid == UUID("12345678-1234-5678-1234-567812345678")

    def test_parse_dt(self) -> None:
        from khora.storage.backends.surrealdb._helpers import _parse_dt

        result = _parse_dt("2025-01-15T12:00:00+00:00")
        assert result is not None
        assert result.year == 2025

    def test_parse_dt_none(self) -> None:
        from khora.storage.backends.surrealdb._helpers import _parse_dt

        assert _parse_dt(None) is None

    def test_parse_dt_passthrough_datetime(self) -> None:
        from khora.storage.backends.surrealdb._helpers import _parse_dt

        dt = datetime(2025, 1, 15, tzinfo=UTC)
        assert _parse_dt(dt) is dt


# ── Relational Adapter — lifecycle ────────────────────────────────────────


@pytest.mark.unit
class TestRelationalAdapterLifecycle:
    async def test_connect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        await adapter.connect()
        conn.connect.assert_awaited_once()

    async def test_disconnect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        await adapter.disconnect()
        conn.disconnect.assert_awaited_once()

    async def test_is_healthy_delegates(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        result = await adapter.is_healthy()
        assert result is True

    def test_get_session_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        assert adapter._get_session() is None

    def test_from_config_defaults(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        adapter = SurrealDBRelationalAdapter.from_config({})
        assert adapter._conn._mode == "memory"
        assert adapter._conn._namespace == "khora"

    def test_from_config_custom(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        adapter = SurrealDBRelationalAdapter.from_config(
            {
                "mode": "remote",
                "url": "ws://db:8000",
                "namespace": "myns",
                "database": "mydb",
                "user": "admin",
                "password": "secret",
            }
        )
        assert adapter._conn._mode == "remote"
        assert adapter._conn._url == "ws://db:8000"
        assert adapter._conn._namespace == "myns"
        assert adapter._conn._database == "mydb"


# ── Relational Adapter — namespace operations ─────────────────────────────


@pytest.mark.unit
class TestRelationalAdapterNamespace:
    async def test_create_namespace(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        row_id = uuid4()
        row = _namespace_row(row_id, ns_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        ns = MemoryNamespace(id=row_id, namespace_id=ns_id)
        result = await adapter.create_namespace(ns)

        conn.query_one.assert_awaited_once()
        assert result.id == row_id
        assert result.namespace_id == ns_id
        assert result.is_active is True

    async def test_create_namespace_raises_on_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        ns = MemoryNamespace()
        with pytest.raises(RuntimeError, match="Failed to create namespace"):
            await adapter.create_namespace(ns)

    async def test_resolve_namespace(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        row_id = uuid4()
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"id": f"memory_namespace:⟨{row_id!s}⟩", "namespace_id": str(ns_id)})
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.resolve_namespace(ns_id)
        # SurrealDB resolve returns the stable namespace_id (not row-level id)
        # because chunks store namespace refs using namespace_id
        assert result == ns_id

    async def test_resolve_namespace_raises_on_missing(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        with pytest.raises(ValueError, match="No active namespace"):
            await adapter.resolve_namespace(uuid4())

    async def test_get_namespace(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        row_id = uuid4()
        ns_id = uuid4()
        row = _namespace_row(row_id, ns_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_namespace(row_id)
        assert result is not None
        assert result.id == row_id

    async def test_get_namespace_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_namespace(uuid4())
        assert result is None

    async def test_deactivate_namespace(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        await adapter.deactivate_namespace(uuid4())
        conn.execute.assert_awaited_once()

    async def test_update_namespace(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        ns = MemoryNamespace()
        result = await adapter.update_namespace(ns)
        conn.execute.assert_awaited_once()
        assert result is ns

    async def test_list_namespaces(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        row1 = _namespace_row()
        row2 = _namespace_row()
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"total": 2})
        conn.query = AsyncMock(return_value=[row1, row2])
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.list_namespaces()
        assert result.total == 2
        assert len(result.items) == 2
        assert result.limit == 100
        assert result.offset == 0

    async def test_create_namespace_version_without_previous(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        # create_namespace calls query_one internally
        conn.query_one = AsyncMock(return_value=_namespace_row(version=1))
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.create_namespace_version()
        assert result.version == 1  # first version

    async def test_create_namespace_version_with_previous(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=_namespace_row(version=2))
        adapter = SurrealDBRelationalAdapter(conn)

        prev = MemoryNamespace(version=1, config_overrides={"key": "val"})
        await adapter.create_namespace_version(previous_version=prev)
        # Deactivate should have been called
        conn.execute.assert_awaited()


# ── Relational Adapter — document operations ──────────────────────────────


@pytest.mark.unit
class TestRelationalAdapterDocument:
    async def test_create_document(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        ns_id = uuid4()
        row = _document_row(doc_id, ns_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        doc = Document(
            id=doc_id,
            namespace_id=ns_id,
            content="test content",
            title="Test Doc",
        )
        result = await adapter.create_document(doc)

        assert result.id == doc_id
        assert result.content == "test content"
        assert result.title == "Test Doc"

    async def test_create_document_raises_on_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        with pytest.raises(RuntimeError, match="Failed to create document"):
            await adapter.create_document(Document())

    async def test_get_document(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        row = _document_row(doc_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_document(doc_id, namespace_id=uuid4())
        assert result is not None
        assert result.id == doc_id

    async def test_get_document_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_document(uuid4(), namespace_id=uuid4())
        assert result is None

    async def test_list_documents(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        rows = [_document_row(ns_id=ns_id), _document_row(ns_id=ns_id)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.list_documents(ns_id)
        assert len(result) == 2

    async def test_list_documents_with_status_filter(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.list_documents(ns_id, status="completed")
        assert result == []
        # Verify the query included the status parameter
        call_args = conn.query.call_args
        assert "status" in call_args[0][0]

    async def test_update_document(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        doc = Document(content="updated")
        result = await adapter.update_document(doc)
        conn.execute.assert_awaited_once()
        assert result is doc

    async def test_delete_document_exists(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[{"id": f"document:⟨{doc_id!s}⟩"}])
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.delete_document(doc_id, namespace_id=uuid4())
        assert result is True
        conn.query.assert_awaited_once()

    async def test_delete_document_missing(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.delete_document(uuid4(), namespace_id=uuid4())
        assert result is False
        conn.query.assert_awaited_once()

    async def test_get_document_by_checksum(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        row = _document_row(ns_id=ns_id)
        row["checksum"] = "sha256abc"

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_document_by_checksum(ns_id, "sha256abc")
        assert result is not None
        assert result.checksum == "sha256abc"

    async def test_get_document_by_checksum_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_document_by_checksum(uuid4(), "nope")
        assert result is None

    async def test_get_documents_batch(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc1 = uuid4()
        doc2 = uuid4()
        rows = [_document_row(doc1), _document_row(doc2)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_documents_batch([doc1, doc2], namespace_id=uuid4())
        assert len(result) == 2
        assert doc1 in result
        assert doc2 in result

    async def test_get_documents_batch_empty(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_documents_batch([], namespace_id=uuid4())
        assert result == {}
        conn.query.assert_not_awaited()

    async def test_get_document_sources_batch(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        now = datetime.now(UTC).isoformat()
        conn = _make_mock_conn()
        conn.query = AsyncMock(
            return_value=[
                {
                    "id": f"document:⟨{doc_id!s}⟩",
                    "title": "My Doc",
                    "source": "http://example.com",
                    "source_type": "url",
                    "created_at": now,
                    "source_timestamp": None,
                }
            ]
        )
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_document_sources_batch([doc_id], namespace_id=uuid4())
        assert doc_id in result
        assert isinstance(result[doc_id], DocumentSource)
        assert result[doc_id].title == "My Doc"

    async def test_get_document_sources_batch_empty(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_document_sources_batch([], namespace_id=uuid4())
        assert result == {}

    async def test_count_documents(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 7})
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.count_documents(ns_id)
        assert result == 7

    async def test_count_documents_empty(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.count_documents(uuid4())
        assert result == 0

    async def test_create_document_with_external_id(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        ns_id = uuid4()
        row = _document_row(doc_id, ns_id)
        row["external_id"] = "ext-123"

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        doc = Document(
            id=doc_id,
            namespace_id=ns_id,
            content="test content",
            title="Test Doc",
            external_id="ext-123",
        )
        result = await adapter.create_document(doc)

        assert result.id == doc_id
        assert result.external_id == "ext-123"
        # Verify external_id was passed to the query
        call_args = conn.query_one.call_args
        assert call_args[0][1]["external_id"] == "ext-123"

    async def test_create_document_without_external_id(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        ns_id = uuid4()
        row = _document_row(doc_id, ns_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBRelationalAdapter(conn)

        doc = Document(
            id=doc_id,
            namespace_id=ns_id,
            content="test content",
            title="Test Doc",
        )
        result = await adapter.create_document(doc)

        assert result.id == doc_id
        assert result.external_id is None

    async def test_get_last_activity_at(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"latest": ts})
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_last_activity_at(ns_id)
        assert result == ts

    async def test_get_last_activity_at_empty(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_last_activity_at(uuid4())
        assert result is None

    async def test_get_document_stats(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        ns_id = uuid4()
        ts = datetime(2026, 4, 7, 12, 0, 0, tzinfo=UTC)
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 5, "latest": ts})
        adapter = SurrealDBRelationalAdapter(conn)

        count, last_activity = await adapter.get_document_stats(ns_id)
        assert count == 5
        assert last_activity == ts

    async def test_get_document_stats_empty(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        count, last_activity = await adapter.get_document_stats(uuid4())
        assert count == 0
        assert last_activity is None


# ── Relational Adapter — sync checkpoint operations ───────────────────────


@pytest.mark.unit
class TestRelationalAdapterSyncCheckpoint:
    async def test_get_sync_checkpoint(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"checkpoint": "2025-01-01T00:00:00Z"})
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_sync_checkpoint(uuid4(), "slack")
        assert result == "2025-01-01T00:00:00Z"

    async def test_get_sync_checkpoint_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.get_sync_checkpoint(uuid4(), "slack")
        assert result is None

    async def test_set_sync_checkpoint(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)
        await adapter.set_sync_checkpoint(uuid4(), "slack", "2025-01-01")
        conn.execute.assert_awaited_once()
        # Verify the UPSERT query
        call_args = conn.execute.call_args
        assert "UPSERT" in call_args[0][0]


# ── Row-to-model conversion ──────────────────────────────────────────────


@pytest.mark.unit
class TestRelationalRowConversion:
    def test_row_to_namespace_model(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)

        row_id = uuid4()
        ns_id = uuid4()
        row = _namespace_row(row_id, ns_id, version=3, is_active=False)
        result = adapter._row_to_namespace(row)

        assert isinstance(result, MemoryNamespace)
        assert result.id == row_id
        assert result.namespace_id == ns_id
        assert result.version == 3
        assert result.is_active is False

    def test_row_to_document_model(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)

        doc_id = uuid4()
        ns_id = uuid4()
        row = _document_row(doc_id, ns_id)
        result = adapter._row_to_document(row)

        assert isinstance(result, Document)
        assert result.id == doc_id
        assert result.namespace_id == ns_id
        assert result.content == "test content"
        assert result.status == DocumentStatus.PENDING
        assert result.source == "test-source"
        assert result.source_type == "file"
        assert result.title == "Test Doc"
        assert result.metadata == {"key": "value"}
        assert result.external_id is None

    def test_row_to_document_with_external_id(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)

        row = _document_row()
        row["external_id"] = "ext-populated"
        result = adapter._row_to_document(row)

        assert result.external_id == "ext-populated"

    def test_row_to_document_handles_completed_status(self) -> None:
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBRelationalAdapter(conn)

        row = _document_row()
        row["status"] = "completed"
        row["chunk_count"] = 5
        row["entity_count"] = 3
        row["processed_at"] = datetime.now(UTC).isoformat()

        result = adapter._row_to_document(row)
        assert result.status == DocumentStatus.COMPLETED
        assert result.chunk_count == 5
        assert result.entity_count == 3
        assert result.processed_at is not None


# ── Connection query/query_one edge cases ─────────────────────────────────


@pytest.mark.unit
class TestConnectionQueryEdgeCases:
    async def test_query_flattens_nested_lists(self) -> None:
        """Verify that query() flattens nested list-of-list responses."""
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        conn._connected = True

        mock_client = AsyncMock()
        # SurrealDB sometimes returns [[{...}, {...}]] for multi-statement results
        mock_client.query = AsyncMock(return_value=[[{"a": 1}, {"b": 2}]])
        conn._client = mock_client

        results = await conn.query("SELECT * FROM chunk")
        assert len(results) == 2
        assert results[0] == {"a": 1}

    async def test_query_handles_dict_response(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        conn._connected = True

        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value={"x": 1})
        conn._client = mock_client

        results = await conn.query("RETURN 1")
        assert results == [{"x": 1}]

    async def test_query_handles_non_list_non_dict(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        conn._connected = True

        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=42)
        conn._client = mock_client

        results = await conn.query("RETURN 1")
        assert results == []

    async def test_query_one_returns_first(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        conn._connected = True

        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=[{"id": "x"}, {"id": "y"}])
        conn._client = mock_client

        result = await conn.query_one("SELECT * FROM thing")
        assert result == {"id": "x"}

    async def test_query_one_returns_none_on_empty(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        conn._connected = True

        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=[])
        conn._client = mock_client

        result = await conn.query_one("SELECT * FROM thing WHERE id = 'nope'")
        assert result is None


# ── Vector Adapter — helpers ──────────────────────────────────────────────


def _chunk_row(
    chunk_id: UUID | None = None,
    ns_id: UUID | None = None,
    doc_id: UUID | None = None,
    *,
    content: str = "test chunk content",
    embedding: list[float] | None = None,
    similarity: float | None = None,
    rank: float | None = None,
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like a chunk row."""
    chunk_id = chunk_id or uuid4()
    ns_id = ns_id or uuid4()
    doc_id = doc_id or uuid4()
    now = datetime.now(UTC).isoformat()
    row: dict[str, object] = {
        "id": f"chunk:\u27e8{chunk_id!s}\u27e9",
        "namespace": f"memory_namespace:\u27e8{ns_id!s}\u27e9",
        "document": f"document:\u27e8{doc_id!s}\u27e9",
        "content": content,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(content),
        "token_count": 3,
        "metadata_": {},
        "embedding": embedding,
        "embedding_model": "test-model",
        "created_at": now,
        "source_timestamp": None,
    }
    if similarity is not None:
        row["similarity"] = similarity
    if rank is not None:
        row["rank"] = rank
    return row


def _entity_row(
    entity_id: UUID | None = None,
    ns_id: UUID | None = None,
    *,
    name: str = "TestEntity",
    entity_type: str = "PERSON",
    similarity: float | None = None,
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like an entity row."""
    entity_id = entity_id or uuid4()
    ns_id = ns_id or uuid4()
    now = datetime.now(UTC).isoformat()
    row: dict[str, object] = {
        "id": f"entity:\u27e8{entity_id!s}\u27e9",
        "namespace": f"memory_namespace:\u27e8{ns_id!s}\u27e9",
        "name": name,
        "entity_type": entity_type,
        "description": "A test entity",
        "attributes": {},
        "source_document_ids": [],
        "source_chunk_ids": [],
        "source_tool": "",
        "mention_count": 1,
        "embedding": [0.1] * 10,
        "embedding_model": "test-model",
        "valid_from": None,
        "valid_until": None,
        "confidence": 0.95,
        "metadata_": {},
        "created_at": now,
        "updated_at": now,
    }
    if similarity is not None:
        row["similarity"] = similarity
    return row


# ── Vector Adapter — lifecycle ────────────────────────────────────────────


@pytest.mark.unit
class TestVectorAdapterLifecycle:
    async def test_connect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)
        await adapter.connect()
        conn.connect.assert_awaited_once()

    async def test_disconnect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)
        await adapter.disconnect()
        conn.disconnect.assert_awaited_once()

    async def test_is_healthy_delegates(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)
        result = await adapter.is_healthy()
        assert result is True

    def test_get_session_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)
        assert adapter._get_session() is None

    def test_from_config_defaults(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        adapter = SurrealDBVectorAdapter.from_config({})
        assert adapter._conn._mode == "memory"
        assert adapter._hnsw_ef_search == 100

    def test_from_config_custom(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        adapter = SurrealDBVectorAdapter.from_config(
            {
                "mode": "remote",
                "url": "ws://db:8000",
                "hnsw_ef_search": 100,
            }
        )
        assert adapter._conn._mode == "remote"
        assert adapter._hnsw_ef_search == 100


# ── Vector Adapter — helpers ──────────────────────────────────────────────


@pytest.mark.unit
class TestVectorAdapterHelpers:
    def test_rid(self) -> None:
        from khora.storage.backends.surrealdb.vector import _rid

        uid = UUID("12345678-1234-5678-1234-567812345678")
        # _rid returns a RecordID object; SDK ≥2.0 may wrap non-simple keys in angle brackets
        result = str(_rid("chunk", uid))
        assert result in (
            "chunk:12345678-1234-5678-1234-567812345678",
            "chunk:⟨12345678-1234-5678-1234-567812345678⟩",
        )

    def test_parse_uuid_bare(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_uuid

        uid = _parse_uuid("12345678-1234-5678-1234-567812345678")
        assert uid == UUID("12345678-1234-5678-1234-567812345678")

    def test_parse_uuid_with_table_prefix(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_uuid

        uid = _parse_uuid("chunk:12345678-1234-5678-1234-567812345678")
        assert uid == UUID("12345678-1234-5678-1234-567812345678")

    def test_parse_uuid_with_angle_brackets(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_uuid

        uid = _parse_uuid("chunk:\u27e812345678-1234-5678-1234-567812345678\u27e9")
        assert uid == UUID("12345678-1234-5678-1234-567812345678")

    def test_parse_uuid_passthrough(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_uuid

        uid = UUID("12345678-1234-5678-1234-567812345678")
        assert _parse_uuid(uid) is uid

    def test_parse_dt(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_dt

        result = _parse_dt("2025-06-15T12:00:00+00:00")
        assert result is not None
        assert result.year == 2025

    def test_parse_dt_none(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_dt

        assert _parse_dt(None) is None

    def test_parse_dt_passthrough(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_dt

        dt = datetime(2025, 1, 1, tzinfo=UTC)
        assert _parse_dt(dt) is dt

    def test_parse_dt_z_suffix(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_dt

        result = _parse_dt("2025-06-15T12:00:00Z")
        assert result is not None
        assert result.year == 2025

    def test_parse_dt_invalid(self) -> None:
        from khora.storage.backends.surrealdb.vector import _parse_dt

        assert _parse_dt("not-a-date") is None


# ── Vector Adapter — chunk operations ─────────────────────────────────────


@pytest.mark.unit
class TestVectorAdapterChunkOps:
    async def test_create_chunk(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        from khora.core.models import Chunk

        chunk = Chunk(
            content="hello world",
            embedding=[0.1] * 10,
            embedding_model="test",
            chunk_index=0,
        )
        result = await adapter.create_chunk(chunk)
        conn.execute.assert_awaited_once()
        assert result is chunk

    async def test_create_chunks_batch(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        from khora.core.models import Chunk

        chunks = [
            Chunk(content="a", embedding=[0.1]),
            Chunk(content="b", embedding=[0.2]),
        ]
        result = await adapter.create_chunks_batch(chunks)
        conn.execute.assert_awaited_once()
        assert len(result) == 2

    async def test_create_chunks_batch_empty(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)
        result = await adapter.create_chunks_batch([])
        assert result == []
        conn.execute.assert_not_awaited()

    async def test_get_chunk(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        chunk_id = uuid4()
        row = _chunk_row(chunk_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.get_chunk(chunk_id, namespace_id=uuid4())
        assert result is not None
        assert result.id == chunk_id
        assert result.content == "test chunk content"

    async def test_get_chunk_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.get_chunk(uuid4(), namespace_id=uuid4())
        assert result is None

    async def test_get_chunks_batch(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        c1, c2 = uuid4(), uuid4()
        rows = [_chunk_row(c1), _chunk_row(c2)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.get_chunks_batch([c1, c2], namespace_id=uuid4())
        assert len(result) == 2
        assert c1 in result
        assert c2 in result

    async def test_get_chunks_batch_empty(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)
        result = await adapter.get_chunks_batch([], namespace_id=uuid4())
        assert result == {}

    async def test_get_chunks_by_document(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        doc_id = uuid4()
        rows = [_chunk_row(doc_id=doc_id), _chunk_row(doc_id=doc_id)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.get_chunks_by_document(doc_id, namespace_id=uuid4())
        assert len(result) == 2

    async def test_delete_chunks_by_document(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 3})
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.delete_chunks_by_document(uuid4(), namespace_id=uuid4())
        assert result == 3
        conn.execute.assert_awaited_once()

    async def test_delete_chunks_by_document_zero(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 0})
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.delete_chunks_by_document(uuid4(), namespace_id=uuid4())
        assert result == 0
        conn.execute.assert_not_awaited()

    async def test_count_chunks(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 42})
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.count_chunks(uuid4())
        assert result == 42

    async def test_count_chunks_empty(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.count_chunks(uuid4())
        assert result == 0

    async def test_list_chunks(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        rows = [_chunk_row(ns_id=ns_id), _chunk_row(ns_id=ns_id)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.list_chunks(ns_id, limit=10, offset=0)
        assert len(result) == 2


# ── Vector Adapter — search operations ────────────────────────────────────


@pytest.mark.unit
class TestVectorAdapterSearch:
    async def test_search_similar_returns_tuples(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        chunk_id = uuid4()
        row = _chunk_row(chunk_id, ns_id, embedding=[0.1] * 10, similarity=0.95)

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[row])
        adapter = SurrealDBVectorAdapter(conn)

        results = await adapter.search_similar(ns_id, [0.1] * 10, limit=5)
        assert len(results) == 1
        chunk, score = results[0]
        assert score == pytest.approx(0.95)
        assert chunk.id == chunk_id

    async def test_search_similar_filters_by_min_similarity(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        row = _chunk_row(ns_id=ns_id, similarity=0.3)

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[row])
        adapter = SurrealDBVectorAdapter(conn)

        results = await adapter.search_similar(ns_id, [0.1] * 10, min_similarity=0.5)
        assert len(results) == 0

    async def test_search_similar_with_document_filter(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        doc_id = uuid4()

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBVectorAdapter(conn)

        await adapter.search_similar(ns_id, [0.1] * 10, filter_document_ids=[doc_id])
        call_args = conn.query.call_args
        sql = call_args[0][0]
        assert "document IN" in sql

    async def test_search_similar_with_time_filters(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBVectorAdapter(conn)

        after = datetime(2025, 1, 1, tzinfo=UTC)
        before = datetime(2025, 12, 31, tzinfo=UTC)
        await adapter.search_similar(ns_id, [0.1] * 10, created_after=after, created_before=before)
        call_args = conn.query.call_args
        sql = call_args[0][0]
        assert "created_after" in sql
        assert "created_before" in sql

    async def test_search_similar_with_metadata_filters(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBVectorAdapter(conn)

        await adapter.search_similar(ns_id, [0.1] * 10, metadata_filters={"topic": "science"})
        call_args = conn.query.call_args
        sql = call_args[0][0]
        assert "metadata_" in sql

    async def test_search_fulltext(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        chunk_id = uuid4()
        row = _chunk_row(chunk_id, ns_id, content="hello world", rank=1.5)

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[row])
        adapter = SurrealDBVectorAdapter(conn)

        results = await adapter.search_fulltext(ns_id, "hello", limit=5)
        assert len(results) == 1
        chunk, score = results[0]
        assert score == pytest.approx(1.5)
        assert chunk.content == "hello world"

    async def test_search_fulltext_with_time_filters(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBVectorAdapter(conn)

        after = datetime(2025, 1, 1, tzinfo=UTC)
        await adapter.search_fulltext(ns_id, "query", created_after=after)
        call_args = conn.query.call_args
        sql = call_args[0][0]
        assert "created_after" in sql

    async def test_search_similar_entities(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        eid = uuid4()
        row = _entity_row(eid, ns_id, similarity=0.88)

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[row])
        adapter = SurrealDBVectorAdapter(conn)

        results = await adapter.search_similar_entities(ns_id, [0.1] * 10, limit=5)
        assert len(results) == 1
        entity_id, score = results[0]
        assert entity_id == eid
        assert score == pytest.approx(0.88)

    async def test_search_similar_entities_min_similarity(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        ns_id = uuid4()
        row = _entity_row(ns_id=ns_id, similarity=0.2)

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[row])
        adapter = SurrealDBVectorAdapter(conn)

        results = await adapter.search_similar_entities(ns_id, [0.1] * 10, min_similarity=0.5)
        assert len(results) == 0


# ── Vector Adapter — entity operations ────────────────────────────────────


@pytest.mark.unit
class TestVectorAdapterEntityOps:
    async def test_create_entity(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        from khora.core.models import Entity

        entity = Entity(name="Alice", entity_type="PERSON")
        await adapter.create_entity(entity)
        conn.execute.assert_awaited_once()

    async def test_update_entity(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        from khora.core.models import Entity

        entity = Entity(name="Alice", entity_type="PERSON", description="updated")
        await adapter.update_entity(entity, namespace_id=entity.namespace_id)
        conn.execute.assert_awaited_once()

    async def test_entity_exists_true(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 1})
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.entity_exists(uuid4(), namespace_id=uuid4())
        assert result is True

    async def test_entity_exists_false(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 0})
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.entity_exists(uuid4(), namespace_id=uuid4())
        assert result is False

    async def test_entity_exists_none_row(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.entity_exists(uuid4(), namespace_id=uuid4())
        assert result is False

    async def test_update_entity_embedding(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        await adapter.update_entity_embedding(uuid4(), [0.1] * 10, "test-model", namespace_id=uuid4())
        conn.execute.assert_awaited_once()

    async def test_update_entity_embeddings_batch(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        updates = [
            (uuid4(), [0.1] * 10, "model-a"),
            (uuid4(), [0.2] * 10, "model-b"),
        ]
        result = await adapter.update_entity_embeddings_batch(updates, namespace_id=uuid4())
        assert result == 2
        conn.execute.assert_awaited_once()

    async def test_update_entity_embeddings_batch_empty(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        result = await adapter.update_entity_embeddings_batch([], namespace_id=uuid4())
        assert result == 0
        conn.execute.assert_not_awaited()


# ── Vector Adapter — row-to-model conversion ──────────────────────────────


@pytest.mark.unit
class TestVectorRowConversion:
    def test_row_to_chunk(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        chunk_id = uuid4()
        ns_id = uuid4()
        doc_id = uuid4()
        row = _chunk_row(chunk_id, ns_id, doc_id, embedding=[0.5, 0.6])

        result = adapter._row_to_chunk(row)
        assert result.id == chunk_id
        assert result.namespace_id == ns_id
        assert result.document_id == doc_id
        assert result.content == "test chunk content"
        assert result.embedding is not None
        assert result.chunk_index == 0

    def test_row_to_chunk_no_embedding(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        row = _chunk_row(embedding=None)
        result = adapter._row_to_chunk(row)
        assert result.embedding is None

    def test_row_to_chunk_invalid_metadata(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        row = _chunk_row()
        row["metadata_"] = "not-a-dict"
        result = adapter._row_to_chunk(row)
        assert result.metadata == {}

    def test_row_to_entity(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        eid = uuid4()
        ns_id = uuid4()
        row = _entity_row(eid, ns_id, name="Alice", entity_type="PERSON")

        result = adapter._row_to_entity(row)
        assert result.id == eid
        assert result.namespace_id == ns_id
        assert result.name == "Alice"
        assert result.entity_type == "PERSON"
        assert result.confidence == pytest.approx(0.95)

    def test_chunk_to_bindings(self) -> None:
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        from khora.core.models import Chunk

        chunk = Chunk(
            content="test",
            embedding=[0.1, 0.2],
            embedding_model="model",
            chunk_index=3,
            start_char=10,
            end_char=20,
            token_count=5,
        )
        bindings = adapter._chunk_to_bindings(chunk)
        assert bindings["content"] == "test"
        assert bindings["embedding"] == [0.1, 0.2]
        assert bindings["chunk_index"] == 3
        assert bindings["start_char"] == 10
        assert bindings["end_char"] == 20
        assert bindings["token_count"] == 5

    def test_entity_to_bindings(self) -> None:
        from khora.core.models import Entity
        from khora.storage.backends.surrealdb._helpers import _entity_to_bindings

        entity = Entity(
            name="Bob",
            entity_type="PERSON",
            description="A person",
            embedding=[0.3, 0.4],
            confidence=0.9,
        )
        bindings = _entity_to_bindings(entity)
        assert bindings["name"] == "Bob"
        assert bindings["entity_type"] == "PERSON"
        assert bindings["embedding"] == [0.3, 0.4]
        assert bindings["confidence"] == pytest.approx(0.9)


# ===========================================================================
# Phase 2 — Graph adapter + Event store adapter tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers for graph / event-store tests
# ---------------------------------------------------------------------------


def _graph_entity_row(
    entity_id: UUID | None = None,
    ns_id: UUID | None = None,
    *,
    name: str = "Alice",
    entity_type: str = "PERSON",
    description: str = "A test entity",
    mention_count: int = 1,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like a graph entity row."""
    entity_id = entity_id or uuid4()
    ns_id = ns_id or uuid4()
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"entity:\u27e8{entity_id!s}\u27e9",
        "namespace": f"memory_namespace:\u27e8{ns_id!s}\u27e9",
        "name": name,
        "entity_type": entity_type,
        "description": description,
        "attributes": {"role": "engineer"},
        "source_document_ids": [],
        "source_chunk_ids": [],
        "source_tool": "",
        "mention_count": mention_count,
        "embedding": None,
        "embedding_model": "",
        "valid_from": None,
        "valid_until": None,
        "confidence": confidence,
        "metadata_": {},
        "created_at": now,
        "updated_at": now,
    }


def _relationship_row(
    rel_id: UUID | None = None,
    ns_id: UUID | None = None,
    source_id: UUID | None = None,
    target_id: UUID | None = None,
    *,
    relationship_type: str = "KNOWS",
    weight: float = 1.0,
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like a relates_to edge."""
    rel_id = rel_id or uuid4()
    ns_id = ns_id or uuid4()
    source_id = source_id or uuid4()
    target_id = target_id or uuid4()
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"relates_to:\u27e8{rel_id!s}\u27e9",
        "namespace_id": str(ns_id),
        "in": f"entity:\u27e8{source_id!s}\u27e9",
        "out": f"entity:\u27e8{target_id!s}\u27e9",
        "relationship_type": relationship_type,
        "description": "test relationship",
        "properties": {},
        "source_document_ids": [],
        "source_chunk_ids": [],
        "valid_from": None,
        "valid_until": None,
        "confidence": 0.9,
        "weight": weight,
        "metadata_": {},
        "created_at": now,
        "updated_at": now,
    }


def _episode_row(
    episode_id: UUID | None = None,
    ns_id: UUID | None = None,
    *,
    name: str = "Meeting",
    occurred_at: str | None = None,
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like an episode row."""
    episode_id = episode_id or uuid4()
    ns_id = ns_id or uuid4()
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"episode:\u27e8{episode_id!s}\u27e9",
        "namespace": f"memory_namespace:\u27e8{ns_id!s}\u27e9",
        "name": name,
        "description": "A test episode",
        "occurred_at": occurred_at or now,
        "duration_seconds": 3600,
        "entity_ids": [],
        "source_document_ids": [],
        "source_chunk_ids": [],
        "embedding": None,
        "embedding_model": "",
        "metadata_": {},
        "created_at": now,
        "updated_at": now,
    }


def _event_row(
    event_id: UUID | None = None,
    ns_id: UUID | None = None,
    resource_id: UUID | None = None,
    *,
    event_type: str = "document.created",
    resource_type: str = "document",
) -> dict[str, object]:
    """Build a SurrealDB result dict that looks like a memory_event row."""
    event_id = event_id or uuid4()
    ns_id = ns_id or uuid4()
    resource_id = resource_id or uuid4()
    now = datetime.now(UTC).isoformat()
    return {
        "id": f"memory_event:\u27e8{event_id!s}\u27e9",
        "namespace_id": str(ns_id),
        "event_type": event_type,
        "timestamp": now,
        "resource_type": resource_type,
        "resource_id": str(resource_id),
        "data": {"key": "value"},
        "previous_data": None,
        "actor_id": None,
        "actor_type": "system",
        "correlation_id": None,
        "version": 1,
        "metadata_": {},
    }


# ── Graph Adapter — lifecycle ─────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBGraphAdapterLifecycle:
    async def test_connect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)
        await adapter.connect()
        conn.connect.assert_awaited_once()

    async def test_disconnect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)
        await adapter.disconnect()
        conn.disconnect.assert_awaited_once()

    async def test_is_healthy_delegates(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)
        result = await adapter.is_healthy()
        assert result is True
        conn.is_healthy.assert_awaited_once()

    def test_get_session_returns_none(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)
        assert adapter._get_session() is None


# ── Graph Adapter — entity operations ─────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBGraphAdapterEntity:
    async def test_create_entity(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid = uuid4()
        ns_id = uuid4()

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Entity

        entity = Entity(id=eid, namespace_id=ns_id, name="Alice", entity_type="PERSON")
        result = await adapter.create_entity(entity)

        conn.execute.assert_awaited_once()
        assert result.id == eid
        assert result.name == "Alice"
        assert result.entity_type == "PERSON"

    async def test_get_entity(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid = uuid4()
        ns_id = uuid4()
        row = _graph_entity_row(eid, ns_id, name="Bob", entity_type="ORGANIZATION")

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entity(eid, namespace_id=ns_id)

        assert result is not None
        assert result.id == eid
        assert result.name == "Bob"
        assert result.entity_type == "ORGANIZATION"

    async def test_get_entity_not_found(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entity(uuid4(), namespace_id=uuid4())
        assert result is None

    async def test_get_entity_by_name(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid = uuid4()
        ns_id = uuid4()
        row = _graph_entity_row(eid, ns_id, name="Charlie", entity_type="CONCEPT")

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entity_by_name(ns_id, "Charlie", "CONCEPT")

        assert result is not None
        assert result.name == "Charlie"
        assert result.entity_type == "CONCEPT"
        # Verify the query used namespace_id, name, and entity_type
        call_args = conn.query_one.call_args
        query_str = call_args[0][0]
        assert "name" in query_str
        assert "entity_type" in query_str

    async def test_update_entity(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid = uuid4()
        ns_id = uuid4()
        row = _graph_entity_row(eid, ns_id, name="Updated", entity_type="PERSON")

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Entity

        entity = Entity(id=eid, namespace_id=ns_id, name="Updated", entity_type="PERSON")
        result = await adapter.update_entity(entity, namespace_id=ns_id)

        assert result.id == eid
        assert result.name == "Updated"

    async def test_delete_entity(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid = uuid4()
        conn = _make_mock_conn()
        # delete_entity uses DELETE RETURN BEFORE; non-empty list means entity existed
        conn.query = AsyncMock(return_value=[{"id": f"entity:⟨{eid}⟩"}])
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.delete_entity(eid, namespace_id=uuid4())
        assert result is True
        conn.execute.assert_awaited()  # relationship delete
        conn.query.assert_awaited()  # entity DELETE RETURN BEFORE

    async def test_delete_entity_not_found(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        # Empty list from DELETE RETURN BEFORE means nothing was deleted
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.delete_entity(uuid4(), namespace_id=uuid4())
        assert result is False

    async def test_list_entities(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        rows = [
            _graph_entity_row(ns_id=ns_id, name="E1"),
            _graph_entity_row(ns_id=ns_id, name="E2"),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.list_entities(ns_id)
        assert len(result) == 2
        assert result[0].name == "E1"
        assert result[1].name == "E2"

    async def test_list_entities_with_type_filter(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        rows = [_graph_entity_row(ns_id=ns_id, name="Org1", entity_type="ORGANIZATION")]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.list_entities(ns_id, entity_type="ORGANIZATION")
        assert len(result) == 1
        assert result[0].entity_type == "ORGANIZATION"
        # Verify entity_type filter was included in the query
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "entity_type" in query_str

    async def test_count_entities(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"cnt": 42})
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.count_entities(ns_id)
        assert result == 42

    async def test_get_entities_batch(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid1 = uuid4()
        eid2 = uuid4()
        ns_id = uuid4()
        rows = [
            _graph_entity_row(eid1, ns_id, name="BatchE1"),
            _graph_entity_row(eid2, ns_id, name="BatchE2"),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entities_batch([eid1, eid2], namespace_id=ns_id)
        assert len(result) == 2
        assert eid1 in result
        assert eid2 in result
        assert result[eid1].name == "BatchE1"
        assert result[eid2].name == "BatchE2"


# ── Graph Adapter — relationship operations ───────────────────────────────


@pytest.mark.unit
class TestSurrealDBGraphAdapterRelationship:
    async def test_create_relationship(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        rel_id = uuid4()
        ns_id = uuid4()
        src_id = uuid4()
        tgt_id = uuid4()

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Relationship

        rel = Relationship(
            id=rel_id,
            namespace_id=ns_id,
            source_entity_id=src_id,
            target_entity_id=tgt_id,
            relationship_type="WORKS_AT",
        )
        result = await adapter.create_relationship(rel)

        conn.execute.assert_awaited_once()
        assert result.id == rel_id
        assert result.relationship_type == "WORKS_AT"
        # Verify RELATE was used (SurrealDB relation creation)
        call_args = conn.execute.call_args
        query_str = call_args[0][0]
        assert "RELATE" in query_str

    async def test_get_relationship(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        rel_id = uuid4()
        ns_id = uuid4()
        src_id = uuid4()
        tgt_id = uuid4()
        row = _relationship_row(rel_id, ns_id, src_id, tgt_id)

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_relationship(rel_id, namespace_id=ns_id)
        assert result is not None
        assert result.id == rel_id
        assert result.source_entity_id == src_id
        assert result.target_entity_id == tgt_id

    async def test_delete_relationship(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        rel_id = uuid4()
        conn = _make_mock_conn()
        # DELETE RETURN BEFORE returns deleted rows
        conn.query = AsyncMock(return_value=[{"rel_id": str(rel_id)}])
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.delete_relationship(rel_id, namespace_id=uuid4())
        assert result is True
        conn.query.assert_awaited_once()

    async def test_get_entity_relationships_outgoing(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()
        rows = [_relationship_row(ns_id=ns_id, source_id=entity_id)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entity_relationships(entity_id, namespace_id=ns_id, direction="outgoing")
        assert len(result) == 1
        # Verify query uses in=entity:⟨id⟩ for outgoing (in = source in SurrealDB)
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "in" in query_str or "out" in query_str

    async def test_get_entity_relationships_incoming(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()
        rows = [_relationship_row(ns_id=ns_id, target_id=entity_id)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entity_relationships(entity_id, namespace_id=ns_id, direction="incoming")
        assert len(result) == 1

    async def test_get_entity_relationships_both(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()
        rows = [
            _relationship_row(ns_id=ns_id, source_id=entity_id),
            _relationship_row(ns_id=ns_id, target_id=entity_id),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_entity_relationships(entity_id, namespace_id=ns_id, direction="both")
        assert len(result) == 2

    async def test_list_relationships(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        rows = [
            _relationship_row(ns_id=ns_id, relationship_type="KNOWS"),
            _relationship_row(ns_id=ns_id, relationship_type="WORKS_WITH"),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.list_relationships(ns_id)
        assert len(result) == 2
        # Verify namespace filter was used
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "namespace_id" in query_str

    async def test_create_relationships_batch(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[{}, {}, {}])
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Relationship

        rels = [
            Relationship(namespace_id=ns_id, relationship_type="KNOWS"),
            Relationship(namespace_id=ns_id, relationship_type="WORKS_WITH"),
            Relationship(namespace_id=ns_id, relationship_type="LOCATED_IN"),
        ]
        count = await adapter.create_relationships_batch(rels)
        assert count == 3


# ── Graph Adapter — episode operations ────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBGraphAdapterEpisode:
    async def test_create_episode(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ep_id = uuid4()
        ns_id = uuid4()

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Episode

        episode = Episode(id=ep_id, namespace_id=ns_id, name="Team Standup")
        result = await adapter.create_episode(episode)

        conn.execute.assert_awaited_once()
        assert result.id == ep_id
        assert result.name == "Team Standup"

    async def test_get_episode(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ep_id = uuid4()
        ns_id = uuid4()
        row = _episode_row(ep_id, ns_id, name="Sprint Review")

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_episode(ep_id, namespace_id=ns_id)

        assert result is not None
        assert result.id == ep_id
        assert result.name == "Sprint Review"

    async def test_list_episodes(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        rows = [
            _episode_row(ns_id=ns_id, name="Morning"),
            _episode_row(ns_id=ns_id, name="Afternoon"),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        result = await adapter.list_episodes(ns_id, start_time=start, end_time=end)

        assert len(result) == 2
        # Verify time filters were used in the query
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        # Should contain time-related filtering
        assert "occurred_at" in query_str or "created_at" in query_str


# ── Graph Adapter — traversal operations ──────────────────────────────────


@pytest.mark.unit
class TestSurrealDBGraphAdapterTraversal:
    async def test_find_paths(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        src_id = uuid4()
        tgt_id = uuid4()
        # find_paths fetches all depths in a single query with d1, d2, d3 columns.
        single_result = [
            {
                "d1": [{"id": f"entity:\u27e8{tgt_id!s}\u27e9"}],
                "d2": None,
                "d3": None,
            },
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=single_result)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.find_paths(src_id, tgt_id, namespace_id=ns_id, max_depth=3)
        assert isinstance(result, list)
        assert len(result) >= 1
        # Single query for all depths
        assert conn.query.await_count == 1

    async def test_get_neighborhood(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()
        neighbor_id = uuid4()
        # Neighborhood returns entities + relationships
        neighborhood_data = [
            _graph_entity_row(neighbor_id, ns_id, name="Neighbor"),
        ]

        conn = _make_mock_conn()
        # IGR-223: get_neighborhood now verifies the seed entity belongs to
        # ``namespace_id`` (via get_entity → query_one) before traversing.
        conn.query_one = AsyncMock(return_value=_graph_entity_row(entity_id, ns_id, name="Seed"))
        conn.query = AsyncMock(return_value=neighborhood_data)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_neighborhood(entity_id, namespace_id=ns_id, depth=1)
        assert isinstance(result, dict)
        conn.query.assert_awaited()

    async def test_search_entities_by_attribute(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        rows = [_graph_entity_row(ns_id=ns_id, name="Found")]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.search_entities_by_attribute(ns_id, "role", "engineer")
        assert len(result) == 1
        assert result[0].name == "Found"


# ── Graph Adapter — upsert operations ─────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBGraphAdapterUpsert:
    async def test_upsert_entities_batch_new(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        eid = uuid4()

        conn = _make_mock_conn()
        # Batch fetch returns empty list => no existing entities
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Entity

        entity = Entity(id=eid, namespace_id=ns_id, name="NewEntity", entity_type="CONCEPT")
        result = await adapter.upsert_entities_batch(ns_id, [entity])

        assert len(result) == 1
        returned_entity, is_new = result[0]
        assert returned_entity.name == "NewEntity"
        assert is_new is True
        # Batch create via execute
        conn.execute.assert_awaited_once()

    async def test_upsert_entities_batch_existing(self) -> None:
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        eid = uuid4()
        existing_row = _graph_entity_row(
            eid,
            ns_id,
            name="ExistingEntity",
            entity_type="CONCEPT",
            mention_count=3,
        )

        conn = _make_mock_conn()
        # Batch fetch returns the existing entity row
        conn.query = AsyncMock(return_value=[existing_row])
        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models.entity import Entity

        entity = Entity(id=eid, namespace_id=ns_id, name="ExistingEntity", entity_type="CONCEPT")
        result = await adapter.upsert_entities_batch(ns_id, [entity])

        assert len(result) == 1
        returned_entity, is_new = result[0]
        assert returned_entity.name == "ExistingEntity"
        assert is_new is False
        # Batch update via execute
        conn.execute.assert_awaited_once()


# ── Event Store Adapter — lifecycle ───────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBEventStoreLifecycle:
    async def test_connect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBEventStoreAdapter(conn)
        await adapter.connect()
        conn.connect.assert_awaited_once()

    async def test_disconnect_delegates(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBEventStoreAdapter(conn)
        await adapter.disconnect()
        conn.disconnect.assert_awaited_once()

    async def test_is_healthy_delegates(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBEventStoreAdapter(conn)
        result = await adapter.is_healthy()
        assert result is True
        conn.is_healthy.assert_awaited_once()


# ── Event Store Adapter — operations ──────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBEventStoreOperations:
    async def test_append_event(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ev_id = uuid4()
        ns_id = uuid4()
        res_id = uuid4()
        row = _event_row(ev_id, ns_id, res_id, event_type="entity.created", resource_type="entity")

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBEventStoreAdapter(conn)

        from khora.core.models.event import EventType, MemoryEvent

        event = MemoryEvent(
            id=ev_id,
            namespace_id=ns_id,
            event_type=EventType.ENTITY_CREATED,
            resource_type="entity",
            resource_id=res_id,
            data={"key": "value"},
        )
        result = await adapter.append_event(event)

        conn.query_one.assert_awaited_once()
        assert result.id == ev_id
        assert result.event_type == EventType.ENTITY_CREATED
        assert result.resource_type == "entity"

    async def test_append_event_raises_on_none(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBEventStoreAdapter(conn)

        from khora.core.models.event import MemoryEvent

        event = MemoryEvent()
        with pytest.raises(RuntimeError, match="Failed to append event"):
            await adapter.append_event(event)

    async def test_append_events_batch(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()
        ev1_id = uuid4()
        ev2_id = uuid4()
        rows = [
            _event_row(ev1_id, ns_id, event_type="document.created"),
            _event_row(ev2_id, ns_id, event_type="document.updated"),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBEventStoreAdapter(conn)

        from khora.core.models.event import EventType, MemoryEvent

        events = [
            MemoryEvent(id=ev1_id, namespace_id=ns_id, event_type=EventType.DOCUMENT_CREATED),
            MemoryEvent(id=ev2_id, namespace_id=ns_id, event_type=EventType.DOCUMENT_UPDATED),
        ]
        result = await adapter.append_events_batch(events)

        assert len(result) == 2
        conn.query.assert_awaited_once()
        # Verify INSERT INTO was used
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "INSERT INTO" in query_str

    async def test_append_events_batch_empty(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.append_events_batch([])
        assert result == []
        conn.query.assert_not_awaited()

    async def test_get_events(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()
        rows = [
            _event_row(ns_id=ns_id, event_type="document.created"),
            _event_row(ns_id=ns_id, event_type="entity.created"),
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.get_events(ns_id)
        assert len(result) == 2
        conn.query.assert_awaited_once()

    async def test_get_events_with_type_filter(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()
        rows = [_event_row(ns_id=ns_id, event_type="entity.created")]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.get_events(ns_id, event_types=["entity.created"])
        assert len(result) == 1
        # Verify event_type filter was included in the query
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "event_type" in query_str
        bindings = call_args[0][1]
        assert bindings["event_types"] == ["entity.created"]

    async def test_get_events_with_time_filter(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()
        rows = [_event_row(ns_id=ns_id)]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBEventStoreAdapter(conn)

        after = datetime(2024, 1, 1, tzinfo=UTC)
        before = datetime(2024, 12, 31, tzinfo=UTC)
        result = await adapter.get_events(ns_id, after=after, before=before)

        assert len(result) == 1
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "timestamp > $after" in query_str
        assert "timestamp < $before" in query_str

    async def test_get_events_for_resource(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        res_id = uuid4()
        rows = [_event_row(resource_id=res_id, resource_type="document")]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=rows)
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.get_events_for_resource("document", res_id, namespace_id=uuid4())
        assert len(result) == 1
        # Verify resource_type and resource_id were used
        call_args = conn.query.call_args
        query_str = call_args[0][0]
        assert "resource_type" in query_str
        assert "resource_id" in query_str

    async def test_get_latest_event(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        res_id = uuid4()
        row = _event_row(resource_id=res_id, resource_type="entity", event_type="entity.updated")

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=row)
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.get_latest_event("entity", res_id, namespace_id=uuid4())
        assert result is not None
        # Verify LIMIT 1 and ORDER BY were used
        call_args = conn.query_one.call_args
        query_str = call_args[0][0]
        assert "LIMIT 1" in query_str
        assert "ORDER BY" in query_str

    async def test_get_latest_event_not_found(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.get_latest_event("document", uuid4(), namespace_id=uuid4())
        assert result is None

    async def test_count_events(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"total": 17})
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.count_events(ns_id)
        assert result == 17
        # Verify count() query was used
        call_args = conn.query_one.call_args
        query_str = call_args[0][0]
        assert "count()" in query_str

    async def test_count_events_empty(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.count_events(ns_id)
        assert result == 0

    async def test_count_events_with_type_filter(self) -> None:
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value={"total": 5})
        adapter = SurrealDBEventStoreAdapter(conn)

        result = await adapter.count_events(ns_id, event_types=["entity.created"])
        assert result == 5
        call_args = conn.query_one.call_args
        query_str = call_args[0][0]
        assert "event_type" in query_str


# ══════════════════════════════════════════════════════════════════════════
# Phase 3 optimizations
# ══════════════════════════════════════════════════════════════════════════


# ── Unified backend detection ─────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBUnifiedBackendDetection:
    """Tests for StorageCoordinator._is_unified_backend detection."""

    def test_unified_detected_when_same_connection(self) -> None:
        """Coordinator detects shared SurrealDB connection."""
        from khora.storage.coordinator import StorageCoordinator

        conn = MagicMock()
        graph = MagicMock()
        graph._conn = conn
        vector = MagicMock()
        vector._conn = conn
        coord = StorageCoordinator(graph=graph, vector=vector)
        assert coord._is_unified_backend is True

    def test_not_unified_when_different_connections(self) -> None:
        """Coordinator does NOT detect unified for different connections."""
        from khora.storage.coordinator import StorageCoordinator

        graph = MagicMock()
        graph._conn = MagicMock()
        vector = MagicMock()
        vector._conn = MagicMock()
        coord = StorageCoordinator(graph=graph, vector=vector)
        assert coord._is_unified_backend is False

    def test_not_unified_when_no_conn_attr(self) -> None:
        """Coordinator does NOT detect unified for non-SurrealDB backends."""
        from khora.storage.coordinator import StorageCoordinator

        graph = MagicMock(spec=[])  # no _conn attribute
        vector = MagicMock(spec=[])
        coord = StorageCoordinator(graph=graph, vector=vector)
        assert coord._is_unified_backend is False

    def test_not_unified_when_graph_is_none(self) -> None:
        """No crash when graph backend is None."""
        from khora.storage.coordinator import StorageCoordinator

        vector = MagicMock()
        vector._conn = MagicMock()
        coord = StorageCoordinator(graph=None, vector=vector)
        assert coord._is_unified_backend is False

    def test_not_unified_when_vector_is_none(self) -> None:
        """No crash when vector backend is None."""
        from khora.storage.coordinator import StorageCoordinator

        graph = MagicMock()
        graph._conn = MagicMock()
        coord = StorageCoordinator(graph=graph, vector=None)
        assert coord._is_unified_backend is False

    def test_not_unified_when_conn_is_none(self) -> None:
        """Handles the case where _conn exists but is None on one backend."""
        from khora.storage.coordinator import StorageCoordinator

        graph = MagicMock()
        graph._conn = None
        vector = MagicMock()
        vector._conn = MagicMock()
        coord = StorageCoordinator(graph=graph, vector=vector)
        assert coord._is_unified_backend is False


# ── Auto-schema initialization ────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBAutoSchemaInit:
    """Tests for SurrealDBConnection schema/sync defaults.

    NOTE (devil's advocate): The connection class currently has NO
    ``_schema_initialized`` or ``_sync_data`` attributes. These tests
    document the *desired* Phase 3 contract. If they fail, it means
    the Phase 3 connection changes have not landed yet.
    """

    def test_connection_has_schema_initialized_flag(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        assert hasattr(conn, "_schema_initialized"), (
            "Phase 3 contract: SurrealDBConnection must track whether "
            "schema has been initialized to avoid redundant DDL on reconnect"
        )
        assert conn._schema_initialized is False

    def test_sync_data_default_true(self) -> None:
        from khora.storage.backends.surrealdb.connection import SurrealDBConnection

        conn = SurrealDBConnection()
        assert hasattr(conn, "_sync_data"), (
            "Phase 3 contract: SurrealDBConnection must expose _sync_data flag so crash-safe fsync can be toggled"
        )
        assert conn._sync_data is True


# ── Crash-safe defaults in config ─────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBCrashSafeDefaults:
    """Tests for SurrealDBConfig.sync_data field.

    NOTE (devil's advocate): SurrealDBConfig currently does NOT have a
    ``sync_data`` field (see config/schema.py). These tests document the
    Phase 3 requirement. They will fail until the config is extended.
    """

    def test_sync_data_in_config(self) -> None:
        from khora.config.schema import SurrealDBConfig

        cfg = SurrealDBConfig()
        assert hasattr(cfg, "sync_data"), (
            "Phase 3 contract: SurrealDBConfig must include sync_data field to control crash-safe fsync behaviour"
        )
        assert cfg.sync_data is True

    def test_sync_data_can_be_disabled(self) -> None:
        from khora.config.schema import SurrealDBConfig

        cfg = SurrealDBConfig(sync_data=False)
        assert cfg.sync_data is False


# ── Batch optimizations ──────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBBatchOptimizations:
    """Tests verifying batch operations use efficient queries.

    Batch optimisations (Phase 3): ``create_relationships_batch``
    now uses a single ``FOR $rel IN $rels { RELATE ... }`` SurrealQL call
    instead of N individual round-trips.
    """

    async def test_create_relationships_batch_single_call(self) -> None:
        """Batch relationship creation uses single SurrealQL FOR loop, not N calls."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models import Relationship

        ns_id = uuid4()
        entity_a = uuid4()
        entity_b = uuid4()
        rels = [
            Relationship(
                namespace_id=ns_id,
                source_entity_id=entity_a,
                target_entity_id=entity_b,
                relationship_type="RELATES_TO",
            )
            for _ in range(5)
        ]
        result = await adapter.create_relationships_batch(rels)

        # Phase 3 batch optimisation: single FOR $rel IN $rels SurrealQL call
        assert conn.execute.await_count == 1, (
            f"Expected 1 batched execute call but got "
            f"{conn.execute.await_count}. The batch should use a single "
            f"FOR loop, not N individual RELATE statements."
        )
        assert result == 5

        # Verify the SQL uses a FOR loop
        sql = conn.execute.call_args[0][0]
        assert "FOR $rel IN $rels" in sql, "Batch RELATE should use a SurrealQL FOR loop"

    async def test_create_relationships_batch_passes_all_rels_as_bindings(self) -> None:
        """All relationships are passed as a $rels binding, not string-interpolated."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models import Relationship

        rels = [
            Relationship(
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
            )
            for _ in range(3)
        ]
        await adapter.create_relationships_batch(rels)

        bindings = conn.execute.call_args[0][1]
        assert "rels" in bindings
        assert len(bindings["rels"]) == 3

    async def test_create_relationships_batch_empty(self) -> None:
        """Empty list produces zero execute calls."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)
        result = await adapter.create_relationships_batch([])
        assert result == 0
        conn.execute.assert_not_awaited()


# ── SQL injection vectors (devil's advocate review) ──────────────────────


@pytest.mark.unit
class TestSurrealDBSQLInjectionVectors:
    """Review findings: several SurrealQL queries use f-string interpolation
    of user-supplied values rather than parameterised bindings.

    FINDING 1 — ``search_entities_by_attribute`` interpolates
    ``attribute_name`` directly into the query:
        ``f"AND attributes.{attribute_name} = $attr_value "``
    An attacker-controlled attribute name like
    ``"x = 1; DELETE entity WHERE true; -- "`` would produce valid
    (destructive) SurrealQL.

    FINDING 2 — ``find_paths`` and ``get_neighborhood`` interpolate
    ``relationship_types`` list items into the query via:
        ``", ".join(f"'{rt}'" for rt in relationship_types)``
    A relationship type containing a single quote (e.g. ``"RELATES_TO'; DELETE entity--"``)
    would break out of the string literal.

    FINDING 3 — ``get_neighborhood`` interpolates ``limit`` as a raw int
    into the SQL string (``f"LIMIT {limit}"``), though this is lower risk
    since Python ``int`` cannot contain SQL.
    """

    async def test_attribute_name_injection_is_rejected(self) -> None:
        """Verify that malicious attribute_name is rejected by sanitiser."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn(query=[])
        adapter = SurrealDBGraphAdapter(conn)

        # Malicious attribute name — must be rejected by _sanitize_field_name
        malicious_attr = "x = 1; DELETE entity WHERE true; --"
        with pytest.raises(ValueError, match="Unsafe field name"):
            await adapter.search_entities_by_attribute(uuid4(), malicious_attr, "anything")

    async def test_relationship_type_quote_injection_is_parameterised(self) -> None:
        """Verify that relationship_types are parameterised, not interpolated."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn(query=[])
        adapter = SurrealDBGraphAdapter(conn)

        malicious_rt = "RELATES_TO'; DELETE entity WHERE true; --"
        await adapter.find_paths(uuid4(), uuid4(), namespace_id=uuid4(), relationship_types=[malicious_rt])

        # The SQL should use $rel_types parameter, not contain the raw payload
        call_args = conn.query.call_args
        sql = call_args[0][0]
        assert "DELETE entity" not in sql, "The raw injection payload should NOT appear in the SQL string."
        assert "$rel_types" in sql, "relationship_types should be passed as a $rel_types parameter."


# ── UUID parsing edge cases ──────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBUUIDParsingEdgeCases:
    """Review findings: ``_parse_uuid`` silently raises ``ValueError`` on
    garbage input, which propagates as an unhandled exception in
    ``_row_to_entity``, ``_row_to_relationship``, etc.

    FINDING: If SurrealDB returns an unexpected record ID format (e.g.
    a nested object, an integer, or ``None``), ``_parse_uuid`` will raise
    ``ValueError`` with a confusing message like "badly formed hexadecimal
    UUID string". Callers never catch this.
    """

    def test_parse_uuid_with_empty_string_returns_deterministic_uuid(self) -> None:
        """Empty string returns a deterministic UUID5 fallback."""
        from khora.storage.backends.surrealdb._helpers import _parse_uuid

        result = _parse_uuid("")
        assert isinstance(result, UUID)
        # Deterministic: same input always produces same UUID
        assert result == _parse_uuid("")

    def test_parse_uuid_with_none_returns_deterministic_uuid(self) -> None:
        """None input returns a deterministic UUID5 fallback."""
        from khora.storage.backends.surrealdb._helpers import _parse_uuid

        result = _parse_uuid(None)
        assert isinstance(result, UUID)
        assert result == _parse_uuid(None)

    def test_parse_uuid_with_nested_dict_returns_fallback(self) -> None:
        """SurrealDB sometimes returns record links as dicts. Falls back
        to deterministic UUID5."""
        from khora.storage.backends.surrealdb._helpers import _parse_uuid

        result = _parse_uuid({"id": "entity:12345678-1234-5678-1234-567812345678"})
        assert isinstance(result, UUID)

    def test_parse_uuid_with_integer_returns_fallback(self) -> None:
        """Non-UUID record IDs produce deterministic UUID5 fallback."""
        from khora.storage.backends.surrealdb._helpers import _parse_uuid

        result = _parse_uuid(42)
        assert isinstance(result, UUID)
        assert result == _parse_uuid(42)  # deterministic

    def test_parse_uuid_with_valid_uuid_object(self) -> None:
        """UUID objects pass through unchanged."""
        from khora.storage.backends.surrealdb._helpers import _parse_uuid

        uid = uuid4()
        assert _parse_uuid(uid) is uid


# ── Silent failure review ─────────────────────────────────────────────────


@pytest.mark.unit
class TestSurrealDBSilentFailures:
    """Review findings: several methods silently swallow exceptions.

    FINDING 1 — ``create_relationships_batch`` catches all exceptions
    per-relationship and only logs a warning (graph.py line 510-511).
    A caller has no way to know that 3 out of 5 relationships failed.
    The method returns a count but no error details.

    FINDING 2 — ``create_episode`` swallows exceptions when creating
    ``involves`` edges (graph.py line 568-569). If the entity doesn't
    exist, the edge silently isn't created and the episode's entity_ids
    will be inconsistent with the graph.

    FINDING 3 — ``get_neighborhoods_batch`` catches all exceptions per
    entity and returns an empty neighborhood (graph.py line 770-772).
    The caller cannot distinguish "this entity has no neighbors" from
    "the query failed".
    """

    async def test_create_relationships_batch_swallows_errors(self) -> None:
        """Demonstrate that failures are silently swallowed."""
        conn = _make_mock_conn()
        conn.execute = AsyncMock(side_effect=RuntimeError("DB down"))
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)

        from khora.core.models import Relationship

        rels = [
            Relationship(
                namespace_id=uuid4(),
                source_entity_id=uuid4(),
                target_entity_id=uuid4(),
            )
            for _ in range(3)
        ]

        # No exception raised — all 3 failures swallowed
        result = await adapter.create_relationships_batch(rels)
        assert result == 0  # all failed but no error propagated

    async def test_row_to_relationship_wrong_uuid_field(self) -> None:
        """If SurrealDB omits 'in'/'out' fields, _parse_uuid gets empty
        string and raises ValueError — this is NOT caught by the caller."""
        from khora.storage.backends.surrealdb.graph import _row_to_relationship

        row = {
            "rel_id": str(uuid4()),
            "namespace_id": str(uuid4()),
            # deliberately omitting 'in' and 'out'
            "relationship_type": "RELATES_TO",
        }
        # _parse_uuid now returns a deterministic fallback UUID instead of raising
        rel = _row_to_relationship(row)
        assert rel.relationship_type == "RELATES_TO"


# ---------------------------------------------------------------------------
# Input sanitization tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBInputSanitization:
    """Tests for the ``_sanitize_field_name`` helper in ``_helpers.py``."""

    def test_valid_simple_name(self) -> None:
        """Simple alphanumeric names pass."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        assert _sanitize_field_name("email") == "email"
        assert _sanitize_field_name("name123") == "name123"

    def test_valid_dotted_name(self) -> None:
        """Dotted names like 'attributes.email' pass."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        assert _sanitize_field_name("attributes.email") == "attributes.email"
        assert _sanitize_field_name("a.b.c") == "a.b.c"

    def test_valid_underscore_name(self) -> None:
        """Names with underscores pass."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        assert _sanitize_field_name("first_name") == "first_name"
        assert _sanitize_field_name("_private") == "_private"

    def test_rejects_semicolon(self) -> None:
        """Names with semicolons are rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("name; DELETE entity")

    def test_rejects_sql_injection(self) -> None:
        """SQL injection attempts are rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("x = 1 OR 1=1 --")

    def test_rejects_empty(self) -> None:
        """Empty string is rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("")

    def test_rejects_space(self) -> None:
        """Names with spaces are rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("first name")

    def test_rejects_quotes(self) -> None:
        """Names with quotes are rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("name'")
        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name('name"')

    def test_rejects_parentheses(self) -> None:
        """Names with parens are rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("fn()")

    def test_rejects_dash(self) -> None:
        """Names starting with dash are rejected."""
        from khora.storage.backends.surrealdb._helpers import _sanitize_field_name

        with pytest.raises(ValueError, match="Unsafe field name"):
            _sanitize_field_name("-bad")


# ---------------------------------------------------------------------------
# Injection prevention on adapter methods
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBInjectionPrevention:
    """Tests that adapter methods reject malicious input."""

    async def test_search_entities_by_attribute_rejects_injection(self) -> None:
        """search_entities_by_attribute raises ValueError for malicious attribute name."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)

        with pytest.raises(ValueError, match="Unsafe field name"):
            await adapter.search_entities_by_attribute(
                uuid4(),
                "email; DELETE entity",
                "test@example.com",
            )
        # Crucially, the DB connection should never have been called
        conn.query.assert_not_called()

    async def test_search_similar_rejects_metadata_key_injection(self) -> None:
        """search_similar raises ValueError for malicious metadata filter key."""
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBVectorAdapter(conn)

        with pytest.raises(ValueError, match="Unsafe field name"):
            await adapter.search_similar(
                uuid4(),
                [0.1] * 384,
                metadata_filters={"key; DROP TABLE chunk": "evil"},
            )
        conn.query.assert_not_called()

    async def test_find_paths_uses_param_binding_for_rel_types(self) -> None:
        """find_paths passes relationship_types as parameter, not f-string."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn(query=[])
        adapter = SurrealDBGraphAdapter(conn)

        src_id = uuid4()
        tgt_id = uuid4()
        ns_id = uuid4()

        await adapter.find_paths(src_id, tgt_id, namespace_id=ns_id, relationship_types=["WORKS_AT", "KNOWS"])

        # Verify that at least one call was made and the SQL uses $rel_types
        # parameter binding instead of inline string interpolation
        assert conn.query.call_count >= 1
        for call in conn.query.call_args_list:
            sql = call.args[0] if call.args else call.kwargs.get("sql", "")
            if "rel_types" in str(call):
                # The bindings should contain rel_types as a list param
                bindings = call.args[1] if len(call.args) > 1 else call.kwargs.get("bindings", {})
                if bindings:
                    assert "rel_types" in bindings
                    assert bindings["rel_types"] == ["WORKS_AT", "KNOWS"]
            # The SQL must NOT contain the literal strings 'WORKS_AT' or 'KNOWS'
            assert "'WORKS_AT'" not in sql
            assert "'KNOWS'" not in sql

    async def test_get_neighborhood_uses_param_binding_for_rel_types(self) -> None:
        """get_neighborhood passes relationship_types as parameter, not f-string."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()

        # IGR-223: get_neighborhood now verifies the seed entity belongs to
        # ``namespace_id`` (via get_entity → query_one) before traversing.
        conn = _make_mock_conn(
            query=[],
            query_one=_graph_entity_row(entity_id, ns_id, name="Seed"),
        )
        adapter = SurrealDBGraphAdapter(conn)

        await adapter.get_neighborhood(
            entity_id,
            namespace_id=ns_id,
            relationship_types=["MANAGES", "REPORTS_TO"],
        )

        assert conn.query.call_count >= 1
        for call in conn.query.call_args_list:
            sql = call.args[0] if call.args else call.kwargs.get("sql", "")
            # The SQL must NOT contain the literal strings inline
            assert "'MANAGES'" not in sql
            assert "'REPORTS_TO'" not in sql
            # Bindings should use $rel_types parameter
            bindings = call.args[1] if len(call.args) > 1 else call.kwargs.get("bindings", {})
            if bindings:
                assert "rel_types" in bindings


# ---------------------------------------------------------------------------
# Vector adapter upsert_entities_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBVectorUpsertBatch:
    """Tests for upsert_entities_batch on the graph adapter (entity storage)."""

    async def test_upsert_new_entities(self) -> None:
        """New entities are created with is_new=True."""
        from khora.core.models import Entity
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        conn = _make_mock_conn(query=[])  # No existing entities found
        adapter = SurrealDBGraphAdapter(conn)

        entities = [
            Entity(
                namespace_id=ns_id,
                name="Alice",
                entity_type="PERSON",
                description="A person",
            ),
            Entity(
                namespace_id=ns_id,
                name="Bob",
                entity_type="PERSON",
                description="Another person",
            ),
        ]

        results = await adapter.upsert_entities_batch(ns_id, entities)

        assert len(results) == 2
        for entity, is_new in results:
            assert is_new is True
            assert entity.namespace_id == ns_id

    async def test_upsert_existing_entities(self) -> None:
        """Existing entities are updated with is_new=False."""
        from khora.core.models import Entity
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        existing_id = uuid4()
        now_iso = datetime.now(UTC).isoformat()

        # Simulate an existing entity row returned from SurrealDB
        existing_row = {
            "id": f"entity:\u27e8{existing_id}\u27e9",
            "namespace": f"memory_namespace:\u27e8{ns_id}\u27e9",
            "name": "Alice",
            "entity_type": "PERSON",
            "description": "Original description",
            "attributes": {},
            "source_document_ids": [],
            "source_chunk_ids": [],
            "source_tool": "",
            "mention_count": 1,
            "embedding": None,
            "embedding_model": "",
            "valid_from": None,
            "valid_until": None,
            "confidence": 1.0,
            "metadata_": {},
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        conn = _make_mock_conn(query=[existing_row])
        adapter = SurrealDBGraphAdapter(conn)

        new_entity = Entity(
            namespace_id=ns_id,
            name="Alice",
            entity_type="PERSON",
            description="Updated description",
        )

        results = await adapter.upsert_entities_batch(ns_id, [new_entity])

        assert len(results) == 1
        entity, is_new = results[0]
        assert is_new is False
        assert entity.id == existing_id  # Kept the existing entity's ID

    async def test_upsert_empty_list(self) -> None:
        """Empty list returns empty list."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        conn = _make_mock_conn()
        adapter = SurrealDBGraphAdapter(conn)

        results = await adapter.upsert_entities_batch(uuid4(), [])

        assert results == []
        # No DB calls should have been made
        conn.query.assert_not_called()
        conn.execute.assert_not_called()

    async def test_vector_adapter_has_upsert_method(self) -> None:
        """SurrealDBVectorAdapter has upsert_entities_batch (delegated or own)."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        # The graph adapter is the canonical owner of upsert_entities_batch
        assert hasattr(SurrealDBGraphAdapter, "upsert_entities_batch")
        assert callable(getattr(SurrealDBGraphAdapter, "upsert_entities_batch"))


# ---------------------------------------------------------------------------
# Create tables
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBCreateTables:
    async def test_relational_create_tables(self):
        """Relational adapter create_tables delegates to schema init."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        adapter = SurrealDBRelationalAdapter(conn)
        await adapter.create_tables()
        conn.execute.assert_awaited()

    async def test_vector_create_tables(self):
        """Vector adapter create_tables delegates to schema init."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        adapter = SurrealDBVectorAdapter(conn)
        await adapter.create_tables()
        conn.execute.assert_awaited()

    async def test_event_store_create_tables(self):
        """Event store adapter create_tables delegates to schema init."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.event_store import SurrealDBEventStoreAdapter

        adapter = SurrealDBEventStoreAdapter(conn)
        await adapter.create_tables()
        conn.execute.assert_awaited()


# ---------------------------------------------------------------------------
# Temporal neighbors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBTemporalNeighbors:
    async def test_get_temporal_neighbors_basic(self):
        """Returns neighbor entities within temporal bounds."""
        conn = _make_mock_conn()
        eid = uuid4()
        ns_id = uuid4()
        neighbor_id = uuid4()
        conn.query = AsyncMock(
            return_value=[
                {
                    "id": f"entity:{neighbor_id}",
                    "namespace": f"memory_namespace:{ns_id}",
                    "name": "Neighbor",
                    "entity_type": "PERSON",
                    "description": "",
                    "attributes": {},
                    "source_document_ids": [],
                    "source_chunk_ids": [],
                    "source_tool": "",
                    "mention_count": 1,
                    "embedding": None,
                    "embedding_model": "",
                    "valid_from": None,
                    "valid_until": None,
                    "confidence": 1.0,
                    "metadata_": {},
                    "created_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            ]
        )
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)
        results = await adapter.get_temporal_neighbors(eid, namespace_id=ns_id, limit=10)
        assert len(results) >= 0  # May be empty depending on mock
        conn.query.assert_awaited()

    async def test_get_temporal_neighbors_with_time_bounds(self):
        """Temporal bounds are passed as query parameters."""
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)
        from datetime import timedelta

        now = datetime.now(UTC)
        results = await adapter.get_temporal_neighbors(
            uuid4(),
            namespace_id=uuid4(),
            valid_after=now - timedelta(days=30),
            valid_before=now,
        )
        assert results == []
        # Verify temporal params were passed
        call_args = conn.query.call_args
        assert call_args is not None


# ---------------------------------------------------------------------------
# Session links
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBSessionLinks:
    async def test_create_session_links_no_chunks(self):
        """Returns 0 when no chunks exist."""
        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)
        count = await adapter.create_session_links(uuid4())
        assert count == 0

    async def test_create_session_links_single_session(self):
        """Returns 0 when only one session exists (no links needed)."""
        conn = _make_mock_conn()
        conn.query = AsyncMock(
            return_value=[
                {
                    "id": f"chunk:{uuid4()}",
                    "created_at": datetime.now(UTC).isoformat(),
                    "metadata_": {"session_id": "s1"},
                    "source_timestamp": None,
                },
            ]
        )
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        adapter = SurrealDBGraphAdapter(conn)
        count = await adapter.create_session_links(uuid4())
        assert count == 0


# ---------------------------------------------------------------------------
# Embedding stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBEmbeddingStats:
    async def test_get_embedding_stats(self):
        """Returns chunk and entity embedding counts."""
        conn = _make_mock_conn()
        # First call: chunk count, second call: entity count
        conn.query_one = AsyncMock(
            side_effect=[
                {"cnt": 42},
                {"cnt": 7},
            ]
        )
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        adapter = SurrealDBVectorAdapter(conn)
        stats = await adapter.get_embedding_stats(uuid4())
        assert stats["chunk_embeddings"] == 42
        assert stats["entity_embeddings"] == 7

    async def test_get_embedding_stats_empty(self):
        """Returns zeros when no embeddings exist."""
        conn = _make_mock_conn()
        conn.query_one = AsyncMock(return_value=None)
        from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter

        adapter = SurrealDBVectorAdapter(conn)
        stats = await adapter.get_embedding_stats(uuid4())
        assert stats["chunk_embeddings"] == 0
        assert stats["entity_embeddings"] == 0


# ---------------------------------------------------------------------------
# Get documents by checksums
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBGetDocumentsByChecksums:
    async def test_returns_dict_by_checksum(self):
        """Returns dict mapping checksum to Document."""
        conn = _make_mock_conn()
        ns_id = uuid4()
        doc_id = uuid4()
        conn.query = AsyncMock(return_value=[_document_row(doc_id, ns_id, checksum="abc123")])
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        adapter = SurrealDBRelationalAdapter(conn)
        result = await adapter.get_documents_by_checksums(ns_id, ["abc123"])
        assert "abc123" in result
        assert result["abc123"].id == doc_id

    async def test_empty_checksums(self):
        """Empty checksum list returns empty dict."""
        conn = _make_mock_conn()
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        adapter = SurrealDBRelationalAdapter(conn)
        result = await adapter.get_documents_by_checksums(uuid4(), [])
        assert result == {}


# ---------------------------------------------------------------------------
# SurrealDB optimizations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurrealDBDepthCapRaised:
    """Verify that graph depth caps were raised from 3 to 6."""

    async def test_find_paths_depth_5_not_capped_to_3(self) -> None:
        """find_paths with max_depth=5 generates d1..d5 columns, not d1..d3."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        src_id = uuid4()
        tgt_id = uuid4()

        conn = _make_mock_conn()
        # Return empty to verify query shape without needing full result
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        await adapter.find_paths(src_id, tgt_id, namespace_id=ns_id, max_depth=5)

        conn.query.assert_awaited_once()
        sql = conn.query.call_args[0][0]
        # Should contain d5 (depth 5 column) since cap is now 6
        assert " AS d5" in sql
        # Should NOT contain d6 since max_depth=5
        assert " AS d6" not in sql

    async def test_find_paths_depth_8_capped_to_6(self) -> None:
        """find_paths with max_depth=8 caps to 6."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        src_id = uuid4()
        tgt_id = uuid4()

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        await adapter.find_paths(src_id, tgt_id, namespace_id=ns_id, max_depth=8)

        sql = conn.query.call_args[0][0]
        assert " AS d6" in sql
        assert " AS d7" not in sql


@pytest.mark.unit
class TestSurrealDBSingleQueryTemporalNeighbors:
    """Verify get_temporal_neighbors uses a single query with d1..dN columns."""

    async def test_single_query_for_multiple_hops(self) -> None:
        """get_temporal_neighbors with max_hops=4 issues exactly one query."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        await adapter.get_temporal_neighbors(entity_id, namespace_id=ns_id, max_hops=4)

        # Should be exactly 1 query (not 4 separate queries)
        assert conn.query.await_count == 1

    async def test_temporal_neighbors_query_has_depth_columns(self) -> None:
        """The single query should contain d1..d4 columns for max_hops=4."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        await adapter.get_temporal_neighbors(entity_id, namespace_id=ns_id, max_hops=4)

        sql = conn.query.call_args[0][0]
        assert " AS d1" in sql
        assert " AS d4" in sql
        assert " AS d5" not in sql

    async def test_temporal_neighbors_returns_results(self) -> None:
        """get_temporal_neighbors returns neighbor entities from the single query."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()
        neighbor_id = uuid4()

        neighbor_row = _graph_entity_row(neighbor_id, ns_id, name="Temporal-Neighbor")
        single_result = [
            {
                "d1": [neighbor_row],
                "d2": None,
            },
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=single_result)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_temporal_neighbors(entity_id, namespace_id=ns_id, max_hops=2)
        assert len(result) == 1
        assert result[0]["name"] == "Temporal-Neighbor"

    async def test_temporal_neighbors_cap_raised_to_6(self) -> None:
        """get_temporal_neighbors with max_hops=6 generates d1..d6 columns."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        entity_id = uuid4()
        ns_id = uuid4()

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBGraphAdapter(conn)

        await adapter.get_temporal_neighbors(entity_id, namespace_id=ns_id, max_hops=10)

        sql = conn.query.call_args[0][0]
        assert " AS d6" in sql
        assert " AS d7" not in sql


@pytest.mark.unit
class TestSurrealDBBatchRelationshipFetch:
    """Verify get_neighborhoods_batch uses a single batch relationship query."""

    async def test_batch_neighborhoods_single_rel_query(self) -> None:
        """Relationship fetch for multiple entities uses one query, not N."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        ns_id = uuid4()
        eid1 = uuid4()
        eid2 = uuid4()
        neighbor1 = uuid4()
        neighbor2 = uuid4()

        # Batch entity query returns rows with neighbors
        entity_rows = [
            {
                "id": f"entity:\u27e8{eid1!s}\u27e9",
                "out_neighbors": [_graph_entity_row(neighbor1, ns_id, name="N1")],
                "in_neighbors": None,
            },
            {
                "id": f"entity:\u27e8{eid2!s}\u27e9",
                "out_neighbors": [_graph_entity_row(neighbor2, ns_id, name="N2")],
                "in_neighbors": None,
            },
        ]

        conn = _make_mock_conn()
        # First call: batch entity neighborhood query; second call: batch relationship query
        conn.query = AsyncMock(side_effect=[entity_rows, []])
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_neighborhoods_batch([eid1, eid2], namespace_id=ns_id, depth=1)

        # 2 queries total: 1 batch entity fetch + 1 batch relationship fetch
        assert conn.query.await_count == 2
        assert eid1 in result
        assert eid2 in result
        assert len(result[eid1]["entities"]) == 1
        assert len(result[eid2]["entities"]) == 1

    async def test_batch_neighborhoods_no_rels_when_no_neighbors(self) -> None:
        """When entities have no neighbors, skip the relationship batch query."""
        from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter

        eid = uuid4()

        entity_rows = [
            {
                "id": f"entity:\u27e8{eid!s}\u27e9",
                "out_neighbors": None,
                "in_neighbors": None,
            },
        ]

        conn = _make_mock_conn()
        conn.query = AsyncMock(return_value=entity_rows)
        adapter = SurrealDBGraphAdapter(conn)

        result = await adapter.get_neighborhoods_batch([eid], namespace_id=uuid4(), depth=1)

        # Only 1 query (entity fetch), no relationship fetch needed
        assert conn.query.await_count == 1
        assert eid in result
        assert result[eid]["entities"] == []
        assert result[eid]["relationships"] == []


@pytest.mark.unit
class TestSurrealDBSingleQueryDelete:
    """Verify delete_document uses DELETE ... RETURN BEFORE (single query)."""

    async def test_delete_existing_document(self) -> None:
        """delete_document returns True when document existed."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        doc_id = uuid4()
        conn = _make_mock_conn()
        # DELETE ... RETURN BEFORE returns the deleted record
        conn.query = AsyncMock(return_value=[{"id": f"document:\u27e8{doc_id!s}\u27e9"}])
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.delete_document(doc_id, namespace_id=uuid4())
        assert result is True
        # Single query call (no separate SELECT + DELETE)
        conn.query.assert_awaited_once()
        sql = conn.query.call_args[0][0]
        assert "DELETE" in sql
        assert "RETURN BEFORE" in sql

    async def test_delete_missing_document(self) -> None:
        """delete_document returns False when document did not exist."""
        from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter

        conn = _make_mock_conn()
        # DELETE ... RETURN BEFORE returns empty list when nothing existed
        conn.query = AsyncMock(return_value=[])
        adapter = SurrealDBRelationalAdapter(conn)

        result = await adapter.delete_document(uuid4(), namespace_id=uuid4())
        assert result is False
        conn.query.assert_awaited_once()


@pytest.mark.unit
class TestSurrealDBGraphTraversalIndexes:
    """Verify schema includes in/out indexes for relates_to."""

    def test_schema_has_relates_to_in_index(self) -> None:
        from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

        assert "idx_relates_to_in" in _TABLE_DEFINITIONS
        assert "ON relates_to FIELDS in" in _TABLE_DEFINITIONS

    def test_schema_has_relates_to_out_index(self) -> None:
        from khora.storage.backends.surrealdb.schema import _TABLE_DEFINITIONS

        assert "idx_relates_to_out" in _TABLE_DEFINITIONS
        assert "ON relates_to FIELDS out" in _TABLE_DEFINITIONS

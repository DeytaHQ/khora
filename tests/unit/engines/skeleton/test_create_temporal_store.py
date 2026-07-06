"""Coverage: ``khora.storage.temporal.create_temporal_store``.

Pins the dispatch table for the supported backends (pgvector, weaviate,
turbopuffer, surrealdb, sqlite_lance) plus the validation errors. Each branch
is covered without requiring the backend's optional dependencies — we mock the
lazy imports so the tests run in any environment. The weaviate / turbopuffer
branches read their connection details from ``config.storage.weaviate`` /
``config.storage.turbopuffer``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from khora.storage.temporal import (
    TemporalChunk,
    TemporalFilter,
    TemporalSearchResult,
    create_temporal_store,
)


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock KhoraConfig — opaque to ``create_temporal_store`` callers."""
    return MagicMock()


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict[str, object]) -> None:
    """Install a stub module in ``sys.modules``. Cleaned up on test teardown via monkeypatch."""
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def stub_pgvector_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the pgvector backend module so the import inside the dispatch works."""
    instance = MagicMock(name="PgVectorTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    _install_module(
        monkeypatch,
        "khora.storage.temporal.pgvector",
        {"PgVectorTemporalStore": cls},
    )
    return cls


@pytest.fixture
def stub_weaviate_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="WeaviateTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    # The factory also imports WeaviateBackendConfig from the same module to
    # translate config.storage.weaviate into the backend config. Capture the
    # kwargs it is built with so the test can assert the values flowed through.
    backend_cfg_cls = MagicMock(name="WeaviateBackendConfig")
    _install_module(
        monkeypatch,
        "khora.storage.temporal.weaviate",
        {"WeaviateTemporalStore": cls, "WeaviateBackendConfig": backend_cfg_cls},
    )
    cls.backend_config_cls = backend_cfg_cls  # type: ignore[attr-defined]
    return cls


@pytest.fixture
def stub_turbopuffer_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="TurbopufferTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    backend_cfg_cls = MagicMock(name="TurbopufferBackendConfig")
    _install_module(
        monkeypatch,
        "khora.storage.temporal.turbopuffer",
        {"TurbopufferTemporalStore": cls, "TurbopufferBackendConfig": backend_cfg_cls},
    )
    cls.backend_config_cls = backend_cfg_cls  # type: ignore[attr-defined]
    return cls


@pytest.fixture
def stub_surrealdb_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="SurrealDBTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    _install_module(
        monkeypatch,
        "khora.storage.temporal.surrealdb",
        {"SurrealDBTemporalStore": cls},
    )
    return cls


@pytest.fixture
def stub_sqlite_lance_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="SQLiteLanceTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    _install_module(
        monkeypatch,
        "khora.storage.temporal.sqlite_lance",
        {"SQLiteLanceTemporalStore": cls},
    )
    return cls


class TestPgVectorDispatch:
    def test_returns_pgvector_store_with_engine(self, mock_config: MagicMock, stub_pgvector_store: MagicMock) -> None:
        engine_sentinel = object()
        store = create_temporal_store("pgvector", mock_config, engine=engine_sentinel)
        stub_pgvector_store.assert_called_once_with(mock_config, engine=engine_sentinel)
        assert store is stub_pgvector_store.return_value

    def test_engine_defaults_to_none(self, mock_config: MagicMock, stub_pgvector_store: MagicMock) -> None:
        create_temporal_store("pgvector", mock_config)
        _, kwargs = stub_pgvector_store.call_args
        assert kwargs["engine"] is None


class TestWeaviateDispatch:
    def test_requires_config(self, mock_config: MagicMock) -> None:
        mock_config.storage.weaviate = None
        with pytest.raises(ValueError, match="requires storage.weaviate config"):
            create_temporal_store("weaviate", mock_config)

    def test_passes_config_to_constructor(self, mock_config: MagicMock, stub_weaviate_store: MagicMock) -> None:
        from khora.config.schema import WeaviateConfig

        mock_config.storage.weaviate = WeaviateConfig(
            url="http://w:8080",
            grpc_port=50061,
            additional_headers={"X-OpenAI-Api-Key": "sk-..."},
        )
        create_temporal_store("weaviate", mock_config)

        # Backend config built from the unwrapped SecretStr URL + scalar fields.
        # additional_headers (code-settable only) threads through verbatim.
        backend_cfg_cls = stub_weaviate_store.backend_config_cls
        backend_cfg_cls.assert_called_once_with(
            url="http://w:8080",
            cluster_url=None,
            api_key=None,
            grpc_port=50061,
            http_secure=False,
            grpc_secure=False,
            additional_headers={"X-OpenAI-Api-Key": "sk-..."},
            skip_init_checks=False,
        )
        stub_weaviate_store.assert_called_once_with(mock_config, backend_cfg_cls.return_value)


class TestTurbopufferDispatch:
    def test_requires_config(self, mock_config: MagicMock) -> None:
        mock_config.storage.turbopuffer = None
        with pytest.raises(ValueError, match="requires storage.turbopuffer config"):
            create_temporal_store("turbopuffer", mock_config)

    def test_requires_api_key(self, mock_config: MagicMock) -> None:
        from khora.config.schema import TurbopufferConfig

        mock_config.storage.turbopuffer = TurbopufferConfig()  # api_key defaults None
        with pytest.raises(ValueError, match="requires storage.turbopuffer config with an api_key"):
            create_temporal_store("turbopuffer", mock_config)

    def test_passes_config_to_constructor(self, mock_config: MagicMock, stub_turbopuffer_store: MagicMock) -> None:
        from khora.config.schema import TurbopufferConfig

        mock_config.storage.turbopuffer = TurbopufferConfig(api_key="tpuf_key", region="gcp-europe-west3")
        create_temporal_store("turbopuffer", mock_config)

        backend_cfg_cls = stub_turbopuffer_store.backend_config_cls
        backend_cfg_cls.assert_called_once_with(
            api_key="tpuf_key",
            region="gcp-europe-west3",
            base_url=None,
            namespace_prefix="khora_",
            ann_distance_threshold=None,
        )
        stub_turbopuffer_store.assert_called_once_with(mock_config, backend_cfg_cls.return_value)


class TestSurrealDBDispatch:
    def test_passes_surrealdb_config(self, mock_config: MagicMock, stub_surrealdb_store: MagicMock) -> None:
        surreal_cfg = MagicMock()
        create_temporal_store("surrealdb", mock_config, surrealdb_config=surreal_cfg)
        stub_surrealdb_store.assert_called_once_with(mock_config, surrealdb_config=surreal_cfg, connection=None)

    def test_passes_shared_connection(self, mock_config: MagicMock, stub_surrealdb_store: MagicMock) -> None:
        """Skeleton must forward the coordinator's SurrealDBConnection (issue #718)."""
        surreal_cfg = MagicMock()
        shared_conn = MagicMock(name="shared-SurrealDBConnection")
        create_temporal_store(
            "surrealdb",
            mock_config,
            surrealdb_config=surreal_cfg,
            surrealdb_connection=shared_conn,
        )
        stub_surrealdb_store.assert_called_once_with(mock_config, surrealdb_config=surreal_cfg, connection=shared_conn)


class TestSQLiteLanceDispatch:
    def test_requires_handle(self, mock_config: MagicMock) -> None:
        with pytest.raises(ValueError, match="sqlite_lance_handle is required"):
            create_temporal_store("sqlite_lance", mock_config)

    def test_passes_handle(self, mock_config: MagicMock, stub_sqlite_lance_store: MagicMock) -> None:
        handle = MagicMock(name="EmbeddedStorageHandle")
        store = create_temporal_store("sqlite_lance", mock_config, sqlite_lance_handle=handle)
        stub_sqlite_lance_store.assert_called_once_with(handle)
        assert store is stub_sqlite_lance_store.return_value


class TestUnknownBackend:
    def test_raises_value_error(self, mock_config: MagicMock) -> None:
        with pytest.raises(ValueError, match="Unknown backend: bogus"):
            create_temporal_store("bogus", mock_config)


# ---------------------------------------------------------------------------
# Lightweight dataclass coverage — TemporalChunk / TemporalFilter / TemporalSearchResult
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    def test_temporal_chunk_defaults(self) -> None:
        from uuid import uuid4

        chunk = TemporalChunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="x")
        assert chunk.embedding is None
        assert chunk.tags == []
        assert chunk.metadata == {}
        assert chunk.confidence == 1.0

    def test_temporal_filter_defaults(self) -> None:
        tf = TemporalFilter()
        assert tf.occurred_after is None
        assert tf.occurred_before is None
        assert tf.tags is None
        assert tf.additional == {}

    def test_temporal_search_result(self) -> None:
        from uuid import uuid4

        chunk = TemporalChunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="x")
        result = TemporalSearchResult(chunk=chunk, similarity=0.9)
        assert result.bm25_score is None
        assert result.combined_score is None

"""Coverage: ``khora.engines.skeleton.backends.create_temporal_store``.

Pins the dispatch table for the four supported backends (pgvector, weaviate,
surrealdb, sqlite_lance) plus the validation errors. Each branch is covered
without requiring the backend's optional dependencies — we mock the lazy
imports so the tests run in any environment.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from khora.engines.skeleton.backends import (
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
        "khora.engines.skeleton.backends.pgvector",
        {"PgVectorTemporalStore": cls},
    )
    return cls


@pytest.fixture
def stub_weaviate_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="WeaviateTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    _install_module(
        monkeypatch,
        "khora.engines.skeleton.backends.weaviate",
        {"WeaviateTemporalStore": cls},
    )
    return cls


@pytest.fixture
def stub_surrealdb_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="SurrealDBTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    _install_module(
        monkeypatch,
        "khora.engines.skeleton.backends.surrealdb",
        {"SurrealDBTemporalStore": cls},
    )
    return cls


@pytest.fixture
def stub_sqlite_lance_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    instance = MagicMock(name="SQLiteLanceTemporalStore-instance")
    cls = MagicMock(return_value=instance)
    _install_module(
        monkeypatch,
        "khora.engines.skeleton.backends.sqlite_lance",
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
    def test_requires_url(self, mock_config: MagicMock) -> None:
        with pytest.raises(ValueError, match="weaviate_url is required"):
            create_temporal_store("weaviate", mock_config)

    def test_passes_url_to_constructor(self, mock_config: MagicMock, stub_weaviate_store: MagicMock) -> None:
        create_temporal_store("weaviate", mock_config, weaviate_url="http://w:8080")
        stub_weaviate_store.assert_called_once_with(mock_config, "http://w:8080")


class TestSurrealDBDispatch:
    def test_passes_surrealdb_config(self, mock_config: MagicMock, stub_surrealdb_store: MagicMock) -> None:
        surreal_cfg = MagicMock()
        create_temporal_store("surrealdb", mock_config, surrealdb_config=surreal_cfg)
        stub_surrealdb_store.assert_called_once_with(mock_config, surrealdb_config=surreal_cfg)


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

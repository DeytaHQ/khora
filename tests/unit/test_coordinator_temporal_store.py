"""Unit tests for StorageCoordinator.temporal_store() backend wiring.

The coordinator's ``temporal_store()`` is a factory that delegates backend
selection to ``khora.storage.temporal.create_temporal_store`` (imported lazily
at call time) and only gathers the per-backend shared resource so the temporal
store reuses the coordinator's existing connections. These tests assert the
kwargs forwarded for each backend, the error guards, and the factory semantics
(connected once, never cached).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.config import KhoraConfig
from khora.storage.coordinator import StorageCoordinator


@pytest.fixture
def config() -> KhoraConfig:
    """A real KhoraConfig (offline, no DB)."""
    return KhoraConfig(app_name="khora-test")


def _make_store() -> MagicMock:
    """A mock temporal store whose connect() is an awaitable."""
    store = MagicMock(name="temporal_store")
    store.connect = AsyncMock()
    return store


def _coordinator(*, vector=None, relational=None) -> StorageCoordinator:
    """Build a coordinator with the given (raw) adapters.

    Passing ``vector=`` / ``relational=`` routes through ``__setattr__`` which
    sets ``_vector`` / ``_relational`` to the raw object, which is what
    ``temporal_store`` reads via ``getattr(self._vector, "_engine", ...)``.
    """
    return StorageCoordinator(relational=relational, vector=vector)


class TestTemporalStorePgvector:
    """pgvector backend shares the coordinator's SQLAlchemy engine."""

    @pytest.mark.asyncio
    async def test_engine_from_vector_adapter(self, config: KhoraConfig) -> None:
        """engine kwarg is taken from the vector adapter's _engine."""
        sentinel_engine = MagicMock(name="vector_engine")
        vec = MagicMock()
        vec._engine = sentinel_engine
        rel = MagicMock()
        rel._engine = MagicMock(name="rel_engine")
        coord = _coordinator(vector=vec, relational=rel)
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            result = await coord.temporal_store("pgvector", config)

        assert result is store
        _, kwargs = factory.call_args
        assert kwargs["engine"] is sentinel_engine

    @pytest.mark.asyncio
    async def test_engine_falls_back_to_relational_when_vector_none(self, config: KhoraConfig) -> None:
        """With no vector adapter, engine falls back to relational._engine."""
        rel_engine = MagicMock(name="rel_engine")
        rel = MagicMock()
        rel._engine = rel_engine
        coord = _coordinator(vector=None, relational=rel)
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            await coord.temporal_store("pgvector", config)

        _, kwargs = factory.call_args
        assert kwargs["engine"] is rel_engine

    @pytest.mark.asyncio
    async def test_engine_falls_back_when_vector_engine_none(self, config: KhoraConfig) -> None:
        """A vector adapter whose _engine is None still falls back to relational."""
        rel_engine = MagicMock(name="rel_engine")
        vec = MagicMock()
        vec._engine = None
        rel = MagicMock()
        rel._engine = rel_engine
        coord = _coordinator(vector=vec, relational=rel)
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            await coord.temporal_store("pgvector", config)

        _, kwargs = factory.call_args
        assert kwargs["engine"] is rel_engine


class TestTemporalStoreSurrealDB:
    """surrealdb backend shares the coordinator's SurrealDBConnection."""

    @pytest.mark.asyncio
    async def test_connection_and_config_forwarded(self, config: KhoraConfig) -> None:
        """surrealdb_connection from relational._conn, surrealdb_config from config."""
        conn = MagicMock(name="surreal_conn")
        rel = MagicMock()
        rel._conn = conn
        coord = _coordinator(relational=rel)
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            await coord.temporal_store("surrealdb", config)

        _, kwargs = factory.call_args
        assert kwargs["surrealdb_connection"] is conn
        assert kwargs["surrealdb_config"] is config.storage.surrealdb


class TestTemporalStoreSqliteLance:
    """sqlite_lance backend shares the vector adapter's EmbeddedStorageHandle."""

    @pytest.mark.asyncio
    async def test_handle_from_vector_adapter(self, config: KhoraConfig) -> None:
        """sqlite_lance_handle is taken from the vector adapter's _handle."""
        handle = MagicMock(name="handle")
        vec = MagicMock()
        vec._handle = handle
        coord = _coordinator(vector=vec)
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            await coord.temporal_store("sqlite_lance", config)

        _, kwargs = factory.call_args
        assert kwargs["sqlite_lance_handle"] is handle

    @pytest.mark.asyncio
    async def test_raises_when_no_vector_adapter(self, config: KhoraConfig) -> None:
        """sqlite_lance without a vector adapter raises RuntimeError."""
        coord = _coordinator(vector=None)

        with patch("khora.storage.temporal.create_temporal_store") as factory:
            with pytest.raises(
                RuntimeError,
                match="sqlite_lance backend requires a vector adapter on the coordinator",
            ):
                await coord.temporal_store("sqlite_lance", config)
        factory.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_handle_none(self, config: KhoraConfig) -> None:
        """sqlite_lance with a handle-less vector adapter raises RuntimeError."""
        vec = MagicMock()
        vec._handle = None
        coord = _coordinator(vector=vec)

        with patch("khora.storage.temporal.create_temporal_store") as factory:
            with pytest.raises(
                RuntimeError,
                match="sqlite_lance vector adapter is missing its EmbeddedStorageHandle",
            ):
                await coord.temporal_store("sqlite_lance", config)
        factory.assert_not_called()


class TestTemporalStoreConfigDriven:
    """weaviate / turbopuffer read their config off ``config.storage.*``.

    The coordinator no longer accepts ``weaviate_url`` / ``turbopuffer_config``
    kwargs — it forwards ``backend`` + ``config`` and the factory reads the
    backend config from ``config.storage.weaviate`` / ``config.storage.turbopuffer``.
    """

    @pytest.mark.asyncio
    async def test_weaviate_no_vendor_kwargs_forwarded(self, config: KhoraConfig) -> None:
        """The coordinator passes config through; no vendor kwargs are sent."""
        from khora.config.schema import WeaviateConfig

        config.storage.weaviate = WeaviateConfig(url="http://w:8080")
        coord = _coordinator()
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            await coord.temporal_store("weaviate", config)

        args, kwargs = factory.call_args
        assert args[0] == "weaviate"
        assert args[1] is config
        assert "weaviate_url" not in kwargs
        assert "turbopuffer_config" not in kwargs
        # The factory will read config.storage.weaviate itself.
        assert config.storage.weaviate.url.get_secret_value() == "http://w:8080"

    @pytest.mark.asyncio
    async def test_turbopuffer_no_vendor_kwargs_forwarded(self, config: KhoraConfig) -> None:
        """The coordinator passes config through; no vendor kwargs are sent."""
        from khora.config.schema import TurbopufferConfig

        config.storage.turbopuffer = TurbopufferConfig(api_key="tpuf_key")
        coord = _coordinator()
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store) as factory:
            await coord.temporal_store("turbopuffer", config)

        args, kwargs = factory.call_args
        assert args[0] == "turbopuffer"
        assert args[1] is config
        assert "weaviate_url" not in kwargs
        assert "turbopuffer_config" not in kwargs
        assert config.storage.turbopuffer.api_key.get_secret_value() == "tpuf_key"


class TestTemporalStoreFactorySemantics:
    """The store is connected once and never cached on the coordinator."""

    @pytest.mark.asyncio
    async def test_returns_connected_store(self, config: KhoraConfig) -> None:
        """The returned store is the factory's, with connect() awaited once."""
        coord = _coordinator()
        store = _make_store()

        with patch("khora.storage.temporal.create_temporal_store", return_value=store):
            result = await coord.temporal_store("weaviate", config)

        assert result is store
        store.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_cached_two_calls_two_stores(self, config: KhoraConfig) -> None:
        """Two calls invoke the factory twice and return distinct stores."""
        coord = _coordinator()
        first, second = _make_store(), _make_store()

        with patch(
            "khora.storage.temporal.create_temporal_store",
            side_effect=[first, second],
        ) as factory:
            r1 = await coord.temporal_store("weaviate", config)
            r2 = await coord.temporal_store("weaviate", config)

        assert factory.call_count == 2
        assert r1 is first
        assert r2 is second
        assert r1 is not r2

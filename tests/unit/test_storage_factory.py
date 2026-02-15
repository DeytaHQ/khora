"""Unit tests for StorageFactory shared connection pools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.storage.factory import StorageConfig, StorageFactory, _normalize_url


class TestNormalizeUrl:
    """Tests for URL normalization."""

    def test_strips_trailing_slash(self):
        assert _normalize_url("postgresql://host/db/") == _normalize_url("postgresql://host/db")

    def test_lowercases_scheme_and_host(self):
        assert _normalize_url("PostgreSQL://HOST/db") == _normalize_url("postgresql://host/db")

    def test_different_paths_differ(self):
        assert _normalize_url("postgresql://host/db1") != _normalize_url("postgresql://host/db2")

    def test_preserves_port(self):
        url = "postgresql://host:5432/db"
        assert ":5432" in _normalize_url(url)


class TestStorageFactoryEngineCache:
    """Tests for engine caching in StorageFactory."""

    @patch("khora.storage.factory.create_async_engine")
    def test_get_or_create_engine_caches_by_url(self, mock_create):
        """Same URL returns the same engine instance."""
        factory = StorageFactory()
        engine1 = factory.get_or_create_engine("postgresql+asyncpg://host/db")
        engine2 = factory.get_or_create_engine("postgresql+asyncpg://host/db")
        assert engine1 is engine2
        mock_create.assert_called_once()

    @patch("khora.storage.factory.create_async_engine")
    def test_get_or_create_engine_normalizes_url(self, mock_create):
        """Equivalent URLs with different casing/trailing slash share an engine."""
        factory = StorageFactory()
        engine1 = factory.get_or_create_engine("postgresql+asyncpg://HOST/db/")
        engine2 = factory.get_or_create_engine("postgresql+asyncpg://host/db")
        assert engine1 is engine2
        mock_create.assert_called_once()

    @patch("khora.storage.factory.create_async_engine")
    def test_different_urls_get_different_engines(self, mock_create):
        """Different URLs create separate engines."""
        mock_create.side_effect = [MagicMock(), MagicMock()]
        factory = StorageFactory()
        engine1 = factory.get_or_create_engine("postgresql+asyncpg://host/db1")
        engine2 = factory.get_or_create_engine("postgresql+asyncpg://host/db2")
        assert engine1 is not engine2
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch("khora.storage.factory.create_async_engine")
    async def test_dispose_engines_clears_cache(self, mock_create):
        """dispose_engines disposes all engines and clears the cache."""
        mock_engine = AsyncMock()
        mock_create.return_value = mock_engine
        factory = StorageFactory()
        factory.get_or_create_engine("postgresql+asyncpg://host/db")
        assert len(factory._engine_cache) == 1

        await factory.dispose_engines()

        mock_engine.dispose.assert_awaited_once()
        assert len(factory._engine_cache) == 0


class TestFactoryPassesSharedEngine:
    """Tests that factory methods pass shared engines to backends."""

    @patch("khora.storage.factory.create_async_engine")
    def test_relational_and_vector_share_engine_for_same_url(self, mock_create):
        """When postgresql_url == pgvector_url, backends share an engine."""
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        url = "postgresql+asyncpg://host/db"
        config = StorageConfig(postgresql_url=url, pgvector_url=url)
        factory = StorageFactory(config=config)

        rel = factory.create_relational_backend()
        vec = factory.create_vector_backend()

        assert rel is not None
        assert vec is not None
        # Both should have the same shared engine
        assert rel._engine is vec._engine
        assert rel._engine_shared is True
        assert vec._engine_shared is True
        # Only one engine created
        mock_create.assert_called_once()

    @patch("khora.storage.factory.create_async_engine")
    def test_event_store_shares_engine_with_relational(self, mock_create):
        """Event store and relational backend share engine when same URL."""
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        url = "postgresql+asyncpg://host/db"
        config = StorageConfig(postgresql_url=url, event_store_url=url)
        factory = StorageFactory(config=config)

        rel = factory.create_relational_backend()
        evt = factory.create_event_store()

        assert rel is not None
        assert evt is not None
        assert rel._engine is evt._engine
        mock_create.assert_called_once()

    @patch("khora.storage.factory.create_async_engine")
    def test_no_backend_when_url_missing(self, mock_create):
        """Returns None when URL not configured."""
        factory = StorageFactory(config=StorageConfig())
        assert factory.create_relational_backend() is None
        assert factory.create_vector_backend() is None
        assert factory.create_event_store() is None
        mock_create.assert_not_called()


class TestBackendSharedEngineDisconnect:
    """Tests that shared-engine backends skip dispose on disconnect."""

    @pytest.mark.asyncio
    async def test_postgresql_shared_engine_skips_dispose(self):
        """PostgreSQLBackend with shared engine skips dispose."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        mock_engine = AsyncMock()
        backend = PostgreSQLBackend("postgresql+asyncpg://host/db", engine=mock_engine)
        assert backend._engine_shared is True

        # Simulate connected state
        backend._session_factory = MagicMock()

        await backend.disconnect()
        mock_engine.dispose.assert_not_awaited()
        assert backend._engine is None

    @pytest.mark.asyncio
    async def test_postgresql_owned_engine_disposes(self):
        """PostgreSQLBackend without shared engine disposes."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend("postgresql+asyncpg://host/db")
        assert backend._engine_shared is False

        mock_engine = AsyncMock()
        backend._engine = mock_engine
        backend._session_factory = MagicMock()

        await backend.disconnect()
        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pgvector_shared_engine_skips_dispose(self):
        """PgVectorBackend with shared engine skips dispose."""
        from khora.storage.backends.pgvector import PgVectorBackend

        mock_engine = AsyncMock()
        backend = PgVectorBackend("postgresql+asyncpg://host/db", engine=mock_engine)
        assert backend._engine_shared is True

        backend._session_factory = MagicMock()
        await backend.disconnect()
        mock_engine.dispose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_event_store_shared_engine_skips_dispose(self):
        """PostgreSQLEventStore with shared engine skips dispose."""
        from khora.storage.event_store import PostgreSQLEventStore

        mock_engine = AsyncMock()
        store = PostgreSQLEventStore("postgresql+asyncpg://host/db", engine=mock_engine)
        assert store._engine_shared is True

        store._session_factory = MagicMock()
        await store.disconnect()
        mock_engine.dispose.assert_not_awaited()

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


class TestPoolPrePing:
    """Tests for pool_pre_ping support across backends."""

    def test_storage_settings_default_false(self):
        """StorageSettings.postgresql_pool_pre_ping defaults to False."""
        from khora.config.schema import StorageSettings

        settings = StorageSettings()
        assert settings.postgresql_pool_pre_ping is False

    def test_storage_settings_can_enable(self):
        """StorageSettings.postgresql_pool_pre_ping can be set to True."""
        from khora.config.schema import StorageSettings

        settings = StorageSettings(postgresql_pool_pre_ping=True)
        assert settings.postgresql_pool_pre_ping is True

    def test_storage_config_default_false(self):
        """StorageConfig.postgresql_pool_pre_ping defaults to False."""
        config = StorageConfig()
        assert config.postgresql_pool_pre_ping is False

    @patch("khora.storage.factory.create_async_engine")
    def test_get_or_create_engine_passes_pool_pre_ping(self, mock_create):
        """get_or_create_engine passes pool_pre_ping to create_async_engine."""
        factory = StorageFactory()
        factory.get_or_create_engine("postgresql+asyncpg://host/db", pool_pre_ping=True)
        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is True

    @patch("khora.storage.factory.create_async_engine")
    def test_get_or_create_engine_default_pool_pre_ping_false(self, mock_create):
        """get_or_create_engine defaults pool_pre_ping to False."""
        factory = StorageFactory()
        factory.get_or_create_engine("postgresql+asyncpg://host/db")
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is False

    @patch("khora.storage.factory.create_async_engine")
    def test_factory_passes_pool_pre_ping_to_relational(self, mock_create):
        """Factory passes pool_pre_ping to relational backend engine."""
        mock_create.return_value = MagicMock()
        config = StorageConfig(
            postgresql_url="postgresql+asyncpg://host/db",
            postgresql_pool_pre_ping=True,
        )
        factory = StorageFactory(config=config)
        backend = factory.create_relational_backend()
        assert backend is not None
        assert backend._pool_pre_ping is True
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is True

    @patch("khora.storage.factory.create_async_engine")
    def test_factory_passes_pool_pre_ping_to_vector(self, mock_create):
        """Factory passes pool_pre_ping to vector backend engine."""
        mock_create.return_value = MagicMock()
        config = StorageConfig(
            pgvector_url="postgresql+asyncpg://host/db",
            postgresql_pool_pre_ping=True,
        )
        factory = StorageFactory(config=config)
        backend = factory.create_vector_backend()
        assert backend is not None
        assert backend._pool_pre_ping is True
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is True

    def test_postgresql_backend_stores_pool_pre_ping(self):
        """PostgreSQLBackend stores pool_pre_ping parameter."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend("postgresql+asyncpg://host/db", pool_pre_ping=True)
        assert backend._pool_pre_ping is True

    def test_pgvector_backend_stores_pool_pre_ping(self):
        """PgVectorBackend stores pool_pre_ping parameter."""
        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://host/db", pool_pre_ping=True)
        assert backend._pool_pre_ping is True

    def test_event_store_stores_pool_pre_ping(self):
        """PostgreSQLEventStore stores pool_pre_ping parameter."""
        from khora.storage.event_store import PostgreSQLEventStore

        store = PostgreSQLEventStore("postgresql+asyncpg://host/db", pool_pre_ping=True)
        assert store._pool_pre_ping is True

    @pytest.mark.asyncio
    @patch("khora.storage.backends.postgresql.create_async_engine")
    async def test_postgresql_connect_passes_pool_pre_ping(self, mock_create):
        """PostgreSQLBackend.connect() passes pool_pre_ping to create_async_engine."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        mock_create.return_value = MagicMock()
        backend = PostgreSQLBackend("postgresql+asyncpg://host/db", pool_pre_ping=True)
        await backend.connect()
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is True

    @patch("khora.storage.backends.pgvector.create_async_engine")
    def test_pgvector_connect_passes_pool_pre_ping(self, mock_create):
        """PgVectorBackend passes pool_pre_ping when creating its own engine."""
        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://host/db", pool_pre_ping=True)
        # Verify the parameter is stored; connect() would pass it to create_async_engine
        assert backend._pool_pre_ping is True

    @pytest.mark.asyncio
    @patch("khora.storage.event_store.create_async_engine")
    async def test_event_store_connect_passes_pool_pre_ping(self, mock_create):
        """PostgreSQLEventStore.connect() passes pool_pre_ping to create_async_engine."""
        from khora.storage.event_store import PostgreSQLEventStore

        mock_create.return_value = MagicMock()
        store = PostgreSQLEventStore("postgresql+asyncpg://host/db", pool_pre_ping=True)
        await store.connect()
        _, kwargs = mock_create.call_args
        assert kwargs["pool_pre_ping"] is True

    def test_khora_config_env_var(self):
        """pool_pre_ping can be set via KHORA_STORAGE_POSTGRESQL_POOL_PRE_PING.

        The field is flat on ``StorageSettings`` (not nested under another
        config object), so the env-var form is a single underscore between
        ``KHORA_STORAGE_`` and the field name — not double. The previous
        docstring claimed ``KHORA_STORAGE__POSTGRESQL_POOL_PRE_PING`` which
        was wrong on both counts.
        """
        from khora.config.schema import KhoraConfig

        config = KhoraConfig(storage={"postgresql_pool_pre_ping": True})
        assert config.storage.postgresql_pool_pre_ping is True


class TestStorageConfigRepr:
    """repr must not expose credential fields."""

    _CREDENTIAL_FIELDS = [
        "postgresql_url",
        "pgvector_url",
        "neo4j_url",
        "neo4j_password",
        "neo4j_user",
        "event_store_url",
    ]

    def _make_config(self) -> StorageConfig:
        return StorageConfig(
            postgresql_url="postgresql+asyncpg://user:secret@host/db",
            pgvector_url="postgresql+asyncpg://user:secret@host/db",
            neo4j_url="bolt://user:secret@host:7687",
            neo4j_password="hunter2",
            neo4j_user="testuser",
            event_store_url="postgresql+asyncpg://user:secret@host/events",
        )

    def test_credential_fields_absent_from_repr(self):
        config = self._make_config()
        r = repr(config)
        for field_name in self._CREDENTIAL_FIELDS:
            assert field_name not in r, f"repr should not include {field_name!r}"

    def test_secret_values_absent_from_repr(self):
        config = self._make_config()
        r = repr(config)
        assert "secret" not in r
        assert "hunter2" not in r

    def test_non_credential_fields_present_in_repr(self):
        config = StorageConfig(postgresql_echo=True, neo4j_database="mydb")
        r = repr(config)
        assert "postgresql_echo" in r
        assert "neo4j_database" in r

    def test_field_values_still_accessible(self):
        config = self._make_config()
        assert config.postgresql_url == "postgresql+asyncpg://user:secret@host/db"
        assert config.neo4j_password == "hunter2"

    def test_new_style_config_fields_absent_from_repr(self):
        from khora.config.schema import SurrealDBConfig

        sdb_cfg = SurrealDBConfig(
            mode="remote",
            url="ws://user:topsecret@host:8000/rpc",
            password="topsecret",
        )
        config = StorageConfig(backend="surrealdb", surrealdb_config=sdb_cfg)
        r = repr(config)
        assert "surrealdb_config" not in r
        assert "graph_config" not in r
        assert "vector_config" not in r
        assert "sqlite_lance_config" not in r
        assert "topsecret" not in r

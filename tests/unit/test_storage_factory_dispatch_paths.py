"""Coverage: ``StorageFactory.create_coordinator`` dispatch paths.

Pre-PR coverage of ``factory.py`` was 54%. The big uncovered surfaces are
the unified-backend branches: ``surrealdb``, ``sqlite_lance``, and
``sqlite``. These tests pin the dispatch contract and the early-fail
guards (missing config, missing optional dep) without requiring the
backend's drivers to be installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from khora.storage.factory import (
    StorageConfig,
    StorageFactory,
    _import_backend_class,
    create_storage_coordinator,
)


def _install(monkeypatch: pytest.MonkeyPatch, name: str, attrs: dict[str, object]) -> None:
    """Install a stub module via monkeypatch so it's cleaned up automatically."""
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


# ---------------------------------------------------------------------------
# StorageConfig.from_dict — was 0% covered (lines 110-128)
# ---------------------------------------------------------------------------


class TestStorageConfigFromDict:
    def test_full_dict_propagates_all_fields(self) -> None:
        d = {
            "storage": {
                "relational": {
                    "url": "postgresql://localhost/db",
                    "echo": True,
                    "pool_size": 7,
                    "max_overflow": 14,
                },
                "vector": {
                    "url": "postgresql://localhost/vec",
                    "embedding_dimension": 768,
                    "hnsw_ef_search": 200,
                },
                "graph": {
                    "url": "bolt://localhost:7687",
                    "user": "u",
                    "password": "p",
                    "database": "g",
                },
                "event_store": {"url": "postgresql://localhost/evt"},
            }
        }
        sc = StorageConfig.from_dict(d)
        assert sc.postgresql_url == "postgresql://localhost/db"
        assert sc.postgresql_echo is True
        assert sc.postgresql_pool_size == 7
        assert sc.postgresql_max_overflow == 14
        assert sc.pgvector_url == "postgresql://localhost/vec"
        assert sc.pgvector_embedding_dimension == 768
        assert sc.pgvector_hnsw_ef_search == 200
        assert sc.neo4j_url == "bolt://localhost:7687"
        assert sc.neo4j_user == "u"
        assert sc.neo4j_password == "p"
        assert sc.neo4j_database == "g"
        assert sc.event_store_url == "postgresql://localhost/evt"

    def test_pgvector_falls_back_to_relational_url(self) -> None:
        d = {"storage": {"relational": {"url": "postgresql://localhost/db"}}}
        sc = StorageConfig.from_dict(d)
        assert sc.pgvector_url == "postgresql://localhost/db"
        # Event store falls back to relational URL too
        assert sc.event_store_url == "postgresql://localhost/db"

    def test_database_url_top_level_used_when_relational_missing(self) -> None:
        d = {"database_url": "postgresql://legacy/db"}
        sc = StorageConfig.from_dict(d)
        assert sc.postgresql_url == "postgresql://legacy/db"

    def test_empty_dict_yields_empty_config(self) -> None:
        sc = StorageConfig.from_dict({})
        assert sc.postgresql_url is None
        assert sc.neo4j_url is None


# ---------------------------------------------------------------------------
# _import_backend_class — error paths (lines 61-66 cover this)
# ---------------------------------------------------------------------------


class TestImportBackendClass:
    def test_returns_class_when_present(self) -> None:
        cls = _import_backend_class("khora.storage.factory", "StorageFactory")
        assert cls is StorageFactory

    def test_returns_none_for_missing_module(self) -> None:
        cls = _import_backend_class("khora.storage.does_not_exist_xyz", "Anything")
        assert cls is None

    def test_returns_none_for_missing_attr(self) -> None:
        cls = _import_backend_class("khora.storage.factory", "DoesNotExist")
        assert cls is None


# ---------------------------------------------------------------------------
# create_coordinator — surrealdb path (without requiring surrealdb installed)
# ---------------------------------------------------------------------------


class TestSurrealDBCoordinator:
    def test_missing_config_raises(self) -> None:
        config = StorageConfig(backend="surrealdb", surrealdb_config=None)
        factory = StorageFactory(config=config)
        with pytest.raises(ValueError, match="surrealdb_config is not set"):
            factory.create_coordinator()

    def test_dispatch_creates_coordinator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All four adapters share the same connection, all instances created.

        We can't safely stub the real ``khora.storage.backends.surrealdb.*``
        modules — they're imported elsewhere and other tests rely on the real
        objects. Instead, drive the dispatch by patching the StorageFactory's
        own import targets via ``patch.dict``-style replacement on the
        factory module's local imports.

        Use ``patch`` on the *concrete classes* the factory imports.
        """
        # Pre-import modules with top-level SurrealDBConnection bindings so
        # those bindings are already set to the real class before the patch
        # context below activates. A lazy first-import inside the patch window
        # would permanently bind the mock in the module's namespace, leaking
        # mock state into sibling tests (test_from_config_* in
        # TestRelationalAdapterLifecycle) when run under xdist -n auto.
        import khora.storage.backends.surrealdb.event_store  # noqa: F401
        import khora.storage.backends.surrealdb.relational  # noqa: F401

        rel_cls = MagicMock(return_value=MagicMock(name="rel"))
        vec_cls = MagicMock(return_value=MagicMock(name="vec"))
        graph_cls = MagicMock(return_value=MagicMock(name="graph"))
        evt_cls = MagicMock(return_value=MagicMock(name="evt"))
        conn_cls = MagicMock(name="SurrealDBConnection")

        # Patch the actual module attributes — these exist because surrealdb
        # subpackage is importable (just may fail at SDK-call time).
        with (
            patch("khora.storage.backends.surrealdb.connection.SurrealDBConnection", conn_cls),
            patch("khora.storage.backends.surrealdb.event_store.SurrealDBEventStoreAdapter", evt_cls),
            patch("khora.storage.backends.surrealdb.graph.SurrealDBGraphAdapter", graph_cls),
            patch("khora.storage.backends.surrealdb.relational.SurrealDBRelationalAdapter", rel_cls),
            patch("khora.storage.backends.surrealdb.vector.SurrealDBVectorAdapter", vec_cls),
        ):
            surreal_cfg = MagicMock()
            surreal_cfg.mode = "memory"
            surreal_cfg.path = None
            surreal_cfg.url = None
            config = StorageConfig(backend="surrealdb", surrealdb_config=surreal_cfg)
            factory = StorageFactory(config=config)
            coord = factory.create_coordinator()

        # SurrealDBConnection instantiated once with kwargs from the surreal_cfg
        conn_cls.assert_called_once()
        # All four adapters got the same connection instance
        conn_instance = conn_cls.return_value
        rel_cls.assert_called_once_with(conn_instance)
        vec_cls.assert_called_once_with(conn_instance)
        graph_cls.assert_called_once_with(conn_instance)
        evt_cls.assert_called_once_with(conn_instance)
        assert coord is not None


# ---------------------------------------------------------------------------
# create_coordinator — sqlite_lance path
# ---------------------------------------------------------------------------


class TestSQLiteLanceCoordinator:
    def test_missing_config_raises(self) -> None:
        config = StorageConfig(backend="sqlite_lance", sqlite_lance_config=None)
        factory = StorageFactory(config=config)
        with pytest.raises(ValueError, match="sqlite_lance_config is not set"):
            factory.create_coordinator()

    def test_missing_lancedb_dep_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without lancedb installed, the import inside the dispatch fails →
        the factory wraps it as a ValueError with an install hint.

        ``monkeypatch.setitem(sys.modules, ...)`` is reverted on test teardown
        so this stub doesn't leak to other tests.
        """
        config = StorageConfig(backend="sqlite_lance", sqlite_lance_config=MagicMock())
        factory = StorageFactory(config=config)
        # Force the import to fail by setting the module to None
        monkeypatch.setitem(sys.modules, "khora.storage.backends.sqlite_lance", None)
        with pytest.raises(ValueError, match="aiosqlite/lancedb are not installed"):
            factory.create_coordinator()


# ---------------------------------------------------------------------------
# _create_sqlite_coordinator — sqlite path
# ---------------------------------------------------------------------------


class TestSQLiteCoordinator:
    def test_missing_vector_config_raises(self) -> None:
        config = StorageConfig(backend="sqlite", vector_config=None)
        factory = StorageFactory(config=config)
        with pytest.raises(ValueError, match="vector config is not set"):
            factory.create_coordinator()

    def test_dispatch_when_aiosqlite_missing(self) -> None:
        """If aiosqlite is unavailable, raise a friendly install-hint error."""
        config = StorageConfig(backend="sqlite", vector_config=MagicMock())
        factory = StorageFactory(config=config)
        # Patch the lazy importer so it returns None for the relational class.
        with patch("khora.storage.factory._import_backend_class", side_effect=[None, MagicMock()]):
            with pytest.raises(ValueError, match="aiosqlite is not installed"):
                factory.create_coordinator()

    def test_dispatch_when_vector_class_missing(self) -> None:
        config = StorageConfig(backend="sqlite", vector_config=MagicMock())
        factory = StorageFactory(config=config)
        rel_cls = MagicMock()
        rel_cls.from_config = MagicMock()
        with patch("khora.storage.factory._import_backend_class", side_effect=[rel_cls, None]):
            with pytest.raises(ValueError, match="SQLite vector backend not available"):
                factory.create_coordinator()

    def test_dispatch_happy_path(self) -> None:
        """Both classes resolve → from_config is called for each."""
        vector_cfg = MagicMock()
        config = StorageConfig(backend="sqlite", vector_config=vector_cfg)
        factory = StorageFactory(config=config)
        rel_cls = MagicMock()
        rel_cls.from_config = MagicMock(return_value=MagicMock(name="rel"))
        vec_cls = MagicMock()
        vec_cls.from_config = MagicMock(return_value=MagicMock(name="vec"))
        with patch("khora.storage.factory._import_backend_class", side_effect=[rel_cls, vec_cls]):
            coord = factory.create_coordinator()
        rel_cls.from_config.assert_called_once_with(vector_cfg)
        vec_cls.from_config.assert_called_once_with(vector_cfg)
        assert coord.graph is None
        assert coord.event_store is None


# ---------------------------------------------------------------------------
# create_storage_coordinator — top-level helper
# ---------------------------------------------------------------------------


class TestCreateStorageCoordinator:
    def test_none_uses_defaults(self) -> None:
        coord = create_storage_coordinator(None)
        assert coord is not None

    def test_storage_config_passthrough(self) -> None:
        sc = StorageConfig()
        coord = create_storage_coordinator(sc)
        assert coord is not None

    def test_dict_input_converted(self) -> None:
        coord = create_storage_coordinator({"storage": {}})
        assert coord is not None

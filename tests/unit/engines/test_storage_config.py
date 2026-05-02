"""DYT-3545 coverage: ``khora.engines._storage_config.build_storage_config``.

Pins the three branches of the unified-backend selector (surrealdb,
sqlite_lance, postgres) plus the ``skip_graph`` toggle. The module is
trivial wiring code; these tests pin the contract so future config-schema
churn doesn't silently change which kwargs the engines see.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from khora.engines._storage_config import build_storage_config


@pytest.fixture
def base_config() -> MagicMock:
    """Mock KhoraConfig with traditional postgres backend defaults."""
    config = MagicMock()
    config.storage.backend = "postgres"
    config.storage.postgresql_pool_size = 5
    config.storage.postgresql_max_overflow = 10
    config.storage.postgresql_pool_pre_ping = False
    config.storage.embedding_dimension = 1536
    config.storage.use_halfvec = True
    config.storage.surrealdb = MagicMock()
    config.storage.sqlite_lance = MagicMock()
    config.get_postgresql_url.return_value = "postgresql://localhost/db"
    config.get_neo4j_url.return_value = "bolt://localhost:7687"
    config.get_neo4j_user.return_value = "neo4j"
    config.get_neo4j_password.return_value = "pw"
    config.get_neo4j_database.return_value = "neo4j"
    config.get_graph_config.return_value = MagicMock()
    config.get_vector_config.return_value = MagicMock()
    return config


class TestSurrealDBBranch:
    def test_surrealdb_returns_unified_storage_config(self, base_config: MagicMock) -> None:
        """When ``backend == "surrealdb"`` the surrealdb config is propagated."""
        base_config.storage.backend = "surrealdb"
        sc = build_storage_config(base_config)
        assert sc.backend == "surrealdb"
        assert sc.surrealdb_config is base_config.storage.surrealdb
        assert sc.postgresql_url is None

    def test_surrealdb_ignores_skip_graph(self, base_config: MagicMock) -> None:
        """``skip_graph`` is irrelevant for unified backends."""
        base_config.storage.backend = "surrealdb"
        sc = build_storage_config(base_config, skip_graph=True)
        assert sc.backend == "surrealdb"


class TestSQLiteLanceBranch:
    def test_sqlite_lance_returns_unified_storage_config(self, base_config: MagicMock) -> None:
        """When ``backend == "sqlite_lance"`` the sqlite_lance config is propagated."""
        base_config.storage.backend = "sqlite_lance"
        sc = build_storage_config(base_config)
        assert sc.backend == "sqlite_lance"
        assert sc.sqlite_lance_config is base_config.storage.sqlite_lance
        assert sc.postgresql_url is None


class TestTraditionalBranch:
    def test_postgres_with_graph_includes_neo4j_kwargs(self, base_config: MagicMock) -> None:
        """The traditional path forwards Neo4j credentials when graph_config is set."""
        sc = build_storage_config(base_config)
        assert sc.backend == "postgres"
        assert sc.postgresql_url == "postgresql://localhost/db"
        assert sc.neo4j_url == "bolt://localhost:7687"
        assert sc.neo4j_user == "neo4j"
        assert sc.neo4j_password == "pw"
        assert sc.neo4j_database == "neo4j"
        assert sc.graph_config is not None

    def test_skip_graph_omits_neo4j_kwargs(self, base_config: MagicMock) -> None:
        """``skip_graph=True`` omits Neo4j fields and graph_config."""
        sc = build_storage_config(base_config, skip_graph=True)
        # Neo4j defaults remain on the dataclass; they're untouched (not set).
        assert sc.graph_config is None
        # neo4j_url should NOT have been set, so it stays at the dataclass default.
        assert sc.neo4j_url is None

    def test_no_graph_config_skips_neo4j_kwargs(self, base_config: MagicMock) -> None:
        """If ``get_graph_config()`` returns None, Neo4j fields are omitted."""
        base_config.get_graph_config.return_value = None
        sc = build_storage_config(base_config)
        assert sc.graph_config is None
        assert sc.neo4j_url is None

    def test_default_backend_is_postgres(self, base_config: MagicMock) -> None:
        """When ``storage.backend`` is missing, ``getattr`` default kicks in."""
        # Simulate an older config that doesn't carry the ``backend`` attr.
        del base_config.storage.backend
        # ``getattr(...) == "postgres"`` path → traditional branch
        sc = build_storage_config(base_config)
        assert sc.backend == "postgres"

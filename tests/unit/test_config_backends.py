"""Tests for discriminated-union backend config parsing and backwards compatibility."""

import pytest
from pydantic import ValidationError

from khora.config.schema import (
    AGEConfig,
    KhoraConfig,
    KuzuConfig,
    MemgraphConfig,
    Neo4jConfig,
    ParsedNeo4jUrl,
    PgVectorConfig,
    StorageSettings,
)


@pytest.mark.unit
class TestStorageSettingsBackwardsCompat:
    """Legacy flat fields should still work and be migrated to new-style configs."""

    def test_legacy_neo4j_fields_migrated(self):
        settings = StorageSettings(
            neo4j_url="bolt://localhost:7687",
            neo4j_user="admin",
            neo4j_password="secret",
            neo4j_database="mydb",
        )
        assert isinstance(settings.graph, Neo4jConfig)
        assert settings.graph.url == "bolt://localhost:7687"
        assert settings.graph.user == "admin"
        assert settings.graph.password == "secret"
        assert settings.graph.database == "mydb"

    def test_legacy_pgvector_fields_migrated(self):
        settings = StorageSettings(
            pgvector_url="postgresql://localhost:5432/vectors",
            embedding_dimension=768,
        )
        assert isinstance(settings.vector, PgVectorConfig)
        assert settings.vector.url == "postgresql://localhost:5432/vectors"
        assert settings.vector.embedding_dimension == 768

    def test_new_style_graph_config_takes_precedence(self):
        settings = StorageSettings(
            neo4j_url="bolt://old:7687",
            graph=KuzuConfig(database_path="/tmp/kuzu"),
        )
        assert isinstance(settings.graph, KuzuConfig)
        assert settings.graph.database_path == "/tmp/kuzu"

    def test_defaults(self):
        settings = StorageSettings()
        # Graph backend is optional - None by default
        assert settings.graph is None
        assert isinstance(settings.vector, PgVectorConfig)
        assert settings.vector.url is None


@pytest.mark.unit
class TestDiscriminatedUnionParsing:
    """Discriminated union configs parsed from dict/YAML."""

    def test_kuzu_config_from_dict(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {"backend": "kuzu", "database_path": "/data/kuzu"},
            }
        )
        assert isinstance(settings.graph, KuzuConfig)
        assert settings.graph.database_path == "/data/kuzu"

    def test_memgraph_config_from_dict(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {"backend": "memgraph", "url": "bolt://mg:7687"},
            }
        )
        assert isinstance(settings.graph, MemgraphConfig)
        assert settings.graph.url == "bolt://mg:7687"

    def test_neptune_config_from_dict(self):
        settings = StorageSettings.model_validate(
            {"graph": {"backend": "neptune", "url": "bolt://cluster:8182", "iam_auth": True}}
        )
        from khora.config.schema import NeptuneConfig

        assert isinstance(settings.graph, NeptuneConfig)
        assert settings.graph.iam_auth is True

    def test_neo4j_is_default_graph_backend(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {"backend": "neo4j", "url": "bolt://neo:7687"},
            }
        )
        assert isinstance(settings.graph, Neo4jConfig)

    def test_age_config_from_dict(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {
                    "backend": "age",
                    "url": "postgresql://localhost:5432/khora",
                    "graph_name": "my_graph",
                },
            }
        )
        assert isinstance(settings.graph, AGEConfig)
        assert settings.graph.url == "postgresql://localhost:5432/khora"
        assert settings.graph.graph_name == "my_graph"

    def test_pgvector_is_default_vector_backend(self):
        settings = StorageSettings.model_validate(
            {
                "vector": {"backend": "pgvector", "url": "postgresql://pg:5432/db"},
            }
        )
        assert isinstance(settings.vector, PgVectorConfig)


@pytest.mark.unit
class TestNeo4jConfigQueryTimeout:
    """Tests for the Neo4jConfig.query_timeout field (DYT-1948)."""

    def test_default_query_timeout_is_5_seconds(self):
        """Default query_timeout caps long-running graph reads at 5 s."""
        cfg = Neo4jConfig()
        assert cfg.query_timeout == 5.0

    def test_query_timeout_can_be_overridden(self):
        """Callers can dial the cap up or down."""
        cfg = Neo4jConfig(query_timeout=10.0)
        assert cfg.query_timeout == 10.0

    def test_query_timeout_can_be_disabled(self):
        """Setting None disables the cap and falls back to the server default."""
        cfg = Neo4jConfig(query_timeout=None)
        assert cfg.query_timeout is None

    def test_query_timeout_round_trips_through_storage_settings(self):
        """The field survives parsing via the discriminated-union loader."""
        settings = StorageSettings.model_validate(
            {
                "graph": {
                    "backend": "neo4j",
                    "url": "bolt://neo:7687",
                    "query_timeout": 7.5,
                },
            }
        )
        assert isinstance(settings.graph, Neo4jConfig)
        assert settings.graph.query_timeout == 7.5

    @pytest.mark.parametrize("bad_value", [0, 0.0, -1, -0.5, 301.0, 1000.0, 1e18])
    def test_query_timeout_rejects_zero_negative_and_over_cap(self, bad_value):
        """Values <= 0 and values > 300s are both rejected.

        <= 0: driver treats 0 as 'run forever' (defeats the purpose);
              negative is nonsense.
        > 300: sanity cap. A 5-minute per-transaction timeout is already
               far beyond any reasonable interactive recall budget; higher
               values almost certainly indicate misconfiguration. Users
               who truly want no ceiling must pass ``None`` explicitly.
        """
        with pytest.raises(ValidationError):
            Neo4jConfig(query_timeout=bad_value)

    def test_query_timeout_accepts_boundary_values(self):
        """Sub-millisecond and exactly 300s are both valid."""
        assert Neo4jConfig(query_timeout=0.001).query_timeout == 0.001
        assert Neo4jConfig(query_timeout=300.0).query_timeout == 300.0


@pytest.mark.unit
class TestKhoraConfigGraphHelpers:
    """KhoraConfig.get_graph_config() / get_vector_config() work correctly."""

    def test_get_graph_config_neo4j(self):
        config = KhoraConfig(
            neo4j_url="bolt://neo4j:pass@localhost:7687",
        )
        graph = config.get_graph_config()
        assert isinstance(graph, Neo4jConfig)
        assert graph.url == "bolt://localhost:7687"
        assert graph.user == "neo4j"
        assert graph.password == "pass"

    def test_get_graph_config_kuzu(self):
        config = KhoraConfig(
            storage=StorageSettings(
                graph=KuzuConfig(database_path="/tmp/kuzu_test"),
            ),
        )
        graph = config.get_graph_config()
        assert isinstance(graph, KuzuConfig)
        assert graph.database_path == "/tmp/kuzu_test"

    def test_get_vector_config_defaults_to_postgresql_url(self):
        config = KhoraConfig(
            database_url="postgresql://localhost:5432/khora",
        )
        vector = config.get_vector_config()
        assert isinstance(vector, PgVectorConfig)
        assert vector.url == "postgresql://localhost:5432/khora"


@pytest.mark.unit
class TestGetNeo4jCredentials:
    """Regression tests for DYT-2049: Neo4j credentials must flow through when split from URL."""

    def test_get_neo4j_password_prefers_explicit_graph_password_over_credential_free_url(self) -> None:
        graph = Neo4jConfig(url="bolt://localhost:7687", user="neo4j", password="secretpass")
        cfg = KhoraConfig(database_url="postgresql://x@y/z", storage=StorageSettings(graph=graph))
        assert cfg.get_neo4j_password() == "secretpass"

    def test_get_neo4j_password_extracts_from_embedded_url(self) -> None:
        graph = Neo4jConfig(url="bolt://neo4j:secretpass@localhost:7687")
        cfg = KhoraConfig(database_url="postgresql://x@y/z", storage=StorageSettings(graph=graph))
        assert cfg.get_neo4j_password() == "secretpass"

    def test_get_neo4j_password_embedded_url_wins_over_field(self) -> None:
        graph = Neo4jConfig(url="bolt://neo4j:urlpass@localhost:7687", password="fieldpass")
        cfg = KhoraConfig(database_url="postgresql://x@y/z", storage=StorageSettings(graph=graph))
        assert cfg.get_neo4j_password() == "urlpass"

    def test_get_neo4j_user_and_database_fall_through_from_credential_free_url(self) -> None:
        graph = Neo4jConfig(
            url="bolt://localhost:7687",
            user="admin",
            password="secretpass",
            database="mydb",
        )
        cfg = KhoraConfig(database_url="postgresql://x@y/z", storage=StorageSettings(graph=graph))
        assert cfg.get_neo4j_user() == "admin"
        assert cfg.get_neo4j_password() == "secretpass"
        assert cfg.get_neo4j_database() == "mydb"

    def test_get_neo4j_password_legacy_flat_fields(self) -> None:
        cfg = KhoraConfig(
            database_url="postgresql://x@y/z",
            storage=StorageSettings(
                neo4j_url="bolt://localhost:7687",
                neo4j_user="legacy",
                neo4j_password="legacypass",
            ),
        )
        assert cfg.get_neo4j_password() == "legacypass"
        assert cfg.get_neo4j_user() == "legacy"


@pytest.mark.unit
class TestParsedNeo4jUrl:
    """Direct contract tests for ParsedNeo4jUrl.parse()."""

    def test_parse_respects_default_password_when_url_has_none(self) -> None:
        parsed = ParsedNeo4jUrl.parse("bolt://localhost:7687", default_password="fallbackpass")
        assert parsed.password == "fallbackpass"

    def test_parse_embedded_password_overrides_default(self) -> None:
        parsed = ParsedNeo4jUrl.parse("bolt://user:embedded@localhost:7687", default_password="fallback")
        assert parsed.password == "embedded"

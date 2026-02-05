"""Tests for discriminated-union backend config parsing and backwards compatibility."""

import pytest

from khora.config.schema import (
    ArcadeDBGraphConfig,
    ArcadeDBVectorConfig,
    KhoraConfig,
    KuzuConfig,
    MemgraphConfig,
    Neo4jConfig,
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

    def test_new_style_vector_config_takes_precedence(self):
        settings = StorageSettings(
            pgvector_url="postgresql://old:5432/db",
            vector=ArcadeDBVectorConfig(url="http://localhost:2480"),
        )
        assert isinstance(settings.vector, ArcadeDBVectorConfig)
        assert settings.vector.url == "http://localhost:2480"

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

    def test_arcadedb_graph_config_from_dict(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {"backend": "arcadedb", "url": "http://arcade:2480"},
            }
        )
        assert isinstance(settings.graph, ArcadeDBGraphConfig)
        assert settings.graph.url == "http://arcade:2480"

    def test_arcadedb_vector_config_from_dict(self):
        settings = StorageSettings.model_validate(
            {
                "vector": {"backend": "arcadedb", "url": "http://arcade:2480", "embedding_dimension": 768},
            }
        )
        assert isinstance(settings.vector, ArcadeDBVectorConfig)
        assert settings.vector.embedding_dimension == 768

    def test_neo4j_is_default_graph_backend(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {"backend": "neo4j", "url": "bolt://neo:7687"},
            }
        )
        assert isinstance(settings.graph, Neo4jConfig)

    def test_pgvector_is_default_vector_backend(self):
        settings = StorageSettings.model_validate(
            {
                "vector": {"backend": "pgvector", "url": "postgresql://pg:5432/db"},
            }
        )
        assert isinstance(settings.vector, PgVectorConfig)


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
class TestArcadeDBDualRole:
    """ArcadeDB dual-role detection."""

    def test_both_graph_and_vector_arcadedb(self):
        settings = StorageSettings.model_validate(
            {
                "graph": {"backend": "arcadedb", "url": "http://localhost:2480"},
                "vector": {"backend": "arcadedb", "url": "http://localhost:2480"},
            }
        )
        assert isinstance(settings.graph, ArcadeDBGraphConfig)
        assert isinstance(settings.vector, ArcadeDBVectorConfig)
        assert settings.graph.url == settings.vector.url

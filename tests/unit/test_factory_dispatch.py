"""Tests for factory registry-based dispatch of graph and vector backends."""

import pytest

from khora.config.schema import (
    AGEConfig,
    KuzuConfig,
    MemgraphConfig,
    Neo4jConfig,
    NeptuneConfig,
    PgVectorConfig,
)
from khora.storage.factory import _GRAPH_REGISTRY, _VECTOR_REGISTRY, StorageConfig, StorageFactory


@pytest.mark.unit
class TestRegistryContents:
    """Registries contain expected backends."""

    def test_graph_registry_has_all_backends(self):
        assert "neo4j" in _GRAPH_REGISTRY
        assert "kuzu" in _GRAPH_REGISTRY
        assert "memgraph" in _GRAPH_REGISTRY
        assert "neptune" in _GRAPH_REGISTRY
        assert "age" in _GRAPH_REGISTRY

    def test_vector_registry_has_all_backends(self):
        assert "pgvector" in _VECTOR_REGISTRY


@pytest.mark.unit
class TestFactoryLegacyPath:
    """Legacy StorageConfig fields still create correct backends."""

    def test_legacy_neo4j_creates_neo4j_backend(self):
        config = StorageConfig(
            neo4j_url="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="test",
        )
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        assert backend is not None
        assert type(backend).__name__ == "Neo4jBackend"

    def test_legacy_pgvector_creates_pgvector_backend(self):
        config = StorageConfig(
            pgvector_url="postgresql://localhost:5432/test",
        )
        factory = StorageFactory(config=config)
        backend = factory.create_vector_backend()
        assert backend is not None
        assert type(backend).__name__ == "PgVectorBackend"

    def test_no_neo4j_url_returns_none(self):
        config = StorageConfig()
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        assert backend is None

    def test_no_pgvector_url_returns_none(self):
        config = StorageConfig()
        factory = StorageFactory(config=config)
        backend = factory.create_vector_backend()
        assert backend is None


@pytest.mark.unit
class TestFactoryNewStyleDispatch:
    """New-style config dispatch creates correct backend types."""

    def test_kuzu_config_dispatch(self):
        config = StorageConfig(
            graph_config=KuzuConfig(database_path="/tmp/test_kuzu"),
        )
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        # Will be None if kuzu package not installed, which is OK
        if backend is not None:
            assert type(backend).__name__ == "KuzuBackend"

    def test_neo4j_config_dispatch(self):
        config = StorageConfig(
            graph_config=Neo4jConfig(url="bolt://localhost:7687"),
        )
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        assert backend is not None
        assert type(backend).__name__ == "Neo4jBackend"

    def test_memgraph_config_dispatch(self):
        config = StorageConfig(
            graph_config=MemgraphConfig(url="bolt://localhost:7687"),
        )
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        # Uses neo4j driver, should succeed
        if backend is not None:
            assert type(backend).__name__ == "MemgraphBackend"

    def test_neptune_config_dispatch(self):
        config = StorageConfig(
            graph_config=NeptuneConfig(url="bolt://cluster:8182"),
        )
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        if backend is not None:
            assert type(backend).__name__ == "NeptuneBackend"

    def test_age_config_dispatch(self):
        config = StorageConfig(
            graph_config=AGEConfig(url="postgresql://localhost:5432/test", graph_name="test"),
        )
        factory = StorageFactory(config=config)
        backend = factory.create_graph_backend()
        if backend is not None:
            assert type(backend).__name__ == "AGEBackend"

    def test_pgvector_config_dispatch(self):
        config = StorageConfig(
            vector_config=PgVectorConfig(url="postgresql://localhost:5432/test"),
        )
        factory = StorageFactory(config=config)
        backend = factory.create_vector_backend()
        assert backend is not None
        assert type(backend).__name__ == "PgVectorBackend"

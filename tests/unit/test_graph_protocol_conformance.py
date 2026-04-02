"""Protocol conformance tests for graph backends.

Parametrized tests that verify all graph backends implement
GraphBackendProtocol correctly. Tests are skipped if the optional
dependency is not installed.
"""

import pytest

# Check which backends are available
HAS_NEO4J = True
try:
    from khora.storage.backends.neo4j import Neo4jBackend
except ImportError:
    HAS_NEO4J = False

HAS_KUZU = True
try:
    from khora.storage.backends.kuzu import KuzuBackend
except ImportError:
    HAS_KUZU = False

HAS_MEMGRAPH = True
try:
    from khora.storage.backends.memgraph import MemgraphBackend
except ImportError:
    HAS_MEMGRAPH = False



@pytest.mark.unit
class TestProtocolConformance:
    """Verify that all backend classes have the required protocol methods."""

    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j package not installed")
    def test_neo4j_implements_protocol(self):
        backend = Neo4jBackend("bolt://localhost:7687")
        self._assert_graph_protocol(backend)

    @pytest.mark.skipif(not HAS_KUZU, reason="kuzu package not installed")
    def test_kuzu_implements_protocol(self, tmp_path):
        backend = KuzuBackend(str(tmp_path / "kuzu_db"))
        self._assert_graph_protocol(backend)

    @pytest.mark.skipif(not HAS_MEMGRAPH, reason="neo4j package not installed")
    def test_memgraph_implements_protocol(self):
        backend = MemgraphBackend("bolt://localhost:7687")
        self._assert_graph_protocol(backend)

    def _assert_graph_protocol(self, backend):
        """Verify all required protocol methods exist."""
        required_methods = [
            "connect",
            "disconnect",
            "is_healthy",
            "create_entity",
            "get_entity",
            "get_entity_by_name",
            "update_entity",
            "delete_entity",
            "list_entities",
            "create_relationship",
            "get_relationship",
            "delete_relationship",
            "get_entity_relationships",
            "list_relationships",
            "create_episode",
            "get_episode",
            "list_episodes",
            "find_paths",
            "get_neighborhood",
            "search_entities_by_attribute",
            # Batch/aggregate ops from GraphBackendBase
            "get_entities_batch",
            "get_neighborhoods_batch",
            "count_entities",
        ]
        for method in required_methods:
            assert hasattr(backend, method), f"Missing method: {method}"
            assert callable(getattr(backend, method)), f"Not callable: {method}"


@pytest.mark.unit
class TestFromConfig:
    """Verify from_config() class method on each backend."""

    @pytest.mark.skipif(not HAS_NEO4J, reason="neo4j package not installed")
    def test_neo4j_from_config(self):
        from khora.config.schema import Neo4jConfig

        config = Neo4jConfig(url="bolt://localhost:7687", user="admin", password="secret", database="testdb")
        backend = Neo4jBackend.from_config(config)
        assert backend._url == "bolt://localhost:7687"
        assert backend._user == "admin"
        assert backend._password == "secret"
        assert backend._database == "testdb"

    @pytest.mark.skipif(not HAS_KUZU, reason="kuzu package not installed")
    def test_kuzu_from_config(self, tmp_path):
        from khora.config.schema import KuzuConfig

        db_path = str(tmp_path / "kuzu_test")
        config = KuzuConfig(database_path=db_path, read_only=True)
        backend = KuzuBackend.from_config(config)
        assert backend._database_path == db_path
        assert backend._read_only is True

    @pytest.mark.skipif(not HAS_MEMGRAPH, reason="neo4j package not installed")
    def test_memgraph_from_config(self):
        from khora.config.schema import MemgraphConfig

        config = MemgraphConfig(url="bolt://mg:7687", user="mg", password="pass")
        backend = MemgraphBackend.from_config(config)
        assert backend._url == "bolt://mg:7687"
        assert backend._user == "mg"


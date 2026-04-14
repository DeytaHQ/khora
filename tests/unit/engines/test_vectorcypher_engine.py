"""Unit tests for the VectorCypher engine."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.engine import (
    ExtractionQualityMetrics,
    VectorCypherConfig,
    VectorCypherEngine,
)
from khora.memory_lake import RecallResult


class TestVectorCypherConfig:
    """Tests for VectorCypherConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = VectorCypherConfig()
        assert config.routing_enabled is True
        assert config.routing_use_llm is False
        assert config.skeleton_core_ratio == 0.70
        assert config.graph_default_depth == 2
        assert config.graph_max_depth == 4
        assert config.graph_max_entry_entities == 10
        assert config.fusion_rrf_k == 60
        assert config.fusion_vector_weight == 0.6
        assert config.fusion_graph_weight == 0.4
        assert config.fusion_simple_vector_weight == 0.8
        assert config.fusion_simple_graph_weight == 0.2
        assert config.fusion_complex_vector_weight == 0.4
        assert config.fusion_complex_graph_weight == 0.6
        assert config.temporal_recency_weight == 0.2
        assert config.temporal_recency_decay_days == 30
        assert config.recency_decay_type == "exponential"
        assert config.query_cache_ttl_seconds == 300
        assert config.query_cache_max_size == 100
        assert config.streaming_pipeline is True
        assert config.enable_smart_resolution is True
        assert config.lazy_entity_expansion is True
        assert config.fusion_hybrid_alpha == 0.7
        assert config.retriever_min_entity_similarity == 0.3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = VectorCypherConfig(
            routing_enabled=False,
            skeleton_core_ratio=0.5,
            graph_default_depth=3,
            fusion_vector_weight=0.7,
            fusion_graph_weight=0.3,
            query_cache_ttl_seconds=300,
        )
        assert config.routing_enabled is False
        assert config.skeleton_core_ratio == 0.5
        assert config.graph_default_depth == 3
        assert config.fusion_vector_weight == 0.7
        assert config.fusion_graph_weight == 0.3
        assert config.query_cache_ttl_seconds == 300


class TestExtractionQualityMetrics:
    """Tests for ExtractionQualityMetrics dataclass."""

    def test_defaults(self) -> None:
        """Test default metrics values."""
        metrics = ExtractionQualityMetrics()
        assert metrics.total_chunks == 0
        assert metrics.chunks_with_entities == 0
        assert metrics.total_entities == 0
        assert metrics.total_relationships == 0
        assert metrics.avg_entities_per_chunk == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.entity_type_distribution == {}

    def test_compute_averages(self) -> None:
        """Test computing averages from totals."""
        metrics = ExtractionQualityMetrics(
            total_chunks=10,
            total_entities=30,
        )
        metrics.compute_averages()
        assert metrics.avg_entities_per_chunk == 3.0

    def test_compute_averages_zero_chunks(self) -> None:
        """Test computing averages with zero chunks does not divide by zero."""
        metrics = ExtractionQualityMetrics(total_chunks=0, total_entities=5)
        metrics.compute_averages()
        assert metrics.avg_entities_per_chunk == 0.0

    def test_entity_type_distribution(self) -> None:
        """Test entity type distribution is mutable dict."""
        metrics = ExtractionQualityMetrics()
        metrics.entity_type_distribution["PERSON"] = 5
        metrics.entity_type_distribution["ORG"] = 3
        assert metrics.entity_type_distribution == {"PERSON": 5, "ORG": 3}


class TestVectorCypherEngineInit:
    """Tests for VectorCypherEngine initialization."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock KhoraConfig."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"
        return config

    def test_init_default_config(self, mock_config: MagicMock) -> None:
        """Test engine initialization with default VectorCypherConfig."""
        engine = VectorCypherEngine(mock_config)
        assert engine._vc_config is not None
        assert engine._vc_config.routing_enabled is True
        assert engine._connected is False
        assert engine._storage is None
        assert engine._neo4j_driver is None

    def test_init_custom_config(self, mock_config: MagicMock) -> None:
        """Test engine initialization with custom VectorCypherConfig."""
        vc_config = VectorCypherConfig(
            skeleton_core_ratio=0.5,
            graph_default_depth=3,
        )
        engine = VectorCypherEngine(mock_config, vectorcypher_config=vc_config)
        assert engine._vc_config.skeleton_core_ratio == 0.5
        assert engine._vc_config.graph_default_depth == 3

    def test_init_with_storage_config(self, mock_config: MagicMock) -> None:
        """Test engine initialization with explicit storage config."""
        storage_config = MagicMock()
        engine = VectorCypherEngine(mock_config, storage_config=storage_config)
        assert engine._storage_config is storage_config


class TestVectorCypherEngineGetters:
    """Tests for engine getter methods when not connected."""

    @pytest.fixture
    def engine(self) -> VectorCypherEngine:
        """Create an unconnected engine."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        return VectorCypherEngine(config)

    def test_get_storage_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_storage raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_storage()

    def test_get_temporal_store_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_temporal_store raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_temporal_store()

    def test_get_embedder_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_embedder raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_embedder()

    def test_get_retriever_raises_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_retriever raises RuntimeError when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            engine._get_retriever()

    def test_get_dual_nodes_returns_none_when_not_connected(self, engine: VectorCypherEngine) -> None:
        """Test _get_dual_nodes returns None when not connected (or SurrealDB backend)."""
        assert engine._get_dual_nodes() is None


@pytest.mark.unit
class TestVectorCypherEngineDisconnect:
    """Tests for engine disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        """Test disconnecting when already disconnected is a no-op."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)
        # Should not raise
        await engine.disconnect()
        assert engine._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_components(self) -> None:
        """Test that disconnect cleans up all component references."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)

        # Simulate connected state
        engine._connected = True
        engine._neo4j_driver = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._storage = AsyncMock()
        engine._embedder = MagicMock()
        engine._retriever = MagicMock()
        engine._dual_nodes = MagicMock()
        engine._router = MagicMock()

        with patch("khora.telemetry.shutdown_telemetry", new_callable=AsyncMock):
            await engine.disconnect()

        assert engine._connected is False
        assert engine._neo4j_driver is None
        assert engine._temporal_store is None
        assert engine._storage is None
        assert engine._embedder is None
        assert engine._retriever is None
        assert engine._dual_nodes is None
        assert engine._router is None


@pytest.mark.unit
class TestVectorCypherEngineRemember:
    """Tests for engine remember() with mocked backends."""

    @pytest.fixture
    def connected_engine(self) -> VectorCypherEngine:
        """Create a mock-connected engine for testing remember/recall/forget."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        config.llm.model = "gpt-4o-mini"
        config.pipeline.extract_entities = True

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._embedder = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._retriever = AsyncMock()
        engine._router = MagicMock()
        engine._neo4j_driver = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_remember_duplicate_document(self, connected_engine: VectorCypherEngine) -> None:
        """Test that remember returns early for duplicate documents."""
        namespace_id = uuid4()
        doc_id = uuid4()

        existing_doc = MagicMock()
        existing_doc.id = doc_id
        existing_doc.status = "completed"
        existing_doc.chunk_count = 5
        existing_doc.entity_count = 3

        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=existing_doc)

        result = await connected_engine.remember(
            "test content",
            namespace_id,
            entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
            relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
        )

        assert result.document_id == doc_id
        assert result.metadata.get("duplicate") is True
        connected_engine._storage.create_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_remember_new_document(self, connected_engine: VectorCypherEngine) -> None:
        """Test remember creates and processes a new document."""
        namespace_id = uuid4()
        doc_id = uuid4()

        connected_engine._storage.get_document_by_checksum = AsyncMock(return_value=None)

        created_doc = MagicMock()
        created_doc.id = doc_id
        created_doc.namespace_id = namespace_id
        created_doc.content = "test content"
        created_doc.metadata = MagicMock()
        created_doc.metadata.custom = {}
        connected_engine._storage.create_document = AsyncMock(return_value=created_doc)

        with patch.object(connected_engine, "_process_document", new_callable=AsyncMock, return_value=(3, 5, 2)):
            result = await connected_engine.remember(
                "test content",
                namespace_id,
                title="Test Doc",
                source="unit_test",
                entity_types=["PERSON", "ORGANIZATION", "LOCATION"],
                relationship_types=["WORKS_FOR", "KNOWS", "LOCATED_IN"],
            )

        assert result.document_id == doc_id
        assert result.chunks_created == 3
        assert result.entities_extracted == 5
        assert result.relationships_created == 2
        connected_engine._storage.create_document.assert_called_once()


@pytest.mark.unit
class TestVectorCypherEngineRecall:
    """Tests for engine recall() with mocked backends."""

    @pytest.fixture
    def connected_engine(self) -> VectorCypherEngine:
        """Create a mock-connected engine."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._embedder = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._neo4j_driver = AsyncMock()

        # Mock retriever
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="test",
        )
        chunk1 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="This is a test chunk with enough content to pass validation",
        )
        chunk2 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Another test chunk that is also long enough to pass validation",
        )
        retriever_result = VectorCypherResult(
            chunks=[(chunk1, 0.9), (chunk2, 0.7)],
            entities=[],
            routing_decision=routing,
            metadata={"search_mode": "simple_vector"},
        )
        engine._retriever = AsyncMock()
        engine._retriever.retrieve = AsyncMock(return_value=retriever_result)
        engine._router = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_recall_returns_results(self, connected_engine: VectorCypherEngine) -> None:
        """Test that recall returns validated results."""
        namespace_id = uuid4()
        result = await connected_engine.recall("test query", namespace_id)

        assert isinstance(result, RecallResult)
        assert result.query == "test query"
        assert result.namespace_id == namespace_id
        assert len(result.chunks) == 2
        assert result.metadata["engine"] == "vectorcypher"

    @pytest.mark.asyncio
    async def test_recall_filters_duplicates(self, connected_engine: VectorCypherEngine) -> None:
        """Test that recall filters duplicate chunks."""
        from khora.engines.vectorcypher.retriever import VectorCypherResult
        from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="test",
        )
        dup_id = uuid4()
        dup_chunk1 = Chunk(
            id=dup_id,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Duplicate chunk content that is long enough for validation",
        )
        dup_chunk2 = Chunk(
            id=dup_id,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Duplicate chunk content that is long enough for validation",
        )
        retriever_result = VectorCypherResult(
            chunks=[(dup_chunk1, 0.9), (dup_chunk2, 0.8)],
            entities=[],
            routing_decision=routing,
            metadata={},
        )
        connected_engine._retriever.retrieve = AsyncMock(return_value=retriever_result)

        namespace_id = uuid4()
        result = await connected_engine.recall("test", namespace_id)

        # Duplicates should be filtered
        assert len(result.chunks) == 1


@pytest.mark.unit
class TestVectorCypherEngineForget:
    """Tests for engine forget() with mocked backends."""

    @pytest.fixture
    def connected_engine(self) -> VectorCypherEngine:
        """Create a mock-connected engine."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)
        engine._connected = True
        engine._storage = AsyncMock()
        engine._temporal_store = AsyncMock()
        engine._dual_nodes = AsyncMock()
        engine._neo4j_driver = AsyncMock()
        return engine

    @pytest.mark.asyncio
    async def test_forget_with_namespace(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget with explicit namespace ID."""
        doc_id = uuid4()
        namespace_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is True
        connected_engine._dual_nodes.delete_chunks_by_document.assert_called_once_with(doc_id, namespace_id)
        connected_engine._temporal_store.delete_chunks_by_document.assert_called_once_with(doc_id, namespace_id)
        connected_engine._storage.delete_document.assert_called_once_with(doc_id)

    @pytest.mark.asyncio
    async def test_forget_namespace_mismatch(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget returns False when namespace doesn't match."""
        doc_id = uuid4()
        namespace_id = uuid4()
        wrong_namespace = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = wrong_namespace
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)

        result = await connected_engine.forget(doc_id, namespace_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_forget_without_namespace(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget without namespace looks up document's namespace."""
        doc_id = uuid4()
        namespace_id = uuid4()

        doc_mock = MagicMock()
        doc_mock.namespace_id = namespace_id
        connected_engine._storage.get_document = AsyncMock(return_value=doc_mock)
        connected_engine._storage.delete_document = AsyncMock(return_value=True)

        result = await connected_engine.forget(doc_id, None)

        assert result is True

    @pytest.mark.asyncio
    async def test_forget_document_not_found(self, connected_engine: VectorCypherEngine) -> None:
        """Test forget returns False when document not found."""
        doc_id = uuid4()
        connected_engine._storage.get_document = AsyncMock(return_value=None)

        result = await connected_engine.forget(doc_id, None)

        assert result is False


@pytest.mark.unit
class TestVectorCypherEngineHealthCheck:
    """Tests for engine health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        """Test health check returns disconnected when not connected."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)
        result = await engine.health_check()

        assert result == {"status": "disconnected"}

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self) -> None:
        """Test health check when all components are healthy."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)
        engine._connected = True

        # Mock storage health
        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {"postgresql": "ok", "graph": "ok"}
        engine._storage = AsyncMock()
        engine._storage.health_check = AsyncMock(return_value=storage_health)

        # Mock temporal store health
        engine._temporal_store = AsyncMock()
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})

        # Mock Neo4j health
        engine._neo4j_driver = AsyncMock()
        engine._neo4j_driver.verify_connectivity = AsyncMock()

        result = await engine.health_check()

        assert result["status"] == "healthy"
        assert result["neo4j"] == "healthy"
        assert result["engine"] == "vectorcypher"

    @pytest.mark.asyncio
    async def test_health_check_neo4j_unhealthy(self) -> None:
        """Test health check when Neo4j is unhealthy."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536

        engine = VectorCypherEngine(config)
        engine._connected = True

        storage_health = MagicMock()
        storage_health.is_healthy = True
        storage_health.summary = {"postgresql": "ok"}
        engine._storage = AsyncMock()
        engine._storage.health_check = AsyncMock(return_value=storage_health)

        engine._temporal_store = AsyncMock()
        engine._temporal_store.health_check = AsyncMock(return_value={"status": "healthy"})

        engine._neo4j_driver = AsyncMock()
        engine._neo4j_driver.verify_connectivity = AsyncMock(side_effect=Exception("connection refused"))

        result = await engine.health_check()

        assert result["status"] == "degraded"
        assert result["neo4j"] == "unhealthy"


@pytest.mark.unit
class TestVectorCypherEngineValidateRecallResults:
    """Tests for _validate_recall_results."""

    @pytest.fixture
    def engine(self) -> VectorCypherEngine:
        """Create an engine for testing validation."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        return VectorCypherEngine(config)

    def test_filters_empty_content(self, engine: VectorCypherEngine) -> None:
        """Test that chunks with empty content are filtered out."""
        c1 = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="")
        c2 = Chunk(
            id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="Valid long enough content for testing"
        )
        chunks = [(c1, 0.9), (c2, 0.8)]
        result = engine._validate_recall_results(chunks, "test query")
        assert len(result) == 1
        assert result[0][0].id == c2.id

    def test_filters_short_content(self, engine: VectorCypherEngine) -> None:
        """Test that chunks with very short content are filtered."""
        c1 = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="short")
        c2 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="This content is long enough to pass minimum length validation",
        )
        chunks = [(c1, 0.9), (c2, 0.8)]
        result = engine._validate_recall_results(chunks, "test")
        assert len(result) == 1

    def test_removes_duplicates(self, engine: VectorCypherEngine) -> None:
        """Test that duplicate chunks are removed."""
        shared_id = uuid4()
        c1 = Chunk(
            id=shared_id, namespace_id=uuid4(), document_id=uuid4(), content="First occurrence with enough content"
        )
        c2 = Chunk(
            id=shared_id, namespace_id=uuid4(), document_id=uuid4(), content="First occurrence with enough content"
        )
        chunks = [(c1, 0.9), (c2, 0.8)]
        result = engine._validate_recall_results(chunks, "test")
        assert len(result) == 1

    def test_normalizes_scores(self, engine: VectorCypherEngine) -> None:
        """Test that scores are clamped to [0, 1]."""
        c1 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Content that has a very high score value assigned to it",
        )
        c2 = Chunk(
            id=uuid4(),
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="Content that has a negative score value assigned to it",
        )
        chunks = [(c1, 1.5), (c2, -0.5)]
        result = engine._validate_recall_results(chunks, "test")
        assert result[0][1] == 1.0
        assert result[1][1] == 0.0

    def test_skips_non_chunk_objects(self, engine: VectorCypherEngine) -> None:
        """Test that non-Chunk objects are skipped."""
        c1 = Chunk(
            id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="Valid content that passes all checks"
        )
        chunks = [("not a chunk", 0.9), (c1, 0.8)]
        result = engine._validate_recall_results(chunks, "test")
        assert len(result) == 1


@pytest.mark.unit
class TestVectorCypherEngineConnectAcquisitionTimeout:
    """Tests for connection_acquisition_timeout being passed to Neo4j driver."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock KhoraConfig."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        config.storage.backend = "postgres"
        config.llm.model = "gpt-4o-mini"
        config.llm.embedding_model = "text-embedding-3-small"
        config.llm.embedding_dimension = 1536
        config.llm.timeout = 30
        config.llm.max_retries = 3
        config.llm.max_concurrent_llm_calls = 5
        config.pipeline.chunking_strategy = "recursive"
        config.pipeline.chunk_size = 1000
        config.pipeline.chunk_overlap = 200
        config.pipeline.extract_entities = True
        config.telemetry_database_url = None
        config.telemetry_service_name = "test"
        return config

    @pytest.mark.asyncio
    async def test_custom_acquisition_timeout(self, mock_config: MagicMock) -> None:
        """Test that connection_acquisition_timeout from config is passed to driver."""
        neo4j_cfg = MagicMock()
        neo4j_cfg.connection_acquisition_timeout = 2.5
        mock_config.get_graph_config.return_value = neo4j_cfg

        engine = VectorCypherEngine(mock_config)

        mock_driver = AsyncMock()
        sentinel = RuntimeError("stop after driver")
        with (
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver) as mock_driver_cls,
            patch(
                "khora.engines.vectorcypher.engine.create_storage_coordinator",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="stop after driver"),
        ):
            await engine.connect()

        mock_driver_cls.assert_called_once()
        call_kwargs = mock_driver_cls.call_args[1]
        assert call_kwargs["connection_acquisition_timeout"] == 2.5

    @pytest.mark.asyncio
    async def test_default_acquisition_timeout(self, mock_config: MagicMock) -> None:
        """Test that default 60.0 is used when config has no connection_acquisition_timeout."""
        # spec=[] prevents MagicMock from auto-creating attributes, so getattr falls through to default
        neo4j_cfg = MagicMock(spec=[])
        mock_config.get_graph_config.return_value = neo4j_cfg

        engine = VectorCypherEngine(mock_config)

        mock_driver = AsyncMock()
        sentinel = RuntimeError("stop after driver")
        with (
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver) as mock_driver_cls,
            patch(
                "khora.engines.vectorcypher.engine.create_storage_coordinator",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="stop after driver"),
        ):
            await engine.connect()

        mock_driver_cls.assert_called_once()
        call_kwargs = mock_driver_cls.call_args[1]
        assert call_kwargs["connection_acquisition_timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_none_graph_config_uses_defaults(self, mock_config: MagicMock) -> None:
        """Test that None graph config falls through to all driver defaults."""
        mock_config.get_graph_config.return_value = None

        engine = VectorCypherEngine(mock_config)

        mock_driver = AsyncMock()
        sentinel = RuntimeError("stop after driver")
        with (
            patch("neo4j.AsyncGraphDatabase.driver", return_value=mock_driver) as mock_driver_cls,
            patch(
                "khora.engines.vectorcypher.engine.create_storage_coordinator",
                side_effect=sentinel,
            ),
            pytest.raises(RuntimeError, match="stop after driver"),
        ):
            await engine.connect()

        mock_driver_cls.assert_called_once()
        call_kwargs = mock_driver_cls.call_args[1]
        assert call_kwargs["connection_acquisition_timeout"] == 60.0
        assert call_kwargs["max_connection_pool_size"] == 100
        assert call_kwargs["max_connection_lifetime"] == 900
        assert call_kwargs["liveness_check_timeout"] == 30.0


@pytest.mark.unit
class TestVectorCypherEngineParseDatetime:
    """Tests for _parse_datetime helper."""

    @pytest.fixture
    def engine(self) -> VectorCypherEngine:
        """Create an engine for testing datetime parsing."""
        config = MagicMock()
        config.get_postgresql_url.return_value = "postgresql://localhost/test"
        config.get_neo4j_url.return_value = "bolt://localhost:7687"
        config.get_neo4j_user.return_value = "neo4j"
        config.get_neo4j_password.return_value = "password"
        config.get_neo4j_database.return_value = "neo4j"
        config.get_graph_config.return_value = MagicMock()
        config.get_vector_config.return_value = MagicMock()
        config.storage.postgresql_pool_size = 5
        config.storage.postgresql_max_overflow = 10
        config.storage.embedding_dimension = 1536
        return VectorCypherEngine(config)

    def test_parse_datetime_object(self, engine: VectorCypherEngine) -> None:
        """Test that datetime objects pass through."""
        now = datetime.now(UTC)
        result = engine._parse_datetime(now)
        assert result == now

    def test_parse_naive_datetime(self, engine: VectorCypherEngine) -> None:
        """Test that naive datetimes get UTC timezone."""
        naive = datetime(2024, 1, 15)
        result = engine._parse_datetime(naive)
        assert result.tzinfo == UTC

    def test_parse_date_string(self, engine: VectorCypherEngine) -> None:
        """Test parsing date-only strings."""
        result = engine._parse_datetime("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == UTC

    def test_parse_iso_string_with_z(self, engine: VectorCypherEngine) -> None:
        """Test parsing ISO string with Z suffix."""
        result = engine._parse_datetime("2024-01-15T10:30:00Z")
        assert result.year == 2024
        assert result.hour == 10

    def test_parse_invalid_raises(self, engine: VectorCypherEngine) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            engine._parse_datetime("not-a-date")

        with pytest.raises(ValueError, match="Cannot parse datetime"):
            engine._parse_datetime(12345)

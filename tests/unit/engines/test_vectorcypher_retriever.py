"""Unit tests for VectorCypher retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherResult,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision


class TestRetrieverConfig:
    """Tests for RetrieverConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default retriever configuration."""
        config = RetrieverConfig()
        assert config.default_depth == 2
        assert config.max_depth == 4
        assert config.max_entry_entities == 10
        assert config.adaptive_depth_enabled is True
        assert config.rrf_k == 60
        assert config.vector_weight == 0.6
        assert config.graph_weight == 0.4
        assert config.simple_vector_weight == 0.8
        assert config.simple_graph_weight == 0.2
        assert config.complex_vector_weight == 0.4
        assert config.complex_graph_weight == 0.6
        assert config.recency_weight == 0.2
        assert config.recency_decay_days == 30
        assert config.recency_decay_type == "exponential"
        assert config.min_entity_similarity == 0.3
        assert config.hybrid_alpha == 0.7
        assert config.query_cache_ttl_seconds == 0
        assert config.query_cache_max_size == 100
        assert config.lazy_entity_expansion is False
        assert config.max_chunks == 50
        assert config.max_entities == 30

    def test_custom_values(self) -> None:
        """Test custom retriever configuration."""
        config = RetrieverConfig(
            default_depth=3,
            max_depth=5,
            rrf_k=100,
            vector_weight=0.7,
            graph_weight=0.3,
            recency_weight=0.0,
            query_cache_ttl_seconds=300,
        )
        assert config.default_depth == 3
        assert config.max_depth == 5
        assert config.rrf_k == 100
        assert config.vector_weight == 0.7
        assert config.graph_weight == 0.3
        assert config.recency_weight == 0.0
        assert config.query_cache_ttl_seconds == 300


class TestVectorCypherResult:
    """Tests for VectorCypherResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a VectorCypherResult."""
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="test",
        )
        chunk = Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content="test")
        result = VectorCypherResult(
            chunks=[(chunk, 0.9)],
            entities=[],
            routing_decision=routing,
            metadata={"test": True},
        )
        assert len(result.chunks) == 1
        assert result.entities == []
        assert result.metadata["test"] is True

    def test_default_metadata(self) -> None:
        """Test VectorCypherResult default metadata is empty dict."""
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.8,
            reasoning="test",
        )
        result = VectorCypherResult(
            chunks=[],
            entities=[],
            routing_decision=routing,
        )
        assert result.metadata == {}


@pytest.mark.unit
class TestRetrieverInit:
    """Tests for VectorCypherRetriever initialization."""

    def test_init_default_config(self) -> None:
        """Test retriever initialization with default config."""
        vector_store = AsyncMock()
        neo4j_driver = AsyncMock()
        embedder = AsyncMock()

        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=neo4j_driver,
            embedder=embedder,
        )

        assert retriever._config.default_depth == 2
        assert retriever._cache == {}

    def test_init_custom_config(self) -> None:
        """Test retriever initialization with custom config."""
        config = RetrieverConfig(default_depth=3, rrf_k=100)

        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=config,
        )

        assert retriever._config.default_depth == 3
        assert retriever._config.rrf_k == 100

    def test_init_with_storage(self) -> None:
        """Test retriever initialization with storage coordinator."""
        storage = AsyncMock()

        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            storage=storage,
        )

        assert retriever._storage is storage


@pytest.mark.unit
class TestRetrieverSimpleRetrieve:
    """Tests for simple vector-only retrieval path."""

    @pytest.fixture
    def retriever(self) -> VectorCypherRetriever:
        """Create a retriever with mocked dependencies."""
        vector_store = AsyncMock()
        neo4j_driver = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        # Mock vector store search
        chunk_id = uuid4()
        doc_id = uuid4()
        ns_id = uuid4()
        mock_result = MagicMock()
        mock_result.chunk = MagicMock()
        mock_result.chunk.id = chunk_id
        mock_result.chunk.namespace_id = ns_id
        mock_result.chunk.content = "test chunk content"
        mock_result.chunk.document_id = doc_id
        mock_result.chunk.occurred_at = None
        mock_result.chunk.created_at = None
        mock_result.chunk.metadata = {}
        mock_result.combined_score = 0.85
        mock_result.similarity = 0.85
        vector_store.search = AsyncMock(return_value=[mock_result])

        config = RetrieverConfig(query_cache_ttl_seconds=0)

        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=neo4j_driver,
            embedder=embedder,
            config=config,
        )

        # Mock the router to always return SIMPLE
        retriever._router = MagicMock()
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.SIMPLE,
                use_graph=False,
                graph_depth=0,
                confidence=0.9,
                reasoning="simple query",
            )
        )

        return retriever

    @pytest.mark.asyncio
    async def test_simple_retrieve(self, retriever: VectorCypherRetriever) -> None:
        """Test simple vector-only retrieval returns chunks."""
        namespace_id = uuid4()
        result = await retriever.retrieve("What is Python?", namespace_id)

        assert isinstance(result, VectorCypherResult)
        assert len(result.chunks) == 1
        assert result.entities == []
        assert result.routing_decision.complexity == QueryComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_simple_retrieve_empty_results(self, retriever: VectorCypherRetriever) -> None:
        """Test simple retrieval with no matches."""
        retriever._vector_store.search = AsyncMock(return_value=[])

        namespace_id = uuid4()
        result = await retriever.retrieve("nonexistent", namespace_id)

        assert result.chunks == []


@pytest.mark.unit
class TestRetrieverFuseResults:
    """Tests for the _fuse_results method."""

    @pytest.fixture
    def retriever(self) -> VectorCypherRetriever:
        """Create a retriever for testing fusion."""
        return VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
        )

    def _make_chunk(self, content: str = "test") -> Chunk:
        """Helper to create a Chunk with given content."""
        return Chunk(id=uuid4(), namespace_id=uuid4(), document_id=uuid4(), content=content)

    def test_fuse_basic(self, retriever: VectorCypherRetriever) -> None:
        """Test basic fusion of vector and graph results."""
        id1, id2 = uuid4(), uuid4()
        c1 = Chunk(id=id1, namespace_id=uuid4(), document_id=uuid4(), content="vec")
        c2 = Chunk(id=id2, namespace_id=uuid4(), document_id=uuid4(), content="graph")
        vector_chunks = [(id1, 0.9, c1)]
        graph_chunks = [(id2, 0.8, c2)]

        fused = retriever._fuse_results(vector_chunks, graph_chunks)

        assert len(fused) == 2

    def test_fuse_with_normalization(self, retriever: VectorCypherRetriever) -> None:
        """Test fusion with score normalization."""
        id1 = uuid4()
        c1 = Chunk(id=id1, namespace_id=uuid4(), document_id=uuid4(), content="test")
        vector_chunks = [(id1, 0.9, c1)]
        graph_chunks = [(id1, 5.0, c1)]

        fused = retriever._fuse_results(
            vector_chunks,
            graph_chunks,
            use_normalization=True,
        )

        assert len(fused) == 1

    def test_fuse_with_routing_simple(self, retriever: VectorCypherRetriever) -> None:
        """Test fusion uses simple weights for SIMPLE routing."""
        id1, id2 = uuid4(), uuid4()
        c1 = Chunk(id=id1, namespace_id=uuid4(), document_id=uuid4(), content="v")
        c2 = Chunk(id=id2, namespace_id=uuid4(), document_id=uuid4(), content="g")
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="test",
        )

        fused = retriever._fuse_results(
            [(id1, 0.9, c1)],
            [(id2, 0.8, c2)],
            routing=routing,
        )

        # Simple routing uses 0.8/0.2 weights -> vector result should rank higher
        assert fused[0].item_id == id1

    def test_fuse_with_routing_complex(self, retriever: VectorCypherRetriever) -> None:
        """Test fusion uses complex weights for COMPLEX routing."""
        id1, id2 = uuid4(), uuid4()
        c1 = Chunk(id=id1, namespace_id=uuid4(), document_id=uuid4(), content="v")
        c2 = Chunk(id=id2, namespace_id=uuid4(), document_id=uuid4(), content="g")
        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX,
            use_graph=True,
            graph_depth=3,
            confidence=0.9,
            reasoning="test",
        )

        fused = retriever._fuse_results(
            [(id1, 0.9, c1)],
            [(id2, 0.8, c2)],
            routing=routing,
        )

        # Complex routing uses 0.4/0.6 weights -> graph result should rank higher
        assert fused[0].item_id == id2

    def test_fuse_empty_inputs(self, retriever: VectorCypherRetriever) -> None:
        """Test fusion with empty inputs."""
        fused = retriever._fuse_results([], [])
        assert fused == []


@pytest.mark.unit
class TestRetrieverRecencyScores:
    """Tests for _calculate_recency_scores."""

    @pytest.fixture
    def retriever(self) -> VectorCypherRetriever:
        """Create a retriever for testing."""
        return VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
        )

    def test_recency_scores_with_dates(self, retriever: VectorCypherRetriever) -> None:
        """Test recency scores are computed from occurred_at."""
        from khora.engines.vectorcypher.fusion import FusedResult

        id1, id2 = uuid4(), uuid4()
        chunk1 = Chunk(
            id=id1,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="old",
            metadata=ChunkMetadata(custom={"occurred_at": "2020-01-01T00:00:00+00:00"}),
        )
        chunk2 = Chunk(
            id=id2,
            namespace_id=uuid4(),
            document_id=uuid4(),
            content="recent",
            metadata=ChunkMetadata(custom={"occurred_at": "2026-02-14T00:00:00+00:00"}),
        )
        results = [
            FusedResult(item_id=id1, item=chunk1, rrf_score=0.9),
            FusedResult(item_id=id2, item=chunk2, rrf_score=0.8),
        ]

        scores = retriever._calculate_recency_scores(results)

        # Recent item should have higher score
        assert scores[id2] > scores[id1]

    def test_recency_scores_missing_date(self, retriever: VectorCypherRetriever) -> None:
        """Test default recency score for missing dates."""
        from khora.engines.vectorcypher.fusion import FusedResult

        id1 = uuid4()
        chunk = Chunk(id=id1, namespace_id=uuid4(), document_id=uuid4(), content="no date")
        results = [FusedResult(item_id=id1, item=chunk, rrf_score=0.9)]

        scores = retriever._calculate_recency_scores(results)

        assert scores[id1] == 0.5  # Default for missing dates

    def test_recency_scores_non_chunk_item(self, retriever: VectorCypherRetriever) -> None:
        """Test default recency score for non-Chunk items."""
        from khora.engines.vectorcypher.fusion import FusedResult

        id1 = uuid4()
        results = [FusedResult(item_id=id1, item="not a chunk", rrf_score=0.9)]

        scores = retriever._calculate_recency_scores(results)

        assert scores[id1] == 0.5


@pytest.mark.unit
class TestRetrieverCaching:
    """Tests for query result caching."""

    @pytest.mark.asyncio
    async def test_cache_disabled_by_default(self) -> None:
        """Test that caching is disabled when ttl is 0."""
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=RetrieverConfig(query_cache_ttl_seconds=0),
        )

        assert retriever._cache_ttl == 0
        assert retriever._cache == {}

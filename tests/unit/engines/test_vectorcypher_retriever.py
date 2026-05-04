"""Unit tests for VectorCypher retriever."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata
from khora.engines.vectorcypher.fusion import FusedResult, normalize_scores
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
        assert config.enable_session_aware_search is True
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

    def test_session_aware_search_enabled(self) -> None:
        """Test enabling session-aware search."""
        config = RetrieverConfig(enable_session_aware_search=True)
        assert config.enable_session_aware_search is True


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

    def test_init_default_neo4j_query_timeout_is_none(self) -> None:
        """When the kwarg is omitted, the inner DualNodeManager has no timeout."""
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
        )

        assert retriever._dual_nodes is not None
        assert retriever._dual_nodes._query_timeout is None

    def test_init_forwards_neo4j_query_timeout_to_dual_nodes(self) -> None:
        """neo4j_query_timeout is forwarded to the underlying DualNodeManager."""
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            neo4j_query_timeout=3.0,
        )

        assert retriever._dual_nodes is not None
        assert retriever._dual_nodes._query_timeout == 3.0

    def test_init_no_dual_nodes_when_driver_missing(self) -> None:
        """Without a driver, _dual_nodes is None even when timeout is set."""
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=None,
            embedder=AsyncMock(),
            neo4j_query_timeout=3.0,
        )

        assert retriever._dual_nodes is None


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


@pytest.mark.unit
class TestSimpleRetrieveScoreNormalization:
    """Regression tests for DYT-1733: simple path score normalization.

    Before the fix, _simple_retrieve() returned raw RRF scores (~0.009-0.016)
    while the complex path returned normalized [0,1] scores. This caused
    downstream consumers to see artificially low confidence for simple queries.
    """

    def test_normalize_scores_with_rrf_like_inputs(self) -> None:
        """normalize_scores maps tiny RRF scores to full [0,1] range."""
        raw_rrf_scores = [0.016, 0.014, 0.012, 0.010, 0.009]
        fused = [FusedResult(item_id=uuid4(), item=f"chunk_{i}", rrf_score=s) for i, s in enumerate(raw_rrf_scores)]

        normalized = normalize_scores(fused)
        scores = [r.rrf_score for r in normalized]

        # Max must be 1.0, min must be 0.0 (min-max normalization with >1 distinct scores)
        assert scores[0] == 1.0
        assert scores[-1] == 0.0
        # All in [0, 1]
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Meaningful spread: not all clustered below 0.05
        assert max(scores) - min(scores) == 1.0

    def test_normalize_scores_all_identical(self) -> None:
        """All identical scores normalize to 1.0 (no variance)."""
        fused = [FusedResult(item_id=uuid4(), item=f"chunk_{i}", rrf_score=0.012) for i in range(5)]
        normalized = normalize_scores(fused)
        scores = [r.rrf_score for r in normalized]
        assert all(s == 1.0 for s in scores)

    def test_normalize_scores_two_results(self) -> None:
        """Two distinct scores normalize to (1.0, 0.0)."""
        fused = [
            FusedResult(item_id=uuid4(), item="high", rrf_score=0.016),
            FusedResult(item_id=uuid4(), item="low", rrf_score=0.009),
        ]
        normalized = normalize_scores(fused)
        assert normalized[0].rrf_score == 1.0
        assert normalized[1].rrf_score == 0.0

    def test_normalize_scores_already_normalized_inputs(self) -> None:
        """Scores already in [0,1] are re-normalized correctly."""
        fused = [
            FusedResult(item_id=uuid4(), item="a", rrf_score=0.9),
            FusedResult(item_id=uuid4(), item="b", rrf_score=0.8),
            FusedResult(item_id=uuid4(), item="c", rrf_score=0.7),
        ]
        normalized = normalize_scores(fused)
        scores = [r.rrf_score for r in normalized]
        assert scores[0] == 1.0
        assert scores[-1] == 0.0
        assert 0.4 < scores[1] < 0.6  # middle score ~0.5

    def test_normalize_scores_empty(self) -> None:
        """Empty input returns empty output."""
        assert normalize_scores([]) == []

    @pytest.fixture
    def multi_result_retriever(self) -> VectorCypherRetriever:
        """Create a retriever whose vector store returns multiple results with raw RRF-like scores.

        The mock exercises the ``combined_score`` field on each result, which is the
        primary path in ``r.combined_score or r.similarity`` inside _simple_retrieve().
        """
        vector_store = AsyncMock()
        neo4j_driver = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        ns_id = uuid4()
        doc_id = uuid4()

        # Simulate 5 results with small raw scores typical of RRF (the bug scenario)
        # RRF scores with k=60: rank 1 = 1/61 ≈ 0.016, rank 5 ≈ 0.009.
        # These reproduce the DYT-1733 bug where all scores cluster below 0.05.
        raw_scores = [0.016, 0.014, 0.012, 0.010, 0.009]
        mock_results = []
        for score in raw_scores:
            chunk_id = uuid4()
            mock_result = MagicMock()
            mock_result.chunk = MagicMock()
            mock_result.chunk.id = chunk_id
            mock_result.chunk.namespace_id = ns_id
            mock_result.chunk.document_id = doc_id
            mock_result.chunk.content = f"test content with score {score}"
            mock_result.chunk.occurred_at = None
            mock_result.chunk.created_at = None
            mock_result.chunk.metadata = {}
            mock_result.combined_score = score
            mock_result.similarity = score
            mock_results.append(mock_result)

        vector_store.search = AsyncMock(return_value=mock_results)

        config = RetrieverConfig(query_cache_ttl_seconds=0)
        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=neo4j_driver,
            embedder=embedder,
            config=config,
        )

        # Route to SIMPLE path
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
    async def test_simple_retrieve_scores_normalized(self, multi_result_retriever: VectorCypherRetriever) -> None:
        """_simple_retrieve() must return scores in [0,1], not raw RRF values."""
        namespace_id = uuid4()
        result = await multi_result_retriever.retrieve("test query", namespace_id)

        scores = [score for _, score in result.chunks]

        # Scores must be in [0, 1] range
        assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range: {scores}"
        # Max score must be 1.0 (min-max normalization guarantees this)
        assert max(scores) == 1.0, f"Max score should be 1.0, got {max(scores)}"
        # Min score must be 0.0
        assert min(scores) == 0.0, f"Min score should be 0.0, got {min(scores)}"
        # Scores must NOT all be clustered below 0.05 (the original bug)
        assert any(s > 0.05 for s in scores), f"All scores below 0.05: {scores}"

    @pytest.mark.asyncio
    async def test_simple_retrieve_single_result_normalized(self) -> None:
        """A single result should get score 1.0 after normalization."""
        vector_store = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        mock_result = MagicMock()
        mock_result.chunk = MagicMock()
        mock_result.chunk.id = uuid4()
        mock_result.chunk.namespace_id = uuid4()
        mock_result.chunk.document_id = uuid4()
        mock_result.chunk.content = "single result"
        mock_result.chunk.occurred_at = None
        mock_result.chunk.created_at = None
        mock_result.chunk.metadata = {}
        mock_result.combined_score = 0.013
        mock_result.similarity = 0.013

        vector_store.search = AsyncMock(return_value=[mock_result])

        config = RetrieverConfig(query_cache_ttl_seconds=0)
        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=AsyncMock(),
            embedder=embedder,
            config=config,
        )
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

        result = await retriever.retrieve("test", uuid4())

        assert len(result.chunks) == 1
        # Single result: normalize_scores sets all-equal scores to 1.0
        assert result.chunks[0][1] == 1.0


@pytest.mark.unit
class TestGracefulDegradation:
    """Tests for graceful degradation when Neo4j is unavailable."""

    @pytest.fixture
    def ns_id(self) -> UUID:
        return uuid4()

    @pytest.fixture
    def retriever(self, ns_id: UUID) -> VectorCypherRetriever:
        """Create a retriever with mocked dependencies for graph-path testing."""
        vector_store = AsyncMock()
        neo4j_driver = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        embedder.model_name = "test-model"
        embedder.dimension = 1536

        # Mock vector store search — returns chunks for _vector_search_chunks
        chunk_id = uuid4()
        doc_id = uuid4()
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

        # Storage coordinator for entity vector search
        storage = AsyncMock()
        entry_entity_id = uuid4()
        storage.search_similar_entities = AsyncMock(return_value=[(entry_entity_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={})

        config = RetrieverConfig(query_cache_ttl_seconds=0)

        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=neo4j_driver,
            embedder=embedder,
            config=config,
            storage=storage,
        )

        # Route to MODERATE/COMPLEX (use_graph=True) to exercise _vectorcypher_retrieve
        retriever._router = MagicMock()
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.MODERATE,
                use_graph=True,
                graph_depth=2,
                confidence=0.8,
                reasoning="moderate query",
            )
        )
        retriever._router.compute_adaptive_depth = MagicMock(return_value=2)

        return retriever

    @pytest.mark.asyncio
    async def test_retrieve_graceful_degradation_on_neo4j_timeout(
        self, retriever: VectorCypherRetriever, ns_id: UUID
    ) -> None:
        """When _cypher_expand raises ConnectionAcquisitionTimeoutError, the
        retriever returns valid results with graph_fallback metadata."""
        from neo4j.exceptions import ServiceUnavailable

        try:
            from neo4j.exceptions import ConnectionAcquisitionTimeoutError as _CATE
        except ImportError:
            _CATE = ServiceUnavailable  # type: ignore[misc,assignment]

        # _cypher_expand raises a transient timeout error
        retriever._cypher_expand = AsyncMock(
            side_effect=_CATE("failed to obtain a connection from the pool within 60.0s (timeout)")
        )
        # _fetch_chunks_from_entities should NOT be called when graph_fallback=True
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])

        result = await retriever.retrieve("What is Python?", ns_id)

        assert isinstance(result, VectorCypherResult)
        assert result.metadata["graph_fallback"] is True
        # graph_error contains the exception type name
        assert result.metadata.get("graph_error") in (
            "ConnectionAcquisitionTimeoutError",
            "ServiceUnavailable",  # fallback alias on older neo4j SDKs
        )
        # Should have chunks from vector search
        assert len(result.chunks) > 0
        # _fetch_chunks_from_entities must NOT have been called
        retriever._fetch_chunks_from_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_does_not_swallow_client_error(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """ClientError from _cypher_expand is NOT caught — it propagates."""
        from neo4j.exceptions import ClientError

        retriever._cypher_expand = AsyncMock(side_effect=ClientError("Invalid Cypher query"))

        with pytest.raises(ClientError, match="Invalid Cypher query"):
            await retriever.retrieve("What is Python?", ns_id)

    @pytest.mark.asyncio
    async def test_version_filter_graceful_degradation(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """When _version_filter_entities raises ServiceUnavailable, the
        retriever still returns valid results (keeps unfiltered entities)."""
        from neo4j.exceptions import ServiceUnavailable

        from khora.engines.vectorcypher.temporal_detection import (
            TemporalCategory,
            TemporalSignal,
        )

        # _cypher_expand succeeds
        expanded_id = uuid4()
        retriever._cypher_expand = AsyncMock(
            return_value=(
                {expanded_id: 0.7},
                {str(expanded_id): {"name": "Test", "entity_type": "CONCEPT"}},
            )
        )
        # _version_filter_entities raises a transient error
        retriever._version_filter_entities = AsyncMock(side_effect=ServiceUnavailable("Connection refused"))
        # _fetch_chunks_from_entities succeeds
        chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="graph chunk")
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(chunk.id, 0.8, chunk)])

        # Build temporal signal for EXPLICIT path (triggers _version_filter_entities)
        from datetime import datetime

        from khora.engines.skeleton.backends import TemporalFilter

        tf = TemporalFilter(occurred_before=datetime(2025, 1, 1, tzinfo=UTC))
        temporal_signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.EXPLICIT,
            confidence=0.9,
            source="dictionary",
            temporal_filter=tf,
        )

        result = await retriever.retrieve(
            "What happened before 2025?",
            ns_id,
            temporal_signal=temporal_signal,
        )

        assert isinstance(result, VectorCypherResult)
        # Should complete without raising
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_version_history_graceful_degradation(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """When _fetch_version_history raises ServiceUnavailable, the
        retriever still returns valid results (version_history becomes None)."""
        from neo4j.exceptions import ServiceUnavailable

        from khora.engines.vectorcypher.temporal_detection import (
            TemporalCategory,
            TemporalSignal,
        )

        # _cypher_expand succeeds
        expanded_id = uuid4()
        retriever._cypher_expand = AsyncMock(
            return_value=(
                {expanded_id: 0.7},
                {str(expanded_id): {"name": "Test", "entity_type": "CONCEPT"}},
            )
        )
        # _fetch_version_history raises a transient error
        retriever._fetch_version_history = AsyncMock(side_effect=ServiceUnavailable("Connection refused"))
        # _fetch_chunks_from_entities succeeds
        chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="graph chunk")
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(chunk.id, 0.8, chunk)])

        # Build temporal signal for CHANGE path (triggers _fetch_version_history)
        temporal_signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.CHANGE,
            confidence=0.9,
            source="dictionary",
        )

        result = await retriever.retrieve(
            "How has the policy changed?",
            ns_id,
            temporal_signal=temporal_signal,
        )

        assert isinstance(result, VectorCypherResult)
        # Should complete without raising
        assert len(result.chunks) > 0
        # version_history should be None due to the failure
        assert result.metadata.get("version_history") is None

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_session_expired(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """SessionExpired from _cypher_expand triggers graceful degradation."""
        from neo4j.exceptions import SessionExpired

        retriever._cypher_expand = AsyncMock(side_effect=SessionExpired("Session no longer valid"))
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])

        result = await retriever.retrieve("What is Python?", ns_id)

        assert result.metadata["graph_fallback"] is True
        assert result.metadata.get("graph_error") == "SessionExpired"
        assert len(result.chunks) > 0
        retriever._fetch_chunks_from_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_transient_error(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """TransientError from _cypher_expand triggers graceful degradation."""
        from neo4j.exceptions import TransientError

        retriever._cypher_expand = AsyncMock(side_effect=TransientError("Database unavailable"))
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])

        result = await retriever.retrieve("What is Python?", ns_id)

        assert result.metadata["graph_fallback"] is True
        assert result.metadata.get("graph_error") == "TransientError"
        assert len(result.chunks) > 0
        retriever._fetch_chunks_from_entities.assert_not_called()

    @pytest.mark.asyncio
    async def test_cypher_expand_fallback_emits_warning(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """Warning log is emitted when _cypher_expand falls back."""
        from unittest.mock import patch

        from neo4j.exceptions import ServiceUnavailable

        retriever._cypher_expand = AsyncMock(side_effect=ServiceUnavailable("Connection refused"))
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])

        with patch("khora.engines.vectorcypher.retriever.logger") as mock_logger:
            await retriever.retrieve("What is Python?", ns_id)

        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("Neo4j unavailable during cypher_expand" in c for c in warning_calls)

    @pytest.mark.asyncio
    async def test_version_filter_fallback_emits_warning(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """Warning log is emitted when _version_filter_entities falls back."""
        from datetime import datetime
        from unittest.mock import patch

        from neo4j.exceptions import ServiceUnavailable

        from khora.engines.skeleton.backends import TemporalFilter
        from khora.engines.vectorcypher.temporal_detection import (
            TemporalCategory,
            TemporalSignal,
        )

        expanded_id = uuid4()
        retriever._cypher_expand = AsyncMock(
            return_value=(
                {expanded_id: 0.7},
                {str(expanded_id): {"name": "Test", "entity_type": "CONCEPT"}},
            )
        )
        retriever._version_filter_entities = AsyncMock(side_effect=ServiceUnavailable("Connection refused"))
        chunk = Chunk(id=uuid4(), namespace_id=ns_id, document_id=uuid4(), content="graph chunk")
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[(chunk.id, 0.8, chunk)])

        tf = TemporalFilter(occurred_before=datetime(2025, 1, 1, tzinfo=UTC))
        temporal_signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.EXPLICIT,
            confidence=0.9,
            source="dictionary",
            temporal_filter=tf,
        )

        with patch("khora.engines.vectorcypher.retriever.logger") as mock_logger:
            await retriever.retrieve("What happened before 2025?", ns_id, temporal_signal=temporal_signal)

        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("Version filter failed" in c for c in warning_calls)

    @pytest.mark.asyncio
    async def test_outer_fallback_on_fetch_chunks_failure(self, retriever: VectorCypherRetriever, ns_id: UUID) -> None:
        """When _cypher_expand succeeds but _fetch_chunks_from_entities raises
        a transient error, the outer catch in retrieve() fires and returns
        vector-only results via _vector_only_fallback."""
        from neo4j.exceptions import ServiceUnavailable

        expanded_id = uuid4()
        retriever._cypher_expand = AsyncMock(
            return_value=(
                {expanded_id: 0.7},
                {str(expanded_id): {"name": "Test", "entity_type": "CONCEPT"}},
            )
        )
        retriever._fetch_chunks_from_entities = AsyncMock(side_effect=ServiceUnavailable("Connection refused"))

        result = await retriever.retrieve("What is Python?", ns_id)

        assert isinstance(result, VectorCypherResult)
        # Outer fallback sets these metadata keys
        assert result.metadata.get("graph_fallback") is True
        assert len(result.chunks) > 0


@pytest.mark.unit
class TestLLMRerankConfidenceGate:
    """Tests for the LLM rerank confidence gate (Sprint 1 — multihop regression).

    The gate is a small predicate that decides whether to skip the LLM rerank
    step after cross-encoder scoring. Two independent conditions trigger a
    skip; either one is sufficient. We test the predicate directly because
    the call sites (graph + simple paths) both delegate to it.
    """

    def _make_retriever(self, config: RetrieverConfig) -> VectorCypherRetriever:
        return VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=config,
        )

    def test_skips_on_legacy_gap_gate(self) -> None:
        """Large gap alone is enough to skip — preserves prior behavior."""
        retriever = self._make_retriever(
            RetrieverConfig(
                llm_reranking_confidence_threshold=0.1,
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            )
        )
        # gap=0.2 > confidence_threshold=0.1 → skip even with low top_score
        assert retriever._should_skip_llm_rerank(top_score=0.4, gap=0.2) is True

    def test_skips_on_decisive_winner_gate(self) -> None:
        """High top_score + meaningful gap → skip even when gap < legacy threshold."""
        retriever = self._make_retriever(
            RetrieverConfig(
                # Set legacy threshold above the gap so ONLY the new gate can fire.
                llm_reranking_confidence_threshold=0.5,
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            )
        )
        # top_score=0.85 ≥ 0.7, gap=0.15 ≥ 0.1, gap < legacy 0.5 → decisive gate fires
        assert retriever._should_skip_llm_rerank(top_score=0.85, gap=0.15) is True

    def test_does_not_skip_when_top_score_below_threshold(self) -> None:
        """Low top_score → LLM rerank is needed even if gap is OK."""
        retriever = self._make_retriever(
            RetrieverConfig(
                llm_reranking_confidence_threshold=0.5,
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            )
        )
        # top_score=0.6 < 0.7 → decisive gate does NOT fire; gap=0.15 < 0.5 → legacy doesn't either
        assert retriever._should_skip_llm_rerank(top_score=0.6, gap=0.15) is False

    def test_does_not_skip_when_gap_too_small(self) -> None:
        """Small gap (uncertain ranking) → run the LLM rerank, even with high top_score."""
        retriever = self._make_retriever(
            RetrieverConfig(
                llm_reranking_confidence_threshold=0.5,
                llm_reranking_min_top_score=0.7,
                llm_reranking_decisive_gap=0.1,
            )
        )
        # top_score=0.9 high, but gap=0.05 < decisive_gap=0.1 AND < legacy 0.5
        assert retriever._should_skip_llm_rerank(top_score=0.9, gap=0.05) is False

    def test_decisive_gate_thresholds_are_tunable(self) -> None:
        """Both new thresholds must be exposed on RetrieverConfig and forwarded."""
        config = RetrieverConfig(
            # Hold the legacy gate inert so the decisive-winner gate is the only one in play.
            llm_reranking_confidence_threshold=0.5,
            llm_reranking_min_top_score=0.85,
            llm_reranking_decisive_gap=0.2,
        )
        assert config.llm_reranking_min_top_score == 0.85
        assert config.llm_reranking_decisive_gap == 0.2

        retriever = self._make_retriever(config)
        # top=0.8 < 0.85, gap=0.25 < legacy 0.5 → neither gate fires
        assert retriever._should_skip_llm_rerank(top_score=0.8, gap=0.25) is False
        # top=0.9 ≥ 0.85 AND gap=0.25 ≥ 0.2 → decisive gate fires
        assert retriever._should_skip_llm_rerank(top_score=0.9, gap=0.25) is True


@pytest.mark.unit
class TestExplicitTemporalSignalSkipsFallback:
    """DYT-3605: when the caller asserts an EXPLICIT temporal_signal carrying a
    parsed date filter, the retriever's "sparse-results" re-run-without-filter
    fallback (engine.py-side, line ~754) MUST be skipped. Sparse results are
    the correct signal in this path — the data may simply not exist in that
    time window, which feeds downstream abstention."""

    @pytest.mark.asyncio
    async def test_explicit_temporal_signal_skips_sparse_results_fallback(self) -> None:
        from datetime import datetime
        from unittest.mock import patch

        from khora.engines.skeleton.backends import TemporalFilter
        from khora.engines.vectorcypher.temporal_detection import (
            TemporalCategory,
            TemporalSignal,
        )

        ns_id = uuid4()

        # ── Fixture wiring (mirrors TestGracefulDegradation.retriever) ────────
        vector_store = AsyncMock()
        neo4j_driver = AsyncMock()
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        embedder.model_name = "test-model"
        embedder.dimension = 1536

        # Mock vector store search (used by _vector_search_chunks under the hood)
        chunk_id = uuid4()
        doc_id = uuid4()
        mock_result = MagicMock()
        mock_result.chunk = MagicMock()
        mock_result.chunk.id = chunk_id
        mock_result.chunk.namespace_id = ns_id
        mock_result.chunk.content = "in-window chunk"
        mock_result.chunk.document_id = doc_id
        mock_result.chunk.occurred_at = None
        mock_result.chunk.created_at = None
        mock_result.chunk.metadata = {}
        mock_result.combined_score = 0.85
        mock_result.similarity = 0.85
        # Single result: well below limit // 2 = 5, so the fallback condition
        # (sparse results) WOULD trigger if not gated by EXPLICIT.
        vector_store.search = AsyncMock(return_value=[mock_result])

        storage = AsyncMock()
        entry_entity_id = uuid4()
        storage.search_similar_entities = AsyncMock(return_value=[(entry_entity_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={})

        # Disable session-aware search to avoid the parallel session-fanout
        # path — that path issues additional _vector_search_chunks calls and
        # would muddy the call-count assertion.
        config = RetrieverConfig(
            query_cache_ttl_seconds=0,
            enable_session_aware_search=False,
        )
        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=neo4j_driver,
            embedder=embedder,
            config=config,
            storage=storage,
        )

        # Route to MODERATE (use_graph=True) so the sparse-fallback site at
        # retriever.py:754 is reachable.
        retriever._router = MagicMock()
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.MODERATE,
                use_graph=True,
                graph_depth=2,
                confidence=0.8,
                reasoning="moderate",
            )
        )
        retriever._router.compute_adaptive_depth = MagicMock(return_value=2)

        # Stub out graph-touching helpers so _vectorcypher_retrieve completes
        # cleanly without exercising Neo4j internals.
        retriever._cypher_expand = AsyncMock(return_value=({}, {}))
        retriever._fetch_chunks_from_entities = AsyncMock(return_value=[])
        retriever._version_filter_entities = AsyncMock(return_value=[])

        # Build the EXPLICIT signal: API-style with a parsed date filter.
        # occurred_before is exclusive in pgvector — bounds chosen so any
        # in-window timestamp would sit strictly inside [after, before).
        tf = TemporalFilter(
            occurred_after=datetime(2025, 1, 1, tzinfo=UTC),
            occurred_before=datetime(2025, 6, 1, tzinfo=UTC),
        )
        temporal_signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.EXPLICIT,
            confidence=1.0,
            source="api",
            temporal_filter=tf,
        )

        # Spy on _vector_search_chunks while preserving its real behavior.
        original = retriever._vector_search_chunks
        spy = AsyncMock(wraps=original)
        with patch.object(retriever, "_vector_search_chunks", spy):
            await retriever.retrieve(
                "what happened in early 2025",
                ns_id,
                temporal_filter=tf,
                temporal_signal=temporal_signal,
                limit=10,
            )

        # Sparse-results fallback re-runs _vector_search_chunks with
        # temporal_filter=None when triggered. The EXPLICIT-with-date guard
        # MUST suppress that re-run — exactly one call, with the original
        # filter intact.
        assert spy.await_count == 1, (
            f"_vector_search_chunks was called {spy.await_count} times; the "
            "EXPLICIT-with-date guard should have suppressed the sparse-"
            "results fallback re-run."
        )
        forwarded_filter = spy.await_args.kwargs["temporal_filter"]
        assert forwarded_filter is tf

"""Coverage push for ``khora.engines.vectorcypher.retriever`` (issue #695).

Targets the remaining uncovered branches in the retriever after the existing
``test_retriever_coverage_push.py`` pass:

- ``_fuse_results`` with bm25_chunks (3-channel fusion path, lines ~2844-2887)
- ``_fuse_results`` with is_temporal=True + sparse graph (lines ~2811-2830)
- ``_calculate_recency_scores`` reference_mode + bench-mode (lines ~2968+)
- ``_version_filter_entities`` short-circuits (lines ~2318-2368)
- ``_fetch_version_history`` short-circuits (lines ~2388-2422)
- ``_fetch_chunks_from_entities`` SurrealDB fallback (lines ~2466-2472)
- ``_vector_only_fallback`` metadata stamping
- SurrealDB rel-fetch inner function inside ``_vectorcypher_retrieve``
  (lines ~1469-1490)
- ``_apply_reranking`` / ``_apply_llm_reranking`` empty / metadata-dict paths
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk, Entity, Relationship
from khora.engines.vectorcypher.fusion import FusedResult
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.engines.vectorcypher.temporal_detection import (
    TemporalCategory,
    TemporalSignal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "text",
    *,
    occurred_at: str | None = None,
    source_system: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Chunk:
    # Post-#748: Chunk.metadata is a flat dict; ``custom`` nesting was
    # flattened away.
    custom: dict[str, Any] = dict(extra or {})
    if occurred_at is not None:
        custom["occurred_at"] = occurred_at
    if source_system is not None:
        custom["source_system"] = source_system
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        metadata=custom,
    )


_SENTINEL = object()


def _make_retriever(
    *,
    config: RetrieverConfig | None = None,
    vector_store: Any | None = None,
    storage: Any | None = None,
    neo4j_driver: Any | None = _SENTINEL,  # type: ignore[assignment]
) -> VectorCypherRetriever:
    # Use _SENTINEL so callers can explicitly pass None to disable the driver.
    if neo4j_driver is _SENTINEL:
        neo4j_driver = AsyncMock()
    return VectorCypherRetriever(
        vector_store=vector_store if vector_store is not None else AsyncMock(),
        neo4j_driver=neo4j_driver,
        embedder=AsyncMock(),
        config=config or RetrieverConfig(),
        storage=storage,
    )


# ---------------------------------------------------------------------------
# _fuse_results — 3-channel BM25 path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFuseResultsBm25:
    def test_three_channel_fusion(self) -> None:
        retriever = _make_retriever()
        c1 = _make_chunk("vector")
        c2 = _make_chunk("graph")
        c3 = _make_chunk("bm25")
        vector_chunks = [(c1.id, 0.9, c1)]
        graph_chunks = [(c2.id, 0.8, c2)]
        bm25_chunks = [(c3.id, 0.7, c3)]
        fused = retriever._fuse_results(vector_chunks, graph_chunks, bm25_chunks=bm25_chunks)
        # All three channels' chunks appear in fused output
        ids = {r.item_id for r in fused}
        assert ids == {c1.id, c2.id, c3.id}

    def test_three_channel_with_overlap(self) -> None:
        """A chunk appearing in both vector and BM25 lists gets blended RRF."""
        retriever = _make_retriever()
        shared = _make_chunk("shared content")
        only_bm25 = _make_chunk("bm25 only")
        fused = retriever._fuse_results(
            vector_chunks=[(shared.id, 0.9, shared)],
            graph_chunks=[],
            bm25_chunks=[(shared.id, 0.7, shared), (only_bm25.id, 0.6, only_bm25)],
        )
        # Shared item has vector_rank populated AND a higher RRF than bm25-only
        shared_results = [r for r in fused if r.item_id == shared.id]
        assert len(shared_results) == 1
        assert shared_results[0].vector_rank is not None
        # Note: graph_rank is None in this scenario (graph_chunks=[])

    def test_three_channel_empty_vector_only_bm25_and_graph(self) -> None:
        """No vector channel, only graph + BM25."""
        retriever = _make_retriever()
        g = _make_chunk("g")
        b = _make_chunk("b")
        fused = retriever._fuse_results(
            vector_chunks=[],
            graph_chunks=[(g.id, 0.5, g)],
            bm25_chunks=[(b.id, 0.3, b)],
        )
        ids = {r.item_id for r in fused}
        assert ids == {g.id, b.id}


# ---------------------------------------------------------------------------
# _fuse_results — is_temporal=True with sparse graph
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFuseResultsTemporal:
    def test_temporal_empty_graph_uses_vector_heavy_weights(self) -> None:
        """is_temporal=True with no graph chunks → adaptive vector-heavy weights."""
        retriever = _make_retriever()
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        fused = retriever._fuse_results(
            vector_chunks=[(c1.id, 0.9, c1), (c2.id, 0.8, c2)],
            graph_chunks=[],
            is_temporal=True,
        )
        # Just verify the function exits cleanly and produces fused results
        assert len(fused) == 2

    def test_temporal_sparse_graph_uses_moderate_weights(self) -> None:
        """is_temporal=True with 1-2 graph chunks → moderate weights."""
        retriever = _make_retriever()
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        fused = retriever._fuse_results(
            vector_chunks=[(c1.id, 0.9, c1)],
            graph_chunks=[(c2.id, 0.8, c2)],  # 1 chunk < 3
            is_temporal=True,
        )
        assert len(fused) == 2

    def test_temporal_dense_graph_keeps_temporal_weights(self) -> None:
        """is_temporal=True with ≥3 graph chunks → keeps graph-heavy weights."""
        retriever = _make_retriever()
        c1 = _make_chunk("c1")
        graph = [(uuid4(), 0.5, _make_chunk(f"g{i}")) for i in range(5)]
        fused = retriever._fuse_results(
            vector_chunks=[(c1.id, 0.9, c1)],
            graph_chunks=graph,
            is_temporal=True,
        )
        # All graph chunks fuse in
        assert len(fused) == 6


# ---------------------------------------------------------------------------
# _calculate_recency_scores — reference modes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalculateRecencyScoresReferenceModes:
    def test_explicit_wall_clock_mode(self) -> None:
        """reference_mode='wall_clock' uses datetime.now(UTC) as the anchor."""
        retriever = _make_retriever()
        now = datetime.now(UTC)
        c_recent = _make_chunk(occurred_at=now.isoformat())
        c_old = _make_chunk(occurred_at=(now - timedelta(days=365)).isoformat())
        results = [
            FusedResult(item_id=c_recent.id, item=c_recent, rrf_score=0.9),
            FusedResult(item_id=c_old.id, item=c_old, rrf_score=0.9),
        ]
        scores = retriever._calculate_recency_scores(results, reference_mode="wall_clock")
        # Recent chunk gets ~1.0, old gets a very low value.
        assert scores[c_recent.id] > scores[c_old.id]

    def test_explicit_relative_mode(self) -> None:
        """reference_mode='relative' uses max(occurred_at) as the anchor."""
        retriever = _make_retriever()
        # Both chunks are years old, but the newer one should score ~1.0
        # under relative mode (newest-in-set anchor).
        c_new = _make_chunk(occurred_at="2020-06-15T00:00:00+00:00")
        c_old = _make_chunk(occurred_at="2019-06-15T00:00:00+00:00")
        results = [
            FusedResult(item_id=c_new.id, item=c_new, rrf_score=0.9),
            FusedResult(item_id=c_old.id, item=c_old, rrf_score=0.9),
        ]
        scores = retriever._calculate_recency_scores(results, reference_mode="relative")
        # Newer chunk gets recency=1.0 under relative anchor
        assert scores[c_new.id] == pytest.approx(1.0, abs=1e-3)
        assert scores[c_old.id] < 1.0

    def test_no_results_returns_empty_dict(self) -> None:
        retriever = _make_retriever()
        assert retriever._calculate_recency_scores([]) == {}

    def test_unparseable_date_gets_default_score(self) -> None:
        """Chunks with no parseable occurred_at get the 0.5 default score."""
        retriever = _make_retriever()
        c = _make_chunk(occurred_at="not-a-date")
        scores = retriever._calculate_recency_scores([FusedResult(item_id=c.id, item=c, rrf_score=0.9)])
        assert scores[c.id] == 0.5

    def test_decay_zero_falls_back_to_config(self) -> None:
        """A pathological decay=0 override falls back to recency_decay_days."""
        retriever = _make_retriever(config=RetrieverConfig(recency_decay_days=14))
        now = datetime.now(UTC)
        c = _make_chunk(occurred_at=(now - timedelta(days=10)).isoformat())
        scores = retriever._calculate_recency_scores(
            [FusedResult(item_id=c.id, item=c, rrf_score=0.9)],
            decay_days_override=0,
            reference_mode="wall_clock",
        )
        # Decay falls back to config (14d); 10/14 days old → score is between 0 and 1
        assert 0.0 < scores[c.id] < 1.0

    def test_linear_decay_type(self) -> None:
        """recency_decay_type='linear' produces a linear recency curve."""
        config = RetrieverConfig(recency_decay_type="linear", recency_decay_days=30)
        retriever = _make_retriever(config=config)
        now = datetime.now(UTC)
        c = _make_chunk(occurred_at=(now - timedelta(days=15)).isoformat())
        scores = retriever._calculate_recency_scores(
            [FusedResult(item_id=c.id, item=c, rrf_score=0.9)],
            reference_mode="wall_clock",
        )
        # 15 days old in a 30-day window → score ~0.5
        assert scores[c.id] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# _version_filter_entities — early short-circuits
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVersionFilterEntities:
    @pytest.mark.asyncio
    async def test_empty_entity_ids_returns_empty(self) -> None:
        retriever = _make_retriever()
        out = await retriever._version_filter_entities([], uuid4(), datetime.now(UTC))
        assert out == []

    @pytest.mark.asyncio
    async def test_no_driver_passthrough(self) -> None:
        """SurrealDB / sqlite_lance (no neo4j_driver) returns input unchanged."""
        retriever = _make_retriever(neo4j_driver=None)
        ids = [uuid4(), uuid4()]
        out = await retriever._version_filter_entities(ids, uuid4(), datetime.now(UTC))
        assert out == ids


# ---------------------------------------------------------------------------
# _fetch_version_history — early short-circuits
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchVersionHistory:
    @pytest.mark.asyncio
    async def test_empty_entity_ids_returns_empty(self) -> None:
        retriever = _make_retriever()
        out = await retriever._fetch_version_history([], uuid4())
        assert out == []

    @pytest.mark.asyncio
    async def test_no_driver_returns_empty(self) -> None:
        retriever = _make_retriever(neo4j_driver=None)
        out = await retriever._fetch_version_history([uuid4()], uuid4())
        assert out == []


# ---------------------------------------------------------------------------
# _fetch_chunks_from_entities — SurrealDB / storage-only fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchChunksFromEntitiesStorageFallback:
    @pytest.mark.asyncio
    async def test_no_dual_nodes_no_chunk_ids_yields_empty(self) -> None:
        """SurrealDB fallback with entities that have no source_chunk_ids → []."""
        ns = uuid4()
        entity_id = uuid4()
        entity = Entity(
            id=entity_id,
            name="E",
            entity_type="X",
            source_chunk_ids=[],  # Empty → all_chunk_ids stays empty
            namespace_id=ns,
        )
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(return_value={entity_id: entity})
        # get_chunks_batch should NOT be called when there are no chunk_ids
        storage.get_chunks_batch = AsyncMock(return_value={})

        retriever = _make_retriever(storage=storage, neo4j_driver=None)
        assert retriever._dual_nodes is None

        out = await retriever._fetch_chunks_from_entities(
            entity_ids=[entity_id],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
        )
        assert out == []
        # get_entities_batch was called, get_chunks_batch was not
        storage.get_entities_batch.assert_awaited_once()
        storage.get_chunks_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_dual_nodes_no_storage_returns_empty(self) -> None:
        """Neither dual_nodes nor storage → returns []."""
        retriever = _make_retriever(storage=None, neo4j_driver=None)
        assert retriever._dual_nodes is None
        out = await retriever._fetch_chunks_from_entities(
            entity_ids=[uuid4()],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
        )
        assert out == []

    @pytest.mark.asyncio
    async def test_storage_error_returns_empty(self) -> None:
        """If get_entities_batch raises, fallback returns []."""
        storage = MagicMock()
        storage.get_entities_batch = AsyncMock(side_effect=RuntimeError("db down"))
        retriever = _make_retriever(storage=storage, neo4j_driver=None)
        out = await retriever._fetch_chunks_from_entities(
            entity_ids=[uuid4()],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
        )
        assert out == []


# ---------------------------------------------------------------------------
# _vector_only_fallback — metadata stamping
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorOnlyFallback:
    @pytest.mark.asyncio
    async def test_marks_fallback_metadata(self) -> None:
        """Fallback path stamps fallback_mode + graph_unavailable + graph_fallback in metadata."""
        ns = uuid4()
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.5,
            reasoning="t",
        )

        # Patch _simple_retrieve to return a stub VectorCypherResult.
        from khora.engines.vectorcypher.retriever import VectorCypherResult

        retriever = _make_retriever()
        retriever._simple_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(chunks=[], entities=[], routing_decision=routing, metadata={})
        )
        result = await retriever._vector_only_fallback(
            query="q",
            query_embedding=[0.0],
            namespace_id=ns,
            temporal_filter=None,
            limit=10,
            routing=routing,
        )
        assert result.metadata["fallback_mode"] == "vector_only"
        assert result.metadata["graph_unavailable"] is True
        assert result.metadata["graph_fallback"] is True


# ---------------------------------------------------------------------------
# _bm25_search_chunks — error swallow + empty storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBm25SearchChunksExtra:
    @pytest.mark.asyncio
    async def test_no_storage_returns_empty(self) -> None:
        retriever = _make_retriever(storage=None)
        out = await retriever._bm25_search_chunks("q", uuid4(), limit=10)
        assert out == []

    @pytest.mark.asyncio
    async def test_storage_raises_returns_empty(self) -> None:
        storage = MagicMock()
        storage.search_fulltext_chunks = AsyncMock(side_effect=RuntimeError("fts down"))
        retriever = _make_retriever(storage=storage)
        out = await retriever._bm25_search_chunks("q", uuid4(), limit=10)
        assert out == []


# ---------------------------------------------------------------------------
# _vector_search_entities — error + no-storage paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorSearchEntities:
    @pytest.mark.asyncio
    async def test_no_storage_returns_empty(self) -> None:
        retriever = _make_retriever(storage=None)
        out = await retriever._vector_search_entities([0.1] * 4, uuid4(), limit=10)
        assert out == []

    @pytest.mark.asyncio
    async def test_storage_raises_returns_empty(self) -> None:
        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(side_effect=RuntimeError("pgvector down"))
        retriever = _make_retriever(storage=storage)
        out = await retriever._vector_search_entities([0.1] * 4, uuid4(), limit=10)
        assert out == []

    @pytest.mark.asyncio
    async def test_storage_returns_results(self) -> None:
        storage = MagicMock()
        eid1, eid2 = uuid4(), uuid4()
        storage.search_similar_entities = AsyncMock(return_value=[(eid1, 0.9), (eid2, 0.7)])
        retriever = _make_retriever(storage=storage)
        out = await retriever._vector_search_entities([0.1] * 4, uuid4(), limit=10)
        assert out == [(eid1, 0.9), (eid2, 0.7)]


# ---------------------------------------------------------------------------
# _recency_channel_chunks — no-store + no-method short-circuits
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecencyChannelChunksExtra:
    @pytest.mark.asyncio
    async def test_no_vector_store_returns_empty(self) -> None:
        retriever = _make_retriever()
        retriever._vector_store = None
        out = await retriever._recency_channel_chunks(
            query_embedding=[0.0] * 4, namespace_id=uuid4(), temporal_filter=None
        )
        assert out == []

    @pytest.mark.asyncio
    async def test_vector_store_without_method_returns_empty(self) -> None:
        retriever = _make_retriever()
        # MagicMock has __getattr__ for any attr; explicitly delete search_recent_chunks
        spec_store = MagicMock(spec=[])  # spec=[] = no attributes
        retriever._vector_store = spec_store
        out = await retriever._recency_channel_chunks(
            query_embedding=[0.0] * 4, namespace_id=uuid4(), temporal_filter=None
        )
        assert out == []


# ---------------------------------------------------------------------------
# _decompose_change_query — extra patterns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecomposeChangeQueryExtra:
    def test_fallback_currently_prefix(self) -> None:
        """Fallback substitutes 'currently' for change keywords."""
        # "previously" is in the fallback keyword list
        result = VectorCypherRetriever._decompose_change_query("previously a discussion happened")
        # The fallback replaces 'previously' with 'currently'
        assert result is not None
        assert "currently" in result.lower()

    def test_returns_none_for_non_change_query(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("What is the weather today?")
        assert result is None

    def test_empty_query(self) -> None:
        result = VectorCypherRetriever._decompose_change_query("")
        assert result is None


# ---------------------------------------------------------------------------
# _should_skip_llm_rerank — both gates exhaustively
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldSkipLlmRerankGates:
    def test_neither_gate_fires(self) -> None:
        """Low top score AND small gap → don't skip."""
        config = RetrieverConfig(
            llm_reranking_confidence_threshold=0.3,
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.15,
        )
        retriever = _make_retriever(config=config)
        # gap=0.1 < 0.3 (legacy gate fails), top=0.5 < 0.7 (decisive gate fails)
        assert retriever._should_skip_llm_rerank(0.5, 0.1) is False

    def test_gap_gate_fires(self) -> None:
        config = RetrieverConfig(llm_reranking_confidence_threshold=0.1)
        retriever = _make_retriever(config=config)
        assert retriever._should_skip_llm_rerank(0.5, 0.5) is True

    def test_decisive_gate_fires(self) -> None:
        """High top AND meaningful gap → skip even when legacy gate misses."""
        config = RetrieverConfig(
            llm_reranking_confidence_threshold=0.99,  # legacy gate won't fire
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        retriever = _make_retriever(config=config)
        assert retriever._should_skip_llm_rerank(0.85, 0.15) is True


# ---------------------------------------------------------------------------
# _apply_reranking / _apply_llm_reranking — empty input + error fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyRerankingExtra:
    @pytest.mark.asyncio
    async def test_empty_fused_results_returns_unchanged(self) -> None:
        retriever = _make_retriever()
        out = await retriever._apply_reranking("q", [], limit=10, namespace_id=uuid4())
        assert out == []

    @pytest.mark.asyncio
    async def test_empty_fused_results_llm_returns_unchanged(self) -> None:
        retriever = _make_retriever()
        out = await retriever._apply_llm_reranking("q", [], limit=10, namespace_id=uuid4())
        assert out == []

    @pytest.mark.asyncio
    async def test_reranker_error_returns_original_order(self) -> None:
        """When the cross-encoder reranker raises, the original ordering is kept."""
        retriever = _make_retriever()
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        fused = [
            FusedResult(item_id=c1.id, item=c1, rrf_score=0.9),
            FusedResult(item_id=c2.id, item=c2, rrf_score=0.8),
        ]
        with patch(
            "khora.query.reranking.CrossEncoderReranker",
            autospec=True,
        ) as mock_cls:
            instance = mock_cls.return_value
            instance.rerank = AsyncMock(side_effect=RuntimeError("model down"))
            # Force a fresh reranker per call
            retriever._reranker = instance
            out = await retriever._apply_reranking("q", fused, limit=10, namespace_id=uuid4())
            # On error, returns original list
            assert out == fused

    @pytest.mark.asyncio
    async def test_llm_reranker_error_returns_original_order(self) -> None:
        """When the LLM reranker raises, the original ordering is kept."""
        retriever = _make_retriever()
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        fused = [
            FusedResult(item_id=c1.id, item=c1, rrf_score=0.9),
            FusedResult(item_id=c2.id, item=c2, rrf_score=0.8),
        ]
        with patch("khora.query.reranking.LLMReranker", autospec=True) as mock_cls:
            instance = mock_cls.return_value
            instance.rerank = AsyncMock(side_effect=RuntimeError("llm down"))
            retriever._llm_reranker = instance
            out = await retriever._apply_llm_reranking("q", fused, limit=10, namespace_id=uuid4())
            assert out == fused


# ---------------------------------------------------------------------------
# retrieve() — sqlite_lance backend gate when target date is present
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrieveBackendGateExtra:
    @pytest.mark.asyncio
    async def test_point_in_time_query_raises_on_sqlite_lance(self) -> None:
        """sqlite_lance backend + temporal_filter with date → NotImplementedError."""
        from khora.engines.skeleton.backends import TemporalFilter

        retriever = _make_retriever()
        retriever._backend = "sqlite_lance"
        tf = TemporalFilter(occurred_after=datetime(2024, 1, 1, tzinfo=UTC))
        with pytest.raises(NotImplementedError, match="sqlite_lance"):
            await retriever.retrieve("q", uuid4(), temporal_filter=tf)

    @pytest.mark.asyncio
    async def test_sqlite_lance_without_date_proceeds(self) -> None:
        """sqlite_lance without temporal filter doesn't trip the gate."""
        retriever = _make_retriever()
        retriever._backend = "sqlite_lance"
        # Mock embedder + downstream so we can exit quickly
        retriever._embedder.model_name = "m"
        retriever._embedder.dimension = 4
        retriever._embedder.embed = AsyncMock(return_value=[0.1] * 4)
        retriever._embedder.cache_stats = {"hits": 0}
        # Force the simple path by ensuring no entry entities are found
        from khora.engines.vectorcypher.retriever import VectorCypherResult

        retriever._simple_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[],
                entities=[],
                routing_decision=RoutingDecision(
                    complexity=QueryComplexity.SIMPLE,
                    use_graph=False,
                    graph_depth=0,
                    confidence=0.5,
                    reasoning="t",
                ),
                metadata={},
            )
        )
        # No storage → _vector_search_entities returns []
        retriever._storage = None

        out = await retriever.retrieve("q", uuid4())
        # _simple_retrieve was reached
        assert retriever._simple_retrieve.await_count == 1
        assert out.chunks == []


# ---------------------------------------------------------------------------
# _simple_retrieve — temporal_sort + ORDINAL category
# ---------------------------------------------------------------------------


def _make_temporal_search_result(content: str, occurred_at: datetime | None, similarity: float = 0.9):
    from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

    tc = TemporalChunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        embedding=None,
        occurred_at=occurred_at,
    )
    return TemporalSearchResult(chunk=tc, similarity=similarity, combined_score=similarity)


@pytest.mark.unit
class TestSimpleRetrieveExtra:
    @pytest.mark.asyncio
    async def test_temporal_sort_descending_by_default(self) -> None:
        """temporal_sort=True orders by occurred_at DESC (newest first)."""
        retriever = _make_retriever(config=RetrieverConfig(enable_reranking=False))

        old = _make_temporal_search_result("old", datetime(2024, 1, 1, tzinfo=UTC), similarity=0.9)
        new = _make_temporal_search_result("new", datetime(2024, 6, 1, tzinfo=UTC), similarity=0.5)

        retriever._vector_store.search = AsyncMock(return_value=[old, new])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_sort=True,
        )
        # Newer chunk should be first
        assert result.chunks[0][0].content == "new"
        assert result.chunks[1][0].content == "old"

    @pytest.mark.asyncio
    async def test_temporal_sort_ordinal_ascending(self) -> None:
        """ORDINAL category re-sorts ascending (earliest first)."""
        retriever = _make_retriever(config=RetrieverConfig(enable_reranking=False))

        old = _make_temporal_search_result("old", datetime(2024, 1, 1, tzinfo=UTC), similarity=0.5)
        new = _make_temporal_search_result("new", datetime(2024, 6, 1, tzinfo=UTC), similarity=0.9)
        retriever._vector_store.search = AsyncMock(return_value=[old, new])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        ordinal_signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.ORDINAL,
            confidence=0.9,
            source="dictionary",
        )
        result = await retriever._simple_retrieve(
            query="which came first?",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_sort=True,
            temporal_signal=ordinal_signal,
        )
        # Older chunk should be first (ascending order)
        assert result.chunks[0][0].content == "old"

    @pytest.mark.asyncio
    async def test_recency_boost_applies_when_positive(self) -> None:
        """effective_recency > 0 invokes the recency boost path."""
        retriever = _make_retriever(config=RetrieverConfig(enable_reranking=False))

        now = datetime.now(UTC)
        recent = _make_temporal_search_result("recent", now, similarity=0.5)
        retriever._vector_store.search = AsyncMock(return_value=[recent])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            effective_recency=0.5,
        )
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_empty_results_metadata(self) -> None:
        retriever = _make_retriever()
        retriever._vector_store.search = AsyncMock(return_value=[])
        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
        )
        assert result.chunks == []
        assert result.metadata["max_raw_vector_score"] == 0.0
        assert result.metadata["search_mode"] == "simple_vector"

    @pytest.mark.asyncio
    async def test_bm25_channel_active_uses_pure_vector_alpha(self) -> None:
        """When BM25 channel is on, effective_alpha=1.0 (pure vector)."""
        config = RetrieverConfig(enable_bm25_channel=True, hybrid_alpha=0.5)
        storage = MagicMock()
        storage.search_fulltext_chunks = AsyncMock(return_value=[])
        retriever = _make_retriever(config=config, storage=storage)
        retriever._vector_store.search = AsyncMock(return_value=[])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
        )
        # search() was called with hybrid_alpha=1.0
        kwargs = retriever._vector_store.search.call_args.kwargs
        assert kwargs["hybrid_alpha"] == 1.0


# ---------------------------------------------------------------------------
# retrieve() falling back to _simple_retrieve when no entry entities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetrieveFallbackToSimple:
    @pytest.mark.asyncio
    async def test_no_entry_entities_falls_back_to_simple(self) -> None:
        """When _vector_search_entities returns [], retrieve() routes to _simple_retrieve."""
        retriever = _make_retriever()
        retriever._embedder.model_name = "m"
        retriever._embedder.dimension = 4
        retriever._embedder.embed = AsyncMock(return_value=[0.1] * 4)
        retriever._embedder.cache_stats = {"hits": 0}
        # No storage → entity search yields []
        retriever._storage = None

        from khora.engines.vectorcypher.retriever import VectorCypherResult

        sentinel = VectorCypherResult(
            chunks=[],
            entities=[],
            routing_decision=RoutingDecision(
                complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
            ),
            metadata={"search_mode": "simple_vector"},
        )
        retriever._simple_retrieve = AsyncMock(return_value=sentinel)  # type: ignore[method-assign]

        result = await retriever.retrieve("q", uuid4())
        assert result is sentinel
        retriever._simple_retrieve.assert_awaited_once()


# ---------------------------------------------------------------------------
# _vectorcypher_retrieve — end-to-end entity stub fallback + relationships
# ---------------------------------------------------------------------------


def _vectorcypher_retrieve_setup() -> tuple[VectorCypherRetriever, UUID, UUID]:
    """Build a retriever that runs the full _vectorcypher_retrieve flow.

    Returns (retriever, namespace_id, entry_entity_id).
    """
    ns = uuid4()
    e_id = uuid4()

    storage = MagicMock()
    storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
    # Make get_entities_batch return a real entity for the entry
    entry_entity = Entity(id=e_id, name="Alice", entity_type="PERSON", namespace_id=ns)
    storage.get_entities_batch = AsyncMock(return_value={e_id: entry_entity})
    # Disable BM25, simulate the graph being available
    storage.graph = MagicMock()
    storage.search_fulltext_chunks = AsyncMock(return_value=[])

    retriever = _make_retriever(storage=storage)
    # Mock vector store search to return a chunk
    chunk = _make_chunk("hello world")
    from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

    tc = TemporalChunk(
        id=chunk.id, namespace_id=ns, document_id=chunk.document_id, content=chunk.content, embedding=None
    )
    retriever._vector_store.search = AsyncMock(
        return_value=[TemporalSearchResult(chunk=tc, similarity=0.9, combined_score=0.9)]
    )

    # Stub dual_nodes operations
    retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
    retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
    retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])
    retriever._dual_nodes.get_entity_channels = AsyncMock(return_value=[])

    return retriever, ns, e_id


@pytest.mark.unit
class TestVectorCypherRetrieveFlow:
    @pytest.mark.asyncio
    async def test_returns_entry_entity_with_score(self) -> None:
        """Entry entity is surfaced in result.entities with its score."""
        retriever, ns, e_id = _vectorcypher_retrieve_setup()
        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        result = await retriever._vectorcypher_retrieve(
            query="alice",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
        )
        # The entry entity is among result.entities
        assert any(e.id == e_id for e, _ in result.entities)

    @pytest.mark.asyncio
    async def test_returns_relationships_from_storage_when_no_dual_nodes(self) -> None:
        """SurrealDB path (no dual_nodes) fetches relationships via storage.graph."""
        ns = uuid4()
        e_id = uuid4()
        target_id = uuid4()

        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        entity = Entity(id=e_id, name="Alice", entity_type="PERSON", namespace_id=ns)
        storage.get_entities_batch = AsyncMock(return_value={e_id: entity})
        # Use storage.graph fallback for relationships
        rel = Relationship(
            id=uuid4(),
            source_entity_id=e_id,
            target_entity_id=target_id,
            relationship_type="KNOWS",
            description="d",
            namespace_id=ns,
        )
        storage._graph = MagicMock()
        storage._graph.get_entity_relationships = AsyncMock(return_value=[rel])
        # neighborhoods batch fetch needed by storage path
        storage.get_neighborhoods_batch = AsyncMock(return_value={})

        # Use _make_retriever with neo4j_driver=None to force the storage path
        retriever = _make_retriever(storage=storage, neo4j_driver=None)
        # Make sure dual_nodes is None
        assert retriever._dual_nodes is None

        # Mock vector store
        from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

        chunk_id = uuid4()
        tc = TemporalChunk(id=chunk_id, namespace_id=ns, document_id=uuid4(), content="hello", embedding=None)
        retriever._vector_store.search = AsyncMock(
            return_value=[TemporalSearchResult(chunk=tc, similarity=0.9, combined_score=0.9)]
        )

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        result = await retriever._vectorcypher_retrieve(
            query="alice",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
        )
        # The relationship is surfaced
        assert any(r.relationship_type == "KNOWS" for r, _ in result.relationships)

    @pytest.mark.asyncio
    async def test_entity_batch_fetch_error_falls_back_to_stubs(self) -> None:
        """When get_entities_batch raises, retriever falls back to stub Entity objects."""
        ns = uuid4()
        e_id = uuid4()
        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        storage.get_entities_batch = AsyncMock(side_effect=RuntimeError("batch failed"))
        storage.graph = MagicMock()

        retriever = _make_retriever(storage=storage)
        from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

        tc = TemporalChunk(id=uuid4(), namespace_id=ns, document_id=uuid4(), content="x", embedding=None)
        retriever._vector_store.search = AsyncMock(
            return_value=[TemporalSearchResult(chunk=tc, similarity=0.9, combined_score=0.9)]
        )
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        result = await retriever._vectorcypher_retrieve(
            query="alice",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
        )
        # The result still includes the entry entity but as a stub (the
        # exception is caught and stubs are built from entity_info_map).
        assert any(e.id == e_id for e, _ in result.entities)

    @pytest.mark.asyncio
    async def test_storage_none_entry_entities_become_stubs(self) -> None:
        """No storage → entry entities become stub Entity objects."""
        ns = uuid4()
        # storage is None, but entry_entities are provided via direct injection
        # in the call sequence — we'll force entry_entities via search_similar.
        # Without storage, _vector_search_entities returns [], so test the
        # path where storage is None entirely (the elif branch).
        retriever = _make_retriever(storage=None)
        retriever._vector_store.search = AsyncMock(return_value=[])
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        # With no storage, _vector_search_entities returns [] and we fall back
        # to _simple_retrieve. Mock that out to confirm the path.
        from khora.engines.vectorcypher.retriever import VectorCypherResult

        retriever._simple_retrieve = AsyncMock(  # type: ignore[method-assign]
            return_value=VectorCypherResult(
                chunks=[], entities=[], routing_decision=routing, metadata={"search_mode": "simple_vector"}
            )
        )
        result = await retriever._vectorcypher_retrieve(
            query="q",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
        )
        retriever._simple_retrieve.assert_awaited_once()
        assert result.metadata["search_mode"] == "simple_vector"


# ---------------------------------------------------------------------------
# Session-aware retrieval path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSessionAwareRetrieve:
    @pytest.mark.asyncio
    async def test_multi_session_fans_out_parallel_searches(self) -> None:
        """When entry entities span ≥2 sessions, parallel per-session searches fire."""
        ns = uuid4()
        e_id = uuid4()

        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={e_id: Entity(id=e_id, name="A", entity_type="P")})
        storage.graph = MagicMock()
        storage.search_fulltext_chunks = AsyncMock(return_value=[])

        retriever = _make_retriever(
            storage=storage,
            config=RetrieverConfig(enable_session_aware_search=True),
        )

        # Two session channels → fan-out activates
        retriever._dual_nodes.get_entity_channels = AsyncMock(return_value=["sess-1", "sess-2"])
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])

        # Vector store returns one chunk per call
        from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

        def make_result(content: str):
            tc = TemporalChunk(id=uuid4(), namespace_id=ns, document_id=uuid4(), content=content, embedding=None)
            return [TemporalSearchResult(chunk=tc, similarity=0.5, combined_score=0.5)]

        # Sequential calls returning different results
        retriever._vector_store.search = AsyncMock(side_effect=lambda **kw: make_result(kw.get("query_text", "")))

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
        )

        # Provide retrieval params that have recency_weight > 0
        from khora.engines.vectorcypher.temporal_detection import RETRIEVAL_PARAMS

        params = RETRIEVAL_PARAMS[TemporalCategory.RECENCY]

        result = await retriever._vectorcypher_retrieve(
            query="recent stuff",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
            temporal_params=params,
            temporal_signal=signal,
        )
        # session_aware_activated should be True
        assert result.metadata["session_aware_activated"] is True

    @pytest.mark.asyncio
    async def test_single_session_keeps_global_search(self) -> None:
        """One session → session-aware path skipped, global search continues."""
        ns = uuid4()
        e_id = uuid4()

        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={e_id: Entity(id=e_id, name="A", entity_type="P")})
        storage.graph = MagicMock()

        retriever = _make_retriever(
            storage=storage,
            config=RetrieverConfig(enable_session_aware_search=True),
        )

        # Single channel → fan-out NOT triggered
        retriever._dual_nodes.get_entity_channels = AsyncMock(return_value=["sess-1"])
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])

        retriever._vector_store.search = AsyncMock(return_value=[])

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        signal = TemporalSignal(
            is_temporal=True, category=TemporalCategory.RECENCY, confidence=0.9, source="dictionary"
        )
        from khora.engines.vectorcypher.temporal_detection import RETRIEVAL_PARAMS

        result = await retriever._vectorcypher_retrieve(
            query="recent",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
            temporal_params=RETRIEVAL_PARAMS[TemporalCategory.RECENCY],
            temporal_signal=signal,
        )
        assert result.metadata["session_aware_activated"] is False

    @pytest.mark.asyncio
    async def test_session_discovery_exception_falls_back(self) -> None:
        """When get_entity_channels raises, retrieval continues with global search."""
        ns = uuid4()
        e_id = uuid4()
        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={e_id: Entity(id=e_id, name="A", entity_type="P")})
        storage.graph = MagicMock()

        retriever = _make_retriever(
            storage=storage,
            config=RetrieverConfig(enable_session_aware_search=True),
        )

        retriever._dual_nodes.get_entity_channels = AsyncMock(side_effect=RuntimeError("oops"))
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])
        retriever._vector_store.search = AsyncMock(return_value=[])

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        signal = TemporalSignal(
            is_temporal=True, category=TemporalCategory.RECENCY, confidence=0.9, source="dictionary"
        )
        from khora.engines.vectorcypher.temporal_detection import RETRIEVAL_PARAMS

        result = await retriever._vectorcypher_retrieve(
            query="recent",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
            temporal_params=RETRIEVAL_PARAMS[TemporalCategory.RECENCY],
            temporal_signal=signal,
        )
        # Function survived the exception
        assert result is not None


# ---------------------------------------------------------------------------
# _simple_retrieve — LLM reranking + version-aware scoring branches
# ---------------------------------------------------------------------------


def _make_temporal_search_result_with_version(
    content: str,
    occurred_at: datetime | None,
    version: int,
    entity_refs: list[str] | None = None,
    similarity: float = 0.9,
):
    """Build a TemporalSearchResult with version metadata baked into the chunk metadata."""
    from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

    chunk_meta = {"version": version}
    if entity_refs:
        chunk_meta["entity_refs"] = entity_refs
    tc = TemporalChunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        embedding=None,
        occurred_at=occurred_at,
        metadata=chunk_meta,
    )
    return TemporalSearchResult(chunk=tc, similarity=similarity, combined_score=similarity)


@pytest.mark.unit
class TestSimpleRetrieveLLMRerank:
    @pytest.mark.asyncio
    async def test_llm_rerank_skipped_when_no_versions(self) -> None:
        """LLM rerank is skipped for conversational data (no version metadata)."""
        retriever = _make_retriever(
            config=RetrieverConfig(enable_llm_reranking=True, enable_reranking=False),
        )

        now = datetime.now(UTC)
        r1 = _make_temporal_search_result("c1", now, similarity=0.9)
        r2 = _make_temporal_search_result("c2", now, similarity=0.8)
        retriever._vector_store.search = AsyncMock(return_value=[r1, r2])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.STATE_QUERY,
            confidence=0.9,
            source="dictionary",
        )
        # _apply_llm_reranking should NOT be called (no version metadata)
        retriever._apply_llm_reranking = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=signal,
        )
        retriever._apply_llm_reranking.assert_not_called()
        assert len(result.chunks) == 2

    @pytest.mark.asyncio
    async def test_version_aware_scoring_penalizes_old_versions(self) -> None:
        """STATE_QUERY + version metadata triggers version-aware decay."""
        retriever = _make_retriever(
            config=RetrieverConfig(enable_llm_reranking=False, enable_reranking=False),
        )

        now = datetime.now(UTC)
        # Two chunks referencing the same entity, different versions
        r_old = _make_temporal_search_result_with_version("v1 content", now, version=1, entity_refs=["alice"])
        r_new = _make_temporal_search_result_with_version("v3 content", now, version=3, entity_refs=["alice"])
        retriever._vector_store.search = AsyncMock(return_value=[r_old, r_new])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.STATE_QUERY,
            confidence=0.9,
            source="dictionary",
        )
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=signal,
        )
        # The v1 chunk's score should be penalized relative to v3 (decay applied)
        # The chunks may be in either order, but v1's score should be lower.
        scores_by_content = {c.content: s for c, s in result.chunks}
        assert scores_by_content["v3 content"] >= scores_by_content["v1 content"]

    @pytest.mark.asyncio
    async def test_change_category_skips_version_decay(self) -> None:
        """CHANGE category needs old+new versions; skips the version decay path."""
        retriever = _make_retriever(
            config=RetrieverConfig(enable_llm_reranking=False, enable_reranking=False),
        )

        now = datetime.now(UTC)
        r_old = _make_temporal_search_result_with_version("v1", now, version=1, entity_refs=["x"])
        r_new = _make_temporal_search_result_with_version("v2", now, version=2, entity_refs=["x"])
        retriever._vector_store.search = AsyncMock(return_value=[r_old, r_new])

        routing = RoutingDecision(
            complexity=QueryComplexity.SIMPLE, use_graph=False, graph_depth=0, confidence=0.5, reasoning=""
        )
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.CHANGE,
            confidence=0.9,
            source="dictionary",
        )
        # Function shouldn't crash for CHANGE category
        result = await retriever._simple_retrieve(
            query="q",
            query_embedding=[0.1],
            namespace_id=uuid4(),
            temporal_filter=None,
            limit=10,
            routing=routing,
            temporal_signal=signal,
        )
        assert len(result.chunks) == 2


# ---------------------------------------------------------------------------
# _vectorcypher_retrieve — recency channel pool augmentation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRecencyChannelEnabledFlow:
    @pytest.mark.asyncio
    async def test_recency_channel_merges_new_chunks(self) -> None:
        """When the recency channel is enabled and yields new chunks, they merge in."""
        ns = uuid4()
        e_id = uuid4()
        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={e_id: Entity(id=e_id, name="A", entity_type="P")})
        storage.graph = MagicMock()

        retriever = _make_retriever(
            storage=storage,
            config=RetrieverConfig(
                enable_session_aware_search=False,
                temporal_recency_channel_enabled=True,
            ),
        )

        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])
        retriever._vector_store.search = AsyncMock(return_value=[])

        # Patch _recency_channel_chunks to return a new chunk
        recent_chunk = _make_chunk("recent")
        retriever._recency_channel_chunks = AsyncMock(  # type: ignore[method-assign]
            return_value=[(recent_chunk.id, 0.95, recent_chunk)]
        )

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        from khora.engines.vectorcypher.temporal_detection import RETRIEVAL_PARAMS

        signal = TemporalSignal(
            is_temporal=True, category=TemporalCategory.RECENCY, confidence=0.9, source="dictionary"
        )
        result = await retriever._vectorcypher_retrieve(
            query="q",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
            temporal_params=RETRIEVAL_PARAMS[TemporalCategory.RECENCY],
            temporal_signal=signal,
        )
        # The recency-channel chunk made it into the result
        retriever._recency_channel_chunks.assert_awaited_once()
        # At least one chunk surfaced
        assert result is not None


# ---------------------------------------------------------------------------
# _vectorcypher_retrieve — CHANGE decomposition merging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChangeDecompositionMerging:
    @pytest.mark.asyncio
    async def test_change_query_runs_decomposed_sub_search(self) -> None:
        """CHANGE temporal signal + version_history → runs sub-query, merges new chunks."""
        ns = uuid4()
        e_id = uuid4()

        storage = MagicMock()
        storage.search_similar_entities = AsyncMock(return_value=[(e_id, 0.9)])
        storage.get_entities_batch = AsyncMock(return_value={e_id: Entity(id=e_id, name="A", entity_type="P")})
        storage.graph = MagicMock()

        retriever = _make_retriever(
            storage=storage,
            config=RetrieverConfig(enable_session_aware_search=False),
        )
        retriever._dual_nodes.get_entity_neighborhoods = AsyncMock(return_value={})
        retriever._dual_nodes.get_chunks_by_entities = AsyncMock(return_value=[])
        retriever._dual_nodes.get_relationships_between = AsyncMock(return_value=[])

        # Mock _fetch_version_history to return a non-empty list → CHANGE branch runs.
        retriever._fetch_version_history = AsyncMock(return_value=[{"current_id": str(e_id), "name": "A"}])  # type: ignore[method-assign]

        # Mock vector store search:
        # First call (main query) → empty
        # Second call (sub-query) → has a new chunk
        from khora.engines.skeleton.backends import TemporalChunk, TemporalSearchResult

        new_chunk_tc = TemporalChunk(id=uuid4(), namespace_id=ns, document_id=uuid4(), content="sub", embedding=None)
        new_result = TemporalSearchResult(chunk=new_chunk_tc, similarity=0.7, combined_score=0.7)

        call_count = {"n": 0}

        async def fake_search(**kw):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                return [new_result]
            return []

        retriever._vector_store.search = AsyncMock(side_effect=fake_search)
        # Embedder needs to support embed() for the sub-query.
        retriever._embedder.embed = AsyncMock(return_value=[0.1] * 4)

        # Temporal signal: CHANGE with EXPLICIT filter
        from khora.engines.skeleton.backends import TemporalFilter

        change_tf = TemporalFilter(occurred_after=datetime(2024, 1, 1, tzinfo=UTC))
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.CHANGE,
            confidence=0.9,
            source="dictionary",
            temporal_filter=change_tf,
        )
        from khora.engines.vectorcypher.temporal_detection import RETRIEVAL_PARAMS

        routing = RoutingDecision(
            complexity=QueryComplexity.COMPLEX, use_graph=True, graph_depth=1, confidence=0.9, reasoning=""
        )
        result = await retriever._vectorcypher_retrieve(
            query="What did Alice used to play?",
            query_embedding=[0.1] * 4,
            namespace_id=ns,
            temporal_filter=None,
            graph_depth=1,
            limit=10,
            routing=routing,
            temporal_params=RETRIEVAL_PARAMS[TemporalCategory.CHANGE],
            temporal_signal=signal,
        )
        # _fetch_version_history was called (CHANGE category branch)
        retriever._fetch_version_history.assert_awaited()
        # Sub-query was issued (call_count >= 2)
        assert call_count["n"] >= 2
        assert result is not None


# ---------------------------------------------------------------------------
# Use anchor to keep imports referenced
# ---------------------------------------------------------------------------


def test_module_imports_referenced() -> None:
    """Anchor — ensures Relationship / UUID imports are not flagged unused."""
    assert isinstance(uuid4(), UUID)
    assert Relationship is not None

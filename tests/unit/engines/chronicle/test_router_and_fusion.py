"""Chronicle #6: tests for ``QueryComplexityRouter`` integration and weighted-normalized fusion.

The router classifies queries SIMPLE / MODERATE / COMPLEX. SIMPLE chronicle
queries skip BM25 + entity channels; MODERATE / COMPLEX run all four. The
temporal channel is ALWAYS preserved on chronicle (its differentiator) — even
when the router would otherwise pick SIMPLE, an explicit ``temporal_filter``
forces the router into MODERATE via the ``temporal_signal`` boost.

The fusion call site swaps rank-only ``reciprocal_rank_fusion`` for
``_weighted_normalized_rrf_multi`` so per-channel score scales (BM25 vs cosine)
no longer dominate the fused ordering.

All tests use mock storage coordinators — no real DB / LLM round-trips.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, ChunkMetadata
from khora.engines.chronicle.engine import (
    ChronicleEngine,
    _weighted_normalized_rrf_multi,
)
from khora.query import SearchMode
from khora.query.router import QueryComplexity, RoutingDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "x", *, namespace_id: UUID | None = None) -> Chunk:
    ns_id = namespace_id or uuid4()
    document_id = uuid4()
    return Chunk(
        namespace_id=ns_id,
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
    )


class _RecordingCoordinator:
    """Coordinator double that records which channel-feeding methods were called.

    Each method returns the canned results passed at construction. The flags
    ``search_similar_chunks_calls`` / ``search_fulltext_chunks_calls`` /
    ``search_similar_entities_calls`` let tests assert which channels were
    actually invoked.
    """

    def __init__(
        self,
        *,
        semantic_results: list[tuple[Chunk, float]] | None = None,
        bm25_results: list[tuple[Chunk, float]] | None = None,
        entity_search_results: list[tuple[UUID, float]] | None = None,
    ) -> None:
        self._semantic_results = semantic_results or []
        self._bm25_results = bm25_results or []
        self._entity_search_results = entity_search_results or []

        self.search_similar_chunks_calls: list[dict[str, Any]] = []
        self.search_fulltext_chunks_calls: list[dict[str, Any]] = []
        self.search_similar_entities_calls: list[dict[str, Any]] = []
        self.query_events_calls: list[dict[str, Any]] = []

    async def search_similar_chunks(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        self.search_similar_chunks_calls.append({"limit": limit})
        return list(self._semantic_results)

    async def search_fulltext_chunks(
        self,
        namespace_id: UUID,
        query: str,
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[Chunk, float]]:
        self.search_fulltext_chunks_calls.append({"limit": limit, "query": query})
        return list(self._bm25_results)

    async def search_similar_entities(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.0,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        self.search_similar_entities_calls.append({"limit": limit})
        return list(self._entity_search_results)

    async def query_events(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.query_events_calls.append({"kwargs": kwargs})
        return []

    async def get_chunks_batch(self, chunk_ids: list[UUID]) -> dict[UUID, Chunk]:
        return {}

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Any]:
        return {}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}


def _bare_engine(**kwargs: Any) -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"), **kwargs)


def _wire(engine: ChronicleEngine, coord: _RecordingCoordinator) -> None:
    engine._storage = coord  # type: ignore[assignment]
    # Stub embedder so recall() doesn't try to hit a real LLM provider.
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    engine._embedder = embedder  # type: ignore[assignment]


def _routing(complexity: QueryComplexity, *, confidence: float = 0.9) -> RoutingDecision:
    """Build a RoutingDecision shaped like the real router output."""
    return RoutingDecision(
        complexity=complexity,
        use_graph=complexity is not QueryComplexity.SIMPLE,
        graph_depth=0 if complexity is QueryComplexity.SIMPLE else 1,
        confidence=confidence,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# Router → channel gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterChannelGating:
    @pytest.mark.asyncio
    async def test_simple_skips_bm25_and_entity(self) -> None:
        """SIMPLE classification → only semantic + temporal channels invoked."""
        ns_id = uuid4()
        coord = _RecordingCoordinator(semantic_results=[(_make_chunk("a"), 0.8)])
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.SIMPLE))):
            result = await engine.recall("what is python", ns_id, limit=5, mode=SearchMode.HYBRID)

        # BM25 and entity channels skipped entirely.
        assert coord.search_fulltext_chunks_calls == []
        assert coord.search_similar_entities_calls == []
        # Semantic still runs (semantic channel + temporal-channel fallback both
        # call search_similar_chunks; we don't gate the count, only that BM25
        # and entity were skipped, and that semantic at least ran).
        assert len(coord.search_similar_chunks_calls) >= 1
        # Routing complexity surfaced in metadata for telemetry.
        assert result.metadata["routing"] == "simple"
        # Channel counters reflect skipped channels.
        assert result.metadata["channels"]["bm25"] == 0
        assert result.metadata["channels"]["entity"] == 0

    @pytest.mark.asyncio
    async def test_moderate_runs_all_four_channels(self) -> None:
        ns_id = uuid4()
        coord = _RecordingCoordinator(
            semantic_results=[(_make_chunk("a"), 0.8)],
            bm25_results=[(_make_chunk("b"), 5.2)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
            result = await engine.recall("query", ns_id, limit=5, mode=SearchMode.HYBRID)

        # Semantic always runs; temporal-channel fallback also calls
        # search_similar_chunks so the count is >= 1.
        assert len(coord.search_similar_chunks_calls) >= 1
        assert len(coord.search_fulltext_chunks_calls) == 1
        assert len(coord.search_similar_entities_calls) == 1
        assert result.metadata["routing"] == "moderate"

    @pytest.mark.asyncio
    async def test_complex_runs_all_four_channels(self) -> None:
        ns_id = uuid4()
        coord = _RecordingCoordinator(
            semantic_results=[(_make_chunk("a"), 0.8)],
            bm25_results=[(_make_chunk("b"), 5.2)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.COMPLEX))):
            result = await engine.recall("query", ns_id, limit=5, mode=SearchMode.HYBRID)

        # Semantic always runs; temporal-channel fallback also calls
        # search_similar_chunks so the count is >= 1.
        assert len(coord.search_similar_chunks_calls) >= 1
        assert len(coord.search_fulltext_chunks_calls) == 1
        assert len(coord.search_similar_entities_calls) == 1
        assert result.metadata["routing"] == "complex"

    @pytest.mark.asyncio
    async def test_router_disabled_runs_all_channels(self) -> None:
        """``router_enabled=False`` short-circuits classification."""
        ns_id = uuid4()
        coord = _RecordingCoordinator(
            semantic_results=[(_make_chunk("a"), 0.8)],
        )
        engine = _bare_engine(router_enabled=False)
        _wire(engine, coord)

        # The router would say SIMPLE, but ``router_enabled=False`` means the
        # classification branch is never even taken.
        with patch.object(
            engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.SIMPLE))
        ) as route_mock:
            result = await engine.recall("query", ns_id, limit=5, mode=SearchMode.HYBRID)

        # Router never called when disabled.
        assert route_mock.await_count == 0
        # All four channels still ran (semantic, bm25, entity); temporal runs by config default.
        # Semantic always runs; temporal-channel fallback also calls
        # search_similar_chunks so the count is >= 1.
        assert len(coord.search_similar_chunks_calls) >= 1
        assert len(coord.search_fulltext_chunks_calls) == 1
        assert len(coord.search_similar_entities_calls) == 1
        assert result.metadata["routing"] == "disabled"

    @pytest.mark.asyncio
    async def test_router_error_falls_back_to_all_channels(self) -> None:
        """A router exception must not break recall — fall back to all channels."""
        ns_id = uuid4()
        coord = _RecordingCoordinator(
            semantic_results=[(_make_chunk("a"), 0.8)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await engine.recall("query", ns_id, limit=5, mode=SearchMode.HYBRID)

        # All four channels still ran despite router exception.
        # Semantic always runs; temporal-channel fallback also calls
        # search_similar_chunks so the count is >= 1.
        assert len(coord.search_similar_chunks_calls) >= 1
        assert len(coord.search_fulltext_chunks_calls) == 1
        assert len(coord.search_similar_entities_calls) == 1
        assert result.metadata["routing"] == "fallback"

    @pytest.mark.asyncio
    async def test_temporal_signal_forces_moderate_for_temporal_query(self) -> None:
        """Explicit ``temporal_filter`` + the real heuristic router → MODERATE.

        Even a query that the router would classify SIMPLE on its own (short,
        starts with "what is") must escalate to MODERATE when a temporal filter
        is present, because the chronicle recall site builds a TemporalSignal
        from the resolver and feeds it to ``router.route(..., temporal_signal=)``.
        The router's documented behaviour is to clamp temporal queries to
        >= MODERATE so the temporal channel keeps running.
        """
        from khora.query.temporal import TemporalFilter

        ns_id = uuid4()
        coord = _RecordingCoordinator(
            semantic_results=[(_make_chunk("a"), 0.8)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        # Use the REAL router (no mock) and pass an explicit temporal_filter.
        # The query alone would otherwise classify SIMPLE.
        result = await engine.recall(
            "what is python",
            ns_id,
            limit=5,
            mode=SearchMode.HYBRID,
            temporal_filter=TemporalFilter(),  # marker: temporal intent present
        )

        # Routing escalated to MODERATE → BM25 and entity channels both ran.
        assert result.metadata["routing"] == "moderate"
        assert len(coord.search_fulltext_chunks_calls) == 1
        assert len(coord.search_similar_entities_calls) == 1


# ---------------------------------------------------------------------------
# Fusion: weighted normalized RRF (multi-channel)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWeightedNormalizedRrfMulti:
    def test_single_channel_returns_input_order(self) -> None:
        ns_id = uuid4()
        c1, c2, c3 = (
            _make_chunk("a", namespace_id=ns_id),
            _make_chunk("b", namespace_id=ns_id),
            _make_chunk("c", namespace_id=ns_id),
        )
        ranked = {"semantic": [(c1, 0.9), (c2, 0.7), (c3, 0.5)]}
        weights = {"semantic": 1.0}

        out = _weighted_normalized_rrf_multi(ranked, weights)
        # Order preserved when only one channel contributes.
        assert [c.id for c, _ in out] == [c1.id, c2.id, c3.id]

    def test_score_scale_mismatch_does_not_dominate(self) -> None:
        """BM25 raw scores ~10x larger than cosine must not flatten the ranking.

        If the fusion did NOT min-max normalize per channel, a single BM25 hit
        with score 20.0 would swamp three semantic hits with scores 0.9 / 0.7 /
        0.5. With normalization, both channels contribute on the same [0, 1]
        scale and their ranks dominate the final order.
        """
        ns_id = uuid4()
        c_sem_top, c_sem_mid, c_sem_low = (
            _make_chunk("s1", namespace_id=ns_id),
            _make_chunk("s2", namespace_id=ns_id),
            _make_chunk("s3", namespace_id=ns_id),
        )
        c_bm25_only = _make_chunk("b1", namespace_id=ns_id)

        ranked = {
            "semantic": [(c_sem_top, 0.9), (c_sem_mid, 0.7), (c_sem_low, 0.5)],
            "bm25": [(c_bm25_only, 20.0), (c_sem_mid, 5.0)],
        }
        weights = {"semantic": 1.0, "bm25": 0.8}

        out = _weighted_normalized_rrf_multi(ranked, weights)
        ids = [c.id for c, _ in out]
        # The chunk that is rank-1 in semantic is still in the top 2 — not
        # buried by the BM25 outlier.
        assert c_sem_top.id in ids[:2]
        # The shared chunk (rank-2 semantic + rank-2 bm25) outscores the
        # chunk that only appears in semantic at rank 3.
        assert ids.index(c_sem_mid.id) < ids.index(c_sem_low.id)

    def test_empty_input_returns_empty(self) -> None:
        assert _weighted_normalized_rrf_multi({}, {}) == []
        assert _weighted_normalized_rrf_multi({"semantic": []}, {"semantic": 1.0}) == []


# ---------------------------------------------------------------------------
# Backward compat: existing recall() contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_recall_returns_chunks_for_simple_query(self) -> None:
        """SIMPLE-classified queries still return chunks via the semantic channel."""
        ns_id = uuid4()
        chunks = [(_make_chunk(f"chunk-{i}"), 1.0 - i * 0.1) for i in range(3)]
        coord = _RecordingCoordinator(semantic_results=chunks)
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.SIMPLE))):
            result = await engine.recall("what is python", ns_id, limit=5, mode=SearchMode.HYBRID)

        # Top chunks come back ranked.
        assert len(result.chunks) == 3
        # Result preserves chunk identity from semantic channel.
        returned_ids = {c.id for c, _ in result.chunks}
        assert returned_ids == {c.id for c, _ in chunks}

    @pytest.mark.asyncio
    async def test_simple_and_moderate_top_results_overlap(self) -> None:
        """For a query where all channels return the same chunk, top result must
        be identical regardless of router classification — only latency differs."""
        ns_id = uuid4()
        shared = _make_chunk("shared", namespace_id=ns_id)
        coord = _RecordingCoordinator(semantic_results=[(shared, 0.95)])
        engine = _bare_engine()
        _wire(engine, coord)

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.SIMPLE))):
            simple_result = await engine.recall("q", ns_id, limit=1, mode=SearchMode.HYBRID)

        # Reset the coordinator to clear call counters.
        coord.search_similar_chunks_calls.clear()

        with patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.MODERATE))):
            moderate_result = await engine.recall("q", ns_id, limit=1, mode=SearchMode.HYBRID)

        # Same top chunk in both routing modes when the channel returning it
        # (semantic) runs in both. (BM25 + entity have no extra results to add.)
        assert simple_result.chunks[0][0].id == moderate_result.chunks[0][0].id == shared.id

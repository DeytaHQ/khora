"""Chronicle #866: pin decay-after-rerank composition.

The cross-encoder reranker is timestamp-blind by construction. Without
reapplying the recency multiplier after rerank, ``chronicle_decay_weight``
has no user-visible effect when ``enable_reranking`` is True (the default).

These tests pin two contracts:

1. ``_compute_recency_multipliers`` returns the same per-chunk multiplier
   ``_apply_temporal_decay`` uses internally, keyed by ``chunk.id``.
2. With reranking enabled AND a cross-encoder that returns the same score
   for every candidate, the final ordering is driven by the recency
   multiplier - the newest chunk ranks first.

Pattern matches Qdrant's decay re-scorer, Vespa's global-phase, and
Elasticsearch's function_score with decay-applied-after-rerank.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk
from khora.engines.chronicle.engine import (
    ChronicleEngine,
    _apply_temporal_decay,
    _compute_recency_multipliers,
)
from khora.query import SearchMode
from khora.query.router import QueryComplexity, RoutingDecision


def _chunk(*, source_timestamp: datetime, content: str = "x", namespace_id: UUID | None = None) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id or uuid4(),
        document_id=uuid4(),
        content=content,
        chunk_index=0,
        created_at=datetime.now(UTC),
        source_timestamp=source_timestamp,
    )


class _RecordingCoordinator:
    """Minimal coordinator double for chronicle recall tests."""

    def __init__(self, *, semantic_results: list[tuple[Chunk, float]] | None = None) -> None:
        self._semantic_results = semantic_results or []
        self.search_similar_chunks_calls: list[dict[str, Any]] = []

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

    async def search_fulltext_chunks(self, *args: Any, **kwargs: Any) -> list[tuple[Chunk, float]]:
        return []

    async def search_similar_entities(self, *args: Any, **kwargs: Any) -> list[tuple[UUID, float]]:
        return []

    async def query_events(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def get_chunks_batch(self, chunk_ids: list[UUID], *, namespace_id: UUID) -> dict[UUID, Chunk]:
        return {}

    async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Any]:
        return {}

    async def get_entities_by_names_batch(self, namespace_id: UUID, names: list[str]) -> dict[str, Any]:
        return {}


def _bare_engine(**kwargs: Any) -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(database_url="postgresql://localhost/test"), **kwargs)


def _wire(engine: ChronicleEngine, coord: _RecordingCoordinator) -> None:
    engine._storage = coord  # type: ignore[assignment]
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    engine._embedder = embedder  # type: ignore[assignment]


def _routing(complexity: QueryComplexity) -> RoutingDecision:
    return RoutingDecision(
        complexity=complexity,
        use_graph=complexity is not QueryComplexity.SIMPLE,
        graph_depth=0 if complexity is QueryComplexity.SIMPLE else 1,
        confidence=0.9,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# Direct tests of the multiplier helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeRecencyMultipliers:
    def test_multipliers_match_apply_temporal_decay_for_unit_relevance(self) -> None:
        """Helper returns the same per-chunk multiplier the existing decay path uses."""
        now = datetime(2026, 5, 28, tzinfo=UTC)
        fresh = _chunk(source_timestamp=now)
        aged = _chunk(source_timestamp=now - timedelta(hours=168))  # one half-life
        ancient = _chunk(source_timestamp=now - timedelta(days=3650))  # ~zero retention

        scored = _apply_temporal_decay(
            [(fresh, 1.0), (aged, 1.0), (ancient, 1.0)],
            decay_weight=0.3,
            half_life_hours=168.0,
            reference_time=now,
        )
        mults = _compute_recency_multipliers(
            [fresh, aged, ancient],
            decay_weight=0.3,
            half_life_hours=168.0,
            reference_time=now,
        )

        for chunk, score in scored:
            assert mults[chunk.id] == pytest.approx(score, abs=1e-6)

    def test_decay_weight_zero_returns_empty_dict(self) -> None:
        """decay_weight=0 short-circuits; callers should treat missing keys as 1.0."""
        now = datetime(2026, 5, 28, tzinfo=UTC)
        chunks = [_chunk(source_timestamp=now - timedelta(days=30))]
        assert _compute_recency_multipliers(chunks, decay_weight=0.0, reference_time=now) == {}

    def test_empty_chunks_returns_empty_dict(self) -> None:
        assert _compute_recency_multipliers([], decay_weight=0.3) == {}

    def test_aggressive_decay_weight_inverts_ordering(self) -> None:
        """With weight=0.9, the ancient chunk's multiplier is much smaller than fresh's."""
        now = datetime(2026, 5, 28, tzinfo=UTC)
        fresh = _chunk(source_timestamp=now)
        ancient = _chunk(source_timestamp=now - timedelta(days=3650))
        mults = _compute_recency_multipliers(
            [fresh, ancient],
            decay_weight=0.9,
            half_life_hours=168.0,
            reference_time=now,
        )
        # fresh stays near 1.0; ancient drops to (1 - 0.9) = 0.1.
        assert mults[fresh.id] == pytest.approx(1.0, abs=1e-3)
        assert mults[ancient.id] == pytest.approx(0.1, abs=1e-3)


# ---------------------------------------------------------------------------
# Integration: full recall path with rerank patched to constant score
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecayAppliedAfterRerank:
    @pytest.mark.asyncio
    async def test_constant_rerank_lets_decay_drive_ordering(self) -> None:
        """When rerank returns the same score for every chunk, decay decides the order.

        Pre-fix: the rerank replaces decay-blended scores with raw cross-encoder
        scores; equal cross-encoder scores produce arbitrary ordering, and
        ``chronicle_decay_weight`` has no effect on the final ranking. Post-fix:
        the decay multiplier is reapplied on top of the rerank output, so the
        newest chunk surfaces first.
        """
        ns_id = uuid4()
        now = datetime.now(UTC)
        fresh = _chunk(source_timestamp=now - timedelta(days=7), content="Matcha", namespace_id=ns_id)
        medium = _chunk(source_timestamp=now - timedelta(days=90), content="Coffee", namespace_id=ns_id)
        ancient = _chunk(source_timestamp=now - timedelta(days=180), content="Tea", namespace_id=ns_id)

        coord = _RecordingCoordinator(
            semantic_results=[(ancient, 0.9), (medium, 0.9), (fresh, 0.9)],
        )
        engine = _bare_engine()
        _wire(engine, coord)

        # Aggressive decay so the multiplier gap is obvious.
        engine._config.query.chronicle_decay_weight = 0.7
        engine._config.query.temporal_half_life_hours = 720.0
        engine._config.query.enable_reranking = True

        async def _constant_rerank(
            query: str, candidates: list[tuple[Chunk, float]], **kwargs: Any
        ) -> list[tuple[Chunk, float]]:
            # Cross-encoder is timestamp-blind: every candidate gets the same score.
            return [(c, 1.0) for c, _ in candidates]

        with (
            patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.SIMPLE))),
            patch("khora.query.reranking.rerank_chunks", new=_constant_rerank),
        ):
            result = await engine.recall("preferred drink?", ns_id, limit=3, mode=SearchMode.HYBRID)

        assert len(result.chunks) >= 1
        # Newest chunk surfaces first because the decay multiplier
        # multiplied onto the (constant) rerank score breaks the tie.
        assert result.chunks[0].id == fresh.id

    @pytest.mark.asyncio
    async def test_constant_rerank_with_decay_off_falls_back_to_rerank_order(self) -> None:
        """decay_weight=0 must not surprise the rerank-on path: order matches rerank output."""
        ns_id = uuid4()
        now = datetime.now(UTC)
        fresh = _chunk(source_timestamp=now - timedelta(days=7), content="Matcha", namespace_id=ns_id)
        ancient = _chunk(source_timestamp=now - timedelta(days=180), content="Tea", namespace_id=ns_id)

        coord = _RecordingCoordinator(semantic_results=[(ancient, 0.9), (fresh, 0.9)])
        engine = _bare_engine()
        _wire(engine, coord)

        engine._config.query.chronicle_decay_weight = 0.0
        engine._config.query.enable_reranking = True

        async def _ancient_first_rerank(
            query: str, candidates: list[tuple[Chunk, float]], **kwargs: Any
        ) -> list[tuple[Chunk, float]]:
            # Cross-encoder picks ``ancient`` over ``fresh``; with decay off, decay
            # must not override the rerank verdict.
            ranked: list[tuple[Chunk, float]] = []
            for c, _ in candidates:
                ranked.append((c, 0.9 if c.id == ancient.id else 0.5))
            ranked.sort(key=lambda pair: pair[1], reverse=True)
            return ranked

        with (
            patch.object(engine._router, "route", new=AsyncMock(return_value=_routing(QueryComplexity.SIMPLE))),
            patch("khora.query.reranking.rerank_chunks", new=_ancient_first_rerank),
        ):
            result = await engine.recall("q", ns_id, limit=2, mode=SearchMode.HYBRID)

        assert len(result.chunks) >= 1
        assert result.chunks[0].id == ancient.id

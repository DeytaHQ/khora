"""Tests for issue #567 Phase A retriever changes.

Covers three behaviours implemented in
``khora.engines.vectorcypher.retriever``:

* A1 — wall-clock vs relative reference in ``_calculate_recency_scores``.
* A2 — synthetic RECENCY/CHANGE date floor at the retriever call site,
  including the anti-recency veto and the feature flag.
* A4 — per-source decay in ``_calculate_recency_scores``.

All flags default OFF, so legacy callers see no behavioural change.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.engines.vectorcypher.fusion import FusedResult
from khora.engines.vectorcypher.retriever import (
    RetrieverConfig,
    VectorCypherRetriever,
)
from khora.engines.vectorcypher.router import QueryComplexity, RoutingDecision
from khora.query.temporal_detection import TemporalCategory, TemporalSignal
from khora.storage.temporal import TemporalFilter


def _chunk(occurred_at: datetime | None, *, source_system: str | None = None) -> Chunk:
    """Build a chunk with optional occurred_at and source_system metadata."""
    custom: dict[str, object] = {}
    if occurred_at is not None:
        custom["occurred_at"] = occurred_at.isoformat()
    if source_system is not None:
        custom["source_system"] = source_system
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="test",
        metadata=custom,
    )


def _fused(chunk: Chunk) -> FusedResult:
    return FusedResult(item_id=chunk.id, item=chunk, rrf_score=0.5)


# ─────────────────────────── A1: wall-clock reference ───────────────────────────


@pytest.mark.unit
class TestRecencyReferenceMode:
    """A1 — _calculate_recency_scores reference_mode resolution."""

    def _retriever(self, *, wall_clock: bool = False) -> VectorCypherRetriever:
        cfg = RetrieverConfig(temporal_reference_wall_clock=wall_clock)
        return VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=cfg,
        )

    def test_wall_clock_punishes_uniformly_stale_results(self) -> None:
        """With wall-clock reference, all-old chunks score near zero.

        This is the production-killer bug being fixed: under the relative
        mode the newest-stale chunk receives recency=1.0 even when it is
        years old. Wall-clock mode must penalize it.
        """
        retriever = self._retriever(wall_clock=True)
        very_old = datetime(2018, 1, 1, tzinfo=UTC)
        slightly_less_old = datetime(2018, 6, 1, tzinfo=UTC)
        results = [_fused(_chunk(very_old)), _fused(_chunk(slightly_less_old))]

        scores = retriever._calculate_recency_scores(results, decay_days_override=7)

        # Both chunks predate the wall-clock by many years → recency ≈ 0.
        assert all(s < 0.01 for s in scores.values()), f"Got {scores}"

    def test_relative_mode_normalizes_to_newest(self) -> None:
        """Relative mode preserves the legacy benchmark-friendly behavior."""
        retriever = self._retriever(wall_clock=False)
        very_old = datetime(2018, 1, 1, tzinfo=UTC)
        slightly_less_old = datetime(2018, 6, 1, tzinfo=UTC)
        c1 = _chunk(very_old)
        c2 = _chunk(slightly_less_old)
        results = [_fused(c1), _fused(c2)]

        scores = retriever._calculate_recency_scores(results, decay_days_override=7)

        # Newest item in set is the reference → recency ≈ 1.0.
        assert scores[c2.id] == pytest.approx(1.0, abs=1e-6)
        # And the older item should be measurably lower.
        assert scores[c1.id] < scores[c2.id]

    def test_explicit_arg_overrides_config_flag(self) -> None:
        """Explicit reference_mode kwarg beats the config flag."""
        retriever = self._retriever(wall_clock=True)  # config says wall_clock
        old = datetime(2018, 1, 1, tzinfo=UTC)
        results = [_fused(_chunk(old))]

        scores_relative = retriever._calculate_recency_scores(results, decay_days_override=7, reference_mode="relative")
        # Single item — relative mode → score == 1.0.
        assert scores_relative[results[0].item_id] == pytest.approx(1.0, abs=1e-6)

    def test_default_resolves_to_relative_when_flag_off(self) -> None:
        """Without the config flag and without env, default = relative."""
        retriever = self._retriever(wall_clock=False)
        old = datetime(2018, 1, 1, tzinfo=UTC)
        results = [_fused(_chunk(old))]

        scores = retriever._calculate_recency_scores(results, decay_days_override=7)
        assert scores[results[0].item_id] == pytest.approx(1.0, abs=1e-6)

    def test_future_dated_chunk_capped_at_one_wall_clock_exponential(self) -> None:
        """GitHub issue #1230: a future-dated chunk in wall-clock mode must not
        get a recency factor > 1.0. days_old goes negative for a future ts, so
        ``exp(-lambda * days_old) > 1.0`` without a clamp — inflating the boost."""
        retriever = self._retriever(wall_clock=True)
        future = datetime.now(UTC) + timedelta(days=30)
        results = [_fused(_chunk(future))]

        scores = retriever._calculate_recency_scores(results, decay_days_override=7)

        assert scores[results[0].item_id] <= 1.0, f"Got {scores}"

    def test_future_dated_chunk_capped_at_one_wall_clock_linear(self) -> None:
        """Same #1230 invariant on the linear decay path. ``1 - days_old/decay``
        also exceeds 1.0 when days_old is negative."""
        cfg = RetrieverConfig(temporal_reference_wall_clock=True, recency_decay_type="linear")
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=cfg,
        )
        future = datetime.now(UTC) + timedelta(days=30)
        results = [_fused(_chunk(future))]

        scores = retriever._calculate_recency_scores(results, decay_days_override=7)

        assert scores[results[0].item_id] <= 1.0, f"Got {scores}"


# ─────────────────────────── A4: per-source decay ───────────────────────────


@pytest.mark.unit
class TestPerSourceDecay:
    """A4 — per-source decay lookup in _calculate_recency_scores."""

    def test_per_source_decay_differentiates_sources(self) -> None:
        """A Slack chunk (3d decay) and a Salesforce chunk (180d decay) of
        the same age must produce different recency scores."""
        cfg = RetrieverConfig(
            temporal_per_source_decay=True,
            # Force wall_clock so the per-source decay actually matters:
            # under "relative" the newest chunk anchors to 1.0 regardless.
            temporal_reference_wall_clock=True,
            temporal_default_decay_by_source={
                "slack": 3,
                "salesforce": 180,
                "_default": 14,
            },
        )
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=cfg,
        )

        days_old = 30
        ts = datetime.now(UTC) - timedelta(days=days_old)
        slack_chunk = _chunk(ts, source_system="slack")
        sf_chunk = _chunk(ts, source_system="salesforce")

        scores = retriever._calculate_recency_scores([_fused(slack_chunk), _fused(sf_chunk)])

        # 30 days with Slack's 3-day half-life → ~2^-10 ≈ 1e-3.
        # 30 days with Salesforce's 180-day half-life → ~2^(-30/180) ≈ 0.89.
        assert scores[sf_chunk.id] > scores[slack_chunk.id]
        assert scores[slack_chunk.id] < 0.01
        assert scores[sf_chunk.id] > 0.5

    def test_unknown_source_falls_back_to_default(self) -> None:
        """Empty / unknown source_system uses the _default decay window."""
        cfg = RetrieverConfig(
            temporal_per_source_decay=True,
            temporal_reference_wall_clock=True,
            temporal_default_decay_by_source={"slack": 3, "_default": 14},
        )
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=cfg,
        )

        ts = datetime.now(UTC) - timedelta(days=14)  # exactly the _default half-life
        # source_system absent
        no_src = _chunk(ts, source_system=None)
        # source_system empty string (treated as None by _extract_source_system)
        empty_src = _chunk(ts, source_system="")
        # unknown source_system
        unknown = _chunk(ts, source_system="quickbooks")

        results = [_fused(no_src), _fused(empty_src), _fused(unknown)]
        scores = retriever._calculate_recency_scores(results)

        expected = math.exp(-math.log(2) * 14 / 14)  # ≈ 0.5
        for r in results:
            assert scores[r.item_id] == pytest.approx(expected, rel=0.05), f"item={r.item_id} got={scores[r.item_id]}"

    def test_per_source_disabled_uses_override(self) -> None:
        """With per-source decay OFF, decay_days_override is honored."""
        cfg = RetrieverConfig(temporal_per_source_decay=False, temporal_reference_wall_clock=True)
        retriever = VectorCypherRetriever(
            vector_store=AsyncMock(),
            neo4j_driver=AsyncMock(),
            embedder=AsyncMock(),
            config=cfg,
        )

        ts = datetime.now(UTC) - timedelta(days=7)
        chunk = _chunk(ts, source_system="slack")
        scores = retriever._calculate_recency_scores([_fused(chunk)], decay_days_override=7)

        # 7 days with 7-day half-life → 0.5.
        assert scores[chunk.id] == pytest.approx(0.5, rel=0.05)


# ───────────────── A2: synthesized RECENCY temporal filter ─────────────────


@pytest.fixture
def synth_retriever() -> VectorCypherRetriever:
    """Retriever wired to a recording vector_store so we can assert on the
    final ``temporal_filter`` actually passed downstream.

    Routes everything to SIMPLE so the unit test stays self-contained.
    """
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=[])  # empty result set is fine
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 8)
    embedder.model_name = "stub"
    embedder.dimension = 8

    cfg = RetrieverConfig(
        temporal_recency_floor_enabled=True,
    )
    retriever = VectorCypherRetriever(
        vector_store=vector_store,
        neo4j_driver=None,  # forces SIMPLE / vector-only paths
        embedder=embedder,
        config=cfg,
    )
    # Force the router to SIMPLE so we hit a deterministic path.
    retriever._router = MagicMock()
    retriever._router.route = AsyncMock(
        return_value=RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="simple",
        )
    )
    return retriever


@pytest.mark.unit
class TestSyntheticRecencyFilter:
    """A2 — RECENCY/CHANGE floor synthesis at the retriever call site."""

    @pytest.mark.asyncio
    async def test_recency_signal_synthesizes_floor(self, synth_retriever: VectorCypherRetriever) -> None:
        """Happy path: bare 'latest' RECENCY signal → 30-day floor applied."""
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
            temporal_filter=None,
        )

        before = datetime.now(UTC)
        await synth_retriever.retrieve(
            query="what's the latest status?",
            namespace_id=uuid4(),
            temporal_filter=None,
            temporal_signal=signal,
        )

        # vector_store.search must have been called with a synthesized filter.
        call_kwargs = synth_retriever._vector_store.search.call_args.kwargs
        tf = call_kwargs["temporal_filter"]
        assert tf is not None, "Expected a synthesized TemporalFilter"
        assert isinstance(tf, TemporalFilter)
        # 30-day window for RECENCY, anchored on wall clock. Was 14d in the
        # initial Phase A commit; widened to 30d after LoCoMo --small showed
        # the 14d cutoff regressing counterfactual_accuracy by 16.7pp.
        expected = before - timedelta(days=30)
        delta = abs((tf.occurred_after - expected).total_seconds())
        # Allow a small clock-drift window between the test's `before` and
        # the retriever's datetime.now(UTC) call.
        assert delta < 5.0, f"Synthesized floor too far from expected: delta={delta}s"
        # The remaining fields must stay clean — we only synthesize the
        # date floor, never a source/author/channel constraint.
        assert tf.occurred_before is None
        assert tf.source_system is None

    @pytest.mark.asyncio
    async def test_anti_recency_token_vetoes_synthesis(self, synth_retriever: VectorCypherRetriever) -> None:
        """'ever' / 'all history' must veto the floor — no filter synthesized."""
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
            temporal_filter=None,
        )

        await synth_retriever.retrieve(
            query="what action items have we ever discussed for Phoenix",
            namespace_id=uuid4(),
            temporal_filter=None,
            temporal_signal=signal,
        )

        call_kwargs = synth_retriever._vector_store.search.call_args.kwargs
        assert call_kwargs["temporal_filter"] is None, "Anti-recency token 'ever' must suppress synthesis"

    @pytest.mark.asyncio
    async def test_feature_flag_disabled_skips_synthesis(self) -> None:
        """temporal_recency_floor_enabled=False (the default) → no synthesis."""
        vector_store = AsyncMock()
        vector_store.search = AsyncMock(return_value=[])
        embedder = AsyncMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 8)
        embedder.model_name = "stub"
        embedder.dimension = 8

        cfg = RetrieverConfig(
            temporal_recency_floor_enabled=False,
        )
        retriever = VectorCypherRetriever(
            vector_store=vector_store,
            neo4j_driver=None,
            embedder=embedder,
            config=cfg,
        )
        retriever._router = MagicMock()
        retriever._router.route = AsyncMock(
            return_value=RoutingDecision(
                complexity=QueryComplexity.SIMPLE,
                use_graph=False,
                graph_depth=0,
                confidence=0.9,
                reasoning="simple",
            )
        )

        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
            temporal_filter=None,
        )
        await retriever.retrieve(
            query="latest status",
            namespace_id=uuid4(),
            temporal_filter=None,
            temporal_signal=signal,
        )

        call_kwargs = vector_store.search.call_args.kwargs
        assert call_kwargs["temporal_filter"] is None

    @pytest.mark.asyncio
    async def test_explicit_filter_passes_through_untouched(self, synth_retriever: VectorCypherRetriever) -> None:
        """When the caller already supplies a TemporalFilter, do not overwrite it."""
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
            temporal_filter=None,
        )
        caller_filter = TemporalFilter(occurred_after=datetime(2024, 1, 1, tzinfo=UTC))

        await synth_retriever.retrieve(
            query="latest status",
            namespace_id=uuid4(),
            temporal_filter=caller_filter,
            temporal_signal=signal,
        )

        call_kwargs = synth_retriever._vector_store.search.call_args.kwargs
        assert call_kwargs["temporal_filter"] is caller_filter, (
            "Existing TemporalFilter must not be overwritten by synthesis"
        )

    @pytest.mark.asyncio
    async def test_non_temporal_signal_skips_synthesis(self, synth_retriever: VectorCypherRetriever) -> None:
        """A non-temporal (NONE) signal must never produce a floor."""
        signal = TemporalSignal(
            is_temporal=False,
            category=TemporalCategory.NONE,
            confidence=1.0,
            source="none",
            temporal_filter=None,
        )
        await synth_retriever.retrieve(
            query="what is the capital of France",
            namespace_id=uuid4(),
            temporal_filter=None,
            temporal_signal=signal,
        )
        call_kwargs = synth_retriever._vector_store.search.call_args.kwargs
        assert call_kwargs["temporal_filter"] is None

    @pytest.mark.asyncio
    async def test_change_category_uses_60_day_window(self, synth_retriever: VectorCypherRetriever) -> None:
        """CHANGE has default_window_days=60 → synthesized floor matches.

        Was 30d in the initial Phase A commit; widened to 60d alongside
        the RECENCY 14→30 tuning to reduce regression on CHANGE-class
        historical queries.
        """
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.CHANGE,
            confidence=0.9,
            source="dictionary",
            temporal_filter=None,
        )
        before = datetime.now(UTC)
        await synth_retriever.retrieve(
            query="what changed in the budget",
            namespace_id=uuid4(),
            temporal_filter=None,
            temporal_signal=signal,
        )
        call_kwargs = synth_retriever._vector_store.search.call_args.kwargs
        tf = call_kwargs["temporal_filter"]
        assert tf is not None
        delta = abs((tf.occurred_after - (before - timedelta(days=60))).total_seconds())
        assert delta < 5.0

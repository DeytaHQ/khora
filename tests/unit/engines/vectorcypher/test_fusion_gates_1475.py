"""Gate + confidence calibration (#1475, leg 2).

Two latent mis-calibrations, fixed without touching default output:

1. The LLM-rerank decisive-winner gate compared ``FusedResult.rrf_score`` against
   the ``0.7`` absolute-topicality threshold (and ``0.1`` gaps). On the raw
   weighted-RRF scale, ``rrf_score`` is ~1/(k+rank) ≈ 0.016, so those thresholds
   could NEVER fire — the whole decisive-winner skip was silently dead there.
   The fix runs the skip on the true raw cosine (an absolute [0,1] scale) that
   the caller now threads. Proven dead below.

2. ``engine_info['confidence']`` saturated the cosine term at
   ``target_cosine=0.5`` (a 0.5 and a 0.95 top hit read identically) and computed
   the gap off post-fusion display scores. ``confidence_calibration="raw_cosine"``
   desaturates the cosine term and uses the raw-cosine gap.

Both are behind opt-in switches (the gate lives behind ``enable_llm_reranking``,
default OFF; confidence behind ``confidence_calibration``, default "legacy"), so
default recall output is unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.core.recall_abstention import compute_confidence
from khora.engines.vectorcypher.fusion import FusedResult
from khora.engines.vectorcypher.retriever import RetrieverConfig, VectorCypherRetriever
from khora.engines.vectorcypher.temporal_detection import TemporalCategory, TemporalSignal


def _retriever(**cfg: Any) -> VectorCypherRetriever:
    base = dict(enable_llm_reranking=True, llm_reranking_mode="always")
    base.update(cfg)
    return VectorCypherRetriever(
        vector_store=AsyncMock(),
        neo4j_driver=None,
        embedder=AsyncMock(),
        config=RetrieverConfig(**base),
        storage=None,
    )


def _temporal() -> TemporalSignal:
    return TemporalSignal(is_temporal=True, category=TemporalCategory.RECENCY, confidence=0.9, source="dictionary")


@pytest.mark.unit
class TestDecisiveWinnerScaleDeadOnRRF:
    """The 0.7/0.1 thresholds are unreachable on the raw weighted-RRF scale."""

    def test_should_skip_is_dead_on_rrf_scale(self) -> None:
        retriever = _retriever(
            llm_reranking_confidence_threshold=0.1,
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        # Raw weighted-RRF magnitudes at k=60: top ~1/(60+1), gap between #1/#2
        # ~1/(60+1)-1/(60+2). NEITHER clause can fire.
        assert retriever._should_skip_llm_rerank(top_score=0.0164, gap=0.0003) is False

    def test_should_skip_fires_on_cosine_scale(self) -> None:
        retriever = _retriever(
            llm_reranking_confidence_threshold=0.99,  # isolate the decisive clause
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        # A genuinely topical, well-separated winner on the raw-cosine scale.
        assert retriever._should_skip_llm_rerank(top_score=0.95, gap=0.35) is True

    def test_gate_dead_without_raw_cosine_on_rrf_candidates(self) -> None:
        """End-to-end: RRF-scale candidates + no threaded cosine → never skips."""
        retriever = _retriever(
            llm_reranking_confidence_threshold=0.1,
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        candidates = [
            FusedResult(item=type("X", (), {"metadata": {}})(), rrf_score=0.0164, item_id=uuid4()),
            FusedResult(item=type("X", (), {"metadata": {}})(), rrf_score=0.0161, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(candidates, _temporal(), namespace_id=uuid4())
        # Decisive-winner skip is DEAD on the RRF scale: rerank runs.
        assert should_run is True
        assert reason is None

    def test_gate_fires_when_raw_cosine_threaded(self) -> None:
        """Same RRF-scale candidates, but the fix threads the true raw cosines."""
        retriever = _retriever(
            llm_reranking_confidence_threshold=0.99,  # isolate the decisive clause
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        candidates = [
            FusedResult(item=type("X", (), {"metadata": {}})(), rrf_score=0.0164, item_id=uuid4()),
            FusedResult(item=type("X", (), {"metadata": {}})(), rrf_score=0.0161, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(
            candidates,
            _temporal(),
            namespace_id=uuid4(),
            top_raw_cosine=0.95,
            second_raw_cosine=0.55,
        )
        assert should_run is False
        assert reason == "decisive_winner"

    def test_threaded_but_weak_topicality_still_runs(self) -> None:
        """A weak top cosine (<0.7) must NOT trip the decisive skip."""
        retriever = _retriever(
            llm_reranking_confidence_threshold=0.99,
            llm_reranking_min_top_score=0.7,
            llm_reranking_decisive_gap=0.1,
        )
        candidates = [
            FusedResult(item=type("X", (), {"metadata": {}})(), rrf_score=0.0164, item_id=uuid4()),
            FusedResult(item=type("X", (), {"metadata": {}})(), rrf_score=0.0161, item_id=uuid4()),
        ]
        should_run, reason = retriever._evaluate_llm_rerank_gate(
            candidates,
            _temporal(),
            namespace_id=uuid4(),
            top_raw_cosine=0.45,  # below min_top_score
            second_raw_cosine=0.05,
        )
        assert should_run is True
        assert reason is None


@pytest.mark.unit
class TestConfidenceSaturation:
    """The legacy cosine term ceilings at target_cosine; raw_cosine desaturates."""

    def test_legacy_saturates_at_target_cosine(self) -> None:
        # target_cosine=0.5: both 0.5 and 0.95 map the cosine term to 1.0, so
        # the two confidences are identical — the saturation the ticket flags.
        low = compute_confidence(top_cosine=0.5, top_score_gap=0.0, target_cosine=0.5)
        high = compute_confidence(top_cosine=0.95, top_score_gap=0.0, target_cosine=0.5)
        assert low == high == pytest.approx(0.8)

    def test_raw_cosine_mode_desaturates(self) -> None:
        # raw_cosine mode uses the full [0,1] cosine magnitude, so 0.95 > 0.5.
        low = compute_confidence(top_cosine=0.5, top_score_gap=0.0, mode="raw_cosine")
        high = compute_confidence(top_cosine=0.95, top_score_gap=0.0, mode="raw_cosine")
        assert low == pytest.approx(0.8 * 0.5)
        assert high == pytest.approx(0.8 * 0.95)
        assert high > low

    def test_raw_cosine_still_in_unit_interval(self) -> None:
        for cosine, gap in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.5, 0.1)]:
            c = compute_confidence(top_cosine=cosine, top_score_gap=gap, mode="raw_cosine")
            assert 0.0 <= c <= 1.0

    def test_legacy_default_mode_unchanged(self) -> None:
        # Default mode is "legacy": explicit == implicit.
        assert compute_confidence(top_cosine=0.6, top_score_gap=0.15) == compute_confidence(
            top_cosine=0.6, top_score_gap=0.15, mode="legacy"
        )

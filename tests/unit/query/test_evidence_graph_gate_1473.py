"""Unit tests for the #1473 evidence-based graph channel gate.

The gate is a router-layer channel-selection decision
(``QueryComplexityRouter.evaluate_graph_gate``): when the vector channel has a
decisive score-gap winner, it recommends suppressing the graph channel (which
injects noise on single-fact questions). Default OFF; COMPLEX (multi-hop)
routing is always protected.
"""

from __future__ import annotations

import pytest

from khora.query.router import (
    GraphGateDecision,
    QueryComplexity,
    QueryComplexityRouter,
    RouterConfig,
)

pytestmark = pytest.mark.unit


def _router(**gate_kwargs: object) -> QueryComplexityRouter:
    cfg = RouterConfig(
        evidence_graph_gate_enabled=True,
        evidence_graph_gate_min_top_score=0.5,
        evidence_graph_gate_min_gap=0.25,
    )
    for k, v in gate_kwargs.items():
        setattr(cfg, k, v)
    return QueryComplexityRouter(cfg)


class TestEvaluateGraphGate:
    def test_disabled_never_suppresses(self) -> None:
        router = QueryComplexityRouter(RouterConfig(evidence_graph_gate_enabled=False))
        decision = router.evaluate_graph_gate([0.95, 0.1], complexity=QueryComplexity.MODERATE)
        assert isinstance(decision, GraphGateDecision)
        assert decision.suppress is False
        assert "disabled" in decision.reasoning

    def test_complex_is_protected(self) -> None:
        router = _router()
        # A decisive gap that WOULD suppress on MODERATE is ignored for COMPLEX.
        decision = router.evaluate_graph_gate([0.95, 0.1], complexity=QueryComplexity.COMPLEX)
        assert decision.suppress is False
        assert "complex" in decision.reasoning.lower()

    def test_decisive_winner_suppresses(self) -> None:
        router = _router()
        decision = router.evaluate_graph_gate([0.9, 0.5], complexity=QueryComplexity.MODERATE)
        assert decision.suppress is True
        assert decision.top_score == pytest.approx(0.9)
        assert decision.gap == pytest.approx(0.4)

    def test_entity_anchored_can_be_suppressed(self) -> None:
        router = _router()
        decision = router.evaluate_graph_gate([0.8, 0.3], complexity=QueryComplexity.ENTITY_ANCHORED)
        assert decision.suppress is True

    def test_small_gap_does_not_suppress(self) -> None:
        router = _router()
        # High top score but the runner-up is nearly as strong -> not decisive.
        decision = router.evaluate_graph_gate([0.9, 0.8], complexity=QueryComplexity.MODERATE)
        assert decision.suppress is False
        assert decision.gap == pytest.approx(0.1)

    def test_low_top_score_does_not_suppress(self) -> None:
        router = _router()
        # Big gap but the top score is below the floor -> weak evidence overall.
        decision = router.evaluate_graph_gate([0.4, 0.05], complexity=QueryComplexity.MODERATE)
        assert decision.suppress is False

    def test_single_score_is_insufficient(self) -> None:
        router = _router()
        decision = router.evaluate_graph_gate([0.99], complexity=QueryComplexity.MODERATE)
        assert decision.suppress is False
        assert "insufficient" in decision.reasoning
        assert decision.top_score == pytest.approx(0.99)

    def test_empty_scores_is_insufficient(self) -> None:
        router = _router()
        decision = router.evaluate_graph_gate([], complexity=QueryComplexity.MODERATE)
        assert decision.suppress is False
        assert decision.top_score == 0.0

    def test_scores_ranked_internally(self) -> None:
        router = _router()
        # Unsorted input still resolves the true top/second.
        decision = router.evaluate_graph_gate([0.2, 0.9, 0.4], complexity=QueryComplexity.MODERATE)
        assert decision.top_score == pytest.approx(0.9)
        assert decision.gap == pytest.approx(0.5)
        assert decision.suppress is True

    def test_thresholds_are_configurable(self) -> None:
        # Tighten the gap threshold so the same evidence no longer fires.
        router = _router(evidence_graph_gate_min_gap=0.6)
        decision = router.evaluate_graph_gate([0.9, 0.5], complexity=QueryComplexity.MODERATE)
        assert decision.suppress is False


class TestGraphGateConfigDefaults:
    def test_router_config_default_off(self) -> None:
        cfg = RouterConfig()
        assert cfg.evidence_graph_gate_enabled is False
        assert cfg.evidence_graph_gate_min_top_score == 0.5
        assert cfg.evidence_graph_gate_min_gap == 0.25

    def test_query_settings_default_off(self) -> None:
        from khora.config.schema import QuerySettings

        qs = QuerySettings()
        assert qs.enable_evidence_graph_gate is False
        assert qs.evidence_graph_gate_min_top_score == 0.5
        assert qs.evidence_graph_gate_min_gap == 0.25

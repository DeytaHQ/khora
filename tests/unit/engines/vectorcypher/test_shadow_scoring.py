"""Unit tests for the shadow-scoring divergence math (#1479).

Pure-Python, no DB, no engine - exercises ``build_candidate_order`` /
``compute_shadow_report`` directly with fake chunks so the rank-correlation,
top-k overlap and mover-list logic is locked deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from khora.engines.vectorcypher.shadow_scoring import (
    build_candidate_order,
    compute_shadow_report,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeChunk:
    id: UUID
    document_id: UUID


def _cands(scores: list[float]) -> list[tuple[_FakeChunk, float]]:
    """A scored-candidate list in the given (incumbent) order."""
    return [(_FakeChunk(id=uuid4(), document_id=uuid4()), s) for s in scores]


def test_identity_strategy_returns_incumbent_unchanged() -> None:
    cands = _cands([0.9, 0.5, 0.1])
    out = build_candidate_order(cands, strategy="identity")
    assert out == cands


def test_score_sort_reorders_by_descending_score() -> None:
    # Incumbent order is ASCENDING score (the #1433 shape: fusion order disagrees
    # with raw score). score_sort must flip it to descending.
    cands = _cands([0.1, 0.5, 0.9])
    out = build_candidate_order(cands, strategy="score_sort")
    assert [s for _, s in out] == [0.9, 0.5, 0.1]


def test_score_sort_is_stable_on_ties() -> None:
    cands = _cands([0.5, 0.5, 0.5])
    out = build_candidate_order(cands, strategy="score_sort")
    # Ties preserve incumbent relative order (stable sort).
    assert [c.id for c, _ in out] == [c.id for c, _ in cands]


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown shadow-scoring candidate strategy"):
        build_candidate_order(_cands([0.1]), strategy="nope")


def test_report_identity_is_perfectly_correlated() -> None:
    cands = _cands([0.9, 0.5, 0.1])
    report = compute_shadow_report(
        cands, build_candidate_order(cands, strategy="identity"), strategy="identity", limit=10
    )
    assert report["identical"] is True
    assert report["spearman_rho"] == 1.0
    assert report["topk_overlap"] == 1.0
    assert report["moved"] == []
    assert report["candidate_count"] == 3


def test_report_full_reversal_is_negative_correlation() -> None:
    # Incumbent ascending, candidate score_sort -> fully reversed order.
    cands = _cands([0.1, 0.5, 0.9])
    report = compute_shadow_report(
        cands, build_candidate_order(cands, strategy="score_sort"), strategy="score_sort", limit=10
    )
    assert report["identical"] is False
    assert report["spearman_rho"] == -1.0  # exact reversal of 3 items
    # Top-k CHUNK ids are the same set (all 3), only reordered -> overlap 1.0.
    assert report["topk_overlap"] == 1.0
    # The two endpoints moved; the middle stayed. 2 movers.
    assert len(report["moved"]) == 2
    for mover in report["moved"]:
        assert mover["delta"] == mover["candidate_rank"] - mover["incumbent_rank"]
        assert mover["delta"] != 0


def test_report_spearman_none_for_single_candidate() -> None:
    cands = _cands([0.5])
    report = compute_shadow_report(
        cands, build_candidate_order(cands, strategy="score_sort"), strategy="score_sort", limit=10
    )
    assert report["spearman_rho"] is None
    assert report["candidate_count"] == 1


def test_report_topk_overlap_narrows_when_a_new_chunk_enters_topk() -> None:
    # 5 candidates; incumbent order is ascending score so score_sort reverses it.
    # With k=2 the incumbent top-2 (the two LOWEST-scored) and the candidate
    # top-2 (the two HIGHEST-scored) are disjoint -> Jaccard 0.0.
    cands = _cands([0.1, 0.2, 0.3, 0.4, 0.5])
    report = compute_shadow_report(
        cands, build_candidate_order(cands, strategy="score_sort"), strategy="score_sort", limit=2
    )
    assert report["topk"] == 2
    assert report["topk_overlap"] == 0.0


def test_report_is_json_serializable() -> None:
    import json

    cands = _cands([0.1, 0.9, 0.4])
    report = compute_shadow_report(
        cands, build_candidate_order(cands, strategy="score_sort"), strategy="score_sort", limit=10
    )
    json.dumps(report)  # must not raise

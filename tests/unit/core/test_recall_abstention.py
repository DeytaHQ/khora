"""Unit tests for ``khora.core.recall_abstention.compute_abstention_signals``.

Scenarios from DYT-4601 pin the formula, flag set, and ``should_abstain``
boundary; the chronicle and vectorcypher engines both delegate to this
helper, so the public contract lives here.
"""

from __future__ import annotations

import math

import pytest

from khora.core.recall_abstention import compute_abstention_signals


@pytest.mark.unit
class TestComputeAbstentionSignals:
    def test_all_signals_fire_when_nothing_retrieved(self) -> None:
        """chunk_count=0, entity_count=0, top_chunk_score=0.0 → every flag True,
        combined_score == 1.0, should_abstain True."""
        result = compute_abstention_signals(
            chunk_count=0,
            top_chunk_score=0.0,
            entity_count=0,
        )

        assert result["entities_empty"] is True
        assert result["chunks_empty"] is True
        assert result["chunks_below_min"] is True
        assert result["top_score_low"] is True
        assert math.isclose(result["combined_score"], 1.0)
        assert result["should_abstain"] is True

    def test_no_signals_fire_when_retrieval_is_healthy(self) -> None:
        """chunk_count=5, entity_count=3, top_chunk_score=0.9 (defaults) →
        every flag False, combined_score == 0.0, should_abstain False."""
        result = compute_abstention_signals(
            chunk_count=5,
            top_chunk_score=0.9,
            entity_count=3,
        )

        assert result["entities_empty"] is False
        assert result["chunks_empty"] is False
        assert result["chunks_below_min"] is False
        assert result["top_score_low"] is False
        assert math.isclose(result["combined_score"], 0.0)
        assert result["should_abstain"] is False

    def test_chunks_empty_but_entities_present_does_not_abstain(self) -> None:
        """chunk_count=0, entity_count=5, top_chunk_score=0.9 → chunks_empty
        and chunks_below_min True, entities_empty and top_score_low False,
        combined_score == 0.4 (only chunks_below_min weight fires),
        should_abstain False (0.4 < 0.5 default threshold)."""
        result = compute_abstention_signals(
            chunk_count=0,
            top_chunk_score=0.9,
            entity_count=5,
        )

        assert result["entities_empty"] is False
        assert result["chunks_empty"] is True
        assert result["chunks_below_min"] is True
        assert result["top_score_low"] is False
        assert result["combined_score"] == pytest.approx(0.4)
        assert result["should_abstain"] is False

    def test_low_score_and_no_entities_triggers_abstain(self) -> None:
        """chunk_count=2, entity_count=0, top_chunk_score=0.2 →
        entities_empty and top_score_low True, chunks_empty and chunks_below_min
        False (since 2 >= min_chunks=1), combined_score == 0.6,
        should_abstain True (0.6 >= 0.5 default threshold)."""
        result = compute_abstention_signals(
            chunk_count=2,
            top_chunk_score=0.2,
            entity_count=0,
        )

        assert result["entities_empty"] is True
        assert result["chunks_empty"] is False
        assert result["chunks_below_min"] is False
        assert result["top_score_low"] is True
        assert result["combined_score"] == pytest.approx(0.6)
        assert result["should_abstain"] is True

    def test_custom_thresholds_round_trip(self) -> None:
        """Custom (min_chunks=3, min_top_score=0.5, combined_threshold=0.7)
        with chunk_count=2, entity_count=0, top_chunk_score=0.4 fires
        entities_empty, chunks_below_min, top_score_low; combined_score == 1.0;
        should_abstain True (only because combined_threshold=0.7 < 1.0)."""
        result = compute_abstention_signals(
            chunk_count=2,
            top_chunk_score=0.4,
            entity_count=0,
            min_chunks=3,
            min_top_score=0.5,
            combined_threshold=0.7,
        )

        assert result["entities_empty"] is True
        assert result["chunks_empty"] is False
        assert result["chunks_below_min"] is True
        assert result["top_score_low"] is True
        assert math.isclose(result["combined_score"], 1.0)
        # Threshold-bound assertion: combined_score (1.0) >= combined_threshold (0.7).
        assert result["should_abstain"] is True

    def test_custom_combined_threshold_blocks_abstention(self) -> None:
        """Same flag profile as scenario 4 (combined_score == 0.6), but a
        custom combined_threshold=0.7 flips should_abstain to False — proving
        the threshold parameter is actually wired through rather than
        hardcoded against the 0.5 default."""
        result = compute_abstention_signals(
            chunk_count=2,
            top_chunk_score=0.2,
            entity_count=0,
            combined_threshold=0.7,
        )

        assert result["combined_score"] == pytest.approx(0.6)
        assert result["should_abstain"] is False

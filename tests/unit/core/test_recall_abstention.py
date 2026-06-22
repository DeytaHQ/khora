"""Unit tests for ``khora.core.recall_abstention.compute_abstention_signals``.

Scenarios pin the formula, flag set, and ``should_abstain`` boundary; the
chronicle and vectorcypher engines both delegate to this helper, so the
public contract lives here.
"""

from __future__ import annotations

import math

import pytest

from khora.core.recall_abstention import compute_abstention_signals, compute_confidence


@pytest.mark.unit
class TestComputeAbstentionSignals:
    def test_all_signals_fire_when_nothing_retrieved(self) -> None:
        """chunk_count=0, entity_count=0, top_vector_score=0.0 → every flag True,
        combined_score == 1.0, should_abstain True."""
        result = compute_abstention_signals(
            chunk_count=0,
            top_vector_score=0.0,
            entity_count=0,
        )

        assert result["entities_empty"] is True
        assert result["chunks_empty"] is True
        assert result["chunks_below_min"] is True
        assert result["top_score_low"] is True
        assert math.isclose(result["combined_score"], 1.0)
        assert result["should_abstain"] is True

    def test_no_signals_fire_when_retrieval_is_healthy(self) -> None:
        """chunk_count=5, entity_count=3, top_vector_score=0.9 (defaults) →
        every flag False, combined_score == 0.0, should_abstain False."""
        result = compute_abstention_signals(
            chunk_count=5,
            top_vector_score=0.9,
            entity_count=3,
        )

        assert result["entities_empty"] is False
        assert result["chunks_empty"] is False
        assert result["chunks_below_min"] is False
        assert result["top_score_low"] is False
        assert math.isclose(result["combined_score"], 0.0)
        assert result["should_abstain"] is False

    def test_chunks_empty_but_entities_present_does_not_abstain(self) -> None:
        """chunk_count=0, entity_count=5, top_vector_score=0.9 → chunks_empty
        and chunks_below_min True, entities_empty and top_score_low False,
        combined_score == 0.4 (only chunks_below_min weight fires),
        should_abstain False (0.4 < 0.5 default threshold)."""
        result = compute_abstention_signals(
            chunk_count=0,
            top_vector_score=0.9,
            entity_count=5,
        )

        assert result["entities_empty"] is False
        assert result["chunks_empty"] is True
        assert result["chunks_below_min"] is True
        assert result["top_score_low"] is False
        assert result["combined_score"] == pytest.approx(0.4)
        assert result["should_abstain"] is False

    def test_low_score_and_no_entities_triggers_abstain(self) -> None:
        """chunk_count=2, entity_count=0, top_vector_score=0.2 →
        entities_empty and top_score_low True, chunks_empty and chunks_below_min
        False (since 2 >= min_chunks=1), combined_score == 0.6,
        should_abstain True (0.6 >= 0.5 default threshold)."""
        result = compute_abstention_signals(
            chunk_count=2,
            top_vector_score=0.2,
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
        with chunk_count=2, entity_count=0, top_vector_score=0.4 fires
        entities_empty, chunks_below_min, top_score_low; combined_score == 1.0;
        should_abstain True (only because combined_threshold=0.7 < 1.0)."""
        result = compute_abstention_signals(
            chunk_count=2,
            top_vector_score=0.4,
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
        hardcoded against the 0.5 default. The combined_threshold only governs
        the decision in mode='weighted' (#1331), so pin that mode here."""
        result = compute_abstention_signals(
            chunk_count=2,
            top_vector_score=0.2,
            entity_count=0,
            combined_threshold=0.7,
            mode="weighted",
        )

        assert result["combined_score"] == pytest.approx(0.6)
        assert result["should_abstain"] is False


@pytest.mark.unit
class TestCosineFloorMode:
    """The default cosine_floor mode (#1331): the topicality floor decides on
    its own; entity/chunk COUNTS only matter in the genuinely-empty case."""

    def test_default_flags_and_combined_score_unchanged(self) -> None:
        """Back-compat: on a populated namespace with a near-zero cosine the
        four boolean flags and combined_score reproduce the historical values.
        Only should_abstain's derivation changed (#1331)."""
        result = compute_abstention_signals(
            chunk_count=3,
            top_vector_score=0.013,
            entity_count=4,
        )

        # Flag set + combined_score are the documented public contract — pinned.
        assert result["entities_empty"] is False
        assert result["chunks_empty"] is False
        assert result["chunks_below_min"] is False
        assert result["top_score_low"] is True
        assert result["combined_score"] == pytest.approx(0.3)

    def test_globex_off_topic_query_abstains(self) -> None:
        """The issue's repro: an absent-account query still returns top-k chunks
        (chunks_below_min False) and connected entities (entities_empty False),
        but its raw cosine 0.013 < min_top_score 0.3. The old weighted rule
        capped combined at 0.3 < 0.5 and would NOT abstain; the new cosine_floor
        default abstains on the topicality floor alone."""
        result = compute_abstention_signals(
            chunk_count=3,
            top_vector_score=0.013,
            entity_count=4,
        )

        assert result["should_abstain"] is True

    def test_weighted_mode_restores_legacy_non_abstention(self) -> None:
        """The escape hatch: mode='weighted' keeps the exact old formula +
        threshold, so the same Globex query does NOT abstain (combined 0.3 <
        0.5) — proving back-compat is one flag away."""
        result = compute_abstention_signals(
            chunk_count=3,
            top_vector_score=0.013,
            entity_count=4,
            mode="weighted",
        )

        assert result["combined_score"] == pytest.approx(0.3)
        assert result["should_abstain"] is False

    def test_healthy_cosine_does_not_abstain(self) -> None:
        """False-abstention guard: a valid query at a healthy cosine (>=
        min_top_score) does NOT abstain under the default mode."""
        result = compute_abstention_signals(
            chunk_count=3,
            top_vector_score=0.62,
            entity_count=4,
        )

        assert result["top_score_low"] is False
        assert result["should_abstain"] is False

    def test_genuinely_empty_retrieval_abstains(self) -> None:
        """When retrieval is genuinely empty (no chunks AND no entities) the
        cosine_floor mode abstains even if the cosine input were somehow at the
        floor — the empty-retrieval liveness signal still counts."""
        result = compute_abstention_signals(
            chunk_count=0,
            top_vector_score=0.62,  # above floor, so top_score_low is False
            entity_count=0,
        )

        assert result["top_score_low"] is False
        assert result["should_abstain"] is True


@pytest.mark.unit
class TestComputeConfidence:
    """Calibrated retrieval confidence (#1331)."""

    def test_in_unit_interval(self) -> None:
        for cosine, gap in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.1), (0.013, 0.0), (2.0, 2.0)]:
            c = compute_confidence(top_cosine=cosine, top_score_gap=gap)
            assert 0.0 <= c <= 1.0

    def test_high_for_on_topic(self) -> None:
        """A topical hit at/above target_cosine with a decisive gap → near 1.0."""
        c = compute_confidence(top_cosine=0.6, top_score_gap=0.15)
        assert c == pytest.approx(1.0)

    def test_low_for_off_topic(self) -> None:
        """The Globex-style near-zero cosine with no gap → near 0.0."""
        c = compute_confidence(top_cosine=0.013, top_score_gap=0.0)
        assert c < 0.05

    def test_blend_weights(self) -> None:
        """top_cosine at half target (0.25/0.5=0.5) and gap at target →
        0.8*0.5 + 0.2*1.0 = 0.6."""
        c = compute_confidence(top_cosine=0.25, top_score_gap=0.1)
        assert c == pytest.approx(0.6)

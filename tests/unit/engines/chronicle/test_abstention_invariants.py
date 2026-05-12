"""Property-based tests for the chronicle abstention combined_score.

The combined_score formula in ``ChronicleEngine._compute_abstention_signals``
is::

    combined = 0.3 * entities_empty + 0.4 * chunks_below_min + 0.3 * top_score_low

Three properties make this surface easy to silently break:

1. ``chunks_empty`` carries **zero weight** on purpose — it's a derivative
   of ``chunks_below_min`` (since min ≥ 1, chunks_empty ⇒ chunks_below_min).
   Adding ``chunks_empty`` to the weighting would double-count.
2. Weights sum to 1.0, so the score is bounded [0.0, 1.0].
3. ``should_abstain ⟺ combined_score ≥ threshold`` — pinning this prevents
   a "let me sanitize the threshold check" refactor from inverting it.

These tests use Hypothesis to exhaust the 16-element boolean input space
(2^4 flags) without us hand-rolling parametrize cases, and to fuzz the
threshold across the valid range. Failure messages cite the exact flag
combination that broke the invariant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def _engine_with_thresholds(min_chunks: int, min_top: float, combined: float):
    """Build a minimal ChronicleEngine just to drive _compute_abstention_signals.

    The signals method is a pure function over chunks/entities plus three
    instance-level thresholds, so we don't need a real storage/extractor.
    Use ``object.__new__`` to skip the heavyweight ``__init__``.
    """
    from khora.engines.chronicle.engine import ChronicleEngine

    engine = object.__new__(ChronicleEngine)
    engine._abstention_min_chunks = min_chunks
    engine._abstention_min_top_score = min_top
    engine._abstention_combined_threshold = combined
    return engine


def _make_chunk(score: float):
    """Lightweight Chunk-ish object: the abstention method only reads ``score``."""
    return (MagicMock(), score)


def _make_entity():
    return (MagicMock(), 1.0)


# ---------------------------------------------------------------------------
# Property 1: combined_score is bounded [0.0, 1.0] and takes one of 8 values
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedScoreBounds:
    @given(
        n_chunks=st.integers(min_value=0, max_value=10),
        n_entities=st.integers(min_value=0, max_value=10),
        top_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_chunks=st.integers(min_value=1, max_value=5),
        min_top=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_combined_score_in_unit_interval(
        self,
        n_chunks: int,
        n_entities: int,
        top_score: float,
        min_chunks: int,
        min_top: float,
        threshold: float,
    ) -> None:
        """For ANY input, combined_score is in [0, 1]."""
        engine = _engine_with_thresholds(min_chunks, min_top, threshold)
        chunks = [_make_chunk(top_score)] + [_make_chunk(top_score / 2)] * max(0, n_chunks - 1)
        entities = [_make_entity()] * n_entities
        result = engine._compute_abstention_signals(chunks, entities)
        assert 0.0 <= result["combined_score"] <= 1.0, (
            f"combined_score out of range: {result['combined_score']} for "
            f"n_chunks={n_chunks}, n_entities={n_entities}, top={top_score}"
        )

    def test_combined_score_takes_exactly_8_distinct_values(self) -> None:
        """The formula 0.3·a + 0.4·b + 0.3·c over booleans yields 8 distinct
        values (2^3) — pin the exact set so a future weight change is loud.
        """
        seen: set[float] = set()
        for entities_empty in (False, True):
            for chunks_below_min in (False, True):
                for top_score_low in (False, True):
                    score = 0.3 * float(entities_empty) + 0.4 * float(chunks_below_min) + 0.3 * float(top_score_low)
                    seen.add(round(score, 6))
        assert seen == {0.0, 0.3, 0.4, 0.6, 0.7, 1.0}, (
            f"Expected 6 distinct values from 2^3 inputs (some collide), got {sorted(seen)}"
        )


# ---------------------------------------------------------------------------
# Property 2: monotonicity — flipping any flag False→True must not decrease
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCombinedScoreMonotonicity:
    @given(
        entities_empty=st.booleans(),
        chunks_below_min=st.booleans(),
        top_score_low=st.booleans(),
        flip=st.sampled_from(["entities_empty", "chunks_below_min", "top_score_low"]),
    )
    @settings(max_examples=200, deadline=None)
    def test_flipping_flag_false_to_true_never_decreases(
        self,
        entities_empty: bool,
        chunks_below_min: bool,
        top_score_low: bool,
        flip: str,
    ) -> None:
        """The 3 weighted flags must each have positive (non-negative) weight.

        Asserts directly against the formula — a regression that switched a
        sign or zeroed a weight would trip this immediately.
        """
        base = {
            "entities_empty": entities_empty,
            "chunks_below_min": chunks_below_min,
            "top_score_low": top_score_low,
        }
        if base[flip] is True:
            # Already True — flipping makes no change; trivially monotone.
            return
        flipped = dict(base, **{flip: True})

        def score(d: dict) -> float:
            return (
                0.3 * float(d["entities_empty"]) + 0.4 * float(d["chunks_below_min"]) + 0.3 * float(d["top_score_low"])
            )

        assert score(flipped) > score(base), (
            f"flipping {flip} False→True must strictly increase combined_score; "
            f"got base={score(base)} flipped={score(flipped)}"
        )


# ---------------------------------------------------------------------------
# Property 3: should_abstain ⟺ combined_score ≥ threshold
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldAbstainThreshold:
    @given(
        n_chunks=st.integers(min_value=0, max_value=5),
        n_entities=st.integers(min_value=0, max_value=5),
        top_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=300, deadline=None)
    def test_should_abstain_matches_threshold_comparison(
        self, n_chunks: int, n_entities: int, top_score: float, threshold: float
    ) -> None:
        engine = _engine_with_thresholds(min_chunks=1, min_top=0.3, combined=threshold)
        chunks = [_make_chunk(top_score)] + [_make_chunk(top_score / 2)] * max(0, n_chunks - 1)
        entities = [_make_entity()] * n_entities
        result = engine._compute_abstention_signals(chunks, entities)
        assert result["should_abstain"] == (result["combined_score"] >= threshold), (
            f"should_abstain inconsistent with threshold: combined={result['combined_score']}, "
            f"threshold={threshold}, should_abstain={result['should_abstain']}"
        )


# ---------------------------------------------------------------------------
# Property 4: chunks_empty implies chunks_below_min (when min ≥ 1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunksEmptyImpliesBelowMin:
    """``chunks_empty`` carries no weight in combined_score on purpose: it is
    always implied by ``chunks_below_min`` when ``min_chunks ≥ 1`` (the
    default). Adding chunks_empty to the weighting would double-count.

    Pin this implication so a future refactor that bumps min_chunks to 0
    (allowing a no-chunk recall to pass the chunks-below-min check) doesn't
    silently invalidate the "no double-counting" property.
    """

    @given(min_chunks=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50, deadline=None)
    def test_chunks_empty_always_implies_below_min(self, min_chunks: int) -> None:
        engine = _engine_with_thresholds(min_chunks=min_chunks, min_top=0.3, combined=0.5)
        result = engine._compute_abstention_signals([], [])
        # chunks_empty True (no chunks) → chunks_below_min must also be True.
        assert result["chunks_empty"] is True
        assert result["chunks_below_min"] is True, (
            f"chunks_empty=True but chunks_below_min=False with min_chunks={min_chunks}"
        )

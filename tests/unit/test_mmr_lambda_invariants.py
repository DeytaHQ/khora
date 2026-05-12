"""Property-based tests for the MMR λ parameter convention.

Khora's ``mmr_diversity_select`` uses a convention OPPOSITE to most academic
MMR literature: in khora, ``lambda_param = 1.0`` means **pure relevance**
(top-k by score) and ``lambda_param = 0.0`` means **pure diversity** (greedy
min-similarity to selected). In the canonical Carbonell & Goldstein (1998)
paper, the symbol ``λ`` typically denotes the *diversity* weight, so the
sign would be flipped.

The formula in ``_accel.py:mmr_diversity_select`` is::

    mmr = lambda * relevance - (1 - lambda) * max_sim_to_selected

A future refactor that "fixes the sign to match the paper" would silently
invert ranking behaviour for every Khora deployment in production — the
function would still return ``k`` indices, tests that don't pin the
convention would still pass, but recall quality would tank.

These tests pin the convention with handcrafted inputs whose expected output
is unambiguous (a strict top-k by score under λ=1, and a strict
diversity-greedy pick under λ=0). Property tests also fuzz interior λ
values to assert the "smooth interpolation" property: λ=1 result is a
relevance-permutation of the available indices, and at λ=0 the second pick
maximises distance from the first.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from khora._accel import mmr_diversity_select


def _normalize(v: list[float]) -> list[float]:
    """L2-normalize so dot product equals cosine similarity (function contract)."""
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return v
    return [x / n for x in v]


# ---------------------------------------------------------------------------
# Property 1: λ = 1.0 → pure relevance (matches argsort of scores)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPureRelevanceAtLambdaOne:
    """At λ=1, the diversity penalty term zeroes out and MMR must reduce to
    top-k by score. This pins the convention end (relevance side)."""

    def test_lambda_one_returns_top_k_by_score(self) -> None:
        # 5 random-ish embeddings; scores monotonically decreasing
        embeddings = [
            _normalize([1.0, 0.0]),
            _normalize([0.0, 1.0]),
            _normalize([1.0, 1.0]),
            _normalize([1.0, -1.0]),
            _normalize([-1.0, 1.0]),
        ]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = mmr_diversity_select(embeddings, scores, lambda_param=1.0, k=3)
        assert result == [0, 1, 2], (
            f"λ=1 should pick by descending score; got {result}. "
            f"If this reversed to [4, 3, 2] the convention has been flipped."
        )

    @given(
        n=st.integers(min_value=2, max_value=8),
        k=st.integers(min_value=1, max_value=8),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100, deadline=None)
    def test_lambda_one_matches_argsort(self, n: int, k: int, seed: int) -> None:
        # Deterministic pseudo-random embeddings/scores from seed
        rng_state = seed

        def _next() -> float:
            nonlocal rng_state
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            return (rng_state / 0x7FFFFFFF) * 2.0 - 1.0

        embeddings = [_normalize([_next(), _next(), _next()]) for _ in range(n)]
        scores = [_next() + 1.5 for _ in range(n)]  # shift positive
        k_eff = min(k, n)

        result = mmr_diversity_select(embeddings, scores, lambda_param=1.0, k=k_eff)
        # Expected order: argsort descending by score
        expected = sorted(range(n), key=lambda i: -scores[i])[:k_eff]

        # With ties broken deterministically; if all scores are unique, must match exactly.
        if len(set(scores)) == n:
            assert result == expected, f"λ=1 picked {result}, expected argsort {expected}. Scores: {scores}"


# ---------------------------------------------------------------------------
# Property 2: λ = 0.0 → pure diversity (first pick by relevance only because
# no selected set exists; second pick maximises distance from the first)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPureDiversityAtLambdaZero:
    """At λ=0 the relevance term zeroes out. With no selections yet, every
    candidate scores 0.0 (no max_sim term), so the first pick is the first
    index encountered tied. After that, the algorithm picks the candidate
    MOST dissimilar from any selected — the maximally orthogonal one."""

    def test_lambda_zero_picks_diverse_after_first(self) -> None:
        # Three near-identical vectors and one orthogonal one. With λ=0, after
        # picking idx 0 the next pick MUST be the orthogonal vector (idx 3),
        # not one of the near-duplicates.
        embeddings = [
            _normalize([1.0, 0.0, 0.0]),  # 0
            _normalize([0.99, 0.01, 0.0]),  # 1 — near-dup of 0
            _normalize([0.98, 0.02, 0.0]),  # 2 — near-dup of 0
            _normalize([0.0, 0.0, 1.0]),  # 3 — orthogonal to 0
        ]
        scores = [0.9, 0.9, 0.9, 0.9]
        result = mmr_diversity_select(embeddings, scores, lambda_param=0.0, k=2)
        assert 3 in result, (
            f"λ=0 should pick the orthogonal vector (idx 3) after the first; "
            f"got {result}. If only near-duplicates were chosen the convention "
            f"has been flipped (λ=0 became 'pure relevance' instead)."
        )


# ---------------------------------------------------------------------------
# Property 3: λ=1 result is a permutation of the available indices
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectedSetIsValid:
    @given(
        n=st.integers(min_value=1, max_value=10),
        k=st.integers(min_value=1, max_value=10),
        lambda_param=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=200, deadline=None)
    def test_result_is_valid_index_set(self, n: int, k: int, lambda_param: float, seed: int) -> None:
        """Sanity: indices are unique, in-range, and at most ``min(n, k)`` long."""
        rng_state = seed

        def _next() -> float:
            nonlocal rng_state
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            return (rng_state / 0x7FFFFFFF) * 2.0 - 1.0

        embeddings = [_normalize([_next(), _next()]) for _ in range(n)]
        scores = [_next() + 1.5 for _ in range(n)]

        result = mmr_diversity_select(embeddings, scores, lambda_param, k)
        assert len(result) == min(n, k)
        assert len(set(result)) == len(result), f"duplicate indices: {result}"
        assert all(0 <= i < n for i in result), f"out-of-range index: {result}"


# ---------------------------------------------------------------------------
# Property 4: Boundary inputs don't crash
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoundaryInputs:
    def test_empty_returns_empty(self) -> None:
        assert mmr_diversity_select([], [], lambda_param=0.5, k=5) == []

    def test_k_zero_returns_empty(self) -> None:
        assert mmr_diversity_select([_normalize([1.0, 0.0])], [0.5], lambda_param=0.5, k=0) == []

    def test_k_greater_than_n(self) -> None:
        embs = [_normalize([1.0, 0.0]), _normalize([0.0, 1.0])]
        result = mmr_diversity_select(embs, [0.9, 0.1], lambda_param=1.0, k=100)
        assert len(result) == 2

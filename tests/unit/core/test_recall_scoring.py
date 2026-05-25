"""Unit tests for ``khora.core.recall_scoring.min_max_normalize``.

Reporter Damir's #834: ``RecallChunk.score`` had three different meanings
across the three engines (VC normalized in [0, 1], Chronicle post-rerank
fused on an arbitrary scale, Skeleton raw cosine / BM25). The shared helper
collapses any score-set into the unified contract - top = 1.0, bottom = 0.0.
"""

from __future__ import annotations

import pytest

from khora.core.recall_scoring import min_max_normalize


def test_min_max_normalize_three_distinct_scores():
    """Top score maps to 1.0, bottom to 0.0, middle strictly between."""
    out = min_max_normalize([0.7285, 0.0236, 0.0171])

    assert out[0] == 1.0
    assert out[-1] == 0.0
    assert 0.0 < out[1] < 1.0


def test_min_max_normalize_preserves_order():
    """Output index i corresponds to input index i."""
    out = min_max_normalize([0.5, 0.9, 0.1])

    # 0.9 is the max -> 1.0 at index 1.
    assert out[1] == 1.0
    # 0.1 is the min -> 0.0 at index 2.
    assert out[2] == 0.0
    # 0.5 lands between, at index 0.
    assert out[0] == pytest.approx((0.5 - 0.1) / (0.9 - 0.1))


def test_min_max_normalize_single_element_is_one():
    """Single-element input collapses to [1.0]."""
    assert min_max_normalize([0.42]) == [1.0]


def test_min_max_normalize_all_tied_collapse_to_one():
    """When max == min, every entry collapses to 1.0."""
    assert min_max_normalize([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_min_max_normalize_empty_list_is_empty():
    """Empty input stays empty - no normalization needed."""
    assert min_max_normalize([]) == []


def test_min_max_normalize_handles_unbounded_bm25_range():
    """BM25 scores are unbounded reals - normalization still produces [0, 1]."""
    out = min_max_normalize([42.7, 12.1, 8.3, 0.04])

    assert out[0] == 1.0
    assert out[-1] == 0.0
    assert all(0.0 <= s <= 1.0 for s in out)

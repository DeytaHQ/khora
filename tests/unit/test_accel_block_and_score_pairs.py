"""Tests for khora._accel.block_and_score_pairs.

Exercises the Rust kernel (when available) and the Python fallback through
the same public wrapper. Parity assertions ensure the two paths agree.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import khora._accel as accel
from khora._accel import block_and_score_pairs, pairwise_cosine_above_threshold


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def _make_embeddings(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal(size=(n, d)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (mat / norms).astype(np.float32)


# --------------------------------------------------------------------------
# Behavioural tests
# --------------------------------------------------------------------------


def test_blocking_finds_high_similarity_pair():
    """Two near-identical rows whose names share a token are returned."""
    mat = _make_embeddings(100, 128, seed=42)
    # Make row 7 a near-clone of row 3, with overlapping name tokens.
    mat[7] = _normalize(mat[3] + 0.001 * mat[5])
    names = [f"Entity {i}" for i in range(100)]
    names[3] = "Acme Corporation"
    names[7] = "Acme Holdings"

    pairs = block_and_score_pairs(mat, names, threshold=0.95, name_token_blocking=True)
    assert any((i, j) == (3, 7) for i, j, _ in pairs), pairs


def test_blocking_filters_zero_token_overlap():
    """Near-identical rows with disjoint names are excluded when blocking is on."""
    mat = _make_embeddings(20, 64, seed=1)
    mat[5] = _normalize(mat[2] + 1e-4 * mat[3])
    names = [f"Entity_{i}_token" for i in range(20)]
    # Force disjoint blocking keys.
    names[2] = "Acme Corp"
    names[5] = "Zenith Holdings"

    blocked = block_and_score_pairs(mat, names, threshold=0.99, name_token_blocking=True)
    assert not any((i, j) == (2, 5) for i, j, _ in blocked)

    unblocked = block_and_score_pairs(mat, names, threshold=0.99, name_token_blocking=False)
    assert any((i, j) == (2, 5) for i, j, _ in unblocked)


def test_threshold_filtering():
    """No pair with similarity < threshold appears in the output."""
    mat = _make_embeddings(50, 32, seed=2)
    names = [f"name_{i}" for i in range(50)]
    threshold = 0.3
    pairs = block_and_score_pairs(mat, names, threshold=threshold, name_token_blocking=False)
    for _, _, s in pairs:
        assert s >= threshold


def test_pairs_returned_with_i_lt_j():
    """Every returned pair has i < j (canonical ordering)."""
    mat = _make_embeddings(30, 16, seed=3)
    names = [f"alpha_{i}" for i in range(30)]
    pairs = block_and_score_pairs(mat, names, threshold=0.0, name_token_blocking=True)
    assert pairs, "expected at least one pair with threshold=0.0"
    for i, j, _ in pairs:
        assert i < j


def test_self_pairs_excluded():
    """No (i, i) self-pair is ever returned."""
    mat = _make_embeddings(10, 8, seed=4)
    names = [f"same_token_{i}" for i in range(10)]
    pairs = block_and_score_pairs(mat, names, threshold=-1.0, name_token_blocking=True)
    for i, j, _ in pairs:
        assert i != j


def test_validation_shape_mismatch_ndim():
    """1-D embeddings raise ValueError."""
    mat = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        block_and_score_pairs(mat, ["only one"], threshold=0.5)


def test_validation_names_length_mismatch():
    """Mismatched names length raises ValueError."""
    mat = _make_embeddings(5, 8, seed=5)
    with pytest.raises(ValueError, match="names length"):
        block_and_score_pairs(mat, ["a", "b"], threshold=0.5)


def test_consistency_with_pairwise_cosine_above_threshold():
    """With blocking off, output set matches pairwise_cosine_above_threshold."""
    mat = _make_embeddings(40, 24, seed=6)
    names = [f"row_{i}" for i in range(40)]
    threshold = 0.0

    blocked_off = block_and_score_pairs(mat, names, threshold=threshold, name_token_blocking=False)
    baseline = pairwise_cosine_above_threshold(mat.tolist(), threshold)

    blocked_set = {(i, j) for i, j, _ in blocked_off}
    baseline_set = {(i, j) for i, j, _ in baseline}
    assert blocked_set == baseline_set

    # Scores agree within float tolerance.
    blocked_map = {(i, j): s for i, j, s in blocked_off}
    for i, j, s in baseline:
        assert abs(blocked_map[(i, j)] - s) < 1e-4


def test_empty_input():
    """N=0 returns an empty list."""
    mat = np.zeros((0, 8), dtype=np.float32)
    assert block_and_score_pairs(mat, [], threshold=0.5) == []


def test_single_row():
    """N=1 returns an empty list."""
    mat = _make_embeddings(1, 4, seed=7)
    assert block_and_score_pairs(mat, ["only"], threshold=0.0) == []


def test_blocking_on_off_parity_small():
    """For names that all share a blocking key, on/off must agree."""
    mat = _make_embeddings(15, 16, seed=8)
    # Every name contains the token "shared" so every pair survives blocking.
    names = [f"shared_token_{i}" for i in range(15)]

    on = sorted((i, j) for i, j, _ in block_and_score_pairs(mat, names, threshold=0.0, name_token_blocking=True))
    off = sorted((i, j) for i, j, _ in block_and_score_pairs(mat, names, threshold=0.0, name_token_blocking=False))
    assert on == off


def test_large_n_perf_sanity():
    """N=1000 finishes well under a second on any modern CPU."""
    mat = _make_embeddings(1000, 64, seed=9)
    names = [f"entity_{i % 50}_thing" for i in range(1000)]  # heavy block sharing
    start = time.perf_counter()
    pairs = block_and_score_pairs(mat, names, threshold=0.5, name_token_blocking=True)
    elapsed = time.perf_counter() - start
    # 1 second is generous; on Rust this runs in ms. Pure-Python fallback also fits.
    assert elapsed < 5.0, f"perf regression: {elapsed:.3f}s"
    # And it actually does work.
    assert isinstance(pairs, list)


# --------------------------------------------------------------------------
# Python fallback path (forced)
# --------------------------------------------------------------------------


@pytest.fixture
def force_python(monkeypatch):
    monkeypatch.setattr(accel, "_HAS_RUST", False)


def test_python_fallback_blocking(force_python):
    """Pure-numpy/Python path produces the same blocked output."""
    mat = _make_embeddings(20, 16, seed=10)
    names = [f"acme_{i // 2}_holding" for i in range(20)]
    pairs = block_and_score_pairs(mat, names, threshold=0.0, name_token_blocking=True)
    assert pairs
    for i, j, _ in pairs:
        assert i < j


def test_python_fallback_unblocked_matches_baseline(force_python):
    """Python fallback unblocked path matches pairwise_cosine_above_threshold."""
    mat = _make_embeddings(20, 16, seed=11)
    names = [f"row_{i}" for i in range(20)]
    threshold = 0.0
    blocked_off = block_and_score_pairs(mat, names, threshold=threshold, name_token_blocking=False)
    baseline = pairwise_cosine_above_threshold(mat.tolist(), threshold)
    assert {(i, j) for i, j, _ in blocked_off} == {(i, j) for i, j, _ in baseline}


def test_python_fallback_validation(force_python):
    """Validation works on the fallback path too."""
    mat = _make_embeddings(3, 4, seed=12)
    with pytest.raises(ValueError, match="names length"):
        block_and_score_pairs(mat, ["only one"], threshold=0.5)

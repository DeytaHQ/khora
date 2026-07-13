"""Tests for khora._accel — Rust-accelerated operations with Python fallbacks.

Each function is tested under two backends:
  1. Default (Rust, if available)
  2. Pure Python (via monkeypatching _HAS_RUST/_HAS_NUMPY/_HAS_RAPIDFUZZ)
"""

from __future__ import annotations

import pytest

import khora._accel as accel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def force_python(monkeypatch):
    """Force the pure-Python path for all functions."""
    monkeypatch.setattr(accel, "_HAS_RUST", False)
    monkeypatch.setattr(accel, "_HAS_NUMPY", False)
    monkeypatch.setattr(accel, "_HAS_RAPIDFUZZ", False)


@pytest.fixture()
def force_numpy(monkeypatch):
    """Force numpy path (skip Rust, keep numpy)."""
    monkeypatch.setattr(accel, "_HAS_RUST", False)
    # Keep _HAS_NUMPY and _HAS_RAPIDFUZZ at their real values


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert accel.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert accel.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_dimension_mismatch_python(self, force_python):
        assert accel.cosine_similarity([1, 0], [1, 0, 0]) == 0.0

    def test_zero_vector_python(self, force_python):
        assert accel.cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_identical_python(self, force_python):
        assert accel.cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_similar_python(self, force_python):
        sim = accel.cosine_similarity([1, 1], [1, 0])
        assert 0.5 < sim < 1.0

    def test_numpy_path(self, force_numpy):
        assert accel.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_numpy_zero_vector(self, force_numpy):
        assert accel.cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_numpy_mismatch(self, force_numpy):
        assert accel.cosine_similarity([1, 0], [1, 0, 0]) == 0.0


class TestBatchCosineSimilarity:
    def test_empty_candidates(self):
        assert accel.batch_cosine_similarity([1, 0], []) == []

    def test_with_threshold(self, force_python):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        results = accel.batch_cosine_similarity(query, candidates, threshold=0.5)
        indices = [idx for idx, _ in results]
        assert 0 in indices  # identical
        assert 1 not in indices  # orthogonal
        assert 2 in indices  # similar enough

    def test_sorted_descending(self, force_python):
        query = [1.0, 0.0]
        candidates = [[0.5, 0.5], [1.0, 0.0], [0.1, 0.9]]
        results = accel.batch_cosine_similarity(query, candidates)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_numpy_path(self, force_numpy):
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        results = accel.batch_cosine_similarity(query, candidates, threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_numpy_zero_query(self, force_numpy):
        results = accel.batch_cosine_similarity([0.0, 0.0], [[1.0, 0.0]])
        assert results == []


# ---------------------------------------------------------------------------
# Dimension-mismatch parity (#1132)
#
# When the query dimension differs from the candidate columns, the NumPy
# fallback (`mat @ q`) raises ValueError. The Rust kernels previously zipped
# to the shorter length and returned a wrong-but-plausible score. Both paths
# must now raise ValueError on the same input so a mixed-embedding-dim store
# fails loudly with or without the compiled wheel.
# ---------------------------------------------------------------------------


class TestBatchDimensionMismatch:
    QUERY = [1.0, 2.0]  # dim 2
    CANDIDATES = [[1.0, 2.0, 3.0, 4.0]]  # dim 4

    def test_cosine_numpy_raises(self, force_numpy):
        if not accel._HAS_NUMPY:
            pytest.skip("numpy not available")
        with pytest.raises(ValueError):
            accel.batch_cosine_similarity(self.QUERY, self.CANDIDATES)

    def test_cosine_rust_raises(self):
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        with pytest.raises(ValueError):
            accel.batch_cosine_similarity(self.QUERY, self.CANDIDATES)

    def test_dot_numpy_raises(self, force_numpy):
        if not accel._HAS_NUMPY:
            pytest.skip("numpy not available")
        with pytest.raises(ValueError):
            accel.batch_dot_product(self.QUERY, self.CANDIDATES)

    def test_dot_rust_raises(self):
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        with pytest.raises(ValueError):
            accel.batch_dot_product(self.QUERY, self.CANDIDATES)


class TestPairwiseCosineSimilarity:
    def test_two_identical(self, force_python):
        result = accel.pairwise_cosine_above_threshold([[1, 0], [1, 0]], 0.5)
        assert len(result) == 1
        i, j, sim = result[0]
        assert i == 0 and j == 1
        assert sim == pytest.approx(1.0)

    def test_below_threshold(self, force_python):
        result = accel.pairwise_cosine_above_threshold([[1, 0], [0, 1]], 0.5)
        assert len(result) == 0

    def test_single_vector(self, force_python):
        assert accel.pairwise_cosine_above_threshold([[1, 0]], 0.5) == []

    def test_empty(self, force_python):
        assert accel.pairwise_cosine_above_threshold([], 0.5) == []

    def test_zero_vector_skipped(self, force_python):
        result = accel.pairwise_cosine_above_threshold([[1, 0], [0, 0], [1, 0]], 0.5)
        # Only (0, 2) should appear, zero vector skipped
        assert len(result) == 1
        assert result[0][0] == 0 and result[0][1] == 2

    def test_numpy_path(self, force_numpy):
        result = accel.pairwise_cosine_above_threshold([[1, 0], [1, 0], [0, 1]], 0.5)
        # (0,1) should match, (0,2) and (1,2) should not
        assert len(result) == 1
        assert result[0][0] == 0 and result[0][1] == 1

    def test_numpy_zero_skipped(self, force_numpy):
        result = accel.pairwise_cosine_above_threshold([[1, 0], [0, 0]], 0.5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Levenshtein similarity
# ---------------------------------------------------------------------------


class TestLevenshteinSimilarity:
    def test_identical(self):
        assert accel.levenshtein_similarity("hello", "hello") == pytest.approx(1.0)

    def test_empty(self):
        assert accel.levenshtein_similarity("", "hello") == pytest.approx(0.0)

    def test_similar_python(self, force_python):
        sim = accel.levenshtein_similarity("hello", "hallo")
        assert 0.5 < sim < 1.0

    def test_case_insensitive_python(self, force_python):
        assert accel.levenshtein_similarity("Hello", "hello") == pytest.approx(1.0)

    def test_rapidfuzz_path(self, force_numpy):
        # force_numpy disables Rust but keeps rapidfuzz
        sim = accel.levenshtein_similarity("hello", "hallo")
        assert 0.5 < sim < 1.0

    def test_rapidfuzz_equal(self, force_numpy):
        assert accel.levenshtein_similarity("abc", "abc") == pytest.approx(1.0)


class TestBatchLevenshtein:
    def test_basic(self, force_python):
        results = accel.batch_levenshtein("cat", ["cat", "car", "dog"], threshold=0.5)
        indices = [idx for idx, _ in results]
        assert 0 in indices  # exact match
        assert 1 in indices  # close match
        # "dog" should have low similarity to "cat"

    def test_sorted_descending(self, force_python):
        results = accel.batch_levenshtein("hello", ["hello", "helo", "world"])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Sequence match ratio
# ---------------------------------------------------------------------------


class TestSequenceMatchRatio:
    def test_identical(self):
        assert accel.sequence_match_ratio("abc", "abc") == pytest.approx(1.0)

    def test_different_python(self, force_python):
        # Use difflib fallback
        ratio = accel.sequence_match_ratio("abc", "xyz")
        assert ratio < 0.5

    def test_rapidfuzz_path(self, force_numpy):
        ratio = accel.sequence_match_ratio("abc", "abc")
        assert ratio == pytest.approx(1.0)


class TestBatchSequenceMatch:
    def test_sorted(self, force_python):
        results = accel.batch_sequence_match("hello", ["hello", "help", "xyz"])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------


class TestPageRank:
    def test_empty_graph(self):
        assert accel.pagerank(0, []) == []

    def test_cycle_graph(self):
        # 3-node cycle: all nodes should have equal score ~0.333
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        scores = accel.pagerank(3, edges, damping=0.85, max_iter=100, tol=1e-6)
        assert len(scores) == 3
        for s in scores:
            assert s == pytest.approx(1.0 / 3, abs=0.01)

    def test_star_graph(self):
        # Node 0 receives links from 1, 2, 3 -> highest score
        edges = [(1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0)]
        scores = accel.pagerank(4, edges, damping=0.85)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
        assert scores[0] > scores[3]

    def test_python_fallback(self, force_python):
        edges = [(0, 1, 1.0), (1, 0, 1.0)]
        scores = accel.pagerank(2, edges, damping=0.85, max_iter=100, tol=1e-6)
        assert len(scores) == 2
        assert scores[0] == pytest.approx(0.5, abs=0.01)

    def test_empty_python(self, force_python):
        assert accel.pagerank(0, []) == []

    def test_star_python(self, force_python):
        edges = [(1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0)]
        scores = accel.pagerank(4, edges, damping=0.85)
        assert scores[0] > scores[1]

    # -----------------------------------------------------------------------
    # Personalized PageRank (Issue #597)
    # -----------------------------------------------------------------------

    def test_ppr_seeded_chain(self):
        """PPR on a 3-node chain seeded at node 0: seed dominates, then 1, then 2."""
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        scores = accel.pagerank(3, edges, damping=0.85, max_iter=200, tol=1e-8, personalization=[1.0, 0.0, 0.0])
        assert scores[0] > scores[1]
        assert scores[1] > scores[2]

    def test_ppr_uniform_matches_default(self):
        """An explicit uniform personalization vector matches None (standard PageRank)."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        default = accel.pagerank(3, edges, damping=0.85, max_iter=200, tol=1e-9)
        uniform = accel.pagerank(3, edges, damping=0.85, max_iter=200, tol=1e-9, personalization=[1.0 / 3] * 3)
        for d, u in zip(default, uniform, strict=True):
            assert d == pytest.approx(u, abs=1e-6)

    def test_ppr_length_mismatch_falls_back_to_uniform(self):
        """Wrong-length vector → uniform fallback; must not raise."""
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
        scores = accel.pagerank(3, edges, damping=0.85, max_iter=100, tol=1e-6, personalization=[1.0, 0.0])
        # symmetric cycle + uniform → equal scores
        assert scores[0] == pytest.approx(scores[1], abs=1e-3)
        assert scores[1] == pytest.approx(scores[2], abs=1e-3)

    def test_ppr_negatives_clipped(self):
        """Negative entries clipped to 0; non-negative subset L1-normalized."""
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        # Seed [0.7, -1.0, 0.3] → clipped to [0.7, 0, 0.3] → normalized [0.7, 0, 0.3]
        scores = accel.pagerank(3, edges, damping=0.85, max_iter=200, tol=1e-8, personalization=[0.7, -1.0, 0.3])
        # Node 0 has the most mass and seeds node 1 via the edge; node 2 is also
        # seeded directly and is a sink with no out-edge, so it accumulates more.
        assert scores[0] > 0
        assert scores[2] > 0

    def test_ppr_python_fallback_seeded_chain(self, force_python):
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        scores = accel.pagerank(3, edges, damping=0.85, max_iter=200, tol=1e-8, personalization=[1.0, 0.0, 0.0])
        assert scores[0] > scores[1]
        assert scores[1] > scores[2]


# ---------------------------------------------------------------------------
# PageRank top-k rank-stability early-stop (#1476)
# ---------------------------------------------------------------------------


def _pr_test_graph(n: int) -> list[tuple[int, int, float]]:
    """Deterministic scale-free-ish weighted graph.

    Low-index hub nodes accumulate the most PR mass, so the top-k set stabilizes
    well before global-L1 convergence — the property the #1476 early-stop
    exploits. Mirrors ``build_test_graph`` in ``pagerank.rs`` so the Rust and
    Python paths are exercised on the same shape.
    """
    edges: list[tuple[int, int, float]] = []
    for dst in range(1, n):
        for step in (1, 2, 3):
            src = (dst * step) % dst
            w = 1.0 + float((dst * 7 + step * 13) % 5)
            edges.append((src, dst, w))
            edges.append((dst, src, w))
    return edges


def _top(scores: list[float], k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


class TestPageRankEarlyStop:
    def test_top_k_indices_helper_tie_break(self, force_python):
        # Ties broken by ascending index; descending by score otherwise.
        assert accel._top_k_indices([0.5, 0.5, 0.5, 0.5], 3) == [0, 1, 2]
        assert accel._top_k_indices([0.1, 0.9, 0.5, 0.7], 2) == [1, 3]
        # k larger than n clamps.
        assert accel._top_k_indices([0.2, 0.4], 10) == [1, 0]

    def test_early_stop_topk_byte_identical_python(self, force_python):
        # Pure-Python path: early-stopped top-30 must be byte-identical to the
        # full-iteration top-30 — the #1476 safety guarantee.
        n = 400
        edges = _pr_test_graph(n)
        seed = [1.0 if i % 40 == 0 else 0.0 for i in range(n)]
        full = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed)
        early = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed, rank_k=40, stable_iters=3)
        assert _top(full, 30) == _top(early, 30)

    def test_early_stop_disabled_matches_legacy_python(self, force_python):
        # rank_k=None and stable_iters=0 both reproduce the pure global-L1 path.
        n = 200
        edges = _pr_test_graph(n)
        seed = [1.0 if i < 3 else 0.0 for i in range(n)]
        legacy = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed)
        rank_none = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed, rank_k=None, stable_iters=3)
        zero_patience = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed, rank_k=40, stable_iters=0)
        assert legacy == rank_none == zero_patience

    @pytest.mark.skipif(not accel._HAS_RUST, reason="Rust extension not built")
    def test_rust_python_parity_early_stop(self, monkeypatch):
        # The Rust kernel and the pure-Python fallback must produce bit-identical
        # scores AND a byte-identical top-30 under the early-stop, so switching
        # backends never changes results (the known Rust-vs-numpy parity concern).
        n = 500
        edges = _pr_test_graph(n)
        seed = [1.0 if i < 5 else 0.0 for i in range(n)]

        rust = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed, rank_k=40, stable_iters=3)

        monkeypatch.setattr(accel, "_HAS_RUST", False)
        py = accel.pagerank(n, edges, 0.85, 50, 1e-5, personalization=seed, rank_k=40, stable_iters=3)

        assert len(rust) == len(py) == n
        for r, p in zip(rust, py, strict=True):
            assert r == pytest.approx(p, abs=1e-12, rel=1e-12)
        assert _top(rust, 30) == _top(py, 30)

    @pytest.mark.skipif(not accel._HAS_RUST, reason="Rust extension not built")
    def test_rust_python_parity_no_early_stop(self, monkeypatch):
        # Parity must also hold on the legacy (no early-stop) path.
        n = 300
        edges = _pr_test_graph(n)
        seed = [1.0 if i < 4 else 0.0 for i in range(n)]

        rust = accel.pagerank(n, edges, 0.85, 50, 1e-6, personalization=seed)
        monkeypatch.setattr(accel, "_HAS_RUST", False)
        py = accel.pagerank(n, edges, 0.85, 50, 1e-6, personalization=seed)
        for r, p in zip(rust, py, strict=True):
            assert r == pytest.approx(p, abs=1e-12, rel=1e-12)


# ---------------------------------------------------------------------------
# Build chunk edges
# ---------------------------------------------------------------------------


class TestBuildChunkEdges:
    def test_basic(self):
        # Two keywords: kw0 shared by chunks [0,1], kw1 shared by [1,2]
        edges = accel.build_chunk_edges(3, [[0, 1], [1, 2]], [1.5, 2.0])
        # kw0 -> (0,1,1.5) + (1,0,1.5)
        # kw1 -> (1,2,2.0) + (2,1,2.0)
        assert (0, 1, 1.5) in edges
        assert (1, 0, 1.5) in edges
        assert (1, 2, 2.0) in edges
        assert (2, 1, 2.0) in edges

    def test_out_of_range(self):
        # Chunk index >= n_chunks should be skipped
        edges = accel.build_chunk_edges(2, [[0, 5]], [1.0])
        assert len(edges) == 0

    def test_python_fallback(self, force_python):
        edges = accel.build_chunk_edges(3, [[0, 1, 2]], [1.0])
        # 3 chunks sharing 1 keyword -> 6 directed edges (3 pairs * 2)
        assert len(edges) == 6

    def test_empty(self, force_python):
        assert accel.build_chunk_edges(0, [], []) == []


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_basic(self):
        kws = accel.extract_keywords("The quick brown fox jumps over the lazy dog")
        assert "quick" in kws
        assert "brown" in kws
        assert "fox" in kws
        # Stopwords removed
        assert "the" not in kws
        assert "and" not in kws

    def test_deduplication(self):
        kws = accel.extract_keywords("hello hello hello world world")
        assert kws.count("hello") == 1
        assert kws.count("world") == 1

    def test_short_words_filtered(self):
        kws = accel.extract_keywords("I am a big fan of AI")
        # "am", "AI", "a", "I" are < 3 chars or stopwords
        assert "fan" in kws
        assert "big" in kws

    def test_python_fallback(self, force_python):
        kws = accel.extract_keywords("The quick brown fox")
        assert "quick" in kws
        assert "the" not in kws


class TestExtractKeywordsBatch:
    def test_batch(self):
        results = accel.extract_keywords_batch(["hello world", "foo bar baz"])
        assert len(results) == 2
        assert "hello" in results[0]
        assert "foo" in results[1]

    def test_python_fallback(self, force_python):
        results = accel.extract_keywords_batch(["The quick fox", "lazy dog"])
        assert len(results) == 2
        assert "quick" in results[0]


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_basic(self):
        results = accel.reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
        ids = [r[0] for r in results]
        # "b" appears in both lists, should have highest score
        assert ids[0] == "b"

    def test_single_list(self):
        results = accel.reciprocal_rank_fusion([["x", "y"]], k=60)
        assert len(results) == 2

    def test_python_fallback(self, force_python):
        results = accel.reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
        ids = [r[0] for r in results]
        assert ids[0] == "b"


class TestWeightedRRF:
    def test_basic(self):
        results = accel.weighted_rrf([(0.8, ["a", "b"]), (0.2, ["b", "c"])], k=60)
        ids = [r[0] for r in results]
        assert "b" in ids

    def test_python_fallback(self, force_python):
        results = accel.weighted_rrf([(0.6, ["x"]), (0.4, ["x", "y"])], k=60)
        ids = [r[0] for r in results]
        assert ids[0] == "x"


class TestNormalizeScores:
    def test_basic(self):
        result = accel.normalize_scores([1.0, 2.0, 3.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_identical_scores(self):
        result = accel.normalize_scores([5.0, 5.0, 5.0])
        assert all(s == pytest.approx(1.0) for s in result)

    def test_empty(self):
        assert accel.normalize_scores([]) == []

    def test_python_fallback(self, force_python):
        result = accel.normalize_scores([0.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


class TestResolveEntitiesBatch:
    def test_exact_match(self):
        results = accel.resolve_entities_batch(
            ["Apple Inc"],
            ["apple inc", "Google"],
            [[], []],
            threshold=0.8,
        )
        assert len(results) == 1
        assert results[0] is not None
        idx, score, match_type = results[0]
        assert idx == 0
        assert score == pytest.approx(1.0)
        assert match_type == "exact"

    def test_alias_match(self):
        results = accel.resolve_entities_batch(
            ["GOOG"],
            ["Google", "Apple"],
            [["goog", "alphabet"], ["aapl"]],
            threshold=0.8,
        )
        assert results[0] is not None
        idx, score, match_type = results[0]
        assert idx == 0
        assert match_type == "alias"

    def test_fuzzy_match(self):
        results = accel.resolve_entities_batch(
            ["Gogle"],  # typo
            ["Google", "Apple"],
            [[], []],
            threshold=0.5,
        )
        assert results[0] is not None
        idx, score, match_type = results[0]
        assert idx == 0
        assert match_type == "fuzzy"

    def test_no_match(self):
        results = accel.resolve_entities_batch(
            ["XYZ Corp"],
            ["Google", "Apple"],
            [[], []],
            threshold=0.99,
        )
        assert results[0] is None

    def test_python_exact(self, force_python):
        results = accel.resolve_entities_batch(
            ["Test"],
            ["test"],
            [[]],
            threshold=0.8,
        )
        assert results[0] is not None
        assert results[0][2] == "exact"

    def test_python_alias(self, force_python):
        results = accel.resolve_entities_batch(
            ["nyc"],
            ["New York City"],
            [["nyc", "ny"]],
            threshold=0.8,
        )
        assert results[0] is not None
        assert results[0][2] == "alias"

    def test_python_fuzzy(self, force_python):
        results = accel.resolve_entities_batch(
            ["Gogle"],
            ["Google"],
            [[]],
            threshold=0.5,
        )
        assert results[0] is not None
        assert results[0][2] == "fuzzy"

    def test_python_no_match(self, force_python):
        results = accel.resolve_entities_batch(
            ["ZZZZZ"],
            ["Google"],
            [[]],
            threshold=0.99,
        )
        assert results[0] is None


# ---------------------------------------------------------------------------
# RustBM25Index
# ---------------------------------------------------------------------------


class TestRustBM25Index:
    def test_available(self):
        """RustBM25Index should be importable when Rust is available."""
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        assert accel.RustBM25Index is not None

    def test_add_and_search(self):
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        idx = accel.RustBM25Index()
        idx.add_document("doc1", "the quick brown fox")
        idx.add_document("doc2", "the lazy brown dog")
        idx.add_document("doc3", "unrelated content here")
        results = idx.search("brown fox", limit=5)
        assert len(results) > 0
        # doc1 should rank highest (exact match for "brown fox")
        assert results[0][0] == "doc1"

    def test_add_documents_batch(self):
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        idx = accel.RustBM25Index()
        idx.add_documents([("d1", "hello world"), ("d2", "world peace")])
        results = idx.search("world", limit=5)
        assert len(results) == 2

    def test_score(self):
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        idx = accel.RustBM25Index()
        idx.add_document("doc1", "test document content")
        score = idx.score("test", "doc1")
        assert score > 0.0
        # Non-existent doc returns 0
        assert idx.score("test", "nonexistent") == 0.0

    def test_none_when_python_forced(self, force_python):
        # When _HAS_RUST is forced False, verify the flag is off
        assert not accel._HAS_RUST


# ---------------------------------------------------------------------------
# Weighted RRF Normalized
# ---------------------------------------------------------------------------


class TestWeightedRRFNormalized:
    def test_basic(self):
        vec = [("a", 0.9), ("b", 0.7)]
        graph = [("b", 5.0), ("c", 3.0)]
        results = accel.weighted_rrf_normalized(vec, graph, k=60, vector_weight=0.6, graph_weight=0.4)
        ids = [r[0] for r in results]
        # "b" appears in both, should rank highly
        assert "b" in ids
        assert len(results) == 3  # a, b, c

    def test_empty_inputs(self):
        results = accel.weighted_rrf_normalized([], [], k=60)
        assert results == []

    def test_vector_only(self):
        results = accel.weighted_rrf_normalized([("x", 1.0), ("y", 0.5)], [])
        assert len(results) == 2
        assert results[0][0] == "x"

    def test_graph_only(self):
        results = accel.weighted_rrf_normalized([], [("a", 3.0), ("b", 1.0)])
        assert len(results) == 2
        assert results[0][0] == "a"

    def test_python_fallback(self, force_python):
        vec = [("a", 0.9), ("b", 0.7)]
        graph = [("b", 5.0), ("c", 3.0)]
        results = accel.weighted_rrf_normalized(vec, graph)
        ids = [r[0] for r in results]
        assert "b" in ids
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Entity name normalization
# ---------------------------------------------------------------------------


class TestNormalizeEntityName:
    def test_lowercase(self):
        assert accel.normalize_entity_name("Alice") == "alice"

    def test_strip_honorific(self):
        assert accel.normalize_entity_name("Dr. John Smith") == "john smith"
        assert accel.normalize_entity_name("Mr. Bob") == "bob"

    def test_collapse_whitespace(self):
        assert accel.normalize_entity_name("John   Smith") == "john smith"

    def test_strip_punctuation(self):
        assert accel.normalize_entity_name('"Hello World"') == "hello world"

    def test_empty(self):
        assert accel.normalize_entity_name("") == ""

    def test_python_fallback(self, force_python):
        assert accel.normalize_entity_name("Dr. Alice") == "alice"
        assert accel.normalize_entity_name("  John  Smith  ") == "john smith"


class TestNormalizeEntityNamesBatch:
    def test_batch(self):
        result = accel.normalize_entity_names_batch(["Alice", "Mr. Bob", "Dr. Charlie"])
        assert result == ["alice", "bob", "charlie"]

    def test_empty(self):
        assert accel.normalize_entity_names_batch([]) == []

    def test_python_fallback(self, force_python):
        result = accel.normalize_entity_names_batch(["Alice", "BOB"])
        assert result == ["alice", "bob"]


# ---------------------------------------------------------------------------
# Fusion diagnostics
# ---------------------------------------------------------------------------


class TestWeightedRrfNormalizedWithDiagnostics:
    def test_basic(self):
        vector = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        graph = [("b", 0.8), ("d", 0.6)]
        results = accel.weighted_rrf_normalized_with_diagnostics(vector, graph)
        # Should return tuples with 9 elements each
        assert len(results) == 4  # a, b, c, d
        assert all(len(r) == 9 for r in results)
        # Results should be sorted descending by score
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
        # "b" should be source=3 (both)
        b_result = next(r for r in results if r[0] == "b")
        assert b_result[2] == 3  # source bitmap = both
        assert b_result[3] > 0  # vector_rank
        assert b_result[4] > 0  # graph_rank

    def test_empty_inputs(self):
        results = accel.weighted_rrf_normalized_with_diagnostics([], [])
        assert results == []

    def test_vector_only(self):
        vector = [("a", 0.9)]
        results = accel.weighted_rrf_normalized_with_diagnostics(vector, [])
        assert len(results) == 1
        assert results[0][0] == "a"
        assert results[0][2] == 1  # vector only

    def test_python_fallback(self, force_python):
        vector = [("a", 0.9), ("b", 0.7)]
        graph = [("b", 0.8), ("c", 0.6)]
        results = accel.weighted_rrf_normalized_with_diagnostics(vector, graph)
        assert len(results) == 3
        b_result = next(r for r in results if r[0] == "b")
        assert b_result[2] == 3  # both sources


class TestBatchScoreStats:
    def test_basic(self):
        mean, std, mn, mx, med = accel.batch_score_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(mean - 3.0) < 1e-10
        assert abs(mn - 1.0) < 1e-10
        assert abs(mx - 5.0) < 1e-10
        assert abs(med - 3.0) < 1e-10
        assert std > 0

    def test_empty(self):
        result = accel.batch_score_stats([])
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0)

    def test_single(self):
        mean, std, mn, mx, med = accel.batch_score_stats([42.0])
        assert abs(mean - 42.0) < 1e-10
        assert abs(std - 0.0) < 1e-10
        assert abs(med - 42.0) < 1e-10

    def test_python_fallback(self, force_python):
        mean, std, mn, mx, med = accel.batch_score_stats([1.0, 3.0])
        assert abs(mean - 2.0) < 1e-10
        assert abs(med - 2.0) < 1e-10


class TestScoreEntropy:
    def test_uniform(self):
        import math

        entropy = accel.score_entropy([1.0, 1.0, 1.0, 1.0])
        assert abs(entropy - math.log(4)) < 1e-10

    def test_peaked(self):
        uniform = accel.score_entropy([1.0, 1.0, 1.0, 1.0])
        peaked = accel.score_entropy([100.0, 1.0, 1.0, 1.0])
        assert peaked < uniform

    def test_empty(self):
        assert accel.score_entropy([]) == 0.0

    def test_all_zeros(self):
        assert accel.score_entropy([0.0, 0.0, 0.0]) == 0.0

    def test_python_fallback(self, force_python):
        import math

        entropy = accel.score_entropy([1.0, 1.0])
        assert abs(entropy - math.log(2)) < 1e-10


class TestDetectTemporalCategoryWithConfidence:
    def test_no_temporal(self):
        cat, conf, terms = accel.detect_temporal_category_with_confidence("What is the capital of France?")
        assert cat == 0
        assert conf == 0.0
        assert terms == []

    def test_single_match(self):
        cat, conf, terms = accel.detect_temporal_category_with_confidence("What happened yesterday?")
        assert cat == 1
        assert conf == 0.6
        assert len(terms) == 1

    def test_multi_match(self):
        cat, conf, terms = accel.detect_temporal_category_with_confidence(
            "When did she switch to piano after the concert?"
        )
        assert cat >= 1
        assert conf >= 0.8
        assert len(terms) >= 2

    def test_python_fallback(self, force_python):
        cat, conf, terms = accel.detect_temporal_category_with_confidence("What happened yesterday?")
        assert cat == 1
        assert conf == 0.6
        assert len(terms) >= 1


# ---------------------------------------------------------------------------
# Community detection — determinism (#1131)
# ---------------------------------------------------------------------------

# Barbell-with-bridge: two triangles {0,1,2} and {3,4,5} joined by a bridge
# node 6 connected with EQUAL weight to one node of each triangle. Node 6's
# modularity gain ties between the two communities, so before #1131 the winner
# depended on hashbrown's per-process-randomly-seeded HashMap iteration order.
_TIE_EDGES = [
    (0, 1, 1.0),
    (1, 0, 1.0),
    (1, 2, 1.0),
    (2, 1, 1.0),
    (0, 2, 1.0),
    (2, 0, 1.0),
    (3, 4, 1.0),
    (4, 3, 1.0),
    (4, 5, 1.0),
    (5, 4, 1.0),
    (3, 5, 1.0),
    (5, 3, 1.0),
    (6, 2, 1.0),
    (2, 6, 1.0),
    (6, 3, 1.0),
    (3, 6, 1.0),
]

_SUBPROCESS_RUNNER = """
import khora._accel as a
edges = {edges!r}
print(",".join(map(str, a.detect_communities(7, edges, 1.0, 10))))
"""


class TestDetectCommunitiesDeterminism:
    def test_repeated_runs_identical_same_process(self):
        """Same input → identical assignment across N in-process calls."""
        first = accel.detect_communities(7, _TIE_EDGES, 1.0, 10)
        for _ in range(20):
            assert accel.detect_communities(7, _TIE_EDGES, 1.0, 10) == first

    def test_repeated_runs_identical_across_processes(self):
        """Same input → identical assignment across fresh processes.

        This is the regression guard for #1131: hashbrown 0.16 seeds its
        default hasher randomly per process, so the tie-break winner flipped
        between process launches before the deterministic smallest-id rule.
        In-process repetition cannot catch it because the seed is fixed once
        per process.
        """
        import subprocess
        import sys

        script = _SUBPROCESS_RUNNER.format(edges=_TIE_EDGES)
        outputs = set()
        for _ in range(16):
            proc = subprocess.run(  # noqa: S603 - test harness, sys.executable is trusted
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.add(proc.stdout.strip())
        assert len(outputs) == 1, f"non-deterministic across processes: {outputs}"

    def test_rust_matches_python_fallback(self, monkeypatch):
        """Rust and the pure-Python fallback agree on the tie graph."""
        rust = accel.detect_communities(7, _TIE_EDGES, 1.0, 10)
        monkeypatch.setattr(accel, "_HAS_RUST", False)
        py = accel.detect_communities(7, _TIE_EDGES, 1.0, 10)
        assert rust == py


# ---------------------------------------------------------------------------
# Entity name normalization — Rust/Python parity (#1133)
# ---------------------------------------------------------------------------

# Inputs spanning markdown decoration, interleaved punctuation/space runs,
# the limited strip-set boundary, and unicode. Before #1133 the Rust path
# trimmed ALL ASCII punctuation while Python stripped only a 16-char set.
_PARITY_INPUTS = [
    "**Acme Corp**",
    "Acme Corp . .",
    "...hello...",
    "  spaces  between  ",
    "Dr. Smith",
    "Mr. Dr. Smith",
    "**Acme**",
    "U.S.A.",
    "foo@bar",
    "a/b",
    "#tag",
    "100%",
    '"Hello World"',
    "(parenthetical)",
    "café—münchen",
    "naïve, résumé;",
    "",
    "   ",
]


class TestNormalizeEntityNameParity:
    @pytest.mark.parametrize("name", _PARITY_INPUTS)
    def test_rust_equals_python(self, name, monkeypatch):
        rust = accel.normalize_entity_name(name)
        monkeypatch.setattr(accel, "_HAS_RUST", False)
        py = accel.normalize_entity_name(name)
        assert rust == py, f"divergence for {name!r}: rust={rust!r} python={py!r}"

    def test_markdown_decoration_preserved(self):
        """The safer less-aggressive behavior keeps '**' decoration."""
        assert accel.normalize_entity_name("**Acme**") == "**acme**"

    def test_batch_parity(self, monkeypatch):
        rust = accel.normalize_entity_names_batch(list(_PARITY_INPUTS))
        monkeypatch.setattr(accel, "_HAS_RUST", False)
        py = accel.normalize_entity_names_batch(list(_PARITY_INPUTS))
        assert rust == py


# ---------------------------------------------------------------------------
# Batch recency scores - Rust/Python future-timestamp parity (#1130)
# ---------------------------------------------------------------------------


class TestBatchRecencyFutureClamp:
    def test_future_timestamp_clamped_rust(self):
        """A forward-dated timestamp must not produce a score above the
        present-time score on the Rust path. Without the age clamp, a future
        ts gives age_days < 0 and decay = 0.5^(negative) > 1.0, inflating the
        recency multiplier without bound (#1130)."""
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        now = 365 * 86400.0
        future = now + 365 * 86400.0  # one year ahead
        out = accel.batch_recency_scores([now, future], now_secs=now, decay_days=30.0, recency_weight=0.5)
        present_score = out[0]
        future_score = out[1]
        # Present-time ts has age 0 -> decay 1.0 -> score == base + weight == 1.0.
        assert present_score == pytest.approx(1.0)
        # Future ts must be clamped to age 0, not inflated above the present.
        assert future_score == pytest.approx(present_score)

    def test_future_timestamp_rust_matches_python_fallback(self, monkeypatch):
        """Rust and the pure-Python fallback must return identical scores for
        a future timestamp (fallback-parity contract)."""
        if not accel._HAS_RUST:
            pytest.skip("Rust extension not available")
        now = 100 * 86400.0
        timestamps = [now - 5 * 86400.0, now, now + 10 * 86400.0]
        rust_out = accel.batch_recency_scores(timestamps, now_secs=now, decay_days=7.0, recency_weight=0.6)
        monkeypatch.setattr(accel, "_HAS_RUST", False)
        monkeypatch.setattr(accel, "_HAS_NUMPY", False)
        py_out = accel.batch_recency_scores(timestamps, now_secs=now, decay_days=7.0, recency_weight=0.6)
        assert rust_out == pytest.approx(py_out)

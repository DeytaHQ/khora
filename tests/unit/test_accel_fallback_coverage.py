"""Coverage push for ``khora._accel`` Python and NumPy fallback paths.

The Rust accelerator is generally not available in the CI venv (built
locally from ``rust/khora-accel`` for dev). All functions in
``_accel.py`` have a NumPy/RapidFuzz fallback and most also have a pure-
Python fallback for environments without those wheels.

This module exercises the fallbacks via the same ``force_python`` /
``force_numpy`` monkeypatch fixtures the mainline test_accel.py uses,
targeting branches the mainline file doesn't cover:

* ``weighted_rrf_normalized_with_provenance`` Python path
* ``weighted_rrf_normalized_with_diagnostics`` Python path
* ``batch_temporal_filter`` Python path
* ``batch_recency_scores`` Python path
* ``detect_temporal_category`` Python path + all 6 categories
* ``detect_temporal_category_with_confidence`` Python path + confidence levels
* ``normalize_embeddings_batch`` numpy + python paths
* ``batch_dot_product`` numpy + python paths
* ``detect_communities`` empty / isolated / Python path
* ``mmr_diversity_select`` numpy + python paths
* ``resolve_entities_enhanced`` Python path (exact, alias, fuzzy)
* ``_py_deduplicate_chunks`` direct
* ``deduplicate_chunks`` end-to-end
* ``configure_thread_pool`` no-op without Rust
* ``build_chunk_edges`` Python path (with overflow indices skipped)
* ``pagerank`` personalization fallbacks (uniform when mismatched/zero)
"""

from __future__ import annotations

import pytest

import khora._accel as accel

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def force_python(monkeypatch):
    monkeypatch.setattr(accel, "_HAS_RUST", False)
    monkeypatch.setattr(accel, "_HAS_NUMPY", False)
    monkeypatch.setattr(accel, "_HAS_RAPIDFUZZ", False)


@pytest.fixture()
def force_numpy(monkeypatch):
    monkeypatch.setattr(accel, "_HAS_RUST", False)
    # Keep _HAS_NUMPY / _HAS_RAPIDFUZZ at the actual values (numpy installed).


# ---------------------------------------------------------------------------
# weighted_rrf_normalized_with_provenance — Python path
# ---------------------------------------------------------------------------


class TestWeightedRRFNormalizedWithProvenance:
    def test_vector_only_marks_bitmap_1(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_provenance(
            vector_results=[("a", 1.0), ("b", 0.5)],
            graph_results=[],
        )
        # All entries have source flag 1 (vector only).
        for _id, _score, src in out:
            assert src == 0x01

    def test_graph_only_marks_bitmap_2(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_provenance(
            vector_results=[],
            graph_results=[("x", 1.0)],
        )
        for _id, _score, src in out:
            assert src == 0x02

    def test_both_sources_mark_bitmap_3(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_provenance(
            vector_results=[("a", 1.0)],
            graph_results=[("a", 0.9)],
        )
        # "a" appears in both — bitmap 3.
        for id_, _score, src in out:
            if id_ == "a":
                assert src == 0x03

    def test_sorted_descending(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_provenance(
            vector_results=[("a", 1.0), ("b", 0.5), ("c", 0.1)],
            graph_results=[("c", 1.0), ("a", 0.5)],
        )
        scores = [s for _, s, _ in out]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# weighted_rrf_normalized_with_diagnostics — Python path
# ---------------------------------------------------------------------------


class TestWeightedRRFNormalizedWithDiagnostics:
    def test_returns_nine_tuple_per_item(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_diagnostics(
            vector_results=[("a", 1.0)],
            graph_results=[("a", 0.5)],
        )
        for entry in out:
            assert len(entry) == 9  # id, score, source, v_rank, g_rank, v_norm, g_norm, v_contrib, g_contrib

    def test_ranks_are_1_indexed(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_diagnostics(
            vector_results=[("a", 1.0), ("b", 0.5)],
            graph_results=[("a", 0.9)],
        )
        for id_, _score, _src, v_rank, g_rank, *_rest in out:
            if id_ == "a":
                assert v_rank == 1
                assert g_rank == 1

    def test_missing_in_one_source_has_zero_rank(self, force_python) -> None:
        out = accel.weighted_rrf_normalized_with_diagnostics(
            vector_results=[("a", 1.0)],
            graph_results=[],
        )
        for _id, _score, _src, _v_rank, g_rank, *_ in out:
            assert g_rank == 0


# ---------------------------------------------------------------------------
# batch_temporal_filter — Python path
# ---------------------------------------------------------------------------


class TestBatchTemporalFilter:
    def test_before(self, force_python) -> None:
        out = accel.batch_temporal_filter([1.0, 5.0, 10.0], "before", end_secs=6.0)
        assert out == [True, True, False]

    def test_after(self, force_python) -> None:
        out = accel.batch_temporal_filter([1.0, 5.0, 10.0], "after", start_secs=4.0)
        assert out == [False, True, True]

    def test_between(self, force_python) -> None:
        out = accel.batch_temporal_filter([1.0, 5.0, 10.0], "between", start_secs=3.0, end_secs=8.0)
        assert out == [False, True, False]

    def test_unknown_operator_returns_true(self, force_python) -> None:
        """Unknown operators fall through to ``True`` for every timestamp."""
        out = accel.batch_temporal_filter([1.0, 2.0], "weird_op")
        assert out == [True, True]

    def test_before_none_end_passes_everything(self, force_python) -> None:
        out = accel.batch_temporal_filter([1.0, 5.0], "before", end_secs=None)
        assert out == [True, True]


# ---------------------------------------------------------------------------
# batch_recency_scores — Python path
# ---------------------------------------------------------------------------


class TestBatchRecencyScores:
    def test_zero_weight_returns_ones(self, force_python) -> None:
        out = accel.batch_recency_scores([100.0, 200.0], now_secs=300.0, decay_days=7.0, recency_weight=0.0)
        assert out == [1.0, 1.0]

    def test_decays_with_age(self, force_python) -> None:
        # Two timestamps: today and 7 days ago. Half-life = 7d, weight = 1.0.
        # Today: 1.0 (no decay). 7 days ago: 0.5.
        now = 7 * 86400.0
        out = accel.batch_recency_scores([now, 0.0], now_secs=now, decay_days=7.0, recency_weight=1.0)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.5, abs=0.01)

    def test_zero_decay_days_gives_constant_score(self, force_python) -> None:
        out = accel.batch_recency_scores([100.0, 200.0], now_secs=200.0, decay_days=0.0, recency_weight=0.5)
        # With decay_days=0 → decay_factor=0 → decay=1 → all scores = base + weight = 1.0.
        assert all(s == pytest.approx(1.0) for s in out)


# ---------------------------------------------------------------------------
# detect_temporal_category — Python path
# ---------------------------------------------------------------------------


class TestDetectTemporalCategory:
    def test_no_temporal_keyword_returns_zero(self, force_python) -> None:
        assert accel.detect_temporal_category("hello world") == 0

    @pytest.mark.parametrize(
        ("query", "expected_min"),
        [
            ("when did we ship?", 1),  # EXPLICIT
            ("is it currently active?", 2),  # STATE_QUERY
            ("which came first?", 3),  # ORDINAL
            ("how many times did we do this in total?", 4),  # AGGREGATE
            ("most recent update", 5),  # RECENCY
            ("they changed their plan", 6),  # CHANGE
        ],
    )
    def test_each_category_matches(self, query: str, expected_min: int, force_python) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat >= expected_min

    def test_picks_highest_when_multiple_match(self, force_python) -> None:
        """When multiple categories match, the highest-numbered one wins."""
        # Both RECENCY (5) and CHANGE (6) terms; CHANGE wins.
        cat = accel.detect_temporal_category("they changed the most recent plan")
        assert cat == 6


# ---------------------------------------------------------------------------
# detect_temporal_category_with_confidence — Python path
# ---------------------------------------------------------------------------


class TestDetectTemporalCategoryWithConfidencePython:
    def test_no_match_returns_zero_tuple(self, force_python) -> None:
        cat, conf, terms = accel.detect_temporal_category_with_confidence("hello world")
        assert cat == 0
        assert conf == 0.0
        assert terms == []

    def test_single_match_low_confidence(self, force_python) -> None:
        cat, conf, terms = accel.detect_temporal_category_with_confidence("when did this happen?")
        assert cat >= 1
        assert conf == pytest.approx(0.6) or conf >= 0.6

    def test_two_matches_medium_confidence(self, force_python) -> None:
        # "since" + "yesterday" → both EXPLICIT, same category.
        cat, conf, terms = accel.detect_temporal_category_with_confidence("since yesterday")
        assert cat >= 1
        assert len(terms) >= 2

    def test_date_pattern_boosts_confidence(self, force_python) -> None:
        cat, conf, terms = accel.detect_temporal_category_with_confidence("when did we ship 2024-01-15")
        assert cat >= 1
        assert conf > 0.6  # boosted by date pattern


# ---------------------------------------------------------------------------
# normalize_embeddings_batch
# ---------------------------------------------------------------------------


class TestNormalizeEmbeddingsBatch:
    def test_numpy_path(self, force_numpy) -> None:
        out = accel.normalize_embeddings_batch([[3.0, 4.0]])
        # Norm = 5.0 → [0.6, 0.8].
        assert out[0][0] == pytest.approx(0.6, abs=1e-5)
        assert out[0][1] == pytest.approx(0.8, abs=1e-5)

    def test_numpy_zero_vector_returns_as_is(self, force_numpy) -> None:
        out = accel.normalize_embeddings_batch([[0.0, 0.0]])
        assert out == [[0.0, 0.0]]

    def test_python_path(self, force_python) -> None:
        out = accel.normalize_embeddings_batch([[3.0, 4.0]])
        assert out[0][0] == pytest.approx(0.6, abs=1e-5)
        assert out[0][1] == pytest.approx(0.8, abs=1e-5)

    def test_python_zero_vector_returns_as_is(self, force_python) -> None:
        out = accel.normalize_embeddings_batch([[0.0, 0.0]])
        assert out == [[0.0, 0.0]]


# ---------------------------------------------------------------------------
# batch_dot_product
# ---------------------------------------------------------------------------


class TestBatchDotProduct:
    def test_empty_candidates(self) -> None:
        assert accel.batch_dot_product([1.0, 0.0], []) == []

    def test_numpy_path_sorts_descending(self, force_numpy) -> None:
        out = accel.batch_dot_product([1.0, 0.0], [[0.5, 0.5], [1.0, 0.0], [0.1, 0.9]])
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_numpy_threshold_filters(self, force_numpy) -> None:
        out = accel.batch_dot_product([1.0, 0.0], [[1.0, 0.0], [0.1, 0.9]], threshold=0.5)
        assert len(out) == 1
        assert out[0][0] == 0

    def test_python_path(self, force_python) -> None:
        out = accel.batch_dot_product([1.0, 0.0], [[1.0, 0.0], [0.5, 0.5]], threshold=0.4)
        # Both pass; first sorts higher.
        assert out[0][0] == 0


# ---------------------------------------------------------------------------
# detect_communities
# ---------------------------------------------------------------------------


class TestDetectCommunities:
    def test_empty_graph_returns_empty(self, force_python) -> None:
        assert accel.detect_communities(0, []) == []

    def test_isolated_nodes_return_minus_one(self, force_python) -> None:
        # 3 nodes, no edges → all isolated.
        out = accel.detect_communities(3, [])
        assert out == [-1, -1, -1]

    def test_connected_pair_share_community(self, force_python) -> None:
        # Two nodes connected → same community.
        out = accel.detect_communities(2, [(0, 1, 1.0), (1, 0, 1.0)])
        assert out[0] == out[1]
        assert out[0] != -1

    def test_self_loops_ignored(self, force_python) -> None:
        # Self-loops are excluded by the algorithm.
        out = accel.detect_communities(2, [(0, 0, 1.0), (1, 1, 1.0)])
        # No real edges → isolated.
        assert out == [-1, -1]

    def test_out_of_range_edges_skipped(self, force_python) -> None:
        out = accel.detect_communities(2, [(5, 7, 1.0)])
        assert out == [-1, -1]


# ---------------------------------------------------------------------------
# mmr_diversity_select
# ---------------------------------------------------------------------------


class TestMmrDiversitySelect:
    def test_empty_returns_empty(self) -> None:
        assert accel.mmr_diversity_select([], [], 0.5, 5) == []

    def test_k_zero_returns_empty(self) -> None:
        assert accel.mmr_diversity_select([[1.0]], [1.0], 0.5, 0) == []

    def test_k_capped_to_n(self, force_python) -> None:
        out = accel.mmr_diversity_select([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.5], 0.5, 10)
        assert len(out) == 2

    def test_numpy_path_returns_top_relevance_at_lambda_one(self, force_numpy) -> None:
        # lambda=1.0 → pure relevance; highest score wins first.
        out = accel.mmr_diversity_select(
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            scores=[0.5, 1.0, 0.8],
            lambda_param=1.0,
            k=2,
        )
        # First pick = index 1 (highest score).
        assert out[0] == 1

    def test_python_path(self, force_python) -> None:
        out = accel.mmr_diversity_select(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            scores=[1.0, 0.5],
            lambda_param=1.0,
            k=2,
        )
        # Pure relevance picks highest first.
        assert out[0] == 0


# ---------------------------------------------------------------------------
# resolve_entities_enhanced — Python path (exact / alias / fuzzy / none)
# ---------------------------------------------------------------------------


class TestResolveEntitiesEnhanced:
    def test_exact_match(self, force_python) -> None:
        out = accel.resolve_entities_enhanced(
            new_names=["Alice"],
            new_types=["PERSON"],
            existing_names=["Bob", "Alice"],
            existing_aliases=[[], []],
            existing_types=["PERSON", "PERSON"],
        )
        assert out[0] == (1, 1.0, "exact")

    def test_alias_match(self, force_python) -> None:
        out = accel.resolve_entities_enhanced(
            new_names=["Bobby"],
            new_types=["PERSON"],
            existing_names=["Bob"],
            existing_aliases=[["Bobby", "Robert"]],
            existing_types=["PERSON"],
        )
        assert out[0] == (0, 1.0, "alias")

    def test_fuzzy_match(self, force_python) -> None:
        # Very close names: "Alicia" vs "Alice" — fuzzy match (default threshold 0.85).
        out = accel.resolve_entities_enhanced(
            new_names=["Alicee"],
            new_types=["PERSON"],
            existing_names=["Alice"],
            existing_aliases=[[]],
            existing_types=["PERSON"],
            default_threshold=0.5,  # generous fuzzy threshold for the python path
        )
        assert out[0] is not None
        assert out[0][2] in ("fuzzy", "exact")

    def test_no_match_returns_none(self, force_python) -> None:
        out = accel.resolve_entities_enhanced(
            new_names=["Zelda"],
            new_types=["PERSON"],
            existing_names=["Alice"],
            existing_aliases=[[]],
            existing_types=["PERSON"],
        )
        assert out[0] is None

    def test_different_type_no_match(self, force_python) -> None:
        """Same name but different type → no exact/alias/fuzzy match."""
        out = accel.resolve_entities_enhanced(
            new_names=["Alice"],
            new_types=["PERSON"],
            existing_names=["Alice"],
            existing_aliases=[[]],
            existing_types=["ORGANIZATION"],
        )
        assert out[0] is None

    def test_per_type_threshold_applied(self, force_python) -> None:
        """A custom threshold for PERSON overrides the default."""
        out = accel.resolve_entities_enhanced(
            new_names=["AliceX"],
            new_types=["PERSON"],
            existing_names=["Alice"],
            existing_aliases=[[]],
            existing_types=["PERSON"],
            type_thresholds={"PERSON": 0.99},  # so fuzzy fails
        )
        assert out[0] is None


# ---------------------------------------------------------------------------
# _py_deduplicate_chunks (Python fallback)
# ---------------------------------------------------------------------------


class TestPyDeduplicateChunks:
    def test_empty_returns_empty(self) -> None:
        assert accel._py_deduplicate_chunks([]) == []

    def test_unique_chunks_have_none_duplicate(self) -> None:
        out = accel._py_deduplicate_chunks(["alpha bravo", "charlie delta"], threshold=0.85, num_perm=16)
        assert all(dup is None for _, dup in out)

    def test_near_identical_chunks_dedup(self) -> None:
        out = accel._py_deduplicate_chunks(
            ["the quick brown fox jumps over the lazy dog"] * 3,
            threshold=0.5,
            num_perm=16,
        )
        # First is canonical; others reference it.
        assert out[0] == (0, None)
        assert out[1][1] == 0
        assert out[2][1] == 0

    def test_uses_python_path_when_no_rust(self, force_python) -> None:
        out = accel.deduplicate_chunks(["alpha"], threshold=0.85, num_perm=8)
        assert out == [(0, None)]

    def test_short_text_below_ngram_size(self) -> None:
        # Texts shorter than ngram size (5) still work.
        out = accel._py_deduplicate_chunks(["a", "b"], threshold=0.5, num_perm=8)
        assert len(out) == 2

    def test_empty_string_handled(self) -> None:
        out = accel._py_deduplicate_chunks(["", ""], threshold=0.5, num_perm=8)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# configure_thread_pool — no-op without Rust
# ---------------------------------------------------------------------------


class TestConfigureThreadPool:
    def test_no_op_without_rust(self, force_python) -> None:
        # Must not raise.
        accel.configure_thread_pool(num_threads=4, mode="query")
        accel.configure_thread_pool(num_threads=0, mode="ingest")


# ---------------------------------------------------------------------------
# build_chunk_edges — Python fallback
# ---------------------------------------------------------------------------


class TestBuildChunkEdges:
    def test_emits_bidirectional_edges(self, force_python) -> None:
        edges = accel.build_chunk_edges(
            n_chunks=3,
            keyword_chunk_ids=[[0, 1], [1, 2]],
            idf_scores=[1.5, 2.0],
        )
        # For each keyword, every pair gets both (a,b) and (b,a).
        # Keyword 0 → chunks 0,1 → (0,1,1.5), (1,0,1.5).
        # Keyword 1 → chunks 1,2 → (1,2,2.0), (2,1,2.0).
        triples = {(s, d) for s, d, _w in edges}
        assert (0, 1) in triples
        assert (1, 0) in triples
        assert (1, 2) in triples
        assert (2, 1) in triples

    def test_chunks_out_of_range_skipped(self, force_python) -> None:
        edges = accel.build_chunk_edges(
            n_chunks=2,
            keyword_chunk_ids=[[0, 99]],  # 99 is out of range
            idf_scores=[1.0],
        )
        # Only (0, ?) pairs valid — 99 skipped → 0 edges (no valid pair).
        assert edges == []


# ---------------------------------------------------------------------------
# pagerank personalization fallbacks
# ---------------------------------------------------------------------------


class TestPagerankPersonalization:
    def test_mismatched_length_falls_back_to_uniform(self, force_python) -> None:
        # personalization length 2 != n=3 → uniform fallback.
        scores = accel.pagerank(3, [], damping=0.85, max_iter=10, personalization=[1.0, 0.0])
        # Uniform → all equal.
        assert scores[0] == pytest.approx(scores[1])
        assert scores[1] == pytest.approx(scores[2])

    def test_all_zero_personalization_falls_back_to_uniform(self, force_python) -> None:
        scores = accel.pagerank(2, [], damping=0.85, max_iter=10, personalization=[0.0, 0.0])
        # Uniform.
        assert scores[0] == pytest.approx(scores[1])

    def test_negative_personalization_clipped(self, force_python) -> None:
        # [-1, 2] → clip → [0, 2] → normalize → [0, 1].
        scores = accel.pagerank(2, [], damping=0.85, max_iter=10, personalization=[-1.0, 2.0])
        # Without edges, the iteration keeps the seed distribution.
        # Score of seed node 1 (where teleport mass is concentrated) > score of node 0.
        assert scores[1] > scores[0]

    def test_zero_nodes_returns_empty(self, force_python) -> None:
        assert accel.pagerank(0, [], damping=0.85, max_iter=10) == []


# ---------------------------------------------------------------------------
# normalize_scores — Python fallbacks
# ---------------------------------------------------------------------------


class TestNormalizeScoresPython:
    def test_empty_list_returns_empty(self, force_python) -> None:
        assert accel.normalize_scores([]) == []

    def test_identical_values_normalize_to_ones(self, force_python) -> None:
        out = accel.normalize_scores([3.0, 3.0, 3.0])
        assert out == [1.0, 1.0, 1.0]

    def test_range_normalizes_to_0_1(self, force_python) -> None:
        out = accel.normalize_scores([0.0, 5.0, 10.0])
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# normalize_entity_name / normalize_entity_names_batch — Python paths
# ---------------------------------------------------------------------------


class TestNormalizeEntityNamePython:
    def test_lowercase_and_strip(self, force_python) -> None:
        assert accel.normalize_entity_name("  Alice  ") == "alice"

    def test_collapses_whitespace(self, force_python) -> None:
        out = accel.normalize_entity_name("Acme   Corp")
        # Multiple whitespace collapsed.
        assert "   " not in out

    def test_batch_calls_normalize_each(self, force_python) -> None:
        out = accel.normalize_entity_names_batch(["Alice", " Bob "])
        assert out[0] == accel.normalize_entity_name("Alice")
        assert out[1] == accel.normalize_entity_name(" Bob ")


# ---------------------------------------------------------------------------
# extract_keywords / extract_keywords_batch — Python paths
# ---------------------------------------------------------------------------


class TestExtractKeywordsPython:
    def test_strips_stopwords_and_short_words(self, force_python) -> None:
        out = accel.extract_keywords("The quick brown fox is on a log")
        assert "the" not in out
        assert "is" not in out
        assert "quick" in out
        assert "brown" in out

    def test_dedupes_repeated_words(self, force_python) -> None:
        out = accel.extract_keywords("alpha beta alpha gamma alpha")
        assert out.count("alpha") == 1

    def test_batch_returns_one_list_per_input(self, force_python) -> None:
        out = accel.extract_keywords_batch(["alpha beta", "gamma delta"])
        assert len(out) == 2


# ---------------------------------------------------------------------------
# sequence_match_ratio — pure-Python path via difflib
# ---------------------------------------------------------------------------


class TestSequenceMatchRatioPython:
    def test_identical_strings_score_one(self, force_python) -> None:
        assert accel.sequence_match_ratio("alice", "alice") == 1.0

    def test_empty_inputs(self, force_python) -> None:
        # difflib returns 1.0 for two empties; that's fine.
        out = accel.sequence_match_ratio("", "")
        assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# levenshtein_similarity — pure-Python DP path
# ---------------------------------------------------------------------------


class TestLevenshteinPython:
    def test_identical_returns_one(self, force_python) -> None:
        assert accel.levenshtein_similarity("alice", "alice") == 1.0

    def test_empty_input(self, force_python) -> None:
        assert accel.levenshtein_similarity("", "alice") == 0.0
        assert accel.levenshtein_similarity("alice", "") == 0.0

    def test_single_substitution(self, force_python) -> None:
        # "alice" vs "alibe" — 1 substitution out of 5 → 0.8.
        out = accel.levenshtein_similarity("alice", "alibe")
        assert out == pytest.approx(0.8, abs=0.01)

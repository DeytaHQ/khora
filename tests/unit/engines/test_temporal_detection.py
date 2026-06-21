"""Unit tests for temporal detection and relative recency."""

from __future__ import annotations

from uuid import uuid4

import pytest

from khora.engines.vectorcypher.temporal_detection import (
    CATEGORY_MAP,
    RETRIEVAL_PARAMS,
    TemporalCategory,
    TemporalDetector,
    TemporalSignal,
    get_retrieval_params,
)

# ---------------------------------------------------------------------------
# TemporalCategory detection tests
# ---------------------------------------------------------------------------


class TestTemporalDetector:
    """Tests for the TemporalDetector cascade."""

    def setup_method(self) -> None:
        self.detector = TemporalDetector()

    def test_state_query_currently(self) -> None:
        signal = self.detector.detect("What instrument is the user currently playing?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.STATE_QUERY
        assert signal.source == "dictionary"
        assert signal.confidence == 0.9

    def test_state_query_right_now(self) -> None:
        signal = self.detector.detect("Where does she live right now?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.STATE_QUERY

    def test_ordinal_first(self) -> None:
        signal = self.detector.detect("Which event happened first ?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.ORDINAL

    def test_explicit_before_date(self) -> None:
        signal = self.detector.detect("What happened before April 2024?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.EXPLICIT

    def test_aggregate_total(self) -> None:
        signal = self.detector.detect("How many times did she visit?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.AGGREGATE

    def test_recency_most_recent(self) -> None:
        signal = self.detector.detect("What is the most recent update?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.RECENCY

    def test_still_is_state_query(self) -> None:
        # "Does she still" matches STATE_QUERY, not CHANGE (still was removed from CHANGE)
        signal = self.detector.detect("Does she still work at Google?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.STATE_QUERY

    def test_change_used_to(self) -> None:
        signal = self.detector.detect("She used to live in Paris")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.CHANGE

    def test_recency_latest(self) -> None:
        signal = self.detector.detect("What is the latest news?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.RECENCY

    def test_change_became(self) -> None:
        signal = self.detector.detect("She became a doctor")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.CHANGE

    def test_change_switched_to(self) -> None:
        signal = self.detector.detect("He switched to piano")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.CHANGE

    def test_aggregate_how_often(self) -> None:
        signal = self.detector.detect("How often does she visit?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.AGGREGATE

    def test_none_non_temporal(self) -> None:
        signal = self.detector.detect("What is the capital of France?")
        assert signal.is_temporal is False
        assert signal.category == TemporalCategory.NONE
        assert signal.source == "none"

    def test_none_programming_query(self) -> None:
        signal = self.detector.detect("How do I implement a binary search tree?")
        assert signal.is_temporal is False
        assert signal.category == TemporalCategory.NONE

    def test_explicit_with_temporal_filter(self) -> None:
        """EXPLICIT category should produce a TemporalFilter when dates are parseable."""
        signal = self.detector.detect("What happened before 2024-03-15?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.EXPLICIT
        assert signal.temporal_filter is not None
        assert signal.temporal_filter.occurred_before is not None

    def test_explicit_after_date(self) -> None:
        signal = self.detector.detect("Events after 2024-01-01?")
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.EXPLICIT
        assert signal.temporal_filter is not None
        assert signal.temporal_filter.occurred_after is not None

    def test_highest_category_wins(self) -> None:
        """When multiple categories match, the highest ID should win."""
        # "used to" (CHANGE=6) + "last year" (EXPLICIT=1) → CHANGE wins
        signal = self.detector.detect("She used to work there last year")
        assert signal.category == TemporalCategory.CHANGE


# ---------------------------------------------------------------------------
# Python fallback detection tests
# ---------------------------------------------------------------------------


class TestPythonFallbackDetection:
    """Test the Python fallback path for detect_temporal_category."""

    def test_python_fallback_state_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force Python backend and verify detection still works."""
        monkeypatch.setattr("khora._accel._HAS_RUST", False)
        from khora._accel import detect_temporal_category

        assert detect_temporal_category("What is she currently doing?") == 2  # STATE_QUERY

    def test_python_fallback_change(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("khora._accel._HAS_RUST", False)
        from khora._accel import detect_temporal_category

        assert detect_temporal_category("She used to work at Apple") == 6  # CHANGE

    def test_python_fallback_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("khora._accel._HAS_RUST", False)
        from khora._accel import detect_temporal_category

        assert detect_temporal_category("What is the capital of France?") == 0  # NONE


# ---------------------------------------------------------------------------
# Word-boundary matching (#981 / #1285)
# ---------------------------------------------------------------------------
#
# Keywords must only fire as whole words/phrases, not as substrings of larger
# words. Each case is checked on BOTH the Rust kernel (default) and the Python
# fallback so the two stay in lockstep parity.


# (query, expected_category_id, reason)
_WORD_BOUNDARY_CASES = [
    # #981 substring false-positives — now correctly NONE.
    ("The contract terms remain unchanged.", 0, "'changed' inside 'unchanged'"),
    ("The exchanged emails were archived.", 0, "'changed' inside 'exchanged'"),
    ("The team marched on with their plan.", 0, "'march' inside 'marched'"),
    # Whole-word matches must still fire (no over-correction).
    ("What happened in March 2024?", 1, "'March' as a whole word -> EXPLICIT"),
    ("The contract terms changed last week.", 6, "'changed' as a whole word -> CHANGE"),
    ("Does the wiki have the more up-to-date status?", 2, "hyphenated 'up-to-date' -> STATE_QUERY"),
    ("Who is the account manager?", 2, "'who is the ' phrase -> STATE_QUERY"),
    # #1285 decision (b): possessive "X's current ..." is a STATE_QUERY.
    ("What is Sarah's current quota attainment?", 2, 'possessive "\'s current " -> STATE_QUERY'),
    # No possessive / compound pattern: plain recency lookup stays NONE.
    ("What is the current pipeline value?", 0, "bare 'current' without a compound pattern"),
]


@pytest.mark.parametrize(("query", "expected", "reason"), _WORD_BOUNDARY_CASES)
def test_word_boundary_rust_path(query: str, expected: int, reason: str) -> None:
    from khora._accel import detect_temporal_category

    assert detect_temporal_category(query) == expected, reason


@pytest.mark.parametrize(("query", "expected", "reason"), _WORD_BOUNDARY_CASES)
def test_word_boundary_python_fallback(query: str, expected: int, reason: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("khora._accel._HAS_RUST", False)
    from khora._accel import detect_temporal_category

    assert detect_temporal_category(query) == expected, reason


# ---------------------------------------------------------------------------
# #981 Tier-1 disambiguation of whole-word ambiguous English keywords
# ---------------------------------------------------------------------------
#
# "may" (month vs. modal verb / proper name), "just" (recency vs. "only/simply"
# adverb), and "active" (state-query vs. generic adjective) are real whole words,
# so the #1313 word-boundary fix does not help and the #1318 Tier-2 LLM fallback
# (which only runs when Tier-1 returns NONE) cannot correct these false
# POSITIVES. They are disambiguated in Tier-1: they classify as temporal only in
# a temporal context. Checked on BOTH the Rust kernel and the Python fallback.

# (query, expected_category_id, reason)
_DISAMBIGUATION_CASES = [
    # "may" — ambiguous false-positives now NONE.
    ("Describe the May Department Stores company.", 0, "'May' as a company name"),
    ("Tell me about May Corp.", 0, "'May' as a proper name"),
    ("You may proceed.", 0, "'may' as a modal verb"),
    # "may" — genuine month, kept (preposition or number adjacency) -> EXPLICIT.
    ("What shipped in May?", 1, "'in May' -> month -> EXPLICIT"),
    ("Everything since May.", 1, "'since May' -> month -> EXPLICIT"),
    ("What did Acme ship May 2024?", 1, "'May 2024' -> month -> EXPLICIT"),
    ("The release on 5 May went out.", 1, "'5 May' -> month -> EXPLICIT"),
    # "just" — adverb false-positive now NONE; recency phrases kept -> RECENCY.
    ("Just confirm the data structure.", 0, "'just' as the adverb 'only/simply'"),
    ("What happened just now?", 5, "'just now' -> RECENCY"),
    ("What was just released?", 5, "'just released' -> RECENCY"),
    # "active" — generic adjective false-positive now NONE; noun phrases kept.
    ("Apply the formula to the active variable.", 0, "'active' (math), not temporal"),
    ("What are all the active deals in the pipeline?", 2, "'active deals' -> STATE_QUERY"),
    ("List the active projects for Q3.", 2, "'active projects' -> STATE_QUERY"),
    ("Is it currently active?", 2, "'currently' -> STATE_QUERY"),
]


@pytest.mark.parametrize(("query", "expected", "reason"), _DISAMBIGUATION_CASES)
def test_disambiguation_rust_path(query: str, expected: int, reason: str) -> None:
    from khora._accel import detect_temporal_category

    assert detect_temporal_category(query) == expected, reason


@pytest.mark.parametrize(("query", "expected", "reason"), _DISAMBIGUATION_CASES)
def test_disambiguation_python_fallback(
    query: str, expected: int, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("khora._accel._HAS_RUST", False)
    from khora._accel import detect_temporal_category

    assert detect_temporal_category(query) == expected, reason


# ---------------------------------------------------------------------------
# Retrieval params mapping tests
# ---------------------------------------------------------------------------


class TestRetrievalParams:
    """Test category → retrieval parameter mapping."""

    def test_none_defaults(self) -> None:
        params = RETRIEVAL_PARAMS[TemporalCategory.NONE]
        assert params.recency_weight == 0.0
        assert params.temporal_sort is False
        assert params.decay_days_override is None

    def test_state_query_params(self) -> None:
        params = RETRIEVAL_PARAMS[TemporalCategory.STATE_QUERY]
        assert params.recency_weight == 0.5
        assert params.temporal_sort is True
        assert params.recency_floor == 0.3

    def test_ordinal_params(self) -> None:
        params = RETRIEVAL_PARAMS[TemporalCategory.ORDINAL]
        assert params.recency_weight == 0.3
        assert params.temporal_sort is True
        assert params.decay_days_override is None

    def test_aggregate_params(self) -> None:
        params = RETRIEVAL_PARAMS[TemporalCategory.AGGREGATE]
        assert params.recency_weight == 0.0
        assert params.temporal_sort is False

    def test_recency_params(self) -> None:
        params = RETRIEVAL_PARAMS[TemporalCategory.RECENCY]
        assert params.recency_weight == 0.5
        assert params.temporal_sort is True
        assert params.decay_days_override == 3
        assert params.recency_floor == 0.3

    def test_change_params(self) -> None:
        params = RETRIEVAL_PARAMS[TemporalCategory.CHANGE]
        assert params.recency_weight == 0.4
        assert params.temporal_sort is True
        assert params.decay_days_override == 14

    def test_get_retrieval_params_helper(self) -> None:
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="dictionary",
        )
        params = get_retrieval_params(signal)
        assert params.recency_weight == 0.5
        assert params.decay_days_override == 3

    def test_all_categories_have_params(self) -> None:
        for cat in TemporalCategory:
            assert cat in RETRIEVAL_PARAMS


# ---------------------------------------------------------------------------
# Relative recency tests
# ---------------------------------------------------------------------------


class TestRelativeRecency:
    """Test the relative recency calculation."""

    def test_relative_recency_discriminates(self) -> None:
        """Two chunks from 2024-01 and 2024-05: should get different scores."""
        from khora.core.models import Chunk
        from khora.engines.vectorcypher.fusion import FusedResult
        from khora.engines.vectorcypher.retriever import RetrieverConfig

        config = RetrieverConfig(recency_decay_days=30)

        # Create mock fused results with occurred_at dates
        jan_id = uuid4()
        may_id = uuid4()
        ns_id = uuid4()

        chunk_jan = Chunk(
            id=jan_id,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="January event",
            metadata={"occurred_at": "2024-01-15T00:00:00+00:00"},
        )
        chunk_may = Chunk(
            id=may_id,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="May event",
            metadata={"occurred_at": "2024-05-15T00:00:00+00:00"},
        )

        fused = [
            FusedResult(item=chunk_jan, rrf_score=0.5, item_id=jan_id),
            FusedResult(item=chunk_may, rrf_score=0.5, item_id=may_id),
        ]

        # Use a mock retriever to call _calculate_recency_scores
        from unittest.mock import MagicMock

        retriever = MagicMock()
        retriever._config = config

        # Call the method directly
        from khora.engines.vectorcypher.retriever import VectorCypherRetriever

        scores = VectorCypherRetriever._calculate_recency_scores(retriever, fused)

        # May (newer) should have higher recency score than January (older)
        assert scores[may_id] > scores[jan_id]

        # May should be close to 1.0 (it's the reference point)
        assert scores[may_id] == pytest.approx(1.0, abs=0.01)

        # January should be significantly lower (4 months old relative to May)
        assert scores[jan_id] < 0.5

    def test_relative_recency_independent_of_execution_year(self) -> None:
        """Recency scores should be same regardless of when test runs."""
        from khora.core.models import Chunk
        from khora.engines.vectorcypher.fusion import FusedResult
        from khora.engines.vectorcypher.retriever import RetrieverConfig

        config = RetrieverConfig(recency_decay_days=30)

        ns_id = uuid4()
        id1, id2 = uuid4(), uuid4()

        chunk1 = Chunk(
            id=id1,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Old",
            metadata={"occurred_at": "2020-01-01T00:00:00+00:00"},
        )
        chunk2 = Chunk(
            id=id2,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="New",
            metadata={"occurred_at": "2020-05-01T00:00:00+00:00"},
        )

        fused = [
            FusedResult(item=chunk1, rrf_score=0.5, item_id=id1),
            FusedResult(item=chunk2, rrf_score=0.5, item_id=id2),
        ]

        from unittest.mock import MagicMock

        from khora.engines.vectorcypher.retriever import VectorCypherRetriever

        retriever = MagicMock()
        retriever._config = config
        scores = VectorCypherRetriever._calculate_recency_scores(retriever, fused)

        # Newest should be ~1.0 (reference time)
        assert scores[id2] == pytest.approx(1.0, abs=0.01)
        # Oldest is 4 months older, should be well below 1.0
        assert scores[id1] < 0.5

    def test_relative_recency_fallback_to_now(self) -> None:
        """When no timestamps exist, falls back to datetime.now(UTC)."""
        from khora.core.models import Chunk
        from khora.engines.vectorcypher.fusion import FusedResult
        from khora.engines.vectorcypher.retriever import RetrieverConfig

        config = RetrieverConfig(recency_decay_days=30)

        ns_id = uuid4()
        id1 = uuid4()

        chunk1 = Chunk(
            id=id1,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="No timestamp",
            metadata={},
        )

        fused = [FusedResult(item=chunk1, rrf_score=0.5, item_id=id1)]

        from unittest.mock import MagicMock

        from khora.engines.vectorcypher.retriever import VectorCypherRetriever

        retriever = MagicMock()
        retriever._config = config
        scores = VectorCypherRetriever._calculate_recency_scores(retriever, fused)

        # Missing timestamps get default 0.5
        assert scores[id1] == 0.5

    def test_decay_days_override(self) -> None:
        """decay_days_override should change the decay rate."""
        from khora.core.models import Chunk
        from khora.engines.vectorcypher.fusion import FusedResult
        from khora.engines.vectorcypher.retriever import RetrieverConfig

        config = RetrieverConfig(recency_decay_days=30)

        ns_id = uuid4()
        id1, id2 = uuid4(), uuid4()

        chunk1 = Chunk(
            id=id1,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="Old",
            metadata={"occurred_at": "2024-01-01T00:00:00+00:00"},
        )
        chunk2 = Chunk(
            id=id2,
            namespace_id=ns_id,
            document_id=uuid4(),
            content="New",
            metadata={"occurred_at": "2024-01-08T00:00:00+00:00"},
        )

        fused = [
            FusedResult(item=chunk1, rrf_score=0.5, item_id=id1),
            FusedResult(item=chunk2, rrf_score=0.5, item_id=id2),
        ]

        from unittest.mock import MagicMock

        from khora.engines.vectorcypher.retriever import VectorCypherRetriever

        retriever = MagicMock()
        retriever._config = config

        # With decay_days_override=7, 7 days old should give score ~0.5
        scores = VectorCypherRetriever._calculate_recency_scores(retriever, fused, decay_days_override=7)
        assert scores[id1] == pytest.approx(0.5, abs=0.05)  # 7 days old with 7-day half-life


# ---------------------------------------------------------------------------
# TemporalSignal dataclass tests
# ---------------------------------------------------------------------------


class TestTemporalSignal:
    """Test TemporalSignal frozen dataclass."""

    def test_creation(self) -> None:
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.STATE_QUERY,
            confidence=0.9,
            source="dictionary",
        )
        assert signal.is_temporal is True
        assert signal.category == TemporalCategory.STATE_QUERY
        assert signal.temporal_filter is None

    def test_frozen(self) -> None:
        signal = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.NONE,
            confidence=1.0,
            source="none",
        )
        with pytest.raises(AttributeError):
            signal.category = TemporalCategory.CHANGE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Category map tests
# ---------------------------------------------------------------------------


class TestCategoryMap:
    """Test integer → TemporalCategory mapping."""

    def test_all_ids_mapped(self) -> None:
        for i in range(7):
            assert i in CATEGORY_MAP

    def test_zero_is_none(self) -> None:
        assert CATEGORY_MAP[0] == TemporalCategory.NONE

    def test_six_is_change(self) -> None:
        assert CATEGORY_MAP[6] == TemporalCategory.CHANGE

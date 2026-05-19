"""Coverage-driven tests for ``khora.query.temporal_detection``.

The module is a three-tier cascade for detecting temporal intent in
queries. Tier 1 (Aho-Corasick dictionary) is tested through the public
``TemporalDetector.detect`` entry point with the Rust accelerator
monkeypatched so tests don't depend on its build state. Tier 2
(semantic centroid) uses a stubbed numpy-style centroid. Tier 3
(LLM disambiguation) mocks ``khora.config.llm.acompletion``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from khora.query import temporal_detection as td
from khora.query.temporal_detection import (
    ANTI_RECENCY_TOKENS,
    CATEGORY_MAP,
    RETRIEVAL_PARAMS,
    TemporalCategory,
    TemporalDetector,
    TemporalIntent,
    TemporalSignal,
    classify_temporal_intent_llm,
    get_retrieval_params,
    has_ambiguity_trigger,
    has_anti_recency_token,
)


@pytest.mark.unit
class TestHasAntiRecencyToken:
    @pytest.mark.parametrize(
        "query",
        [
            "what action items have we ever discussed",
            "the entire history of Phoenix",
            "any time we mentioned it",
            "if we had shipped on time",
            "what would have happened",
            "back in 2021",
            "since the beginning",
            "all-time favorites",
        ],
    )
    def test_returns_true_when_phrase_present(self, query: str) -> None:
        assert has_anti_recency_token(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "latest action items",
            "what did Alice say today",
            "show me recent emails",
            "",  # empty
        ],
    )
    def test_returns_false_when_absent(self, query: str) -> None:
        assert has_anti_recency_token(query) is False

    def test_anti_recency_token_set_contains_expected_phrases(self) -> None:
        # Spot-check: catch accidental token removal in future refactors
        assert "ever" in ANTI_RECENCY_TOKENS
        assert "history of" in ANTI_RECENCY_TOKENS
        assert "would have" in ANTI_RECENCY_TOKENS


@pytest.mark.unit
class TestHasAmbiguityTrigger:
    @pytest.mark.parametrize(
        "query",
        [
            "what would happen if we shipped",
            "imagine the team disagreed",
            "previously discussed in slack",
            "back when we used Mongo",
            "earlier this year",
        ],
    )
    def test_true_for_ambiguous(self, query: str) -> None:
        assert has_ambiguity_trigger(query) is True

    def test_false_for_clear_query(self) -> None:
        assert has_ambiguity_trigger("latest action items") is False

    def test_empty_string(self) -> None:
        assert has_ambiguity_trigger("") is False


@pytest.mark.unit
class TestTemporalDetectorTier1:
    def test_dictionary_hit_returns_temporal_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force tier 1 to return RECENCY (cat_id=5)
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 5)
        detector = TemporalDetector()
        result = detector.detect("latest stuff")
        assert result.is_temporal is True
        assert result.category == TemporalCategory.RECENCY
        assert result.confidence == 0.9
        assert result.source == "dictionary"
        assert result.temporal_filter is None

    def test_explicit_category_extracts_date_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 1)
        detector = TemporalDetector()
        result = detector.detect("notes after 2024-01-15")
        assert result.is_temporal is True
        assert result.category == TemporalCategory.EXPLICIT
        # filter should be a TemporalFilter object
        assert result.temporal_filter is not None
        assert result.temporal_filter.occurred_after is not None

    def test_explicit_category_with_no_date_returns_none_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 1)
        detector = TemporalDetector()
        result = detector.detect("explicit but no date here")
        assert result.category == TemporalCategory.EXPLICIT
        assert result.temporal_filter is None

    def test_no_temporal_returns_none_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 0)
        detector = TemporalDetector()
        result = detector.detect("the cat sat on the mat")
        assert result.is_temporal is False
        assert result.category == TemporalCategory.NONE
        assert result.source == "none"


@pytest.mark.unit
class TestTemporalDetectorTier2:
    def test_semantic_tier_fires_above_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 0)
        # Centroid and embedding chosen so np.dot ~= 0.9 (above 0.20 threshold)
        centroid = [1.0, 0.0, 0.0]
        query_embedding = [0.9, 0.0, 0.0]
        detector = TemporalDetector(semantic_enabled=True, centroid=centroid)
        result = detector.detect("vague query", query_embedding=query_embedding)
        assert result.is_temporal is True
        assert result.category == TemporalCategory.STATE_QUERY
        assert result.source == "semantic"
        assert result.confidence > 0.20

    def test_semantic_tier_below_threshold_no_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 0)
        centroid = [1.0, 0.0, 0.0]
        query_embedding = [0.05, 0.0, 0.0]
        detector = TemporalDetector(semantic_enabled=True, centroid=centroid)
        result = detector.detect("vague query", query_embedding=query_embedding)
        assert result.is_temporal is False
        assert result.source == "none"

    def test_semantic_disabled_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 0)
        # semantic_enabled=False — even with centroid present, skips tier 2
        detector = TemporalDetector(semantic_enabled=False, centroid=[1.0, 0.0, 0.0])
        result = detector.detect("vague query", query_embedding=[1.0, 0.0, 0.0])
        assert result.is_temporal is False


@pytest.mark.unit
class TestExtractDateFilter:
    def test_before_keyword(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 1)
        detector = TemporalDetector()
        result = detector.detect("anything before 2024-06-15")
        assert result.temporal_filter is not None
        assert result.temporal_filter.occurred_before is not None
        assert result.temporal_filter.occurred_after is None

    def test_since_keyword(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 1)
        detector = TemporalDetector()
        result = detector.detect("anything since 2024-06-15")
        assert result.temporal_filter.occurred_after is not None
        assert result.temporal_filter.occurred_before is None

    def test_default_window_around_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(td, "detect_temporal_category", lambda q: 1)
        detector = TemporalDetector()
        result = detector.detect("notes near 2024-06-15")
        assert result.temporal_filter.occurred_after is not None
        assert result.temporal_filter.occurred_before is not None


@pytest.mark.unit
class TestParseDatetime:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2024-06-15", datetime(2024, 6, 15, tzinfo=UTC)),
            ("2024/06/15", datetime(2024, 6, 15, tzinfo=UTC)),
            ("June 15, 2024", datetime(2024, 6, 15, tzinfo=UTC)),
            ("15 June 2024", datetime(2024, 6, 15, tzinfo=UTC)),
        ],
    )
    def test_parses_known_formats(self, value: str, expected: datetime) -> None:
        assert TemporalDetector._parse_datetime(value) == expected

    def test_iso_with_timezone(self) -> None:
        out = TemporalDetector._parse_datetime("2024-06-15T10:00:00Z")
        assert out.tzinfo is not None

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError):
            TemporalDetector._parse_datetime("not a date")


@pytest.mark.unit
class TestGetRetrievalParams:
    def test_returns_params_for_each_category(self) -> None:
        for cat in TemporalCategory:
            sig = TemporalSignal(
                is_temporal=cat != TemporalCategory.NONE,
                category=cat,
                confidence=0.9,
                source="dictionary",
            )
            params = get_retrieval_params(sig)
            assert params is RETRIEVAL_PARAMS[cat]

    def test_category_map_round_trips(self) -> None:
        # 0..6 must map to a TemporalCategory member
        for cat_id in range(7):
            assert cat_id in CATEGORY_MAP


@pytest.mark.unit
class TestClassifyTemporalIntentLLM:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        td._TEMPORAL_INTENT_CACHE.clear()

    async def test_cache_hit_returns_cached_value(self) -> None:
        td._TEMPORAL_INTENT_CACHE["hello"] = (TemporalIntent.RECENT, 1.0)
        intent, conf = await classify_temporal_intent_llm("Hello")
        assert intent == TemporalIntent.RECENT
        assert conf == 1.0

    async def test_parses_recent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(prompt: str, config, **kwargs) -> str:
            return "RECENT"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        intent, conf = await classify_temporal_intent_llm("any new query")
        assert intent == TemporalIntent.RECENT
        assert conf == 1.0

    async def test_parses_historical_with_trailing_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(prompt: str, config, **kwargs) -> str:
            return "HISTORICAL. Background context."

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        intent, _ = await classify_temporal_intent_llm("history query")
        assert intent == TemporalIntent.HISTORICAL

    async def test_unparseable_returns_neutral_with_zero_conf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(prompt: str, config, **kwargs) -> str:
            return "I'm not sure"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        intent, conf = await classify_temporal_intent_llm("ambiguous")
        assert intent == TemporalIntent.NEUTRAL
        assert conf == 0.0

    async def test_llm_exception_returns_neutral_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        intent, conf = await classify_temporal_intent_llm("any q")
        assert intent == TemporalIntent.NEUTRAL
        assert conf == 0.0

    async def test_cache_eviction_when_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pre-fill cache to the max
        for i in range(td._TEMPORAL_INTENT_CACHE_MAX_SIZE):
            td._TEMPORAL_INTENT_CACHE[f"q{i}"] = (TemporalIntent.NEUTRAL, 1.0)

        async def fake_acomp(prompt: str, config, **kwargs) -> str:
            return "NEUTRAL"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        await classify_temporal_intent_llm("brand new query")
        # Cache size should still be <= max
        assert len(td._TEMPORAL_INTENT_CACHE) <= td._TEMPORAL_INTENT_CACHE_MAX_SIZE
        # Oldest key should have been evicted
        assert "q0" not in td._TEMPORAL_INTENT_CACHE

    async def test_unknown_word_neutral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_acomp(prompt: str, config, **kwargs) -> str:
            return "MAYBE"

        monkeypatch.setattr("khora.config.llm.acompletion", fake_acomp)
        intent, conf = await classify_temporal_intent_llm("weird query")
        assert intent == TemporalIntent.NEUTRAL
        assert conf == 0.0

"""Unit tests for Retrieval Quality Phase 3: temporal pattern expansion + engine profiles."""

from __future__ import annotations

import pytest

from khora import _accel as accel
from khora.query.engine_profiles import (
    apply_engine_profile,
    get_engine_profile,
    list_engine_profiles,
)


@pytest.fixture(autouse=True)
def force_python(monkeypatch):
    """Force the pure-Python path so we test the Python TEMPORAL_DICTIONARY.

    The Rust backend has its own compiled Aho-Corasick automaton that must be
    updated separately via khora-accel. These tests validate the Python-side
    dictionary expansions.
    """
    monkeypatch.setattr(accel, "_HAS_RUST", False)
    monkeypatch.setattr(accel, "_HAS_NUMPY", False)


# ---------------------------------------------------------------------------
# Temporal detection: new high-precision patterns only
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExplicitDayOfWeek:
    """Category 1 — day-of-week and period patterns are unambiguous temporal markers."""

    @pytest.mark.parametrize(
        "query",
        [
            "What did we discuss on Monday?",
            "The meeting on Wednesday was productive",
            "What happened on Friday?",
        ],
    )
    def test_day_of_week(self, query: str) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat >= 1

    @pytest.mark.parametrize(
        "query",
        [
            "What happened this week?",
            "Summarize this month's progress",
            "What have we done this year?",
            "She mentioned it this morning",
        ],
    )
    def test_this_period(self, query: str) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat >= 1

    def test_tomorrow(self) -> None:
        cat = accel.detect_temporal_category("What's planned for tomorrow?")
        assert cat >= 1


@pytest.mark.unit
class TestStateQueryStillVariants:
    """Category 2 — expanded 'still' pronoun coverage."""

    @pytest.mark.parametrize(
        "query",
        [
            "Does it still work after the upgrade?",
            "Is it still broken?",
            "Are we still using that library?",
            "Do we still support Python 3.9?",
        ],
    )
    def test_still_patterns(self, query: str) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat >= 2

    @pytest.mark.parametrize(
        "query",
        [
            "As of now, who is leading the project?",
            "As we speak, the migration is running",
            "At this time we don't have a solution",
        ],
    )
    def test_temporal_adverbs(self, query: str) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat >= 2


@pytest.mark.unit
class TestOrdinalExpansions:
    """Category 3 — ordering/sequence patterns."""

    @pytest.mark.parametrize(
        "query",
        [
            "What came before the migration?",
            "What came after the announcement?",
            "In what order did the releases happen?",
            "List the events in chronological order",
            "What preceded the decision to pivot?",
        ],
    )
    def test_ordinal_patterns(self, query: str) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat >= 3


@pytest.mark.unit
class TestAggregateExpansion:
    """Category 4 — 'how frequently' is a direct synonym of 'how often'."""

    def test_how_frequently(self) -> None:
        cat = accel.detect_temporal_category("How frequently does she travel?")
        assert cat >= 4


@pytest.mark.unit
class TestRecencyExpansion:
    """Category 5 — 'most recently' is the adverb form of 'most recent'."""

    def test_most_recently(self) -> None:
        cat = accel.detect_temporal_category("What was most recently discussed?")
        assert cat >= 5


@pytest.mark.unit
class TestFalsePositives:
    """Ensure common non-temporal queries remain category 0."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is the capital of France?",
            "Explain how vector search works",
            "Define knowledge graph",
            "What are the benefits of microservices?",
            "How does pgvector store embeddings?",
            "The config was modified by the linter",
            "She dropped the connection",
            "I started the server",
        ],
    )
    def test_non_temporal_stays_zero(self, query: str) -> None:
        cat = accel.detect_temporal_category(query)
        assert cat == 0, f"Expected NONE for: {query}"


# ---------------------------------------------------------------------------
# Engine profiles
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEngineProfiles:
    """Tests for per-engine tuning profiles."""

    def test_list_profiles(self) -> None:
        profiles = list_engine_profiles()
        assert "graphrag" in profiles
        assert "vectorcypher" in profiles
        assert "skeleton" in profiles
        assert "chronicle" in profiles

    def test_get_graphrag_profile(self) -> None:
        profile = get_engine_profile("graphrag")
        assert profile["fusion_strategy"] == "combmnz"
        assert profile["graph_weight"] == 0.40
        assert profile["linked_entity_boost"] == 2.0

    def test_get_vectorcypher_profile(self) -> None:
        profile = get_engine_profile("vectorcypher")
        assert profile["graph_weight"] == 0.50
        assert profile["linked_entity_boost"] == 2.5
        assert profile["graph_chunk_query_sim_weight"] == 0.4

    def test_get_skeleton_profile(self) -> None:
        profile = get_engine_profile("skeleton")
        assert profile["enable_reranking"] is False
        assert profile["enable_hyde"] == "never"
        assert profile["enable_entity_linking"] is False
        assert profile["apply_recency_bias"] is True
        assert profile["graph_weight"] == 0.0

    def test_get_chronicle_profile(self) -> None:
        profile = get_engine_profile("chronicle")
        assert profile["apply_recency_bias"] is True
        assert profile["temporal_half_life_hours"] == 168.0
        assert profile["temporal_hard_cutoff_days"] == 60.0

    def test_get_unknown_returns_empty(self) -> None:
        profile = get_engine_profile("nonexistent")
        assert profile == {}

    def test_get_profile_returns_copy(self) -> None:
        p1 = get_engine_profile("graphrag")
        p2 = get_engine_profile("graphrag")
        p1["fusion_strategy"] = "rrf"
        assert p2["fusion_strategy"] == "combmnz"

    def test_apply_profile_to_dataclass(self) -> None:
        """Apply profile to a QueryConfig-like dataclass."""
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            vector_weight: float = 0.5
            graph_weight: float = 0.3
            fusion_strategy: str = "rrf"
            enable_reranking: bool = True

        config = FakeConfig()
        result = apply_engine_profile(config, "skeleton")
        assert result is config  # mutates in place
        assert config.vector_weight == 0.70
        assert config.graph_weight == 0.0
        assert config.enable_reranking is False

    def test_apply_unknown_engine_is_noop(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            vector_weight: float = 0.5

        config = FakeConfig()
        apply_engine_profile(config, "unknown_engine")
        assert config.vector_weight == 0.5

    def test_skeleton_and_chronicle_no_graph(self) -> None:
        for engine in ("skeleton", "chronicle"):
            profile = get_engine_profile(engine)
            assert profile["graph_weight"] == 0.0, f"{engine} should have no graph weight"

    def test_weights_sum_reasonable(self) -> None:
        for engine in list_engine_profiles():
            profile = get_engine_profile(engine)
            total = (
                profile.get("vector_weight", 0.5)
                + profile.get("graph_weight", 0.3)
                + profile.get("keyword_weight", 0.2)
            )
            assert 0.5 <= total <= 1.1, f"{engine} weights sum to {total}"

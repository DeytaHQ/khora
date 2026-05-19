"""Coverage-driven tests for ``khora.query.router``.

Focuses on the heuristic branches not exercised by the existing
``test_typed_entity_recent_routing.py`` and the chronicle / vectorcypher
integration tests:

* constructor overrides (``use_llm`` / ``llm_config``)
* every pattern bucket (relationship, comparison, multi-hop, causal,
  counterfactual, hierarchical, aggregation, temporal)
* simple / factual reducers
* word-count / sentence-count / question-word boosts
* temporal-signal boost to MODERATE floor
* ENTITY_ANCHORED promotion + blocking patterns
* the disabled-router default path
* the LLM fallback (success, malformed response, exception)
* ``compute_adaptive_depth`` high / low / mid ranges + disabled flag
* stats reset
* ``_count_potential_entities`` entity sources (proper nouns, quoted,
  CamelCase, snake_case, @ / # mentions)

External LLM is mocked at the import site inside ``_llm_route``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from khora.query.router import (
    QueryComplexity,
    QueryComplexityRouter,
    RouterConfig,
    match_typed_entity_recent,
)
from khora.query.temporal_detection import TemporalCategory, TemporalSignal

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# constructor overrides
# ---------------------------------------------------------------------------


class TestConstructorOverrides:
    def test_use_llm_override_true(self) -> None:
        r = QueryComplexityRouter(use_llm=True)
        assert r._config.use_llm is True

    def test_use_llm_override_false(self) -> None:
        r = QueryComplexityRouter(use_llm=False)
        assert r._config.use_llm is False

    def test_llm_config_override(self) -> None:
        from khora.config.llm import LiteLLMConfig

        cfg = LiteLLMConfig(model="custom-model")
        r = QueryComplexityRouter(llm_config=cfg)
        assert r._config.llm_config is cfg


# ---------------------------------------------------------------------------
# route() — disabled, simple, complex
# ---------------------------------------------------------------------------


class TestRouteEntry:
    @pytest.mark.asyncio
    async def test_disabled_returns_moderate(self) -> None:
        r = QueryComplexityRouter(RouterConfig(enabled=False))
        decision = await r.route("anything")
        assert decision.complexity == QueryComplexity.MODERATE
        assert decision.confidence == 1.0
        assert "Routing disabled" in decision.reasoning


# ---------------------------------------------------------------------------
# Pattern matching — all branches
# ---------------------------------------------------------------------------


class TestHeuristicPatterns:
    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        return QueryComplexityRouter(RouterConfig(use_llm=False))

    @pytest.mark.asyncio
    async def test_relationship_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("what is the relationship between Alice and Bob")
        assert "relationship keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_comparison_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("compare X versus Y and contrast their advantages")
        assert "comparison keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_multi_hop_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("how does A flow through B via C upstream")
        assert "multi-hop keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_causal_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("why does this cause the system failure to occur")
        assert "causal keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_counterfactual_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("what would have happened if she had not joined")
        assert "counterfactual keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_hierarchical_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("what items are part of the parent project")
        assert "hierarchical keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_aggregation_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("how many total commits did the team make")
        assert "aggregation keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_temporal_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("show me the timeline of changes since the launch")
        assert "temporal keywords" in d.reasoning

    @pytest.mark.asyncio
    async def test_simple_question_pattern_reduces_score(self, router: QueryComplexityRouter) -> None:
        d = await router.route("what is Python")
        # Should land in SIMPLE
        assert d.complexity in (QueryComplexity.SIMPLE, QueryComplexity.ENTITY_ANCHORED)

    @pytest.mark.asyncio
    async def test_factual_pattern(self, router: QueryComplexityRouter) -> None:
        d = await router.route("what does foo mean?")
        assert "factual query pattern" in d.reasoning


# ---------------------------------------------------------------------------
# Structural heuristics
# ---------------------------------------------------------------------------


class TestStructuralHeuristics:
    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        return QueryComplexityRouter(RouterConfig(use_llm=False))

    @pytest.mark.asyncio
    async def test_long_query_boost(self, router: QueryComplexityRouter) -> None:
        # 30 words
        q = "alpha " * 30
        d = await router.route(q.strip())
        assert "long query" in d.reasoning

    @pytest.mark.asyncio
    async def test_medium_query_boost(self, router: QueryComplexityRouter) -> None:
        # 20 words — between 15 and 25, gets +0.05 but no reason text
        q = " ".join(["alpha"] * 20)
        d = await router.route(q)
        # No "long query" tag but should not be very short either
        assert "long query" not in d.reasoning
        assert "very short query" not in d.reasoning

    @pytest.mark.asyncio
    async def test_very_short_query(self, router: QueryComplexityRouter) -> None:
        d = await router.route("hi there")
        assert "very short query" in d.reasoning

    @pytest.mark.asyncio
    async def test_short_query(self, router: QueryComplexityRouter) -> None:
        d = await router.route("alpha beta gamma delta epsilon")
        assert "short query" in d.reasoning

    @pytest.mark.asyncio
    async def test_multiple_sentences(self, router: QueryComplexityRouter) -> None:
        d = await router.route("First sentence. Second sentence! Third sentence?")
        assert "multi-sentence" in d.reasoning

    @pytest.mark.asyncio
    async def test_multiple_question_words(self, router: QueryComplexityRouter) -> None:
        d = await router.route("what who when did this happen and why")
        assert "multiple question words" in d.reasoning


# ---------------------------------------------------------------------------
# Temporal signal boost
# ---------------------------------------------------------------------------


class TestTemporalSignalBoost:
    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        return QueryComplexityRouter(RouterConfig(use_llm=False))

    @pytest.mark.asyncio
    async def test_temporal_signal_boosts_to_moderate(self, router: QueryComplexityRouter) -> None:
        # "When did Alice move?" — short query, would normally be SIMPLE without temporal
        sig = TemporalSignal(
            is_temporal=True,
            category=TemporalCategory.RECENCY,
            confidence=0.9,
            source="test",
        )
        d = await router.route("did she move", temporal_signal=sig)
        assert "temporal signal boosted to MODERATE" in d.reasoning
        # Should land at MODERATE (or higher)
        assert d.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.ENTITY_ANCHORED)

    @pytest.mark.asyncio
    async def test_temporal_signal_not_temporal_no_boost(self, router: QueryComplexityRouter) -> None:
        sig = TemporalSignal(
            is_temporal=False,
            category=TemporalCategory.NONE,
            confidence=0.0,
            source="test",
        )
        d = await router.route("hi there", temporal_signal=sig)
        assert "temporal signal boosted" not in d.reasoning


# ---------------------------------------------------------------------------
# ENTITY_ANCHORED branch
# ---------------------------------------------------------------------------


class TestEntityAnchored:
    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        return QueryComplexityRouter(RouterConfig(use_llm=False))

    @pytest.mark.asyncio
    async def test_named_entity_short_query_promoted(self, router: QueryComplexityRouter) -> None:
        d = await router.route("tell me about Alice")
        assert d.complexity == QueryComplexity.ENTITY_ANCHORED

    @pytest.mark.asyncio
    async def test_comparison_blocks_entity_anchored(self, router: QueryComplexityRouter) -> None:
        d = await router.route("compare Alice and Bob")
        # comparison pattern blocks ENTITY_ANCHORED promotion
        assert d.complexity != QueryComplexity.ENTITY_ANCHORED


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_llm_fallback_simple(self) -> None:
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5)  # always trigger LLM
        r = QueryComplexityRouter(cfg)
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="SIMPLE|just a lookup")):
            d = await r.route("anything")
        assert d.complexity == QueryComplexity.SIMPLE
        assert d.use_graph is False
        assert "LLM:" in d.reasoning
        assert r.get_routing_stats()["llm_fallback"] == 1

    @pytest.mark.asyncio
    async def test_llm_fallback_moderate(self) -> None:
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5)
        r = QueryComplexityRouter(cfg)
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="MODERATE|single rel")):
            d = await r.route("show me X")
        assert d.complexity == QueryComplexity.MODERATE

    @pytest.mark.asyncio
    async def test_llm_fallback_complex(self) -> None:
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5)
        r = QueryComplexityRouter(cfg)
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="COMPLEX|multi-hop")):
            d = await r.route("how is A connected to B through C")
        assert d.complexity == QueryComplexity.COMPLEX

    @pytest.mark.asyncio
    async def test_llm_fallback_unknown_response_falls_back_to_heuristic(self) -> None:
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5)
        r = QueryComplexityRouter(cfg)
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="UNKNOWN|something")):
            d = await r.route("alpha beta gamma")
        # Returns the heuristic_result with an extra annotation
        assert "(LLM unclear: UNKNOWN" in d.reasoning

    @pytest.mark.asyncio
    async def test_llm_fallback_exception_returns_heuristic(self) -> None:
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5)
        r = QueryComplexityRouter(cfg)
        with patch(
            "khora.config.llm.acompletion",
            new=AsyncMock(side_effect=RuntimeError("api fail")),
        ):
            d = await r.route("alpha beta gamma")
        # Heuristic result, with LLM error annotated
        assert "LLM error" in d.reasoning or "LLM fallback failed" in d.reasoning

    @pytest.mark.asyncio
    async def test_llm_fallback_with_custom_llm_config(self) -> None:
        from khora.config.llm import LiteLLMConfig

        custom_cfg = LiteLLMConfig(model="custom-router-model")
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5, llm_config=custom_cfg)
        r = QueryComplexityRouter(cfg)
        ac = AsyncMock(return_value="SIMPLE|test")
        with patch("khora.config.llm.acompletion", new=ac):
            await r.route("alpha")
        # The config used should be derived from custom_cfg model
        call_kwargs = ac.await_args.kwargs
        assert call_kwargs["config"].model == "custom-router-model"

    @pytest.mark.asyncio
    async def test_llm_response_without_pipe(self) -> None:
        cfg = RouterConfig(use_llm=True, llm_confidence_threshold=1.5)
        r = QueryComplexityRouter(cfg)
        with patch("khora.config.llm.acompletion", new=AsyncMock(return_value="SIMPLE")):
            d = await r.route("alpha")
        assert d.complexity == QueryComplexity.SIMPLE
        # default reasoning when no pipe present
        assert "LLM classification" in d.reasoning


# ---------------------------------------------------------------------------
# _count_potential_entities
# ---------------------------------------------------------------------------


class TestCountPotentialEntities:
    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        return QueryComplexityRouter()

    def test_first_word_capital_does_not_count(self, router: QueryComplexityRouter) -> None:
        # "What" at position 0 should not be counted as entity
        assert router._count_potential_entities("What is going on") == 0

    def test_capitalized_middle_word(self, router: QueryComplexityRouter) -> None:
        assert router._count_potential_entities("tell me about Alice") == 1

    def test_multi_word_entity(self, router: QueryComplexityRouter) -> None:
        # "Acme Corp" should count as 1 entity
        assert router._count_potential_entities("tell me about Acme Corp today") == 1

    def test_quoted_string(self, router: QueryComplexityRouter) -> None:
        n = router._count_potential_entities('find "project alpha" status')
        assert n >= 1

    def test_camel_case(self, router: QueryComplexityRouter) -> None:
        # CamelCase regex requires uppercase-start: PascalCase form
        assert router._count_potential_entities("look up MyFunction in code") >= 1

    def test_snake_case(self, router: QueryComplexityRouter) -> None:
        assert router._count_potential_entities("look up my_function in code") >= 1

    def test_mentions(self, router: QueryComplexityRouter) -> None:
        assert router._count_potential_entities("ping @alice and #devops") >= 2

    def test_all_caps_word_skipped(self, router: QueryComplexityRouter) -> None:
        # "API" is all caps — currently the heuristic skips it
        assert router._count_potential_entities("show the API") == 0


# ---------------------------------------------------------------------------
# compute_adaptive_depth
# ---------------------------------------------------------------------------


class TestComputeAdaptiveDepth:
    def test_disabled_returns_base_depth(self) -> None:
        cfg = RouterConfig(adaptive_depth_enabled=False)
        r = QueryComplexityRouter(cfg)
        assert r.compute_adaptive_depth(entry_entity_count=100, base_depth=2) == 2

    def test_high_entity_count_reduces_depth(self) -> None:
        r = QueryComplexityRouter(RouterConfig(adaptive_depth_high_entity_threshold=10))
        assert r.compute_adaptive_depth(entry_entity_count=15, base_depth=3) == 1

    def test_low_entity_count_increases_depth(self) -> None:
        r = QueryComplexityRouter(RouterConfig(adaptive_depth_low_entity_threshold=2, complex_depth=3))
        # base 1 → +1 → 2, but capped at complex_depth+1=4
        adjusted = r.compute_adaptive_depth(entry_entity_count=1, base_depth=1)
        assert adjusted == 2

    def test_low_entity_caps_at_complex_plus_one(self) -> None:
        r = QueryComplexityRouter(RouterConfig(complex_depth=3))
        adjusted = r.compute_adaptive_depth(entry_entity_count=1, base_depth=10)
        # min(10+1, 3+1) = 4
        assert adjusted == 4

    def test_mid_range_returns_base(self) -> None:
        r = QueryComplexityRouter(
            RouterConfig(
                adaptive_depth_high_entity_threshold=10,
                adaptive_depth_low_entity_threshold=2,
            )
        )
        assert r.compute_adaptive_depth(entry_entity_count=5, base_depth=2) == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_increment(self) -> None:
        r = QueryComplexityRouter(RouterConfig(use_llm=False))
        await r.route("hi there")
        await r.route("compare A versus B versus C")
        stats = r.get_routing_stats()
        assert sum(stats.values()) == 2

    @pytest.mark.asyncio
    async def test_reset_clears_stats(self) -> None:
        r = QueryComplexityRouter(RouterConfig(use_llm=False))
        await r.route("hi there")
        r.reset_routing_stats()
        stats = r.get_routing_stats()
        assert all(v == 0 for v in stats.values())


# ---------------------------------------------------------------------------
# match_typed_entity_recent edge cases
# ---------------------------------------------------------------------------


class TestMatchTypedEntityRecent:
    def test_match_singular(self) -> None:
        assert match_typed_entity_recent("latest decision") == "DECISION"

    def test_match_plural(self) -> None:
        assert match_typed_entity_recent("most recent risks") == "RISK"

    def test_no_match(self) -> None:
        assert match_typed_entity_recent("recent emails") is None

    def test_anti_recency_token_blocks(self) -> None:
        assert match_typed_entity_recent("what action items have we ever discussed") is None

    def test_empty_string(self) -> None:
        assert match_typed_entity_recent("") is None

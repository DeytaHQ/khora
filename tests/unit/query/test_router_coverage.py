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
from uuid import uuid4

import pytest

from khora.query.degree_stats import DegreeStats
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
# Frontier-budgeted adaptive depth (#1477)
# ---------------------------------------------------------------------------


def _stats(degree_map: dict, mean: float, *, num_entities: int = 10_000) -> DegreeStats:
    return DegreeStats(
        num_entities=num_entities,
        mean_degree=mean,
        median_degree=1.0,
        max_degree=max(degree_map.values(), default=0),
        degree_by_entity=degree_map,
    )


class TestFrontierBudgetedDepth:
    """The #1477 rule: predict frontier from seed degree x branching factor.

    The old count-only rule is sign-wrong on power-law graphs. With a degree
    histogram, hub seeds (few but high-degree) must go SHALLOWER than the old
    rule, many low-degree seeds must go DEEPER, and a missing histogram must
    fall back to the exact old count rule.
    """

    def test_hub_seeds_go_shallower_than_old_rule(self) -> None:
        # Two hub seeds (degree 5000 each) on a power-law graph. The old rule
        # keys on count (2 <= low threshold) and DEEPENS to base+1=3 - exactly
        # when the one-hop frontier (~10000) explodes. The frontier rule sees
        # the seeds' actual degrees blow the budget and drops to depth 1.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        seeds = [uuid4(), uuid4()]
        stats = _stats({seeds[0]: 5000, seeds[1]: 5000}, mean=4.0)

        old = r.compute_adaptive_depth(entry_entity_count=2, base_depth=2)
        new = r.compute_adaptive_depth(
            entry_entity_count=2,
            base_depth=2,
            degree_stats=stats,
            seed_entity_ids=seeds,
        )

        assert old == 3  # sign-wrong: deepens the hub query
        assert new == 1  # frontier rule: shallow because the frontier explodes
        assert new < old

    def test_many_low_degree_seeds_go_deeper_than_old_rule(self) -> None:
        # Fifteen low-degree seeds (degree 2 each). The old rule keys on count
        # (15 >= high threshold) and caps depth at 1 - leaving cheap context
        # unexplored. The frontier rule predicts a small frontier and goes deep.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        seeds = [uuid4() for _ in range(15)]
        stats = _stats({s: 2 for s in seeds}, mean=2.0)

        old = r.compute_adaptive_depth(entry_entity_count=15, base_depth=2)
        new = r.compute_adaptive_depth(
            entry_entity_count=15,
            base_depth=2,
            degree_stats=stats,
            seed_entity_ids=seeds,
        )

        assert old == 1  # sign-wrong: caps a cheap frontier
        assert new == 3  # frontier rule: deep because the frontier stays small
        assert new > old

    def test_missing_stats_falls_back_to_old_count_rule(self) -> None:
        # No histogram (fresh namespace / graph-less stack) -> the frontier rule
        # is not consulted at all; the exact old count rule decides.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        # High-count fallback: 15 >= threshold -> shallow (1).
        assert r.compute_adaptive_depth(entry_entity_count=15, base_depth=2, degree_stats=None) == 1
        # Low-count fallback: 2 <= threshold -> deeper (3).
        assert r.compute_adaptive_depth(entry_entity_count=2, base_depth=2, degree_stats=None) == 3
        # Mid-range fallback: base_depth unchanged.
        assert r.compute_adaptive_depth(entry_entity_count=5, base_depth=2, degree_stats=None) == 2

    def test_empty_graph_stats_fall_back_to_count_rule(self) -> None:
        # A DegreeStats with num_entities == 0 (edgeless namespace) is treated
        # as "no signal" and falls back to the count rule.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        empty = _stats({}, mean=0.0, num_entities=0)
        assert r.compute_adaptive_depth(entry_entity_count=15, base_depth=2, degree_stats=empty) == 1

    def test_typical_graph_reproduces_base_depth(self) -> None:
        # Conservatism guard: on a typical mid-range recall (5 seeds, mean
        # degree 5) the frontier rule reproduces the old base_depth, so the
        # default budget does not silently deepen every ordinary query.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        seeds = [uuid4() for _ in range(5)]
        stats = _stats({s: 5 for s in seeds}, mean=5.0)
        new = r.compute_adaptive_depth(
            entry_entity_count=5,
            base_depth=2,
            degree_stats=stats,
            seed_entity_ids=seeds,
        )
        assert new == 2

    def test_moderate_route_not_pushed_past_base_plus_one(self) -> None:
        # A MODERATE route (base_depth=1) with a cheap frontier can reach 2 but
        # never 3 - the frontier rule respects the router's complexity ceiling
        # (min(base+1, complex+1)) exactly like the old count rule did.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        seeds = [uuid4()]
        stats = _stats({seeds[0]: 3}, mean=3.0)
        new = r.compute_adaptive_depth(
            entry_entity_count=1,
            base_depth=1,
            degree_stats=stats,
            seed_entity_ids=seeds,
        )
        assert new == 2

    def test_disabled_flag_ignores_stats(self) -> None:
        r = QueryComplexityRouter(RouterConfig(adaptive_depth_enabled=False))
        seeds = [uuid4()]
        stats = _stats({seeds[0]: 5000}, mean=4.0)
        assert (
            r.compute_adaptive_depth(
                entry_entity_count=1,
                base_depth=2,
                degree_stats=stats,
                seed_entity_ids=seeds,
            )
            == 2
        )

    def test_unknown_seed_uses_mean_degree(self) -> None:
        # A seed absent from the histogram (e.g. dropped past the list cap) is
        # charged the mean degree, not zero - so it still contributes to the
        # frontier prediction rather than silently vanishing.
        r = QueryComplexityRouter(RouterConfig(complex_depth=2, adaptive_depth_frontier_budget=300))
        known = uuid4()
        unknown = uuid4()
        # mean 400 -> a single unknown seed alone predicts 400 > budget 300 at
        # depth 1, forcing the shallow floor.
        stats = _stats({known: 1}, mean=400.0)
        assert (
            r.compute_adaptive_depth(
                entry_entity_count=1,
                base_depth=2,
                degree_stats=stats,
                seed_entity_ids=[unknown],
            )
            == 1
        )


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

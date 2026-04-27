"""Tests for ENTITY_ANCHORED router classification (DYT-3147)."""

from __future__ import annotations

import pytest

from khora.query.router import QueryComplexity, QueryComplexityRouter, RouterConfig


@pytest.fixture
def router() -> QueryComplexityRouter:
    return QueryComplexityRouter(RouterConfig(use_llm=False))


class TestEntityAnchoredClassification:
    @pytest.mark.parametrize(
        "query",
        [
            "What did Alice work on?",
            "Tell me about Project Apollo",
            "Who is John Smith?",
            'Find notes about "Project Apollo"',
            "When did Alice join?",
        ],
    )
    async def test_entity_anchored_promotion(self, router: QueryComplexityRouter, query: str) -> None:
        decision = await router.route(query)
        assert decision.complexity == QueryComplexity.ENTITY_ANCHORED, (
            f"expected ENTITY_ANCHORED for {query!r}, got {decision.complexity}"
        )
        assert decision.use_graph is True

    async def test_no_entity_stays_simple(self, router: QueryComplexityRouter) -> None:
        decision = await router.route("what is the weather")
        assert decision.complexity == QueryComplexity.SIMPLE

    async def test_long_query_with_entity_not_anchored(self, router: QueryComplexityRouter) -> None:
        query = (
            "What did Alice work on during her time at the company "
            "and how did her contributions change over the years across multiple teams?"
        )
        decision = await router.route(query)
        assert decision.complexity != QueryComplexity.ENTITY_ANCHORED

    async def test_multi_hop_with_entity_not_anchored(self, router: QueryComplexityRouter) -> None:
        decision = await router.route("How is Alice connected to Bob through Project Apollo?")
        assert decision.complexity != QueryComplexity.ENTITY_ANCHORED
        assert decision.complexity == QueryComplexity.COMPLEX

    async def test_comparison_with_entity_not_anchored(self, router: QueryComplexityRouter) -> None:
        decision = await router.route("Compare Alice and Bob")
        assert decision.complexity != QueryComplexity.ENTITY_ANCHORED

    async def test_aggregation_with_entity_not_anchored(self, router: QueryComplexityRouter) -> None:
        decision = await router.route("List all projects Alice worked on")
        assert decision.complexity != QueryComplexity.ENTITY_ANCHORED

    async def test_entity_anchored_logged_in_stats(self, router: QueryComplexityRouter) -> None:
        await router.route("Who is Alice Smith?")
        stats = router.get_routing_stats()
        assert "entity_anchored" in stats
        assert stats["entity_anchored"] == 1

    async def test_reset_clears_entity_anchored(self, router: QueryComplexityRouter) -> None:
        await router.route("Who is Alice?")
        router.reset_routing_stats()
        stats = router.get_routing_stats()
        assert stats["entity_anchored"] == 0

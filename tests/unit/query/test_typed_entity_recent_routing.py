"""Tests for the typed-entity recency fast path router classification (issue #569)."""

from __future__ import annotations

import pytest

from khora.query.router import (
    QueryComplexity,
    QueryComplexityRouter,
    RouterConfig,
    match_typed_entity_recent,
)


@pytest.fixture
def router() -> QueryComplexityRouter:
    return QueryComplexityRouter(RouterConfig(use_llm=False))


class TestTypedEntityRecentClassification:
    @pytest.mark.parametrize(
        ("query", "expected_type"),
        [
            ("latest action items", "ACTION_ITEM"),
            ("most recent decisions from the offsite", "DECISION"),
            ("newest blockers in the sprint", "BLOCKER"),
            ("recent risks for Q3", "RISK"),
            ("latest commitments from the team", "COMMITMENT"),
            ("most recent open questions", "OPEN_QUESTION"),
        ],
    )
    async def test_typed_entity_recent_routed(
        self,
        router: QueryComplexityRouter,
        query: str,
        expected_type: str,
    ) -> None:
        decision = await router.route(query)
        assert decision.complexity == QueryComplexity.TYPED_ENTITY_RECENT, (
            f"expected TYPED_ENTITY_RECENT for {query!r}, got {decision.complexity}"
        )
        assert decision.use_graph is True
        # match_typed_entity_recent should return the same entity type
        assert match_typed_entity_recent(query) == expected_type

    async def test_anti_recency_token_vetoes_fast_path(self, router: QueryComplexityRouter) -> None:
        # "ever" is an anti-recency token — the user is asking historical scope,
        # not freshness, so the fast path must NOT apply.
        decision = await router.route("what action items have we ever discussed")
        assert decision.complexity != QueryComplexity.TYPED_ENTITY_RECENT
        assert match_typed_entity_recent("what action items have we ever discussed") is None

    async def test_non_typed_noun_not_routed(self, router: QueryComplexityRouter) -> None:
        # "recent emails" — has recency adjective but no typed-entity noun.
        decision = await router.route("recent emails about pricing")
        assert decision.complexity != QueryComplexity.TYPED_ENTITY_RECENT

    async def test_no_typed_noun_long_query(self, router: QueryComplexityRouter) -> None:
        # "what changed in the last sprint" — temporal but no typed-entity noun.
        decision = await router.route("what changed in the last sprint")
        assert decision.complexity != QueryComplexity.TYPED_ENTITY_RECENT

    async def test_history_of_action_items_vetoed(self, router: QueryComplexityRouter) -> None:
        # Multi-word anti-recency phrase "history of" must veto.
        decision = await router.route("show me the history of action items")
        assert decision.complexity != QueryComplexity.TYPED_ENTITY_RECENT

    async def test_singular_form_matches(self, router: QueryComplexityRouter) -> None:
        decision = await router.route("latest action item from yesterday")
        assert decision.complexity == QueryComplexity.TYPED_ENTITY_RECENT

    async def test_stats_increment_on_fast_path(self, router: QueryComplexityRouter) -> None:
        await router.route("latest action items")
        stats = router.get_routing_stats()
        assert "typed_entity_recent" in stats
        assert stats["typed_entity_recent"] == 1

    async def test_reset_clears_stats(self, router: QueryComplexityRouter) -> None:
        await router.route("latest decisions")
        router.reset_routing_stats()
        stats = router.get_routing_stats()
        assert stats["typed_entity_recent"] == 0

    async def test_fast_path_confidence_high(self, router: QueryComplexityRouter) -> None:
        decision = await router.route("most recent blockers")
        assert decision.confidence >= 0.9

    async def test_empty_query_no_match(self) -> None:
        assert match_typed_entity_recent("") is None
        assert match_typed_entity_recent("   ") is None

"""Unit tests for VectorCypher query complexity router."""

from __future__ import annotations

import pytest

from khora.engines.vectorcypher.router import (
    QueryComplexity,
    QueryComplexityRouter,
    RouterConfig,
    RoutingDecision,
)


class TestQueryComplexity:
    """Tests for QueryComplexity enum."""

    def test_enum_values(self) -> None:
        """Test all complexity levels exist."""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MODERATE.value == "moderate"
        assert QueryComplexity.COMPLEX.value == "complex"


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_create_routing_decision(self) -> None:
        """Test creating a RoutingDecision with all fields."""
        decision = RoutingDecision(
            complexity=QueryComplexity.COMPLEX,
            use_graph=True,
            graph_depth=3,
            confidence=0.85,
            reasoning="multi-hop query",
            suggested_entry_limit=15,
        )
        assert decision.complexity == QueryComplexity.COMPLEX
        assert decision.use_graph is True
        assert decision.graph_depth == 3
        assert decision.confidence == 0.85
        assert decision.suggested_entry_limit == 15

    def test_default_entry_limit(self) -> None:
        """Test default suggested_entry_limit."""
        decision = RoutingDecision(
            complexity=QueryComplexity.SIMPLE,
            use_graph=False,
            graph_depth=0,
            confidence=0.9,
            reasoning="simple",
        )
        assert decision.suggested_entry_limit == 10


class TestRouterConfig:
    """Tests for RouterConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default router configuration."""
        config = RouterConfig()
        assert config.enabled is True
        assert config.use_llm is False
        assert config.simple_depth == 0
        assert config.moderate_depth == 1
        assert config.complex_depth == 2
        assert config.simple_entry_limit == 5
        assert config.moderate_entry_limit == 10
        assert config.complex_entry_limit == 15
        assert config.multi_entity_threshold == 2
        assert config.adaptive_depth_enabled is True

    def test_custom_config(self) -> None:
        """Test custom router configuration."""
        config = RouterConfig(
            enabled=False,
            complex_depth=4,
            adaptive_depth_high_entity_threshold=20,
        )
        assert config.enabled is False
        assert config.complex_depth == 4
        assert config.adaptive_depth_high_entity_threshold == 20


class TestQueryComplexityRouterInit:
    """Tests for QueryComplexityRouter initialization."""

    def test_default_init(self) -> None:
        """Test router with default config."""
        router = QueryComplexityRouter()
        assert router._config.enabled is True

    def test_custom_config_init(self) -> None:
        """Test router with custom config."""
        config = RouterConfig(enabled=False, use_llm=True)
        router = QueryComplexityRouter(config)
        assert router._config.enabled is False
        assert router._config.use_llm is True

    def test_use_llm_override(self) -> None:
        """Test that use_llm kwarg overrides config."""
        config = RouterConfig(use_llm=False)
        router = QueryComplexityRouter(config, use_llm=True)
        assert router._config.use_llm is True

    def test_routing_stats_initialized(self) -> None:
        """Test that routing stats are initialized to zero."""
        router = QueryComplexityRouter()
        stats = router.get_routing_stats()
        assert stats == {
            "simple": 0,
            "moderate": 0,
            "complex": 0,
            "entity_anchored": 0,
            "typed_entity_recent": 0,
            "llm_fallback": 0,
        }


@pytest.mark.unit
class TestSimpleQueryRouting:
    """Tests for SIMPLE query classification."""

    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        """Create a router for testing."""
        return QueryComplexityRouter()

    @pytest.mark.asyncio
    async def test_simple_what_is(self, router: QueryComplexityRouter) -> None:
        """Test 'what is' queries without a named entity are classified as SIMPLE.

        Queries with a named entity (e.g. 'What is Python?') now route to
        ENTITY_ANCHORED instead — see test_router_entity_anchored.py.
        """
        decision = await router.route("what is the weather")
        assert decision.complexity == QueryComplexity.SIMPLE
        assert decision.use_graph is False
        assert decision.graph_depth == 0

    @pytest.mark.asyncio
    async def test_simple_who_is(self, router: QueryComplexityRouter) -> None:
        """Test 'who is' queries are classified as SIMPLE."""
        decision = await router.route("Who is the CEO?")
        assert decision.complexity == QueryComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_simple_define(self, router: QueryComplexityRouter) -> None:
        """Test 'define' queries are classified as SIMPLE."""
        decision = await router.route("Define machine learning")
        assert decision.complexity == QueryComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_simple_short_factual(self, router: QueryComplexityRouter) -> None:
        """Test short factual queries are classified as SIMPLE."""
        decision = await router.route("What is the date?")
        assert decision.complexity == QueryComplexity.SIMPLE


@pytest.mark.unit
class TestModerateQueryRouting:
    """Tests for MODERATE query classification."""

    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        """Create a router for testing."""
        return QueryComplexityRouter()

    @pytest.mark.asyncio
    async def test_moderate_relationship_query(self, router: QueryComplexityRouter) -> None:
        """Test queries with relationship keywords are at least MODERATE."""
        decision = await router.route("How is the database connected to the API?")
        assert decision.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)
        assert decision.use_graph is True

    @pytest.mark.asyncio
    async def test_moderate_aggregation(self, router: QueryComplexityRouter) -> None:
        """Test aggregation queries are at least MODERATE."""
        decision = await router.route(
            "Show me every project that belongs to the Engineering and Infrastructure departments"
        )
        assert decision.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    @pytest.mark.asyncio
    async def test_moderate_temporal(self, router: QueryComplexityRouter) -> None:
        """Test temporal queries get at least MODERATE routing."""
        decision = await router.route("Show me the timeline of changes that happened before the database migration")
        assert decision.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    @pytest.mark.asyncio
    async def test_moderate_hierarchical(self, router: QueryComplexityRouter) -> None:
        """Test hierarchical queries are at least MODERATE."""
        decision = await router.route("Which services belong to the infrastructure team and what subcategories exist?")
        assert decision.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)


@pytest.mark.unit
class TestComplexQueryRouting:
    """Tests for COMPLEX query classification."""

    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        """Create a router for testing."""
        return QueryComplexityRouter()

    @pytest.mark.asyncio
    async def test_complex_comparison(self, router: QueryComplexityRouter) -> None:
        """Test comparison queries are classified as COMPLEX."""
        decision = await router.route(
            "Compare the advantages and disadvantages of PostgreSQL versus MongoDB for our use case"
        )
        assert decision.complexity == QueryComplexity.COMPLEX
        assert decision.use_graph is True

    @pytest.mark.asyncio
    async def test_complex_multi_hop(self, router: QueryComplexityRouter) -> None:
        """Test multi-hop queries are classified as COMPLEX."""
        decision = await router.route(
            "How does the auth service indirectly affect the billing system through the API gateway?"
        )
        assert decision.complexity == QueryComplexity.COMPLEX

    @pytest.mark.asyncio
    async def test_complex_causal_reasoning(self, router: QueryComplexityRouter) -> None:
        """Test causal reasoning queries are classified as COMPLEX."""
        decision = await router.route(
            "Why did the deployment failure cause a cascade of impacts across dependent services?"
        )
        assert decision.complexity == QueryComplexity.COMPLEX

    @pytest.mark.asyncio
    async def test_complex_multi_entity_comparison(self, router: QueryComplexityRouter) -> None:
        """Test multi-entity comparison queries are COMPLEX."""
        decision = await router.route(
            "What is the difference between Redis and Memcached for caching, and how do they compare to Hazelcast?"
        )
        assert decision.complexity == QueryComplexity.COMPLEX


@pytest.mark.unit
class TestEdgeCaseRouting:
    """Tests for edge cases in query routing."""

    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        """Create a router for testing."""
        return QueryComplexityRouter()

    @pytest.mark.asyncio
    async def test_empty_query(self, router: QueryComplexityRouter) -> None:
        """Test empty query gets a valid routing decision."""
        decision = await router.route("")
        assert isinstance(decision, RoutingDecision)
        assert decision.complexity in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    @pytest.mark.asyncio
    async def test_single_word(self, router: QueryComplexityRouter) -> None:
        """Test single word query is classified as SIMPLE."""
        decision = await router.route("hello")
        assert decision.complexity == QueryComplexity.SIMPLE

    @pytest.mark.asyncio
    async def test_very_long_query(self, router: QueryComplexityRouter) -> None:
        """Test very long query gets appropriate routing."""
        long_query = "How does " + " and ".join([f"component_{i}" for i in range(30)]) + " work together?"
        decision = await router.route(long_query)
        # Long queries with many entities should be at least MODERATE
        assert decision.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    @pytest.mark.asyncio
    async def test_routing_disabled(self) -> None:
        """Test that disabled routing returns MODERATE by default."""
        config = RouterConfig(enabled=False)
        router = QueryComplexityRouter(config)
        decision = await router.route("What is Python?")
        assert decision.complexity == QueryComplexity.MODERATE
        assert decision.confidence == 1.0

    @pytest.mark.asyncio
    async def test_query_with_entities(self, router: QueryComplexityRouter) -> None:
        """Test that queries with entity mentions are detected."""
        decision = await router.route("How is Google related to Alphabet and YouTube?")
        # Multiple capitalized entities → higher complexity
        assert decision.complexity in (QueryComplexity.MODERATE, QueryComplexity.COMPLEX)

    @pytest.mark.asyncio
    async def test_query_with_quoted_terms(self, router: QueryComplexityRouter) -> None:
        """Test that quoted terms are counted as entities."""
        decision = await router.route('Find "machine learning" and "deep learning" documents')
        assert isinstance(decision, RoutingDecision)


@pytest.mark.unit
class TestAdaptiveDepth:
    """Tests for adaptive depth computation."""

    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        """Create a router for testing."""
        return QueryComplexityRouter()

    def test_many_entities_shallow_depth(self, router: QueryComplexityRouter) -> None:
        """Test that many entry entities reduce depth."""
        depth = router.compute_adaptive_depth(entry_entity_count=15, base_depth=3)
        assert depth <= 1

    def test_few_entities_deeper_depth(self, router: QueryComplexityRouter) -> None:
        """Test that few entry entities increase depth."""
        depth = router.compute_adaptive_depth(entry_entity_count=1, base_depth=2)
        assert depth > 2

    def test_moderate_entities_same_depth(self, router: QueryComplexityRouter) -> None:
        """Test that moderate entity count preserves base depth."""
        depth = router.compute_adaptive_depth(entry_entity_count=5, base_depth=2)
        assert depth == 2

    def test_adaptive_disabled(self) -> None:
        """Test that disabled adaptive depth returns base depth."""
        config = RouterConfig(adaptive_depth_enabled=False)
        router = QueryComplexityRouter(config)
        depth = router.compute_adaptive_depth(entry_entity_count=100, base_depth=3)
        assert depth == 3


@pytest.mark.unit
class TestRoutingStats:
    """Tests for routing statistics."""

    @pytest.mark.asyncio
    async def test_stats_tracked(self) -> None:
        """Test that routing decisions update stats."""
        router = QueryComplexityRouter()
        await router.route("What is Python?")
        await router.route("Compare X versus Y with advantages and disadvantages")

        stats = router.get_routing_stats()
        total = stats["simple"] + stats["moderate"] + stats["complex"] + stats["entity_anchored"]
        assert total == 2

    @pytest.mark.asyncio
    async def test_reset_stats(self) -> None:
        """Test resetting routing stats."""
        router = QueryComplexityRouter()
        await router.route("What is Python?")

        router.reset_routing_stats()
        stats = router.get_routing_stats()
        assert all(v == 0 for v in stats.values())

    def test_stats_are_copy(self) -> None:
        """Test that get_routing_stats returns a copy."""
        router = QueryComplexityRouter()
        stats1 = router.get_routing_stats()
        stats1["simple"] = 999
        stats2 = router.get_routing_stats()
        assert stats2["simple"] == 0


@pytest.mark.unit
class TestCountPotentialEntities:
    """Tests for _count_potential_entities helper."""

    @pytest.fixture
    def router(self) -> QueryComplexityRouter:
        """Create a router for testing."""
        return QueryComplexityRouter()

    def test_capitalized_words(self, router: QueryComplexityRouter) -> None:
        """Test capitalized words are counted as entities."""
        count = router._count_potential_entities("Talk about Google and Microsoft")
        assert count >= 2

    def test_quoted_strings(self, router: QueryComplexityRouter) -> None:
        """Test quoted strings are counted."""
        count = router._count_potential_entities('Find "machine learning" documents')
        assert count >= 1

    def test_snake_case(self, router: QueryComplexityRouter) -> None:
        """Test snake_case identifiers are counted."""
        count = router._count_potential_entities("Check the user_profile table")
        assert count >= 1

    def test_no_entities(self, router: QueryComplexityRouter) -> None:
        """Test query with no entity mentions."""
        count = router._count_potential_entities("what is this about?")
        assert count == 0

    def test_mentions_and_tags(self, router: QueryComplexityRouter) -> None:
        """Test @mentions and #tags are counted."""
        count = router._count_potential_entities("Ask @alice about #project")
        assert count >= 2

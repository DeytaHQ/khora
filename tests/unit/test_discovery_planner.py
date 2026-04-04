"""Unit tests for discovery LLM planner.

All LLM calls are mocked — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from khora.discovery.planner import DiscoveryPlanner, FetchStrategy, QueryPlan
from khora.discovery.state import DiscoveredSource, SourceType

# ---------------------------------------------------------------------------
# QueryPlan
# ---------------------------------------------------------------------------


class TestQueryPlan:
    def test_defaults(self) -> None:
        plan = QueryPlan()
        assert plan.domain == ""
        assert plan.search_queries == []

    def test_fields(self) -> None:
        plan = QueryPlan(
            domain="wine",
            description="European wine datasets",
            search_queries=["wine CSV", "wine API"],
            preferred_formats=["csv", "json"],
        )
        assert plan.domain == "wine"
        assert len(plan.search_queries) == 2


# ---------------------------------------------------------------------------
# FetchStrategy
# ---------------------------------------------------------------------------


class TestFetchStrategy:
    def test_defaults(self) -> None:
        fs = FetchStrategy(method="direct_download")
        assert fs.script is None
        assert fs.params == {}


# ---------------------------------------------------------------------------
# DiscoveryPlanner.formulate_queries
# ---------------------------------------------------------------------------


class TestFormulateQueries:
    @pytest.mark.asyncio
    async def test_returns_query_plan(self) -> None:
        planner = DiscoveryPlanner(budget_usd=1.0)
        mock_result = {
            "domain": "wine",
            "description": "European wine quality datasets",
            "search_queries": [
                "wine quality dataset CSV download",
                "wine data API free",
            ],
            "preferred_formats": ["csv", "json"],
        }

        with patch.object(planner._planning_llm, "complete", new_callable=AsyncMock, return_value=mock_result):
            plan = await planner.formulate_queries("I need European wine data")

        assert plan.domain == "wine"
        assert len(plan.search_queries) == 2
        assert "wine" in plan.search_queries[0].lower()

    @pytest.mark.asyncio
    async def test_includes_previous_queries(self) -> None:
        planner = DiscoveryPlanner()
        mock_result = {
            "domain": "wine",
            "description": "test",
            "search_queries": ["new query"],
        }

        with patch.object(planner._planning_llm, "complete", new_callable=AsyncMock, return_value=mock_result) as mock:
            await planner.formulate_queries("wine data", previous_queries=["old query"])

        call_args = mock.call_args
        assert "old query" in call_args.kwargs.get("user", "") or "old query" in call_args.args[1]

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        planner = DiscoveryPlanner()

        with patch.object(
            planner._planning_llm, "complete", new_callable=AsyncMock, side_effect=Exception("API error")
        ):
            plan = await planner.formulate_queries("wine data")

        # Should fall back to using the intent as the query
        assert plan.search_queries == ["wine data"]


# ---------------------------------------------------------------------------
# DiscoveryPlanner.classify_sources
# ---------------------------------------------------------------------------


class TestClassifySources:
    @pytest.mark.asyncio
    async def test_classifies_and_ranks(self) -> None:
        planner = DiscoveryPlanner()
        mock_result = {
            "sources": [
                {
                    "url": "https://example.com/data.csv",
                    "title": "Wine Dataset",
                    "source_type": "csv",
                    "access_method": "direct_download",
                    "relevance": 0.9,
                    "requires_auth": False,
                    "description": "UCI wine quality dataset",
                },
                {
                    "url": "https://api.example.com/wine",
                    "title": "Wine API",
                    "source_type": "api",
                    "access_method": "api_call",
                    "relevance": 0.6,
                    "requires_auth": True,
                    "description": "REST API for wine data",
                },
            ]
        }

        with patch.object(planner._planning_llm, "complete", new_callable=AsyncMock, return_value=mock_result):
            sources = await planner.classify_sources(
                "wine",
                ["https://example.com/data.csv", "https://api.example.com/wine"],
            )

        assert len(sources) == 2
        # Should be sorted by relevance (0.9 first, 0.6 second)
        assert sources[0].relevance_score == 0.9
        assert sources[0].source_type == SourceType.CSV
        assert sources[1].source_type == SourceType.API
        assert sources[1].requires_auth is True

    @pytest.mark.asyncio
    async def test_empty_citations(self) -> None:
        planner = DiscoveryPlanner()
        sources = await planner.classify_sources("wine", [])
        assert sources == []

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        planner = DiscoveryPlanner()

        with patch.object(
            planner._planning_llm, "complete", new_callable=AsyncMock, side_effect=Exception("API error")
        ):
            sources = await planner.classify_sources("wine", ["https://a.com", "https://b.com"])

        # Should fall back to raw citations with default scores
        assert len(sources) == 2
        assert all(s.relevance_score == 0.5 for s in sources)

    @pytest.mark.asyncio
    async def test_unknown_source_type(self) -> None:
        planner = DiscoveryPlanner()
        mock_result = {
            "sources": [
                {
                    "url": "https://example.com",
                    "title": "Unknown",
                    "source_type": "invalid_type",
                    "relevance": 0.5,
                }
            ]
        }

        with patch.object(planner._planning_llm, "complete", new_callable=AsyncMock, return_value=mock_result):
            sources = await planner.classify_sources("test", ["https://example.com"])

        assert sources[0].source_type == SourceType.OTHER


# ---------------------------------------------------------------------------
# DiscoveryPlanner.plan_fetch_strategy
# ---------------------------------------------------------------------------


class TestPlanFetchStrategy:
    def test_csv_direct_download(self) -> None:
        planner = DiscoveryPlanner()
        src = DiscoveredSource(
            url="https://example.com/data.csv",
            title="Test",
            source_type=SourceType.CSV,
            access_method="direct_download",
        )
        strategy = planner.plan_fetch_strategy(src)
        assert strategy.method == "direct_download"

    def test_api_generates_script(self) -> None:
        planner = DiscoveryPlanner()
        src = DiscoveredSource(
            url="https://api.example.com/v1",
            title="Test",
            source_type=SourceType.API,
        )
        strategy = planner.plan_fetch_strategy(src)
        assert strategy.method == "generated_script"

    def test_webpage_uses_firecrawl(self) -> None:
        planner = DiscoveryPlanner()
        src = DiscoveredSource(
            url="https://example.com/page",
            title="Test",
            source_type=SourceType.WEBPAGE,
        )
        strategy = planner.plan_fetch_strategy(src, has_firecrawl=True)
        assert strategy.method == "firecrawl_scrape"

    def test_webpage_fallback_without_firecrawl(self) -> None:
        planner = DiscoveryPlanner()
        src = DiscoveredSource(
            url="https://example.com/page",
            title="Test",
            source_type=SourceType.WEBPAGE,
        )
        strategy = planner.plan_fetch_strategy(src, has_firecrawl=False)
        assert strategy.method == "direct_download"

    def test_repo_generates_script(self) -> None:
        planner = DiscoveryPlanner()
        src = DiscoveredSource(
            url="https://github.com/org/repo",
            title="Test",
            source_type=SourceType.REPO,
        )
        strategy = planner.plan_fetch_strategy(src)
        assert strategy.method == "generated_script"


# ---------------------------------------------------------------------------
# DiscoveryPlanner.generate_fetch_script
# ---------------------------------------------------------------------------


class TestGenerateFetchScript:
    @pytest.mark.asyncio
    async def test_returns_script(self) -> None:
        planner = DiscoveryPlanner()

        with patch.object(
            planner._codegen_llm,
            "complete_raw",
            new_callable=AsyncMock,
            return_value='```python\nimport httpx\nprint("hello")\n```',
        ):
            script = await planner.generate_fetch_script(
                DiscoveredSource(url="https://api.example.com", title="Test", source_type=SourceType.API),
                "/tmp/output",
            )

        assert "httpx" in script

    @pytest.mark.asyncio
    async def test_usage_tracking(self) -> None:
        planner = DiscoveryPlanner(budget_usd=1.0)
        assert planner.cost_usd == 0.0
        assert planner.usage_summary["calls"] == 0

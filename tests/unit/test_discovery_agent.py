"""Unit tests for the DiscoveryAgent state machine.

Uses mock UI, planner, and clients to test phase transitions
without any real API calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.discovery.agent import DiscoveryAgent
from khora.discovery.clients.firecrawl import FirecrawlClient, FirecrawlScrapeResult
from khora.discovery.clients.perplexity import PerplexityClient, PerplexitySearchResponse
from khora.discovery.planner import DiscoveryPlanner, FetchStrategy, QueryPlan
from khora.discovery.state import AgentPhase, DiscoveredSource, SessionState, SourceType
from khora.discovery.ui import DiscoveryUI


def _make_mock_ui() -> DiscoveryUI:
    """Create a mock DiscoveryUI with all methods as MagicMock/AsyncMock."""
    ui = MagicMock(spec=DiscoveryUI)
    # Async methods need AsyncMock
    ui.prompt_intent = AsyncMock(return_value="wine datasets")
    ui.prompt_url = AsyncMock(return_value="")
    ui.prompt_source_selection = AsyncMock(return_value=[0])
    ui.prompt_review_action = AsyncMock(return_value="accept")
    # Sync methods that return context managers
    ui.show_searching = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    ui.show_fetching = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    return ui


def _make_mock_planner() -> DiscoveryPlanner:
    """Create a mock planner."""
    planner = MagicMock(spec=DiscoveryPlanner)
    planner.cost_usd = 0.0
    planner.formulate_queries = AsyncMock(
        return_value=QueryPlan(
            domain="wine",
            description="wine datasets",
            search_queries=["wine dataset CSV"],
        )
    )
    planner.classify_sources = AsyncMock(
        return_value=[
            DiscoveredSource(
                url="https://example.com/wine.csv",
                title="Wine Dataset",
                source_type=SourceType.CSV,
                relevance_score=0.9,
                access_method="direct_download",
            )
        ]
    )
    planner.plan_fetch_strategy = MagicMock(return_value=FetchStrategy(method="firecrawl_scrape"))
    planner.generate_fetch_script = AsyncMock(return_value="print('hello')")
    return planner


def _make_mock_perplexity() -> PerplexityClient:
    """Create a mock Perplexity client."""
    client = MagicMock(spec=PerplexityClient)
    client.search = AsyncMock(
        return_value=PerplexitySearchResponse(
            answer="Here are some wine datasets...",
            citations=["https://example.com/wine.csv"],
            usage={"input_tokens": 100, "output_tokens": 200},
        )
    )
    return client


def _make_mock_firecrawl() -> FirecrawlClient:
    """Create a mock Firecrawl client."""
    client = MagicMock(spec=FirecrawlClient)
    client.scrape = AsyncMock(
        return_value=FirecrawlScrapeResult(
            markdown="# Wine Data\nSome wine data content...",
            metadata={"title": "Wine"},
            success=True,
        )
    )
    return client


# ---------------------------------------------------------------------------
# Happy path: full flow
# ---------------------------------------------------------------------------


class TestAgentHappyPath:
    @pytest.mark.asyncio
    async def test_full_flow_gather_to_done(self, tmp_path: Path) -> None:
        """Test the complete happy path: intent → search → select → fetch → accept."""
        ui = _make_mock_ui()
        planner = _make_mock_planner()
        perplexity = _make_mock_perplexity()
        firecrawl = _make_mock_firecrawl()

        state = SessionState(max_iterations=20)
        agent = DiscoveryAgent(
            ui=ui,
            output_dir=tmp_path,
            state=state,
            planner=planner,
            perplexity=perplexity,
            firecrawl=firecrawl,
        )

        result = await agent.run()

        assert result.phase == AgentPhase.DONE
        assert len(result.successful_fetches) == 1
        assert (tmp_path / "Wine_Dataset.md").exists()
        ui.prompt_intent.assert_called_once()
        ui.prompt_source_selection.assert_called_once()
        ui.prompt_review_action.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_from_search(self, tmp_path: Path) -> None:
        """Resume a session that's already past GATHER_INTENT."""
        ui = _make_mock_ui()
        planner = _make_mock_planner()
        perplexity = _make_mock_perplexity()
        firecrawl = _make_mock_firecrawl()

        state = SessionState(
            phase=AgentPhase.SEARCH,
            user_intent="wine data",
            search_queries=["wine dataset"],
        )

        agent = DiscoveryAgent(
            ui=ui,
            output_dir=tmp_path,
            state=state,
            planner=planner,
            perplexity=perplexity,
            firecrawl=firecrawl,
        )

        result = await agent.run()
        assert result.phase == AgentPhase.DONE
        # Should NOT have prompted for intent since we started at SEARCH
        ui.prompt_intent.assert_not_called()


# ---------------------------------------------------------------------------
# Phase transition tests
# ---------------------------------------------------------------------------


class TestPhaseTransitions:
    @pytest.mark.asyncio
    async def test_quit_at_intent(self, tmp_path: Path) -> None:
        """User quits at the intent prompt."""
        ui = _make_mock_ui()
        ui.prompt_intent = AsyncMock(return_value="")  # empty = quit

        agent = DiscoveryAgent(ui=ui, output_dir=tmp_path, planner=_make_mock_planner())
        state = await agent.run()
        assert state.phase == AgentPhase.DONE

    @pytest.mark.asyncio
    async def test_search_no_results_loops_back(self, tmp_path: Path) -> None:
        """When search finds nothing, agent goes back to GATHER_INTENT."""
        ui = _make_mock_ui()
        # First call: provide intent, second call: quit
        ui.prompt_intent = AsyncMock(side_effect=["wine data", ""])

        perplexity = _make_mock_perplexity()
        perplexity.search = AsyncMock(return_value=PerplexitySearchResponse(answer="Nothing found", citations=[]))

        planner = _make_mock_planner()
        planner.classify_sources = AsyncMock(return_value=[])

        agent = DiscoveryAgent(ui=ui, output_dir=tmp_path, planner=planner, perplexity=perplexity)
        state = await agent.run()
        assert state.phase == AgentPhase.DONE
        assert ui.show_no_results.called

    @pytest.mark.asyncio
    async def test_select_search_goes_back(self, tmp_path: Path) -> None:
        """User chooses 'search' at selection → back to GATHER_INTENT."""
        ui = _make_mock_ui()
        # First intent: provide, then quit
        ui.prompt_intent = AsyncMock(side_effect=["wine data", ""])
        ui.prompt_source_selection = AsyncMock(return_value=None)  # None = search

        agent = DiscoveryAgent(
            ui=ui,
            output_dir=tmp_path,
            planner=_make_mock_planner(),
            perplexity=_make_mock_perplexity(),
        )
        state = await agent.run()
        assert state.phase == AgentPhase.DONE

    @pytest.mark.asyncio
    async def test_review_retry_refetches(self, tmp_path: Path) -> None:
        """User chooses 'retry' at review → goes back to FETCH."""
        ui = _make_mock_ui()
        # First review: retry, second review: accept
        ui.prompt_review_action = AsyncMock(side_effect=["retry", "accept"])

        state = SessionState(max_iterations=20)
        agent = DiscoveryAgent(
            ui=ui,
            output_dir=tmp_path,
            state=state,
            planner=_make_mock_planner(),
            perplexity=_make_mock_perplexity(),
            firecrawl=_make_mock_firecrawl(),
        )
        result = await agent.run()
        assert result.phase == AgentPhase.DONE
        # Firecrawl should have been called twice (once per fetch cycle)
        assert ui.show_fetch_saved.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_stops(self, tmp_path: Path) -> None:
        """Agent stops after max_iterations."""
        ui = _make_mock_ui()
        ui.prompt_intent = AsyncMock(return_value="wine data")
        ui.prompt_review_action = AsyncMock(return_value="search")  # always loop

        state = SessionState(max_iterations=3)

        agent = DiscoveryAgent(
            ui=ui,
            output_dir=tmp_path,
            state=state,
            planner=_make_mock_planner(),
            perplexity=_make_mock_perplexity(),
            firecrawl=_make_mock_firecrawl(),
        )
        result = await agent.run()
        assert result.iteration >= 3
        ui.show_max_iterations.assert_called_once()


# ---------------------------------------------------------------------------
# Fetch fallbacks
# ---------------------------------------------------------------------------


class TestFetchFallbacks:
    @pytest.mark.asyncio
    async def test_no_firecrawl_uses_direct(self, tmp_path: Path) -> None:
        """Without Firecrawl, agent falls back to direct HTTP download."""
        ui = _make_mock_ui()
        planner = _make_mock_planner()
        planner.plan_fetch_strategy = MagicMock(return_value=FetchStrategy(method="direct_download"))

        state = SessionState(
            phase=AgentPhase.SEARCH,
            user_intent="test",
            search_queries=["test"],
            max_iterations=20,
        )

        # Mock httpx response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_resp = MagicMock()
            mock_resp.text = "Hello World"
            mock_resp.content = b"Hello World"
            mock_resp.headers = {"content-type": "text/html"}
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            agent = DiscoveryAgent(
                ui=ui,
                output_dir=tmp_path,
                state=state,
                planner=planner,
                perplexity=_make_mock_perplexity(),
            )
            agent._has_firecrawl = False

            result = await agent.run()
            assert result.phase == AgentPhase.DONE
            assert len(result.successful_fetches) == 1

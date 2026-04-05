"""Tests for the EXPLORE phase and exploration suggestions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from khora.discovery.agent import DiscoveryAgent
from khora.discovery.state import AgentPhase, DiscoveredSource, FetchResult, SessionState


@pytest.mark.unit
class TestExplorePhase:
    """Test EXPLORE phase in the agent state machine."""

    @pytest.fixture
    def agent(self, tmp_path):
        ui = MagicMock()
        ui.show_exploration_depth = MagicMock()
        ui.show_exploration_suggestions = MagicMock()
        ui.show_info = MagicMock()
        ui.prompt_exploration_choice = AsyncMock(return_value=None)

        planner = MagicMock()
        planner.summarize_content = AsyncMock(return_value="Summary of data")
        planner.suggest_exploration = AsyncMock(return_value=["follow-up query 1", "follow-up query 2"])
        planner.cost_usd = 0.01

        state = SessionState(
            user_intent="European wine data",
            search_queries=["wine production data"],
            output_dir=str(tmp_path),
        )

        return DiscoveryAgent(
            ui=ui,
            output_dir=tmp_path,
            state=state,
            planner=planner,
        )

    def _make_fetch(self, tmp_path):
        """Helper: create a fake successful fetch with a temp file."""
        f = tmp_path / "test_data.txt"
        f.write_text("Some sample data about wine production in Europe.")
        src = DiscoveredSource(url="https://example.com/wine.csv", title="Wine Data")
        return FetchResult(source=src, local_path=str(f), success=True, content_type="text/plain", size_bytes=100)

    @pytest.mark.asyncio
    async def test_explore_returns_review_when_user_skips(self, agent):
        """Skipping exploration returns to REVIEW."""
        agent._ui.prompt_exploration_choice = AsyncMock(return_value=None)
        agent._state.fetched.append(self._make_fetch(agent._output_dir))

        result = await agent._handle_explore()
        assert result == AgentPhase.REVIEW

    @pytest.mark.asyncio
    async def test_explore_returns_search_with_query(self, agent):
        """Choosing a suggestion returns SEARCH with new query."""
        agent._ui.prompt_exploration_choice = AsyncMock(return_value="wine export statistics CSV")
        agent._state.fetched.append(self._make_fetch(agent._output_dir))

        result = await agent._handle_explore()
        assert result == AgentPhase.SEARCH
        assert agent._state.search_queries == ["wine export statistics CSV"]
        assert agent._state.exploration_depth == 1

    @pytest.mark.asyncio
    async def test_explore_respects_max_depth(self, agent):
        """Exceeding max exploration depth returns REVIEW."""
        agent._state.exploration_depth = 3
        agent._state.max_exploration_depth = 3

        result = await agent._handle_explore()
        assert result == AgentPhase.REVIEW
        agent._ui.show_info.assert_called()

    @pytest.mark.asyncio
    async def test_explore_returns_review_when_no_fetches(self, agent):
        """No successful fetches means nothing to analyze."""
        result = await agent._handle_explore()
        assert result == AgentPhase.REVIEW

    @pytest.mark.asyncio
    async def test_explore_increments_depth(self, agent):
        """Exploration should increment depth counter."""
        agent._ui.prompt_exploration_choice = AsyncMock(return_value="new query")
        agent._state.fetched.append(self._make_fetch(agent._output_dir))

        assert agent._state.exploration_depth == 0
        await agent._handle_explore()
        assert agent._state.exploration_depth == 1

    @pytest.mark.asyncio
    async def test_explore_custom_query(self, agent):
        """User can provide a custom query."""
        agent._ui.prompt_exploration_choice = AsyncMock(return_value="my custom search")
        agent._state.fetched.append(self._make_fetch(agent._output_dir))

        result = await agent._handle_explore()
        assert result == AgentPhase.SEARCH
        assert agent._state.search_queries == ["my custom search"]

    @pytest.mark.asyncio
    async def test_explore_updates_cost(self, agent):
        """Exploration should update total cost from planner."""
        agent._ui.prompt_exploration_choice = AsyncMock(return_value="new query")
        agent._state.fetched.append(self._make_fetch(agent._output_dir))

        await agent._handle_explore()
        assert agent._state.total_cost_usd == 0.01


@pytest.mark.unit
class TestExploreInStateEnum:
    """Test that EXPLORE is a valid AgentPhase."""

    def test_explore_exists(self):
        assert AgentPhase.EXPLORE == "explore"
        assert AgentPhase.EXPLORE.value == "explore"

    def test_explore_in_enum(self):
        phases = [p.value for p in AgentPhase]
        assert "explore" in phases


@pytest.mark.unit
class TestExplorationDepthSerialization:
    """Test exploration depth persists across save/load."""

    def test_exploration_depth_in_dict(self):
        state = SessionState(exploration_depth=2, max_exploration_depth=5)
        d = state.to_dict()
        assert d["exploration_depth"] == 2
        assert d["max_exploration_depth"] == 5

    def test_exploration_depth_from_dict(self):
        state = SessionState.from_dict({"exploration_depth": 2, "max_exploration_depth": 5})
        assert state.exploration_depth == 2
        assert state.max_exploration_depth == 5

    def test_exploration_depth_defaults(self):
        state = SessionState.from_dict({})
        assert state.exploration_depth == 0
        assert state.max_exploration_depth == 3

    def test_exploration_depth_round_trip(self, tmp_path):
        state = SessionState(exploration_depth=2, max_exploration_depth=5)
        path = tmp_path / "session.json"
        state.save(path)
        loaded = SessionState.load(path)
        assert loaded.exploration_depth == 2
        assert loaded.max_exploration_depth == 5

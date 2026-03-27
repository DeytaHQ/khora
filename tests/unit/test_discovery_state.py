"""Unit tests for discovery state models."""

from __future__ import annotations

import json
from pathlib import Path

from khora.discovery.state import (
    AgentPhase,
    DiscoveredSource,
    FetchAttempt,
    FetchMethod,
    FetchResult,
    SessionState,
    SourceStatus,
    SourceType,
)


class TestAgentPhase:
    """Test the AgentPhase state machine enum."""

    def test_all_phases_exist(self) -> None:
        phases = {p.value for p in AgentPhase}
        assert phases == {
            "gather_intent",
            "search",
            "present_results",
            "select_sources",
            "fetch",
            "review",
            "augment",
            "ingest",
            "done",
        }

    def test_string_value(self) -> None:
        assert AgentPhase.GATHER_INTENT == "gather_intent"
        assert AgentPhase.DONE == "done"


class TestSourceType:
    def test_all_types(self) -> None:
        assert len(SourceType) == 10
        assert SourceType.CSV == "csv"
        assert SourceType.API == "api"


class TestDiscoveredSource:
    def test_defaults(self) -> None:
        src = DiscoveredSource(url="https://example.com", title="Test")
        assert src.source_type == SourceType.OTHER
        assert src.status == SourceStatus.DISCOVERED
        assert src.relevance_score == 0.0
        assert src.requires_auth is False
        assert src.metadata == {}

    def test_round_trip(self) -> None:
        src = DiscoveredSource(
            url="https://example.com/data.csv",
            title="Test Dataset",
            description="A test dataset",
            source_type=SourceType.CSV,
            status=SourceStatus.SELECTED,
            relevance_score=0.85,
            access_method="direct_download",
            requires_auth=False,
            discovered_via="perplexity",
            discovery_query="economic data CSV",
        )
        data = src.to_dict()
        restored = DiscoveredSource.from_dict(data)

        assert restored.url == src.url
        assert restored.title == src.title
        assert restored.description == src.description
        assert restored.source_type == src.source_type
        assert restored.status == src.status
        assert restored.relevance_score == src.relevance_score
        assert restored.access_method == src.access_method
        assert restored.discovered_via == src.discovered_via


class TestFetchAttempt:
    def test_defaults(self) -> None:
        attempt = FetchAttempt()
        assert attempt.method == FetchMethod.DIRECT_DOWNLOAD
        assert attempt.success is False
        assert attempt.error is None
        assert attempt.timestamp  # should be set

    def test_round_trip(self) -> None:
        attempt = FetchAttempt(
            method=FetchMethod.FIRECRAWL_SCRAPE,
            success=True,
            bytes_fetched=1024,
            duration_seconds=2.5,
        )
        restored = FetchAttempt.from_dict(attempt.to_dict())
        assert restored.method == FetchMethod.FIRECRAWL_SCRAPE
        assert restored.success is True
        assert restored.bytes_fetched == 1024


class TestFetchResult:
    def test_round_trip(self) -> None:
        src = DiscoveredSource(url="https://example.com", title="Test")
        result = FetchResult(
            source=src,
            local_path="/tmp/test_data",
            content_type="text/csv",
            size_bytes=2048,
            success=True,
            attempts=[
                FetchAttempt(method=FetchMethod.DIRECT_DOWNLOAD, success=True, bytes_fetched=2048),
            ],
        )
        restored = FetchResult.from_dict(result.to_dict())
        assert restored.source.url == "https://example.com"
        assert restored.success is True
        assert len(restored.attempts) == 1
        assert restored.attempts[0].bytes_fetched == 2048


class TestSessionState:
    def test_defaults(self) -> None:
        state = SessionState()
        assert state.phase == AgentPhase.GATHER_INTENT
        assert state.user_intent == ""
        assert state.discovered == []
        assert state.selected_indices == []
        assert state.iteration == 0
        assert state.max_iterations == 5
        assert state.max_cost_usd == 2.0

    def test_selected_sources(self) -> None:
        state = SessionState()
        state.discovered = [
            DiscoveredSource(url="https://a.com", title="A"),
            DiscoveredSource(url="https://b.com", title="B"),
            DiscoveredSource(url="https://c.com", title="C"),
        ]
        state.selected_indices = [0, 2]
        selected = state.selected_sources
        assert len(selected) == 2
        assert selected[0].title == "A"
        assert selected[1].title == "C"

    def test_selected_sources_out_of_bounds(self) -> None:
        state = SessionState()
        state.discovered = [DiscoveredSource(url="https://a.com", title="A")]
        state.selected_indices = [0, 5]  # 5 is out of bounds
        assert len(state.selected_sources) == 1

    def test_successful_fetches(self) -> None:
        src = DiscoveredSource(url="https://example.com", title="Test")
        state = SessionState()
        state.fetched = [
            FetchResult(source=src, local_path="/tmp/a", success=True),
            FetchResult(source=src, local_path="/tmp/b", success=False, error="timeout"),
            FetchResult(source=src, local_path="/tmp/c", success=True),
        ]
        assert len(state.successful_fetches) == 2

    def test_round_trip_json(self) -> None:
        state = SessionState(
            phase=AgentPhase.SEARCH,
            user_intent="European wine datasets",
            search_queries=["wine dataset CSV download", "wine API free"],
            iteration=2,
            total_cost_usd=0.15,
        )
        state.discovered.append(
            DiscoveredSource(
                url="https://wine.com/data.csv",
                title="Wine Data",
                source_type=SourceType.CSV,
                relevance_score=0.9,
            )
        )
        state.selected_indices = [0]

        data = state.to_dict()
        json_str = json.dumps(data)
        restored = SessionState.from_dict(json.loads(json_str))

        assert restored.phase == AgentPhase.SEARCH
        assert restored.user_intent == "European wine datasets"
        assert len(restored.search_queries) == 2
        assert len(restored.discovered) == 1
        assert restored.discovered[0].source_type == SourceType.CSV
        assert restored.selected_indices == [0]
        assert restored.iteration == 2
        assert restored.total_cost_usd == 0.15

    def test_save_and_load(self, tmp_path: Path) -> None:
        state = SessionState(
            phase=AgentPhase.REVIEW,
            user_intent="test data",
        )
        state.discovered.append(DiscoveredSource(url="https://example.com", title="Test"))

        save_path = tmp_path / "session.json"
        state.save(save_path)

        assert save_path.exists()
        loaded = SessionState.load(save_path)
        assert loaded.phase == AgentPhase.REVIEW
        assert loaded.user_intent == "test data"
        assert len(loaded.discovered) == 1
        assert loaded.session_id == state.session_id

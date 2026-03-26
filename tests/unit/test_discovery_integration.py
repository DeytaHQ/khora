"""Integration tests for the discovery feature.

Tests cross-module interactions: agent + planner + clients + codegen +
validation working together.  All external API calls are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from khora.discovery.agent import DiscoveryAgent
from khora.discovery.clients.firecrawl import FirecrawlClient, FirecrawlScrapeResult
from khora.discovery.clients.perplexity import PerplexityClient, PerplexitySearchResponse
from khora.discovery.codegen import execute_script, render_template, validate_script
from khora.discovery.planner import DiscoveryPlanner, FetchStrategy, QueryPlan
from khora.discovery.state import (
    AgentPhase,
    DiscoveredSource,
    FetchResult,
    SessionState,
    SourceType,
)
from khora.discovery.ui import DiscoveryUI
from khora.discovery.validation import validate_batch, validate_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ui() -> DiscoveryUI:
    ui = MagicMock(spec=DiscoveryUI)
    ui.prompt_intent = AsyncMock(return_value="wine quality datasets")
    ui.prompt_url = AsyncMock(return_value="")
    ui.prompt_source_selection = AsyncMock(return_value=[0])
    ui.prompt_review_action = AsyncMock(return_value="accept")
    ui.show_searching = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    ui.show_fetching = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    return ui


# ---------------------------------------------------------------------------
# Agent → Planner → Client integration
# ---------------------------------------------------------------------------


class TestAgentPlannerIntegration:
    """Test that the agent correctly delegates to the planner for query
    formulation and source classification."""

    @pytest.mark.asyncio
    async def test_agent_uses_planner_for_queries(self, tmp_path: Path) -> None:
        """Agent should call planner.formulate_queries to generate search queries."""
        ui = _mock_ui()

        planner = MagicMock(spec=DiscoveryPlanner)
        planner.cost_usd = 0.0
        planner.formulate_queries = AsyncMock(
            return_value=QueryPlan(
                domain="wine",
                search_queries=["wine quality CSV", "wine dataset API"],
            )
        )
        planner.classify_sources = AsyncMock(
            return_value=[
                DiscoveredSource(
                    url="https://archive.ics.uci.edu/dataset/186/wine+quality",
                    title="UCI Wine Quality",
                    source_type=SourceType.DATASET,
                    relevance_score=0.95,
                )
            ]
        )
        planner.plan_fetch_strategy = MagicMock(return_value=FetchStrategy(method="firecrawl_scrape"))

        perplexity = MagicMock(spec=PerplexityClient)
        perplexity.search = AsyncMock(
            return_value=PerplexitySearchResponse(
                answer="UCI has a wine quality dataset...",
                citations=["https://archive.ics.uci.edu/dataset/186/wine+quality"],
            )
        )

        firecrawl = MagicMock(spec=FirecrawlClient)
        firecrawl.scrape = AsyncMock(
            return_value=FirecrawlScrapeResult(markdown="# Wine Quality\nData from UCI...", success=True)
        )

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

        # Verify the planner was consulted
        planner.formulate_queries.assert_called_once()
        planner.classify_sources.assert_called_once()
        assert result.phase == AgentPhase.DONE
        assert len(result.successful_fetches) == 1

    @pytest.mark.asyncio
    async def test_agent_with_no_citations(self, tmp_path: Path) -> None:
        """When Perplexity returns no citations, agent loops back to intent."""
        ui = _mock_ui()
        ui.prompt_intent = AsyncMock(side_effect=["wine data", ""])

        planner = MagicMock(spec=DiscoveryPlanner)
        planner.cost_usd = 0.0
        planner.formulate_queries = AsyncMock(return_value=QueryPlan(search_queries=["wine data"]))

        perplexity = MagicMock(spec=PerplexityClient)
        perplexity.search = AsyncMock(return_value=PerplexitySearchResponse(answer="No results found.", citations=[]))

        state = SessionState(max_iterations=20)
        agent = DiscoveryAgent(ui=ui, output_dir=tmp_path, state=state, planner=planner, perplexity=perplexity)

        result = await agent.run()
        assert result.phase == AgentPhase.DONE
        assert ui.show_no_results.called


# ---------------------------------------------------------------------------
# Codegen → Validation integration
# ---------------------------------------------------------------------------


class TestCodegenValidationIntegration:
    """Test that generated scripts are properly validated before execution."""

    def test_rendered_template_passes_validation(self) -> None:
        """A well-formed template should pass AST validation."""
        script = render_template(
            title="Test API",
            url="https://api.example.com/data",
            fetch_body=(
                'response = httpx.get("https://api.example.com/data")\n'
                "data = response.json()\n"
                'output_file = OUTPUT_DIR / "data.json"\n'
                "output_file.write_text(json.dumps(data, indent=2))\n"
                "files_written.append(str(output_file))\n"
                "total_records = len(data)\n"
            ),
        )
        violations = validate_script(script)
        assert violations == [], f"Unexpected violations: {violations}"

    def test_malicious_body_caught(self) -> None:
        """A fetch body with dangerous calls should be caught."""
        script = render_template(
            title="Evil Script",
            url="https://evil.com",
            fetch_body='import subprocess\nsubprocess.run(["rm", "-rf", "/"])\n',
        )
        violations = validate_script(script)
        assert len(violations) > 0
        assert any("subprocess" in str(v) for v in violations)

    @pytest.mark.asyncio
    async def test_execute_script_creates_files(self, tmp_path: Path) -> None:
        """A safe script should execute and create output files."""
        script = render_template(
            title="Simple Fetch",
            url="https://example.com",
            fetch_body=(
                'data = [{"name": "test", "value": 42}]\n'
                'out = OUTPUT_DIR / "output.json"\n'
                "out.write_text(json.dumps(data))\n"
                'files_written.append("output.json")\n'
                "total_records = 1\n"
            ),
        )
        violations = validate_script(script)
        assert violations == []

        result = await execute_script(script, tmp_path, timeout=30)
        assert result.success
        assert "output.json" in result.files_created
        assert (tmp_path / "output.json").exists()

        # Verify the output is valid JSON
        content = json.loads((tmp_path / "output.json").read_text())
        assert content[0]["name"] == "test"


# ---------------------------------------------------------------------------
# Fetch → Validation pipeline integration
# ---------------------------------------------------------------------------


class TestFetchValidationIntegration:
    """Test that fetched data flows through the validation pipeline."""

    def test_good_fetched_data_accepted(self, tmp_path: Path) -> None:
        """Well-structured fetched data should pass validation."""
        data_file = tmp_path / "wine_data.json"
        data = [
            {"wine": "Bordeaux", "region": "France", "rating": 92, "year": 2018},
            {"wine": "Chianti", "region": "Italy", "rating": 88, "year": 2019},
        ] * 15  # enough content to score well
        data_file.write_text(json.dumps(data, indent=2))

        result = validate_file(data_file, query="wine quality data France Italy")
        assert result.format_detected == "json"
        assert result.format_score >= 0.9
        assert result.decision in ("accept", "review")

    def test_scraped_html_flagged(self, tmp_path: Path) -> None:
        """Scraped HTML with boilerplate should score lower."""
        html_file = tmp_path / "page.html"
        boilerplate = "<p>Cookie policy and privacy policy terms of service.</p>" * 20
        lorem = "<p>Lorem ipsum dolor sit amet.</p>" * 10
        html_file.write_text(f"<html><body>{boilerplate}{lorem}</body></html>")

        result = validate_file(html_file, query="wine data")
        assert result.format_detected == "html"
        # Should have boilerplate warnings
        assert any("boilerplate" in r for r in result.reasons)

    def test_duplicate_files_rejected(self, tmp_path: Path) -> None:
        """Duplicate files in a batch should be detected and rejected."""
        content = json.dumps([{"wine": "test"} for _ in range(50)])
        f1 = tmp_path / "source1.json"
        f2 = tmp_path / "source2.json"
        f1.write_text(content)
        f2.write_text(content)

        results = validate_batch([f1, f2], query="wine")
        duplicates = [r for r in results if r.duplicate]
        assert len(duplicates) == 1


# ---------------------------------------------------------------------------
# Session state persistence integration
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_save_load_with_fetched_data(self, tmp_path: Path) -> None:
        """Session with fetched results should survive save/load cycle."""
        state = SessionState(
            phase=AgentPhase.REVIEW,
            user_intent="wine quality data",
            search_queries=["wine CSV", "wine API"],
            iteration=3,
            total_cost_usd=0.25,
        )

        src = DiscoveredSource(
            url="https://example.com/wine.csv",
            title="Wine Data",
            source_type=SourceType.CSV,
            relevance_score=0.9,
        )
        state.discovered.append(src)
        state.selected_indices = [0]
        state.fetched.append(
            FetchResult(
                source=src,
                local_path="/tmp/wine.csv",
                content_type="text/csv",
                size_bytes=10240,
                success=True,
            )
        )

        # Save
        session_file = tmp_path / "session.json"
        state.save(session_file)

        # Load
        loaded = SessionState.load(session_file)

        assert loaded.phase == AgentPhase.REVIEW
        assert loaded.user_intent == "wine quality data"
        assert len(loaded.discovered) == 1
        assert loaded.discovered[0].source_type == SourceType.CSV
        assert len(loaded.fetched) == 1
        assert loaded.fetched[0].success is True
        assert loaded.total_cost_usd == 0.25


# ---------------------------------------------------------------------------
# Codegen adversarial tests
# ---------------------------------------------------------------------------


class TestCodegenAdversarial:
    """Adversarial tests for the AST validation sandbox."""

    @pytest.mark.parametrize(
        "code,description",
        [
            ("import subprocess; subprocess.run(['ls'])", "direct subprocess"),
            ("import os; os.system('ls')", "os.system"),
            ("eval('1+1')", "eval call"),
            ("exec('print(1)')", "exec call"),
            ("__import__('os')", "__import__ call"),
            ("getattr(os, 'system')('ls')", "getattr os.system"),
            ("import shutil; shutil.rmtree('/')", "shutil.rmtree"),
            ("import socket; socket.socket()", "raw socket"),
            ("import pickle; pickle.loads(b'')", "pickle deserialization"),
            ("import ctypes; ctypes.CDLL('libc.so')", "ctypes FFI"),
            ("from importlib import import_module", "importlib"),
            ("import asyncio", "asyncio import"),
            ("import multiprocessing", "multiprocessing"),
            ("from os import environ", "os.environ access"),
        ],
    )
    def test_dangerous_patterns_blocked(self, code: str, description: str) -> None:
        """Verify that known dangerous patterns are caught by AST validation."""
        violations = validate_script(code)
        assert len(violations) > 0, f"Expected violation for: {description}"

    @pytest.mark.parametrize(
        "code,description",
        [
            ("import httpx\nresponse = httpx.get('https://example.com')", "httpx GET"),
            ("import json\ndata = json.loads('{}')", "json parsing"),
            ("from pathlib import Path\np = Path('.')", "pathlib usage"),
            ("import csv, io\nreader = csv.reader(io.StringIO('a,b'))", "csv reading"),
            ("import re\nre.findall(r'\\d+', 'abc123')", "regex"),
            ("import time\ntime.sleep(1)", "time.sleep"),
            ("from datetime import datetime\nnow = datetime.now()", "datetime"),
        ],
    )
    def test_safe_patterns_allowed(self, code: str, description: str) -> None:
        """Verify that legitimate fetch patterns are not blocked."""
        violations = validate_script(code)
        assert violations == [], f"Unexpected violation for {description}: {violations}"


# ---------------------------------------------------------------------------
# End-to-end: agent run → validation
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    @pytest.mark.asyncio
    async def test_agent_fetch_then_validate(self, tmp_path: Path) -> None:
        """Full flow: agent fetches data, then we validate the output."""
        ui = _mock_ui()

        planner = MagicMock(spec=DiscoveryPlanner)
        planner.cost_usd = 0.01
        planner.formulate_queries = AsyncMock(return_value=QueryPlan(domain="wine", search_queries=["wine data"]))
        planner.classify_sources = AsyncMock(
            return_value=[
                DiscoveredSource(
                    url="https://example.com/wine.json",
                    title="Wine JSON",
                    source_type=SourceType.JSON,
                    relevance_score=0.9,
                )
            ]
        )
        planner.plan_fetch_strategy = MagicMock(return_value=FetchStrategy(method="firecrawl_scrape"))

        perplexity = MagicMock(spec=PerplexityClient)
        perplexity.search = AsyncMock(
            return_value=PerplexitySearchResponse(
                answer="Found wine data...",
                citations=["https://example.com/wine.json"],
            )
        )

        # Firecrawl returns good structured content
        wine_content = (
            "# Wine Quality Dataset\n\n"
            "This dataset contains wine quality measurements from Portugal.\n\n"
            "## Features\n"
            "- fixed acidity\n- volatile acidity\n- citric acid\n"
            "- residual sugar\n- chlorides\n- free sulfur dioxide\n\n"
            "## Data\n"
            "| Wine | Region | Rating |\n"
            "| Bordeaux | France | 92 |\n"
            "| Chianti | Italy | 88 |\n"
        )
        firecrawl = MagicMock(spec=FirecrawlClient)
        firecrawl.scrape = AsyncMock(return_value=FirecrawlScrapeResult(markdown=wine_content, success=True))

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

        # Now validate the fetched files
        fetched_files = list(tmp_path.glob("*.md"))
        assert len(fetched_files) >= 1

        validation_results = validate_batch(fetched_files, query="wine quality data")
        assert len(validation_results) >= 1
        # The wine content should score reasonably well
        best = validation_results[0]
        assert best.format_detected == "markdown"
        assert best.quality_score > 0.3

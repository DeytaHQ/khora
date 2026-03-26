"""Unit tests for the discovery TUI renderer."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from khora.discovery.state import (
    DiscoveredSource,
    FetchResult,
    SourceStatus,
    SourceType,
)
from khora.discovery.ui import (
    DiscoveryUI,
    render_fetch_results_table,
    render_sources_table,
)


def _make_console() -> tuple[Console, StringIO]:
    """Create a Console that writes to a StringIO buffer (no ANSI codes)."""
    buf = StringIO()
    return Console(file=buf, no_color=True, highlight=False, width=120), buf


# ---------------------------------------------------------------------------
# render_sources_table
# ---------------------------------------------------------------------------


class TestRenderSourcesTable:
    def test_renders_all_sources(self) -> None:
        sources = [
            DiscoveredSource(
                url="https://example.com/data.csv",
                title="Test Dataset",
                source_type=SourceType.CSV,
                relevance_score=0.85,
            ),
            DiscoveredSource(
                url="https://api.example.com/v1",
                title="Test API",
                source_type=SourceType.API,
                relevance_score=0.3,
            ),
        ]
        table = render_sources_table(sources)
        assert "DISCOVERED SOURCES" in (table.title or "")
        assert table.row_count == 2

    def test_empty_sources(self) -> None:
        table = render_sources_table([])
        assert table.row_count == 0

    def test_status_display(self) -> None:
        sources = [
            DiscoveredSource(
                url="https://a.com",
                title="A",
                status=SourceStatus.SELECTED,
            ),
            DiscoveredSource(
                url="https://b.com",
                title="B",
                status=SourceStatus.FAILED,
            ),
        ]
        table = render_sources_table(sources)
        assert table.row_count == 2


# ---------------------------------------------------------------------------
# render_fetch_results_table
# ---------------------------------------------------------------------------


class TestRenderFetchResultsTable:
    def test_renders_results(self) -> None:
        src = DiscoveredSource(url="https://example.com", title="Test")
        results = [
            FetchResult(source=src, local_path="/tmp/test.md", success=True, size_bytes=1024),
            FetchResult(source=src, local_path="", success=False, error="timeout"),
        ]
        table = render_fetch_results_table(results)
        assert "FETCH RESULTS" in (table.title or "")
        assert table.row_count == 2


# ---------------------------------------------------------------------------
# DiscoveryUI output methods
# ---------------------------------------------------------------------------


class TestDiscoveryUIOutput:
    def test_show_welcome(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_welcome(["Perplexity (search)", "Firecrawl (scrape)"], "/tmp/data")
        output = buf.getvalue()
        assert "SOURCE DISCOVERY" in output
        assert "Perplexity" in output
        assert "/tmp/data" in output

    def test_show_sources(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        sources = [
            DiscoveredSource(url="https://a.com", title="A", relevance_score=0.9),
        ]
        ui.show_sources(sources)
        output = buf.getvalue()
        assert "1 source" in output

    def test_show_review_summary(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        src = DiscoveredSource(url="https://a.com", title="A")
        results = [
            FetchResult(source=src, local_path="/tmp/a.md", success=True, size_bytes=500),
            FetchResult(source=src, local_path="", success=False, error="404"),
        ]
        ui.show_review_summary(results)
        output = buf.getvalue()
        assert "1 ok" in output
        assert "500" in output

    def test_show_data_preview(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_data_preview("/tmp/test.md", "Hello world content here", max_chars=10)
        output = buf.getvalue()
        assert "test.md" in output
        assert "..." in output

    def test_show_done_with_paths(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_done(["/tmp/a.md", "/tmp/b.md"], "/tmp/data")
        output = buf.getvalue()
        assert "2 file(s)" in output
        assert "khora ontology construct" in output

    def test_show_done_no_paths(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_done([], "/tmp/data")
        output = buf.getvalue()
        assert "no data fetched" in output

    def test_show_no_keys(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_no_keys()
        output = buf.getvalue()
        assert "PERPLEXITY_API_KEY" in output

    def test_show_cost(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_cost(0.42)
        output = buf.getvalue()
        assert "$0.42" in output

    def test_show_search_failed(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_search_failed("connection refused")
        output = buf.getvalue()
        assert "connection refused" in output

    def test_show_fetch_saved(self) -> None:
        console, buf = _make_console()
        ui = DiscoveryUI(console)
        ui.show_fetch_saved("test.md", 1500)
        output = buf.getvalue()
        assert "test.md" in output
        assert "1,500" in output


# ---------------------------------------------------------------------------
# DiscoveryUI prompt methods (async)
# ---------------------------------------------------------------------------


class TestDiscoveryUIPrompts:
    @pytest.mark.asyncio
    async def test_prompt_intent_returns_text(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="wine datasets"):
            result = await ui.prompt_intent()
        assert result == "wine datasets"

    @pytest.mark.asyncio
    async def test_prompt_intent_quit(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="quit"):
            result = await ui.prompt_intent()
        assert result == ""

    @pytest.mark.asyncio
    async def test_prompt_source_selection_all(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="all"):
            result = await ui.prompt_source_selection(5)
        assert result == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_prompt_source_selection_specific(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="1, 3"):
            result = await ui.prompt_source_selection(5)
        assert result == [0, 2]

    @pytest.mark.asyncio
    async def test_prompt_source_selection_search(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="search"):
            result = await ui.prompt_source_selection(5)
        assert result is None

    @pytest.mark.asyncio
    async def test_prompt_review_action(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="accept"):
            result = await ui.prompt_review_action()
        assert result == "accept"

    @pytest.mark.asyncio
    async def test_prompt_url(self) -> None:
        ui = DiscoveryUI()
        with patch("khora.discovery.ui.Prompt.ask", return_value="https://example.com"):
            result = await ui.prompt_url()
        assert result == "https://example.com"

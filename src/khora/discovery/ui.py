"""Rich TUI renderer for the interactive discovery agent.

Visual language: monochrome base + orange accent (TE-inspired).
Provides the ``DiscoveryUI`` class that encapsulates all terminal
rendering and user prompting, keeping the agent logic free of
presentation concerns.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from .state import DiscoveredSource, FetchResult

# -- Palette (shared with ontology TUI) ------------------------------------

_ACCENT = "#FF6600"
_ACCENT2 = "#FF9933"
_MUTED = "bright_black"
_DIM = "dim"
_SUCCESS = "#33FF66"
_WARN = "#FFCC00"
_ERR = "#FF3333"


class DiscoveryUI:
    """Rich-based TUI for the discovery agent.

    All output goes through a Rich ``Console`` instance.  All prompts
    use ``rich.prompt`` and are wrapped in ``asyncio.to_thread`` so the
    event loop stays free during blocking input.

    The class is designed to be replaceable with a mock for testing --
    every method that interacts with the user has a clear return type.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    @property
    def console(self) -> Console:
        return self._console

    # ------------------------------------------------------------------
    # Welcome / headers
    # ------------------------------------------------------------------

    def show_welcome(
        self,
        services: list[str],
        output_dir: str,
    ) -> None:
        """Show the welcome panel with available services."""
        svc_text = ", ".join(f"[{_ACCENT}]{s}[/]" for s in services)
        self._console.print(
            Panel(
                f"[bold {_ACCENT}]SOURCE DISCOVERY[/]\n"
                f"[{_MUTED}]{'─' * 40}[/]\n"
                f"  [{_DIM}]services[/]  {svc_text}\n"
                f"  [{_DIM}]output  [/]  [{_ACCENT2}]{output_dir}[/]\n"
                f"[{_MUTED}]{'─' * 40}[/]\n"
                f"[{_DIM}]Describe the data you need. Type [bold {_ACCENT}]quit[/] to exit.[/]",
                border_style=_MUTED,
                padding=(1, 2),
            )
        )

    def show_rule(self, title: str) -> None:
        """Print a horizontal rule with a title."""
        self._console.print(Rule(f"[bold {_ACCENT}]{title}[/]", style=_MUTED))

    # ------------------------------------------------------------------
    # Intent gathering
    # ------------------------------------------------------------------

    async def prompt_intent(self) -> str:
        """Ask the user what data they need."""
        raw = await asyncio.to_thread(Prompt.ask, f"\n[bold {_ACCENT}]>[/] [bold]What data do you need?[/]")
        if raw.strip().lower() in ("quit", "exit", "q"):
            return ""
        return raw.strip()

    async def prompt_url(self) -> str:
        """Ask the user for a URL directly."""
        return await asyncio.to_thread(Prompt.ask, f"  [{_ACCENT}]>[/] URL", default="")

    # ------------------------------------------------------------------
    # Search phase
    # ------------------------------------------------------------------

    def show_searching(self, query: str) -> Any:
        """Return a console status context manager for the search phase."""
        return self._console.status(f"[{_ACCENT}]searching:[/] {query[:60]}...")

    def show_search_results(self, answer: str, citation_count: int) -> None:
        """Display the Perplexity search answer."""
        if answer:
            label = f"[bold {_ACCENT}]SEARCH RESULTS[/] [{_DIM}]// {citation_count} source{'s' if citation_count != 1 else ''}[/]"
            self._console.print(
                Panel(
                    answer[:2000],
                    title=label,
                    border_style=_MUTED,
                    padding=(1, 2),
                )
            )

    def show_search_failed(self, error: str) -> None:
        self._console.print(f"  [{_ERR}]error:[/] {error}")

    def show_no_results(self) -> None:
        self._console.print(f"  [{_WARN}]no sources found.[/] [{_DIM}]Try a different query.[/]")

    def show_no_perplexity(self) -> None:
        self._console.print(f"  [{_WARN}]perplexity unavailable.[/] [{_DIM}]Provide URLs directly.[/]")

    # ------------------------------------------------------------------
    # Source presentation
    # ------------------------------------------------------------------

    def show_sources(self, sources: list[DiscoveredSource]) -> None:
        """Render discovered sources as a numbered table."""
        self._console.print(f"\n  [{_ACCENT}]{len(sources)}[/] source(s) discovered")
        self._console.print(render_sources_table(sources))

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------

    async def prompt_source_selection(self, count: int) -> list[int] | None:
        """Ask the user which sources to fetch."""
        raw = await asyncio.to_thread(
            Prompt.ask,
            f"[{_ACCENT}]>[/] Select sources [{_DIM}](numbers, 'all', or 'search')[/]",
            default="all",
        )
        cmd = raw.strip().lower()
        if cmd in ("search", "back", "new"):
            return None
        if cmd == "all" or "all" in cmd.split():
            return list(range(count))
        try:
            return [int(x.strip()) - 1 for x in raw.split(",")]
        except ValueError:
            self._console.print(f"  [{_ERR}]invalid selection.[/] [{_DIM}]Use numbers separated by commas.[/]")
            return await self.prompt_source_selection(count)

    # ------------------------------------------------------------------
    # Fetch phase
    # ------------------------------------------------------------------

    def show_fetching(self, url: str) -> Any:
        """Return a console status context manager for fetching."""
        return self._console.status(f"[{_ACCENT}]fetching:[/] {url}...")

    def show_fetch_start(self, url: str) -> None:
        self._console.print(f"\n  [{_MUTED}]>[/] [{_ACCENT2}]{url}[/]")

    def show_fetch_saved(self, filename: str, size: int, unit: str = "chars") -> None:
        self._console.print(f"    [{_SUCCESS}]saved[/] {filename} [{_DIM}]({size:,} {unit})[/]")

    def show_fetch_empty(self, url: str) -> None:
        self._console.print(f"    [{_WARN}]empty[/] [{_DIM}]{url}[/]")

    def show_fetch_failed(self, error: str) -> None:
        self._console.print(f"    [{_ERR}]fail[/]  {error}")

    def show_no_selection(self) -> None:
        self._console.print(f"  [{_WARN}]no sources selected.[/]")

    # ------------------------------------------------------------------
    # Review phase
    # ------------------------------------------------------------------

    def show_review_summary(self, fetched: list[FetchResult]) -> None:
        """Show a summary of fetch results for review."""
        successful = [f for f in fetched if f.success]
        failed = [f for f in fetched if not f.success]

        self._console.print(f"\n[bold {_ACCENT}]FETCH SUMMARY[/]")
        self._console.print(f"[{_MUTED}]{'─' * 50}[/]")
        self._console.print(
            f"  [{_SUCCESS}]{len(successful)}[/] ok  [{_ERR}]{len(failed)}[/] failed  [{_DIM}]/ {len(fetched)} total[/]"
        )
        self._console.print(f"[{_MUTED}]{'─' * 50}[/]")

        for f in successful:
            name = Path(f.local_path).name if f.local_path else "?"
            self._console.print(f"  [{_SUCCESS}]OK[/]   {name} [{_DIM}]({f.size_bytes:,} bytes)[/]")
        for f in failed:
            self._console.print(f"  [{_ERR}]FAIL[/] {f.source.url}: [{_DIM}]{f.error or 'unknown'}[/]")

    def show_data_preview(self, path: str, content: str, max_chars: int = 500) -> None:
        """Show a preview of fetched data content."""
        preview = content[:max_chars]
        if len(content) > max_chars:
            preview += f"\n[{_DIM}]... ({len(content) - max_chars:,} more chars)[/]"
        self._console.print(
            Panel(
                preview,
                title=f"[{_ACCENT}]{Path(path).name}[/]",
                border_style=_MUTED,
                padding=(0, 2),
            )
        )

    async def prompt_review_action(self) -> Literal["accept", "retry", "search", "quit"]:
        """Ask the user what to do after reviewing fetched data."""
        choice = await asyncio.to_thread(
            Prompt.ask,
            f"[{_ACCENT}]>[/] Action [{_DIM}](accept/retry/search/quit)[/]",
            choices=["accept", "retry", "search", "quit"],
            default="accept",
        )
        return choice  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Completion / errors
    # ------------------------------------------------------------------

    def show_done(self, paths: list[str], output_dir: str) -> None:
        """Show completion message with next steps."""
        if paths:
            self._console.print(
                Panel(
                    f"[{_SUCCESS}]{len(paths)} file(s)[/] fetched to [{_ACCENT2}]{output_dir}[/]\n\n"
                    f"[{_DIM}]next step:[/]\n"
                    f"  [{_ACCENT}]khora ontology construct --source {output_dir}[/]",
                    title=f"[bold {_ACCENT}]DONE[/]",
                    border_style=_MUTED,
                    padding=(1, 2),
                )
            )
        else:
            self._console.print(f"[{_DIM}]no data fetched.[/]")

    def show_session_saved(self, path: str) -> None:
        self._console.print(f"[{_DIM}]session saved: {path}[/]")

    def show_max_iterations(self) -> None:
        self._console.print(f"[{_WARN}]max iterations reached.[/]")

    def show_no_keys(self) -> None:
        self._console.print(
            f"[{_ERR}]no discovery API keys found.[/]\n"
            f"[{_DIM}]Set PERPLEXITY_API_KEY and/or FIRECRAWL_API_KEY to enable source discovery.[/]"
        )

    def show_error(self, message: str) -> None:
        self._console.print(f"[{_ERR}]{message}[/]")

    def show_info(self, message: str) -> None:
        self._console.print(message)

    def show_cost(self, cost_usd: float) -> None:
        """Show running cost estimate."""
        self._console.print(f"[{_DIM}]cost: ~${cost_usd:.2f}[/]")


# ---------------------------------------------------------------------------
# Standalone rendering functions
# ---------------------------------------------------------------------------


def render_sources_table(sources: list[DiscoveredSource]) -> Table:
    """Build a Rich Table showing discovered datasources."""
    table = Table(
        title=f"[bold {_ACCENT}]DISCOVERED SOURCES[/]",
        show_lines=True,
        expand=True,
        title_style=f"bold {_ACCENT}",
        border_style=_MUTED,
    )
    table.add_column("#", style=_DIM, width=4, justify="right")
    table.add_column("Title", style="bold white", min_width=20)
    table.add_column("Type", style=_ACCENT2, width=12)
    table.add_column("URL", style=_DIM, ratio=2)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Status", justify="center", width=10)

    for i, src in enumerate(sources, 1):
        score = src.relevance_score
        score_color = _SUCCESS if score >= 0.7 else _WARN if score >= 0.4 else _DIM
        src_type = src.source_type.value if hasattr(src.source_type, "value") else str(src.source_type)

        status_display = {
            "discovered": f"[{_DIM}]--[/]",
            "selected": f"[{_ACCENT}]SEL[/]",
            "fetching": f"[{_WARN}]...[/]",
            "fetched": f"[{_SUCCESS}]OK[/]",
            "validated": f"[{_SUCCESS}]OK[/]",
            "failed": f"[{_ERR}]FAIL[/]",
        }.get(src.status.value if hasattr(src.status, "value") else str(src.status), f"[{_DIM}]--[/]")

        table.add_row(
            str(i),
            src.title,
            src_type,
            src.url,
            f"[{score_color}]{score:.0%}[/]",
            status_display,
        )

    return table


def render_fetch_results_table(results: list[FetchResult]) -> Table:
    """Build a Rich Table showing fetch results."""
    table = Table(
        title=f"[bold {_ACCENT}]FETCH RESULTS[/]",
        show_lines=True,
        expand=True,
        title_style=f"bold {_ACCENT}",
        border_style=_MUTED,
    )
    table.add_column("#", style=_DIM, width=4, justify="right")
    table.add_column("Source", min_width=20, style="white")
    table.add_column("Status", width=8, justify="center")
    table.add_column("Size", width=12, justify="right", style=_DIM)
    table.add_column("Path", style=_DIM, ratio=2)

    for i, r in enumerate(results, 1):
        status = f"[{_SUCCESS}]OK[/]" if r.success else f"[{_ERR}]FAIL[/]"
        size = f"{r.size_bytes:,} B" if r.success else "-"
        path = r.local_path if r.success else (r.error or "unknown error")[:60]
        table.add_row(
            str(i),
            r.source.title,
            status,
            size,
            path,
        )

    return table

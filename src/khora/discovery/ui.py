"""Rich TUI renderer for the interactive discovery agent.

Provides the ``DiscoveryUI`` class that encapsulates all terminal
rendering and user prompting, keeping the agent logic free of
presentation concerns.  Every prompt method returns structured data
(strings, indices, enum values) rather than Rich objects, making the
agent testable with a mock UI.
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


class DiscoveryUI:
    """Rich-based TUI for the discovery agent.

    All output goes through a Rich ``Console`` instance.  All prompts
    use ``rich.prompt`` and are wrapped in ``asyncio.to_thread`` so the
    event loop stays free during blocking input.

    The class is designed to be replaceable with a mock for testing —
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
        svc_text = ", ".join(f"[green]{s}[/green]" for s in services)
        self._console.print(
            Panel(
                "[bold]Khora Source Discovery[/bold]\n"
                f"Services: {svc_text}\n"
                f"Output: [cyan]{output_dir}[/cyan]\n\n"
                "Describe the data you need, and I will search the internet for sources.\n"
                "Type [bold cyan]quit[/bold cyan] to exit.",
                border_style="cyan",
            )
        )

    def show_rule(self, title: str) -> None:
        """Print a horizontal rule with a title."""
        self._console.print(Rule(f"[bold magenta]{title}[/]"))

    # ------------------------------------------------------------------
    # Intent gathering
    # ------------------------------------------------------------------

    async def prompt_intent(self) -> str:
        """Ask the user what data they need.

        Returns:
            The user's intent string, or empty string if they want to quit.
        """
        raw = await asyncio.to_thread(Prompt.ask, "\n[bold green]What data do you need?[/bold green]")
        if raw.strip().lower() in ("quit", "exit", "q"):
            return ""
        return raw.strip()

    async def prompt_url(self) -> str:
        """Ask the user for a URL directly (fallback when Perplexity unavailable)."""
        return await asyncio.to_thread(Prompt.ask, "  URL", default="")

    # ------------------------------------------------------------------
    # Search phase
    # ------------------------------------------------------------------

    def show_searching(self, query: str) -> Any:
        """Return a console status context manager for the search phase."""
        return self._console.status(f"[bold cyan]Searching: {query[:60]}...[/]")

    def show_search_results(self, answer: str, citation_count: int) -> None:
        """Display the Perplexity search answer."""
        if answer:
            self._console.print(
                Panel(
                    answer[:2000],
                    title=f"Search Results ({citation_count} source{'s' if citation_count != 1 else ''})",
                    border_style="green",
                )
            )

    def show_search_failed(self, error: str) -> None:
        self._console.print(f"[red]Search failed: {error}[/]")

    def show_no_results(self) -> None:
        self._console.print("[yellow]No sources found. Try a different query.[/]")

    def show_no_perplexity(self) -> None:
        self._console.print("[yellow]Perplexity not available. Please provide URLs directly.[/]")

    # ------------------------------------------------------------------
    # Source presentation
    # ------------------------------------------------------------------

    def show_sources(self, sources: list[DiscoveredSource]) -> None:
        """Render discovered sources as a numbered table."""
        self._console.print(f"\n[green]Found {len(sources)} source(s):[/]")
        self._console.print(render_sources_table(sources))

    # ------------------------------------------------------------------
    # Source selection
    # ------------------------------------------------------------------

    async def prompt_source_selection(self, count: int) -> list[int] | None:
        """Ask the user which sources to fetch.

        Returns:
            List of 0-based indices, or None to go back to search.
        """
        raw = await asyncio.to_thread(
            Prompt.ask,
            "Select sources (comma-separated numbers, 'all', or 'search' for new query)",
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
            self._console.print("[red]Invalid selection. Use numbers separated by commas.[/]")
            return await self.prompt_source_selection(count)

    # ------------------------------------------------------------------
    # Fetch phase
    # ------------------------------------------------------------------

    def show_fetching(self, url: str) -> Any:
        """Return a console status context manager for fetching."""
        return self._console.status(f"[bold cyan]Fetching {url}...[/]")

    def show_fetch_start(self, url: str) -> None:
        self._console.print(f"\n  Fetching: [cyan]{url}[/cyan]")

    def show_fetch_saved(self, filename: str, size: int, unit: str = "chars") -> None:
        self._console.print(f"  [green]Saved: {filename} ({size:,} {unit})[/]")

    def show_fetch_empty(self, url: str) -> None:
        self._console.print(f"  [yellow]No content extracted from {url}[/]")

    def show_fetch_failed(self, error: str) -> None:
        self._console.print(f"  [red]Failed: {error}[/]")

    def show_no_selection(self) -> None:
        self._console.print("[yellow]No sources selected.[/]")

    # ------------------------------------------------------------------
    # Review phase
    # ------------------------------------------------------------------

    def show_review_summary(self, fetched: list[FetchResult]) -> None:
        """Show a summary of fetch results for review."""
        successful = [f for f in fetched if f.success]
        self._console.print(f"\n[bold]Fetched {len(successful)}/{len(fetched)} source(s) successfully.[/]")
        for f in successful:
            self._console.print(f"  [green]✓[/] {f.local_path} ({f.size_bytes:,} bytes)")
        for f in fetched:
            if not f.success:
                self._console.print(f"  [red]✗[/] {f.source.url}: {f.error or 'unknown error'}")

    def show_data_preview(self, path: str, content: str, max_chars: int = 500) -> None:
        """Show a preview of fetched data content."""
        preview = content[:max_chars]
        if len(content) > max_chars:
            preview += "\n..."
        self._console.print(
            Panel(
                preview,
                title=f"Preview: {Path(path).name}",
                border_style="cyan",
            )
        )

    async def prompt_review_action(self) -> Literal["accept", "retry", "search", "quit"]:
        """Ask the user what to do after reviewing fetched data."""
        choice = await asyncio.to_thread(
            Prompt.ask,
            "Action",
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
                    f"[bold green]{len(paths)} file(s) fetched to {output_dir}[/bold green]\n\n"
                    f"Next step:\n  [cyan]khora ontology construct --source {output_dir}[/cyan]",
                    title="Discovery Complete",
                    border_style="green",
                )
            )
        else:
            self._console.print("[dim]No data fetched.[/]")

    def show_session_saved(self, path: str) -> None:
        self._console.print(f"[dim]Session saved: {path}[/]")

    def show_max_iterations(self) -> None:
        self._console.print("[yellow]Maximum iterations reached.[/]")

    def show_no_keys(self) -> None:
        self._console.print(
            "[red]No discovery API keys found.[/]\n"
            "Set PERPLEXITY_API_KEY and/or FIRECRAWL_API_KEY to enable source discovery."
        )

    def show_error(self, message: str) -> None:
        self._console.print(f"[red]{message}[/]")

    def show_info(self, message: str) -> None:
        self._console.print(message)

    def show_cost(self, cost_usd: float) -> None:
        """Show running cost estimate."""
        self._console.print(f"[dim]Session cost: ~${cost_usd:.2f}[/]")


# ---------------------------------------------------------------------------
# Standalone rendering functions (used by both UI class and CLI)
# ---------------------------------------------------------------------------


def render_sources_table(sources: list[DiscoveredSource]) -> Table:
    """Build a Rich Table showing discovered datasources."""
    table = Table(
        title="Discovered Sources",
        show_lines=True,
        expand=True,
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", style="bold white", min_width=20)
    table.add_column("Type", style="green", width=12)
    table.add_column("URL", style="blue", ratio=2)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Status", justify="center", width=10)

    for i, src in enumerate(sources, 1):
        score = src.relevance_score
        score_color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "dim"
        src_type = src.source_type.value if hasattr(src.source_type, "value") else str(src.source_type)

        status_display = {
            "discovered": "[dim]--[/dim]",
            "selected": "[cyan]SEL[/cyan]",
            "fetching": "[yellow]...[/yellow]",
            "fetched": "[green]OK[/green]",
            "validated": "[green]✓[/green]",
            "failed": "[red]FAIL[/red]",
        }.get(src.status.value if hasattr(src.status, "value") else str(src.status), "[dim]--[/]")

        table.add_row(
            str(i),
            src.title,
            src_type,
            src.url,
            f"[{score_color}]{score:.0%}[/{score_color}]",
            status_display,
        )

    return table


def render_fetch_results_table(results: list[FetchResult]) -> Table:
    """Build a Rich Table showing fetch results."""
    table = Table(
        title="Fetch Results",
        show_lines=True,
        expand=True,
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Source", min_width=20)
    table.add_column("Status", width=8, justify="center")
    table.add_column("Size", width=12, justify="right")
    table.add_column("Path", style="dim", ratio=2)

    for i, r in enumerate(results, 1):
        status = "[green]OK[/green]" if r.success else "[red]FAIL[/red]"
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

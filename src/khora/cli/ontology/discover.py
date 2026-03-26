"""CLI command for interactive datasource discovery.

Provides the ``khora ontology discover`` subcommand that launches an
interactive agent to find and pull datasources from the internet using
Perplexity (search) and Firecrawl (scraping).  Discovered data is saved
to a local directory and can feed into ``khora ontology construct``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import click
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from .tui.console import console


def _has_discovery_keys() -> dict[str, bool]:
    """Check which discovery API keys are available."""
    return {
        "perplexity": bool(os.environ.get("PERPLEXITY_API_KEY")),
        "firecrawl": bool(os.environ.get("FIRECRAWL_API_KEY")),
    }


def _render_discovered_sources(sources: list) -> Table:
    """Render discovered sources as a Rich table."""
    from khora.discovery.state import DiscoveredSource

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

    for i, src in enumerate(sources, 1):
        score = src.relevance_score if isinstance(src, DiscoveredSource) else 0.0
        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "dim"
        src_type = src.source_type.value if hasattr(src.source_type, "value") else str(src.source_type)
        table.add_row(
            str(i),
            src.title,
            src_type,
            src.url,
            f"[{color}]{score:.0%}[/{color}]",
        )

    return table


async def run_discovery_session(
    output_dir: Path,
    *,
    topic: str = "",
    resume_path: str | None = None,
) -> list[str]:
    """Run the interactive discovery session and return local file paths.

    This is the async entry point called by both the ``discover`` CLI
    command and the ``_phase_sources`` integration in the construct flow.

    Returns:
        List of local file/directory paths containing fetched data.
    """
    from khora.discovery.state import AgentPhase, SessionState

    keys = _has_discovery_keys()
    if not keys["perplexity"] and not keys["firecrawl"]:
        console.print(
            "[red]No discovery API keys found.[/]\n"
            "Set PERPLEXITY_API_KEY and/or FIRECRAWL_API_KEY to enable source discovery."
        )
        return []

    # Load or create session
    if resume_path:
        state = SessionState.load(resume_path)
        console.print(f"[dim]Resumed session {state.session_id[:8]}... (phase: {state.phase.value})[/]")
    else:
        state = SessionState(
            user_intent=topic,
            output_dir=str(output_dir),
        )
        if topic:
            state.phase = AgentPhase.SEARCH

    # Show capabilities
    services = []
    if keys["perplexity"]:
        services.append("[green]Perplexity[/green] (search)")
    if keys["firecrawl"]:
        services.append("[green]Firecrawl[/green] (scrape)")

    console.print(
        Panel(
            "[bold]Khora Source Discovery[/bold]\n"
            f"Services: {', '.join(services)}\n"
            f"Output: [cyan]{output_dir}[/cyan]\n\n"
            "Describe the data you need, and I will search the internet for sources.\n"
            "Type [bold cyan]quit[/bold cyan] to exit.",
            border_style="cyan",
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Interactive loop — simplified version; full agent (DYT-1220) replaces this
    discovered_paths: list[str] = []

    while state.phase != AgentPhase.DONE:
        if state.phase == AgentPhase.GATHER_INTENT:
            intent = Prompt.ask("\n[bold green]What data do you need?[/bold green]")
            if intent.strip().lower() in ("quit", "exit", "q"):
                break
            state.user_intent = intent
            state.phase = AgentPhase.SEARCH

        elif state.phase == AgentPhase.SEARCH:
            # Search using Perplexity
            if keys["perplexity"]:
                from khora.discovery.clients.perplexity import PerplexityClient

                with console.status("[bold cyan]Searching with Perplexity...[/]"):
                    try:
                        async with PerplexityClient() as client:
                            response = await client.search(
                                state.user_intent,
                                domain_hint=state.user_intent.split()[0] if state.user_intent else "",
                            )
                    except Exception as e:
                        console.print(f"[red]Search failed: {e}[/]")
                        state.phase = AgentPhase.GATHER_INTENT
                        continue

                # Show the answer
                if response.answer:
                    console.print(Panel(response.answer[:2000], title="Search Results", border_style="green"))

                # Convert citations to DiscoveredSource objects
                if response.citations:
                    from khora.discovery.state import DiscoveredSource

                    state.discovered = []
                    for url in response.citations:
                        state.discovered.append(
                            DiscoveredSource(
                                url=url,
                                title=url.split("/")[-1] or url,
                                discovered_via="perplexity",
                                discovery_query=state.user_intent,
                                relevance_score=0.5,
                            )
                        )
                    state.phase = AgentPhase.PRESENT_RESULTS
                else:
                    console.print("[yellow]No sources found. Try a different query.[/]")
                    state.phase = AgentPhase.GATHER_INTENT
            else:
                console.print("[yellow]Perplexity not available. Please provide URLs directly.[/]")
                url = Prompt.ask("  URL", default="")
                if url:
                    from khora.discovery.state import DiscoveredSource

                    state.discovered.append(DiscoveredSource(url=url, title=url, discovered_via="user"))
                    state.phase = AgentPhase.PRESENT_RESULTS
                else:
                    state.phase = AgentPhase.GATHER_INTENT

        elif state.phase == AgentPhase.PRESENT_RESULTS:
            if state.discovered:
                console.print(f"\n[green]Found {len(state.discovered)} source(s):[/]")
                console.print(_render_discovered_sources(state.discovered))
                state.phase = AgentPhase.SELECT_SOURCES
            else:
                state.phase = AgentPhase.GATHER_INTENT

        elif state.phase == AgentPhase.SELECT_SOURCES:
            raw = Prompt.ask(
                "Select sources (comma-separated numbers, 'all', or 'search' for new query)",
                default="all",
            )
            if raw.strip().lower() == "search":
                state.phase = AgentPhase.GATHER_INTENT
                continue
            if raw.strip().lower() == "all":
                state.selected_indices = list(range(len(state.discovered)))
            else:
                try:
                    state.selected_indices = [int(x.strip()) - 1 for x in raw.split(",")]
                except ValueError:
                    console.print("[red]Invalid selection. Use numbers separated by commas.[/]")
                    continue
            state.phase = AgentPhase.FETCH

        elif state.phase == AgentPhase.FETCH:
            selected = state.selected_sources
            if not selected:
                console.print("[yellow]No sources selected.[/]")
                state.phase = AgentPhase.SELECT_SOURCES
                continue

            for src in selected:
                console.print(f"\n  Fetching: [cyan]{src.url}[/cyan]")

                if keys["firecrawl"]:
                    from khora.discovery.clients.firecrawl import FirecrawlClient
                    from khora.discovery.state import FetchAttempt, FetchMethod, FetchResult

                    try:
                        with console.status(f"[bold cyan]Scraping {src.url}...[/]"):
                            async with FirecrawlClient() as fc:
                                result = await fc.scrape(src.url)

                        if result.markdown:
                            # Save to output dir
                            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in src.title[:50])
                            out_file = output_dir / f"{safe_name}.md"
                            out_file.write_text(result.markdown, encoding="utf-8")
                            discovered_paths.append(str(out_file))

                            state.fetched.append(
                                FetchResult(
                                    source=src,
                                    local_path=str(out_file),
                                    content_type="text/markdown",
                                    size_bytes=len(result.markdown.encode()),
                                    success=True,
                                    attempts=[
                                        FetchAttempt(
                                            method=FetchMethod.FIRECRAWL_SCRAPE,
                                            success=True,
                                            bytes_fetched=len(result.markdown.encode()),
                                        )
                                    ],
                                )
                            )
                            console.print(f"  [green]Saved: {out_file.name} ({len(result.markdown):,} chars)[/]")
                        else:
                            console.print(f"  [yellow]No content extracted from {src.url}[/]")
                    except Exception as e:
                        console.print(f"  [red]Failed: {e}[/]")
                else:
                    # Fallback: try direct download with httpx
                    import httpx

                    try:
                        with console.status(f"[bold cyan]Downloading {src.url}...[/]"):
                            async with httpx.AsyncClient(timeout=30.0) as http:
                                resp = await http.get(src.url, follow_redirects=True)
                                resp.raise_for_status()

                        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in src.title[:50])
                        ext = ".html" if "html" in resp.headers.get("content-type", "") else ".txt"
                        out_file = output_dir / f"{safe_name}{ext}"
                        out_file.write_bytes(resp.content)
                        discovered_paths.append(str(out_file))
                        console.print(f"  [green]Saved: {out_file.name} ({len(resp.content):,} bytes)[/]")
                    except Exception as e:
                        console.print(f"  [red]Failed: {e}[/]")

            state.phase = AgentPhase.REVIEW

        elif state.phase == AgentPhase.REVIEW:
            successful = state.successful_fetches
            console.print(f"\n[bold]Fetched {len(successful)}/{len(state.fetched)} source(s) successfully.[/]")
            for f in successful:
                console.print(f"  [green]✓[/] {f.local_path} ({f.size_bytes:,} bytes)")

            choice = Prompt.ask(
                "Action",
                choices=["accept", "retry", "search", "quit"],
                default="accept",
            )
            if choice == "accept":
                state.phase = AgentPhase.DONE
            elif choice == "retry":
                state.fetched.clear()
                state.phase = AgentPhase.FETCH
            elif choice == "search":
                state.phase = AgentPhase.GATHER_INTENT
            else:
                break

        else:
            break

        state.iteration += 1
        if state.iteration >= state.max_iterations:
            console.print("[yellow]Maximum iterations reached.[/]")
            break

    # Save session for potential resume
    session_file = output_dir / ".discovery_session.json"
    state.save(session_file)
    console.print(f"[dim]Session saved: {session_file}[/]")

    return discovered_paths


@click.command(name="discover")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="./khora_discovery_data",
    show_default=True,
    help="Directory to store fetched data.",
)
@click.option(
    "--topic",
    "-t",
    type=str,
    default="",
    help="Start with a topic (skip first prompt).",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume a saved discovery session.",
)
def discover(output_dir: Path, topic: str, resume: str | None) -> None:
    """Interactively discover and fetch datasources from the internet.

    Uses Perplexity for search and Firecrawl for web scraping.
    Fetched data is saved to OUTPUT_DIR and can be used with
    ``khora ontology construct --source OUTPUT_DIR``.
    """
    from .tui.console import print_header

    print_header()
    console.print(Rule("[bold magenta]Source Discovery[/]"))

    try:
        paths = asyncio.run(
            run_discovery_session(
                output_dir,
                topic=topic,
                resume_path=resume,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Discovery interrupted.[/]")
        paths = []

    if paths:
        console.print(
            Panel(
                f"[bold green]{len(paths)} file(s) fetched to {output_dir}[/bold green]\n\n"
                f"Next step:\n  [cyan]khora ontology construct --source {output_dir}[/cyan]",
                title="Discovery Complete",
                border_style="green",
            )
        )
    else:
        console.print("[dim]No data fetched.[/]")

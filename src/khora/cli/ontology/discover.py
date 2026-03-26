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

from .tui.console import console


def _has_discovery_keys() -> dict[str, bool]:
    """Check which discovery API keys are available."""
    return {
        "perplexity": bool(os.environ.get("PERPLEXITY_API_KEY")),
        "firecrawl": bool(os.environ.get("FIRECRAWL_API_KEY")),
    }


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
    from khora.discovery.state import AgentPhase, DiscoveredSource, FetchAttempt, FetchMethod, FetchResult, SessionState
    from khora.discovery.ui import DiscoveryUI

    ui = DiscoveryUI(console)
    keys = _has_discovery_keys()

    if not keys["perplexity"] and not keys["firecrawl"]:
        ui.show_no_keys()
        return []

    # Load or create session
    if resume_path:
        state = SessionState.load(resume_path)
        ui.show_info(f"[dim]Resumed session {state.session_id[:8]}... (phase: {state.phase.value})[/]")
    else:
        state = SessionState(user_intent=topic, output_dir=str(output_dir))
        if topic:
            state.phase = AgentPhase.SEARCH

    # Show capabilities
    services = []
    if keys["perplexity"]:
        services.append("Perplexity (search)")
    if keys["firecrawl"]:
        services.append("Firecrawl (scrape)")
    ui.show_welcome(services, str(output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Interactive loop — simplified version; full agent (DYT-1220) replaces this
    discovered_paths: list[str] = []

    while state.phase != AgentPhase.DONE:
        if state.phase == AgentPhase.GATHER_INTENT:
            intent = await ui.prompt_intent()
            if not intent:
                break
            state.user_intent = intent
            state.phase = AgentPhase.SEARCH

        elif state.phase == AgentPhase.SEARCH:
            if keys["perplexity"]:
                from khora.discovery.clients.perplexity import PerplexityClient

                with ui.show_searching(state.user_intent):
                    try:
                        async with PerplexityClient() as client:
                            response = await client.search(
                                state.user_intent,
                                domain_hint=state.user_intent.split()[0] if state.user_intent else "",
                            )
                    except Exception as e:
                        ui.show_search_failed(str(e))
                        state.phase = AgentPhase.GATHER_INTENT
                        continue

                ui.show_search_results(response.answer, len(response.citations))

                if response.citations:
                    state.discovered = [
                        DiscoveredSource(
                            url=url,
                            title=url.split("/")[-1] or url,
                            discovered_via="perplexity",
                            discovery_query=state.user_intent,
                            relevance_score=0.5,
                        )
                        for url in response.citations
                    ]
                    state.phase = AgentPhase.PRESENT_RESULTS
                else:
                    ui.show_no_results()
                    state.phase = AgentPhase.GATHER_INTENT
            else:
                ui.show_no_perplexity()
                url = await ui.prompt_url()
                if url:
                    state.discovered.append(DiscoveredSource(url=url, title=url, discovered_via="user"))
                    state.phase = AgentPhase.PRESENT_RESULTS
                else:
                    state.phase = AgentPhase.GATHER_INTENT

        elif state.phase == AgentPhase.PRESENT_RESULTS:
            if state.discovered:
                ui.show_sources(state.discovered)
                state.phase = AgentPhase.SELECT_SOURCES
            else:
                state.phase = AgentPhase.GATHER_INTENT

        elif state.phase == AgentPhase.SELECT_SOURCES:
            indices = await ui.prompt_source_selection(len(state.discovered))
            if indices is None:
                state.phase = AgentPhase.GATHER_INTENT
                continue
            state.selected_indices = indices
            state.phase = AgentPhase.FETCH

        elif state.phase == AgentPhase.FETCH:
            selected = state.selected_sources
            if not selected:
                ui.show_no_selection()
                state.phase = AgentPhase.SELECT_SOURCES
                continue

            for src in selected:
                ui.show_fetch_start(src.url)

                if keys["firecrawl"]:
                    from khora.discovery.clients.firecrawl import FirecrawlClient

                    try:
                        with ui.show_fetching(src.url):
                            async with FirecrawlClient() as fc:
                                result = await fc.scrape(src.url)

                        if result.markdown:
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
                            ui.show_fetch_saved(out_file.name, len(result.markdown))
                        else:
                            ui.show_fetch_empty(src.url)
                    except Exception as e:
                        ui.show_fetch_failed(str(e))
                else:
                    import httpx

                    try:
                        with ui.show_fetching(src.url):
                            async with httpx.AsyncClient(timeout=30.0) as http:
                                resp = await http.get(src.url, follow_redirects=True)
                                resp.raise_for_status()

                        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in src.title[:50])
                        ext = ".html" if "html" in resp.headers.get("content-type", "") else ".txt"
                        out_file = output_dir / f"{safe_name}{ext}"
                        out_file.write_bytes(resp.content)
                        discovered_paths.append(str(out_file))
                        ui.show_fetch_saved(out_file.name, len(resp.content), unit="bytes")
                    except Exception as e:
                        ui.show_fetch_failed(str(e))

            state.phase = AgentPhase.REVIEW

        elif state.phase == AgentPhase.REVIEW:
            ui.show_review_summary(state.fetched)

            choice = await ui.prompt_review_action()
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
            ui.show_max_iterations()
            break

    # Save session for potential resume
    session_file = output_dir / ".discovery_session.json"
    state.save(session_file)
    ui.show_session_saved(str(session_file))

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
    from khora.discovery.ui import DiscoveryUI

    from .tui.console import print_header

    print_header()

    ui = DiscoveryUI(console)
    ui.show_rule("Source Discovery")

    try:
        paths = asyncio.run(run_discovery_session(output_dir, topic=topic, resume_path=resume))
    except KeyboardInterrupt:
        console.print("\n[yellow]Discovery interrupted.[/]")
        paths = []

    ui.show_done(paths, str(output_dir))

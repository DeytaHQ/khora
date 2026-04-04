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


def _default_output_dir(session_id: str) -> Path:
    """Compute the default output directory using XDG conventions.

    Respects KHORA_DATA_DIR env var, falls back to ~/.local/share/khora/discovery/.
    """
    base = os.environ.get("KHORA_DATA_DIR")
    if base:
        return Path(base) / "discovery" / session_id[:8]
    return Path.home() / ".local" / "share" / "khora" / "discovery" / session_id[:8]


async def run_discovery_session(
    output_dir: Path | None = None,
    *,
    topic: str = "",
    resume_path: str | None = None,
    litellm_config: str | None = None,
) -> list[str]:
    """Run the interactive discovery session and return local file paths.

    This is the async entry point called by both the ``discover`` CLI
    command and the ``_phase_sources`` integration in the construct flow.

    Args:
        output_dir: Where to save fetched data. If None, prompts the user.
        topic: Pre-fill the intent (skip first prompt).
        resume_path: Path to a saved session JSON to resume.
        litellm_config: Path to LiteLLM YAML config for discovery models.

    Returns:
        List of local file/directory paths containing fetched data.
    """
    from khora.discovery.agent import DiscoveryAgent
    from khora.discovery.state import AgentPhase, SessionState
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
        state = SessionState(user_intent=topic)
        if topic:
            state.phase = AgentPhase.SEARCH

    # Resolve output directory
    if output_dir is None:
        default_dir = _default_output_dir(state.session_id)
        chosen = await ui.prompt_output_dir(str(default_dir))
        output_dir = Path(chosen)

    state.output_dir = str(output_dir)

    # Show capabilities
    services = []
    if keys["perplexity"]:
        services.append("Perplexity (search)")
    if keys["firecrawl"]:
        services.append("Firecrawl (scrape)")
    ui.show_welcome(services, str(output_dir))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load discovery settings from config (if available)
    try:
        from khora.config.schema import KhoraConfig

        discovery_settings = KhoraConfig().discovery
    except Exception:
        discovery_settings = None

    # Apply litellm config override
    if discovery_settings and litellm_config:
        discovery_settings.litellm_config = litellm_config
    elif litellm_config:
        from khora.config.schema import DiscoverySettings

        discovery_settings = DiscoverySettings(litellm_config=litellm_config)

    # Run the agent
    agent = DiscoveryAgent(ui=ui, output_dir=output_dir, state=state, settings=discovery_settings)
    final_state = await agent.run()

    # Save session for potential resume
    session_file = output_dir / ".discovery_session.json"
    final_state.save(session_file)
    ui.show_session_saved(str(session_file))

    return [f.local_path for f in final_state.successful_fetches]


@click.command(name="discover")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store fetched data (default: ~/.local/share/khora/discovery/).",
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
@click.option(
    "--construct/--no-construct",
    default=None,
    help="Continue to ontology construction after discovery (default: ask).",
)
@click.option(
    "--litellm",
    "-l",
    "litellm_config",
    type=click.Path(exists=True),
    default=None,
    help="Path to LiteLLM YAML config for discovery models.",
)
def discover(
    output_dir: Path | None, topic: str, resume: str | None, construct: bool | None, litellm_config: str | None
) -> None:
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

    resolved_dir: Path | None = output_dir
    try:
        paths = asyncio.run(
            run_discovery_session(resolved_dir, topic=topic, resume_path=resume, litellm_config=litellm_config)
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Discovery interrupted.[/]")
        paths = []

    if not paths:
        ui.show_done(paths, str(resolved_dir or ""))
        return

    # Determine the actual output directory from the first path
    actual_dir = Path(paths[0]).parent if paths else resolved_dir
    ui.show_done(paths, str(actual_dir))

    # Offer to continue to ontology construction
    should_construct = construct
    if should_construct is None and paths:
        should_construct = asyncio.run(ui.prompt_continue_to_construct())

    if should_construct and actual_dir:
        _run_construct(actual_dir)


def _run_construct(source_dir: Path) -> None:
    """Launch ontology construction on the discovered data."""
    from .flow import run_construct

    console.print()
    run_construct(
        source=(str(source_dir),),
        output="./ontology.yaml",
        model="gpt-4o",
        budget=1.0,
        extends_skill=None,
        non_interactive=False,
        resume=None,
    )

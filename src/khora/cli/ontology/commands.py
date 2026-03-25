"""Click command definitions for khora ontology."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from khora.extraction.skills.base import ExpertiseConfig

from .tui.console import console, print_header
from .tui.panels import (
    render_entity_types_table,
    render_ontology_tree,
    render_relationship_types_table,
)


@click.group(name="ontology")
def ontology_group() -> None:
    """AI-powered ontology construction and management."""


# ---------------------------------------------------------------------------
# construct
# ---------------------------------------------------------------------------


@ontology_group.command()
@click.option(
    "--source",
    "-s",
    multiple=True,
    type=str,
    help="Data source paths (repeatable).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./ontology.yaml",
    show_default=True,
    help="Output path for the generated ontology YAML.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="gpt-4o",
    show_default=True,
    help="LLM model to use for construction.",
)
@click.option(
    "--budget",
    type=float,
    default=1.0,
    show_default=True,
    help="Maximum USD budget for LLM calls.",
)
@click.option(
    "--extends",
    "extends_skill",
    type=str,
    default=None,
    help='Base skill to extend (e.g. "builtin:general").',
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Run without interactive prompts.",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume from a saved session file.",
)
def construct(
    source: tuple[str, ...],
    output: str,
    model: str,
    budget: float,
    extends_skill: str | None,
    non_interactive: bool,
    resume: str | None,
) -> None:
    """Construct an ontology from data sources using LLM analysis."""
    print_header()

    from .flow import run_construct

    run_construct(
        source=source,
        output=output,
        model=model,
        budget=budget,
        extends_skill=extends_skill,
        non_interactive=non_interactive,
        resume=resume,
    )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@ontology_group.command()
@click.argument("file", type=click.Path(exists=True))
def validate(file: str) -> None:
    """Validate an ontology YAML file against ExpertiseConfig."""
    path = Path(file)
    console.print(f"Validating [bold]{path.name}[/] ...")

    # Load YAML
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        console.print(f"[red bold]FAIL[/]  Could not parse YAML: {exc}")
        raise SystemExit(1) from exc

    if not isinstance(data, dict):
        console.print("[red bold]FAIL[/]  Top-level YAML value must be a mapping.")
        raise SystemExit(1)

    # Parse into ExpertiseConfig
    try:
        config = ExpertiseConfig.from_dict(data)
    except Exception as exc:
        console.print(f"[red bold]FAIL[/]  Invalid ExpertiseConfig: {exc}")
        raise SystemExit(1) from exc

    # Semantic checks
    errors: list[str] = []
    entity_names = {et.name for et in config.entity_types}

    for rt in config.relationship_types:
        for src in rt.source_types:
            if src != "*" and src not in entity_names:
                errors.append(f"Relationship '{rt.name}' references unknown source type '{src}'.")
        for tgt in rt.target_types:
            if tgt != "*" and tgt not in entity_names:
                errors.append(f"Relationship '{rt.name}' references unknown target type '{tgt}'.")

    if errors:
        console.print()
        for err in errors:
            console.print(f"  [red]- {err}[/]")
        console.print()
        console.print(f"[red bold]FAIL[/]  {len(errors)} validation error(s).")
        raise SystemExit(1)

    console.print(f"[green bold]PASS[/]  '{config.name}' v{config.version} is valid.")
    console.print(
        f"  {len(config.entity_types)} entity type(s), "
        f"{len(config.relationship_types)} relationship type(s), "
        f"{len(config.correlation_rules)} correlation rule(s), "
        f"{len(config.inference_rules)} inference rule(s)."
    )


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


@ontology_group.command()
@click.argument("file", type=click.Path(exists=True))
def preview(file: str) -> None:
    """Preview an ontology YAML file in a rich display."""
    path = Path(file)

    # Load YAML
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        console.print(f"[red bold]Error[/]  Could not parse YAML: {exc}")
        raise SystemExit(1) from exc

    if not isinstance(data, dict):
        console.print("[red bold]Error[/]  Top-level YAML value must be a mapping.")
        raise SystemExit(1)

    try:
        config = ExpertiseConfig.from_dict(data)
    except Exception as exc:
        console.print(f"[red bold]Error[/]  Invalid ExpertiseConfig: {exc}")
        raise SystemExit(1) from exc

    # Display
    print_header()

    if config.entity_types:
        console.print(render_entity_types_table(config.entity_types))
        console.print()

    if config.relationship_types:
        console.print(render_relationship_types_table(config.relationship_types))
        console.print()

    console.print(render_ontology_tree(config))

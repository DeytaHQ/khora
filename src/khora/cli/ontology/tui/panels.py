"""Rich panel and table builders for ontology display."""

from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.tree import Tree

from khora.extraction.skills.base import (
    EntityTypeConfig,
    ExpertiseConfig,
    RelationshipTypeConfig,
)


def render_entity_types_table(types: list[EntityTypeConfig]) -> Table:
    """Build a Rich table of entity type definitions."""
    table = Table(title="Entity Types", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bright_blue bold")
    table.add_column("Description")
    table.add_column("Attributes")
    table.add_column("Identifiers")

    for idx, et in enumerate(types, 1):
        attrs = ", ".join(f"{k}: [{', '.join(v)}]" for k, v in et.attributes.items()) if et.attributes else "-"
        ids = ", ".join(et.identifiers) if et.identifiers else "-"
        table.add_row(str(idx), et.name, et.description or "-", attrs, ids)

    return table


def render_relationship_types_table(types: list[RelationshipTypeConfig]) -> Table:
    """Build a Rich table of relationship type definitions."""
    table = Table(title="Relationship Types", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="bright_green bold")
    table.add_column("Source Types")
    table.add_column("Target Types")
    table.add_column("Bidir", width=6)
    table.add_column("Description")

    for idx, rt in enumerate(types, 1):
        src = ", ".join(rt.source_types) if rt.source_types else "*"
        tgt = ", ".join(rt.target_types) if rt.target_types else "*"
        bidir = "Yes" if rt.bidirectional else "No"
        table.add_row(str(idx), rt.name, src, tgt, bidir, rt.description or "-")

    return table


def render_ontology_tree(config: ExpertiseConfig) -> Tree:
    """Build a Rich tree view of the full ontology config."""
    tree = Tree(f"[bold]{config.name}[/] v{config.version}")

    if config.description:
        tree.add(f"[dim]{config.description}[/]")

    # Entity types
    if config.entity_types:
        et_branch = tree.add("[bright_blue bold]Entity Types[/]")
        for et in config.entity_types:
            label = f"[bright_blue]{et.name}[/]"
            if et.description:
                label += f"  [dim]{et.description}[/]"
            node = et_branch.add(label)
            if et.attributes:
                for key, vals in et.attributes.items():
                    node.add(f"[dim]{key}:[/] {', '.join(vals)}")
            if et.identifiers:
                node.add(f"[dim]identifiers:[/] {', '.join(et.identifiers)}")

    # Relationship types
    if config.relationship_types:
        rt_branch = tree.add("[bright_green bold]Relationship Types[/]")
        for rt in config.relationship_types:
            bidir = " (bidir)" if rt.bidirectional else ""
            label = f"[bright_green]{rt.name}[/]{bidir}"
            if rt.description:
                label += f"  [dim]{rt.description}[/]"
            node = rt_branch.add(label)
            src = ", ".join(rt.source_types) if rt.source_types else "*"
            tgt = ", ".join(rt.target_types) if rt.target_types else "*"
            node.add(f"[dim]{src} -> {tgt}[/]")

    # Correlation rules
    if config.correlation_rules:
        cr_branch = tree.add("[bright_yellow bold]Correlation Rules[/]")
        for cr in config.correlation_rules:
            cr_branch.add(f"[bright_yellow]{cr.name}[/]  [dim]{cr.description}[/]")

    # Inference rules
    if config.inference_rules:
        ir_branch = tree.add("[bright_yellow bold]Inference Rules[/]")
        for ir in config.inference_rules:
            ir_branch.add(f"[bright_yellow]{ir.name}[/]  [dim]{ir.description}[/]")

    # Confidence
    conf = config.confidence
    conf_branch = tree.add("[bold]Confidence Thresholds[/]")
    conf_branch.add(f"min_entity: {conf.min_entity}")
    conf_branch.add(f"min_relationship: {conf.min_relationship}")
    conf_branch.add(f"min_inferred: {conf.min_inferred}")

    return tree


def render_source_summary_table(sources: list[dict[str, Any]]) -> Table:
    """Build a Rich table summarising data sources."""
    table = Table(title="Data Sources", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Path", style="bold")
    table.add_column("Type")
    table.add_column("Size")

    for idx, src in enumerate(sources, 1):
        table.add_row(
            str(idx),
            str(src.get("path", "-")),
            str(src.get("type", "-")),
            str(src.get("size", "-")),
        )

    return table

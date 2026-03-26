"""Rich panel and table builders for ontology display.

Uses the TE-inspired palette from console.py — orange accent on
monochrome, clean grid layout, minimal decoration.
"""

from __future__ import annotations

from typing import Any

from rich.table import Table
from rich.tree import Tree

from khora.extraction.skills.base import (
    EntityTypeConfig,
    ExpertiseConfig,
    RelationshipTypeConfig,
)

from .console import ACCENT, ACCENT2, DIM, MUTED, WARN


def render_entity_types_table(types: list[EntityTypeConfig]) -> Table:
    """Build a Rich table of entity type definitions."""
    table = Table(
        title=f"[bold {ACCENT}]ENTITY TYPES[/]", show_lines=True, border_style=MUTED, title_style=f"bold {ACCENT}"
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("Name", style=f"bold {ACCENT}")
    table.add_column("Description", style="white")
    table.add_column("Attributes", style=DIM)
    table.add_column("Identifiers", style=DIM)

    for idx, et in enumerate(types, 1):
        attrs = ", ".join(f"{k}: [{', '.join(v)}]" for k, v in et.attributes.items()) if et.attributes else "-"
        ids = ", ".join(et.identifiers) if et.identifiers else "-"
        table.add_row(str(idx), et.name, et.description or "-", attrs, ids)

    return table


def render_relationship_types_table(types: list[RelationshipTypeConfig]) -> Table:
    """Build a Rich table of relationship type definitions."""
    table = Table(
        title=f"[bold {ACCENT2}]RELATIONSHIP TYPES[/]",
        show_lines=True,
        border_style=MUTED,
        title_style=f"bold {ACCENT2}",
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("Name", style=f"bold {ACCENT2}")
    table.add_column("Source Types", style="white")
    table.add_column("Target Types", style="white")
    table.add_column("Bidir", width=6, style=DIM)
    table.add_column("Description", style="white")

    for idx, rt in enumerate(types, 1):
        src = ", ".join(rt.source_types) if rt.source_types else "*"
        tgt = ", ".join(rt.target_types) if rt.target_types else "*"
        bidir = "Yes" if rt.bidirectional else "No"
        table.add_row(str(idx), rt.name, src, tgt, bidir, rt.description or "-")

    return table


def render_ontology_tree(config: ExpertiseConfig) -> Tree:
    """Build a Rich tree view of the full ontology config."""
    tree = Tree(f"[bold {ACCENT}]{config.name}[/] [dim]v{config.version}[/]")

    if config.description:
        tree.add(f"[{DIM}]{config.description}[/]")

    # Entity types
    if config.entity_types:
        et_branch = tree.add(f"[bold {ACCENT}]Entity Types[/]")
        for et in config.entity_types:
            label = f"[{ACCENT}]{et.name}[/]"
            if et.description:
                label += f"  [{DIM}]{et.description}[/]"
            node = et_branch.add(label)
            if et.attributes:
                for key, vals in et.attributes.items():
                    node.add(f"[{DIM}]{key}:[/] {', '.join(vals)}")
            if et.identifiers:
                node.add(f"[{DIM}]identifiers:[/] {', '.join(et.identifiers)}")

    # Relationship types
    if config.relationship_types:
        rt_branch = tree.add(f"[bold {ACCENT2}]Relationship Types[/]")
        for rt in config.relationship_types:
            bidir = " (bidir)" if rt.bidirectional else ""
            label = f"[{ACCENT2}]{rt.name}[/]{bidir}"
            if rt.description:
                label += f"  [{DIM}]{rt.description}[/]"
            node = rt_branch.add(label)
            src = ", ".join(rt.source_types) if rt.source_types else "*"
            tgt = ", ".join(rt.target_types) if rt.target_types else "*"
            node.add(f"[{DIM}]{src} -> {tgt}[/]")

    # Correlation rules
    if config.correlation_rules:
        cr_branch = tree.add(f"[bold {WARN}]Correlation Rules[/]")
        for cr in config.correlation_rules:
            cr_branch.add(f"[{WARN}]{cr.name}[/]  [{DIM}]{cr.description}[/]")

    # Inference rules
    if config.inference_rules:
        ir_branch = tree.add(f"[bold {WARN}]Inference Rules[/]")
        for ir in config.inference_rules:
            ir_branch.add(f"[{WARN}]{ir.name}[/]  [{DIM}]{ir.description}[/]")

    # Confidence
    conf = config.confidence
    conf_branch = tree.add("[bold]Confidence Thresholds[/]")
    conf_branch.add(f"min_entity: [{ACCENT}]{conf.min_entity}[/]")
    conf_branch.add(f"min_relationship: [{ACCENT}]{conf.min_relationship}[/]")
    conf_branch.add(f"min_inferred: [{ACCENT}]{conf.min_inferred}[/]")

    return tree


def render_source_summary_table(sources: list[dict[str, Any]]) -> Table:
    """Build a Rich table summarising data sources."""
    table = Table(
        title=f"[bold {ACCENT}]DATA SOURCES[/]", show_lines=True, border_style=MUTED, title_style=f"bold {ACCENT}"
    )
    table.add_column("#", style=DIM, width=4)
    table.add_column("Path", style="bold white")
    table.add_column("Type", style=ACCENT2)
    table.add_column("Size", style=DIM)

    for idx, src in enumerate(sources, 1):
        table.add_row(
            str(idx),
            str(src.get("path", "-")),
            str(src.get("type", "-")),
            str(src.get("size", "-")),
        )

    return table

"""Shared Rich console, theme, and branding for ontology CLI.

Visual language inspired by Teenage Engineering — monochrome base with
a single high-contrast accent colour (orange), clean grid layout, and
brutalist industrial typography.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

# -- Palette (TE-inspired: mono + single accent) ----------------------------

ACCENT = "#FF6600"  # TE orange
ACCENT2 = "#FF9933"  # softer orange for secondary highlights
MUTED = "bright_black"  # dark grey for structure
TEXT = "white"
DIM = "dim"
SUCCESS = "#33FF66"  # terminal green
WARN = "#FFCC00"  # amber
ERR = "#FF3333"  # red

# Semantic roles
ENTITY_COLOR = ACCENT
RELATIONSHIP_COLOR = ACCENT2
RULE_COLOR = WARN

# -- Theme -------------------------------------------------------------------

khora_theme = Theme(
    {
        "header": f"bold {ACCENT}",
        "accent": ACCENT,
        "accent2": ACCENT2,
        "muted": MUTED,
        "entity": f"bold {ACCENT}",
        "relationship": ACCENT2,
        "rule": WARN,
        "success": SUCCESS,
        "warning": WARN,
        "error": ERR,
        "dim": DIM,
        "label": f"bold {TEXT}",
        "value": ACCENT,
    }
)

# -- Shared console instance -------------------------------------------------

console = Console(theme=khora_theme)

# -- ASCII header ------------------------------------------------------------
# Block letters built from box-drawing + braille characters — dense, compact,
# industrial feel.  Orange accent with dim structural lines.

_HEADER = f"""\
[{ACCENT} bold]  ┃ ┃┏━ ┃ ┃ ┏━┃ ┏━┃  ┏━┃[/]
[{ACCENT} bold]  ┣┻┃┃  ┣━┃ ┃ ┃ ┣┳┛  ┣━┃[/]
[{ACCENT} bold]  ┃ ┃┗━ ┃ ┃ ┗━┃ ┃ ┗  ┃ ┃[/]
[{MUTED}]  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[{DIM}]  ontology processor[/]  [{MUTED}]//[/]  [{ACCENT2}]v0.5[/]  [{MUTED}]//[/]  [{DIM}]deyta.ai[/]"""

_TAGLINE = ""


def print_header() -> None:
    """Print the branded header if running in an interactive terminal."""
    if sys.stdout.isatty():
        console.print()
        console.print(_HEADER, highlight=False)
        if _TAGLINE:
            console.print(_TAGLINE, highlight=False)
        console.print()

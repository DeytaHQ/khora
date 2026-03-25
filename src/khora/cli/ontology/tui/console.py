"""Shared Rich console, theme, and ASCII art for ontology CLI."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

# -- Colour constants --------------------------------------------------------

CYAN = "bright_cyan"
ENTITY_COLOR = "bright_blue"
RELATIONSHIP_COLOR = "bright_green"
RULE_COLOR = "bright_yellow"
DIM = "dim"

# -- Theme -------------------------------------------------------------------

khora_theme = Theme(
    {
        "header": CYAN,
        "entity": ENTITY_COLOR,
        "relationship": RELATIONSHIP_COLOR,
        "rule": RULE_COLOR,
        "dim": DIM,
    }
)

# -- Shared console instance -------------------------------------------------

console = Console(theme=khora_theme)

# -- ASCII art ---------------------------------------------------------------

_HEADER = r"""
  [bright_cyan]╦╔═╦ ╦╔═╗╦═╗╔═╗  ╔═╗╔╗╔╔╦╗╔═╗╦  ╔═╗╔═╗╦ ╦[/]
  [bright_cyan]╠╩╗╠═╣║ ║╠╦╝╠═╣  ║ ║║║║ ║ ║ ║║  ║ ║║ ╦╚╦╝[/]
  [bright_cyan]╩ ╩╩ ╩╚═╝╩╚═╩ ╩  ╚═╝╝╚╝ ╩ ╚═╝╩═╝╚═╝╚═╝ ╩[/]
"""

_TAGLINE = "  [dim]AI-powered ontology construction[/]\n"


def print_header() -> None:
    """Print the ASCII art header if running in an interactive terminal."""
    if sys.stdout.isatty():
        console.print(_HEADER, highlight=False)
        console.print(_TAGLINE, highlight=False)

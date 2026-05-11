"""Controlled source vocabulary for Khora.

Defines canonical source type identifiers shared between extraction
and query pipelines, ensuring consistent provenance tracking.

The registry is extensible — downstream projects can register
additional source types and aliases via ``register_source_type()``
and ``register_source_alias()``.
"""

from __future__ import annotations

from enum import Enum


class SourceTool(str, Enum):
    """Canonical source type identifiers.

    Used for source provenance on documents, chunks, and entities,
    and for query-time source priority boosting.

    Only generic/transport source types are defined here.
    Domain-specific tools (Slack, Linear, Jira, etc.) should be
    registered by downstream projects via register_source_type().
    """

    FILE = "file"
    URL = "url"
    API = "api"
    UNKNOWN = "unknown"


# Dynamic registry for source types added by downstream projects.
# Maps lowercase canonical name -> display name.
_REGISTERED_SOURCE_TYPES: dict[str, str] = {}

# Mapping from loose source_type strings to canonical source name.
# Covers common variations encountered during ingestion.
# Extensible via register_source_alias().
SOURCE_TYPE_ALIASES: dict[str, str] = {
    # Generic
    "file": "file",
    "url": "url",
    "api": "api",
}


def register_source_type(name: str, *, display_name: str | None = None) -> None:
    """Register a new source type.

    Args:
        name: Canonical source type name (e.g. "slack", "linear")
        display_name: Optional human-readable name (defaults to name)
    """
    canonical = name.lower().strip()
    _REGISTERED_SOURCE_TYPES[canonical] = display_name or canonical
    # Also register as its own alias
    SOURCE_TYPE_ALIASES[canonical] = canonical


def register_source_alias(alias: str, canonical: str) -> None:
    """Register an alias that maps to a canonical source type.

    Args:
        alias: Alias string (e.g. "slack_message", "github_pr")
        canonical: Canonical source type name (e.g. "slack", "github")
    """
    SOURCE_TYPE_ALIASES[alias.lower().strip()] = canonical.lower().strip()


def normalize_source_type(source_type: str) -> str:
    """Normalize a loose source type string to a canonical source name.

    Returns the canonical string rather than an enum value to support
    dynamically registered source types.

    Args:
        source_type: Free-form source type string from ingestion

    Returns:
        Canonical source type string, "unknown" if not recognized
    """
    if not source_type:
        return SourceTool.UNKNOWN.value
    return SOURCE_TYPE_ALIASES.get(source_type.lower().strip(), SourceTool.UNKNOWN.value)


def is_known_source(source_type: str) -> bool:
    """Check whether a source type is registered (built-in or dynamic).

    Args:
        source_type: Source type string to check

    Returns:
        True if the source type or an alias for it is registered
    """
    normalized = source_type.lower().strip()
    return normalized in SOURCE_TYPE_ALIASES or normalized in _REGISTERED_SOURCE_TYPES

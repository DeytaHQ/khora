"""Shared CLI utilities: output formatting, config resolution, namespace helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger

# Exit codes
EXIT_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_CONFIG_ERROR = 2
EXIT_CONNECTION_ERROR = 3


def detect_output_format(explicit_format: str | None) -> str:
    """Return 'json' or 'text'. Auto-detect based on TTY when not explicit."""
    if explicit_format:
        return explicit_format
    return "text" if sys.stdout.isatty() else "json"


def write_json(data: dict[str, Any]) -> None:
    """Write JSON to stdout."""
    json.dump(data, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def write_text(lines: list[str]) -> None:
    """Write plain text lines to stdout."""
    for line in lines:
        sys.stdout.write(line + "\n")


def resolve_lake_kwargs(
    *,
    config_path: str | None = None,
    database_url: str | None = None,
    graph_url: str | None = None,
    model: str | None = None,
    engine: str = "vectorcypher",
) -> dict[str, Any]:
    """Build kwargs dict suitable for ``MemoryLake(...)`` constructor.

    Priority: CLI flags > env vars > config file > defaults.
    Returns a dict that can be unpacked into ``MemoryLake(**kwargs)``.
    """
    import os

    from khora.config.schema import KhoraConfig

    # Start with config file if provided
    if config_path and Path(config_path).exists():
        config = KhoraConfig.from_yaml(config_path)
    else:
        # Picks up KHORA_* env vars automatically via pydantic-settings
        config = KhoraConfig()

    # CLI flags override env/config
    db_url = database_url or os.environ.get("KHORA_DATABASE_URL") or config.database_url
    neo4j = graph_url or os.environ.get("KHORA_NEO4J_URL") or config.neo4j_url

    if model:
        config.llm.model = model

    kwargs: dict[str, Any] = {"engine": engine}
    if db_url:
        kwargs["database_url"] = db_url
    else:
        # Pass config object so MemoryLake uses all its settings
        kwargs["database_url"] = config
    if neo4j:
        kwargs["graph_url"] = neo4j

    return kwargs


async def resolve_namespace(lake: Any, namespace_arg: str | None) -> UUID:
    """Resolve namespace argument to a namespace_id UUID.

    If *namespace_arg* looks like a UUID, verify it exists.
    If None, create a new namespace.
    """
    if namespace_arg:
        try:
            ns_id = UUID(namespace_arg)
        except ValueError:
            # Not a UUID — treat as opaque; create new namespace
            logger.debug("Namespace arg '{}' is not a UUID, creating new namespace", namespace_arg)
            ns = await lake.create_namespace()
            return ns.namespace_id

        existing = await lake.get_namespace_by_stable_id(ns_id)
        if existing:
            return existing.namespace_id
        # UUID provided but doesn't exist — create with that ID not possible,
        # just create a fresh one and warn
        logger.warning("Namespace {} not found, creating new one", ns_id)

    ns = await lake.create_namespace()
    return ns.namespace_id

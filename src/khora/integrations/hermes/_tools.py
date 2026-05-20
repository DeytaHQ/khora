"""Hermes tool schemas and dispatch for khora-backed memory.

Hermes discovers a plugin's tools by calling ``get_tool_schemas()`` and
dispatches them through the plugin runtime. This module owns the two
schemas the khora plugin exposes and the dispatch functions that turn a
schema-validated arguments dict into a ``kb.recall`` call plus a
plain-text response formatted for the LLM.

Two tools, deliberately distinct so the model picks the right one:

* ``memory_search`` — semantic-only search, no time filter.
* ``memory_recall`` — same plus ``before`` / ``after`` ISO 8601 bounds
  for "what did we discuss last week" style questions.

Schemas follow Hermes's OpenAI-style envelope
(``{"name", "description", "input_schema"}``) — same shape Honcho and
Mem0 plugins use in the hermes-agent reference plugins.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.integrations.hermes._mapping import format_memory_context
from khora.telemetry import bounded_text_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from khora.khora import Khora


_TOP_K_MAX = 50
_TOP_K_DEFAULT = 10
_MIN_SIM_DEFAULT = 0.1


def memory_search_schema() -> dict[str, Any]:
    """Return the JSON schema for the ``memory_search`` tool."""
    return {
        "name": "memory_search",
        "description": (
            "Search the user's long-term memory for context relevant to the current "
            "question. Use this whenever the user references prior conversations, "
            "facts they've shared, or projects/people you should know about. Returns "
            "ranked memory snippets with timestamps. Use `memory_recall` instead when "
            "the user constrains the question to a specific time range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what to find — the exact phrasing the user used works well."
                    ),
                    "minLength": 1,
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of memory snippets to return "
                        f"(default {_TOP_K_DEFAULT}, hard cap {_TOP_K_MAX})."
                    ),
                    "minimum": 1,
                    "maximum": _TOP_K_MAX,
                    "default": _TOP_K_DEFAULT,
                },
                "min_similarity": {
                    "type": "number",
                    "description": (
                        "Lower bound on cosine similarity (0.0–1.0). Raise it to "
                        "drop weak matches; leave at the default for recall-heavy use."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": _MIN_SIM_DEFAULT,
                },
            },
            "required": ["query"],
        },
    }


def memory_recall_schema() -> dict[str, Any]:
    """Return the JSON schema for the ``memory_recall`` tool."""
    base = memory_search_schema()
    properties = dict(base["input_schema"]["properties"])
    properties["before"] = {
        "type": "string",
        "description": (
            "ISO 8601 datetime upper bound (inclusive). Use when the user asks "
            "about something that happened before a specific date."
        ),
        "format": "date-time",
    }
    properties["after"] = {
        "type": "string",
        "description": (
            "ISO 8601 datetime lower bound (inclusive). Use when the user asks "
            "about something that happened after a specific date."
        ),
        "format": "date-time",
    }
    return {
        "name": "memory_recall",
        "description": (
            "Search the user's long-term memory restricted to a time window. Prefer "
            "this over `memory_search` whenever the question carries a temporal "
            "anchor — 'last week', 'since the meeting', 'before March', a specific "
            "date. Pass `after`, `before`, or both as ISO 8601 datetimes."
        ),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["query"],
        },
    }


def _validated_args(args: dict[str, Any]) -> tuple[str, int, float]:
    """Validate the args shared by both tools. Raises ``ValueError`` on bad input."""
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("`query` is required and must be a non-empty string")

    top_k = args.get("top_k", _TOP_K_DEFAULT)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"`top_k` must be an integer, got {type(top_k).__name__}")
    if top_k < 1 or top_k > _TOP_K_MAX:
        raise ValueError(f"`top_k` must be between 1 and {_TOP_K_MAX}, got {top_k}")

    min_sim = args.get("min_similarity", _MIN_SIM_DEFAULT)
    if not isinstance(min_sim, (int, float)) or isinstance(min_sim, bool):
        raise ValueError(f"`min_similarity` must be a number, got {type(min_sim).__name__}")
    if not 0.0 <= float(min_sim) <= 1.0:
        raise ValueError(f"`min_similarity` must be in [0.0, 1.0], got {min_sim}")

    return query.strip(), top_k, float(min_sim)


def _parse_iso(value: Any, name: str) -> datetime | None:
    """Parse an optional ISO 8601 datetime arg. Returns ``None`` if absent."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be an ISO 8601 datetime string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"`{name}` is not a valid ISO 8601 datetime: {value}") from exc


async def dispatch_memory_search(kb: Khora, namespace_id: UUID, args: dict[str, Any]) -> str:
    """Run a semantic ``memory_search`` and return formatted text for the LLM."""
    query, top_k, min_sim = _validated_args(args)
    logger.debug("hermes.memory_search query_hash={} top_k={}", bounded_text_hash(query), top_k)
    result = await kb.recall(
        query,
        namespace=namespace_id,
        limit=top_k,
        min_similarity=min_sim,
    )
    return format_memory_context(list(result.chunks), list(result.entities))


async def dispatch_memory_recall(kb: Khora, namespace_id: UUID, args: dict[str, Any]) -> str:
    """Run a temporal-bounded ``memory_recall`` and return formatted text for the LLM."""
    query, top_k, min_sim = _validated_args(args)
    after = _parse_iso(args.get("after"), "after")
    before = _parse_iso(args.get("before"), "before")
    if after is not None and before is not None and after > before:
        raise ValueError("`after` must be <= `before`")
    logger.debug(
        "hermes.memory_recall query_hash={} top_k={} after={} before={}",
        bounded_text_hash(query),
        top_k,
        after.isoformat() if after else None,
        before.isoformat() if before else None,
    )
    result = await kb.recall(
        query,
        namespace=namespace_id,
        limit=top_k,
        min_similarity=min_sim,
        start_time=after,
        end_time=before,
    )
    return format_memory_context(list(result.chunks), list(result.entities))


__all__ = [
    "dispatch_memory_recall",
    "dispatch_memory_search",
    "memory_recall_schema",
    "memory_search_schema",
]

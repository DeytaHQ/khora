"""Trace propagation via contextvars.

Provides request-level trace_id and parent_event_id propagation through
the async call stack without threading parameters through every function.

Also provides a request-scoped LLMUsage accumulator via asyncio.Queue
for collecting usage data across concurrent asyncio.gather() calls.

Usage::

    from khora.telemetry.context import set_trace_id, ensure_trace_id

    # At request entry point (remember/recall/chat):
    set_trace_id(uuid4())

    # Downstream code automatically inherits trace_id:
    trace_id = get_trace_id()  # returns the UUID set above

    # Usage accumulation:
    from khora.telemetry.context import start_usage_collection, record_usage, collect_usage

    start_usage_collection()
    # ... LLM calls record_usage(usage) internally ...
    entries = collect_usage()  # drains queue, resets contextvar
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from khora.khora import LLMUsage

_trace_id_var: ContextVar[UUID | None] = ContextVar("khora_trace_id", default=None)
_parent_event_id_var: ContextVar[int | None] = ContextVar("khora_parent_event_id", default=None)
_usage_accumulator: ContextVar[asyncio.Queue[LLMUsage] | None] = ContextVar("khora_usage_accumulator", default=None)


def get_trace_id() -> UUID | None:
    """Get the current trace ID, or None if not set."""
    return _trace_id_var.get()


def set_trace_id(trace_id: UUID | None) -> None:
    """Set the trace ID for the current async context."""
    _trace_id_var.set(trace_id)


def ensure_trace_id() -> UUID:
    """Get or create a trace ID for the current context."""
    tid = _trace_id_var.get()
    if tid is None:
        tid = uuid4()
        _trace_id_var.set(tid)
    return tid


def clear_trace_id() -> None:
    """Clear the trace ID (call at end of request)."""
    _trace_id_var.set(None)


def get_parent_event_id() -> int | None:
    """Get the current parent event ID, or None."""
    return _parent_event_id_var.get()


def set_parent_event_id(event_id: int | None) -> None:
    """Set the parent event ID for child event linking."""
    _parent_event_id_var.set(event_id)


def clear_parent_event_id() -> None:
    """Clear the parent event ID."""
    _parent_event_id_var.set(None)


# ---------------------------------------------------------------------------
# Request-scoped LLM usage accumulator
# ---------------------------------------------------------------------------


def start_usage_collection() -> None:
    """Create a new asyncio.Queue in the contextvar for this request scope.

    Call at the start of a Khora facade method (remember, recall, etc.).
    """
    _usage_accumulator.set(asyncio.Queue())


def record_usage(usage: LLMUsage) -> None:
    """Record a single LLMUsage entry into the current request's queue.

    No-op if ``start_usage_collection()`` was not called (i.e. when Khora
    internals are invoked outside the Khora facade).

    ``put_nowait()`` is safe under cooperative multitasking (asyncio.gather).
    """
    q = _usage_accumulator.get()
    if q is not None:
        q.put_nowait(usage)


def collect_usage() -> list[LLMUsage]:
    """Drain the queue and reset the contextvar.

    Returns an empty list if collection was never started.
    """
    q = _usage_accumulator.get()
    _usage_accumulator.set(None)
    if q is None:
        return []
    entries: list[LLMUsage] = []
    while not q.empty():
        entries.append(q.get_nowait())
    return entries

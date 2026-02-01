"""Trace propagation via contextvars.

Provides request-level trace_id and parent_event_id propagation through
the async call stack without threading parameters through every function.

Usage::

    from khora.telemetry.context import set_trace_id, ensure_trace_id

    # At request entry point (remember/recall/chat):
    set_trace_id(uuid4())

    # Downstream code automatically inherits trace_id:
    trace_id = get_trace_id()  # returns the UUID set above
"""

from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID, uuid4

_trace_id_var: ContextVar[UUID | None] = ContextVar("khora_trace_id", default=None)
_parent_event_id_var: ContextVar[int | None] = ContextVar("khora_parent_event_id", default=None)


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

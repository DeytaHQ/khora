"""Optional OTEL span integration for Khora telemetry.

When ``logfire`` is installed, :func:`trace_span` emits real OpenTelemetry
spans via the Logfire SDK, wrapped in a :class:`LogfireSpan`.  Otherwise it
yields a :class:`NoOpSpan` that silently discards all attribute writes.

All consumers interact exclusively with the :class:`Span` base class,
keeping the logfire dependency fully encapsulated.

Install with::

    pip install khora[logfire]
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    import logfire as _logfire

    _HAS_LOGFIRE = True
except ImportError:
    _logfire = None
    _HAS_LOGFIRE = False


__all__ = [
    "Span",
    "NoOpSpan",
    "LogfireSpan",
    "trace_span",
    "install_neo4j_logfire_handler",
]

# Marker attribute used to identify neo4j logfire handlers that khora owns so
# repeated ``install_neo4j_logfire_handler()`` calls can strip stale handlers
# before attaching a fresh one (idempotent behaviour).
_NEO4J_LOGFIRE_HANDLER_MARK = "_khora_neo4j_logfire_handler"


class Span(ABC):
    """Abstract base for telemetry spans."""

    __slots__ = ()

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def set_attributes(self, attributes: dict[str, Any]) -> None: ...


class NoOpSpan(Span):
    """Span that silently discards all writes."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:  # noqa: ARG002
        pass


class LogfireSpan(Span):
    """Span that delegates to a real Logfire span."""

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def set_attribute(self, key: str, value: Any) -> None:
        self._inner.set_attribute(key, value)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        for key, value in attributes.items():
            self._inner.set_attribute(key, value)


_NOOP_SPAN = NoOpSpan()


@contextmanager
def trace_span(name: str, /, **attributes: Any) -> Iterator[Span]:
    """Emit an OTEL span if an exporter is installed, otherwise no-op."""
    if _HAS_LOGFIRE:
        with _logfire.span(name, **attributes) as span:
            yield LogfireSpan(span)
    else:
        yield _NOOP_SPAN


def install_neo4j_logfire_handler() -> bool:
    """Attach a ``LogfireLoggingHandler`` to the ``neo4j`` stdlib logger.

    Neo4j driver DEBUG records (enabled by ``KHORA_NEO4J_LOG_LEVEL``) are
    emitted through Python's stdlib ``logging``. Khora's loguru sinks only
    pick records up at the level of the main sink (typically INFO), so
    driver DEBUG records are dropped by the sink filter. Attaching a
    dedicated ``LogfireLoggingHandler`` directly on the ``neo4j`` logger
    bypasses the sink-level filter — records reach Logfire alongside the
    existing stderr handler installed by downstream services (DYT-2721).

    No-op when ``logfire`` is not installed, or when
    ``KHORA_NEO4J_LOG_LEVEL`` is unset/empty. Idempotent: removes any
    previously attached marked handler before adding a new one, so
    repeated calls do not stack duplicates.

    Returns True when a handler was attached, False otherwise.
    """
    if not _HAS_LOGFIRE:
        return False
    if not os.environ.get("KHORA_NEO4J_LOG_LEVEL"):
        return False
    from logfire.integrations.logging import LogfireLoggingHandler

    neo4j_logger = logging.getLogger("neo4j")
    for existing in list(neo4j_logger.handlers):
        if getattr(existing, _NEO4J_LOGFIRE_HANDLER_MARK, False):
            neo4j_logger.removeHandler(existing)
    handler = LogfireLoggingHandler()
    setattr(handler, _NEO4J_LOGFIRE_HANDLER_MARK, True)
    neo4j_logger.addHandler(handler)
    return True

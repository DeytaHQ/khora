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

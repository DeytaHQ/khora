"""Optional OTEL span integration for Khora telemetry.

When ``logfire`` is installed, :func:`trace_span` emits real OpenTelemetry
spans via the Logfire SDK.  Otherwise it yields a :class:`NoOpSpan` that
silently discards all attribute writes (zero-cost no-op).

Install with::

    pip install khora[logfire]
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    import logfire as _logfire

    _HAS_LOGFIRE = True
except ImportError:
    _logfire = None
    _HAS_LOGFIRE = False


class NoOpSpan:
    """Dummy span that silently discards all writes."""

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:  # noqa: ARG002
        pass


_NOOP_SPAN = NoOpSpan()


@contextmanager
def trace_span(name: str, /, **attributes: Any) -> Iterator[NoOpSpan]:
    """Emit an OTEL span if an exporter is installed, otherwise no-op."""
    if _HAS_LOGFIRE:
        with _logfire.span(name, **attributes) as span:
            yield span
    else:
        yield _NOOP_SPAN

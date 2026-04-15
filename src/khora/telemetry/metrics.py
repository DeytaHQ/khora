"""OTel metric wrappers with no-op fallbacks.

When ``logfire`` is installed the helpers delegate to real OTel instruments.
Otherwise they return lightweight no-op objects so callers never need to
check availability themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .logfire_integration import _HAS_LOGFIRE, _logfire

# --- No-op stubs ---


class _NoOpCounter:
    __slots__ = ()

    def add(self, amount: int | float, attributes: Any = None) -> None: ...


class _NoOpHistogram:
    __slots__ = ()

    def record(self, amount: int | float, attributes: Any = None) -> None: ...


_NOOP_COUNTER = _NoOpCounter()
_NOOP_HISTOGRAM = _NoOpHistogram()


# --- Public helpers ---


def metric_counter(name: str, *, unit: str = "", description: str = "") -> Any:
    if _HAS_LOGFIRE:
        return _logfire.metric_counter(name, unit=unit, description=description)
    return _NOOP_COUNTER


def metric_histogram(name: str, *, unit: str = "", description: str = "") -> Any:
    if _HAS_LOGFIRE:
        return _logfire.metric_histogram(name, unit=unit, description=description)
    return _NOOP_HISTOGRAM


def metric_gauge_callback(name: str, callbacks: Sequence[Any], *, unit: str = "", description: str = "") -> None:
    if _HAS_LOGFIRE:
        _logfire.metric_gauge_callback(name, callbacks, unit=unit, description=description)

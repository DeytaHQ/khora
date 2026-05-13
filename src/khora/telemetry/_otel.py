"""OpenTelemetry tracer + meter wiring for khora.

khora emits spans and metrics through the OpenTelemetry API. The actual
delivery (OTLP to a collector, logfire, in-memory test exporter,
nothing) is determined by which ``TracerProvider`` / ``MeterProvider``
is installed globally — see :mod:`khora.telemetry.bootstrap` for the
configuration entry point.

This module is the single place that talks to the OTel API. Everything
else in khora calls :func:`trace_span` and the helpers in
:mod:`khora.telemetry.metrics` — they don't need to know whether the
underlying SDK is logfire, vanilla OTel SDK, or nothing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from opentelemetry import metrics as _otel_metrics
from opentelemetry import trace as _otel_trace

__all__ = [
    "trace_span",
    "get_tracer",
    "get_meter",
    "install_neo4j_log_bridge",
]


_NEO4J_LOG_BRIDGE_MARK = "_khora_neo4j_log_bridge"


def _khora_version() -> str:
    try:
        return version("khora")
    except PackageNotFoundError:
        return "0.0.0+unknown"


_KHORA_VERSION = _khora_version()
_TRACER = _otel_trace.get_tracer("khora", _KHORA_VERSION)
_METER = _otel_metrics.get_meter("khora", _KHORA_VERSION)


def get_tracer() -> _otel_trace.Tracer:
    """Return khora's OTel tracer (scope ``khora``, version = pkg version)."""
    return _TRACER


def get_meter() -> _otel_metrics.Meter:
    """Return khora's OTel meter (scope ``khora``, version = pkg version)."""
    return _METER


@contextmanager
def trace_span(name: str, /, **attributes: Any) -> Iterator[_otel_trace.Span]:
    """Open an OTel span with *attributes*.

    Yields the real :class:`opentelemetry.trace.Span` (which has
    ``set_attribute`` and ``set_attributes``). When no real
    ``TracerProvider`` is installed, the OTel API returns a
    ``NonRecordingSpan`` and the body becomes effectively free.

    Attribute values must be OTel-permitted scalar types (str, int,
    float, bool) or sequences thereof. Free-text inputs should be
    pre-hashed via :func:`khora.telemetry.bounded_text_hash` to keep
    cardinality bounded.
    """
    with _TRACER.start_as_current_span(name, attributes=attributes) as span:
        yield span


def install_neo4j_log_bridge() -> bool:
    """Attach a logging handler to the ``neo4j`` stdlib logger.

    The neo4j Python driver emits DEBUG records (enabled by
    ``KHORA_NEO4J_LOG_LEVEL``) through Python's stdlib ``logging``.
    Khora's loguru sinks filter at the main sink level (typically INFO),
    so driver DEBUG records are dropped by default. Attaching a
    dedicated handler directly on the ``neo4j`` logger bypasses the
    sink-level filter.

    Preference order for the handler:

    1. If ``logfire`` is importable, use ``LogfireLoggingHandler`` — it
       emits records as OTel logs through the logfire processor chain.
    2. If the OTel logs SDK is importable, use it with a handler bound
       to the global ``LoggerProvider`` (set up by
       :func:`khora.telemetry.bootstrap.configure_telemetry`). This is
       the vanilla-OTel path.
    3. Otherwise, no bridge is attached and the function returns False.

    No-op when ``KHORA_NEO4J_LOG_LEVEL`` is unset/empty. Idempotent:
    removes any previously attached marked handler before adding a new
    one, so repeated calls do not stack duplicates.

    Returns True when a handler was attached, False otherwise.
    """
    if not os.environ.get("KHORA_NEO4J_LOG_LEVEL"):
        return False

    handler: logging.Handler | None = None
    try:
        from logfire.integrations.logging import LogfireLoggingHandler

        handler = LogfireLoggingHandler()
    except ImportError:
        try:
            from opentelemetry.sdk._logs import LoggingHandler

            handler = LoggingHandler()
        except ImportError:
            return False

    if handler is None:
        return False

    setattr(handler, _NEO4J_LOG_BRIDGE_MARK, True)
    neo4j_logger = logging.getLogger("neo4j")
    for existing in list(neo4j_logger.handlers):
        if getattr(existing, _NEO4J_LOG_BRIDGE_MARK, False):
            neo4j_logger.removeHandler(existing)
    neo4j_logger.addHandler(handler)
    return True

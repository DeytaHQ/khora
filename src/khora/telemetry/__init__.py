"""Khora telemetry — spans, metrics, and structured event recording.

khora emits spans and metrics through the OpenTelemetry API. The
export path (vanilla OTel, logfire, or nothing) is configured via
:func:`configure_telemetry`. See :mod:`khora.telemetry.bootstrap` for
the precedence rules and the supported backends.

Quick start (vanilla OTel)::

    pip install khora[otel]
    # then, in your app startup:
    from khora.telemetry import configure_telemetry
    configure_telemetry()  # honors OTEL_* env vars

Quick start (logfire)::

    pip install khora[logfire]
    import logfire
    logfire.configure()    # khora picks up the provider automatically

Khora also records structured ``LLMEvent`` / ``StorageEvent`` /
``PipelineEvent`` rows to PostgreSQL when ``KHORA_TELEMETRY_DATABASE_URL``
is set; otherwise a zero-cost :class:`NoOpCollector` is used.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ._attrs import bounded_text_hash
from ._otel import install_neo4j_log_bridge, trace_span
from .bootstrap import (
    Backend,
    TelemetryHandle,
    configure_telemetry,
    diagnostics,
    shutdown_telemetry_providers,
)
from .collector import TelemetryCollector
from .context import (
    clear_trace_id,
    collect_usage,
    ensure_trace_id,
    get_trace_id,
    record_usage,
    set_trace_id,
    start_usage_collection,
)
from .metrics import metric_counter, metric_gauge_callback, metric_histogram
from .noop import NoOpCollector
from .trace_decorator import trace

if TYPE_CHECKING:
    from .config import TelemetryConfig

__all__ = [
    "TelemetryCollector",
    "NoOpCollector",
    "get_collector",
    "init_telemetry",
    "shutdown_telemetry",
    "get_trace_id",
    "set_trace_id",
    "ensure_trace_id",
    "clear_trace_id",
    "start_usage_collection",
    "record_usage",
    "collect_usage",
    "trace_span",
    "bounded_text_hash",
    "install_neo4j_log_bridge",
    "trace",
    "metric_counter",
    "metric_gauge_callback",
    "metric_histogram",
    "Backend",
    "TelemetryHandle",
    "configure_telemetry",
    "shutdown_telemetry_providers",
    "diagnostics",
]


def __getattr__(name: str) -> Any:
    """Deprecation shim for renamed exports.

    Currently used to keep ``install_neo4j_logfire_handler`` importable
    for one minor release after the rename to
    :func:`install_neo4j_log_bridge`. Slated for removal in khora 0.12.
    """
    if name == "install_neo4j_logfire_handler":
        warnings.warn(
            "khora.telemetry.install_neo4j_logfire_handler has been renamed to "
            "install_neo4j_log_bridge. The old name is kept as an alias for one "
            "minor release and will be removed in khora 0.12.",
            DeprecationWarning,
            stacklevel=2,
        )
        return install_neo4j_log_bridge
    raise AttributeError(f"module 'khora.telemetry' has no attribute {name!r}")


_collector: TelemetryCollector | NoOpCollector = NoOpCollector()


def get_collector() -> TelemetryCollector | NoOpCollector:
    """Return the current global telemetry collector."""
    return _collector


async def init_telemetry(config: TelemetryConfig | None = None) -> TelemetryCollector | NoOpCollector:
    """Initialise the structured event collector.

    Separate from :func:`configure_telemetry`, which wires the
    span/metric export path. ``init_telemetry`` connects to the
    PostgreSQL-backed event collector when ``KHORA_TELEMETRY_DATABASE_URL``
    is set; otherwise a :class:`NoOpCollector` is returned.
    """
    global _collector

    if config is None:
        from .config import TelemetryConfig

        config = TelemetryConfig.from_env()

    if not config.database_url:
        _collector = NoOpCollector()
        return _collector

    from .session import create_telemetry_engine

    # TelemetryConfig.database_url is a SecretStr; unwrap exactly here so
    # the SQLAlchemy engine receives the plaintext DSN.
    database_url = config.database_url.get_secret_value()
    engine = create_telemetry_engine(database_url)
    _collector = TelemetryCollector(
        engine,
        service_name=config.service_name,
        flush_interval=config.flush_interval_seconds,
        flush_threshold=config.flush_threshold,
    )
    await _collector.start()
    return _collector


async def shutdown_telemetry() -> None:
    """Shut down the structured event collector (final flush + engine dispose)."""
    global _collector
    await _collector.shutdown()
    _collector = NoOpCollector()

"""Khora internal telemetry.

Records LLM usage, storage operations, and pipeline performance to a
dedicated PostgreSQL database.  Enabled when ``KHORA_TELEMETRY_DATABASE_URL``
is set; otherwise a zero-cost :class:`NoOpCollector` is used.

Quick start::

    from khora.telemetry import get_collector

    get_collector().record_llm_call(
        operation="entity_extraction",
        model="gpt-4o-mini",
        prompt_tokens=120,
        completion_tokens=350,
        total_tokens=470,
        latency_ms=812.3,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from .logfire_integration import trace_span
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
    "trace",
]

_collector: TelemetryCollector | NoOpCollector = NoOpCollector()


def get_collector() -> TelemetryCollector | NoOpCollector:
    """Return the current global telemetry collector."""
    return _collector


async def init_telemetry(config: TelemetryConfig | None = None) -> TelemetryCollector | NoOpCollector:
    """Initialise the telemetry subsystem.

    If *config* is ``None`` the configuration is read from environment
    variables.  When no ``KHORA_TELEMETRY_DATABASE_URL`` is found a
    :class:`NoOpCollector` is returned.
    """
    global _collector

    if config is None:
        from .config import TelemetryConfig

        config = TelemetryConfig.from_env()

    if not config.database_url:
        _collector = NoOpCollector()
        return _collector

    from .session import create_telemetry_engine

    engine = create_telemetry_engine(config.database_url)
    _collector = TelemetryCollector(
        engine,
        service_name=config.service_name,
        flush_interval=config.flush_interval_seconds,
        flush_threshold=config.flush_threshold,
    )
    await _collector.start()
    return _collector


async def shutdown_telemetry() -> None:
    """Shut down the telemetry collector (final flush + engine dispose)."""
    global _collector
    await _collector.shutdown()
    _collector = NoOpCollector()

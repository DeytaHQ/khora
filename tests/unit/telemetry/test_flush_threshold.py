"""flush_threshold triggers an immediate flush without waiting for the timer (#934)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from khora.telemetry.collector import TelemetryCollector


def _make_collector(threshold: int) -> TelemetryCollector:
    # _flush is patched out, so the engine is never touched; pass a sentinel.
    collector = TelemetryCollector(
        engine=object(),  # type: ignore[arg-type]
        flush_interval=3600.0,  # effectively disables the timer for this test
        flush_threshold=threshold,
    )
    collector._flush = AsyncMock()  # type: ignore[method-assign]
    return collector


@pytest.mark.asyncio
async def test_record_triggers_flush_at_threshold() -> None:
    collector = _make_collector(threshold=3)

    collector.record_storage_op(operation="read", backend="pg", latency_ms=1.0)
    collector.record_storage_op(operation="read", backend="pg", latency_ms=1.0)
    # Below threshold: no flush scheduled yet.
    assert collector._threshold_flush_task is None

    collector.record_storage_op(operation="read", backend="pg", latency_ms=1.0)
    # Threshold hit: a flush task is scheduled.
    assert collector._threshold_flush_task is not None

    await collector._threshold_flush_task
    collector._flush.assert_awaited_once()

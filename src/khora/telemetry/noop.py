"""No-op telemetry collector -- zero overhead when telemetry is disabled."""

from __future__ import annotations

from typing import Any


class NoOpCollector:
    """Drop-in replacement for TelemetryCollector that does nothing."""

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def record_llm_call(self, **kwargs: Any) -> None:
        pass

    def record_storage_op(self, **kwargs: Any) -> None:
        pass

    def record_pipeline_stage(self, **kwargs: Any) -> None:
        pass

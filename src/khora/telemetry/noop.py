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
        # Aggregate OTel metrics still fire even when DB telemetry is
        # disabled — they're a separate transport (logfire/OTel exporter)
        # and provide SLO signals for downstream consumers.
        from .aggregate_metrics import record_llm_call_metrics

        record_llm_call_metrics(
            model=kwargs.get("model", ""),
            operation=kwargs.get("operation", ""),
            status=kwargs.get("status", "success"),
            prompt_tokens=kwargs.get("prompt_tokens", 0) or 0,
            completion_tokens=kwargs.get("completion_tokens", 0) or 0,
            cost_usd=kwargs.get("cost_usd"),
        )

    def record_storage_op(self, **kwargs: Any) -> None:
        pass

    def record_pipeline_stage(self, **kwargs: Any) -> None:
        pass

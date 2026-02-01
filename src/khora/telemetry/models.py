"""Pydantic models for telemetry events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class LLMEvent(BaseModel):
    """A single LLM API call event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service_name: str = "khora"
    operation: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    status: str = "success"
    error_message: str | None = None
    namespace_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Tracing
    trace_id: UUID | None = None
    parent_event_id: int | None = None
    cache_hit: bool = False
    batch_size: int = 1


class StorageEvent(BaseModel):
    """A single storage operation event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service_name: str = "khora"
    backend: str = ""
    operation: str = ""
    latency_ms: float = 0.0
    record_count: int = 0
    status: str = "success"
    error_message: str | None = None
    namespace_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Tracing
    trace_id: UUID | None = None
    parent_event_id: int | None = None


class PipelineEvent(BaseModel):
    """A single pipeline stage event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service_name: str = "khora"
    pipeline: str = ""
    stage: str = ""
    run_id: UUID | None = None
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    status: str = "success"
    error_message: str | None = None
    namespace_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Tracing
    trace_id: UUID | None = None
    parent_event_id: int | None = None

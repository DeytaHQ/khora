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
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    error_message: str | None = None
    namespace_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class PipelineEvent(BaseModel):
    """A single pipeline stage event."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service_name: str = "khora"
    pipeline: str = ""
    stage: str = ""
    run_id: UUID | None = None
    latency_ms: float = 0.0
    record_count: int = 0
    status: str = "success"
    error_message: str | None = None
    namespace_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

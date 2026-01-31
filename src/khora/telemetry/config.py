"""Telemetry configuration."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class TelemetryConfig(BaseModel):
    """Configuration for the telemetry subsystem."""

    database_url: str | None = Field(
        default=None,
        description="PostgreSQL URL for telemetry database",
    )
    service_name: str = Field(
        default="khora",
        description="Service name tag for telemetry events",
    )
    flush_interval_seconds: float = Field(default=5.0, description="Seconds between background flushes")
    flush_threshold: int = Field(default=100, description="Buffer size that triggers an immediate flush")

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        """Build config from environment variables."""
        return cls(
            database_url=os.getenv("KHORA_TELEMETRY_DATABASE_URL"),
            service_name=os.getenv("KHORA_TELEMETRY_SERVICE_NAME", "khora"),
        )

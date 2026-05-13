"""Telemetry configuration."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class TelemetryConfig(BaseModel):
    """Configuration for the telemetry subsystem."""

    database_url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL URL for telemetry database (SecretStr)",
    )
    service_name: str = Field(
        default="khora",
        description="Service name tag for telemetry events",
    )
    flush_interval_seconds: float = Field(default=5.0, description="Seconds between background flushes")
    flush_threshold: int = Field(default=100, description="Buffer size that triggers an immediate flush")
    # ADR-084 migration window: "warn" emits WARNING logs for str-typed secrets;
    # "fail" raises on startup. Removed by D3-C once all fields are re-typed.
    secret_typing_mode: Literal["warn", "fail"] = Field(
        default="warn",
        description="ADR-084 secret typing enforcement mode (warn|fail)",
    )

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        """Build config from environment variables."""
        return cls(
            database_url=os.getenv("KHORA_TELEMETRY_DATABASE_URL"),
            service_name=os.getenv("KHORA_TELEMETRY_SERVICE_NAME", "khora"),
            secret_typing_mode=os.getenv("KHORA_TELEMETRY_SECRET_TYPING_MODE", "warn"),  # type: ignore[arg-type]
        )

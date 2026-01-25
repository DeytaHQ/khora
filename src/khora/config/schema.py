"""Pydantic configuration models for Khora."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KhoraConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="KHORA_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = Field(
        default="khora",
        description="Application name",
    )
    environment: str = Field(
        default="development",
        description="Environment: development, staging, or production",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Authentication settings
    auth_enabled: bool = Field(
        default=True,
        description="Enable authentication (set to False for local development)",
    )

    # API settings
    api_host: str = Field(
        default="127.0.0.1",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        description="API server port",
    )

    # Database for Khora internal state
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL URL for Khora database",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> KhoraConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            KhoraConfig instance
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

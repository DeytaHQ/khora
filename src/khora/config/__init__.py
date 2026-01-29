"""Configuration module for Khora."""

from .llm import LiteLLMConfig, acompletion, aembedding, configure_litellm, create_litellm_router
from .schema import (
    KhoraConfig,
    LLMSettings,
    PipelineSettings,
    QuerySettings,
    StorageSettings,
    TenancySettings,
)

# Default config path
DEFAULT_CONFIG_PATH = "config/khora.yaml"


def load_config(path: str | None = None) -> KhoraConfig:
    """Load configuration from file or environment.

    Args:
        path: Optional path to YAML configuration file

    Returns:
        KhoraConfig instance
    """
    import os
    from pathlib import Path

    if path:
        return KhoraConfig.from_yaml(Path(path))

    # Try default paths
    config_path = os.getenv("KHORA_CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if Path(config_path).exists():
        return KhoraConfig.from_yaml(Path(config_path))

    # Fall back to environment variables only
    return KhoraConfig()


__all__ = [
    # Main config
    "KhoraConfig",
    "load_config",
    # Config sections
    "StorageSettings",
    "LLMSettings",
    "PipelineSettings",
    "TenancySettings",
    "QuerySettings",
    # LiteLLM
    "LiteLLMConfig",
    "configure_litellm",
    "create_litellm_router",
    "acompletion",
    "aembedding",
]

"""Pydantic configuration models for Khora."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseModel):
    """Storage backend configuration."""

    # PostgreSQL (relational)
    postgresql_url: str | None = Field(default=None, description="PostgreSQL connection URL")

    # pgvector
    pgvector_url: str | None = Field(default=None, description="pgvector connection URL (defaults to postgresql_url)")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")

    # Neo4j
    neo4j_url: str | None = Field(default=None, description="Neo4j connection URL")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")


class LLMSettings(BaseModel):
    """LLM configuration settings."""

    model: str = Field(default="gpt-4o-mini", description="Primary LLM model")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=2000, description="Maximum tokens to generate")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retries on failure")
    max_concurrent_llm_calls: int = Field(default=10, description="Maximum concurrent LLM calls")

    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    embedding_dimension: int = Field(default=1536, description="Embedding dimension")

    # Router configuration
    config_file: str | None = Field(default=None, description="Path to LiteLLM config YAML")
    model_list: list[dict[str, Any]] | None = Field(default=None, description="Model list for router")
    router_settings: dict[str, Any] | None = Field(default=None, description="Router settings")


class PipelineSettings(BaseModel):
    """Pipeline configuration settings."""

    # Chunking settings
    chunking_strategy: str = Field(default="semantic", description="Chunking strategy: fixed, semantic, recursive")
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks in tokens")

    # Extraction settings
    extract_entities: bool = Field(default=True, description="Extract entities from documents")
    entity_types: list[str] = Field(
        default=["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION"],
        description="Entity types to extract",
    )


class TenancySettings(BaseModel):
    """Multi-tenancy configuration settings."""

    default_mode: str = Field(default="shared", description="Default tenancy mode: shared or isolated")
    enforce_namespace: bool = Field(default=True, description="Enforce namespace isolation")


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

    # Database for Khora internal state (shortcut for storage.postgresql_url)
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL URL for Khora database",
    )

    # Storage configuration
    storage: StorageSettings = Field(default_factory=StorageSettings)

    # LLM configuration
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Pipeline configuration
    pipelines: PipelineSettings = Field(default_factory=PipelineSettings)

    # Tenancy configuration
    tenancy: TenancySettings = Field(default_factory=TenancySettings)

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

    def get_postgresql_url(self) -> str | None:
        """Get PostgreSQL URL from config."""
        return self.storage.postgresql_url or self.database_url

    def get_neo4j_url(self) -> str | None:
        """Get Neo4j URL from config."""
        return self.storage.neo4j_url

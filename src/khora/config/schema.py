"""Pydantic configuration models for Khora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class ParsedNeo4jUrl:
    """Parsed Neo4j URL components."""

    url: str  # URL without credentials (bolt://host:port)
    user: str
    password: str
    database: str

    @classmethod
    def parse(cls, url: str, default_user: str = "neo4j", default_database: str = "neo4j") -> ParsedNeo4jUrl:
        """Parse a Neo4j URL with optional embedded credentials.

        Supports formats:
        - bolt://host:port
        - bolt://user:password@host:port
        - bolt://user:password@host:port/database
        """
        parsed = urlparse(url)

        # Extract user and password from URL
        user = parsed.username or default_user
        password = parsed.password or ""

        # Extract database from path (e.g., /mydb -> mydb)
        database = parsed.path.lstrip("/") if parsed.path and parsed.path != "/" else default_database

        # Reconstruct URL without credentials
        host_port = parsed.hostname or "localhost"
        if parsed.port:
            host_port = f"{host_port}:{parsed.port}"
        clean_url = f"{parsed.scheme}://{host_port}"

        return cls(url=clean_url, user=user, password=password, database=database)


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


class QuerySettings(BaseModel):
    """Query pipeline configuration.

    All fields are flat to allow single-underscore env vars, e.g.:
    KHORA_QUERY__DEFAULT_MODE, KHORA_QUERY__ENABLE_RERANKING, etc.
    """

    # Basic search settings
    default_mode: str = Field(default="hybrid", description="Default search mode: vector, graph, hybrid, all")
    min_chunk_similarity: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum chunk similarity threshold")
    min_entity_similarity: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum entity similarity threshold")

    # Fusion weights
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector search in fusion")
    graph_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for graph search in fusion")
    keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for keyword search in fusion")

    # Temporal settings
    apply_recency_bias: bool = Field(default=False, description="Apply recency bias to results")
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight of recency in scoring")
    recency_decay_days: float = Field(default=30.0, ge=1.0, description="Days for recency score to decay by half")

    # Query understanding
    enable_understanding: bool = Field(default=True, description="Enable LLM-based query understanding")
    understanding_expand_query: bool = Field(default=True, description="Generate query expansions/reformulations")
    understanding_extract_entities: bool = Field(default=True, description="Extract entity mentions from query")
    understanding_detect_temporal: bool = Field(default=True, description="Detect temporal references in query")
    understanding_model: str | None = Field(
        default=None, description="Model to use for query understanding (defaults to main LLM)"
    )

    # Entity linking
    enable_entity_linking: bool = Field(default=True, description="Enable entity linking")
    entity_linking_exact_match: bool = Field(default=True, description="Use exact name matching")
    entity_linking_fuzzy_match: bool = Field(default=True, description="Use fuzzy name matching")
    entity_linking_embedding_match: bool = Field(default=True, description="Use embedding similarity matching")
    entity_linking_fuzzy_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum fuzzy match ratio")
    entity_linking_embedding_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum embedding similarity"
    )
    entity_linking_max_candidates: int = Field(default=5, ge=1, description="Maximum entity candidates per mention")

    # Reranking
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    reranking_method: str = Field(default="cross_encoder", description="Reranking method: cross_encoder, llm")
    reranking_model: str | None = Field(default=None, description="Model for reranking (cross-encoder model or LLM)")
    reranking_top_n: int = Field(default=50, ge=1, description="Number of candidates to rerank")
    reranking_final_k: int = Field(default=10, ge=1, description="Number of results after reranking")

    # Keyword search
    enable_keyword_search: bool = Field(default=True, description="Enable keyword search")
    keyword_search_method: str = Field(default="fulltext", description="Keyword search method: bm25, fulltext")
    keyword_search_use_stemming: bool = Field(default=True, description="Apply stemming to search terms")
    keyword_search_use_stopwords: bool = Field(default=True, description="Remove stopwords from search")
    keyword_search_language: str = Field(default="english", description="Language for stemming and stopwords")

    # HyDE
    enable_hyde: bool = Field(default=False, description="Enable HyDE query expansion")
    hyde_num_hypotheticals: int = Field(
        default=1, ge=1, le=5, description="Number of hypothetical documents to generate"
    )


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

    # Database for Khora internal state (shortcuts for storage.* URLs)
    # These can be set via KHORA_DATABASE_URL and KHORA_NEO4J_URL environment variables
    # Programmatic values take priority over environment variables
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL URL for Khora database (shortcut for storage.postgresql_url)",
    )
    neo4j_url: str | None = Field(
        default=None,
        description="Neo4j URL for graph storage (shortcut for storage.neo4j_url)",
    )

    # Storage configuration
    storage: StorageSettings = Field(default_factory=StorageSettings)

    # LLM configuration
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Pipeline configuration
    pipelines: PipelineSettings = Field(default_factory=PipelineSettings)

    # Tenancy configuration
    tenancy: TenancySettings = Field(default_factory=TenancySettings)

    # Query pipeline configuration
    query: QuerySettings = Field(default_factory=QuerySettings)

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

    def _get_raw_neo4j_url(self) -> str | None:
        """Get raw Neo4j URL (may contain credentials)."""
        return self.storage.neo4j_url or self.neo4j_url

    def _parse_neo4j_url(self) -> ParsedNeo4jUrl | None:
        """Parse Neo4j URL and extract components."""
        raw_url = self._get_raw_neo4j_url()
        if not raw_url:
            return None
        return ParsedNeo4jUrl.parse(
            raw_url,
            default_user=self.storage.neo4j_user,
            default_database=self.storage.neo4j_database,
        )

    def get_neo4j_url(self) -> str | None:
        """Get Neo4j URL without credentials (for driver connection).

        Parses URL like bolt://user:pass@host:port and returns bolt://host:port
        """
        parsed = self._parse_neo4j_url()
        return parsed.url if parsed else None

    def get_neo4j_user(self) -> str:
        """Get Neo4j username from URL or config."""
        parsed = self._parse_neo4j_url()
        if parsed:
            return parsed.user
        return self.storage.neo4j_user

    def get_neo4j_password(self) -> str:
        """Get Neo4j password from URL or config."""
        parsed = self._parse_neo4j_url()
        if parsed:
            return parsed.password
        return self.storage.neo4j_password

    def get_neo4j_database(self) -> str:
        """Get Neo4j database from URL or config."""
        parsed = self._parse_neo4j_url()
        if parsed:
            return parsed.database
        return self.storage.neo4j_database

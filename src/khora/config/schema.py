"""Pydantic configuration models for Khora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Discriminator, Field, SecretStr, Tag, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _secret_value(value: SecretStr | str | None, default: str = "") -> str:
    """Return the plain-text value of a ``SecretStr`` (or pass-through ``str``).

    Storage-backend engine factories unwrap ``SecretStr`` exactly once when
    handing credentials to the underlying driver. Accepts ``str`` as a
    back-compat fallback so legacy call sites that still pass plain strings
    continue to work during the migration window.
    """
    if value is None:
        return default
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


@dataclass
class ParsedNeo4jUrl:
    """Parsed Neo4j URL components."""

    url: str  # URL without credentials (bolt://host:port)
    user: str
    password: SecretStr
    database: str

    @classmethod
    def parse(
        cls,
        url: str,
        default_user: str = "neo4j",
        default_password: SecretStr | str = "",
        default_database: str = "neo4j",
    ) -> ParsedNeo4jUrl:
        """Parse a Neo4j URL with optional embedded credentials.

        Supports formats:
        - bolt://host:port
        - bolt://user:password@host:port
        - bolt://user:password@host:port/database

        ``default_password`` is used when the URL has no embedded password, so
        callers can pass a separately-configured password (e.g. from
        ``Neo4jConfig.password``) and have it flow through correctly.
        """
        parsed = urlparse(url)

        # Extract user and password from URL
        user = parsed.username or default_user
        if parsed.password:
            password: SecretStr = SecretStr(parsed.password)
        elif isinstance(default_password, SecretStr):
            password = default_password
        else:
            password = SecretStr(default_password)

        # Extract database from path (e.g., /mydb -> mydb)
        database = parsed.path.lstrip("/") if parsed.path and parsed.path != "/" else default_database

        # Reconstruct URL without credentials
        host_port = parsed.hostname or "localhost"
        if parsed.port:
            host_port = f"{host_port}:{parsed.port}"
        clean_url = f"{parsed.scheme}://{host_port}"

        return cls(url=clean_url, user=user, password=password, database=database)


# ---------------------------------------------------------------------------
# Graph backend configs (discriminated union on "backend" field)
# ---------------------------------------------------------------------------


class Neo4jConfig(BaseModel):
    """Neo4j graph backend configuration."""

    backend: Literal["neo4j"] = "neo4j"
    url: SecretStr | None = Field(default=None, description="Neo4j connection URL (bolt:// or neo4j://)")
    user: str = Field(default="neo4j", description="Neo4j username")
    password: SecretStr = Field(default=SecretStr(""), description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    max_connection_pool_size: int = Field(default=100, description="Neo4j connection pool size")
    connection_acquisition_timeout: float = Field(
        default=60.0, description="Timeout in seconds waiting for a connection from the pool"
    )
    retry_delay_jitter_factor: float = Field(
        default=0.5, description="Jitter factor for transaction retry delays (0.0-1.0)"
    )
    max_connection_lifetime: int = Field(
        default=900,
        description="Max seconds a connection stays in the pool before rotation. "
        "Set below the server-side TTL (e.g. Aura ~20min) to avoid reset errors.",
    )
    liveness_check_timeout: float | None = Field(
        default=30.0,
        description="Seconds of idle time after which connections are checked for liveness "
        "before being returned from the pool. None disables the check.",
    )
    query_timeout: float | None = Field(
        default=5.0,
        gt=0,
        le=300,
        description=(
            "Per-transaction timeout in seconds for bounded Neo4j read queries. "
            "Applied to all read methods on DualNodeManager and Neo4jBackend "
            "that issue MATCH traversals. Write paths (upsert / link / delete) "
            "are not bounded by this setting and require separate design. The "
            "Neo4j server terminates transactions exceeding this duration, "
            "raising ClientError with code "
            "Neo.ClientError.Transaction.TransactionTimedOut* — the client "
            "catches this and returns an empty result. Set to None to disable "
            "entirely. Values <= 0 are rejected (the driver would interpret 0 "
            "as 'run forever', which defeats the purpose). Values above 300 "
            "seconds (5 minutes) are also rejected as a sanity cap. Override "
            "via env var KHORA_STORAGE__GRAPH__QUERY_TIMEOUT."
        ),
    )
    entity_write_concurrency: int = Field(default=12, description="Max concurrent entity write transactions")
    relationship_write_concurrency: int = Field(default=8, description="Max concurrent relationship write transactions")
    pool_sampler_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in high-frequency Neo4j pool sampler. When True, Khora starts a "
            "background task that samples driver pool state at "
            "``pool_sampler_interval_ms`` cadence and emits the observations on the "
            "``khora.neo4j.pool.sampled.*`` histograms. Zero-cost when False. "
            "Set via env: KHORA_STORAGE__GRAPH__POOL_SAMPLER_ENABLED=true."
        ),
    )
    pool_sampler_interval_ms: int = Field(
        default=500,
        ge=50,
        le=60_000,
        description=(
            "Interval in milliseconds between Neo4j pool samples when "
            "``pool_sampler_enabled`` is True. Clamped to [50, 60000]. "
            "Set via env: KHORA_STORAGE__GRAPH__POOL_SAMPLER_INTERVAL_MS=250."
        ),
    )


class KuzuConfig(BaseModel):
    """Kùzu embedded graph backend configuration.

    .. deprecated::
        KuzuDB backend is deprecated. Kuzu was acquired by Apple in October 2025
        and the repository is archived. Consider using :class:`Neo4jConfig` or
        :class:`SurrealDBConfig` instead.
    """

    backend: Literal["kuzu"] = "kuzu"  # DEPRECATED in 0.9.0 — removal in 0.10.0
    database_path: str = Field(default="./kuzu_db", description="Path to Kùzu database directory")
    read_only: bool = Field(default=False, description="Open database in read-only mode")


class MemgraphConfig(BaseModel):
    """Memgraph graph backend configuration."""

    backend: Literal["memgraph"] = "memgraph"
    url: SecretStr | None = Field(default=None, description="Memgraph connection URL (bolt://)")
    user: str = Field(default="memgraph", description="Memgraph username")
    password: SecretStr = Field(default=SecretStr(""), description="Memgraph password")


class NeptuneConfig(BaseModel):
    """AWS Neptune graph backend configuration.

    Neptune supports openCypher via Bolt protocol. Uses the same neo4j
    Python driver as Neo4j and Memgraph backends.
    """

    backend: Literal["neptune"] = "neptune"
    url: SecretStr | None = Field(default=None, description="Neptune Bolt endpoint (bolt://cluster:8182)")
    user: str = Field(default="", description="Username (empty for IAM auth)")
    password: SecretStr = Field(default=SecretStr(""), description="Password (empty for IAM auth)")
    iam_auth: bool = Field(default=False, description="Use AWS IAM SigV4 authentication")
    aws_region: str = Field(default="us-east-1", description="AWS region for IAM auth signing")
    max_connection_pool_size: int = Field(default=100, description="Bolt connection pool size (Neptune max: 1000)")


class SurrealDBConfig(BaseModel):
    """SurrealDB unified backend configuration (graph role).

    SurrealDB serves as a unified backend providing graph, vector, and
    relational storage in a single database.
    """

    backend: Literal["surrealdb"] = "surrealdb"
    mode: str = Field(default="memory", description="Connection mode: memory, embedded, or remote")
    url: SecretStr | None = Field(default=None, description="SurrealDB WebSocket URL (for remote mode)")
    path: str | None = Field(default=None, description="Database file path (for embedded mode)")
    namespace: str = Field(default="khora", description="SurrealDB namespace")
    database: str = Field(default="default", description="SurrealDB database")
    user: str = Field(default="root", description="SurrealDB username")
    password: SecretStr = Field(default=SecretStr("root"), description="SurrealDB password")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    sync_data: bool = Field(default=True, description="Enable SURREAL_SYNC_DATA for crash-safe writes")


class SQLiteLanceConfig(BaseModel):
    """SQLite + LanceDB embedded unified backend configuration.

    Pairs an on-disk SQLite database (graph + relational + event store)
    with a sibling LanceDB directory (vector search). Zero infrastructure —
    both backends run in-process.
    """

    backend: Literal["sqlite_lance"] = "sqlite_lance"
    db_path: str = Field(default="./khora.db", description="SQLite database path")
    lance_path: str | None = Field(
        default=None,
        description="LanceDB directory path. When None, derived from db_path (sibling .lance dir).",
    )
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    use_halfvec: bool = Field(
        default=False,
        description="Store embeddings as float16 to halve index size (minor recall loss).",
    )
    lance_index: Literal["auto", "ivf_pq", "hnsw", "brute"] = Field(
        default="auto",
        description="Vector index type. 'auto' picks based on table size.",
    )
    ivf_partitions: int | None = Field(
        default=None,
        description="IVF partition count (ivf_pq only). None = auto from row count.",
    )
    hnsw_m: int = Field(default=16, description="HNSW M parameter (max connections per layer)")
    retrain_factor: float = Field(
        default=2.0,
        description=(
            "Rebuild the LanceDB ANN index once the row count grows to "
            "retrain_factor * (rows at last training). Default 2.0 retrains "
            "when the corpus has doubled. Set <= 1.0 to disable retraining."
        ),
    )

    model_config = {"extra": "forbid"}


class AGEConfig(BaseModel):
    """PostgreSQL AGE graph backend configuration.

    Uses Apache AGE extension to run openCypher queries inside PostgreSQL.
    Can share the same connection pool as the relational backend.
    """

    backend: Literal["age"] = "age"
    url: SecretStr | None = Field(default=None, description="PostgreSQL URL (can share with relational backend)")
    graph_name: str = Field(default="khora_graph", description="Name of the AGE graph")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")


def _graph_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("backend", "neo4j")
    return getattr(v, "backend", "neo4j")


GraphConfig = Annotated[
    Annotated[Neo4jConfig, Tag("neo4j")]
    | Annotated[KuzuConfig, Tag("kuzu")]  # DEPRECATED in 0.9.0
    | Annotated[MemgraphConfig, Tag("memgraph")]
    | Annotated[NeptuneConfig, Tag("neptune")]
    | Annotated[SurrealDBConfig, Tag("surrealdb")]
    | Annotated[AGEConfig, Tag("age")],
    Discriminator(_graph_discriminator),
]


# ---------------------------------------------------------------------------
# Vector backend configs (discriminated union on "backend" field)
# ---------------------------------------------------------------------------


class PgVectorConfig(BaseModel):
    """pgvector vector backend configuration."""

    backend: Literal["pgvector"] = "pgvector"
    url: SecretStr | None = Field(default=None, description="pgvector connection URL")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")


class SurrealDBVectorConfig(BaseModel):
    """SurrealDB unified backend configuration (vector role).

    Shares the same SurrealDB instance as the graph role.
    """

    backend: Literal["surrealdb"] = "surrealdb"
    mode: str = Field(default="memory", description="Connection mode: memory, embedded, or remote")
    url: SecretStr | None = Field(default=None, description="SurrealDB WebSocket URL (for remote mode)")
    path: str | None = Field(default=None, description="Database file path (for embedded mode)")
    namespace: str = Field(default="khora", description="SurrealDB namespace")
    database: str = Field(default="default", description="SurrealDB database")
    user: str = Field(default="root", description="SurrealDB username")
    password: SecretStr = Field(default=SecretStr("root"), description="SurrealDB password")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")


class SQLiteVectorConfig(BaseModel):
    """SQLite embedded backend configuration (relational + vector).

    Uses a single SQLite file for both relational and vector storage.
    Vector search is brute-force cosine via khora._accel.
    """

    backend: Literal["sqlite"] = "sqlite"
    url: str | None = Field(default=None, description="SQLite path (sqlite:///path.db or :memory:)")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")


def _vector_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("backend", "pgvector")
    return getattr(v, "backend", "pgvector")


VectorConfig = Annotated[
    Annotated[PgVectorConfig, Tag("pgvector")]
    | Annotated[SurrealDBVectorConfig, Tag("surrealdb")]
    | Annotated[SQLiteVectorConfig, Tag("sqlite")],
    Discriminator(_vector_discriminator),
]


# ---------------------------------------------------------------------------
# Storage settings
# ---------------------------------------------------------------------------


class StorageSettings(BaseSettings):
    """Storage backend configuration.

    Supports both the new discriminated-union graph/vector configs and
    the legacy flat fields (neo4j_url, pgvector_url, etc.) for backwards
    compatibility.

    Env vars: ``KHORA_STORAGE_BACKEND``, ``KHORA_STORAGE_POSTGRESQL_URL``, etc.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_STORAGE_", case_sensitive=False)

    # Unified backend selector (postgres = traditional PG+pgvector+Neo4j, surrealdb = unified)
    backend: str = Field(
        default="postgres",
        description="Storage backend strategy: 'postgres' (traditional PG+pgvector+graph) or 'surrealdb' (unified)",
    )

    # SurrealDB unified backend config (used when backend='surrealdb')
    surrealdb: SurrealDBConfig | None = Field(
        default=None,
        description="SurrealDB unified backend configuration (used when backend='surrealdb')",
    )

    # SQLite + LanceDB unified backend config (used when backend='sqlite_lance')
    sqlite_lance: SQLiteLanceConfig | None = Field(
        default=None,
        description="SQLite + LanceDB unified backend configuration (used when backend='sqlite_lance')",
    )

    # PostgreSQL (relational)
    postgresql_url: SecretStr | None = Field(default=None, description="PostgreSQL connection URL")
    postgresql_pool_size: int = Field(default=50, description="PostgreSQL connection pool size")
    postgresql_max_overflow: int = Field(default=30, description="PostgreSQL max overflow connections")
    postgresql_pool_pre_ping: bool = Field(
        default=False,
        description="Enable pool pre-ping to detect stale connections before checkout. "
        "Adds a small latency overhead per checkout but prevents errors from idle connections "
        "dropped by the server or network infrastructure.",
    )

    # New-style backend configs
    graph: GraphConfig | None = Field(default=None, description="Graph backend configuration (optional)")
    vector: VectorConfig = Field(default_factory=PgVectorConfig, description="Vector backend configuration")

    # Legacy flat fields — kept for backwards compatibility
    pgvector_url: SecretStr | None = Field(default=None, description="[deprecated] pgvector connection URL")
    embedding_dimension: int = Field(default=1536, description="[deprecated] Embedding vector dimension")
    neo4j_url: SecretStr | None = Field(default=None, description="[deprecated] Neo4j connection URL")
    neo4j_user: str = Field(default="neo4j", description="[deprecated] Neo4j username")
    neo4j_password: SecretStr = Field(default=SecretStr(""), description="[deprecated] Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="[deprecated] Neo4j database name")

    # HNSW index tuning
    hnsw_m: int = Field(default=24, description="HNSW index M parameter (max connections per layer)")
    hnsw_ef_construction: int = Field(default=128, description="HNSW index ef_construction (build-time search width)")
    hnsw_ef_search: int = Field(default=100, description="HNSW ef_search for query-time accuracy")

    # Half-precision vectors (requires pgvector extension >= 0.7.0)
    use_halfvec: bool = Field(
        default=True,
        description="Use halfvec (float16) for HNSW indexes. Halves index size with minimal recall loss. "
        "Requires pgvector extension >= 0.7.0. Column data remains full precision (vector type). "
        "Falls back to full precision if pgvector < 0.7.0.",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: Any) -> Any:
        """Migrate legacy flat fields into the new graph/vector config objects."""
        if not isinstance(data, dict):
            return data

        # Only migrate if the new-style configs are not explicitly provided
        if "graph" not in data:
            neo4j_url = data.get("neo4j_url")
            neo4j_user = data.get("neo4j_user", "neo4j")
            neo4j_password = data.get("neo4j_password", "")
            neo4j_database = data.get("neo4j_database", "neo4j")
            if neo4j_url:
                data["graph"] = {
                    "backend": "neo4j",
                    "url": neo4j_url,
                    "user": neo4j_user,
                    "password": neo4j_password,
                    "database": neo4j_database,
                }

        if "vector" not in data:
            pgvector_url = data.get("pgvector_url")
            embedding_dim = data.get("embedding_dimension", 1536)
            if pgvector_url:
                data["vector"] = {
                    "backend": "pgvector",
                    "url": pgvector_url,
                    "embedding_dimension": embedding_dim,
                }

        return data


class LLMSettings(BaseSettings):
    """LLM configuration settings.

    Env vars: ``KHORA_LLM_MODEL``, ``KHORA_LLM_EMBEDDING_MODEL``, etc.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_LLM_", case_sensitive=False)

    model: str = Field(default="gpt-4o-mini", description="Primary LLM model")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=12288, description="Maximum tokens for LLM extraction output")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retries on failure")
    max_concurrent_llm_calls: int = Field(default=10, description="Maximum concurrent LLM calls")

    # Embedding settings
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    embedding_dimension: int = Field(default=1536, description="Embedding dimension")

    # Extraction model (defaults to primary model if not set)
    extraction_model: str | None = Field(
        default=None,
        description="Model for entity extraction (defaults to primary model). "
        "Smaller/faster models like claude-3-5-haiku or gemini-2.0-flash work well for extraction.",
    )

    # Router configuration
    config_file: str | None = Field(default=None, description="Path to LiteLLM config YAML")
    model_list: list[dict[str, Any]] | None = Field(default=None, description="Model list for router")
    router_settings: dict[str, Any] | None = Field(default=None, description="Router settings")

    # Shared aiohttp session connector settings (mirrors LiteLLMConfig fields).
    # The shared session is created once per process on first engine connect;
    # see LiteLLMConfig docstring for first-call-wins semantics.
    max_total_connections: int = Field(
        default=200,
        gt=0,
        description="Total cap on simultaneous connections in the shared aiohttp session, summed across all hosts.",
    )
    max_connections_per_host: int = Field(
        default=0,
        ge=0,
        description="Per-host cap on simultaneous connections in the shared "
        "aiohttp session. 0 = unlimited (matches pre-0.9.0 behaviour).",
    )
    keepalive_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description="Idle keepalive seconds for connections in the shared aiohttp session.",
    )


class PipelineSettings(BaseSettings):
    """Pipeline configuration settings.

    Env vars: ``KHORA_PIPELINES_CHUNK_SIZE``, ``KHORA_PIPELINES_SELECTIVE_EXTRACTION``, etc.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_PIPELINES_", case_sensitive=False)

    # Chunking settings
    chunking_strategy: str = Field(default="semantic", description="Chunking strategy: fixed, semantic, recursive")
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks in tokens")

    # Conversation chunking settings
    conversation_time_gap_minutes: int = Field(default=15, description="Time gap (minutes) to split conversations")
    conversation_max_group_size: int = Field(default=50, description="Max messages per conversation chunk")
    conversation_min_group_size: int = Field(default=2, description="Min messages per chunk (merges below this)")
    conversation_semantic_threshold: float | None = Field(
        default=None, description="Optional cosine similarity threshold for semantic splitting"
    )

    # Extraction settings
    extract_entities: bool = Field(default=True, description="Extract entities from documents")
    entity_types: list[str] = Field(
        default=["PERSON", "ORGANIZATION", "CONCEPT", "LOCATION"],
        description="Entity types to extract",
    )

    # Selective extraction (KET-RAG style importance scoring)
    # When enabled, chunks are scored by importance and only the top fraction
    # are sent to LLM extraction. The rest get lightweight rule-based edges.
    selective_extraction: bool = Field(
        default=True,
        description="Enable importance-based selective extraction to reduce LLM cost. "
        "When True, only the most important chunks are sent to LLM extraction; "
        "the rest get lightweight co-occurrence edges.",
    )
    extraction_importance_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of chunks to send to full LLM extraction (top-K by importance score). "
        "Lower values save more cost but may miss entities in low-importance chunks.",
    )
    extraction_min_importance: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum importance score threshold. Chunks scoring above this are always "
        "sent to LLM extraction regardless of the ratio cutoff.",
    )

    # Entity embedding skip rules — skip embedding generation for low-value entity types
    skip_embedding_entity_types: list[str] = Field(
        default=["DATE", "URL", "EMAIL"],
        description="Entity types to skip embedding for when mention_count is below threshold. "
        "These types rarely benefit from vector similarity search.",
    )
    skip_embedding_mention_threshold: int = Field(
        default=1,
        description="Skip embedding for entities in skip_embedding_entity_types with "
        "mention_count <= this value. Set to 0 to skip all single-mention entities of these types.",
    )

    # Unified PENDING document processor.
    # Replaces the separate _submit_batch_worker and _recover_pending_documents paths.
    pending_processor_enabled: bool = Field(
        default=True,
        description="Retained for backwards compatibility. No longer auto-consulted on connect() — "
        "the processor must now be started explicitly via Khora.start_pending_processor(). "
        "Setting this env var has no effect on processor startup.",
    )
    pending_processor_max_concurrent: int = Field(
        default=20,
        description="Maximum documents to process concurrently in the unified pending processor.",
    )
    pending_processor_grace_period_minutes: int = Field(
        default=5,
        description="Minimum age (minutes) a PENDING document must have before it is eligible for "
        "crash-recovery processing. Avoids racing with in-flight writes.",
    )

    # Deprecated aliases — kept for backwards compat with existing env vars.
    pending_recovery_enabled: bool | None = Field(
        default=None,
        description="Deprecated: use pending_processor_enabled instead.",
    )
    pending_recovery_grace_period_minutes: int | None = Field(
        default=None,
        description="Deprecated: use pending_processor_grace_period_minutes instead.",
    )


class TenancySettings(BaseSettings):
    """Multi-tenancy configuration settings.

    Env vars: ``KHORA_TENANCY_DEFAULT_MODE``, ``KHORA_TENANCY_ENFORCE_NAMESPACE``.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_TENANCY_", case_sensitive=False)

    default_mode: str = Field(default="shared", description="Default tenancy mode: shared or isolated")
    enforce_namespace: bool = Field(default=True, description="Enforce namespace isolation")


class QuerySettings(BaseSettings):
    """Query pipeline configuration.

    Env vars: ``KHORA_QUERY_DEFAULT_MODE``, ``KHORA_QUERY_ENABLE_HYDE``, etc.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_QUERY_", case_sensitive=False)

    # Basic search settings
    default_mode: str = Field(default="hybrid", description="Default search mode: vector, graph, hybrid, all")
    min_chunk_similarity: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum chunk similarity threshold")
    min_entity_similarity: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Minimum entity similarity threshold"
    )

    # Fusion weights
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for vector search in fusion")
    graph_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for graph search in fusion")
    keyword_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for keyword search in fusion")

    # Temporal settings.
    # `recency_weight` and `recency_decay_days` were tightened in DYT-3780
    # from (0.2, 30) to (0.35, 7) after BEAM 100K showed the four weakest
    # categories (event_ordering, contradiction_resolution, temporal_reasoning,
    # knowledge_update) all share a weak-recency / supersession root cause —
    # 30-day half-life is wider than most session lifetimes, and 0.2 is too
    # gentle to break ties between an old fact and its in-session update.
    # `apply_recency_bias` still defaults False: callers must opt in.
    apply_recency_bias: bool = Field(default=False, description="Apply recency bias to results")
    recency_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Weight of recency in scoring")
    recency_decay_days: float = Field(default=7.0, ge=1.0, description="Days for recency score to decay by half")

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
    entity_linking_fuzzy_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum fuzzy match ratio")
    entity_linking_embedding_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Minimum embedding similarity"
    )
    entity_linking_max_candidates: int = Field(default=10, ge=1, description="Maximum entity candidates per mention")

    # Reranking
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    reranking_method: str = Field(default="cross_encoder", description="Reranking method: cross_encoder, llm")
    reranking_model: str | None = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="Model for reranking (cross-encoder model or LLM)",
    )
    reranking_top_n: int = Field(default=50, ge=1, description="Number of candidates to rerank")
    reranking_final_k: int = Field(default=10, ge=1, description="Number of results after reranking")

    # Keyword search
    enable_keyword_search: bool = Field(default=True, description="Enable keyword search")
    keyword_search_method: str = Field(default="fulltext", description="Keyword search method: bm25, fulltext")
    keyword_search_use_stemming: bool = Field(default=True, description="Apply stemming to search terms")
    keyword_search_use_stopwords: bool = Field(default=True, description="Remove stopwords from search")
    keyword_search_language: str = Field(default="english", description="Language for stemming and stopwords")

    # HyDE
    enable_hyde: str = Field(
        default="auto",
        description="HyDE mode: 'auto' (enable for complex/temporal queries), 'always', 'never'. "
        "Also accepts bool for backward compatibility (True='always', False='never').",
    )
    hyde_num_hypotheticals: int = Field(
        default=1, ge=1, le=5, description="Number of hypothetical documents to generate"
    )

    @field_validator("enable_hyde", mode="before")
    @classmethod
    def _normalize_enable_hyde(cls, v: Any) -> str:
        if v is True:
            return "always"
        if v is False:
            return "never"
        if isinstance(v, str) and v in ("auto", "always", "never"):
            return v
        return "auto"

    # Multi-stage ranking pipeline
    enable_multi_stage: bool = Field(
        default=True, description="Enable multi-stage ranking pipeline for improved quality"
    )
    stage1_recall_limit: int = Field(
        default=200, ge=50, le=500, description="Number of candidates to retrieve in Stage 1 (broad recall)"
    )
    stage3_filter_limit: int = Field(
        default=50, ge=20, le=200, description="Number of candidates after Stage 3 filtering"
    )
    stage4_rerank_limit: int = Field(
        default=50, ge=10, le=100, description="Number of candidates to send to neural reranking in Stage 4"
    )
    enable_diversity: bool = Field(default=True, description="Enable MMR-style diversity selection in Stage 5")
    diversity_lambda: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Diversity vs relevance tradeoff (0=pure diversity, 1=pure relevance)"
    )

    # Stage 1 recall budget distribution (must sum to ~1.0)
    stage1_vector_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of Stage 1 recall budget allocated to vector search",
    )
    stage1_graph_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of Stage 1 recall budget allocated to graph search",
    )
    stage1_keyword_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of Stage 1 recall budget allocated to keyword search",
    )

    # Reranking blend weight (how much to trust reranker vs original score)
    reranking_blend_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for reranker score in blend (remainder goes to original). "
        "0.7 = trust reranker 70%, keep 30% original.",
    )

    # Temporal scoring parameters (previously hardcoded)
    temporal_hard_cutoff_days: float = Field(
        default=30.0,
        ge=0.0,
        description="Hard cutoff for soft temporal scoring. Chunks outside the temporal "
        "window by more than this many days are scored zero.",
    )
    temporal_half_life_hours: float = Field(
        default=24.0,
        ge=1.0,
        description="Half-life in hours for temporal decay outside the query window.",
    )

    # Graph search scoring (previously hardcoded)
    graph_chunk_query_sim_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for query similarity in graph chunk scoring (remainder = entity score).",
    )

    # Expanded query discount (previously hardcoded)
    expanded_query_discount: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Discount factor for HyDE/expansion-generated query results.",
    )

    # Linked entity boost (previously hardcoded)
    linked_entity_boost: float = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Score multiplier for entities matched via entity linking.",
    )

    # Structured document features (default off — enable per-namespace to avoid per-query DB lookups)
    enable_relationship_expansion: bool = Field(
        default=False,
        description="After retrieval, follow relationship edges from top entities to inject "
        "related chunks. Enable per-namespace for structured documents with dense cross-references. "
        "When True, adds a DB round-trip per query to check for relationship data.",
    )
    relationship_expansion_max: int = Field(
        default=5, ge=1, le=20, description="Maximum additional chunks from relationship expansion"
    )
    enable_taxonomy_boost: bool = Field(
        default=False,
        description="Classify query against document hierarchy (chapters/topics) and boost chunks "
        "from matching scope. Reads from namespace.metadata['khora']['taxonomy']. "
        "When True, adds a DB round-trip per query to fetch namespace metadata.",
    )
    taxonomy_boost_factor: float = Field(
        default=1.5, ge=1.0, le=3.0, description="Score multiplier for chunks from taxonomy-matched scope"
    )

    # Chronicle engine tuning
    chronicle_temporal_window_days: float = Field(
        default=0.0,
        ge=-1.0,
        description="Temporal channel window: 0=unlimited (search all data), >0=N-day window, -1=disable channel",
    )
    chronicle_decay_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight of temporal decay in Chronicle scoring (blended with relevance score)",
    )
    chronicle_overfetch_multiplier: int = Field(
        default=4, ge=2, le=10, description="Over-fetch multiplier for Chronicle retrieval channels"
    )
    chronicle_rrf_semantic_weight: float = Field(
        default=1.0, ge=0.0, le=2.0, description="RRF weight for semantic channel in Chronicle fusion"
    )
    chronicle_rrf_bm25_weight: float = Field(
        default=0.8, ge=0.0, le=2.0, description="RRF weight for BM25 channel in Chronicle fusion"
    )
    chronicle_rrf_temporal_weight: float = Field(
        default=0.9, ge=0.0, le=2.0, description="RRF weight for temporal channel in Chronicle fusion"
    )
    chronicle_rrf_entity_weight: float = Field(
        default=0.85, ge=0.0, le=2.0, description="RRF weight for entity co-occurrence channel in Chronicle fusion"
    )

    # LLM listwise reranking
    enable_llm_reranking: bool = Field(
        default=False, description="Enable LLM-based listwise reranking after cross-encoder stage"
    )
    llm_reranking_model: str = Field(default="gpt-4o-mini", description="Model for LLM listwise reranking")
    llm_reranking_top_n: int = Field(default=10, ge=3, le=30, description="Number of top candidates to rerank with LLM")
    llm_reranking_confidence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Only trigger LLM reranking when cross-encoder score gap between rank 1 and 2 is below this",
    )

    # Query normalization
    enable_query_normalization: bool = Field(
        default=True, description="Normalize queries (filler removal, contraction expansion) before embedding"
    )
    enable_multi_vector_query: bool = Field(
        default=False, description="Average original and normalized query embeddings for more robust retrieval"
    )
    # Two-tier temporal resolver
    enable_temporal_resolver: bool = Field(
        default=True, description="Enable two-tier temporal resolver (dateparser + LLM)"
    )
    temporal_resolver_strategy: str = Field(
        default="hybrid",
        description="Temporal resolver strategy: 'dateparser' (fast only), 'llm' (LLM only), 'hybrid' (dateparser + LLM fallback)",
    )
    temporal_sql_pushdown: bool = Field(
        default=True,
        description="Push temporal filters to Stage 1 SQL WHERE clauses instead of post-retrieval filtering",
    )
    temporal_date_validation: bool = Field(
        default=True, description="Validate LLM-generated dates (swap inverted, cap future, reject ancient)"
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

    # Database for Khora internal state (shortcuts for storage.* URLs)
    # These can be set via KHORA_DATABASE_URL and KHORA_NEO4J_URL environment variables
    # Programmatic values take priority over environment variables
    database_url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL URL for Khora database (shortcut for storage.postgresql_url)",
    )
    neo4j_url: SecretStr | None = Field(
        default=None,
        description="Neo4j URL for graph storage (shortcut for storage.neo4j_url)",
    )

    # LLM extraction model shortcut (single-underscore env var: KHORA_LLM_EXTRACTION_MODEL)
    # Propagated to llm.extraction_model — see model_validator below.
    llm_extraction_model: str | None = Field(
        default=None,
        description="Model for entity extraction (shortcut for llm.extraction_model)",
    )

    # Storage configuration
    storage: StorageSettings = Field(default_factory=StorageSettings)

    # LLM configuration
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Pipeline configuration
    pipelines: PipelineSettings = Field(default_factory=PipelineSettings)

    @property
    def pipeline(self) -> PipelineSettings:
        """Alias so engine code can use ``config.pipeline.*``."""
        return self.pipelines

    # Tenancy configuration
    tenancy: TenancySettings = Field(default_factory=TenancySettings)

    # Query pipeline configuration
    query: QuerySettings = Field(default_factory=QuerySettings)

    # Semantic hooks configuration
    hooks: Any = Field(default=None, description="Semantic hooks configuration (SemanticHooksConfig)")

    # Telemetry
    telemetry_database_url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL URL for telemetry database (set KHORA_TELEMETRY_DATABASE_URL to enable)",
    )
    telemetry_service_name: str = Field(
        default="khora",
        description="Service name tag for telemetry events",
    )

    @model_validator(mode="after")
    def _propagate_shortcuts(self) -> KhoraConfig:
        """Propagate top-level shortcut fields into nested configs."""
        if self.llm_extraction_model and not self.llm.extraction_model:
            self.llm.extraction_model = self.llm_extraction_model
        return self

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
        """Get PostgreSQL URL from config (plaintext, for driver consumption).

        Boundary unwrap for ``SecretStr``: callers receive a plain string
        suitable for handing to SQLAlchemy / asyncpg. Returns ``None`` when
        neither ``storage.postgresql_url`` nor ``database_url`` is set.
        """
        if self.storage.postgresql_url is not None:
            return _secret_value(self.storage.postgresql_url) or None
        if self.database_url is not None:
            return _secret_value(self.database_url) or None
        return None

    def _get_raw_neo4j_url(self) -> str | None:
        """Get raw Neo4j URL (may contain credentials).

        Boundary unwrap for ``SecretStr``. Callers feed this URL into
        ``ParsedNeo4jUrl.parse`` (which itself splits the URL into
        cleaned-URL + credentials before forwarding to the driver).
        """
        # Check new-style graph config first
        graph = self.storage.graph
        if isinstance(graph, Neo4jConfig) and graph.url:
            return _secret_value(graph.url) or None
        # Fall back to legacy fields
        if self.storage.neo4j_url is not None:
            value = _secret_value(self.storage.neo4j_url)
            if value:
                return value
        if self.neo4j_url is not None:
            value = _secret_value(self.neo4j_url)
            if value:
                return value
        return None

    def _parse_neo4j_url(self) -> ParsedNeo4jUrl | None:
        """Parse Neo4j URL and extract components."""
        raw_url = self._get_raw_neo4j_url()
        if not raw_url:
            return None
        # Use graph config defaults if available
        graph = self.storage.graph
        default_user = graph.user if isinstance(graph, Neo4jConfig) else self.storage.neo4j_user
        default_password = graph.password if isinstance(graph, Neo4jConfig) else self.storage.neo4j_password
        default_db = graph.database if isinstance(graph, Neo4jConfig) else self.storage.neo4j_database
        return ParsedNeo4jUrl.parse(
            raw_url,
            default_user=default_user,
            default_password=default_password,
            default_database=default_db,
        )

    def get_neo4j_url(self) -> str | None:
        """Get Neo4j URL without credentials (for driver connection).

        Parses URL like bolt://user:pass@host:port and returns bolt://host:port
        """
        parsed = self._parse_neo4j_url()
        return parsed.url if parsed else None

    def get_neo4j_user(self) -> str:
        """Get Neo4j username.

        Precedence: username embedded in ``Neo4jConfig.url`` (or the legacy
        ``neo4j_url``) wins. Otherwise, falls back to the separately-configured
        ``Neo4jConfig.user`` (or the legacy ``neo4j_user``). Returns ``"neo4j"``
        when neither is set.
        """
        parsed = self._parse_neo4j_url()
        if parsed:
            return parsed.user
        graph = self.storage.graph
        if isinstance(graph, Neo4jConfig):
            return graph.user
        return self.storage.neo4j_user

    def get_neo4j_password(self) -> str:
        """Get Neo4j password (plaintext, for driver consumption).

        Boundary unwrap for ``SecretStr``.

        Precedence: password embedded in ``Neo4jConfig.url`` (or the legacy
        ``neo4j_url``) wins. Otherwise, falls back to the separately-configured
        ``Neo4jConfig.password`` (or the legacy ``neo4j_password``). Returns an
        empty string when neither is set.
        """
        parsed = self._parse_neo4j_url()
        if parsed:
            return parsed.password.get_secret_value()
        graph = self.storage.graph
        if isinstance(graph, Neo4jConfig):
            return graph.password.get_secret_value()
        return _secret_value(self.storage.neo4j_password)

    def get_neo4j_database(self) -> str:
        """Get Neo4j database name.

        Precedence: database embedded in ``Neo4jConfig.url`` path (or the legacy
        ``neo4j_url``) wins. Otherwise, falls back to the separately-configured
        ``Neo4jConfig.database`` (or the legacy ``neo4j_database``). Returns
        ``"neo4j"`` when neither is set.
        """
        parsed = self._parse_neo4j_url()
        if parsed:
            return parsed.database
        graph = self.storage.graph
        if isinstance(graph, Neo4jConfig):
            return graph.database
        return self.storage.neo4j_database

    def get_graph_config(self) -> GraphConfig | None:
        """Get the graph backend configuration.

        Returns None if no graph backend is configured, allowing graph-free operation.
        If using legacy config, builds a Neo4jConfig from the parsed URL.
        """
        graph = self.storage.graph
        # If it's already set from new-style config with a URL, return as-is
        if isinstance(graph, Neo4jConfig) and graph.url:
            return graph
        # If it's a non-Neo4j backend (Kuzu, Memgraph, etc.), return as-is
        if graph is not None and not isinstance(graph, Neo4jConfig):
            return graph
        # Check legacy neo4j_url (covers both graph=None and Neo4jConfig without URL)
        neo4j_url = self.get_neo4j_url()
        if neo4j_url:
            return Neo4jConfig(
                url=neo4j_url,
                user=self.get_neo4j_user(),
                password=self.get_neo4j_password(),
                database=self.get_neo4j_database(),
            )
        # No URL configured - graph backend is disabled
        return None

    def get_vector_config(self) -> VectorConfig:
        """Get the vector backend configuration.

        If using legacy config, builds a PgVectorConfig from the flat fields.
        """
        vector = self.storage.vector
        if isinstance(vector, PgVectorConfig) and not vector.url:
            # Populate from legacy fields. Both ``pgvector_url`` and
            # ``PgVectorConfig.url`` are ``SecretStr``; unwrap the legacy
            # field here so Pydantic re-wraps consistently.
            url = _secret_value(self.storage.pgvector_url) or self.get_postgresql_url()
            return PgVectorConfig(
                url=url,
                embedding_dimension=self.storage.embedding_dimension,
            )
        return vector

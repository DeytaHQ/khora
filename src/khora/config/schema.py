"""Pydantic configuration models for Khora."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import AliasChoices, BaseModel, Discriminator, Field, SecretStr, Tag, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Lazy module-level import of SemanticHooksConfig avoids the circular-import
# chain (config → hooks → telemetry → config) that previously forced
# ``KhoraConfig.hooks: Any = None``. Imported here at module load — by the
# time KhoraConfig is instantiated all chains are resolved. See
# Issue #576 Phase 1 Item 4.
from khora.config._secrets import AllowSecretTyping
from khora.config.llm import DEFAULT_API_KEY_ENV, derive_api_key_env
from khora.dream.config import DreamConfig
from khora.hooks.models import SemanticHooksConfig as _SemanticHooksConfig

SemanticHooksConfig = _SemanticHooksConfig  # public re-export


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
# Env-var alias helper (Issue #789)
# ---------------------------------------------------------------------------


def _env_aliases(*names: str) -> AliasChoices:
    """Build AliasChoices from a list of accepted env var names (in priority order).

    Used to expose the same field under both the canonical single-underscore
    form (preferred, documented) and the legacy double-underscore form
    (kept working forever for backward compat with existing .env files).

    Priority order matters: the first name wins when multiple aliases are
    set to the same value. Conflicts between different values across the
    aliases are surfaced by ``KhoraConfig._reject_alias_conflicts``.
    """
    return AliasChoices(*names)


def _inject_canonical_env(
    data: dict[str, Any], legacy_env_name: str, value: str, *, env_prefix: str = "KHORA_"
) -> None:
    """Inject a single-underscore env var into the nested-dict slot.

    pydantic-settings's ``env_nested_delimiter="__"`` ingests only the
    double-underscore form. We translate the legacy env-var name into the
    nested-key path it would have produced (e.g.
    ``KHORA_STORAGE__GRAPH__URL`` → ``data["storage"]["graph"]["url"]``)
    and write the value there if the slot is empty.

    ``env_prefix`` is stripped from the legacy name before splitting, so
    callers must pass the prefix that matches the BaseSettings owner of
    the field (``KHORA_`` for ``KhoraConfig``, ``KHORA_DREAM_`` for
    ``DreamConfig``).
    """
    if not legacy_env_name.startswith(env_prefix):
        return
    tail = legacy_env_name[len(env_prefix) :]
    parts = [p.lower() for p in tail.split("__")]
    if not parts:
        return
    node: Any = data
    for key in parts[:-1]:
        existing = node.get(key)
        if existing is None:
            existing = {}
            node[key] = existing
        elif not isinstance(existing, dict):
            # Caller has already provided a concrete value here — don't
            # clobber it with a dict.
            return
        node = existing
    leaf_key = parts[-1]
    if node.get(leaf_key) is None:
        node[leaf_key] = value


def _process_alias_env_pairs(data: Any, pairs: tuple[tuple[str, str], ...], *, env_prefix: str) -> list[str]:
    """Shared body of the alias-conflict + canonical-promotion validators.

    Mutates ``data`` in place to inject canonical-form values into the
    nested-dict slot pydantic-settings expects, and returns a list of
    conflict messages for the caller to assemble into a ``ValueError``.
    """
    env = os.environ
    conflicts: list[str] = []
    for canonical, legacy in pairs:
        new_val = env.get(canonical)
        old_val = env.get(legacy)
        if new_val is None and old_val is None:
            continue
        if new_val is not None and old_val is not None and new_val != old_val:
            conflicts.append(f"  - {canonical} and {legacy} are both set to different values")
            continue
        if new_val is not None and old_val is None and isinstance(data, dict):
            _inject_canonical_env(data, legacy, new_val, env_prefix=env_prefix)
    return conflicts


# Mapping from canonical single-underscore env var to its legacy double-
# underscore counterpart for the conflict-detection validator. Generated
# once here so the validator does not have to walk the model. Pairs whose
# canonical form is owned by a nested BaseSettings (``DreamConfig``) are
# tagged via ``_DREAM_ENV_ALIAS_PAIRS`` instead so the right validator
# handles them.
_ENV_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    # Discriminator fields — must promote so the union resolver sees the
    # operator-chosen backend before any backend-specific field validation
    # runs. Without these, ``KHORA_STORAGE_GRAPH_BACKEND=neptune`` plus
    # Neptune-specific knobs would route to the default Neo4jConfig and
    # the Neptune fields would fail ``extra=forbid``.
    ("KHORA_STORAGE_GRAPH_BACKEND", "KHORA_STORAGE__GRAPH__BACKEND"),
    ("KHORA_STORAGE_VECTOR_BACKEND", "KHORA_STORAGE__VECTOR__BACKEND"),
    # storage.graph (Neo4j)
    ("KHORA_STORAGE_GRAPH_URL", "KHORA_STORAGE__GRAPH__URL"),
    ("KHORA_STORAGE_GRAPH_USER", "KHORA_STORAGE__GRAPH__USER"),
    ("KHORA_STORAGE_GRAPH_PASSWORD", "KHORA_STORAGE__GRAPH__PASSWORD"),
    ("KHORA_STORAGE_GRAPH_DATABASE", "KHORA_STORAGE__GRAPH__DATABASE"),
    ("KHORA_STORAGE_GRAPH_MAX_CONNECTION_POOL_SIZE", "KHORA_STORAGE__GRAPH__MAX_CONNECTION_POOL_SIZE"),
    ("KHORA_STORAGE_GRAPH_CONNECTION_ACQUISITION_TIMEOUT", "KHORA_STORAGE__GRAPH__CONNECTION_ACQUISITION_TIMEOUT"),
    ("KHORA_STORAGE_GRAPH_RETRY_DELAY_JITTER_FACTOR", "KHORA_STORAGE__GRAPH__RETRY_DELAY_JITTER_FACTOR"),
    ("KHORA_STORAGE_GRAPH_MAX_CONNECTION_LIFETIME", "KHORA_STORAGE__GRAPH__MAX_CONNECTION_LIFETIME"),
    ("KHORA_STORAGE_GRAPH_LIVENESS_CHECK_TIMEOUT", "KHORA_STORAGE__GRAPH__LIVENESS_CHECK_TIMEOUT"),
    ("KHORA_STORAGE_GRAPH_QUERY_TIMEOUT", "KHORA_STORAGE__GRAPH__QUERY_TIMEOUT"),
    ("KHORA_STORAGE_GRAPH_ENTITY_WRITE_CONCURRENCY", "KHORA_STORAGE__GRAPH__ENTITY_WRITE_CONCURRENCY"),
    ("KHORA_STORAGE_GRAPH_RELATIONSHIP_WRITE_CONCURRENCY", "KHORA_STORAGE__GRAPH__RELATIONSHIP_WRITE_CONCURRENCY"),
    ("KHORA_STORAGE_GRAPH_POOL_SAMPLER_ENABLED", "KHORA_STORAGE__GRAPH__POOL_SAMPLER_ENABLED"),
    ("KHORA_STORAGE_GRAPH_POOL_SAMPLER_INTERVAL_MS", "KHORA_STORAGE__GRAPH__POOL_SAMPLER_INTERVAL_MS"),
    ("KHORA_STORAGE_GRAPH_POOL_KEEPALIVE_ENABLED", "KHORA_STORAGE__GRAPH__POOL_KEEPALIVE_ENABLED"),
    ("KHORA_STORAGE_GRAPH_POOL_KEEPALIVE_INTERVAL_MS", "KHORA_STORAGE__GRAPH__POOL_KEEPALIVE_INTERVAL_MS"),
    (
        "KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX",
        "KHORA_STORAGE__GRAPH__RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX",
    ),
    (
        "KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_CHUNK_IDS_MAX",
        "KHORA_STORAGE__GRAPH__RELATIONSHIP_SOURCE_CHUNK_IDS_MAX",
    ),
    ("KHORA_STORAGE_GRAPH_ENTITY_SOURCE_DOCUMENT_IDS_MAX", "KHORA_STORAGE__GRAPH__ENTITY_SOURCE_DOCUMENT_IDS_MAX"),
    ("KHORA_STORAGE_GRAPH_ENTITY_SOURCE_CHUNK_IDS_MAX", "KHORA_STORAGE__GRAPH__ENTITY_SOURCE_CHUNK_IDS_MAX"),
    # Neptune-specific
    ("KHORA_STORAGE_GRAPH_IAM_AUTH", "KHORA_STORAGE__GRAPH__IAM_AUTH"),
    ("KHORA_STORAGE_GRAPH_AWS_REGION", "KHORA_STORAGE__GRAPH__AWS_REGION"),
    # SurrealDB graph role
    ("KHORA_STORAGE_GRAPH_MODE", "KHORA_STORAGE__GRAPH__MODE"),
    ("KHORA_STORAGE_GRAPH_PATH", "KHORA_STORAGE__GRAPH__PATH"),
    ("KHORA_STORAGE_GRAPH_NAMESPACE", "KHORA_STORAGE__GRAPH__NAMESPACE"),
    ("KHORA_STORAGE_GRAPH_EMBEDDING_DIMENSION", "KHORA_STORAGE__GRAPH__EMBEDDING_DIMENSION"),
    ("KHORA_STORAGE_GRAPH_SYNC_DATA", "KHORA_STORAGE__GRAPH__SYNC_DATA"),
    # AGE
    ("KHORA_STORAGE_GRAPH_GRAPH_NAME", "KHORA_STORAGE__GRAPH__GRAPH_NAME"),
    ("KHORA_STORAGE_GRAPH_POOL_SIZE", "KHORA_STORAGE__GRAPH__POOL_SIZE"),
    ("KHORA_STORAGE_GRAPH_MAX_OVERFLOW", "KHORA_STORAGE__GRAPH__MAX_OVERFLOW"),
    # storage.surrealdb (unified)
    ("KHORA_STORAGE_SURREALDB_MODE", "KHORA_STORAGE__SURREALDB__MODE"),
    ("KHORA_STORAGE_SURREALDB_URL", "KHORA_STORAGE__SURREALDB__URL"),
    ("KHORA_STORAGE_SURREALDB_PATH", "KHORA_STORAGE__SURREALDB__PATH"),
    ("KHORA_STORAGE_SURREALDB_NAMESPACE", "KHORA_STORAGE__SURREALDB__NAMESPACE"),
    ("KHORA_STORAGE_SURREALDB_DATABASE", "KHORA_STORAGE__SURREALDB__DATABASE"),
    ("KHORA_STORAGE_SURREALDB_USER", "KHORA_STORAGE__SURREALDB__USER"),
    ("KHORA_STORAGE_SURREALDB_PASSWORD", "KHORA_STORAGE__SURREALDB__PASSWORD"),
    ("KHORA_STORAGE_SURREALDB_EMBEDDING_DIMENSION", "KHORA_STORAGE__SURREALDB__EMBEDDING_DIMENSION"),
    ("KHORA_STORAGE_SURREALDB_SYNC_DATA", "KHORA_STORAGE__SURREALDB__SYNC_DATA"),
    # storage.vector (PgVector / SurrealDBVector / SQLiteVector)
    ("KHORA_STORAGE_VECTOR_URL", "KHORA_STORAGE__VECTOR__URL"),
    ("KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION", "KHORA_STORAGE__VECTOR__EMBEDDING_DIMENSION"),
    ("KHORA_STORAGE_VECTOR_MODE", "KHORA_STORAGE__VECTOR__MODE"),
    ("KHORA_STORAGE_VECTOR_PATH", "KHORA_STORAGE__VECTOR__PATH"),
    ("KHORA_STORAGE_VECTOR_NAMESPACE", "KHORA_STORAGE__VECTOR__NAMESPACE"),
    ("KHORA_STORAGE_VECTOR_DATABASE", "KHORA_STORAGE__VECTOR__DATABASE"),
    ("KHORA_STORAGE_VECTOR_USER", "KHORA_STORAGE__VECTOR__USER"),
    ("KHORA_STORAGE_VECTOR_PASSWORD", "KHORA_STORAGE__VECTOR__PASSWORD"),
    # storage.sqlite_lance
    ("KHORA_STORAGE_SQLITE_LANCE_DB_PATH", "KHORA_STORAGE__SQLITE_LANCE__DB_PATH"),
    ("KHORA_STORAGE_SQLITE_LANCE_LANCE_PATH", "KHORA_STORAGE__SQLITE_LANCE__LANCE_PATH"),
    ("KHORA_STORAGE_SQLITE_LANCE_EMBEDDING_DIMENSION", "KHORA_STORAGE__SQLITE_LANCE__EMBEDDING_DIMENSION"),
    ("KHORA_STORAGE_SQLITE_LANCE_USE_HALFVEC", "KHORA_STORAGE__SQLITE_LANCE__USE_HALFVEC"),
    ("KHORA_STORAGE_SQLITE_LANCE_LANCE_INDEX", "KHORA_STORAGE__SQLITE_LANCE__LANCE_INDEX"),
    ("KHORA_STORAGE_SQLITE_LANCE_IVF_PARTITIONS", "KHORA_STORAGE__SQLITE_LANCE__IVF_PARTITIONS"),
    ("KHORA_STORAGE_SQLITE_LANCE_HNSW_M", "KHORA_STORAGE__SQLITE_LANCE__HNSW_M"),
    ("KHORA_STORAGE_SQLITE_LANCE_RETRAIN_FACTOR", "KHORA_STORAGE__SQLITE_LANCE__RETRAIN_FACTOR"),
)


# dream.ops aliases — owned by ``DreamConfig`` (a ``BaseSettings`` with
# its own env-prefix ``KHORA_DREAM_``), so the canonical-→-legacy
# promotion + conflict check fires inside ``DreamConfig._reject_alias_conflicts``.
_DREAM_ENV_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("KHORA_DREAM_OPS_DEDUPE_ENTITIES", "KHORA_DREAM_OPS__DEDUPE_ENTITIES"),
    ("KHORA_DREAM_OPS_PRUNE_EDGES", "KHORA_DREAM_OPS__PRUNE_EDGES"),
    ("KHORA_DREAM_OPS_COMPACT_FACTS", "KHORA_DREAM_OPS__COMPACT_FACTS"),
    ("KHORA_DREAM_OPS_CLUSTER_EVENTS", "KHORA_DREAM_OPS__CLUSTER_EVENTS"),
    ("KHORA_DREAM_OPS_RECOMPUTE_CENTROIDS", "KHORA_DREAM_OPS__RECOMPUTE_CENTROIDS"),
)


# ---------------------------------------------------------------------------
# Graph backend configs (discriminated union on "backend" field)
# ---------------------------------------------------------------------------


class Neo4jConfig(BaseModel):
    """Neo4j graph backend configuration."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["neo4j"] = "neo4j"
    url: SecretStr | None = Field(
        default=None,
        description="Neo4j connection URL (bolt:// or neo4j://)",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_URL", "KHORA_STORAGE__GRAPH__URL"),
    )
    user: str = Field(
        default="neo4j",
        description="Neo4j username",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_USER", "KHORA_STORAGE__GRAPH__USER"),
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Neo4j password",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_PASSWORD", "KHORA_STORAGE__GRAPH__PASSWORD"),
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_DATABASE", "KHORA_STORAGE__GRAPH__DATABASE"),
    )
    max_connection_pool_size: int = Field(
        default=100,
        description="Neo4j connection pool size",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_MAX_CONNECTION_POOL_SIZE",
            "KHORA_STORAGE__GRAPH__MAX_CONNECTION_POOL_SIZE",
        ),
    )
    connection_acquisition_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds waiting for a connection from the pool",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_CONNECTION_ACQUISITION_TIMEOUT",
            "KHORA_STORAGE__GRAPH__CONNECTION_ACQUISITION_TIMEOUT",
        ),
    )
    retry_delay_jitter_factor: float = Field(
        default=0.5,
        description="Jitter factor for transaction retry delays (0.0-1.0)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_RETRY_DELAY_JITTER_FACTOR",
            "KHORA_STORAGE__GRAPH__RETRY_DELAY_JITTER_FACTOR",
        ),
    )
    max_connection_lifetime: int = Field(
        default=900,
        description="Max seconds a connection stays in the pool before rotation. "
        "Set below the server-side TTL (e.g. Aura ~20min) to avoid reset errors.",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_MAX_CONNECTION_LIFETIME",
            "KHORA_STORAGE__GRAPH__MAX_CONNECTION_LIFETIME",
        ),
    )
    liveness_check_timeout: float | None = Field(
        default=30.0,
        description="Seconds of idle time after which connections are checked for liveness "
        "before being returned from the pool. None disables the check.",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_LIVENESS_CHECK_TIMEOUT",
            "KHORA_STORAGE__GRAPH__LIVENESS_CHECK_TIMEOUT",
        ),
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
            "via env var KHORA_STORAGE_GRAPH_QUERY_TIMEOUT."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_QUERY_TIMEOUT",
            "KHORA_STORAGE__GRAPH__QUERY_TIMEOUT",
        ),
    )
    entity_write_concurrency: int = Field(
        default=12,
        description="Max concurrent entity write transactions",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_ENTITY_WRITE_CONCURRENCY",
            "KHORA_STORAGE__GRAPH__ENTITY_WRITE_CONCURRENCY",
        ),
    )
    relationship_write_concurrency: int = Field(
        default=8,
        description="Max concurrent relationship write transactions",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_RELATIONSHIP_WRITE_CONCURRENCY",
            "KHORA_STORAGE__GRAPH__RELATIONSHIP_WRITE_CONCURRENCY",
        ),
    )
    pool_sampler_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in high-frequency Neo4j pool sampler. When True, Khora starts a "
            "background task that samples driver pool state at "
            "``pool_sampler_interval_ms`` cadence and emits the observations on the "
            "``khora.neo4j.pool.sampled.*`` histograms. Zero-cost when False. "
            "Set via env: KHORA_STORAGE_GRAPH_POOL_SAMPLER_ENABLED=true."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_POOL_SAMPLER_ENABLED",
            "KHORA_STORAGE__GRAPH__POOL_SAMPLER_ENABLED",
        ),
    )
    pool_sampler_interval_ms: int = Field(
        default=500,
        ge=50,
        le=60_000,
        description=(
            "Interval in milliseconds between Neo4j pool samples when "
            "``pool_sampler_enabled`` is True. Clamped to [50, 60000]. "
            "Set via env: KHORA_STORAGE_GRAPH_POOL_SAMPLER_INTERVAL_MS=250."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_POOL_SAMPLER_INTERVAL_MS",
            "KHORA_STORAGE__GRAPH__POOL_SAMPLER_INTERVAL_MS",
        ),
    )
    pool_keepalive_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in Neo4j connection-pool keepalive. When True, Khora starts a "
            "background task that fires periodic ``RETURN 1`` pings on idle pooled "
            "connections at ``pool_keepalive_interval_ms`` cadence so they are never "
            "idle-dropped by an intermediary (load balancer, firewall) before the "
            "driver's own liveness check would catch a stale connection. Zero-cost "
            "when False. Set via env: KHORA_STORAGE_GRAPH_POOL_KEEPALIVE_ENABLED=true."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_POOL_KEEPALIVE_ENABLED",
            "KHORA_STORAGE__GRAPH__POOL_KEEPALIVE_ENABLED",
        ),
    )
    pool_keepalive_interval_ms: int = Field(
        default=15000,
        ge=50,
        le=60_000,
        description=(
            "Interval in milliseconds between Neo4j keepalive pings when "
            "``pool_keepalive_enabled`` is True. Clamped to [50, 60000]. Default "
            "15000 is intentionally below the driver's 30s liveness window so idle "
            "connections are exercised before they go stale. "
            "Set via env: KHORA_STORAGE_GRAPH_POOL_KEEPALIVE_INTERVAL_MS=15000."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_POOL_KEEPALIVE_INTERVAL_MS",
            "KHORA_STORAGE__GRAPH__POOL_KEEPALIVE_INTERVAL_MS",
        ),
    )
    relationship_source_document_ids_max: int = Field(
        default=100,
        ge=1,
        description=(
            "Max provenance ``source_document_ids`` retained on a relationship "
            "after MERGE. When (existing + incoming) exceeds this cap the "
            "most-recent tail is kept and the over-limit entries are dropped — "
            "a warning is logged and ``khora.neo4j.relationship.source_id_truncated`` "
            "is incremented with field=source_document_ids. Default 100 preserves "
            "pre-#737 behavior; deep-provenance workloads should raise it. "
            "Set via env: KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX=500."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX",
            "KHORA_STORAGE__GRAPH__RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX",
        ),
    )
    relationship_source_chunk_ids_max: int = Field(
        default=250,
        ge=1,
        description=(
            "Max provenance ``source_chunk_ids`` retained on a relationship "
            "after MERGE. When (existing + incoming) exceeds this cap the "
            "most-recent tail is kept and the over-limit entries are dropped — "
            "a warning is logged and ``khora.neo4j.relationship.source_id_truncated`` "
            "is incremented with field=source_chunk_ids. Default 250 preserves "
            "pre-#737 behavior; deep-provenance workloads should raise it. "
            "Set via env: KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_CHUNK_IDS_MAX=1000."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_CHUNK_IDS_MAX",
            "KHORA_STORAGE__GRAPH__RELATIONSHIP_SOURCE_CHUNK_IDS_MAX",
        ),
    )
    entity_source_document_ids_max: int = Field(
        default=100,
        ge=1,
        description=(
            "Max provenance ``source_document_ids`` retained on an entity after "
            "MERGE. When (existing + incoming) exceeds this cap the most-recent "
            "tail is kept and the over-limit entries are dropped — a warning is "
            "logged and ``khora.neo4j.entity.source_id_truncated`` is incremented "
            "with field=source_document_ids. Default 100 preserves pre-#777 "
            "behavior; deep-provenance workloads should raise it. "
            "Set via env: KHORA_STORAGE_GRAPH_ENTITY_SOURCE_DOCUMENT_IDS_MAX=500."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_ENTITY_SOURCE_DOCUMENT_IDS_MAX",
            "KHORA_STORAGE__GRAPH__ENTITY_SOURCE_DOCUMENT_IDS_MAX",
        ),
    )
    entity_source_chunk_ids_max: int = Field(
        default=250,
        ge=1,
        description=(
            "Max provenance ``source_chunk_ids`` retained on an entity after "
            "MERGE. When (existing + incoming) exceeds this cap the most-recent "
            "tail is kept and the over-limit entries are dropped — a warning is "
            "logged and ``khora.neo4j.entity.source_id_truncated`` is incremented "
            "with field=source_chunk_ids. Default 250 preserves pre-#777 behavior; "
            "deep-provenance workloads should raise it. "
            "Set via env: KHORA_STORAGE_GRAPH_ENTITY_SOURCE_CHUNK_IDS_MAX=1000."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_ENTITY_SOURCE_CHUNK_IDS_MAX",
            "KHORA_STORAGE__GRAPH__ENTITY_SOURCE_CHUNK_IDS_MAX",
        ),
    )


class MemgraphConfig(BaseModel):
    """Memgraph graph backend configuration."""

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["memgraph"] = "memgraph"
    url: SecretStr | None = Field(
        default=None,
        description="Memgraph connection URL (bolt://)",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_URL", "KHORA_STORAGE__GRAPH__URL"),
    )
    user: str = Field(
        default="memgraph",
        description="Memgraph username",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_USER", "KHORA_STORAGE__GRAPH__USER"),
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Memgraph password",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_PASSWORD", "KHORA_STORAGE__GRAPH__PASSWORD"),
    )


class NeptuneConfig(BaseModel):
    """AWS Neptune graph backend configuration.

    Neptune supports openCypher via Bolt protocol. Uses the same neo4j
    Python driver as Neo4j and Memgraph backends.
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["neptune"] = "neptune"
    url: SecretStr | None = Field(
        default=None,
        description="Neptune Bolt endpoint (bolt://cluster:8182)",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_URL", "KHORA_STORAGE__GRAPH__URL"),
    )
    user: str = Field(
        default="",
        description="Username (empty for IAM auth)",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_USER", "KHORA_STORAGE__GRAPH__USER"),
    )
    password: SecretStr = Field(
        default=SecretStr(""),
        description="Password (empty for IAM auth)",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_PASSWORD", "KHORA_STORAGE__GRAPH__PASSWORD"),
    )
    iam_auth: bool = Field(
        default=False,
        description="Use AWS IAM SigV4 authentication",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_IAM_AUTH", "KHORA_STORAGE__GRAPH__IAM_AUTH"),
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for IAM auth signing",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_AWS_REGION", "KHORA_STORAGE__GRAPH__AWS_REGION"),
    )
    max_connection_pool_size: int = Field(
        default=100,
        description="Bolt connection pool size (Neptune max: 1000)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_MAX_CONNECTION_POOL_SIZE",
            "KHORA_STORAGE__GRAPH__MAX_CONNECTION_POOL_SIZE",
        ),
    )


class SurrealDBConfig(BaseModel):
    """SurrealDB unified backend configuration (graph role).

    SurrealDB serves as a unified backend providing graph, vector, and
    relational storage in a single database.

    Note: this class is reachable through TWO parent paths — ``storage.graph``
    (when ``backend=surrealdb``) and ``storage.surrealdb`` (unified-mode slot).
    Each aliased field therefore declares both ``_GRAPH_`` and ``_SURREALDB_``
    env-var prefixes (single- and double-underscore variants of each).
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["surrealdb"] = "surrealdb"
    mode: str = Field(
        default="memory",
        description="Connection mode: memory, embedded, or remote",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_MODE",
            "KHORA_STORAGE_SURREALDB_MODE",
            "KHORA_STORAGE__GRAPH__MODE",
            "KHORA_STORAGE__SURREALDB__MODE",
        ),
    )
    url: SecretStr | None = Field(
        default=None,
        description="SurrealDB WebSocket URL (for remote mode)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_URL",
            "KHORA_STORAGE_SURREALDB_URL",
            "KHORA_STORAGE__GRAPH__URL",
            "KHORA_STORAGE__SURREALDB__URL",
        ),
    )
    path: str | None = Field(
        default=None,
        description="Database file path (for embedded mode)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_PATH",
            "KHORA_STORAGE_SURREALDB_PATH",
            "KHORA_STORAGE__GRAPH__PATH",
            "KHORA_STORAGE__SURREALDB__PATH",
        ),
    )
    namespace: str = Field(
        default="khora",
        description="SurrealDB namespace",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_NAMESPACE",
            "KHORA_STORAGE_SURREALDB_NAMESPACE",
            "KHORA_STORAGE__GRAPH__NAMESPACE",
            "KHORA_STORAGE__SURREALDB__NAMESPACE",
        ),
    )
    database: str = Field(
        default="default",
        description="SurrealDB database",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_DATABASE",
            "KHORA_STORAGE_SURREALDB_DATABASE",
            "KHORA_STORAGE__GRAPH__DATABASE",
            "KHORA_STORAGE__SURREALDB__DATABASE",
        ),
    )
    user: str = Field(
        default="root",
        description="SurrealDB username",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_USER",
            "KHORA_STORAGE_SURREALDB_USER",
            "KHORA_STORAGE__GRAPH__USER",
            "KHORA_STORAGE__SURREALDB__USER",
        ),
    )
    password: SecretStr = Field(
        default=SecretStr("root"),
        description="SurrealDB password",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_PASSWORD",
            "KHORA_STORAGE_SURREALDB_PASSWORD",
            "KHORA_STORAGE__GRAPH__PASSWORD",
            "KHORA_STORAGE__SURREALDB__PASSWORD",
        ),
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_EMBEDDING_DIMENSION",
            "KHORA_STORAGE_SURREALDB_EMBEDDING_DIMENSION",
            "KHORA_STORAGE__GRAPH__EMBEDDING_DIMENSION",
            "KHORA_STORAGE__SURREALDB__EMBEDDING_DIMENSION",
        ),
    )
    sync_data: bool = Field(
        default=True,
        description="Enable SURREAL_SYNC_DATA for crash-safe writes",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_GRAPH_SYNC_DATA",
            "KHORA_STORAGE_SURREALDB_SYNC_DATA",
            "KHORA_STORAGE__GRAPH__SYNC_DATA",
            "KHORA_STORAGE__SURREALDB__SYNC_DATA",
        ),
    )


class SQLiteLanceConfig(BaseModel):
    """SQLite + LanceDB embedded unified backend configuration.

    Pairs an on-disk SQLite database (graph + relational + event store)
    with a sibling LanceDB directory (vector search). Zero infrastructure —
    both backends run in-process.
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["sqlite_lance"] = "sqlite_lance"
    db_path: str = Field(
        default="./khora.db",
        description="SQLite database path",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_DB_PATH",
            "KHORA_STORAGE__SQLITE_LANCE__DB_PATH",
        ),
    )
    lance_path: str | None = Field(
        default=None,
        description="LanceDB directory path. When None, derived from db_path (sibling .lance dir).",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_LANCE_PATH",
            "KHORA_STORAGE__SQLITE_LANCE__LANCE_PATH",
        ),
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_EMBEDDING_DIMENSION",
            "KHORA_STORAGE__SQLITE_LANCE__EMBEDDING_DIMENSION",
        ),
    )
    use_halfvec: bool = Field(
        default=False,
        description="Store embeddings as float16 to halve index size (minor recall loss).",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_USE_HALFVEC",
            "KHORA_STORAGE__SQLITE_LANCE__USE_HALFVEC",
        ),
    )
    lance_index: Literal["auto", "ivf_pq", "hnsw", "brute"] = Field(
        default="auto",
        description="Vector index type. 'auto' picks based on table size.",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_LANCE_INDEX",
            "KHORA_STORAGE__SQLITE_LANCE__LANCE_INDEX",
        ),
    )
    ivf_partitions: int | None = Field(
        default=None,
        description="IVF partition count (ivf_pq only). None = auto from row count.",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_IVF_PARTITIONS",
            "KHORA_STORAGE__SQLITE_LANCE__IVF_PARTITIONS",
        ),
    )
    hnsw_m: int = Field(
        default=16,
        description="HNSW M parameter (max connections per layer)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_HNSW_M",
            "KHORA_STORAGE__SQLITE_LANCE__HNSW_M",
        ),
    )
    retrain_factor: float = Field(
        default=2.0,
        description=(
            "Rebuild the LanceDB ANN index once the row count grows to "
            "retrain_factor * (rows at last training). Default 2.0 retrains "
            "when the corpus has doubled. Set <= 1.0 to disable retraining."
        ),
        validation_alias=_env_aliases(
            "KHORA_STORAGE_SQLITE_LANCE_RETRAIN_FACTOR",
            "KHORA_STORAGE__SQLITE_LANCE__RETRAIN_FACTOR",
        ),
    )


class AGEConfig(BaseModel):
    """PostgreSQL AGE graph backend configuration.

    Uses Apache AGE extension to run openCypher queries inside PostgreSQL.
    Can share the same connection pool as the relational backend.
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["age"] = "age"
    url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL URL (can share with relational backend)",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_URL", "KHORA_STORAGE__GRAPH__URL"),
    )
    graph_name: str = Field(
        default="khora_graph",
        description="Name of the AGE graph",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_GRAPH_NAME", "KHORA_STORAGE__GRAPH__GRAPH_NAME"),
    )
    pool_size: int = Field(
        default=10,
        description="Connection pool size",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_POOL_SIZE", "KHORA_STORAGE__GRAPH__POOL_SIZE"),
    )
    max_overflow: int = Field(
        default=20,
        description="Max overflow connections",
        validation_alias=_env_aliases("KHORA_STORAGE_GRAPH_MAX_OVERFLOW", "KHORA_STORAGE__GRAPH__MAX_OVERFLOW"),
    )


def _graph_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("backend", "neo4j")
    return getattr(v, "backend", "neo4j")


GraphConfig = Annotated[
    Annotated[Neo4jConfig, Tag("neo4j")]
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

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["pgvector"] = "pgvector"
    url: SecretStr | None = Field(
        default=None,
        description="pgvector connection URL",
        validation_alias=_env_aliases("KHORA_STORAGE_VECTOR_URL", "KHORA_STORAGE__VECTOR__URL"),
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION",
            "KHORA_STORAGE__VECTOR__EMBEDDING_DIMENSION",
        ),
    )


class SurrealDBVectorConfig(BaseModel):
    """SurrealDB unified backend configuration (vector role).

    Shares the same SurrealDB instance as the graph role. Like
    :class:`SurrealDBConfig` (graph role), this class is parented under
    ``storage.vector`` but the env-var contract has historically routed
    SurrealDB unified-mode config through both ``KHORA_STORAGE_VECTOR_*``
    and the central ``KHORA_STORAGE_SURREALDB_*`` slots — both forms are
    accepted here.
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["surrealdb"] = "surrealdb"
    mode: str = Field(
        default="memory",
        description="Connection mode: memory, embedded, or remote",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_MODE",
            "KHORA_STORAGE_SURREALDB_MODE",
            "KHORA_STORAGE__VECTOR__MODE",
            "KHORA_STORAGE__SURREALDB__MODE",
        ),
    )
    url: SecretStr | None = Field(
        default=None,
        description="SurrealDB WebSocket URL (for remote mode)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_URL",
            "KHORA_STORAGE_SURREALDB_URL",
            "KHORA_STORAGE__VECTOR__URL",
            "KHORA_STORAGE__SURREALDB__URL",
        ),
    )
    path: str | None = Field(
        default=None,
        description="Database file path (for embedded mode)",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_PATH",
            "KHORA_STORAGE_SURREALDB_PATH",
            "KHORA_STORAGE__VECTOR__PATH",
            "KHORA_STORAGE__SURREALDB__PATH",
        ),
    )
    namespace: str = Field(
        default="khora",
        description="SurrealDB namespace",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_NAMESPACE",
            "KHORA_STORAGE_SURREALDB_NAMESPACE",
            "KHORA_STORAGE__VECTOR__NAMESPACE",
            "KHORA_STORAGE__SURREALDB__NAMESPACE",
        ),
    )
    database: str = Field(
        default="default",
        description="SurrealDB database",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_DATABASE",
            "KHORA_STORAGE_SURREALDB_DATABASE",
            "KHORA_STORAGE__VECTOR__DATABASE",
            "KHORA_STORAGE__SURREALDB__DATABASE",
        ),
    )
    user: str = Field(
        default="root",
        description="SurrealDB username",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_USER",
            "KHORA_STORAGE_SURREALDB_USER",
            "KHORA_STORAGE__VECTOR__USER",
            "KHORA_STORAGE__SURREALDB__USER",
        ),
    )
    password: SecretStr = Field(
        default=SecretStr("root"),
        description="SurrealDB password",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_PASSWORD",
            "KHORA_STORAGE_SURREALDB_PASSWORD",
            "KHORA_STORAGE__VECTOR__PASSWORD",
            "KHORA_STORAGE__SURREALDB__PASSWORD",
        ),
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION",
            "KHORA_STORAGE_SURREALDB_EMBEDDING_DIMENSION",
            "KHORA_STORAGE__VECTOR__EMBEDDING_DIMENSION",
            "KHORA_STORAGE__SURREALDB__EMBEDDING_DIMENSION",
        ),
    )


class SQLiteVectorConfig(BaseModel):
    """SQLite embedded backend configuration (relational + vector).

    Uses a single SQLite file for both relational and vector storage.
    Vector search is brute-force cosine via khora._accel.
    """

    model_config = {"extra": "forbid", "populate_by_name": True}

    backend: Literal["sqlite"] = "sqlite"
    url: str | None = Field(
        default=None,
        description="SQLite path (sqlite:///path.db or :memory:)",
        validation_alias=_env_aliases("KHORA_STORAGE_VECTOR_URL", "KHORA_STORAGE__VECTOR__URL"),
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
        validation_alias=_env_aliases(
            "KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION",
            "KHORA_STORAGE__VECTOR__EMBEDDING_DIMENSION",
        ),
    )


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

    @model_validator(mode="after")
    def _guard_postgres_embedding_dimension(self) -> StorageSettings:
        """Reject non-1536 embedding dimensions on the Postgres backend.

        The Postgres / pgvector schema hardcodes ``Vector(1536)`` columns
        (#925), so a configured dimension other than 1536 would pass
        through the embedder (#926) and then crash at store time. Until the
        parameterized migration lands (tracked in #925 for a future
        release) we fail fast at config time on the Postgres path only.
        sqlite_lance and surrealdb size their vector columns from config
        and support arbitrary dimensions, so this guard does not apply to
        them.
        """
        if self.backend != "postgres":
            return self
        pgvector_dim = self.vector.embedding_dimension if isinstance(self.vector, PgVectorConfig) else 1536
        if pgvector_dim != 1536 or self.embedding_dimension != 1536:
            raise ValueError(
                "Postgres backend currently supports only embedding_dimension=1536; "
                "arbitrary dimensions are tracked in #925 for a future release. "
                "Use sqlite_lance for other dimensions, or set embedding_dimension=1536."
            )
        return self


class LLMSettings(BaseSettings):
    """LLM configuration settings.

    Env vars: ``KHORA_LLM_MODEL``, ``KHORA_LLM_EMBEDDING_MODEL``, etc.
    """

    model_config = SettingsConfigDict(env_prefix="KHORA_LLM_", case_sensitive=False)

    model: str = Field(default="gpt-4o-mini", description="Primary LLM model")
    api_key_env: Annotated[str, AllowSecretTyping(reason="env-var name pointer, not a credential")] = Field(
        default=DEFAULT_API_KEY_ENV,
        description=(
            "Environment variable that holds the API key. When left at the "
            "OpenAI default but ``model`` names another provider (``gemini/``, "
            "``claude``, ``anthropic/``, ``vertex_ai/``, ...) it is auto-derived "
            "from the model prefix so the wrong provider's key is not read. An "
            "explicit non-default value is always honored verbatim."
        ),
    )
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=12288, description="Maximum tokens for LLM extraction output")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retries on failure")
    max_concurrent_llm_calls: int = Field(default=10, description="Maximum concurrent LLM calls")
    extraction_wave_size: int = Field(
        default=20,
        ge=1,
        description=(
            "Number of extraction batches dispatched concurrently per wave in "
            "LLMEntityExtractor.extract_multi(). The circuit breaker is checked "
            "between waves. Raising this above max_concurrent_llm_calls has no "
            "effect (the per-call semaphore is the binding limit); raising both "
            "increases throughput but also the worst-case doomed-call count when "
            "the circuit breaker trips."
        ),
    )

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

    @model_validator(mode="after")
    def _derive_api_key_env(self) -> LLMSettings:
        """Co-evolve ``api_key_env`` with ``model`` when left at the OpenAI default.

        Without this, ``LLMSettings(model="gemini/...")`` silently keeps
        ``api_key_env="OPENAI_API_KEY"`` and reads the OpenAI key for a Google
        model. We only override the default; an explicit value is left alone.
        """
        if self.api_key_env == DEFAULT_API_KEY_ENV:
            derived = derive_api_key_env(self.model)
            if derived is not None and derived != self.api_key_env:
                self.api_key_env = derived
        return self


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
    extract_entities: bool = Field(
        default=True,
        description=(
            "Global switch for entity/relationship extraction at ingest. False "
            "leaves the entity graph empty (vector + keyword search still work, but "
            "graph/Cypher/multi-hop retrieval returns nothing and list_entities() is 0)."
        ),
    )
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
    ketrag_skeleton_channel: bool = Field(
        default=False,
        description="Opt-in KET-RAG-faithful skeleton: multilingual keyword tokenizer + "
        "keyword-PageRank chunk selection + (later) a separate keyword-chunk retrieval channel "
        "kept out of the entity graph. Default off; no behavior change when off.",
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
        ge=1,
        description="Maximum documents to process concurrently in the unified pending processor.",
    )
    pending_processor_grace_period_minutes: int = Field(
        default=5,
        description="Minimum age (minutes) a PENDING document must have before it is eligible for "
        "crash-recovery processing. Avoids racing with in-flight writes.",
    )
    pending_processor_orphan_stale_after_seconds: int = Field(
        default=900,
        description="Minimum age (seconds) a PROCESSING document must have before it is reclaimed as an "
        "orphan (a worker that crashed mid-process). Default 900s (15 min) is generous to avoid "
        "preempting a slow-but-alive worker.",
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
    default_mode: str = Field(default="hybrid", description="Default search mode: vector, graph, hybrid, keyword, all")
    # Wired onto the default VectorCypher engine in #1406 as the chunk-channel
    # cosine floor. Default canonicalized to 0.0 (the engine's previous
    # effective behavior - no floor) so the wiring changed nothing silently;
    # the old 0.05 here never took effect on the default recall() path. Set
    # KHORA_QUERY_MIN_CHUNK_SIMILARITY to opt in to a floor.
    min_chunk_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum chunk similarity threshold")
    min_entity_similarity: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Minimum entity similarity threshold"
    )

    # Fusion weights. Wired onto the default VectorCypher engine in #1406
    # (previously dead there - the engine read only VectorCypherConfig). The
    # defaults were canonicalized to the values the engine already used
    # (``fusion_vector_weight=0.6`` / ``fusion_graph_weight=0.4`` /
    # ``bm25_weight=0.3``) so the wiring changed no default behavior; the old
    # 0.5/0.3/0.2 here never took effect on the default recall() path.
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight for vector search in fusion")
    graph_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for graph search in fusion")
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of the lexical (BM25 keyword) channel in fusion (fills the bm25_weight slot)",
    )

    # Independent lexical (BM25 full-text) channel fused alongside vector + graph
    # via RRF. Default OFF (unchanged). Exposing it here (#1330) makes the
    # channel operable from public config via KHORA_QUERY_ENABLE_BM25_CHANNEL;
    # it previously lived only on VectorCypherConfig and was unreachable.
    enable_bm25_channel: bool = Field(
        default=False, description="Enable the independent BM25 lexical channel in fusion"
    )

    # Lexical-channel selector (#1391). Picks which retriever fills the lexical
    # recall slot: "bm25" (default, current behavior, byte-identical) or
    # "keyword_ppr" (experimental KET-RAG text-keyword channel: per-query
    # personalized PageRank over the namespace keyword->chunk bipartite). Default
    # "bm25" = unchanged. An earlier synthetic-corpus spike found the PPR channel
    # adds ~no marginal recall over BM25, so this exists to A/B it on real data,
    # not as a recommended default. Switching to "keyword_ppr" requires a
    # re-ingest to populate the keyword_chunks edge table.
    lexical_channel: Literal["bm25", "keyword_ppr"] = Field(
        default="bm25",
        description=(
            "Which retriever fills the lexical recall slot: 'bm25' (default, "
            "unchanged) or 'keyword_ppr' (experimental keyword-chunk PageRank "
            "channel; requires re-ingest to populate keyword_chunks)."
        ),
    )
    # Damping factor for the keyword_ppr channel's per-query PageRank. Only used
    # when lexical_channel == "keyword_ppr".
    keyword_ppr_damping: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Damping factor for the keyword_ppr lexical channel's per-query PageRank.",
    )
    # Cap on the number of keyword->chunk bipartite edges loaded per query for the
    # keyword_ppr channel. Bounds per-query PageRank cost (heavier than BM25's
    # inverted-index lookup). Only used when lexical_channel == "keyword_ppr".
    keyword_ppr_max_edges: int = Field(
        default=50_000,
        ge=1,
        description=(
            "Max keyword->chunk bipartite edges loaded per query for the "
            "keyword_ppr lexical channel (bounds per-query PageRank cost)."
        ),
    )

    # Coherence re-rank: a small post-fusion nudge that demotes word-shuffled /
    # disfluent confounders. Applied to [0,1]-normalized fused scores so it acts
    # as a true ~w nudge (#1056). Set to 0.0 to disable
    # (``KHORA_QUERY_COHERENCE_WEIGHT=0``).
    coherence_weight: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Weight of the bigram-coherence re-rank after fusion (0.0 disables)"
    )

    # Temporal settings.
    # `recency_weight` and `recency_decay_days` were tightened
    # from (0.2, 30) to (0.35, 7) after BEAM 100K showed the four weakest
    # categories (event_ordering, contradiction_resolution, temporal_reasoning,
    # knowledge_update) all share a weak-recency / supersession root cause —
    # 30-day half-life is wider than most session lifetimes, and 0.2 is too
    # gentle to break ties between an old fact and its in-session update.
    # `apply_recency_bias` still defaults False: callers must opt in.
    apply_recency_bias: bool = Field(default=False, description="Apply recency bias to results")
    recency_weight: float = Field(default=0.35, ge=0.0, le=1.0, description="Weight of recency in scoring")
    recency_decay_days: float = Field(default=7.0, ge=1.0, description="Days for recency score to decay by half")

    # Issue #567 — temporal recency Phase A. All four flags default OFF so the
    # PR is shippable without changing existing-consumer score distributions.
    # Operators opt in per-namespace via KHORA_QUERY_TEMPORAL_* env vars.
    temporal_recency_floor_enabled: bool = Field(
        default=False,
        description=(
            "When True, RECENCY/CHANGE queries that lack an explicit date "
            "and contain no anti-recency token ('ever', 'all', 'history', "
            "'over time', etc.) get a synthetic date floor from "
            "RETRIEVAL_PARAMS.default_window_days. See docs/observability.md."
        ),
    )
    temporal_reference_wall_clock: bool = Field(
        default=False,
        description=(
            "When True, _calculate_recency_scores uses datetime.now(UTC) as "
            "the reference time instead of max(occurred_at in result set). "
            "The latter is correct for benchmark replay (KHORA_BENCH_MODE) "
            "but wrong for production where 'recent' means recent-in-wall-clock."
        ),
    )
    temporal_recency_channel_enabled: bool = Field(
        default=False,
        description=(
            "When True, RECENCY/CHANGE queries fuse a parallel 'recency "
            "channel' (ORDER BY COALESCE(source_timestamp, created_at) DESC) "
            "alongside the cosine and BM25 channels via RRF. Chunks from the "
            "recency channel only enter fusion when their cosine to the query "
            "embedding exceeds temporal_query_relevance_floor."
        ),
    )
    temporal_per_source_decay: bool = Field(
        default=False,
        description=(
            "When True, _calculate_recency_scores looks up decay_days per "
            "chunk via chunk.metadata['source_system'] in "
            "temporal_default_decay_by_source. Falls back to the category's "
            "decay_days_override when no per-source entry exists."
        ),
    )
    temporal_query_relevance_floor: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold a chunk must exceed to enter the "
            "recency-channel fusion. Prevents today's HR-channel chunks from "
            "muscling into top-K for a niche query. Raised from 0.30 to 0.40 "
            "after LoCoMo --small (PR #571) showed a persistent 4.2pp "
            "abstention_accuracy regression from chunks just-above-floor "
            "diluting the engine's confidence-to-abstain signal."
        ),
    )
    temporal_recency_channel_limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Per-channel limit for the parallel recency channel SQL.",
    )
    temporal_llm_disambiguation_enabled: bool = Field(
        default=False,
        description=(
            "When True, queries that fire RECENCY/CHANGE in the Aho-Corasick "
            "tier AND contain ambiguity-trigger tokens ('would', 'if', "
            "'previously', etc.) are routed to an LLM classifier that "
            "outputs RECENT/HISTORICAL/COUNTERFACTUAL/NEUTRAL. The "
            "synthetic floor is vetoed for non-RECENT outputs. Cost is "
            "bounded by query distinct-count (results cached per-query). "
            "Targets the LoCoMo counterfactual regression seen in PR #571."
        ),
    )
    temporal_llm_disambiguation_model: str | None = Field(
        default=None,
        description=(
            "Override model for temporal intent classification. None uses "
            "gpt-4o-mini (small + fast). Pass any LiteLLM-supported model."
        ),
    )
    temporal_semantic_fallback_enabled: bool = Field(
        default=False,
        description=(
            "When True, queries that the English Aho-Corasick keyword tier "
            "classifies as NONE are routed to a small LLM classifier that "
            "resolves German / multilingual and paraphrased temporal intent "
            "into the correct TemporalCategory (#981). Default OFF: keyword-only "
            "and zero LLM cost. Cost is bounded by distinct keyword-missed "
            "queries (results cached per-query); any LLM failure/timeout "
            "degrades back to the keyword (NONE) result."
        ),
    )
    temporal_semantic_fallback_model: str | None = Field(
        default=None,
        description=(
            "Override model for the Tier-2 temporal-category classifier. None "
            "uses gpt-4o-mini (small + fast). Pass any LiteLLM-supported model."
        ),
    )
    temporal_default_decay_by_source: dict[str, int] = Field(
        default_factory=lambda: {
            "slack": 3,
            "email": 7,
            "calendar": 14,
            "salesforce": 180,
            "_default": 14,
        },
        description=(
            "Per-source decay-days override. Keys are source_system values "
            "set by connectors via metadata.custom['source_system']. "
            "'_default' applies when the chunk's source_system is None or "
            "absent from the dict."
        ),
    )

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
        default="BAAI/bge-reranker-v2-m3",
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
    # HyDE-Cypher (Issue #595, Phase D2). Default OFF — flip after a positive
    # A/B on a hand-curated structured-query set. When ON, the engine asks an
    # LLM to pick a parameterized Cypher template for the query and executes
    # it as an additional retrieval channel; on selection "none" or any
    # validation error, falls back to text HyDE.
    enable_hyde_cypher: bool = Field(
        default=False,
        description=(
            "Enable HyDE-Cypher templated graph queries as an additional retrieval "
            "channel for structured queries. See khora#595."
        ),
    )
    hyde_cypher_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Max entities returned by a HyDE-Cypher template execution.",
    )

    # Personalized PageRank retrieval (Issue #542 — HippoRAG 2 style).
    # Default OFF: when ON, the VectorCypher path replaces BFS graph
    # expansion with query-time Personalized PageRank seeded from the
    # entry entities discovered via vector search. Falls back to vector-
    # only when no entry entities are found or the entity graph is empty.
    enable_ppr_retrieval: bool = Field(
        default=False,
        description=(
            "Enable query-time Personalized PageRank in VectorCypher retrieval "
            "(HippoRAG 2). Replaces the BFS graph-expansion channel with a "
            "PPR-weighted entity walk seeded from query entities. Falls back "
            "to vector-only when the entity graph is empty or no entry "
            "entities are found. See khora#542."
        ),
    )
    ppr_damping: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Damping factor for Personalized PageRank.",
    )
    ppr_max_iter: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum iterations for Personalized PageRank power method.",
    )
    ppr_tol: float = Field(
        default=1e-5,
        ge=0.0,
        le=1.0,
        description="Convergence tolerance for Personalized PageRank.",
    )
    ppr_top_entities: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Number of top PR-scored entities used to score chunks in the PPR retrieval path.",
    )
    ppr_neighborhood_per_seed_limit: int = Field(
        default=64,
        ge=1,
        le=1000,
        description=(
            "When the PPR entity slice hits its cap (namespace larger than the "
            "~5000-entity slice), augment it with each query seed's 1-hop "
            "neighborhood, fetching at most this many relationships per seed so "
            "the resolved seeds survive into the graph. Below the cap this is "
            "inert (no extra round-trips). See khora#1373."
        ),
    )
    ppr_max_neighborhood_entities: int = Field(
        default=2000,
        ge=1,
        le=50_000,
        description=(
            "Upper bound on how far seed-anchored augmentation may grow the PPR "
            "entity set above the global slice (khora#1373). The effective bound "
            "is max(this, len(slice)), so the base slice (the multi-hop backbone) "
            "is never shrunk; seeds are kept first when trimming."
        ),
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
        # Must match khora.engines.chronicle.engine.DEFAULT_CHRONICLE_HALF_LIFE_HOURS
        # (kept duplicated to avoid engines -> config import cycle).
        default=168.0,
        ge=1.0,
        description="Half-life in hours for temporal decay. Default 168h (7 days): "
        "a memory retains ~50% strength after one week, ~25% after two weeks. "
        "Used by Chronicle's Ebbinghaus decay; also consulted by VectorCypher's "
        "soft temporal scoring.",
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
        # Must match khora.engines.chronicle.engine.DEFAULT_CHRONICLE_DECAY_WEIGHT
        # (kept duplicated to avoid engines -> config import cycle).
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight of temporal decay in Chronicle's multiplicative scoring "
        "blend: final = relevance * ((1 - w) + w * retention). Default 0.30 means "
        "a fully-faded memory (retention -> 0) keeps 70% of its relevance score, "
        "while a fresh memory keeps 100%. Higher values lean more heavily on recency.",
    )
    chronicle_overfetch_multiplier: int = Field(
        default=4, ge=2, le=10, description="Over-fetch multiplier for Chronicle retrieval channels"
    )
    metadata_overfetch_multiplier: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Over-fetch multiplier for the VectorCypher graph chunk channel when a "
        "residual metadata predicate must be applied as a Python post-filter (metadata is not "
        "pushable to Cypher). Capped at min(limit*multiplier, 200).",
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
    chronicle_enable_recall_reinforcement: bool = Field(
        default=False,
        description="When True, Chronicle updates chunk.last_accessed_at on recall "
        "and the decay function uses max(source_timestamp, last_accessed_at) "
        "as the effective event time. Frequently-recalled chunks stay 'fresh' "
        "even as their source_timestamp ages. See issue #855.",
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

    # Recall abstention (#1331). Both engines compute a passive
    # ``abstention_signals`` block whose ``should_abstain`` tells callers the
    # knowledge base doesn't cover the query. These knobs were hardcoded at the
    # call sites; they are now operator-tunable here. Defaults reproduce the
    # historical literals so flags + combined_score are unchanged.
    abstention_min_chunks: int = Field(
        default=1, ge=0, description="Chunk count below which the chunks_below_min abstention flag fires."
    )
    abstention_min_top_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Raw top-chunk cosine below which the top_score_low abstention flag fires (the topicality floor).",
    )
    abstention_combined_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Combined-score threshold at or above which should_abstain is True in abstention_mode='weighted'.",
    )
    abstention_weight_entities_empty: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of entities_empty in the weighted-mode combined_score.",
    )
    abstention_weight_chunks_below_min: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight of chunks_below_min in the weighted-mode combined_score.",
    )
    abstention_weight_top_score_low: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of top_score_low in the weighted-mode combined_score.",
    )
    abstention_mode: Literal["cosine_floor", "weighted"] = Field(
        default="cosine_floor",
        description=(
            "should_abstain derivation. 'cosine_floor' (default) abstains when "
            "top_score_low fires OR retrieval is genuinely empty (chunks_empty "
            "AND entities_empty) - the topicality floor decides on its own. "
            "'weighted' is the legacy escape hatch: combined_score >= "
            "abstention_combined_threshold. The four flags + combined_score are "
            "identical across modes; only should_abstain differs. See khora#1331."
        ),
    )

    @model_validator(mode="after")
    def _validate_abstention_weights(self) -> QuerySettings:
        # combined_score is a documented [0,1] contract; cap the simplex so a
        # mis-set weight triple can't push it above 1.0 in weighted mode.
        total = (
            self.abstention_weight_entities_empty
            + self.abstention_weight_chunks_below_min
            + self.abstention_weight_top_score_low
        )
        if total > 1.0:
            raise ValueError(
                f"abstention weights must sum to <= 1.0 (got {total:.3f}); a larger sum "
                "would push combined_score above the documented [0,1] range."
            )
        return self

    # Calibrated retrieval confidence (#1331), surfaced as
    # ``engine_info['confidence']``: 0.8*clip01(top_cosine/target_cosine)
    # + 0.2*clip01(top_score_gap/target_gap).
    abstention_confidence_target_cosine: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Cosine at which the confidence cosine-component saturates to 1.0.",
    )
    abstention_confidence_target_gap: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Top-two score gap at which the confidence gap-component saturates to 1.0.",
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

    # Semantic hooks configuration. Was ``Any = None`` historically due to
    # a circular-import worry; lazy import below resolves it without
    # introducing a forward ref in the type position. Issue #576 Phase 1
    # Item 4 — replaces the unused-field placeholder with the real config.
    hooks: SemanticHooksConfig = Field(
        default_factory=lambda: _SemanticHooksConfig(),
        description="Semantic hooks configuration",
    )

    # Telemetry
    telemetry_database_url: SecretStr | None = Field(
        default=None,
        description="PostgreSQL URL for telemetry database (set KHORA_TELEMETRY_DATABASE_URL to enable)",
    )
    telemetry_service_name: str = Field(
        default="khora",
        description="Service name tag for telemetry events",
    )

    # Dream-phase configuration (#649 / #650). Scaffolding only — orchestrator
    # bodies are stubbed and raise NotImplementedError until #661 lands.
    dream: DreamConfig = Field(
        default_factory=DreamConfig,
        description="Dream-phase configuration",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_alias_conflicts(cls, data: Any) -> Any:
        """Reject conflicting single-underscore and double-underscore env vars.

        Issue #789: each nested-config field accepts both ``KHORA_FOO_BAR``
        (canonical) and ``KHORA_FOO__BAR`` (legacy) forms. If the operator
        sets BOTH forms to **different** values we refuse to silently pick
        one — the wrong half of the .env file would be ignored without
        warning. Same value on both forms is fine (just redundant).

        Beyond conflict detection, this validator also **promotes** the
        canonical single-underscore env var into the legacy double-
        underscore slot inside ``data`` — pydantic-settings's
        ``env_nested_delimiter="__"`` only ingests the double-underscore
        form, so without this promotion the new spelling would silently
        be ignored. We mutate ``data`` (the dict pydantic-settings already
        built from env vars) by walking the canonical name and stitching
        the value into the nested-dict structure.
        """
        conflicts = _process_alias_env_pairs(data, _ENV_ALIAS_PAIRS, env_prefix="KHORA_")
        if conflicts:
            raise ValueError(
                "Conflicting Khora env vars: the same field is configured via "
                "both the new single-underscore form and the legacy "
                "double-underscore form with different values. Pick one — the "
                "single-underscore form is preferred; the double-underscore "
                "form is kept for backward compatibility.\n" + "\n".join(conflicts)
            )
        return data

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
        data = data or {}
        # model_validate bypasses pydantic-settings env sourcing, so the
        # documented KHORA_DATABASE_URL / KHORA_NEO4J_URL overrides (advertised
        # in the example YAML headers and the field docstrings above) would be
        # silently ignored. Merge them over the YAML dict so env wins over YAML
        # for these top-level shortcuts. Precedence: env > yaml > defaults.
        for env_name, key in (("KHORA_DATABASE_URL", "database_url"), ("KHORA_NEO4J_URL", "neo4j_url")):
            env_value = os.environ.get(env_name)
            if env_value:
                data[key] = env_value
        return cls.model_validate(data)

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
        # If it's a non-Neo4j backend (Memgraph, Neptune, SurrealDB, AGE, etc.), return as-is
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

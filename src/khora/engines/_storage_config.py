"""Shared storage config builder for all memory engines.

Centralises StorageConfig construction so that new backend support
(SurrealDB, pool_pre_ping, bulk_mode) is automatically available to
every engine without duplicating config wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from khora.storage.factory import StorageConfig

if TYPE_CHECKING:
    from khora.config import KhoraConfig


def build_storage_config(config: KhoraConfig, *, skip_graph: bool = False) -> StorageConfig:
    """Build a StorageConfig from a KhoraConfig, handling all backend types.

    This replaces the ~20-line inline config construction that was
    duplicated in Skeleton and VectorCypher engines.

    Args:
        config: The KhoraConfig to build from.
        skip_graph: When True, omit graph backend config even if Neo4j URL
            is set. Used by engines that don't need a graph database
            (skeleton, chronicle).

    Supports:
    - Traditional PostgreSQL + pgvector + Neo4j/Memgraph stack
    - SurrealDB unified backend (when ``config.storage.backend == "surrealdb"``)
    - pool_pre_ping for connection health checking
    """
    # --- SurrealDB unified backend ---
    if getattr(config.storage, "backend", "postgres") == "surrealdb":
        return StorageConfig(
            backend="surrealdb",
            surrealdb_config=config.storage.surrealdb,
            # HNSW index params for the unified backend's deferred vector
            # indexes (#1386). Dimension is the real embedder output
            # (llm.embedding_dimension) so every SurrealDB table (chunk /
            # entity / episode and the temporal_chunk store) sizes its index
            # from one source of truth; build params from StorageSettings.
            surrealdb_embedding_dimension=config.llm.embedding_dimension or 1536,
            surrealdb_hnsw_m=config.storage.hnsw_m,
            surrealdb_hnsw_ef_construction=config.storage.hnsw_ef_construction,
            # PostgreSQL fields are unused with SurrealDB but we still
            # populate them so callers that inspect the config don't crash.
            postgresql_url=None,
        )

    # --- SQLite + LanceDB embedded unified backend ---
    if getattr(config.storage, "backend", "postgres") == "sqlite_lance":
        return StorageConfig(
            backend="sqlite_lance",
            sqlite_lance_config=config.storage.sqlite_lance,
            postgresql_url=None,
        )

    # --- Traditional PostgreSQL + graph backend ---
    postgresql_url = config.get_postgresql_url()
    graph_config = None if skip_graph else config.get_graph_config()

    storage_kwargs: dict[str, Any] = {
        "postgresql_url": postgresql_url,
        "pgvector_url": postgresql_url,
        "postgresql_pool_size": config.storage.postgresql_pool_size,
        "postgresql_max_overflow": config.storage.postgresql_max_overflow,
        "postgresql_pool_pre_ping": getattr(config.storage, "postgresql_pool_pre_ping", False),
        "pgvector_embedding_dimension": config.storage.embedding_dimension,
        "pgvector_use_halfvec": config.storage.use_halfvec,
        "graph_config": graph_config,
        "vector_config": config.get_vector_config(),
        # Sentinels for #877: silence misleading "URL not configured" WARNINGs
        # on engines that intentionally opt out of graph / event store. The
        # legitimate operator-misconfiguration warning still fires when the
        # caller did NOT skip and the URL is genuinely missing.
        "graph_skipped": skip_graph,
        # No engine wires the legacy PostgreSQL event store through this
        # builder today (chronicle uses sqlite_lance; vectorcypher /
        # skeleton don't use an event store). Always opt out here.
        "event_store_skipped": True,
    }

    if graph_config is not None:
        storage_kwargs["neo4j_url"] = config.get_neo4j_url()
        storage_kwargs["neo4j_user"] = config.get_neo4j_user()
        storage_kwargs["neo4j_password"] = config.get_neo4j_password()
        storage_kwargs["neo4j_database"] = config.get_neo4j_database()

    return StorageConfig(**storage_kwargs)

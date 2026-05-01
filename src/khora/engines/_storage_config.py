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
    duplicated in Skeleton, GraphRAG, and VectorCypher engines.

    Args:
        config: The KhoraConfig to build from.
        skip_graph: When True, omit graph backend config even if Neo4j URL
            is set. Used by engines that don't need a graph database
            (skeleton, chronicle).

    Supports:
    - Traditional PostgreSQL + pgvector + Neo4j/Kuzu/Memgraph stack
      (Kuzu DEPRECATED in 0.9.0 — removal in 0.10.0)
    - SurrealDB unified backend (when ``config.storage.backend == "surrealdb"``)
    - pool_pre_ping for connection health checking
    """
    # --- SurrealDB unified backend ---
    if getattr(config.storage, "backend", "postgres") == "surrealdb":
        return StorageConfig(
            backend="surrealdb",
            surrealdb_config=config.storage.surrealdb,
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
    }

    if graph_config is not None:
        storage_kwargs["neo4j_url"] = config.get_neo4j_url()
        storage_kwargs["neo4j_user"] = config.get_neo4j_user()
        storage_kwargs["neo4j_password"] = config.get_neo4j_password()
        storage_kwargs["neo4j_database"] = config.get_neo4j_database()

    return StorageConfig(**storage_kwargs)

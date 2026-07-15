"""Storage layer for Khora.

Provides unified access to multiple storage backends:
- PostgreSQL: Relational data (documents, events, tenancy, ACLs)
- pgvector: Vector embeddings for semantic search
- Neo4j: Knowledge graph (entities, relationships)
- Kùzu: Embedded graph database (optional)
- Memgraph: In-memory graph database (optional)
- SurrealDB: Unified multi-model backend (graph + vector + relational)
"""

from __future__ import annotations

from .backends.base import (
    EventStoreProtocol,
    GraphBackendProtocol,
    PaginatedResult,
    RelationalBackendProtocol,
    VectorBackendProtocol,
)
from .backends.neo4j import Neo4jBackend
from .backends.pgvector import PgVectorBackend
from .backends.postgresql import PostgreSQLBackend
from .coordinator import NamespaceDeletionResult, StorageCoordinator, TransactionContext
from .event_store import PostgreSQLEventStore
from .expertise_store import ExpertiseStore
from .factory import StorageConfig, StorageFactory, create_storage_coordinator
from .optimize import (
    demote_namespace_hnsw,
    list_partial_hnsw_indexes,
    maybe_promote_namespace,
    optimize_neo4j,
    optimize_postgresql,
    optimize_storage,
    promote_namespace_hnsw,
)

__all__ = [
    # Protocols
    "PaginatedResult",
    "RelationalBackendProtocol",
    "VectorBackendProtocol",
    "GraphBackendProtocol",
    "EventStoreProtocol",
    # Implementations
    "PostgreSQLBackend",
    "PgVectorBackend",
    "Neo4jBackend",
    "PostgreSQLEventStore",
    # Coordinator
    "StorageCoordinator",
    "NamespaceDeletionResult",
    "TransactionContext",
    "StorageConfig",
    "StorageFactory",
    "create_storage_coordinator",
    # Optimization
    "optimize_storage",
    "optimize_postgresql",
    "optimize_neo4j",
    # Per-namespace partial HNSW (policy-gated, operator-driven)
    "maybe_promote_namespace",
    "promote_namespace_hnsw",
    "demote_namespace_hnsw",
    "list_partial_hnsw_indexes",
    # Expertise
    "ExpertiseStore",
]

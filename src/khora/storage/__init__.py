"""Storage layer for Khora Memory Lake.

Provides unified access to multiple storage backends:
- PostgreSQL: Relational data (documents, events, tenancy, ACLs)
- pgvector: Vector embeddings for semantic search
- Neo4j: Knowledge graph (entities, relationships)
- Kùzu: Embedded graph database (optional)
- Memgraph: In-memory graph database (optional)
- ArcadeDB: Multi-model graph + vector database (optional)
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
from .coordinator import StorageCoordinator, TransactionContext
from .event_store import PostgreSQLEventStore
from .expertise_store import ExpertiseStore
from .factory import StorageConfig, StorageFactory, create_storage_coordinator
from .optimize import optimize_neo4j, optimize_postgresql, optimize_storage

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
    "TransactionContext",
    "StorageConfig",
    "StorageFactory",
    "create_storage_coordinator",
    # Optimization
    "optimize_storage",
    "optimize_postgresql",
    "optimize_neo4j",
    # Expertise
    "ExpertiseStore",
]

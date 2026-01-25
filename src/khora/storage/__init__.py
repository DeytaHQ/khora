"""Storage layer for Khora Memory Lake.

Provides unified access to multiple storage backends:
- PostgreSQL: Relational data (documents, events, tenancy, ACLs)
- pgvector: Vector embeddings for semantic search
- Neo4j: Knowledge graph (entities, relationships)
"""

from __future__ import annotations

from .backends.base import (
    EventStoreProtocol,
    GraphBackendProtocol,
    RelationalBackendProtocol,
    VectorBackendProtocol,
)
from .backends.neo4j import Neo4jBackend
from .backends.pgvector import PgVectorBackend
from .backends.postgresql import PostgreSQLBackend
from .coordinator import StorageCoordinator
from .event_store import PostgreSQLEventStore
from .factory import StorageConfig, StorageFactory, create_storage_coordinator

__all__ = [
    # Protocols
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
    "StorageConfig",
    "StorageFactory",
    "create_storage_coordinator",
]

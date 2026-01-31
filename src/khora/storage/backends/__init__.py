"""Storage backend implementations for Khora Memory Lake."""

from __future__ import annotations

from .base import (
    EventStoreProtocol,
    GraphBackendProtocol,
    RelationalBackendProtocol,
    VectorBackendProtocol,
)
from .mixins import GraphBackendBase, VectorBackendBase
from .neo4j import Neo4jBackend
from .pgvector import PgVectorBackend
from .postgresql import PostgreSQLBackend

# Lazy imports for optional backends
try:
    from .kuzu import KuzuBackend
except ImportError:
    KuzuBackend = None  # type: ignore[assignment,misc]

try:
    from .memgraph import MemgraphBackend
except ImportError:
    MemgraphBackend = None  # type: ignore[assignment,misc]

try:
    from .arcadedb import ArcadeDBBackend
except ImportError:
    ArcadeDBBackend = None  # type: ignore[assignment,misc]

__all__ = [
    # Protocols
    "RelationalBackendProtocol",
    "VectorBackendProtocol",
    "GraphBackendProtocol",
    "EventStoreProtocol",
    # Mixins
    "GraphBackendBase",
    "VectorBackendBase",
    # Implementations
    "PostgreSQLBackend",
    "PgVectorBackend",
    "Neo4jBackend",
    "KuzuBackend",
    "MemgraphBackend",
    "ArcadeDBBackend",
]

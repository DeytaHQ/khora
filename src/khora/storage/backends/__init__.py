"""Storage backend implementations for Khora."""

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
    from .memgraph import MemgraphBackend
except ImportError:
    MemgraphBackend = None  # type: ignore[assignment,misc]

try:
    from .neptune import NeptuneBackend
except ImportError:
    NeptuneBackend = None  # type: ignore[assignment,misc]

try:
    from .age import AGEBackend
except ImportError:
    AGEBackend = None  # type: ignore[assignment,misc]

try:
    from .sqlite import SQLiteRelationalBackend, SQLiteVectorBackend
except ImportError:
    SQLiteRelationalBackend = None  # type: ignore[assignment,misc]
    SQLiteVectorBackend = None  # type: ignore[assignment,misc]

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
    "MemgraphBackend",
    "NeptuneBackend",
    "AGEBackend",
    "SQLiteRelationalBackend",
    "SQLiteVectorBackend",
]

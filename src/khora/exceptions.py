"""Khora exception hierarchy.

All domain-specific exceptions inherit from KhoraError, enabling callers
to catch broad or narrow exception types as needed.
"""

from __future__ import annotations


class KhoraError(Exception):
    """Base exception for all Khora errors."""


class StorageError(KhoraError):
    """Storage backend operation failed."""


class GraphError(StorageError):
    """Graph backend operation failed."""


class VectorError(StorageError):
    """Vector backend operation failed."""


class RelationalError(StorageError):
    """Relational backend operation failed."""


class QueryError(KhoraError):
    """Query execution failed."""


class EntityNotFoundError(QueryError):
    """Requested entity does not exist."""


class NamespaceNotFoundError(QueryError):
    """Requested namespace does not exist."""


class ExtractionError(KhoraError):
    """Entity/relationship extraction failed."""


class EmbeddingError(KhoraError):
    """Embedding generation failed."""


class ConfigurationError(KhoraError):
    """Invalid configuration."""


class MigrationError(KhoraError):
    """Database migration failed."""

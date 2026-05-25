"""Khora exception hierarchy.

All domain-specific exceptions inherit from KhoraError, enabling callers
to catch broad or narrow exception types as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from khora.query import SearchMode


__all__ = [
    "ConfigurationError",
    "EmbeddingError",
    "EngineCapabilityError",
    "EntityNotFoundError",
    "ExtractionError",
    "GraphError",
    "KhoraError",
    "KhoraIntegrationError",
    "MigrationError",
    "NamespaceNotFoundError",
    "QueryError",
    "RelationalError",
    "StorageError",
    "VectorError",
]


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


class KhoraIntegrationError(KhoraError):
    """Adapter (``khora.integrations.*``) configuration or runtime error.

    Raised by adapter factories and storage backends when caller input
    violates the adapter's invariants (e.g. an empty or placeholder
    ``user_id`` that would silently cross-share memory between users).
    """


class EngineCapabilityError(KhoraError):
    """Raised when a caller asks an engine for an unsupported ``SearchMode``.

    Each engine declares its honest mode contract via the
    ``supported_modes`` class attribute. Asking VectorCypher for KEYWORD,
    or Chronicle for GRAPH, fails fast with this error rather than
    silently degrading to HYBRID or returning empty results - both of
    which previously misled downstream agentic code into treating the
    response as authoritative.

    The exception carries the engine name, the requested mode, and the
    set of modes the engine does support so callers can either retry
    with a different engine, a different mode, or surface the constraint
    to the user.
    """

    def __init__(
        self,
        engine_name: str,
        mode: SearchMode,
        supported_modes: Iterable[SearchMode],
    ) -> None:
        self.engine_name = engine_name
        self.mode = mode
        # Frozen tuple for stable repr / equality regardless of caller's
        # container choice.
        self.supported_modes = tuple(sorted(supported_modes, key=lambda m: m.name))
        supported_names = sorted(m.name for m in supported_modes)
        super().__init__(
            f"Engine {engine_name!r} does not support SearchMode.{mode.name}. Supported modes: {supported_names}"
        )

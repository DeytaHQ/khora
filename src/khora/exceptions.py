"""Khora exception hierarchy.

All domain-specific exceptions inherit from KhoraError, enabling callers
to catch broad or narrow exception types as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    from khora.query import SearchMode


__all__ = [
    "ConfigurationError",
    "EmbeddingError",
    "EngineCapabilityError",
    "EntityNotFoundError",
    "ExtractionError",
    "GraphError",
    "GraphMirrorFailedAfterPGCommitError",
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


class GraphMirrorFailedAfterPGCommitError(StorageError):
    """Signals that a ``replace_document_extraction`` PG transaction
    committed (chunks + document status are durable) but the post-commit
    graph-mirror phase (retire / remap / upsert / create relationships)
    raised, leaving the graph backend in a partial-mirror state (#884).

    Carries the underlying exception via ``__cause__`` and exposes the
    document_id / namespace_id so the caller can record the divergence
    as a degradation on the user-facing result without losing the
    durable-write information PG already accepted.

    Out of scope for this PR: a reconciler that replays the missing
    graph work. Until that lands the next successful replace for the
    same ``external_id`` heals the row via the same MERGE / retire path.
    """

    def __init__(
        self,
        *,
        document_id: UUID,
        namespace_id: UUID,
        original: BaseException,
    ) -> None:
        self.document_id = document_id
        self.namespace_id = namespace_id
        # Surface the original exception class name so caller-side
        # observability (RememberResult.metadata) can record it without
        # importing the underlying backend's exception types.
        self.original_exception_type = type(original).__name__
        super().__init__(
            f"replace_document_extraction: PG committed for document "
            f"{document_id} in namespace {namespace_id} but the graph-mirror "
            f"phase raised {self.original_exception_type}: {original}"
        )


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

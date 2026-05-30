"""Dream-phase exception hierarchy (#661).

All inherit from :class:`khora.exceptions.KhoraError` so callers can
catch the broad family or narrow to a specific failure.

Stability: **public** — these are part of the orchestrator's surface
that callers will pattern-match on (e.g. ``except DreamDisabledError``
in a job runner). They live in their own module so an import of the
orchestrator does not pull in the engines / config tree.
"""

from __future__ import annotations

from uuid import UUID

from khora.exceptions import KhoraError


class DreamDisabledError(KhoraError):
    """Raised when ``Khora.dream()`` is called but ``DreamConfig.enabled`` is False.

    The dream phase is opt-in. Operators wire ``KHORA_DREAM_ENABLED=true``
    (or pass ``DreamConfig(enabled=True)``) to authorize runs.
    """


class DreamForbiddenOpError(KhoraError):
    """Raised when a plan contains a forbidden op (safety-floor breach).

    Forbidden ops include:

    - Document delete (Documents are tombstone-only)
    - Writes that would violate the ``(namespace_id, name, entity_type)``
      UNIQUE constraint on entities
    - Any write into a namespace marked read-only

    Enforced at plan time AND apply time (defense in depth).
    """


class DreamRunStuckError(KhoraError):
    """Raised when a previous run is in ``applying`` with a stale heartbeat.

    The caller must explicitly resolve by passing ``resume_from=run_id``
    or ``abandon=True`` on the next ``Khora.dream()`` call.
    """

    def __init__(self, run_id: UUID, heartbeat_age_seconds: float) -> None:
        self.run_id = run_id
        self.heartbeat_age_seconds = heartbeat_age_seconds
        super().__init__(
            f"Dream run {run_id} is in state='applying' with stale heartbeat "
            f"({heartbeat_age_seconds:.0f}s old). Pass resume_from={run_id} "
            f"to resume, or abandon=True to mark it crashed."
        )


class DreamApplyDisabled(KhoraError):
    """Raised when ``mode='apply'`` is requested but the kill-switch is set.

    The ``KHORA_DREAM_DISABLE_APPLY`` environment variable is the global
    five-second escape hatch operators flip to halt all destructive dream
    runs without restarting the process. It is read at orchestrator
    construction; a truthy value (anything other than ``""``, ``"0"``,
    ``"false"``) makes :meth:`DreamOrchestrator._apply_phase` raise this
    error before touching any database row.
    """


class DreamBackendUnsupported(KhoraError):
    """Raised when an apply handler requires a backend the active session lacks.

    The vectorcypher mutation handlers (``apply_vectorcypher_dedupe_entities``
    and siblings) bind raw ``uuid.UUID`` values into ``session.execute``;
    PostgreSQL's ``UUID`` adapter handles those natively, but SQLite (the
    ``sqlite_lance`` test stack) raises ``sqlite3.ProgrammingError`` on the
    first parameter bind. Rather than crashing with that opaque error, the
    orchestrator raises this exception, logs a warning, and marks the op
    as skipped so the run completes cleanly.

    See ``docs/dream-phase.md`` for the list of Postgres-only ops.
    """


__all__ = [
    "DreamApplyDisabled",
    "DreamBackendUnsupported",
    "DreamDisabledError",
    "DreamForbiddenOpError",
    "DreamRunStuckError",
]

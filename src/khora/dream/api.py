"""Public ``Khora.dream`` entry points (#661).

These functions are re-exposed as bound methods on :class:`khora.Khora`.
They validate inputs, resolve namespace IDs, construct a fresh
:class:`DreamOrchestrator`, and delegate execution to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from khora.dream.config import DreamConfig
from khora.dream.exceptions import DreamDisabledError
from khora.dream.orchestrator import DreamOrchestrator, request_cancel
from khora.dream.plan import DreamScope, OpKind
from khora.dream.result import DreamProgress, DreamResult, DreamRunInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from khora.khora import Khora


async def dream(
    kb: Khora,
    namespace: str | UUID,
    *,
    mode: str = "dry-run",
    scope: DreamScope | None = None,
    ops: Iterable[OpKind] | None = None,
    config: DreamConfig | None = None,
    on_progress: Callable[[DreamProgress], None] | None = None,
    resume_from: UUID | None = None,
) -> DreamResult:
    """Run a dream-phase pass over ``namespace``.

    Validates the master switch, resolves the namespace to a UUID,
    constructs a :class:`DreamOrchestrator`, and delegates the full
    state machine to it.

    Raises:
        DreamDisabledError: when ``config.enabled`` is ``False``. The
            master switch is opt-in by design — see #649.
        ValueError: when ``mode`` is not one of ``"dry-run"`` / ``"apply"``,
            or when ``namespace`` is a string that does not parse as UUID.
    """
    effective_config = config if config is not None else kb._config.dream
    if not effective_config.enabled:
        raise DreamDisabledError(
            "Khora.dream() requires DreamConfig.enabled=True. "
            "Set KHORA_DREAM_ENABLED=true or pass DreamConfig(enabled=True)."
        )

    if mode not in ("dry-run", "apply"):
        raise ValueError(f"mode must be 'dry-run' or 'apply', got {mode!r}")

    namespace_id = _coerce_namespace(namespace)
    orchestrator = DreamOrchestrator(kb, effective_config)
    return await orchestrator.run(
        namespace_id,
        mode=mode,
        scope=scope,
        ops=ops,
        on_progress=on_progress,
        resume_from=resume_from,
    )


async def dream_status(kb: Khora, run_id: UUID) -> dict[str, object]:
    """Return ``khora_dream_runs`` row metadata for ``run_id``.

    Returns an empty dict when the row does not exist or the backend
    isn't Postgres (embedded path mirrors checkpoints to a JSONL file —
    out of scope for this entry point).
    """
    orchestrator = DreamOrchestrator(kb, kb._config.dream)
    info = await orchestrator.status(run_id)
    if info is None:
        return {}
    return {
        "run_id": str(info.run_id),
        "namespace_id": str(info.namespace_id),
        "mode": info.mode,
        "started_at": info.started_at.isoformat(),
        "finished_at": info.finished_at.isoformat() if info.finished_at else None,
        "duration_ms": info.duration_ms,
    }


async def dream_history(
    kb: Khora,
    namespace: str | UUID,
    *,
    limit: int = 20,
) -> list[DreamRunInfo]:
    """Return recent dream-run records for ``namespace`` (newest first)."""
    namespace_id = _coerce_namespace(namespace)
    orchestrator = DreamOrchestrator(kb, kb._config.dream)
    return await orchestrator.history(namespace_id, limit=limit)


async def dream_cancel(kb: Khora, run_id: UUID) -> bool:
    """Signal a running dream run to halt between ops.

    Returns True if the run was found and the cancel flag was flipped,
    False if no in-process run matched ``run_id`` (already terminal or
    running in a different process).
    """
    del kb  # Cancel flags are in-process; the Khora handle is unused.
    return await request_cancel(run_id)


def _coerce_namespace(namespace: str | UUID) -> UUID:
    if isinstance(namespace, UUID):
        return namespace
    try:
        return UUID(namespace)
    except ValueError as exc:
        raise ValueError(f"namespace must be a UUID or UUID-shaped string, got {namespace!r}") from exc


__all__ = [
    "dream",
    "dream_cancel",
    "dream_history",
    "dream_status",
]

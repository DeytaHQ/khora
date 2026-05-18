"""Public ``Khora.dream`` entry points (#661).

These functions are re-exposed as bound methods on :class:`khora.Khora`.
They validate inputs, resolve namespace IDs, construct a fresh
:class:`DreamOrchestrator`, and delegate execution to it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
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


# ---------------------------------------------------------------------------
# Dream undo (#667 — Phase 4.1)
# ---------------------------------------------------------------------------


async def dream_undo(
    kb: Khora,
    op_id: UUID,
    *,
    base_dir: str | Path | None = None,
) -> bool:
    """Reverse a previously-applied dream op by ``op_id``.

    Reads the ``{run_id}.undo.json`` (schema ``dream-undo/1``) under
    ``base_dir`` (defaulting to the orchestrator's default file-sink
    location — ``<tmp>/khora-dream-reports``), locates the op record
    with ``op_id`` equal to the argument, and dispatches to the
    op-type-specific reverse handler inside one
    ``coordinator.transaction()`` block.

    Idempotent: re-calling on an already-undone op returns ``False``
    without touching any rows.

    Currently supported op types:

      * :data:`khora.dream.plan.OpKind.VECTORCYPHER_DEDUPE_ENTITIES` —
        restores the absorbed entity tombstone, re-points each
        previously-rewritten relationship at the absorbed side, and
        clears the bi-temporal invalidation on any post-rewrite
        self-loops the apply created.

    Other op types fall through with a ``False`` return — they don't
    have a reverse implementation yet (separate tickets).

    Args:
        op_id: The op_id captured on the :class:`DreamOp` at apply time
            (visible on the ``khora.dream.op`` span and in the ``ops[]``
            array of the run's ``undo.json``).
        base_dir: Override for the file-sink root. Defaults to the same
            path :class:`khora.dream.orchestrator.DreamOrchestrator`
            uses when no operator-supplied sink directory exists.

    Returns:
        ``True`` when at least one DB row was restored. ``False`` for an
        unknown op_id, an op whose op_type lacks a reverse handler, or
        an already-undone op (idempotent re-undo).
    """
    found = _locate_undo_op(op_id, base_dir=base_dir)
    if found is None:
        return False

    op_entry = found
    op_type = str(op_entry.get("op_type") or "")
    coordinator = kb.storage

    if op_type == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES):
        from khora.dream.engines.vectorcypher.dedupe_entities import (
            reverse_vectorcypher_dedupe_entities,
        )

        try:
            async with coordinator.transaction() as txn:
                return await reverse_vectorcypher_dedupe_entities(op_entry, session=txn.session)
        except RuntimeError as exc:
            # No SQL backend — embedded paths don't carry the
            # bi-temporal rows the reverse handler needs.
            if "No SQL backend" in str(exc):
                return False
            raise

    # Unknown op type — Phase 5+ apply handlers can wire their own
    # reverse paths here as they land.
    return False


def _default_file_sink_dir() -> Path:
    """Mirror :func:`khora.dream.orchestrator._default_file_sink_dir`."""
    import tempfile

    return Path(tempfile.gettempdir()) / "khora-dream-reports"


def _locate_undo_op(op_id: UUID, *, base_dir: str | Path | None) -> dict[str, Any] | None:
    """Walk the file-sink tree until we find the op with ``op_id``.

    The dream file sink lays out runs as
    ``{base_dir}/{namespace_id}/{date}/{run_id}.undo.json``. We scan
    every undo file (small N: one per dream run) and return the first
    op record whose ``op_id`` matches. Returns ``None`` when no match
    exists — including when the base_dir doesn't exist at all.
    """
    root = Path(base_dir) if base_dir is not None else _default_file_sink_dir()
    if not root.exists() or not root.is_dir():
        return None
    target = str(op_id)
    for path in root.rglob("*.undo.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") != "dream-undo/1":
            continue
        for op_entry in payload.get("ops") or []:
            if str(op_entry.get("op_id") or "") == target:
                return op_entry
    return None


__all__ = [
    "dream",
    "dream_cancel",
    "dream_history",
    "dream_status",
    "dream_undo",
]

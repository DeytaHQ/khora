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

    from khora.extraction.skills.base import ExpertiseConfig
    from khora.khora import Khora


async def dream(
    kb: Khora,
    namespace: str | UUID,
    *,
    mode: str = "dry-run",
    scope: DreamScope | None = None,
    ops: Iterable[OpKind] | None = None,
    config: DreamConfig | None = None,
    expertise: ExpertiseConfig | None = None,
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
        expertise=expertise,
        on_progress=on_progress,
        resume_from=resume_from,
    )


async def dream_status(kb: Khora, run_id: UUID) -> dict[str, object]:
    """Return ``khora_dream_runs`` row metadata for ``run_id``.

    Resolved through the stack's :class:`~khora.dream.runstore.DreamRunStore`
    (PostgreSQL, SQLite sidecar, or SurrealDB-relational since #1274), so
    it works on non-PG stacks. Returns an empty dict when the row does not
    exist or no run-state backend is reachable.
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

    Graph reverse mirror (#1275): after the PG reverse commits, the
    matching forward graph mirror (#1272/#1273) is reversed too — the
    absorbed entity is un-retired (its ``:EntityVersion`` /
    ``[:SUPERSEDES]`` snapshot deleted), self-loops are un-invalidated,
    and incident edges re-pointed back onto the absorbed entity — so undo
    restores PG and graph to identical pre-apply live sets rather than a
    half-revert. A reverse-mirror failure does NOT roll back the committed
    PG reverse; it records a structured degradation (ADR-001) and
    increments ``khora.dream.graph_unmirror.partial_failure`` so the
    divergence is observable. Backends without a native reverse (or with
    no graph configured) record a skip and leave the graph untouched.

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
                reverted = await reverse_vectorcypher_dedupe_entities(op_entry, session=txn.session)
        except RuntimeError as exc:
            # No SQL backend — embedded paths don't carry the
            # bi-temporal rows the reverse handler needs.
            if "No SQL backend" in str(exc):
                return False
            raise
        # PG reverse committed. Reverse the forward graph mirror too so undo
        # converges both stores (#1275). A failure here never rolls back the
        # committed PG reverse - it records a degradation (ADR-001).
        if reverted:
            await _unmirror_dream_op(kb, op_entry, op_type)
        return reverted

    # Unknown op type — Phase 5+ apply handlers can wire their own
    # reverse paths here as they land.
    return False


async def _unmirror_dream_op(kb: Khora, op_entry: dict[str, Any], op_type: str) -> None:
    """Reverse the forward graph mirror for one undone op (#1275).

    Runs AFTER the PG reverse committed (eventual consistency), reading the same
    ``before`` snapshot the PG reverse used as the source of truth. Translates
    the reverse onto the graph via the #1275 capability-gated restore verbs.
    Idempotent by-id.

    Gating (ADR-001):

      - No graph backend, or the backend cannot reverse this op kind -> records
        a skip (logged) and returns; the graph was never mirrored forward (or
        cannot be reversed), so there is nothing to diverge.
      - The reverse raises after the PG commit (or the namespace resolve fails)
        -> increments ``khora.dream.graph_unmirror.partial_failure``, logs a
        WARNING degradation, and returns. The committed PG reverse is NOT rolled
        back; the divergence is observable via the counter.
    """
    from khora.dream.graph_mirror import extract_unmirror_targets, unmirror_targets
    from khora.dream.orchestrator import _graph_backend, _supported_mirror_kinds

    graph = _graph_backend(kb.storage)
    if graph is None:
        return
    if op_type not in _supported_mirror_kinds(graph):
        # The forward mirror never touched the graph for this op kind, so the
        # PG-only reverse already converges. Nothing to undo on the graph.
        return

    before = op_entry.get("before") or {}
    targets = extract_unmirror_targets(op_type, before)
    if (
        not targets["restore_entity_ids"]
        and not targets["restore_relationship_ids"]
        and not targets["restore_endpoints"]
    ):
        return

    ns_raw = op_entry.get("_namespace_id")
    namespace_id = _coerce_namespace(str(ns_raw)) if ns_raw else None
    try:
        if namespace_id is None:
            raise ValueError("undo file carries no namespace_id; cannot resolve graph rows for the reverse mirror")
        row_namespace_id = await _resolve_namespace_for_unmirror(kb, namespace_id)
        await unmirror_targets(graph, targets, namespace_id=row_namespace_id)
    except Exception as exc:
        from loguru import logger

        from khora.dream.graph_mirror import GRAPH_UNMIRROR_PARTIAL_FAILURE_COUNTER

        GRAPH_UNMIRROR_PARTIAL_FAILURE_COUNTER.add(1)
        logger.warning(
            "dream graph un-mirror failed for op {op_type} (PG reverted, graph left diverged): {exc}",
            op_type=op_type,
            exc=exc,
            exc_info=True,
        )


async def _resolve_namespace_for_unmirror(kb: Khora, namespace_id: UUID) -> UUID:
    """Resolve a stable namespace id to the row id the graph rows carry (#1275).

    Mirrors :meth:`DreamOrchestrator._resolve_namespace_for_mirror`. A resolver
    error propagates so the caller queues a degradation rather than matching zero
    graph rows yet reporting success (silent divergence).
    """
    resolver = getattr(kb.storage, "resolve_namespace", None)
    if resolver is None:
        return namespace_id
    return await resolver(namespace_id)


def _default_file_sink_dir() -> Path:
    """Mirror :func:`khora.dream.orchestrator._default_file_sink_dir`."""
    import tempfile

    return Path(tempfile.gettempdir()) / "khora-dream-reports"


def _locate_undo_op(op_id: UUID, *, base_dir: str | Path | None) -> dict[str, Any] | None:
    """Walk the file-sink tree until we find the op with ``op_id``.

    The dream file sink lays out runs as
    ``{base_dir}/{namespace_id}/{date}/{run_id}.undo.json``. We scan
    every undo file (small N: one per dream run) and return the first
    op record whose ``op_id`` matches. The payload-level ``namespace_id``
    is injected onto the returned entry under ``_namespace_id`` so the
    graph reverse mirror (#1275) can resolve the graph rows' row id.
    Returns ``None`` when no match exists — including when the base_dir
    doesn't exist at all.
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
                op_entry["_namespace_id"] = payload.get("namespace_id")
                return op_entry
    return None


__all__ = [
    "dream",
    "dream_cancel",
    "dream_history",
    "dream_status",
    "dream_undo",
]

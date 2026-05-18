"""Plan + apply soft-deletion of low-confidence orphan edges (#671, Phase 5.2).

The vectorcypher edge-pruning dream op. **Off by default** —
``DreamConfig.prune_edges_enabled`` must be ``True`` and the operator
must opt-in to broader pruning by extending
``DreamConfig.prune_edges_target_predicates`` past the single default
``"ASSOCIATED_WITH"``.

**Predicate** (all three conjuncts):

1. ``confidence < threshold`` (default 0.4).
2. ``valid_to IS NULL`` (the edge is still live).
3. Every UUID in ``source_chunk_ids`` has no matching live ``chunks`` row.

When all three hold, the edge is a co-occurrence orphan: its
confidence is too low to trust, and the chunks that originally
established it are gone. The apply handler stamps
``valid_to = NOW()`` — a bi-temporal soft-delete (migration 033).

Reuses :func:`khora.storage.backends.neo4j.Neo4jBackend.retire_orphaned_relationships_batch`
is **not** appropriate here: that primitive is keyed on a single
replaced document and only mutates sole-sourced relationships. The
Phase 5.2 predicate is independent of document replacement. The
soft-delete still goes through the orchestrator-owned SQL session
(idempotent, dual-store-safe — the Neo4j ``valid_until`` mirror is
written by Phase 4's apply mode in v0.15).

Stability: **internal**.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.storage.coordinator import StorageCoordinator


_PHASE = "mutation"
_DECISION = "planned"
_DEFAULT_THRESHOLD = 0.4
_DEFAULT_PREDICATES: tuple[str, ...] = ("ASSOCIATED_WITH",)


async def plan_vectorcypher_prune_edges(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    target_predicates: tuple[str, ...] | list[str] = _DEFAULT_PREDICATES,
    confidence_threshold: float = _DEFAULT_THRESHOLD,
    mode: Literal["dry-run", "apply"] = "dry-run",
) -> tuple[DreamOp, ...]:
    """Plan soft-deletes of low-confidence orphan relationship edges.

    Args:
        namespace_id: The stable namespace identifier; resolved via the
            coordinator before any SELECT runs.
        coordinator: Storage coordinator — must expose a SQL session via
            ``coordinator.transaction()``.
        target_predicates: Whitelist of ``relationship_type`` values to
            consider. Default is ``("ASSOCIATED_WITH",)`` — operators
            must opt in to broader pruning by passing a wider tuple.
        confidence_threshold: Edges with ``confidence < threshold``
            (default 0.4) match the first conjunct of the predicate.
        mode: ``"dry-run"`` (default) plans without writes. ``"apply"``
            raises :class:`NotImplementedError` — apply is invoked
            through the orchestrator's per-op handler dispatch, not
            this planner.

    Returns:
        Tuple of :class:`DreamOp` instances — one per candidate
        relationship. Empty when no edge satisfies the predicate.

    Raises:
        NotImplementedError: When ``mode != "dry-run"``.

    Read-only: SELECTs against ``relationships`` and ``chunks`` only.
    No LLM calls.
    """
    if mode != "dry-run":
        raise NotImplementedError(
            "apply mode runs through the orchestrator's per-op handler "
            "dispatch (apply_vectorcypher_prune_edges), not the planner."
        )

    predicates = tuple(target_predicates)
    resolved_id = await coordinator.resolve_namespace(namespace_id)

    started_perf = perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.prune_edges",
        namespace_id=str(namespace_id),
        phase=_PHASE,
        threshold=float(confidence_threshold),
        target_predicates_count=len(predicates),
    ) as span:
        rows = await _collect_candidates(
            coordinator,
            resolved_namespace_id=resolved_id,
            target_predicates=predicates,
            confidence_threshold=confidence_threshold,
        )
        span.set_attribute("candidate_count", len(rows))

        ops: list[DreamOp] = []
        for row in rows:
            rel_id: UUID = row["id"]
            rel_type: str = row["relationship_type"]
            confidence: float = row["confidence"]

            started_at = datetime.now(UTC)
            duration_ms = (perf_counter() - started_perf) * 1000.0

            ops.append(
                DreamOp(
                    op_id=uuid4(),
                    phase=_PHASE,
                    op_type=OpKind.VECTORCYPHER_PRUNE_EDGES,
                    inputs=(
                        {
                            "relationship_id": str(rel_id),
                            "relationship_type": rel_type,
                            "confidence": confidence,
                            "threshold": float(confidence_threshold),
                        },
                    ),
                    outputs=(),
                    decision=_DECISION,
                    rationale=(
                        f"soft-delete {rel_type} edge: confidence={confidence:.3f} "
                        f"< {confidence_threshold:.3f}, valid_to IS NULL, all "
                        f"source chunks deleted."
                    ),
                    started_at=started_at,
                    duration_ms=duration_ms,
                    namespace_id=namespace_id,
                )
            )

    return tuple(ops)


# ---------------------------------------------------------------------------
# SQL: candidate collection
# ---------------------------------------------------------------------------


async def _collect_candidates(
    coordinator: StorageCoordinator,
    *,
    resolved_namespace_id: UUID,
    target_predicates: tuple[str, ...],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Return relationship rows satisfying the three-conjunct predicate.

    Postgres: a single statement uses ``unnest(source_chunk_ids)`` +
    ``LEFT JOIN chunks`` to assert every source chunk is dead.

    SQLite: the test fixture path — pulls the rows that satisfy the
    cheap conjuncts (``confidence`` + ``valid_to``) and filters the
    chunk-liveness conjunct in Python.
    """
    async with coordinator.transaction() as txn:
        session = txn.session
        dialect = session.bind.dialect.name if session.bind is not None else ""

        if dialect == "postgresql":
            return await _collect_postgres(
                session,
                resolved_namespace_id=resolved_namespace_id,
                target_predicates=target_predicates,
                confidence_threshold=confidence_threshold,
            )
        return await _collect_sqlite(
            session,
            resolved_namespace_id=resolved_namespace_id,
            target_predicates=target_predicates,
            confidence_threshold=confidence_threshold,
        )


async def _collect_postgres(
    session: Any,
    *,
    resolved_namespace_id: UUID,
    target_predicates: tuple[str, ...],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Postgres path — one SQL statement covers the full predicate."""
    sql = text(
        """
        SELECT r.id, r.relationship_type, r.confidence
        FROM relationships AS r
        WHERE r.namespace_id = :ns
          AND r.relationship_type = ANY(:rel_types)
          AND r.confidence < :threshold
          AND r.valid_to IS NULL
          AND (
              cardinality(r.source_chunk_ids) = 0
              OR NOT EXISTS (
                  SELECT 1
                  FROM unnest(r.source_chunk_ids) AS u(cid)
                  JOIN chunks AS c ON c.id = u.cid
                  WHERE c.namespace_id = r.namespace_id
              )
          )
        """
    )
    result = await session.execute(
        sql,
        {
            "ns": resolved_namespace_id,
            "rel_types": list(target_predicates),
            "threshold": float(confidence_threshold),
        },
    )
    out: list[dict[str, Any]] = []
    for row in result:
        out.append(
            {
                "id": _coerce_uuid(row.id),
                "relationship_type": str(row.relationship_type),
                "confidence": float(row.confidence),
            }
        )
    return out


async def _collect_sqlite(
    session: Any,
    *,
    resolved_namespace_id: UUID,
    target_predicates: tuple[str, ...],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """SQLite test path — Python filters the chunk-liveness conjunct."""
    # Pin the bind for the IN clause via expanding param.
    # `placeholders` is a comma-separated bind-name list, never user input.
    placeholders = ", ".join(f":t{i}" for i in range(len(target_predicates)))
    sql_text = f"SELECT id, relationship_type, confidence, source_chunk_ids FROM relationships WHERE namespace_id = :ns AND relationship_type IN ({placeholders or ':empty'}) AND confidence < :threshold AND valid_to IS NULL"  # noqa: S608
    sql = text(sql_text)
    params: dict[str, Any] = {
        "ns": resolved_namespace_id.hex,
        "threshold": float(confidence_threshold),
        "rel_types": list(target_predicates),  # kept for test introspection
    }
    for i, t in enumerate(target_predicates):
        params[f"t{i}"] = t
    if not target_predicates:
        params["empty"] = ""

    result = await session.execute(sql, params)

    # Build the live-chunk set once for the namespace; the relationship
    # set is small relative to chunks here, but the helper stays cheap
    # because SQLite is only used for tests.
    chunk_rows = await session.execute(
        text("SELECT id FROM chunks WHERE namespace_id = :ns"),
        {"ns": resolved_namespace_id.hex},
    )
    live_chunk_ids: set[UUID] = set()
    for row in chunk_rows:
        cid = row.id if hasattr(row, "id") else row[0]
        coerced = _coerce_uuid(cid)
        if coerced is not None:
            live_chunk_ids.add(coerced)

    out: list[dict[str, Any]] = []
    for row in result:
        rel_id = _coerce_uuid(row.id if hasattr(row, "id") else row[0])
        if rel_id is None:
            continue
        chunk_ids_raw = row.source_chunk_ids if hasattr(row, "source_chunk_ids") else row[3]
        chunk_ids = _parse_uuid_list(chunk_ids_raw)
        # Predicate: every source chunk is dead (or the array is empty).
        if chunk_ids and any(cid in live_chunk_ids for cid in chunk_ids):
            continue
        out.append(
            {
                "id": rel_id,
                "relationship_type": str(row.relationship_type),
                "confidence": float(row.confidence),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Apply handler
# ---------------------------------------------------------------------------


async def apply_vectorcypher_prune_edges(
    op: DreamOp,
    *,
    coordinator: StorageCoordinator,
    session: AsyncSession,
) -> UndoRecord:
    """Stamp ``valid_to = NOW()`` on one planned relationship.

    Caller (the orchestrator) owns the transaction. Idempotent on
    replay: if the relationship is already pruned (``valid_to IS NOT
    NULL``) or has vanished (FK-cascade from a deleted entity), no
    UPDATE fires and the returned :class:`UndoRecord` carries
    ``before={"noop": True}``.

    Args:
        op: The planned op. ``inputs[0]`` carries the
            ``relationship_id``.
        coordinator: Storage coordinator — unused (session is the only
            write surface). Kept for handler signature uniformity.
        session: Orchestrator-owned async session.

    Returns:
        :class:`UndoRecord` with ``before["relationships"]`` carrying
        the pre-state ``previous_confidence`` and ``previous_valid_to``
        per pruned relationship. The undoer reverses the soft-delete by
        writing ``UPDATE relationships SET valid_to = previous_valid_to``.
    """
    del coordinator  # unused — session is the only write surface
    inputs = op.inputs[0]
    rel_id = UUID(str(inputs["relationship_id"]))

    pre_state = await _read_pre_state(session, rel_id)
    if pre_state is None:
        # Relationship has vanished (FK cascade) — nothing to do.
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"noop": True},
            applied_at=datetime.now(UTC),
        )

    previous_confidence, previous_valid_to = pre_state
    if previous_valid_to is not None:
        # Already pruned — idempotent replay.
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"noop": True},
            applied_at=datetime.now(UTC),
        )

    now = datetime.now(UTC)
    await session.execute(
        text("UPDATE relationships SET valid_to = :ts, updated_at = :ts WHERE id = :rid AND valid_to IS NULL"),
        {"ts": now, "rid": rel_id},
    )

    return UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "relationships": [
                {
                    "relationship_id": str(rel_id),
                    "previous_confidence": previous_confidence,
                    "previous_valid_to": previous_valid_to.isoformat() if previous_valid_to else None,
                }
            ]
        },
        applied_at=now,
    )


async def _read_pre_state(
    session: AsyncSession,
    rel_id: UUID,
) -> tuple[float, datetime | None] | None:
    """Return ``(confidence, valid_to)`` for the relationship, or None if missing."""
    result = await session.execute(
        text("SELECT confidence, valid_to FROM relationships WHERE id = :rid"),
        {"rid": rel_id},
    )
    row = result.first()
    if row is None:
        return None
    confidence = float(row.confidence) if row.confidence is not None else 1.0
    valid_to = row.valid_to
    return confidence, valid_to


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _parse_uuid_list(value: Any) -> list[UUID]:
    """Parse the source_chunk_ids column for the SQLite/test path."""
    if value is None:
        return []
    if isinstance(value, list):
        items: list[Any] = value
    else:
        # SQLite stores arrays as JSON text.
        import json

        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        items = parsed
    out: list[UUID] = []
    for item in items:
        coerced = _coerce_uuid(item)
        if coerced is not None:
            out.append(coerced)
    return out

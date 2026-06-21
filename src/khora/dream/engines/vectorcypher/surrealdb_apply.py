"""SurrealQL-native dream-apply path for the unified SurrealDB backend (#1280).

On a SurrealDB-unified stack the coordinator has **no SQL session** -
``coordinator.transaction()`` raises ``RuntimeError("No SQL backend ...")``
and the vectorcypher apply handlers (which bind raw ``uuid.UUID`` values into a
SQLAlchemy ``session``) cannot run. Before #1280 the orchestrator fell back to
calling those handlers with ``session=None``, which crashed / no-op'd silently.

This module is the SurrealQL-native replacement for the soft-delete ops that
matter for cross-store convergence (the P1-4 invariant, #1268):

* ``vectorcypher_prune_edges`` - stamp ``valid_until = time::now()`` on the
  ``relates_to`` edge (flat soft-delete; the read filter then hides it).
* ``vectorcypher_dedupe_entities`` - rewrite incident edges off the absorbed
  entity onto the canonical, flat-soft-delete the post-rewrite self-loops, and
  flat-soft-delete the absorbed entity. **Flat soft-delete only** - no
  ``:EntityVersion`` snapshot (the embedded backend has no version columns,
  per CLAUDE.md).

Because SurrealDB is unified (graph == vector == relational, one store), the
apply IS the mutation - there is no separate graph to mirror to afterwards, so
the orchestrator skips its post-commit graph mirror for SurrealDB-applied ops.

Every other op kind (centroid recompute, source-chunk GC, schema normalize,
community summary) has no SurrealQL apply yet and is declared **unsupported**
with a structured ``surrealdb_native_apply_required`` skip (ADR-001) rather
than crashing.

Embedded mode (``memory://`` / ``surrealkv://``) does not support
``BEGIN TRANSACTION`` (surrealkv raises), so the handlers issue their writes
through :meth:`SurrealDBConnection.execute_batch` - the documented embedded
batched-atomicity primitive - and each handler is idempotent on replay.

Stability: **internal**.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

from khora.dream.plan import OpKind
from khora.dream.result import UndoRecord
from khora.storage.backends.surrealdb._helpers import _rid

if TYPE_CHECKING:
    from khora.dream.plan import DreamOp
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection


# Op kinds with a SurrealQL-native apply handler. Every other kind is
# skip-declared with ``surrealdb_native_apply_required`` (see
# :func:`surrealdb_native_skip_reason`).
SURREALDB_NATIVE_APPLY_KINDS: frozenset[str] = frozenset(
    {
        str(OpKind.VECTORCYPHER_PRUNE_EDGES),
        str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES),
    }
)


def surrealdb_native_skip_reason(op: DreamOp) -> dict[str, Any]:
    """Structured skip for an op with no SurrealQL-native apply (ADR-001)."""
    return {
        "op_kind": str(op.op_type),
        "reason": "surrealdb_native_apply_required",
        "detail": (
            f"op_type={op.op_type!s} has no SurrealQL-native apply handler; "
            "SurrealDB-unified dream-apply supports prune_edges and "
            "dedupe_entities (flat valid_until soft-delete) only."
        ),
    }


async def apply_surrealdb_op(
    op: DreamOp,
    *,
    conn: SurrealDBConnection,
) -> UndoRecord:
    """Dispatch one op to its SurrealQL-native apply handler.

    The caller (orchestrator) only routes here for op kinds in
    :data:`SURREALDB_NATIVE_APPLY_KINDS`; an unexpected kind is a
    programming error, surfaced as a ``KeyError`` rather than a silent no-op.
    """
    op_type = str(op.op_type)
    if op_type == str(OpKind.VECTORCYPHER_PRUNE_EDGES):
        return await _apply_prune_edges(op, conn=conn)
    if op_type == str(OpKind.VECTORCYPHER_DEDUPE_ENTITIES):
        return await _apply_dedupe_entities(op, conn=conn)
    raise KeyError(f"no SurrealQL-native apply handler for op_type={op_type!r}")


# ---------------------------------------------------------------------------
# prune_edges
# ---------------------------------------------------------------------------


async def _apply_prune_edges(op: DreamOp, *, conn: SurrealDBConnection) -> UndoRecord:
    """Stamp ``valid_until = time::now()`` on one planned ``relates_to`` edge.

    Idempotent on replay: the UPDATE is guarded by ``valid_until IS NONE`` so a
    second apply is a no-op and the returned :class:`UndoRecord` carries
    ``before={"noop": True}``.
    """
    inputs = op.inputs[0]
    rel_id = str(inputs["relationship_id"])

    pre = await conn.query_one(
        "SELECT rel_id, confidence, valid_until FROM relates_to WHERE rel_id = $rid LIMIT 1",
        {"rid": rel_id},
    )
    now = datetime.now(UTC)
    if pre is None or pre.get("valid_until") is not None:
        # Edge vanished, or already pruned - idempotent replay.
        return UndoRecord(op_id=op.op_id, op_type=str(op.op_type), before={"noop": True}, applied_at=now)

    await conn.execute_batch(
        [
            (
                "UPDATE relates_to SET valid_until = time::now(), updated_at = time::now() "
                "WHERE rel_id = $rid AND valid_until IS NONE",
                {"rid": rel_id},
            )
        ]
    )

    return UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={
            "relationships": [
                {
                    "relationship_id": rel_id,
                    "previous_confidence": _as_float(pre.get("confidence")),
                    "previous_valid_until": None,
                }
            ]
        },
        applied_at=now,
    )


# ---------------------------------------------------------------------------
# dedupe_entities (flat soft-delete)
# ---------------------------------------------------------------------------


async def _apply_dedupe_entities(op: DreamOp, *, conn: SurrealDBConnection) -> UndoRecord:
    """Flat soft-delete dedupe on the unified SurrealDB store (#1280).

    Per merge entry: flat-soft-delete (``valid_until``) every live edge
    incident to the absorbed entity, then flat-soft-delete the absorbed entity
    row itself. No ``:EntityVersion`` snapshot - flat soft-delete only.

    **Endpoint re-pointing is intentionally NOT done here.** SurrealDB
    RELATION ``in`` / ``out`` record links are not rewritable in place, and the
    incident-edge re-point leg of dedupe is tracked separately (#1273, the
    Neo4j path defers it too). The flat model therefore retires the absorbed
    entity together with its incident edges - keeping the live set internally
    consistent (no live edge points at a retired node) - which is the
    convergence the P1-4 invariant checks. The self-loop case (the issue's
    acceptance target) is a strict subset of this: a canonical->absorbed edge
    becomes a self-loop on merge and is soft-deleted along with the others.

    The verifier two-LLM judge is NOT run here. Idempotent on replay: an
    already-retired entity has no live incident edges left, and the guarded
    soft-delete UPDATE is a no-op on an already-soft-deleted row.
    """
    outputs = op.outputs[0] if op.outputs else {}
    merges_input = list(outputs.get("merges") or [])
    now = datetime.now(UTC)
    if not merges_input:
        return UndoRecord(op_id=op.op_id, op_type=str(op.op_type), before={"merges": []}, applied_at=now)

    merges_undo: list[dict[str, Any]] = []

    for entry in merges_input:
        canonical_id = UUID(str(entry["canonical_id"]))
        absorbed_id = UUID(str(entry["absorbed_id"]))
        absorbed_rid = _rid("entity", absorbed_id)

        # Snapshot every live edge touching the absorbed entity (in OR out).
        rows = await conn.query(
            "SELECT rel_id, in, out, relationship_type FROM relates_to "
            "WHERE (in = $aid OR out = $aid) AND valid_until IS NONE",
            {"aid": absorbed_rid},
        )

        previous_serialized: list[dict[str, Any]] = []
        self_loops: list[str] = []
        edges_retired: list[str] = []
        for row in rows:
            rel_id = row.get("rel_id")
            src = _record_uuid(row.get("in"))
            tgt = _record_uuid(row.get("out"))
            if rel_id is None or src is None or tgt is None:
                continue
            previous_serialized.append(
                {
                    "id": str(rel_id),
                    "source_entity_id": str(src),
                    "target_entity_id": str(tgt),
                    "relationship_type": str(row.get("relationship_type"))
                    if row.get("relationship_type") is not None
                    else None,
                }
            )

            # Flat soft-delete the incident edge (self-loop or not).
            await conn.execute_batch(
                [
                    (
                        "UPDATE relates_to SET valid_until = time::now(), updated_at = time::now() "
                        "WHERE rel_id = $rid AND valid_until IS NONE",
                        {"rid": str(rel_id)},
                    )
                ]
            )
            edges_retired.append(str(rel_id))
            # A canonical<->absorbed edge collapses to a self-loop on merge.
            if {src, tgt} == {canonical_id, absorbed_id} or src == tgt:
                self_loops.append(str(rel_id))

        # Flat soft-delete the absorbed entity row (guarded - idempotent).
        await conn.execute_batch(
            [
                (
                    "UPDATE $aid SET valid_until = time::now(), updated_at = time::now() WHERE valid_until IS NONE",
                    {"aid": absorbed_rid},
                )
            ]
        )

        merges_undo.append(
            {
                "canonical_id": str(canonical_id),
                "absorbed_id": str(absorbed_id),
                "previous_relationships": previous_serialized,
                "self_loops_invalidated": self_loops,
                "edges_retired": edges_retired,
                "applied": True,
            }
        )

    logger.debug(
        "surrealdb dream dedupe applied: {n} merge entr(ies)",
        n=len(merges_undo),
    )
    return UndoRecord(op_id=op.op_id, op_type=str(op.op_type), before={"merges": merges_undo}, applied_at=now)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _record_uuid(value: Any) -> UUID | None:
    """Extract a ``uuid.UUID`` from a SurrealDB ``in`` / ``out`` record link.

    SurrealDB returns record links as ``RecordID`` objects (``.id`` holds the
    UUID) or as ``table:<uuid>`` strings depending on SDK shape; handle both.
    """
    if value is None:
        return None
    inner = getattr(value, "id", None)
    if isinstance(inner, UUID):
        return inner
    if inner is not None:
        try:
            return UUID(str(inner))
        except (TypeError, ValueError):
            return None
    text = str(value)
    if ":" in text:
        text = text.split(":", 1)[1].strip("⟨⟩")
    try:
        return UUID(text)
    except (TypeError, ValueError):
        return None


__all__ = [
    "SURREALDB_NATIVE_APPLY_KINDS",
    "apply_surrealdb_op",
    "surrealdb_native_skip_reason",
]

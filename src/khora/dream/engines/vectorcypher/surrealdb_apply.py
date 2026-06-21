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
* ``vectorcypher_dedupe_entities`` - re-point each incident edge off the
  absorbed entity onto the canonical (soft-delete the old edge + create an
  equivalent live edge to/from the canonical, #1303), flat-soft-delete the
  post-merge self-loops (canonical<->absorbed collapses to a self-loop and is
  retired, not re-pointed), and flat-soft-delete the absorbed entity. **Flat
  soft-delete only** - no ``:EntityVersion`` snapshot (the embedded backend has
  no version columns, per CLAUDE.md).

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
from uuid import NAMESPACE_URL, UUID, uuid5

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
    """Dedupe with incident-edge re-pointing on the unified SurrealDB store (#1303).

    Per merge entry, for every live edge incident to the absorbed entity:

    * a **self-loop** (canonical<->absorbed collapses to a self-loop on merge,
      or an existing absorbed self-loop) is flat-soft-deleted only - it is NOT
      re-pointed (a self-loop on the canonical adds no information).
    * any other edge is **re-pointed** onto the canonical: the old edge is
      flat-soft-deleted (``valid_until = time::now()``) and an equivalent live
      edge is created to/from the canonical with the same type + properties.
      SurrealDB RELATION ``in`` / ``out`` are not rewritable in place, so
      re-pointing = soft-delete-old + create-new (the SurrealDB analog of the
      Neo4j #1273 endpoint rewrite). The new edge gets a deterministic
      ``rel_id`` (``uuid5(old_rel_id, canonical)``) so a replay produces the
      same id rather than a duplicate.

    Finally the absorbed entity row is flat-soft-deleted. No ``:EntityVersion``
    snapshot - flat soft-delete only.

    Idempotent on replay: the live-incident-edge SELECT is guarded by
    ``valid_until IS NONE``, so once the old edges are soft-deleted the absorbed
    entity has no live incident edges left and a second apply re-points nothing.
    The re-pointed edges point at the canonical (never the absorbed entity), so
    they never resurface in that SELECT. The verifier two-LLM judge is NOT run
    here.
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
        canonical_rid = _rid("entity", canonical_id)

        # Snapshot every live edge touching the absorbed entity (in OR out),
        # including the columns we must carry onto a re-pointed edge.
        rows = await conn.query(
            "SELECT rel_id, in, out, namespace_id, relationship_type, description, properties, "
            "source_document_ids, source_chunk_ids, valid_from, confidence, weight, metadata_ "
            "FROM relates_to WHERE (in = $aid OR out = $aid) AND valid_until IS NONE",
            {"aid": absorbed_rid},
        )

        previous_serialized: list[dict[str, Any]] = []
        self_loops: list[str] = []
        edges_retired: list[str] = []
        edges_repointed: list[dict[str, Any]] = []
        # Collect every write for this merge entry into ONE batch so the whole
        # merge is a single round-trip. ``execute_batch`` rejects parameter-name
        # collisions, so each statement gets indexed bind names.
        batch_stmts: list[tuple[str, dict[str, Any]]] = []
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
            # ``param`` is an internally-generated bind name, never user input;
            # the rel_id value still binds as a parameter. S608 noqa per the
            # repo SurrealQL convention.
            n = len(batch_stmts)
            param = f"rid_{n}"
            batch_stmts.append(
                (
                    "UPDATE relates_to SET valid_until = time::now(), updated_at = time::now() "  # noqa: S608
                    f"WHERE rel_id = ${param} AND valid_until IS NONE",
                    {param: str(rel_id)},
                )
            )
            edges_retired.append(str(rel_id))

            # Map the absorbed endpoint onto the canonical.
            new_src = canonical_id if src == absorbed_id else src
            new_tgt = canonical_id if tgt == absorbed_id else tgt
            # A canonical<->absorbed edge collapses to a self-loop on merge, as
            # does an absorbed self-loop. Soft-delete only - never re-pointed.
            if new_src == new_tgt:
                self_loops.append(str(rel_id))
                continue

            # Re-point: create an equivalent live edge to/from the canonical
            # with the same type + properties. Deterministic rel_id keeps replay
            # a no-op (a second apply finds the old edge already soft-deleted,
            # so it re-points nothing).
            new_rel_id = uuid5(NAMESPACE_URL, f"{rel_id}->{canonical_id}")
            src_rid = canonical_rid if src == absorbed_id else _rid("entity", src)
            tgt_rid = canonical_rid if tgt == absorbed_id else _rid("entity", tgt)
            batch_stmts.append(
                (
                    f"RELATE $rp_src_{n}->relates_to->$rp_tgt_{n} SET "  # noqa: S608
                    f"rel_id = $rp_id_{n}, "
                    f"namespace_id = $rp_ns_{n}, "
                    f"relationship_type = $rp_type_{n}, "
                    f"description = $rp_desc_{n}, "
                    f"properties = $rp_props_{n}, "
                    f"source_document_ids = $rp_docs_{n}, "
                    f"source_chunk_ids = $rp_chunks_{n}, "
                    f"valid_from = $rp_from_{n}, "
                    "valid_until = NONE, "
                    f"confidence = $rp_conf_{n}, "
                    f"weight = $rp_weight_{n}, "
                    f"metadata_ = $rp_meta_{n}, "
                    "created_at = time::now(), "
                    "updated_at = time::now()",
                    {
                        f"rp_src_{n}": src_rid,
                        f"rp_tgt_{n}": tgt_rid,
                        f"rp_id_{n}": str(new_rel_id),
                        f"rp_ns_{n}": row.get("namespace_id"),
                        f"rp_type_{n}": row.get("relationship_type"),
                        f"rp_desc_{n}": row.get("description"),
                        f"rp_props_{n}": row.get("properties") or {},
                        f"rp_docs_{n}": row.get("source_document_ids") or [],
                        f"rp_chunks_{n}": row.get("source_chunk_ids") or [],
                        f"rp_from_{n}": row.get("valid_from"),
                        f"rp_conf_{n}": _as_float(row.get("confidence")),
                        f"rp_weight_{n}": _as_float(row.get("weight")),
                        f"rp_meta_{n}": row.get("metadata_") or {},
                    },
                )
            )
            edges_repointed.append(
                {
                    "old_rel_id": str(rel_id),
                    "new_rel_id": str(new_rel_id),
                    "source_entity_id": str(new_src),
                    "target_entity_id": str(new_tgt),
                }
            )

        # Flat soft-delete the absorbed entity row (guarded - idempotent).
        batch_stmts.append(
            (
                "UPDATE $aid SET valid_until = time::now(), updated_at = time::now() WHERE valid_until IS NONE",
                {"aid": absorbed_rid},
            )
        )
        await conn.execute_batch(batch_stmts)

        merges_undo.append(
            {
                "canonical_id": str(canonical_id),
                "absorbed_id": str(absorbed_id),
                "previous_relationships": previous_serialized,
                "self_loops_invalidated": self_loops,
                "edges_retired": edges_retired,
                "edges_repointed": edges_repointed,
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

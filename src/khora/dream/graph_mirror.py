"""Neo4j dream tombstone-mirror: the committed-PG -> graph convergence step (#1272).

Dream apply stamps soft-delete columns on PostgreSQL only (``prune_edges``
writes ``relationships.valid_to``; ``dedupe_entities`` writes
``relationships.invalidated_at`` on post-rewrite self-loops and
``entities.valid_until`` on the absorbed node). The graph backend (Neo4j /
Memgraph / Neptune / AGE) reflects only ``valid_until`` on read, so without a
mirror the pruned / merged shapes stay silently live in graph recall while the
PG read filter (#888 / #970) hides them - cross-store divergence.

This module is the post-commit convergence step. It reads the just-committed
``UndoRecord`` (the source of truth for what PG accepted) and translates each
soft-delete into the #1271 capability-gated graph verbs, folding the three PG
soft-delete columns onto the single graph ``valid_until`` the read path honors:

  - ``prune_edges``     -> ``soft_invalidate_relationships_batch`` (PG ``valid_to``)
  - ``dedupe_entities`` -> ``soft_retire_entities_batch`` (PG ``entities.valid_until``)
                           + ``soft_invalidate_relationships_batch`` for the
                             self-loops (PG ``relationships.invalidated_at``)

The endpoint-rewrite leg of dedupe (re-pointing incident edges, folding
duplicate-key edges) is the #1273 entity-merge mirror and is intentionally NOT
mirrored here - this PR ships the soft-delete shapes only.

The step runs OUTSIDE the apply transaction (the checkpoint already advanced
inside the PG commit), so it is eventually-consistent and idempotent by-id
(MERGE/SET on ``valid_until IS NULL``). A failure after the PG commit increments
``khora.dream.graph_mirror.partial_failure`` and is re-attempted by the
reconciler via the run store's ``graph_mirror_pending`` slot.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.telemetry.metrics import metric_counter

if TYPE_CHECKING:
    from khora.dream.plan import DreamOp
    from khora.dream.result import UndoRecord
    from khora.storage.backends.base import GraphBackendProtocol

# Op kinds this mirror translates. Subset of the graph backend's
# ``supports_dream_mirror()`` probe - the entity-merge endpoint rewrite
# (VECTORCYPHER_DEDUPE_ENTITIES also advertises the rewrite verb) is #1273.
_PRUNE_EDGES = "vectorcypher_prune_edges"
_DEDUPE_ENTITIES = "vectorcypher_dedupe_entities"

# Op kinds whose soft-deletes this module knows how to mirror. Anything outside
# this set (centroid recompute, normalize_schema relabel, source-chunk GC) is
# recorded as a structured skip rather than mirrored - see ``mirror_skip_reason``.
MIRRORABLE_OP_KINDS: frozenset[str] = frozenset({_PRUNE_EDGES, _DEDUPE_ENTITIES})


# Emitted once per op whose PG commit succeeded but whose graph mirror raised.
# No labels - the namespace_id cardinality rule forbids a per-tenant label.
GRAPH_MIRROR_PARTIAL_FAILURE_COUNTER = metric_counter(
    "khora.dream.graph_mirror.partial_failure",
    description=(
        "Dream apply committed on PostgreSQL but the post-commit Neo4j "
        "tombstone-mirror raised. PG is the source of truth; the op is queued "
        "in graph_mirror_pending and re-attempted by the reconciler. NO "
        "namespace_id label - cardinality rule."
    ),
)


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def extract_mirror_targets(op_type: str, undo: UndoRecord) -> dict[str, list[UUID]]:
    """Translate a just-committed ``UndoRecord`` into graph mirror targets.

    Returns a dict with two keys:

      - ``retire_entity_ids``: entity ids to soft-retire (absorbed nodes).
      - ``invalidate_relationship_ids``: relationship ids to soft-invalidate
        (pruned edges + dedupe self-loops).

    The ``UndoRecord.before`` snapshot is the source of truth: it records
    exactly the rows the PG handler committed. A no-op apply (already-pruned,
    vanished, verifier-rejected merge) yields empty lists, so the mirror is a
    no-op too. Endpoint rewrites in ``before["merges"][*]["previous_relationships"]``
    are deliberately ignored here - they are the #1273 entity-merge mirror.
    """
    before = undo.before or {}
    retire_entity_ids: list[UUID] = []
    invalidate_relationship_ids: list[UUID] = []

    if op_type == _PRUNE_EDGES:
        # prune_edges: before == {"noop": True} or {"relationships": [{...}]}.
        for row in before.get("relationships") or []:
            rel_id = _coerce_uuid(row.get("relationship_id"))
            if rel_id is not None:
                invalidate_relationship_ids.append(rel_id)
    elif op_type == _DEDUPE_ENTITIES:
        # dedupe_entities: before == {"merges": [{...}]}. Each applied merge
        # soft-deletes the absorbed entity (entities.valid_until) and may
        # invalidate self-loop relationships (relationships.invalidated_at).
        for merge in before.get("merges") or []:
            if not merge.get("applied"):
                continue
            absorbed_id = _coerce_uuid(merge.get("absorbed_id"))
            if absorbed_id is not None:
                retire_entity_ids.append(absorbed_id)
            for rid in merge.get("self_loops_invalidated") or []:
                rel_id = _coerce_uuid(rid)
                if rel_id is not None:
                    invalidate_relationship_ids.append(rel_id)

    return {
        "retire_entity_ids": retire_entity_ids,
        "invalidate_relationship_ids": invalidate_relationship_ids,
    }


def mirror_payload(op: DreamOp, undo: UndoRecord) -> dict[str, Any]:
    """Serialize the mirror targets for the ``graph_mirror_pending`` slot.

    The reconciler re-attempts from this JSON payload rather than re-reading the
    undo file - it carries the exact id lists plus the stamp time, so a replay
    after a crash mirrors precisely what PG committed.
    """
    targets = extract_mirror_targets(str(op.op_type), undo)
    return {
        "retire_entity_ids": [str(eid) for eid in targets["retire_entity_ids"]],
        "invalidate_relationship_ids": [str(rid) for rid in targets["invalidate_relationship_ids"]],
        "applied_at": undo.applied_at.isoformat() if undo.applied_at else None,
    }


def targets_from_payload(payload: dict[str, Any]) -> dict[str, list[UUID]]:
    """Inverse of :func:`mirror_payload` for the reconciler replay path."""
    return {
        "retire_entity_ids": [u for u in (_coerce_uuid(x) for x in payload.get("retire_entity_ids") or []) if u],
        "invalidate_relationship_ids": [
            u for u in (_coerce_uuid(x) for x in payload.get("invalidate_relationship_ids") or []) if u
        ],
    }


def _parse_applied_at(payload: dict[str, Any], fallback: datetime) -> datetime:
    raw = payload.get("applied_at")
    if not raw:
        return fallback
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return fallback


async def apply_mirror_targets(
    graph: GraphBackendProtocol,
    targets: dict[str, list[UUID]],
    *,
    namespace_id: UUID,
    stamp_at: datetime,
) -> dict[str, int]:
    """Push the soft-deletes to the graph via the #1271 verbs.

    Idempotent by-id (the verbs only touch rows with ``valid_until IS NULL``).
    Empty target lists short-circuit inside the verbs (return 0), so a no-op
    apply costs zero Cypher round-trips. Returns the per-verb affected counts.
    """
    retired = 0
    invalidated = 0
    if targets["retire_entity_ids"]:
        retired = await graph.soft_retire_entities_batch(
            targets["retire_entity_ids"],
            namespace_id=namespace_id,
            retired_at=stamp_at,
            reason="dream_consolidated",
        )
    if targets["invalidate_relationship_ids"]:
        invalidated = await graph.soft_invalidate_relationships_batch(
            targets["invalidate_relationship_ids"],
            namespace_id=namespace_id,
            invalidated_at=stamp_at,
        )
    return {"entities_retired": retired, "relationships_invalidated": invalidated}


async def apply_mirror_payload(
    graph: GraphBackendProtocol,
    payload: dict[str, Any],
    *,
    namespace_id: UUID,
    fallback_stamp: datetime,
) -> dict[str, int]:
    """Reconciler entry: re-mirror from a persisted ``graph_mirror_pending`` payload."""
    targets = targets_from_payload(payload)
    stamp_at = _parse_applied_at(payload, fallback_stamp)
    return await apply_mirror_targets(graph, targets, namespace_id=namespace_id, stamp_at=stamp_at)

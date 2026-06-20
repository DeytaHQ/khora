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
                           + ``rewrite_relationship_endpoints_batch`` for the
                             incident-edge re-pointing (#1273)

The entity-merge endpoint-rewrite leg of dedupe re-points every incident edge
from each absorbed entity to the canonical (#1273), honoring the #806
absorbed->canonical id-remap: the Phase-1 planner produces union-find merge
components with a single transitive survivor, so a global absorbed->canonical map
across all merges in the op resolves both endpoints of every snapshotted edge to
that one canonical (A->B->C collapses to one survivor, no edge points at a
retired intermediate). Edges that become self-loops after the remap are already
carried in ``self_loops_invalidated`` and routed to the invalidate verb - they
are excluded from the rewrite list so the two paths never double-apply.

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

from khora.core.models import CommunityNode
from khora.telemetry.metrics import metric_counter

if TYPE_CHECKING:
    from khora.dream.plan import DreamOp
    from khora.dream.result import UndoRecord
    from khora.storage.backends.base import GraphBackendProtocol

# Op kinds this mirror translates. Subset of the graph backend's
# ``supports_dream_mirror()`` probe. ``VECTORCYPHER_DEDUPE_ENTITIES`` now also
# routes through ``rewrite_relationship_endpoints_batch`` for the entity-merge
# incident-edge re-pointing (#1273).
_PRUNE_EDGES = "vectorcypher_prune_edges"
_DEDUPE_ENTITIES = "vectorcypher_dedupe_entities"
# Additive community materialization (#1276): the GraphRAG payoff. Unlike the
# prune / dedupe legs (soft-deletes), this MERGEs :Community nodes + member
# edges into the graph so the dream summaries are queryable at recall.
_COMMUNITY_SUMMARY = "vectorcypher_community_summary"

# Op kinds whose soft-deletes / materializations this module knows how to
# mirror. Anything outside this set (centroid recompute, normalize_schema
# relabel, source-chunk GC) is recorded as a structured skip rather than
# mirrored - see ``mirror_skip_reason``.
MIRRORABLE_OP_KINDS: frozenset[str] = frozenset({_PRUNE_EDGES, _DEDUPE_ENTITIES, _COMMUNITY_SUMMARY})


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


def extract_mirror_targets(op_type: str, undo: UndoRecord) -> dict[str, list[Any]]:
    """Translate a just-committed ``UndoRecord`` into graph mirror targets.

    Returns a dict with three keys:

      - ``retire_entity_ids``: entity ids (``UUID``) to soft-retire (absorbed
        nodes).
      - ``invalidate_relationship_ids``: relationship ids (``UUID``) to
        soft-invalidate (pruned edges + post-rewrite dedupe self-loops).
      - ``rewrite_relationships``: dedupe incident-edge re-pointings (#1273),
        each a dict ``{"relationship_id", "source_entity_id",
        "target_entity_id", "relationship_type"}`` carrying the POST-rewrite
        endpoints the graph verb expects.

    The ``UndoRecord.before`` snapshot is the source of truth: it records
    exactly the rows the PG handler committed. A no-op apply (already-pruned,
    vanished, verifier-rejected merge) yields empty lists, so the mirror is a
    no-op too.

    The dedupe ``previous_relationships`` snapshot carries each incident edge's
    PRE-rewrite endpoints; this function applies the #806 absorbed->canonical
    id-remap (built globally across every applied merge in the op so a
    cross-component endpoint that is itself absorbed resolves to its canonical)
    to compute the post-rewrite endpoints. An edge that becomes a self-loop
    after the remap is already in ``self_loops_invalidated`` and is excluded
    from ``rewrite_relationships`` so the invalidate and rewrite paths never
    double-apply.
    """
    before = undo.before or {}
    retire_entity_ids: list[UUID] = []
    invalidate_relationship_ids: list[UUID] = []
    rewrite_relationships: list[dict[str, str]] = []

    if op_type == _PRUNE_EDGES:
        # prune_edges: before == {"noop": True} or {"relationships": [{...}]}.
        for row in before.get("relationships") or []:
            rel_id = _coerce_uuid(row.get("relationship_id"))
            if rel_id is not None:
                invalidate_relationship_ids.append(rel_id)
    elif op_type == _DEDUPE_ENTITIES:
        # dedupe_entities: before == {"merges": [{...}]}. Each applied merge
        # soft-deletes the absorbed entity (entities.valid_until), may invalidate
        # post-rewrite self-loops (relationships.invalidated_at), and re-points
        # incident edges from absorbed -> canonical (#1273).
        applied_merges = [m for m in (before.get("merges") or []) if m.get("applied")]

        # #806 id-remap: a single global absorbed -> canonical map across every
        # applied merge in the op. The planner produces union-find components
        # with one transitive survivor each, so chaining (canonical itself being
        # absorbed by another merge) does not occur within one op; the global map
        # still resolves a cross-component endpoint that happens to be absorbed.
        absorbed_to_canonical: dict[UUID, UUID] = {}
        for merge in applied_merges:
            absorbed_id = _coerce_uuid(merge.get("absorbed_id"))
            canonical_id = _coerce_uuid(merge.get("canonical_id"))
            if absorbed_id is not None and canonical_id is not None:
                absorbed_to_canonical[absorbed_id] = canonical_id

        for merge in applied_merges:
            absorbed_id = _coerce_uuid(merge.get("absorbed_id"))
            if absorbed_id is not None:
                retire_entity_ids.append(absorbed_id)
            self_loop_ids = {
                rel_id for rid in merge.get("self_loops_invalidated") or [] if (rel_id := _coerce_uuid(rid))
            }
            invalidate_relationship_ids.extend(self_loop_ids)

            for prev in merge.get("previous_relationships") or []:
                rel_id = _coerce_uuid(prev.get("id"))
                src = _coerce_uuid(prev.get("source_entity_id"))
                tgt = _coerce_uuid(prev.get("target_entity_id"))
                if rel_id is None or src is None or tgt is None:
                    continue
                # The self-loop path already owns this edge - don't also rewrite.
                if rel_id in self_loop_ids:
                    continue
                new_src = absorbed_to_canonical.get(src, src)
                new_tgt = absorbed_to_canonical.get(tgt, tgt)
                if new_src == new_tgt:
                    # Became a self-loop under the global remap but the PG handler
                    # did not list it as one (cross-component). Mirror it as an
                    # invalidate, not a rewrite, to match recall semantics.
                    invalidate_relationship_ids.append(rel_id)
                    continue
                if new_src == src and new_tgt == tgt:
                    # No endpoint moved (idempotent replay / unrelated edge).
                    continue
                rewrite_relationships.append(
                    {
                        "relationship_id": str(rel_id),
                        "source_entity_id": str(new_src),
                        "target_entity_id": str(new_tgt),
                        "relationship_type": str(prev.get("relationship_type") or ""),
                    }
                )

    return {
        "retire_entity_ids": retire_entity_ids,
        "invalidate_relationship_ids": invalidate_relationship_ids,
        "rewrite_relationships": rewrite_relationships,
    }


def extract_community_targets(op_type: str, undo: UndoRecord) -> list[CommunityNode]:
    """Translate a just-committed community_summary ``UndoRecord`` into nodes (#1276).

    The community apply handler stamps the persisted summary text + member ids
    onto ``UndoRecord.before`` (the same source-of-truth pattern the soft-delete
    legs use). A no-op apply (already-live replay, no grounded claims) carries
    ``before["noop"]`` and yields no node, so the materialization is a no-op too.
    """
    if op_type != _COMMUNITY_SUMMARY:
        return []
    before = undo.before or {}
    if before.get("noop"):
        return []
    community_id = _coerce_uuid(before.get("community_id"))
    if community_id is None:
        return []
    member_ids = [u for u in (_coerce_uuid(m) for m in before.get("member_ids") or []) if u]
    return [
        CommunityNode(
            id=community_id,
            summary=str(before.get("summary_text") or ""),
            member_ids=member_ids,
            summary_depth=int(before.get("summary_depth") or 1),
        )
    ]


def mirror_payload(op: DreamOp, undo: UndoRecord) -> dict[str, Any]:
    """Serialize the mirror targets for the ``graph_mirror_pending`` slot.

    The reconciler re-attempts from this JSON payload rather than re-reading the
    undo file - it carries the exact id lists plus the stamp time, so a replay
    after a crash mirrors precisely what PG committed.
    """
    op_type = str(op.op_type)
    targets = extract_mirror_targets(op_type, undo)
    communities = extract_community_targets(op_type, undo)
    return {
        "retire_entity_ids": [str(eid) for eid in targets["retire_entity_ids"]],
        "invalidate_relationship_ids": [str(rid) for rid in targets["invalidate_relationship_ids"]],
        # Already JSON-safe (all str) - persisted verbatim for the reconciler.
        "rewrite_relationships": list(targets["rewrite_relationships"]),
        "communities": [
            {
                "id": str(c.id),
                "summary": c.summary,
                "member_ids": [str(m) for m in c.member_ids],
                "summary_depth": c.summary_depth,
            }
            for c in communities
        ],
        "applied_at": undo.applied_at.isoformat() if undo.applied_at else None,
    }


def targets_from_payload(payload: dict[str, Any]) -> dict[str, list[Any]]:
    """Inverse of :func:`mirror_payload` for the reconciler replay path."""
    rewrites: list[dict[str, str]] = []
    for rw in payload.get("rewrite_relationships") or []:
        rel_id = _coerce_uuid(rw.get("relationship_id"))
        src = _coerce_uuid(rw.get("source_entity_id"))
        tgt = _coerce_uuid(rw.get("target_entity_id"))
        if rel_id is None or src is None or tgt is None:
            continue
        rewrites.append(
            {
                "relationship_id": str(rel_id),
                "source_entity_id": str(src),
                "target_entity_id": str(tgt),
                "relationship_type": str(rw.get("relationship_type") or ""),
            }
        )
    return {
        "retire_entity_ids": [u for u in (_coerce_uuid(x) for x in payload.get("retire_entity_ids") or []) if u],
        "invalidate_relationship_ids": [
            u for u in (_coerce_uuid(x) for x in payload.get("invalidate_relationship_ids") or []) if u
        ],
        "rewrite_relationships": rewrites,
    }


def communities_from_payload(payload: dict[str, Any]) -> list[CommunityNode]:
    """Inverse of the ``communities`` slot in :func:`mirror_payload` (reconciler)."""
    out: list[CommunityNode] = []
    for entry in payload.get("communities") or []:
        cid = _coerce_uuid(entry.get("id"))
        if cid is None:
            continue
        member_ids = [u for u in (_coerce_uuid(m) for m in entry.get("member_ids") or []) if u]
        out.append(
            CommunityNode(
                id=cid,
                summary=str(entry.get("summary") or ""),
                member_ids=member_ids,
                summary_depth=int(entry.get("summary_depth") or 1),
            )
        )
    return out


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
    targets: dict[str, list[Any]],
    *,
    namespace_id: UUID,
    stamp_at: datetime,
    communities: list[CommunityNode] | None = None,
) -> dict[str, int]:
    """Push the soft-deletes + entity-merge re-pointings + community materialization to the graph (#1271/#1273/#1276).

    Idempotent by-id (the soft-delete verbs only touch rows with
    ``valid_until IS NULL`` / edges still on the old endpoints; community
    materialization MERGEs on community id). Empty inputs short-circuit inside
    the verbs (return 0), so a no-op apply costs zero Cypher round-trips.

    Ordering: re-point incident edges onto the canonical FIRST, then invalidate
    self-loops and retire the absorbed node. The rewrite verb matches the
    canonical node by id (the canonical is never retired), so the soft-retire of
    the absorbed node afterwards cannot strand a still-to-be-moved edge. Returns
    the per-verb affected counts.
    """
    rewritten = 0
    retired = 0
    invalidated = 0
    materialized = 0
    if targets["rewrite_relationships"]:
        rewritten = await graph.rewrite_relationship_endpoints_batch(
            targets["rewrite_relationships"],
            namespace_id=namespace_id,
            rewritten_at=stamp_at,
        )
    if targets["invalidate_relationship_ids"]:
        invalidated = await graph.soft_invalidate_relationships_batch(
            targets["invalidate_relationship_ids"],
            namespace_id=namespace_id,
            invalidated_at=stamp_at,
        )
    if targets["retire_entity_ids"]:
        retired = await graph.soft_retire_entities_batch(
            targets["retire_entity_ids"],
            namespace_id=namespace_id,
            retired_at=stamp_at,
            reason="dream_consolidated",
        )
    if communities:
        materialized = await graph.materialize_communities_batch(
            communities,
            namespace_id=namespace_id,
            materialized_at=stamp_at,
        )
    return {
        "entities_retired": retired,
        "relationships_invalidated": invalidated,
        "relationships_rewritten": rewritten,
        "communities_materialized": materialized,
    }


async def apply_mirror_payload(
    graph: GraphBackendProtocol,
    payload: dict[str, Any],
    *,
    namespace_id: UUID,
    fallback_stamp: datetime,
) -> dict[str, int]:
    """Reconciler entry: re-mirror from a persisted ``graph_mirror_pending`` payload."""
    targets = targets_from_payload(payload)
    communities = communities_from_payload(payload)
    stamp_at = _parse_applied_at(payload, fallback_stamp)
    return await apply_mirror_targets(
        graph, targets, namespace_id=namespace_id, stamp_at=stamp_at, communities=communities
    )

"""Vectorcypher PageRank-based orphan-entity report (#657).

Read-only dream operation that surfaces entities looking like archive
candidates: bottom Nth-percentile PageRank score AND ``mention_count`` at
or below 1. ``ASSOCIATED_WITH`` co-occurrence edges are down-weighted to
keep the PR distribution meaningful — selective extraction emits these
by default for non-LLM chunks and they would otherwise dominate.

The op is read-only: zero mutations, zero LLM calls. Apply-side
archival is explicitly out of scope for v0.14 (see #649 Phase 4+).

Recall-hit counts are not yet wired through Phase 0 telemetry — the
ticket calls this gap out and accepts zeroed recall hits as the v1
behaviour. ``mention_count`` carries the load alone here.
"""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from khora import _accel
from khora.dream.plan import DreamOp, OpKind
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from khora.extraction.skills.base import ExpertiseConfig
    from khora.storage.coordinator import StorageCoordinator


# Canonical co-occurrence relationship label emitted by selective
# extraction. Matched case-insensitively so adapter-specific casing
# (Neo4j upper-case vs SurrealDB free-form) doesn't slip through.
_COOCCURRENCE_REL_TYPE = "ASSOCIATED_WITH"

_PHASE = "audit"


async def plan_vectorcypher_orphan_report(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    expertise: ExpertiseConfig | None = None,
    pr_percentile_threshold: float = 5.0,
    cooccurrence_edge_weight: float = 0.2,
) -> DreamOp:
    """Surface entities that look orphaned based on PageRank + mention_count.

    Reads ``list_entities`` / ``list_relationships`` from the coordinator
    (works on graph-backed and graph-less stacks alike — see
    ``coordinator.list_entities`` for the fallback chain), builds an
    edge list with ``ASSOCIATED_WITH`` edges down-weighted to
    ``cooccurrence_edge_weight``, runs :func:`khora._accel.pagerank`, and
    returns a :class:`DreamOp` carrying the archive candidates.

    Args:
        namespace_id: Namespace to audit.
        coordinator: Storage coordinator (DI for tests).
        expertise: Optional expertise config — reserved for future entity-type
            filtering; not consumed in v1.
        pr_percentile_threshold: Bottom-percentile cut-off (0-100). Entities
            scoring at or below this percentile AND with ``mention_count <= 1``
            are flagged. Default 5.0 (the bottom 5%).
        cooccurrence_edge_weight: Weight for ``ASSOCIATED_WITH`` edges.
            Default 0.2.

    Returns:
        :class:`DreamOp` with ``op_type=VECTORCYPHER_ORPHAN_REPORT``. The
        ``outputs`` tuple holds one dict per archive candidate, each with
        ``entity_id``, ``name``, ``entity_type``, ``pr_score``,
        ``mention_count``, ``archive_candidate=True``. ``decision`` is one
        of ``"audit_complete"`` or ``"empty_namespace"``.
    """
    del expertise  # Reserved for future filtering; unused in v1.

    op_id = uuid4()
    started_at_wall = _utcnow()
    t0 = time.perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.orphan_report",
        run_id="",
        op_id=str(op_id),
        namespace_id=str(namespace_id),
        phase=_PHASE,
        weighting_used=cooccurrence_edge_weight,
    ) as span:
        entities = await coordinator.list_entities(namespace_id, limit=100_000)
        total_entities = len(entities)
        span.set_attribute("total_entities", total_entities)

        if total_entities == 0:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            span.set_attribute("candidate_count", 0)
            span.set_attribute("p5_combined_score", 0.0)
            return DreamOp(
                op_id=op_id,
                phase=_PHASE,
                op_type=OpKind.VECTORCYPHER_ORPHAN_REPORT,
                inputs=(
                    {
                        "pr_percentile_threshold": pr_percentile_threshold,
                        "cooccurrence_edge_weight": cooccurrence_edge_weight,
                    },
                ),
                outputs=(),
                decision="empty_namespace",
                rationale="Namespace has zero entities; nothing to score.",
                started_at=started_at_wall,
                duration_ms=duration_ms,
                namespace_id=namespace_id,
            )

        relationships = await coordinator.list_relationships(namespace_id, limit=1_000_000)

        index_by_id = {entity.id: idx for idx, entity in enumerate(entities)}
        edges: list[tuple[int, int, float]] = []
        for rel in relationships:
            src_idx = index_by_id.get(rel.source_entity_id)
            tgt_idx = index_by_id.get(rel.target_entity_id)
            if src_idx is None or tgt_idx is None:
                continue
            weight = cooccurrence_edge_weight if rel.relationship_type.upper() == _COOCCURRENCE_REL_TYPE else 1.0
            edges.append((src_idx, tgt_idx, weight))

        scores = _accel.pagerank(total_entities, edges)

        # Percentile cut-off on the PR distribution. ``quantiles`` with
        # ``n=100`` returns the 99 internal cut-points (p1..p99); index 4
        # picks p5. Fall back to ``min(scores)`` when there are too few
        # entities for ``quantiles`` to compute (n<2 raises).
        if total_entities >= 2:
            cut_points = statistics.quantiles(scores, n=100, method="inclusive")
            cut_idx = max(0, min(98, int(pr_percentile_threshold) - 1))
            pr_threshold = cut_points[cut_idx]
        else:
            pr_threshold = scores[0]

        candidates: list[dict[str, Any]] = []
        for entity, score in zip(entities, scores, strict=True):
            if score <= pr_threshold and entity.mention_count <= 1:
                candidates.append(
                    {
                        "entity_id": str(entity.id),
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "pr_score": score,
                        "mention_count": entity.mention_count,
                        "archive_candidate": True,
                    }
                )

        span.set_attribute("candidate_count", len(candidates))
        span.set_attribute("p5_combined_score", float(pr_threshold))

        duration_ms = (time.perf_counter() - t0) * 1000.0

    return DreamOp(
        op_id=op_id,
        phase=_PHASE,
        op_type=OpKind.VECTORCYPHER_ORPHAN_REPORT,
        inputs=(
            {
                "pr_percentile_threshold": pr_percentile_threshold,
                "cooccurrence_edge_weight": cooccurrence_edge_weight,
                "total_entities": total_entities,
                "total_relationships": len(relationships),
            },
        ),
        outputs=tuple(candidates),
        decision="audit_complete",
        rationale=(
            f"Flagged {len(candidates)} of {total_entities} entities below "
            f"p{pr_percentile_threshold:g} PR ({pr_threshold:.6g}) with "
            f"mention_count <= 1."
        ),
        started_at=started_at_wall,
        duration_ms=duration_ms,
        namespace_id=namespace_id,
    )


def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(UTC)

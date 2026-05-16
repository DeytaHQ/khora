"""Engine-plugin registry for dream-phase orchestration (#661).

Each registered engine plugin implements :class:`DreamCapable` and
encapsulates the per-op ``plan_*`` calls under one ``plan_dream`` entry
point. The orchestrator runtime-checks every plugin against the
Protocol before dispatching.

Phase 1 only ships read-only audit ops, so ``apply_dream`` is a
pass-through that re-stamps the op records and returns a
:class:`DreamResult`. Phase 2+ tickets land destructive apply handlers
on top of the same scaffolding.

Stability: **internal** (Phase 1.0). The Protocol itself
(:class:`DreamCapable`) is internal until the dream surface stabilizes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from khora.dream.engines.chronicle import (
    plan_chronicle_abstention_drift,
    plan_chronicle_event_clustering,
    plan_chronicle_fact_compaction,
    plan_chronicle_tombstone_audit,
)
from khora.dream.engines.vectorcypher import (
    plan_vectorcypher_centroid_recompute,
    plan_vectorcypher_dedupe_entities,
    plan_vectorcypher_orphan_report,
    plan_vectorcypher_schema_drift,
    plan_vectorcypher_source_chunk_ids_audit,
    plan_vectorcypher_source_chunk_ids_gc,
)
from khora.dream.exceptions import DreamForbiddenOpError
from khora.dream.plan import Checkpoint, DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.result import DreamDiff, DreamProgress, DreamResult, DreamRunInfo, OpSummary
from khora.exceptions import KhoraError

if TYPE_CHECKING:
    from khora.dream.config import DreamConfig
    from khora.extraction.skills.base import ExpertiseConfig
    from khora.khora import Khora


# ---------------------------------------------------------------------------
# Plan-hash canonicalization
# ---------------------------------------------------------------------------


def canonical_plan_payload(plan: DreamPlan) -> str:
    """Return a stable JSON serialization for hashing a :class:`DreamPlan`.

    Only the ops' identity-bearing fields (``op_type``, ``inputs``,
    ``decision``) feed the hash. ``op_id`` is excluded because it is
    randomly generated per plan-build and would defeat the purpose of a
    drift-detection hash on resume.
    """
    payload = {
        "namespace_id": str(plan.namespace_id),
        "ops": [
            {
                "op_type": str(op.op_type),
                "inputs": _jsonable(op.inputs),
                "decision": op.decision,
            }
            for op in plan.ops
        ],
    }
    return json.dumps(payload, sort_keys=True, default=str)


def plan_hash(plan: DreamPlan) -> str:
    """SHA1[:16] of ``canonical_plan_payload``. Hex, lowercase.

    SHA1 is used for a stability/identity digest only, never for
    authentication — bandit S324 is silenced via ``usedforsecurity``.
    """
    return hashlib.sha1(  # noqa: S324
        canonical_plan_payload(plan).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]


def _jsonable(value: object) -> object:
    """Recursively coerce a value to a json-serializable form."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Engine plugins
# ---------------------------------------------------------------------------


# Document-deletion op kinds. Banned today (Documents are tombstone-only).
# Phase 4+ may introduce a soft-delete variant but it will not appear in
# this list — it would be a separate, bi-temporal op kind.
_FORBIDDEN_OP_KINDS: frozenset[str] = frozenset(
    {
        "delete_document",
        "document_delete",
        "documents_delete",
    }
)


def _validate_no_forbidden_ops(plan: DreamPlan) -> None:
    """Reject any plan that touches the safety floor.

    Called at both plan-time (after the engine builds the plan) and
    apply-time (before any op runs). Defense in depth.
    """
    for op in plan.ops:
        if str(op.op_type) in _FORBIDDEN_OP_KINDS:
            raise DreamForbiddenOpError(
                f"Plan op {op.op_id} carries forbidden op_type={op.op_type!r}; "
                "Document deletes are not permitted in the dream phase."
            )


class _ChroniclePlugin:
    """``DreamCapable`` plugin wrapping chronicle Phase 1 audit ops."""

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return frozenset(
            {
                OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT,
                OpKind.CHRONICLE_TOMBSTONE_AUDIT,
                OpKind.CHRONICLE_FACT_COMPACTION,
                OpKind.CHRONICLE_EVENT_CLUSTERING,
            }
        )

    async def plan_dream(
        self,
        kb: Khora,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: ExpertiseConfig | None = None,
    ) -> DreamPlan:
        del expertise  # chronicle audits don't consult expertise in Phase 1
        ops: list[DreamOp] = []
        wanted = _resolved_scope(scope, self.dream_capabilities)
        coordinator = kb.storage

        if OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT in wanted:
            engine = kb._get_engine()
            # The op only reads three threshold attrs. The
            # vectorcypher engine doesn't have them; if the orchestrator
            # routed an abstention-drift op to it, that's a misconfig —
            # raise rather than silently produce a meaningless report.
            if not hasattr(engine, "_abstention_min_top_score"):
                raise KhoraError("chronicle abstention drift requested but active engine is not a ChronicleEngine")
            op = await plan_chronicle_abstention_drift(namespace_id, engine=engine, config=config)
            ops.append(op)

        if OpKind.CHRONICLE_TOMBSTONE_AUDIT in wanted:
            async with coordinator.transaction() as txn:
                op = await plan_chronicle_tombstone_audit(namespace_id, session=txn.session, config=config)
            ops.append(op)

        if OpKind.CHRONICLE_FACT_COMPACTION in wanted:
            async with coordinator.transaction() as txn:
                phase2_ops = await plan_chronicle_fact_compaction(namespace_id, session=txn.session, config=config)
            ops.extend(phase2_ops)

        if OpKind.CHRONICLE_EVENT_CLUSTERING in wanted:
            async with coordinator.transaction() as txn:
                phase2_ops = await plan_chronicle_event_clustering(namespace_id, session=txn.session, config=config)
            ops.extend(phase2_ops)

        return DreamPlan(
            plan_id=uuid4(),
            namespace_id=namespace_id,
            ops=tuple(ops),
        )

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Checkpoint | None = None,
        on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    ) -> DreamResult:
        del checkpoint, on_progress  # Phase 1 ops are pure observation
        _validate_no_forbidden_ops(plan)
        return _build_pass_through_result(plan, mode="apply")


class _VectorCypherPlugin:
    """``DreamCapable`` plugin wrapping vectorcypher Phase 1 audit ops."""

    @property
    def dream_capabilities(self) -> frozenset[OpKind]:
        return frozenset(
            {
                OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT,
                OpKind.VECTORCYPHER_ORPHAN_REPORT,
                OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT,
                OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
                OpKind.VECTORCYPHER_CENTROID_RECOMPUTE,
                OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC,
            }
        )

    async def plan_dream(
        self,
        kb: Khora,
        namespace_id: UUID,
        *,
        scope: DreamScope,
        config: DreamConfig,
        expertise: ExpertiseConfig | None = None,
    ) -> DreamPlan:
        ops: list[DreamOp] = []
        wanted = _resolved_scope(scope, self.dream_capabilities)
        coordinator = kb.storage

        if OpKind.VECTORCYPHER_SCHEMA_DRIFT_REPORT in wanted:
            # schema_drift requires an ExpertiseConfig; skip when caller
            # didn't supply one rather than crashing.
            if expertise is not None:
                op = await plan_vectorcypher_schema_drift(
                    namespace_id,
                    coordinator=coordinator,
                    expertise=expertise,
                )
                ops.append(op)

        if OpKind.VECTORCYPHER_ORPHAN_REPORT in wanted:
            op = await plan_vectorcypher_orphan_report(
                namespace_id,
                coordinator=coordinator,
                expertise=expertise,
                pr_percentile_threshold=config.orphan_pr_percentile_threshold,
                cooccurrence_edge_weight=config.cooccurrence_edge_weight,
            )
            ops.append(op)

        if OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT in wanted:
            op = await plan_vectorcypher_source_chunk_ids_audit(namespace_id, coordinator=coordinator)
            ops.append(op)

        if OpKind.VECTORCYPHER_DEDUPE_ENTITIES in wanted:
            phase2_ops = await plan_vectorcypher_dedupe_entities(
                namespace_id,
                coordinator=coordinator,
                default_threshold=config.dedupe_entities_default_threshold,
                per_type_thresholds=config.dedupe_entities_per_type_thresholds,
                mode="dry-run",
            )
            ops.extend(phase2_ops)

        if OpKind.VECTORCYPHER_CENTROID_RECOMPUTE in wanted:
            # centroid_recompute needs merge_clusters as input. In the
            # default dispatch we pass [] — operators who want real
            # centroid plans either invoke the planner function directly
            # with clusters from a prior dedupe run, or wait for the
            # auto-chaining off dedupe results that lands in v0.15.x.
            phase2_ops = await plan_vectorcypher_centroid_recompute(
                namespace_id,
                coordinator=coordinator,
                merge_clusters=[],
                mode="dry-run",
                lev_threshold=config.centroid_lev_threshold,
                min_intra_cluster_cosine=config.centroid_min_intra_cluster_cosine,
            )
            ops.extend(phase2_ops)

        if OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC in wanted:
            phase2_ops = await plan_vectorcypher_source_chunk_ids_gc(
                namespace_id,
                coordinator=coordinator,
                min_dead=config.source_chunk_ids_gc_min_dead,
                mode="dry-run",
            )
            ops.extend(phase2_ops)

        return DreamPlan(
            plan_id=uuid4(),
            namespace_id=namespace_id,
            ops=tuple(ops),
        )

    async def apply_dream(
        self,
        plan: DreamPlan,
        *,
        checkpoint: Checkpoint | None = None,
        on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    ) -> DreamResult:
        del checkpoint, on_progress
        _validate_no_forbidden_ops(plan)
        return _build_pass_through_result(plan, mode="apply")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolved_scope(scope: DreamScope, capabilities: frozenset[OpKind]) -> frozenset[OpKind]:
    """Intersect requested op_kinds with the plugin's capabilities."""
    if scope.op_kinds is None:
        return capabilities
    requested = frozenset(scope.op_kinds)
    return requested & capabilities


def _build_pass_through_result(plan: DreamPlan, *, mode: str) -> DreamResult:
    """Build a :class:`DreamResult` from a plan whose ops are all complete.

    Phase 1 ops carry their decision and outputs from the plan call
    itself (they're pure observation), so the apply pass just re-shapes
    the plan into a result.
    """
    now = datetime.now(UTC)
    summaries: dict[str, OpSummary] = {}
    for op in plan.ops:
        key = str(op.op_type)
        cur = summaries.get(key) or OpSummary(op_type=key)
        summaries[key] = OpSummary(
            op_type=key,
            planned=cur.planned + 1,
            applied=cur.applied + 1,
            skipped=cur.skipped,
            failed=cur.failed,
        )

    info = DreamRunInfo(
        run_id=uuid4(),
        namespace_id=plan.namespace_id,
        mode="apply" if mode == "apply" else "dry-run",
        started_at=now,
        finished_at=now,
        duration_ms=0.0,
    )
    return DreamResult(
        run=info,
        diff=DreamDiff(),
        ops=tuple(summaries.values()),
    )


# ---------------------------------------------------------------------------
# Public registry surface
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, object] = {
    "chronicle": _ChroniclePlugin(),
    "vectorcypher": _VectorCypherPlugin(),
}


def get_engine_plugin(engine_name: str) -> object:
    """Return the registered plugin for ``engine_name``.

    Raises:
        KhoraError: when no plugin is registered for the engine.
    """
    plugin = _REGISTRY.get(engine_name)
    if plugin is None:
        raise KhoraError(f"engine {engine_name!r} doesn't support dream phase (no DreamCapable plugin registered)")
    return plugin


__all__ = [
    "canonical_plan_payload",
    "get_engine_plugin",
    "plan_hash",
]

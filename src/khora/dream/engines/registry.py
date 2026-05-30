"""Engine-plugin registry for dream-phase orchestration (#661, #667).

Each registered engine plugin implements :class:`DreamCapable` and
encapsulates the per-op ``plan_*`` calls under one ``plan_dream`` entry
point. The orchestrator runtime-checks every plugin against the
Protocol before dispatching.

Phase 4 (#667) introduces real per-op apply handlers. The orchestrator
walks each :class:`DreamOp` in the plan, looks up the handler via
:func:`get_apply_handler`, and invokes it inside its own
coordinator transaction. Handlers return an :class:`UndoRecord`; ops
without a handler (Phase 1 audit ops) return ``None`` and are skipped.

Stability: **internal** (Phase 1.0). The Protocol itself
(:class:`DreamCapable`) is internal until the dream surface stabilizes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from loguru import logger

from khora.dream.engines.chronicle import (
    plan_chronicle_abstention_drift,
    plan_chronicle_tombstone_audit,
)
from khora.dream.engines.vectorcypher import (
    plan_vectorcypher_community_summary,
    plan_vectorcypher_contradiction_detect,
    plan_vectorcypher_normalize_schema,
    plan_vectorcypher_orphan_report,
    plan_vectorcypher_prune_edges,
    plan_vectorcypher_schema_drift,
    plan_vectorcypher_source_chunk_ids_audit,
)
from khora.dream.exceptions import DreamForbiddenOpError
from khora.dream.plan import Checkpoint, DreamOp, DreamPlan, DreamScope, OpKind
from khora.dream.result import DreamDiff, DreamProgress, DreamResult, DreamRunInfo, OpSummary, UndoRecord
from khora.exceptions import KhoraError

if TYPE_CHECKING:
    from khora.dream.config import DreamConfig
    from khora.extraction.skills.base import ExpertiseConfig
    from khora.khora import Khora


# Per-op apply-handler signature. Handlers run inside an orchestrator-owned
# coordinator transaction; they take the op + the open session and return
# an :class:`UndoRecord`. Returning ``None`` is reserved for ops that have
# no apply handler (Phase 1 audit ops).
ApplyHandler = Callable[..., Awaitable["UndoRecord"]]


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


# ---------------------------------------------------------------------------
# Apply-handler dispatch (#667)
# ---------------------------------------------------------------------------
#
# Each OpKind that has an apply path maps to the name of a function the
# corresponding engine submodule exports. Lookup is lazy via importlib so
# this module can stay decoupled from individual handler modules — the
# parallel agents writing the handlers only need to export the name listed
# below. Phase 1 audit ops (e.g. CHRONICLE_TOMBSTONE_AUDIT) are absent
# from the map; the orchestrator treats that as a no-op pass-through.

_APPLY_HANDLER_NAMES: dict[OpKind, tuple[str, str]] = {
    # (module-path, attribute-name)
    OpKind.VECTORCYPHER_DEDUPE_ENTITIES: (
        "khora.dream.engines.vectorcypher.dedupe_entities",
        "apply_vectorcypher_dedupe_entities",
    ),
    OpKind.VECTORCYPHER_CENTROID_RECOMPUTE: (
        "khora.dream.engines.vectorcypher.centroid_recompute",
        "apply_vectorcypher_centroid_recompute",
    ),
    OpKind.VECTORCYPHER_SOURCE_CHUNK_IDS_GC: (
        "khora.dream.engines.vectorcypher.source_chunk_ids_gc",
        "apply_vectorcypher_source_chunk_ids_gc",
    ),
    OpKind.VECTORCYPHER_PRUNE_EDGES: (
        "khora.dream.engines.vectorcypher.prune_edges",
        "apply_vectorcypher_prune_edges",
    ),
    OpKind.CHRONICLE_FACT_COMPACTION: (
        "khora.dream.engines.chronicle.fact_compaction",
        "apply_chronicle_fact_compaction",
    ),
    OpKind.CHRONICLE_EVENT_CLUSTERING: (
        "khora.dream.engines.chronicle.event_clustering",
        "apply_chronicle_event_clustering",
    ),
    OpKind.VECTORCYPHER_COMMUNITY_SUMMARY: (
        "khora.dream.engines.vectorcypher.community_summary",
        "apply_vectorcypher_community_summary",
    ),
    OpKind.VECTORCYPHER_CONTRADICTION_DETECT: (
        "khora.dream.engines.vectorcypher.contradiction_detect",
        "apply_vectorcypher_contradiction_detect",
    ),
    OpKind.VECTORCYPHER_NORMALIZE_SCHEMA: (
        "khora.dream.engines.vectorcypher.normalize_schema",
        "apply_vectorcypher_normalize_schema",
    ),
}


def get_apply_handler(op_type: OpKind | str) -> ApplyHandler | None:
    """Resolve the apply handler for ``op_type``.

    Returns ``None`` for Phase 1 audit ops that don't carry an apply
    handler. The orchestrator treats ``None`` as "skip — no mutation
    needed".

    Looking up via importlib means the engine submodule does not need
    to import its handler from the registry, and the registry does not
    need to import the handler at module load. The parallel handler
    implementations land in their respective engine submodules.

    Raises:
        DreamForbiddenOpError: ``op_type`` lives in
            :data:`_FORBIDDEN_OP_KINDS`. Defense in depth — the safety
            floor check should have rejected the plan first, but this
            re-check makes the apply path safe against a buggy
            ``_validate_no_forbidden_ops`` call site.
    """
    op_type_str = str(op_type)
    if op_type_str in _FORBIDDEN_OP_KINDS:
        raise DreamForbiddenOpError(
            f"Apply-handler lookup for forbidden op_type={op_type_str!r}; "
            "_validate_no_forbidden_ops should have aborted the run earlier."
        )

    try:
        op_kind = OpKind(op_type_str)
    except ValueError:
        # Unknown op_type — treat as no-handler so the orchestrator's
        # pass-through path runs.
        return None

    entry = _APPLY_HANDLER_NAMES.get(op_kind)
    if entry is None:
        return None

    import importlib

    module_path, attr = entry
    module = importlib.import_module(module_path)
    handler = getattr(module, attr, None)
    if handler is None:
        # Handler module exists but the symbol is missing — a parallel
        # agent has not landed yet. Treat as no-handler so the
        # orchestrator continues (and the missing handler surfaces as a
        # test failure in the handler's own ticket, not this one).
        return None
    return handler


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
        skip_reasons: list[dict[str, Any]] = []
        wanted = _resolved_scope(
            scope,
            self.dream_capabilities,
            skip_reasons=skip_reasons,
            engine_name="chronicle",
        )

        if OpKind.CHRONICLE_ABSTENTION_DRIFT_REPORT in wanted:
            engine = kb._get_engine()
            # The op only reads three threshold attrs. The
            # vectorcypher engine doesn't have them; if the orchestrator
            # routed an abstention-drift op to it, that's a misconfig -
            # raise rather than silently produce a meaningless report.
            if not hasattr(engine, "_abstention_min_top_score"):
                raise KhoraError("chronicle abstention drift requested but active engine is not a ChronicleEngine")
            op = await plan_chronicle_abstention_drift(namespace_id, engine=engine, config=config)
            ops.append(op)

        if OpKind.CHRONICLE_TOMBSTONE_AUDIT in wanted:
            coordinator = kb.storage
            async with coordinator.transaction() as txn:
                op = await plan_chronicle_tombstone_audit(namespace_id, session=txn.session, config=config)
            ops.append(op)

        return DreamPlan(
            plan_id=uuid4(),
            namespace_id=namespace_id,
            ops=tuple(ops),
            metadata={"skip_reasons": skip_reasons} if skip_reasons else {},
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
                OpKind.VECTORCYPHER_COMMUNITY_SUMMARY,
                OpKind.VECTORCYPHER_PRUNE_EDGES,
                OpKind.VECTORCYPHER_CONTRADICTION_DETECT,
                OpKind.VECTORCYPHER_NORMALIZE_SCHEMA,
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
        skip_reasons: list[dict[str, Any]] = []
        wanted = _resolved_scope(
            scope,
            self.dream_capabilities,
            skip_reasons=skip_reasons,
            engine_name="vectorcypher",
        )
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

        if OpKind.VECTORCYPHER_COMMUNITY_SUMMARY in wanted and config.community_summary_enabled:
            community_ops = await plan_vectorcypher_community_summary(
                namespace_id,
                coordinator=coordinator,
                min_size=config.community_summary_min_size,
                cooccurrence_edge_weight=config.cooccurrence_edge_weight,
                max_members_per_prompt=config.community_summary_max_members_per_prompt,
            )
            ops.extend(community_ops)

        if OpKind.VECTORCYPHER_PRUNE_EDGES in wanted and config.prune_edges_enabled:
            prune_ops = await plan_vectorcypher_prune_edges(
                namespace_id,
                coordinator=coordinator,
                target_predicates=tuple(config.prune_edges_target_predicates),
                confidence_threshold=config.prune_edges_confidence_threshold,
            )
            ops.extend(prune_ops)

        if OpKind.VECTORCYPHER_CONTRADICTION_DETECT in wanted and config.contradiction_detect_enabled:
            op = await plan_vectorcypher_contradiction_detect(
                namespace_id,
                coordinator=coordinator,
                similarity_threshold=config.contradiction_detect_similarity_threshold,
            )
            ops.append(op)

        if OpKind.VECTORCYPHER_NORMALIZE_SCHEMA in wanted:
            normalize_ops = await plan_vectorcypher_normalize_schema(
                namespace_id,
                coordinator=coordinator,
                config=config,
            )
            ops.extend(normalize_ops)

        return DreamPlan(
            plan_id=uuid4(),
            namespace_id=namespace_id,
            ops=tuple(ops),
            metadata={"skip_reasons": skip_reasons} if skip_reasons else {},
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


def _resolved_scope(
    scope: DreamScope,
    capabilities: frozenset[OpKind],
    *,
    skip_reasons: list[dict[str, Any]] | None = None,
    engine_name: str | None = None,
) -> frozenset[OpKind]:
    """Intersect requested op_kinds with the plugin's capabilities.

    When ``skip_reasons`` is supplied, every requested op kind that the
    plugin does not own is appended as
    ``{"op_kind": str, "reason": "op_not_supported_by_engine", "detail": ...}``
    and a single ``logger.warning`` is emitted naming the dropped kinds.
    This is the observability fix for #876: callers were previously
    unable to distinguish "no work needed" from "your op silently fell
    off the floor".
    """
    if scope.op_kinds is None:
        return capabilities
    requested = frozenset(scope.op_kinds)
    resolved = requested & capabilities
    if skip_reasons is not None:
        dropped = requested - capabilities
        if dropped:
            dropped_names = sorted(str(op) for op in dropped)
            engine_label = engine_name or "<unknown>"
            logger.warning(
                "dream: dropping op_kinds not supported by engine {engine}: {ops}",
                engine=engine_label,
                ops=dropped_names,
            )
            for op_kind in dropped:
                skip_reasons.append(
                    {
                        "op_kind": str(op_kind),
                        "reason": "op_not_supported_by_engine",
                        "detail": (f"engine={engine_label!r} does not list this op kind in dream_capabilities"),
                    }
                )
    return resolved


def _build_pass_through_result(plan: DreamPlan, *, mode: str) -> DreamResult:
    """Build a :class:`DreamResult` from a plan whose ops are all complete.

    Phase 1 ops carry their decision and outputs from the plan call
    itself (they're pure observation), so the apply pass just re-shapes
    the plan into a result. ``skip_reasons`` attached to
    :attr:`DreamPlan.metadata` are forwarded onto the result so callers
    can distinguish empty outcomes (no candidates, op not supported,
    guardrail tripped) from work that simply happened to be empty.
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
    skip_reasons = list(plan.metadata.get("skip_reasons", ()))
    return DreamResult(
        run=info,
        diff=DreamDiff(),
        ops=tuple(summaries.values()),
        metadata={"skip_reasons": skip_reasons},
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
    "ApplyHandler",
    "canonical_plan_payload",
    "get_apply_handler",
    "get_engine_plugin",
    "plan_hash",
]

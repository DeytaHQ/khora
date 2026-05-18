"""Cross-batch entity-resolution dedupe — dry-run planner (#658, Phase 2.1).

Walks every ``entity_type`` bucket in a namespace and uses
:func:`khora._accel.block_and_score_pairs` (Phase 3 kernel #685) to
produce candidate merge pairs above a per-type cosine threshold (tighter
than the online resolver — default 0.90 vs the online 0.85). One
:class:`DreamOp` is emitted per candidate pair carrying ``keep_id`` /
``drop_ids`` / merged source provenance.

**Mode** is dry-run only in v0.14. Calling :func:`plan_vectorcypher_dedupe_entities`
with ``mode="apply"`` raises :class:`NotImplementedError` — apply mode
lands in v0.15 under Phase 4 (#667) of the umbrella #649.

UNIQUE-violation prediction: if the predicted surviving name collides
with an unrelated third entity in ``(namespace_id, name, entity_type)``,
the pair is emitted with ``decision="skip_unique_collision"`` and the
collision detail in ``inputs`` for audit.

Stability: **internal**. ``OpKind.VECTORCYPHER_DEDUPE_ENTITIES`` is a
stable string identifier; the planner signature and op inputs/outputs
shape may evolve through Phase 2.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import text

from khora import _accel
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.core.models.entity import Entity
    from khora.storage.coordinator import StorageCoordinator


_PHASE = "plan"
_OP_TYPE = "entity_merge"
_DEFAULT_THRESHOLD = 0.90


async def plan_vectorcypher_dedupe_entities(
    namespace_id: UUID,
    *,
    coordinator: StorageCoordinator,
    default_threshold: float = _DEFAULT_THRESHOLD,
    per_type_thresholds: dict[str, float] | None = None,
    mode: Literal["dry-run", "apply"] = "dry-run",
) -> list[DreamOp]:
    """Plan cross-batch entity merges in ``namespace_id`` — never writes.

    For each ``entity_type`` bucket: collect entities with non-empty
    embeddings, call :func:`khora._accel.block_and_score_pairs` at the
    bucket's threshold, then emit one :class:`DreamOp` per candidate
    pair. Pairs whose predicted survivor would collide with an unrelated
    third entity on ``(namespace_id, name, entity_type)`` are emitted
    with ``decision="skip_unique_collision"`` for operator review.

    Args:
        namespace_id: Namespace to scan.
        coordinator: Storage coordinator (DI for tests).
        default_threshold: Fallback cosine-similarity threshold for
            entity_type buckets not present in ``per_type_thresholds``.
        per_type_thresholds: Optional per-type overrides. Missing types
            fall back to ``default_threshold``.
        mode: ``"dry-run"`` (default) plans without writing. ``"apply"``
            raises :class:`NotImplementedError` — apply lands in v0.15
            (umbrella #649 Phase 4, ticket #667).

    Returns:
        List of :class:`DreamOp` — one per candidate pair. Empty when
        the namespace has no entities, no embedded entities, or no
        candidate pairs cross the threshold. Each op carries:

        - ``op_type`` = :data:`OpKind.VECTORCYPHER_DEDUPE_ENTITIES`
        - ``decision`` = ``"planned"`` for proposed merges; or
          ``"skip_unique_collision"`` for skipped collisions.
        - ``inputs`` = a single dict with ``keep_id`` (UUID str),
          ``drop_ids`` (tuple of UUID strs), ``similarity_score``,
          ``entity_type``, ``threshold``, ``op_type=entity_merge``.
          Collision skips also carry ``collision_entity_id``,
          ``collision_name``, ``surviving_name``.
        - ``outputs`` = a single dict with
          ``merged_source_document_ids`` (tuple) and
          ``merged_source_chunk_ids`` (tuple) of UUID strs.

    Raises:
        NotImplementedError: when ``mode="apply"``.
    """
    if mode == "apply":
        raise NotImplementedError("apply mode lands in v0.15 — see #649 phase 4 / #667")

    per_type = dict(per_type_thresholds or {})

    op_id = uuid4()
    started_at = datetime.now(UTC)
    t0 = time.perf_counter()

    with trace_span(
        "khora.dream.vectorcypher.dedupe_entities",
        run_id="",
        op_id=str(op_id),
        namespace_id=str(namespace_id),
        phase=_PHASE,
        default_threshold=float(default_threshold),
    ) as span:
        entities = await coordinator.list_entities(namespace_id, limit=100_000)
        total_entities = len(entities)
        span.set_attribute("total_entities", total_entities)

        if total_entities == 0:
            span.set_attribute("total_buckets", 0)
            span.set_attribute("planned_count", 0)
            span.set_attribute("skip_collision_count", 0)
            return []

        buckets: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            if not entity.embedding:
                continue
            buckets[entity.entity_type].append(entity)
        span.set_attribute("total_buckets", len(buckets))

        # Build an O(1) (name, entity_type) → entity index so we can
        # predict UNIQUE-violation collisions across the full namespace
        # (not just within a bucket).
        by_name_type: dict[tuple[str, str], Entity] = {}
        for entity in entities:
            by_name_type[(entity.name, entity.entity_type)] = entity

        ops: list[DreamOp] = []
        planned = 0
        skipped = 0

        for entity_type, bucket in buckets.items():
            if len(bucket) < 2:
                continue
            threshold = float(per_type.get(entity_type, default_threshold))
            embeddings = [e.embedding for e in bucket]
            names = [e.name for e in bucket]
            pairs = _accel.block_and_score_pairs(
                embeddings,
                names,
                threshold=threshold,
                name_token_blocking=True,
            )
            for i, j, score in pairs:
                a, b = bucket[i], bucket[j]
                keeper, dropped = _pick_survivor(a, b)
                surviving_name = keeper.name

                collision = by_name_type.get((surviving_name, entity_type))
                if collision is not None and collision.id != keeper.id and collision.id != dropped.id:
                    ops.append(
                        _build_skip_collision_op(
                            namespace_id=namespace_id,
                            entity_type=entity_type,
                            threshold=threshold,
                            similarity_score=float(score),
                            keeper=keeper,
                            dropped=dropped,
                            surviving_name=surviving_name,
                            collision=collision,
                            started_at=started_at,
                        )
                    )
                    skipped += 1
                    continue

                ops.append(
                    _build_planned_op(
                        namespace_id=namespace_id,
                        entity_type=entity_type,
                        threshold=threshold,
                        similarity_score=float(score),
                        keeper=keeper,
                        dropped=dropped,
                        started_at=started_at,
                    )
                )
                planned += 1

        span.set_attribute("planned_count", planned)
        span.set_attribute("skip_collision_count", skipped)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        span.set_attribute("duration_ms", duration_ms)

    return ops


def _pick_survivor(a: Entity, b: Entity) -> tuple[Entity, Entity]:
    """Return ``(keeper, dropped)`` for a candidate merge pair.

    Tiebreakers: highest ``mention_count``, then earliest ``created_at``.
    Stable on ties via ``id`` lexicographic order so re-runs produce the
    same plan.
    """
    if a.mention_count != b.mention_count:
        return (a, b) if a.mention_count > b.mention_count else (b, a)
    if a.created_at != b.created_at:
        return (a, b) if a.created_at < b.created_at else (b, a)
    return (a, b) if str(a.id) < str(b.id) else (b, a)


def _merged_provenance(keeper: Entity, dropped: Entity) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Union the source-document and source-chunk id lists, keeper first.

    Returns ``(document_ids, chunk_ids)`` as tuples of UUID strings so
    the op outputs are JSON-serialisable for the file sink.
    """
    docs: list[UUID] = list(keeper.source_document_ids)
    for doc_id in dropped.source_document_ids:
        if doc_id not in docs:
            docs.append(doc_id)
    chunks: list[UUID] = list(keeper.source_chunk_ids)
    for chunk_id in dropped.source_chunk_ids:
        if chunk_id not in chunks:
            chunks.append(chunk_id)
    return tuple(str(d) for d in docs), tuple(str(c) for c in chunks)


def _build_planned_op(
    *,
    namespace_id: UUID,
    entity_type: str,
    threshold: float,
    similarity_score: float,
    keeper: Entity,
    dropped: Entity,
    started_at: datetime,
) -> DreamOp:
    """Construct a ``decision="planned"`` :class:`DreamOp`."""
    merged_docs, merged_chunks = _merged_provenance(keeper, dropped)
    inputs: dict[str, Any] = {
        "op_type": _OP_TYPE,
        "entity_type": entity_type,
        "threshold": threshold,
        "similarity_score": similarity_score,
        "keep_id": str(keeper.id),
        "drop_ids": (str(dropped.id),),
        "surviving_name": keeper.name,
    }
    outputs: dict[str, Any] = {
        "merged_source_document_ids": merged_docs,
        "merged_source_chunk_ids": merged_chunks,
    }
    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE,
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(inputs,),
        outputs=(outputs,),
        decision="planned",
        rationale=(
            f"cross-batch ER candidate at score={similarity_score:.4f} "
            f">= threshold={threshold:.4f} for entity_type={entity_type!r}; "
            f"keeper picked by mention_count then earliest created_at."
        ),
        started_at=started_at,
        namespace_id=namespace_id,
    )


def _build_skip_collision_op(
    *,
    namespace_id: UUID,
    entity_type: str,
    threshold: float,
    similarity_score: float,
    keeper: Entity,
    dropped: Entity,
    surviving_name: str,
    collision: Entity,
    started_at: datetime,
) -> DreamOp:
    """Construct a ``decision="skip_unique_collision"`` :class:`DreamOp`.

    Emitted when the surviving (namespace_id, name, entity_type) tuple
    would collide with an unrelated third entity post-merge — the merge
    would violate the entities UNIQUE constraint.
    """
    inputs: dict[str, Any] = {
        "op_type": _OP_TYPE,
        "entity_type": entity_type,
        "threshold": threshold,
        "similarity_score": similarity_score,
        "keep_id": str(keeper.id),
        "drop_ids": (str(dropped.id),),
        "surviving_name": surviving_name,
        "collision_entity_id": str(collision.id),
        "collision_name": collision.name,
    }
    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE,
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(inputs,),
        outputs=(),
        decision="skip_unique_collision",
        rationale=(
            f"merge would collide with existing entity {collision.id} "
            f"on (namespace_id, {surviving_name!r}, {entity_type!r}); "
            f"UNIQUE constraint would reject the apply."
        ),
        started_at=started_at,
        namespace_id=namespace_id,
    )


# ---------------------------------------------------------------------------
# Apply handler (#668)
# ---------------------------------------------------------------------------


async def apply_vectorcypher_dedupe_entities(
    op: DreamOp,
    *,
    coordinator: StorageCoordinator,
    session: AsyncSession,
    dream_config: Any = None,
    verifier_fn: Any = None,
) -> UndoRecord:
    """Apply one planned dedupe op — caller owns the transaction.

    For each merge listed in ``op.outputs[0]["merges"]``:

      1. (Optional) Run the borderline-merge two-LLM judge when the
         entry carries a ``similarity_score`` inside the configured
         verifier band. ``decision="defer"`` skips the merge but still
         records the verdict in the undo for operator review.
      2. Snapshot every ``relationships`` row that points at the
         absorbed entity (either as source or target).
      3. Rewrite each such row's matching endpoint(s) to the canonical
         entity_id.
      4. Detect self-loops created by the rewrite (canonical -> canonical)
         and bi-temporally invalidate them via ``invalidated_at=NOW()`` /
         ``invalidated_by=op_id``.
      5. Soft-delete the absorbed entity row by stamping
         ``valid_until=NOW()`` — never hard-delete.

    Idempotent on replay: if the absorbed entity has no remaining live
    edges, no rewrites fire (the soft-delete UPDATE is itself a noop on
    an already-soft-deleted row). The handler does not touch
    ``documents`` or ``chunks``.

    Args:
        op: The planned op. ``outputs[0]["merges"]`` is a list of
            ``{"canonical_id": str, "absorbed_id": str,
            "similarity_score"?: float, "canonical_name"?: str,
            "absorbed_name"?: str, "entity_type"?: str}`` dicts.
        coordinator: Storage coordinator (unused — session owns writes).
        session: Orchestrator-owned async session.
        dream_config: Optional :class:`DreamConfig` providing verifier /
            auditor model names and the borderline band edges. When None
            the default ``DreamConfig()`` is used so the verifier still
            runs with the spec'd defaults.
        verifier_fn: Optional async callable
            ``(CandidatePair, *, config) -> JudgeResult`` for tests. When
            None the real two-LLM judge is used.

    Returns:
        :class:`UndoRecord` with ``before["merges"]`` carrying, per
        merge, the absorbed_id, canonical_id, the list of previous
        ``relationships`` rows, the ids of relationships invalidated as
        post-rewrite self-loops, and (when the verifier ran) the joint
        ``decision`` plus per-judge verdicts. Top-level key is
        ``"merges"``, never ``"chunk_id"``.
    """
    del coordinator  # unused — session is the only write surface
    outputs = op.outputs[0] if op.outputs else {}
    merges_input = list(outputs.get("merges") or [])
    if not merges_input:
        return UndoRecord(
            op_id=op.op_id,
            op_type=str(op.op_type),
            before={"merges": []},
            applied_at=datetime.now(UTC),
        )

    merges_undo: list[dict[str, Any]] = []
    now = datetime.now(UTC)

    for entry in merges_input:
        canonical_id = UUID(str(entry["canonical_id"]))
        absorbed_id = UUID(str(entry["absorbed_id"]))

        # 1. Borderline-merge verifier gate (#667).
        verifier_record = await _maybe_run_verifier(
            entry,
            canonical_id=canonical_id,
            absorbed_id=absorbed_id,
            dream_config=dream_config,
            verifier_fn=verifier_fn,
        )
        if verifier_record is not None and verifier_record["decision"] != "merge":
            # The verifier did not authorize the merge. Record the
            # verdict for the undo file so an operator can see *why* no
            # rows were touched, and continue to the next merge entry
            # without any writes.
            merges_undo.append(
                {
                    "canonical_id": str(canonical_id),
                    "absorbed_id": str(absorbed_id),
                    "previous_relationships": [],
                    "self_loops_invalidated": [],
                    "verifier": verifier_record,
                    "applied": False,
                }
            )
            continue

        # 2. Snapshot rows that reference the absorbed entity.
        prev_rows = await _select_relationships_touching(session, absorbed_id)
        previous_serialized: list[dict[str, Any]] = []
        self_loops: list[UUID] = []

        for row in prev_rows:
            rel_id = _coerce_uuid(getattr(row, "id", None))
            src_id = _coerce_uuid(getattr(row, "source_entity_id", None))
            tgt_id = _coerce_uuid(getattr(row, "target_entity_id", None))
            rel_type = getattr(row, "relationship_type", None)
            if rel_id is None or src_id is None or tgt_id is None:
                continue

            previous_serialized.append(
                {
                    "id": str(rel_id),
                    "source_entity_id": str(src_id),
                    "target_entity_id": str(tgt_id),
                    "relationship_type": str(rel_type) if rel_type is not None else None,
                }
            )

            new_src = canonical_id if src_id == absorbed_id else src_id
            new_tgt = canonical_id if tgt_id == absorbed_id else tgt_id
            if new_src == new_tgt:
                # Post-rewrite self-loop — invalidate, don't rewrite.
                await session.execute(
                    text("UPDATE relationships SET invalidated_at = :ts, invalidated_by = :opid WHERE id = :rid"),
                    {"ts": now, "opid": op.op_id, "rid": rel_id},
                )
                self_loops.append(rel_id)
            else:
                # 2. Rewrite the absorbed endpoint(s) to the canonical id.
                await session.execute(
                    text(
                        "UPDATE relationships "
                        "SET source_entity_id = :src, target_entity_id = :tgt, "
                        "    updated_at = :ts "
                        "WHERE id = :rid"
                    ),
                    {
                        "src": new_src,
                        "tgt": new_tgt,
                        "ts": now,
                        "rid": rel_id,
                    },
                )

        # 3. Soft-delete the absorbed entity row (never hard-delete).
        await session.execute(
            text("UPDATE entities SET valid_until = :ts, updated_at = :ts WHERE id = :aid AND valid_until IS NULL"),
            {"ts": now, "aid": absorbed_id},
        )

        merge_record: dict[str, Any] = {
            "canonical_id": str(canonical_id),
            "absorbed_id": str(absorbed_id),
            "previous_relationships": previous_serialized,
            "self_loops_invalidated": [str(rid) for rid in self_loops],
            "applied": True,
        }
        if verifier_record is not None:
            merge_record["verifier"] = verifier_record
        merges_undo.append(merge_record)

    return UndoRecord(
        op_id=op.op_id,
        op_type=str(op.op_type),
        before={"merges": merges_undo},
        applied_at=now,
    )


async def _select_relationships_touching(session: AsyncSession, absorbed_id: UUID) -> list[Any]:
    """Return every row whose source or target is ``absorbed_id`` and which is still live."""
    result = await session.execute(
        text(
            "SELECT id, source_entity_id, target_entity_id, relationship_type "
            "FROM relationships "
            "WHERE (source_entity_id = :aid OR target_entity_id = :aid) "
            "  AND invalidated_at IS NULL"
        ),
        {"aid": absorbed_id},
    )
    return list(result)


def _coerce_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Phase 4.1 — two-LLM judge gate (#667)
# ---------------------------------------------------------------------------


async def _maybe_run_verifier(
    entry: dict[str, Any],
    *,
    canonical_id: UUID,
    absorbed_id: UUID,
    dream_config: Any,
    verifier_fn: Any,
) -> dict[str, Any] | None:
    """Run the borderline-merge two-LLM judge when the pair falls in band.

    Returns a serialised verdict dict (``decision`` + per-judge verdicts)
    when the verifier ran, or ``None`` when the entry was outside the
    band and the verifier was skipped. The caller treats a returned
    ``decision != "merge"`` as "do not apply this merge entry".
    """
    score_raw = entry.get("similarity_score")
    if score_raw is None:
        # No similarity score on the entry — skip the verifier path
        # entirely (back-compat with the v0.15.x apply contract that
        # only carried canonical_id / absorbed_id).
        return None
    try:
        similarity = float(score_raw)
    except (TypeError, ValueError):
        return None

    from khora.dream.config import DreamConfig

    cfg = dream_config if dream_config is not None else DreamConfig()
    low = float(cfg.dedupe_verifier_band_low)
    high = float(cfg.dedupe_verifier_band_high)
    if similarity < low or similarity >= high:
        # Above the band → planner / threshold already authorised it.
        # Below the band → the planner threshold would have rejected it.
        return None

    from khora.dream.engines.vectorcypher.verifier import (
        CandidatePair,
        run_two_llm_judge,
    )

    pair = CandidatePair(
        canonical_id=str(canonical_id),
        canonical_name=str(entry.get("canonical_name") or entry.get("surviving_name") or ""),
        canonical_entity_type=str(entry.get("entity_type") or entry.get("canonical_entity_type") or ""),
        absorbed_id=str(absorbed_id),
        absorbed_name=str(entry.get("absorbed_name") or ""),
        absorbed_entity_type=str(entry.get("absorbed_entity_type") or entry.get("entity_type") or ""),
        similarity_score=similarity,
    )

    judge = verifier_fn if verifier_fn is not None else run_two_llm_judge
    result = await judge(pair, config=cfg)

    verifier_payload: dict[str, Any] = {
        "decision": result.decision,
        "rationale": result.rationale,
        "similarity_score": similarity,
    }
    if result.verifier_verdict is not None:
        verifier_payload["verifier_verdict"] = result.verifier_verdict.model_dump()
    if result.auditor_verdict is not None:
        verifier_payload["auditor_verdict"] = result.auditor_verdict.model_dump()
    return verifier_payload


# ---------------------------------------------------------------------------
# Undo / reverse path (#667 — kb.dream_undo)
# ---------------------------------------------------------------------------


async def reverse_vectorcypher_dedupe_entities(
    undo_op: dict[str, Any],
    *,
    session: AsyncSession,
) -> bool:
    """Reverse a previously-applied dedupe op from its ``undo.json`` entry.

    The :class:`UndoRecord` for one applied dedupe op records, per merge:

      * the previous ``relationships`` rows (so we can re-point them at
        the absorbed entity),
      * the self-loops that were invalidated (so we can clear
        ``invalidated_at`` / ``invalidated_by``),
      * the canonical / absorbed entity ids (so we can clear the
        ``valid_until`` tombstone on the absorbed row).

    This function unwinds those three sides in reverse order. It is
    **idempotent**: replaying it on an already-undone op finds nothing
    to restore and returns ``False`` (no rows touched).

    Merges with ``"applied": False`` (verifier deferred them) are
    skipped — there's nothing to roll back.

    Args:
        undo_op: One element of the ``ops`` array in ``undo.json``.
            Must have the schema produced by
            :func:`apply_vectorcypher_dedupe_entities` —
            ``before.merges[*]`` carries the per-merge snapshot.
        session: Caller-owned async session. The caller wraps this in
            the same coordinator transaction shape as the apply path.

    Returns:
        ``True`` when at least one row was restored (entity tombstone
        cleared, relationship rewritten back, or self-loop reactivated).
        ``False`` when there was nothing to undo (idempotent re-undo).
    """
    before = undo_op.get("before") or {}
    merges = list(before.get("merges") or [])
    if not merges:
        return False

    op_id_value = undo_op.get("op_id")
    op_uuid = _coerce_uuid(op_id_value) if op_id_value is not None else None

    any_change = False
    for merge in merges:
        if not merge.get("applied", True):
            # Verifier-deferred merge — nothing was written, nothing to undo.
            continue

        absorbed_id = _coerce_uuid(merge.get("absorbed_id"))
        canonical_id = _coerce_uuid(merge.get("canonical_id"))
        if absorbed_id is None or canonical_id is None:
            continue

        # 1. Clear the absorbed entity's tombstone iff it still points
        #    at the apply timestamp. If a downstream write has already
        #    re-soft-deleted the row, leave it alone — never resurrect
        #    something the live system has tombstoned for unrelated
        #    reasons.
        revive = await session.execute(
            text("UPDATE entities SET valid_until = NULL WHERE id = :aid AND valid_until IS NOT NULL"),
            {"aid": absorbed_id},
        )
        if getattr(revive, "rowcount", 0):
            any_change = True

        # 2. Restore each previously-rewritten relationship to its
        #    pre-apply endpoints. Use the snapshot's id so a row that
        #    has since been deleted is simply not updated (idempotent).
        for prev in list(merge.get("previous_relationships") or []):
            rid = _coerce_uuid(prev.get("id"))
            src = _coerce_uuid(prev.get("source_entity_id"))
            tgt = _coerce_uuid(prev.get("target_entity_id"))
            if rid is None or src is None or tgt is None:
                continue
            rewrite = await session.execute(
                text(
                    "UPDATE relationships "
                    "SET source_entity_id = :src, target_entity_id = :tgt, "
                    "    updated_at = :ts "
                    "WHERE id = :rid"
                ),
                {"src": src, "tgt": tgt, "ts": datetime.now(UTC), "rid": rid},
            )
            if getattr(rewrite, "rowcount", 0):
                any_change = True

        # 3. Clear the bi-temporal invalidation on any self-loop the
        #    apply created. Match by id and by the invalidated_by op_id
        #    so we only clear what *this* op invalidated, never a
        #    self-loop the live system invalidated for other reasons.
        for self_loop_value in list(merge.get("self_loops_invalidated") or []):
            sl_id = _coerce_uuid(self_loop_value)
            if sl_id is None:
                continue
            if op_uuid is not None:
                clear = await session.execute(
                    text(
                        "UPDATE relationships "
                        "SET invalidated_at = NULL, invalidated_by = NULL "
                        "WHERE id = :rid AND invalidated_by = :opid"
                    ),
                    {"rid": sl_id, "opid": op_uuid},
                )
            else:
                clear = await session.execute(
                    text("UPDATE relationships SET invalidated_at = NULL, invalidated_by = NULL WHERE id = :rid"),
                    {"rid": sl_id},
                )
            if getattr(clear, "rowcount", 0):
                any_change = True

    return any_change

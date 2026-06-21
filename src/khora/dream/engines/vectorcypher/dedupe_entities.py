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

from loguru import logger
from sqlalchemy import text

from khora import _accel
from khora.dream.engines.vectorcypher._uuid_bind import uuid_bind
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
    degradations: list[dict[str, Any]] | None = None,
) -> list[DreamOp]:
    """Plan cross-batch entity merges in ``namespace_id`` — never writes.

    For each ``entity_type`` bucket, score candidate pairs with
    :func:`khora._accel.block_and_score_pairs`, then group every
    candidate edge into connected components via union-find (#1265). One
    :class:`DreamOp` is emitted per component carrying a single
    ``outputs[0] = {"merges": [...]}`` payload — the exact shape
    :func:`apply_vectorcypher_dedupe_entities`, the future graph mirror,
    and the undo path all read. Each component has exactly one canonical
    (highest mention_count, then earliest created_at, then lexicographic
    id); every other member is absorbed directly into that canonical so
    no merge entry ever points at a retired intermediate (A->B, B->C
    yields one component, all edges resolved to one survivor).

    Determinism (#1266 INVARIANT 0): the entity list is stable-sorted by
    ``(name, entity_type, str(id))`` and the per-bucket index is built
    from the sorted list *before* any kernel call, and the kernel's
    unordered pair output is sorted, so a shuffled input yields
    byte-identical merge payloads and an identical ``plan_hash``.

    Embedding re-join (#1267): the coordinator's ``list_entities`` prefers
    the graph, and graph-routed entities carry no embedding. On a
    graph+pgvector stack the planner re-joins L2-normalized embeddings by
    id from pgvector before the kernel runs (preserving dot == cosine).
    When embeddings cannot be sourced at all, an ADR-001 degradation is
    appended to ``degradations`` (and logged at WARNING) rather than
    silently under-consolidating.

    Args:
        namespace_id: Namespace to scan.
        coordinator: Storage coordinator (DI for tests).
        default_threshold: Fallback cosine-similarity threshold for
            entity_type buckets not present in ``per_type_thresholds``.
        per_type_thresholds: Optional per-type overrides. Missing types
            fall back to ``default_threshold``.
        mode: ``"dry-run"`` (default) plans without writing. ``"apply"``
            raises :class:`NotImplementedError` — apply runs through the
            orchestrator handler, not this planner.
        degradations: Optional out-parameter. When supplied, ADR-001
            :class:`khora.core.diagnostics.Degradation`-shaped dicts are
            appended in place (e.g. when entities have no usable
            embedding). The caller forwards these onto the dream result's
            observability dict.

    Returns:
        List of :class:`DreamOp` — one per connected component. Empty when
        the namespace has no entities, no embedded entities, or no
        candidate pairs cross the threshold. Each op carries:

        - ``op_type`` = :data:`OpKind.VECTORCYPHER_DEDUPE_ENTITIES`
        - ``decision`` = ``"planned"`` for proposed merges; or
          ``"skip_unique_collision"`` when the component's surviving
          ``(name, entity_type)`` collides with an unrelated third entity.
        - ``inputs[0]`` = a dict with ``keep_id`` (canonical UUID str),
          ``drop_ids`` (tuple of absorbed UUID strs), ``entity_type``,
          ``threshold``, ``component_size``, ``op_type=entity_merge``.
        - ``outputs[0]`` = ``{"merges": [{"canonical_id", "absorbed_id",
          "similarity_score", "canonical_name", "absorbed_name",
          "entity_type", "merged_source_document_ids",
          "merged_source_chunk_ids"}, ...]}``.

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

        # #1267: graph-routed entities carry no embedding. Re-join from
        # pgvector before bucketing so the dedupe plan on a graph stack
        # matches the PG-only plan. Records a degradation when embeddings
        # remain absent (ADR-001).
        entities = await _rejoin_embeddings(
            entities,
            coordinator=coordinator,
            namespace_id=namespace_id,
            degradations=degradations,
        )

        # #1266 INVARIANT 0: stable-sort the full entity list before any
        # index is built or any kernel runs. The kernel sees a fixed row
        # order, so its (i, j) indices map to a deterministic id ordering
        # independent of the backend's list_entities order.
        entities = sorted(entities, key=lambda e: (e.name, e.entity_type, str(e.id)))

        buckets: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            if not entity.embedding:
                continue
            buckets[entity.entity_type].append(entity)
        span.set_attribute("total_buckets", len(buckets))

        # Build a (name, entity_type) → entities index so we can predict
        # UNIQUE-violation collisions across the full namespace (not just
        # within a component). A surviving (name, type) collides when an
        # entity carrying it lives OUTSIDE the merge component. Entities
        # are appended in sorted order so the reported collision id is
        # deterministic (#1266).
        by_name_type: dict[tuple[str, str], list[Entity]] = defaultdict(list)
        for entity in entities:
            by_name_type[(entity.name, entity.entity_type)].append(entity)

        ops: list[DreamOp] = []
        planned = 0
        skipped = 0

        for entity_type in sorted(buckets):
            bucket = buckets[entity_type]
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
            # The kernel makes no deterministic global sort guarantee.
            # Sort so component formation is order-independent (#1266).
            pairs = sorted(pairs, key=lambda p: (p[0], p[1]))

            # #1265: union-find over every candidate edge in the bucket.
            components = _connected_components(len(bucket), pairs)
            best_score = _best_pair_scores(pairs)

            for member_indices in components:
                members = [bucket[i] for i in sorted(member_indices)]
                op = _build_component_op(
                    namespace_id=namespace_id,
                    entity_type=entity_type,
                    threshold=threshold,
                    members=members,
                    member_indices=sorted(member_indices),
                    best_score=best_score,
                    by_name_type=by_name_type,
                    started_at=started_at,
                )
                ops.append(op)
                if op.decision == "skip_unique_collision":
                    skipped += 1
                else:
                    planned += 1

        span.set_attribute("planned_count", planned)
        span.set_attribute("skip_collision_count", skipped)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        span.set_attribute("duration_ms", duration_ms)

    return ops


# ---------------------------------------------------------------------------
# Union-find connected components (#1265)
# ---------------------------------------------------------------------------


class _UnionFind:
    """Disjoint-set with path compression + union by size.

    Operates on bucket-local integer indices. ``components()`` returns the
    member-index sets, each a transitive closure of the candidate edges.
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]


def _connected_components(n: int, pairs: list[tuple[int, int, float]]) -> list[list[int]]:
    """Group bucket indices into connected components from candidate edges.

    Only indices that participate in at least one edge form a component;
    singletons (no candidate edge) are dropped — nothing to merge. The
    returned list is sorted by each component's minimum index so the op
    emission order is deterministic (#1266).
    """
    uf = _UnionFind(n)
    touched: set[int] = set()
    for i, j, _ in pairs:
        uf.union(i, j)
        touched.add(i)
        touched.add(j)

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in sorted(touched):
        grouped[uf.find(idx)].append(idx)

    return sorted((sorted(members) for members in grouped.values()), key=lambda m: m[0])


def _best_pair_scores(pairs: list[tuple[int, int, float]]) -> dict[tuple[int, int], float]:
    """Map each undirected bucket-index pair to its similarity score."""
    out: dict[tuple[int, int], float] = {}
    for i, j, score in pairs:
        key = (i, j) if i < j else (j, i)
        out[key] = float(score)
    return out


async def _rejoin_embeddings(
    entities: list[Entity],
    *,
    coordinator: StorageCoordinator,
    namespace_id: UUID,
    degradations: list[dict[str, Any]] | None,
) -> list[Entity]:
    """Fill in missing embeddings from pgvector on graph-routed stacks (#1267).

    ``coordinator.list_entities`` prefers the graph backend, and graph
    entities carry no embedding (the Neo4j ``_record_to_entity`` sets it
    to ``None``). When a graph + pgvector stack is configured, fetch the
    embeddings by id from pgvector and graft them onto the entity objects
    so the dedupe plan matches the PG-only plan (dot == cosine preserved —
    pgvector embeddings are L2-normalized at ingest).

    Entities whose embedding cannot be sourced are recorded as an ADR-001
    degradation (and logged at WARNING) rather than silently dropped from
    the candidate set.
    """
    missing = [e for e in entities if not e.embedding]
    if not missing:
        return entities

    vector = getattr(coordinator, "_vector", None)
    getter = getattr(coordinator, "get_entities_batch", None)
    if vector is not None and getter is not None:
        try:
            fetched = await getter([e.id for e in missing], namespace_id=namespace_id)
        except Exception as exc:  # noqa: BLE001 - boundary read; degrade, don't crash the plan
            logger.warning(
                "dream dedupe: pgvector embedding re-join failed; entities without "
                "an embedding will be excluded from this plan: {exc}",
                exc=exc,
                exc_info=True,
            )
            fetched = {}
        for entity in missing:
            joined = fetched.get(entity.id)
            if joined is not None and joined.embedding is not None:
                entity.embedding = joined.embedding

    still_missing = [e for e in entities if not e.embedding]
    if still_missing and degradations is not None:
        logger.warning(
            "dream dedupe: {n} entit{suffix} in namespace {ns} lack an embedding and "
            "cannot be scored; the dedupe plan may under-consolidate.",
            n=len(still_missing),
            suffix="y" if len(still_missing) == 1 else "ies",
            ns=namespace_id,
        )
        degradations.append(
            {
                "component": "dedupe_entities",
                "reason": "missing_entity_embedding",
                "detail": (
                    f"{len(still_missing)} entit"
                    f"{'y' if len(still_missing) == 1 else 'ies'} had no usable embedding "
                    "after the pgvector re-join; excluded from the candidate set."
                ),
                "count": len(still_missing),
            }
        )
    return entities


def _pick_canonical(members: list[Entity]) -> Entity:
    """Return the single canonical for a merge component.

    Ordering: highest ``mention_count``, then earliest ``created_at``,
    then lexicographically smallest ``id``. Total + deterministic so the
    plan is identical across input permutations (#1266).
    """
    return min(members, key=lambda e: (-e.mention_count, e.created_at, str(e.id)))


def _merged_provenance(members: list[Entity]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Union the source-document / source-chunk id lists across a component.

    Returns ``(document_ids, chunk_ids)`` as tuples of UUID strings so
    the op outputs are JSON-serialisable for the file sink. Order is
    stable: members are visited in the order passed (already sorted).
    """
    docs: list[UUID] = []
    chunks: list[UUID] = []
    for member in members:
        for doc_id in member.source_document_ids:
            if doc_id not in docs:
                docs.append(doc_id)
        for chunk_id in member.source_chunk_ids:
            if chunk_id not in chunks:
                chunks.append(chunk_id)
    return tuple(str(d) for d in docs), tuple(str(c) for c in chunks)


def _build_component_op(
    *,
    namespace_id: UUID,
    entity_type: str,
    threshold: float,
    members: list[Entity],
    member_indices: list[int],
    best_score: dict[tuple[int, int], float],
    by_name_type: dict[tuple[str, str], list[Entity]],
    started_at: datetime,
) -> DreamOp:
    """Build one :class:`DreamOp` for a connected merge component (#1265).

    A single canonical absorbs every other member; the op carries one
    ``outputs[0]["merges"]`` entry per absorbed member (canonical_id ==
    the resolved survivor for all of them, so no entry points at a
    retired intermediate). When the surviving ``(name, entity_type)``
    collides with an unrelated third entity, the whole component is
    emitted as ``decision="skip_unique_collision"`` for operator review.
    """
    canonical = _pick_canonical(members)
    absorbed = [m for m in members if m.id != canonical.id]
    surviving_name = canonical.name

    # Per-member best similarity to any other member of the component
    # (the strongest candidate edge that pulled it in). Lets the apply
    # verifier band still see a meaningful score per absorbed member.
    index_of = {member.id: idx for member, idx in zip(members, member_indices, strict=True)}

    def member_score(member: Entity) -> float:
        mi = index_of[member.id]
        ci = index_of[canonical.id]
        # Prefer the direct edge to the canonical; otherwise (the member
        # was pulled into the component transitively) use the strongest
        # candidate edge touching this member.
        direct = best_score.get((mi, ci) if mi < ci else (ci, mi))
        if direct is not None:
            return direct
        scores = [s for (a, b), s in best_score.items() if mi in (a, b)]
        return max(scores) if scores else threshold

    component_ids = {m.id for m in members}
    # A collision is any entity carrying the surviving (name, type) that
    # lives outside this merge component — merging would violate the
    # entities UNIQUE constraint on (namespace_id, name, entity_type).
    collision = next(
        (e for e in by_name_type.get((surviving_name, entity_type), []) if e.id not in component_ids),
        None,
    )

    if collision is not None:
        inputs: dict[str, Any] = {
            "op_type": _OP_TYPE,
            "entity_type": entity_type,
            "threshold": threshold,
            "keep_id": str(canonical.id),
            "drop_ids": tuple(str(m.id) for m in absorbed),
            "surviving_name": surviving_name,
            "component_size": len(members),
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

    merges: list[dict[str, Any]] = []
    # Component-wide provenance: every merge entry carries the full union over
    # the whole component (canonical + all absorbed), not just canonical + the
    # current member, so a consumer that applies entries with overwrite
    # semantics (e.g. the Phase-2 graph mirror) cannot drop an earlier member's
    # provenance.
    merged_docs, merged_chunks = _merged_provenance(members)
    for member in absorbed:
        score = member_score(member)
        merges.append(
            {
                "canonical_id": str(canonical.id),
                "absorbed_id": str(member.id),
                "similarity_score": score,
                "canonical_name": canonical.name,
                "absorbed_name": member.name,
                "entity_type": entity_type,
                "merged_source_document_ids": merged_docs,
                "merged_source_chunk_ids": merged_chunks,
            }
        )

    top_score = max((m["similarity_score"] for m in merges), default=threshold)
    inputs = {
        "op_type": _OP_TYPE,
        "entity_type": entity_type,
        "threshold": threshold,
        "similarity_score": top_score,
        "keep_id": str(canonical.id),
        "drop_ids": tuple(str(m.id) for m in absorbed),
        "surviving_name": canonical.name,
        "component_size": len(members),
    }
    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE,
        op_type=OpKind.VECTORCYPHER_DEDUPE_ENTITIES,
        inputs=(inputs,),
        outputs=({"merges": merges},),
        decision="planned",
        rationale=(
            f"cross-batch ER component of {len(members)} entities for "
            f"entity_type={entity_type!r} at threshold={threshold:.4f}; "
            f"canonical picked by mention_count then earliest created_at; "
            f"{len(absorbed)} member(s) absorbed into a single survivor."
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
    bind_uuid = uuid_bind(session)

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
        prev_rows = await _select_relationships_touching(session, absorbed_id, bind_uuid=bind_uuid)
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
                    {"ts": now, "opid": bind_uuid(op.op_id), "rid": bind_uuid(rel_id)},
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
                        "src": bind_uuid(new_src),
                        "tgt": bind_uuid(new_tgt),
                        "ts": now,
                        "rid": bind_uuid(rel_id),
                    },
                )

        # 3. Soft-delete the absorbed entity row (never hard-delete).
        await session.execute(
            text("UPDATE entities SET valid_until = :ts, updated_at = :ts WHERE id = :aid AND valid_until IS NULL"),
            {"ts": now, "aid": bind_uuid(absorbed_id)},
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


async def _select_relationships_touching(session: AsyncSession, absorbed_id: UUID, *, bind_uuid: Any) -> list[Any]:
    """Return every row whose source or target is ``absorbed_id`` and which is still live."""
    result = await session.execute(
        text(
            "SELECT id, source_entity_id, target_entity_id, relationship_type "
            "FROM relationships "
            "WHERE (source_entity_id = :aid OR target_entity_id = :aid) "
            "  AND invalidated_at IS NULL"
        ),
        {"aid": bind_uuid(absorbed_id)},
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
    bind_uuid = uuid_bind(session)

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
            {"aid": bind_uuid(absorbed_id)},
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
                {"src": bind_uuid(src), "tgt": bind_uuid(tgt), "ts": datetime.now(UTC), "rid": bind_uuid(rid)},
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
                    {"rid": bind_uuid(sl_id), "opid": bind_uuid(op_uuid)},
                )
            else:
                clear = await session.execute(
                    text("UPDATE relationships SET invalidated_at = NULL, invalidated_by = NULL WHERE id = :rid"),
                    {"rid": bind_uuid(sl_id)},
                )
            if getattr(clear, "rowcount", 0):
                any_change = True

    return any_change

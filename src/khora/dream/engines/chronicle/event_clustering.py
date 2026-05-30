"""Chronicle ``chronicle_events`` near-duplicate clustering (#665, Phase 2.5 of #649).

Pure plan-builder. For each ``(namespace_id, subject)`` bucket, cluster
events whose SVO-summary embeddings are within
``DreamConfig.event_clustering_cosine_threshold`` cosine of one another
*and* whose ``referenced_date`` values fall within
``DreamConfig.event_clustering_window_days`` of each other. Pick the
highest-confidence event in each cluster as canonical (ties broken by
``observation_date`` descending, then ``id`` ascending — deterministic).

The op emits one :class:`DreamOp` per cluster with
``op_type=OpKind.CHRONICLE_EVENT_CLUSTERING`` and
``decision="planned"``. Its ``inputs`` carry the canonical id and the
full merged-id list; ``outputs`` carry the *planned* merged
``source_chunk_ids`` count and the retained ``observation_date`` list.

**Critical invariant — never mutate ``chronicle_events.chunk_id``.**

The temporal recall channel (see ``engine.py:1620``) dedupes events by
``chunk_id`` and surfaces the back-pointer to the chunk that owns the
event. Mutating ``chunk_id`` on the canonical row breaks that linkage.
The planner only records an aggregate ``source_chunk_ids`` count —
never a write directive against the FK column. The apply path
(:func:`apply_chronicle_event_clustering`) only touches three columns:
``invalidated_at`` / ``invalidated_by`` / ``merged_into_event_id``
introduced in migration 034.

Apply mode (Phase 4, #669) soft-merges tails into the canonical event
via the bi-temporal columns. Pre-existing ``apply_event_clustering``
shim is gone — callers go through
:func:`apply_chronicle_event_clustering`.

Span: ``khora.dream.chronicle.event_clustering`` (internal stability).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora._accel import batch_dot_product
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.dream.config import DreamConfig


_PHASE = "plan"
_OP_SPAN = "khora.dream.chronicle.event_clustering"
_APPLY_SPAN = "khora.dream.chronicle.event_clustering.apply"
# Columns the apply path is allowed to touch. Any UPDATE that mentions a
# column NOT in this set is a bug — chunk_id in particular is load-bearing
# for the temporal-recall back-pointer (see engine.py:1620).
_ALLOWED_UPDATE_COLUMNS = frozenset({"invalidated_at", "invalidated_by", "merged_into_event_id"})
_FORBIDDEN_UPDATE_COLUMNS = frozenset({"chunk_id", "id", "namespace_id", "subject"})


async def plan_chronicle_event_clustering(
    namespace_id: UUID,
    *,
    session: AsyncSession,
    config: DreamConfig,
    _skip_reasons: list[dict[str, Any]] | None = None,
) -> tuple[DreamOp, ...]:
    """Plan near-duplicate clustering of ``chronicle_events``.

    Returns one :class:`DreamOp` per cluster found. When no clusters
    form, the return value is an empty tuple - the orchestrator records
    the planner ran but produced nothing to apply.

    The planner is pure SELECT; it never inserts, updates, or deletes a
    row. Cosine similarity is evaluated only within
    ``(namespace_id, subject)`` buckets so the O(N^2) inner loop stays
    bounded by per-subject event volume.

    When ``_skip_reasons`` is supplied (callers wiring the planner into
    the chronicle plugin's ``plan_dream`` so result observability survives
    an empty plan, see #876), this function appends a single
    ``{"op_kind": "chronicle_event_clustering", "reason": "no_candidates",
    "detail": ...}`` entry when ``_fetch_events`` returned no rows. The
    underscore prefix marks the kwarg as orchestrator-internal: existing
    test call sites that don't pass it observe the unchanged
    ``tuple[DreamOp, ...]`` return shape.
    """
    threshold = float(config.event_clustering_cosine_threshold)
    window_days = int(config.event_clustering_window_days)

    rows = await _fetch_events(session, namespace_id)

    with trace_span(
        _OP_SPAN,
        namespace_id=str(namespace_id),
        threshold=threshold,
        window_days=window_days,
        event_count=len(rows),
    ):
        clusters = _cluster_events(rows, threshold=threshold, window_days=window_days)

    ops: list[DreamOp] = []
    for cluster in clusters:
        ops.append(_build_op(cluster, namespace_id=namespace_id))
    if _skip_reasons is not None and not ops:
        # Distinguish "no rows at fetch time" from "rows existed but no
        # cluster of >=2 formed within threshold + window". Both surface
        # as no_candidates for the caller - the detail string carries the
        # finer-grained signal.
        detail = (
            "no chronicle_events rows with embedding + referenced_date"
            if not rows
            else "no near-duplicate clusters of size >= 2 within threshold + window"
        )
        _skip_reasons.append(
            {
                "op_kind": OpKind.CHRONICLE_EVENT_CLUSTERING.value,
                "reason": "no_candidates",
                "detail": detail,
            }
        )
    return tuple(ops)


async def apply_chronicle_event_clustering(
    op: DreamOp,
    *,
    coordinator: Any = None,
    session: AsyncSession,
) -> UndoRecord:
    """Apply one event-clustering DreamOp via bi-temporal soft-merge.

    Reads ``op.inputs[0]["canonical_id"]`` (the canonical event chosen
    by the planner) and ``op.inputs[0]["merged_event_ids"]`` (all
    cluster members including the canonical). For every tail
    (everything except the canonical), UPDATEs the migration-034
    columns:

    * ``invalidated_at`` = NOW()
    * ``invalidated_by`` = ``op.op_id``
    * ``merged_into_event_id`` = canonical_id

    The UPDATE is gated by ``invalidated_at IS NULL`` so re-applying
    the same op is a no-op (idempotency).

    Returns an :class:`UndoRecord` whose ``before`` payload uses the
    top-level key ``"clusters"`` (NEVER ``"chunk_id"``) and records the
    pre-apply state of each tail so a later revert can restore it.

    The ``coordinator`` argument is accepted to match the protocol
    shared with other apply handlers (some need it for multi-backend
    transactions); event-clustering only touches the SQL session and
    so receives it for symmetry.
    """
    payload = op.inputs[0] if op.inputs else {}
    canonical_id = _as_uuid(payload["canonical_id"])
    merged_ids = [_as_uuid(eid) for eid in payload.get("merged_event_ids", ())]
    tail_ids = [eid for eid in merged_ids if eid != canonical_id]

    with trace_span(
        _APPLY_SPAN,
        op_id=str(op.op_id),
        canonical_id=str(canonical_id),
        tail_count=len(tail_ids),
    ):
        previous_states = await _capture_previous_states(session, tail_ids)

        if tail_ids:
            await _soft_merge_tails(
                session,
                canonical_id=canonical_id,
                tail_ids=tail_ids,
                op_id=op.op_id,
            )

    return UndoRecord(
        op_id=op.op_id,
        op_type=OpKind.CHRONICLE_EVENT_CLUSTERING.value,
        before={
            "clusters": [
                {
                    "canonical_id": str(canonical_id),
                    "tail_ids": [str(t) for t in tail_ids],
                    "previous_states": previous_states,
                }
            ]
        },
        applied_at=datetime.now(UTC),
    )


# Compile-time guard for the column-allowlist invariant. Forbidden columns
# must not intersect the allowed set; if a refactor accidentally moves a
# load-bearing FK into the allow set, this trips at import time.
assert _ALLOWED_UPDATE_COLUMNS.isdisjoint(_FORBIDDEN_UPDATE_COLUMNS), (
    "apply path's allow-set overlaps the forbidden-set — chunk_id, id, namespace_id, and subject must never be mutated"
)


async def _capture_previous_states(session: AsyncSession, tail_ids: list[UUID]) -> list[dict[str, Any]]:
    """Snapshot the soft-delete columns for each tail before the update.

    Used by the returned UndoRecord so a revert can restore the
    pre-apply state. Only the three migration-034 columns are
    snapshotted — by construction, the apply path cannot have touched
    anything else.
    """
    if not tail_ids:
        return []
    bound = [_bind_uuid(session, t) for t in tail_ids]
    # ``placeholders`` is a programmatically-generated bind-param list
    # (``:t0, :t1, ...``); no user input flows into the SQL string. The
    # actual UUID values bind via the ``params`` dict below.
    placeholders = ", ".join(f":t{i}" for i in range(len(bound)))
    params = {f"t{i}": v for i, v in enumerate(bound)}
    stmt = text(
        "SELECT id, invalidated_at, invalidated_by, merged_into_event_id "  # noqa: S608
        f"FROM chronicle_events WHERE id IN ({placeholders})"
    )
    result = await session.execute(stmt, params)
    snapshots: list[dict[str, Any]] = []
    for r in result.all():
        snapshots.append(
            {
                "id": str(_as_uuid(r.id)),
                "invalidated_at": _iso_or_none(r.invalidated_at),
                "invalidated_by": _str_or_none(r.invalidated_by),
                "merged_into_event_id": _str_or_none(r.merged_into_event_id),
            }
        )
    return snapshots


async def _soft_merge_tails(
    session: AsyncSession,
    *,
    canonical_id: UUID,
    tail_ids: list[UUID],
    op_id: UUID,
) -> None:
    """Issue the UPDATE that soft-merges each tail into the canonical.

    Per-row UPDATE (instead of a single ``WHERE id IN (...)``) because
    SQLite's ``CURRENT_TIMESTAMP`` resolves to a single value per
    statement and the test fixture relies on per-row timestamps for
    deterministic ordering checks. The N is bounded by cluster size —
    typically <10. Postgres N=many UPDATE is identical in cost.

    Column-allowlist invariant: the SET clause references only
    ``invalidated_at`` / ``invalidated_by`` / ``merged_into_event_id``.
    Compile-time-asserted via :data:`_ALLOWED_UPDATE_COLUMNS`.
    """
    stmt = text(
        "UPDATE chronicle_events SET "
        "invalidated_at = CURRENT_TIMESTAMP, "
        "invalidated_by = :op_id, "
        "merged_into_event_id = :canonical "
        "WHERE id = :tail_id AND invalidated_at IS NULL"
    )
    canonical_bind = _bind_uuid(session, canonical_id)
    op_bind = _bind_uuid(session, op_id)
    for tail in tail_ids:
        await session.execute(
            stmt,
            {
                "op_id": op_bind,
                "canonical": canonical_bind,
                "tail_id": _bind_uuid(session, tail),
            },
        )


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# Row fetch
# ---------------------------------------------------------------------------


async def _fetch_events(session: AsyncSession, namespace_id: UUID) -> list[_EventRow]:
    """Pull every row that can participate in clustering.

    Rows missing an ``embedding`` or a ``referenced_date`` cannot be
    cosine-scored or date-windowed and so cannot cluster — they're
    filtered out at fetch time rather than in the inner loop.
    """
    stmt = text(
        """
        SELECT id, chunk_id, subject, verb, object, referenced_date,
               observation_date, confidence, embedding
        FROM chronicle_events
        WHERE namespace_id = :ns
          AND embedding IS NOT NULL
          AND referenced_date IS NOT NULL
        """
    )
    result = await session.execute(stmt, {"ns": _bind_uuid(session, namespace_id)})
    rows: list[_EventRow] = []
    for r in result.all():
        emb = _coerce_embedding(r.embedding)
        if emb is None:
            continue
        rows.append(
            _EventRow(
                event_id=_as_uuid(r.id),
                chunk_id=_as_uuid(r.chunk_id),
                subject=str(r.subject),
                referenced_date=_as_aware(r.referenced_date),
                observation_date=_as_aware(r.observation_date),
                confidence=float(r.confidence),
                embedding=emb,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class _EventRow:
    """Compact view of one chronicle_events row used by the clusterer."""

    __slots__ = (
        "event_id",
        "chunk_id",
        "subject",
        "referenced_date",
        "observation_date",
        "confidence",
        "embedding",
    )

    def __init__(
        self,
        *,
        event_id: UUID,
        chunk_id: UUID,
        subject: str,
        referenced_date: datetime,
        observation_date: datetime,
        confidence: float,
        embedding: list[float],
    ) -> None:
        self.event_id = event_id
        self.chunk_id = chunk_id
        self.subject = subject
        self.referenced_date = referenced_date
        self.observation_date = observation_date
        self.confidence = confidence
        self.embedding = embedding


def _cluster_events(rows: list[_EventRow], *, threshold: float, window_days: int) -> list[list[_EventRow]]:
    """Group rows into near-duplicate clusters.

    Per ``(namespace_id, subject)`` bucket: union-find using a single
    Rust-accelerated batched dot product per anchor row. Two rows merge
    when both:

    1. cosine(a.embedding, b.embedding) >= threshold (embeddings are
       L2-normalized at ingest, so dot product == cosine), AND
    2. |a.referenced_date - b.referenced_date| <= window_days.

    Singleton clusters (size 1) are dropped — they don't represent a
    near-duplicate.
    """
    by_subject: dict[str, list[_EventRow]] = defaultdict(list)
    for row in rows:
        by_subject[row.subject].append(row)

    window = timedelta(days=window_days)
    clusters: list[list[_EventRow]] = []
    for bucket in by_subject.values():
        n = len(bucket)
        if n < 2:
            continue
        parent = list(range(n))

        def find(i: int, p: list[int] = parent) -> int:
            while p[i] != i:
                p[i] = p[p[i]]
                i = p[i]
            return i

        def union(a: int, b: int, p: list[int] = parent) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                p[ra] = rb

        candidate_embeddings = [row.embedding for row in bucket]
        for i in range(n):
            row_i = bucket[i]
            scored = batch_dot_product(row_i.embedding, candidate_embeddings, threshold)
            for j, _score in scored:
                if j <= i:
                    continue
                row_j = bucket[j]
                if abs(row_i.referenced_date - row_j.referenced_date) <= window:
                    union(i, j)

        groups: dict[int, list[_EventRow]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(bucket[i])
        for members in groups.values():
            if len(members) >= 2:
                clusters.append(members)
    return clusters


# ---------------------------------------------------------------------------
# Op construction
# ---------------------------------------------------------------------------


def _build_op(cluster: list[_EventRow], *, namespace_id: UUID) -> DreamOp:
    """Build the per-cluster :class:`DreamOp`.

    The canonical row is the highest-confidence member; ties resolve to
    the most-recently observed event, then to the lowest UUID — making
    the planner deterministic for a fixed input set.

    The op's ``inputs`` carry the canonical id and the full member id
    list; ``outputs`` carry merged-list summaries (chunk count, retained
    observation dates). Neither side proposes touching the canonical
    row's ``chunk_id`` column — that FK stays put.
    """
    canonical = max(
        cluster,
        key=lambda r: (r.confidence, r.observation_date, -int(r.event_id.int)),
    )
    merged_ids = sorted((r.event_id for r in cluster), key=lambda u: u.int)
    chunk_ids = sorted({r.chunk_id for r in cluster}, key=lambda u: u.int)
    retained_dates = sorted({r.observation_date for r in cluster})

    started_at = datetime.now(UTC)
    inputs: tuple[dict[str, Any], ...] = (
        {
            "canonical_id": str(canonical.event_id),
            "merged_event_ids": [str(eid) for eid in merged_ids],
        },
    )
    outputs: tuple[dict[str, Any], ...] = (
        {
            "subject": canonical.subject,
            "cluster_size": len(cluster),
            "retained_observation_dates": [d.isoformat() for d in retained_dates],
            "merged_source_chunk_ids_count": len(chunk_ids),
        },
    )
    rationale = (
        f"Subject {canonical.subject!r}: {len(cluster)} near-duplicate events "
        f"within window; canonical={canonical.event_id} (confidence={canonical.confidence:.3f})."
    )
    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE,
        op_type=OpKind.CHRONICLE_EVENT_CLUSTERING,
        inputs=inputs,
        outputs=outputs,
        decision="planned",
        rationale=rationale,
        started_at=started_at,
        duration_ms=0.0,
        namespace_id=namespace_id,
    )


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _bind_uuid(session: AsyncSession, value: UUID) -> str | UUID:
    """Convert UUID to str for SQLite; pass-through for Postgres asyncpg."""
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        return value
    return str(value)


def _as_uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _as_aware(value: datetime | str) -> datetime:
    """Coerce a row value to a tz-aware UTC datetime."""
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _coerce_embedding(value: Any) -> list[float] | None:
    """Turn a row-side embedding into ``list[float]`` or ``None``.

    pgvector returns a list-like; SQLite stores the vector as a JSON
    string via the SQLAlchemy ``Text`` column the test fixture uses
    (the embedded production path keeps vectors in LanceDB instead).
    """
    import json as _json

    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = _json.loads(value)
        except (TypeError, ValueError):
            return None
    try:
        coerced = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return coerced or None

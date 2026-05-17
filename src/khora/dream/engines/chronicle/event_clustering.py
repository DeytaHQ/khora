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
``decision="planned"``. Its ``outputs`` carry the *planned* merged
``source_chunk_ids`` count and the retained ``observation_date`` list —
data the Phase 4 apply path (#649 / #669) will consume.

**Critical invariant — never propose mutating ``chronicle_events.chunk_id``.**

The temporal recall channel (see ``engine.py:1620``) dedupes events by
``chunk_id`` and surfaces the back-pointer to the chunk that owns the
event. Mutating ``chunk_id`` on the canonical row breaks that linkage.
The planned merge therefore unifies an aggregate ``source_chunk_ids``
list (the chunk_ids of cluster members, recorded for downstream apply
consumption) — it never targets the ``chunk_id`` FK column itself.

Apply mode is blocked here: ``apply_event_clustering`` raises
``NotImplementedError`` until v0.15 lands the bi-temporal apply path.

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
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.dream.config import DreamConfig


_PHASE = "plan"
_OP_SPAN = "khora.dream.chronicle.event_clustering"


async def plan_chronicle_event_clustering(
    namespace_id: UUID,
    *,
    session: AsyncSession,
    config: DreamConfig,
) -> tuple[DreamOp, ...]:
    """Plan near-duplicate clustering of ``chronicle_events``.

    Returns one :class:`DreamOp` per cluster found. When no clusters
    form, the return value is an empty tuple — the orchestrator records
    the planner ran but produced nothing to apply.

    The planner is pure SELECT; it never inserts, updates, or deletes a
    row. Cosine similarity is evaluated only within
    ``(namespace_id, subject)`` buckets so the O(N^2) inner loop stays
    bounded by per-subject event volume.
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
    return tuple(ops)


async def apply_event_clustering(*_args: Any, **_kwargs: Any) -> None:
    """Apply path is blocked pending a ``chronicle_events`` schema migration.

    Phase 4 (#669) wires the orchestrator's per-op apply dispatch, but
    the bi-temporal soft-delete columns required by this op
    (``invalidated_at``, ``invalidated_by``, ``merged_into_event_id``)
    are **not** present on the ``chronicle_events`` table as of
    migration 033 — that migration added bi-temporal columns to
    ``relationships`` and ``memory_facts`` only. Landing apply mode for
    this op requires migration 034 (out of scope for #669).

    The chunk_id invariant assertion the runtime handler will carry is
    documented in the module docstring above ("never propose mutating
    ``chronicle_events.chunk_id``"). When migration 034 lands, this
    function will be replaced with an apply handler matching the
    :func:`apply_chronicle_fact_compaction` shape: snapshot canonical /
    tail rows before mutation, ``UPDATE`` the soft-delete columns
    (never ``chunk_id``), return an :class:`UndoRecord`.
    """
    raise NotImplementedError(
        "apply_event_clustering requires chronicle_events bi-temporal columns "
        "(invalidated_at, invalidated_by, merged_into_event_id) — not present "
        "until migration 034 lands (see #669 follow-up). Plan mode is supported."
    )


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

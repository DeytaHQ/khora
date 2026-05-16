"""Chronicle ``memory_facts`` compaction planner (#664, Phase 2.4 of #649).

Pure-SELECT, zero-LLM. Plans the hard-delete of every ``memory_facts``
row that has been tombstoned (legacy ``is_active=False`` OR bi-temporal
``invalidated_at IS NOT NULL`` from migration 033) for longer than
``DreamConfig.fact_compaction_retention_days``. Cascade is safe — the
``superseded_by`` FK is declared ``ON DELETE SET NULL``.

This is the **only Phase 2 op that hard-deletes** when applied, because
the tombstone *is* the soft-delete marker; compacting it is row
reclamation, not data loss. Every other Phase 2 op uses bi-temporal
soft-delete via ``valid_to`` / ``invalidated_at``.

Apply mode is intentionally blocked in v0.14 — see Phase 4 / #669.

Span: ``khora.dream.chronicle.fact_compaction`` (internal stability).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.plan import DreamOp, OpKind
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.dream.config import DreamConfig


_PHASE = "compact"
_OP_SPAN = "khora.dream.chronicle.fact_compaction"


async def plan_chronicle_fact_compaction(
    namespace_id: UUID,
    *,
    session: AsyncSession,
    config: DreamConfig,
    apply: bool = False,
) -> tuple[DreamOp, ...]:
    """Plan the hard-delete of stale tombstoned ``memory_facts`` rows.

    A row is a candidate when either:

    - ``is_active = False`` (legacy tombstone flag), AND/OR
    - ``invalidated_at IS NOT NULL`` (bi-temporal soft-delete from
      migration 033),

    AND the most-recent tombstone marker among the two is older than
    ``config.fact_compaction_retention_days``. For the legacy flag we
    use ``updated_at`` as the tombstone timestamp (it's the only signal
    available before migration 033); for the bi-temporal column we use
    ``invalidated_at`` directly. Rows tripping both gates take the
    *later* of the two timestamps — i.e. the row must be stale by
    *both* signals.

    Returns one :class:`DreamOp` per row that would be reclaimed. Zero
    writes. Apply mode raises :class:`NotImplementedError` — flip lands
    in v0.15 (#669).
    """
    if apply:
        raise NotImplementedError("apply mode lands in v0.15 — see #649 phase 4 / #669")

    started_at = datetime.now(UTC)
    retention_days = int(config.fact_compaction_retention_days)
    threshold = started_at - timedelta(days=retention_days)

    with trace_span(
        _OP_SPAN,
        namespace_id=str(namespace_id),
        retention_days=retention_days,
    ):
        rows = await _select_candidates(session, namespace_id, threshold)

    finished_at = datetime.now(UTC)
    duration_ms = (finished_at - started_at).total_seconds() * 1000.0

    ops: list[DreamOp] = []
    for row in rows:
        marker_at = _as_aware(row["marker_at"])
        age_days = (started_at - marker_at).total_seconds() / 86400.0
        inputs: tuple[dict[str, Any], ...] = (
            {
                "fact_id": str(row["id"]),
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "age_days": age_days,
                "superseded_by": (str(row["superseded_by"]) if row["superseded_by"] else None),
            },
        )
        ops.append(
            DreamOp(
                op_id=uuid4(),
                phase=_PHASE,
                op_type=OpKind.CHRONICLE_FACT_COMPACTION,
                inputs=inputs,
                outputs=(),
                decision="planned",
                rationale=(f"Tombstoned for {age_days:.1f}d (> {retention_days}d retention)."),
                started_at=started_at,
                duration_ms=duration_ms,
                namespace_id=namespace_id,
            )
        )

    return tuple(ops)


async def _select_candidates(
    session: AsyncSession,
    namespace_id: UUID,
    threshold: datetime,
) -> list[dict[str, Any]]:
    """Return tombstoned rows whose marker timestamp is older than ``threshold``.

    The ``marker_at`` returned per row is the timestamp the row crossed
    into the tombstoned state: ``invalidated_at`` when set, else
    ``updated_at`` (the legacy-flag fallback).
    """
    dialect = session.bind.dialect.name if session.bind is not None else ""

    if dialect == "postgresql":
        stmt = text(
            """
            SELECT
                id,
                subject,
                predicate,
                object,
                superseded_by,
                COALESCE(invalidated_at, updated_at) AS marker_at
            FROM memory_facts
            WHERE namespace_id = :ns
              AND (NOT is_active OR invalidated_at IS NOT NULL)
              AND COALESCE(invalidated_at, updated_at) < :threshold
            ORDER BY marker_at ASC
            """
        )
    else:
        # SQLite / aiosqlite path — no COALESCE-on-timestamp pitfalls,
        # but timestamps come back as strings (see ``_as_aware``).
        stmt = text(
            """
            SELECT
                id,
                subject,
                predicate,
                object,
                superseded_by,
                COALESCE(invalidated_at, updated_at) AS marker_at
            FROM memory_facts
            WHERE namespace_id = :ns
              AND (NOT is_active OR invalidated_at IS NOT NULL)
              AND COALESCE(invalidated_at, updated_at) < :threshold
            ORDER BY marker_at ASC
            """
        )

    result = await session.execute(
        stmt,
        {
            "ns": _bind_uuid(session, namespace_id),
            "threshold": threshold if dialect == "postgresql" else threshold.isoformat(),
        },
    )
    return [dict(row._mapping) for row in result.all()]


def _bind_uuid(session: AsyncSession, value: UUID) -> str | UUID:
    """Convert UUID to str for SQLite; pass-through for Postgres asyncpg."""
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        return value
    return str(value)


def _as_aware(value: datetime | str) -> datetime:
    """Coerce a row value to a tz-aware UTC datetime.

    SQLite stores timestamps as strings via aiosqlite; Postgres asyncpg
    returns native ``datetime``.
    """
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt

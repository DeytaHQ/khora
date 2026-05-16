"""Chronicle ``memory_facts`` tombstone audit (#654, Phase 1.2 of #649).

Pure-SELECT, zero-LLM, zero-mutation. Counts active vs inactive facts in
the legacy ``is_active`` flag, separately counts the bi-temporal
``invalidated_at IS NOT NULL`` set (migration 033, #653), reports the
tombstone ratio and the age distribution (oldest, p50, p90) of inactive
rows, and surfaces a recommended retention threshold for the Phase 2
compaction op (#664) to consume. This op never deletes or flips flags.

The op decision values:

- ``"audit_complete"`` — always, when the namespace has at least one
  ``memory_facts`` row (active or inactive).
- ``"empty_namespace"`` — when the namespace has zero facts.

Span: ``khora.dream.chronicle.tombstone_audit`` (internal stability).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.plan import DreamOp, OpKind
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.dream.config import DreamConfig


_PHASE = "audit"
_OP_SPAN = "khora.dream.chronicle.tombstone_audit"

# Default retention recommendation when the caller / config doesn't
# specify one. Surfaced in the op outputs; never applied here.
_DEFAULT_RETENTION_DAYS = 365


async def plan_chronicle_tombstone_audit(
    namespace_id: UUID,
    *,
    session: AsyncSession,
    config: DreamConfig,  # noqa: ARG001 — reserved for future knobs (#664)
    recommended_retention_days: int = _DEFAULT_RETENTION_DAYS,
) -> DreamOp:
    """Read-only audit of memory_facts tombstone ratio and age distribution.

    Pure SELECT queries; zero LLM, zero writes. The op's outputs include:

    - ``active_count``, ``inactive_count`` (legacy ``is_active`` flag)
    - ``invalidated_count`` (bi-temporal ``invalidated_at IS NOT NULL``)
    - ``total_count`` (active + inactive)
    - ``tombstone_ratio`` = ``inactive / total`` (float in [0.0, 1.0]),
      ``0.0`` when the namespace is empty
    - ``oldest_tombstone_age_days``, ``p50_age_days``, ``p90_age_days``
      (None when there are zero inactive facts)
    - ``recommended_retention_days``: lifted from caller / config
      (default ``365``) — never applied here; surfaces in the report
      for the Phase 2 compaction op (#664) to consume
    """
    started_at = datetime.now(UTC)

    with trace_span(
        _OP_SPAN,
        namespace_id=str(namespace_id),
    ):
        active_count, inactive_count, invalidated_count = await _count_facts(session, namespace_id)
        total_count = active_count + inactive_count

        if total_count == 0:
            decision = "empty_namespace"
            rationale = "Namespace has zero memory_facts rows; nothing to audit."
            oldest_age_days: float | None = None
            p50_age_days: float | None = None
            p90_age_days: float | None = None
            tombstone_ratio = 0.0
        else:
            decision = "audit_complete"
            tombstone_ratio = inactive_count / total_count
            oldest_age_days, p50_age_days, p90_age_days = await _age_distribution(session, namespace_id)
            rationale = (
                f"Inactive {inactive_count}/{total_count} facts "
                f"(tombstone_ratio={tombstone_ratio:.3f}); "
                f"recommended retention {recommended_retention_days} days."
            )

    outputs: tuple[dict[str, Any], ...] = (
        {
            "active_count": active_count,
            "inactive_count": inactive_count,
            "invalidated_count": invalidated_count,
            "total_count": total_count,
            "tombstone_ratio": tombstone_ratio,
            "oldest_tombstone_age_days": oldest_age_days,
            "p50_age_days": p50_age_days,
            "p90_age_days": p90_age_days,
            "recommended_retention_days": recommended_retention_days,
        },
    )

    finished_at = datetime.now(UTC)
    duration_ms = (finished_at - started_at).total_seconds() * 1000.0

    return DreamOp(
        op_id=uuid4(),
        phase=_PHASE,
        op_type=OpKind.CHRONICLE_TOMBSTONE_AUDIT,
        inputs=(),
        outputs=outputs,
        decision=decision,
        rationale=rationale,
        started_at=started_at,
        duration_ms=duration_ms,
        namespace_id=namespace_id,
    )


async def _count_facts(session: AsyncSession, namespace_id: UUID) -> tuple[int, int, int]:
    """Return ``(active_count, inactive_count, invalidated_count)``.

    Aggregates computed by the database — single round-trip.
    """
    stmt = text(
        """
        SELECT
            COALESCE(SUM(CASE WHEN is_active THEN 1 ELSE 0 END), 0) AS active,
            COALESCE(SUM(CASE WHEN NOT is_active THEN 1 ELSE 0 END), 0) AS inactive,
            COALESCE(SUM(CASE WHEN invalidated_at IS NOT NULL THEN 1 ELSE 0 END), 0) AS invalidated
        FROM memory_facts
        WHERE namespace_id = :ns
        """
    )
    row = (await session.execute(stmt, {"ns": _bind_uuid(session, namespace_id)})).one()
    return int(row.active), int(row.inactive), int(row.invalidated)


async def _age_distribution(
    session: AsyncSession, namespace_id: UUID
) -> tuple[float | None, float | None, float | None]:
    """Return ``(oldest_age_days, p50_age_days, p90_age_days)`` for inactive facts.

    Postgres path uses native ``percentile_disc`` — aggregates land in
    the DB. SQLite path (the embedded fixture and the unit-test path)
    has no percentile function, so we pull the inactive ``updated_at``
    column once and compute in Python.

    Returns ``(None, None, None)`` if there are no inactive facts.
    """
    dialect = session.bind.dialect.name if session.bind is not None else ""

    if dialect == "postgresql":
        stmt = text(
            """
            SELECT
                EXTRACT(EPOCH FROM (NOW() - MIN(updated_at))) / 86400.0 AS oldest_days,
                EXTRACT(
                    EPOCH FROM (NOW() - percentile_disc(0.5) WITHIN GROUP (ORDER BY updated_at))
                ) / 86400.0 AS p50_days,
                EXTRACT(
                    EPOCH FROM (NOW() - percentile_disc(0.1) WITHIN GROUP (ORDER BY updated_at))
                ) / 86400.0 AS p90_days
            FROM memory_facts
            WHERE namespace_id = :ns AND NOT is_active
            """
        )
        row = (await session.execute(stmt, {"ns": _bind_uuid(session, namespace_id)})).one()
        if row.oldest_days is None:
            return (None, None, None)
        return (
            float(row.oldest_days),
            float(row.p50_days),
            float(row.p90_days),
        )

    # Embedded / non-Postgres path — pull the column, compute in Python.
    rows = (
        await session.execute(
            text("SELECT updated_at FROM memory_facts WHERE namespace_id = :ns AND NOT is_active"),
            {"ns": _bind_uuid(session, namespace_id)},
        )
    ).all()
    if not rows:
        return (None, None, None)
    now = datetime.now(UTC)
    ages_days = sorted((now - _as_aware(r[0])).total_seconds() / 86400.0 for r in rows)
    return (
        ages_days[-1],
        _percentile(ages_days, 0.5),
        _percentile(ages_days, 0.9),
    )


def _bind_uuid(session: AsyncSession, value: UUID) -> str | UUID:
    """Convert UUID to str for SQLite; pass-through for Postgres asyncpg."""
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        return value
    return str(value)


def _as_aware(value: datetime | str) -> datetime:
    """Coerce a row value to a tz-aware UTC datetime.

    SQLite stores timestamps as strings via aiosqlite; Postgres asyncpg
    returns native ``datetime``. Both paths land here.
    """
    if isinstance(value, str):
        # SQLite returns ISO 8601 strings; tolerate both the "Z" suffix
        # and "+00:00" offsets.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _percentile(sorted_values: list[float], q: float) -> float:
    """Discrete percentile matching Postgres ``percentile_disc`` semantics.

    ``percentile_disc(q)`` returns the smallest value v such that the
    cumulative distribution of v in the sorted set is >= q. With a
    1-indexed rank that's ``ceil(q * n)``.
    """
    import math

    n = len(sorted_values)
    rank = max(1, math.ceil(q * n))
    return sorted_values[min(rank - 1, n - 1)]

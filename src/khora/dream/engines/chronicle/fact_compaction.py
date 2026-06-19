"""Chronicle ``memory_facts`` compaction planner + apply (#664 / #669).

Pure-SELECT planner (#664, Phase 2.4) and hard-delete apply handler
(#669, Phase 4) for tombstoned ``memory_facts`` rows that are older than
``DreamConfig.fact_compaction_retention_days``. A row counts as
tombstoned when either the legacy ``is_active=False`` flag is set OR the
bi-temporal ``invalidated_at IS NOT NULL`` column (migration 033) is
populated. Cascade is safe — the ``superseded_by`` FK is declared
``ON DELETE SET NULL`` and no external table FKs into ``memory_facts``.

This is the **only Phase 4 op that hard-deletes**. Every other dream op
uses bi-temporal soft-delete via ``valid_to`` / ``invalidated_at``. Bug
here = irrecoverable data loss — the apply path therefore snapshots
each row's full content into the :class:`UndoRecord` *before* the
DELETE statement runs (see :func:`apply_chronicle_fact_compaction`).

Apply-side safety invariants:

1. **Snapshot before delete.** ``SELECT *`` per target row is captured
   into the ``UndoRecord.before["rows"]`` list before any DELETE
   statement is issued. If the snapshot loop fails partway through,
   the op aborts with no rows touched (the caller's transaction
   rolls back).
2. **Retention-days floor (defense in depth).** A tampered plan with
   ``inputs[*]['retention_days'] < 7`` is rejected by the handler even
   if the :class:`DreamConfig` validator was bypassed.
3. **Caller owns the transaction.** The handler does NOT commit, log,
   or emit telemetry — the orchestrator owns commit / checkpoint /
   span boundaries.

Span: ``khora.dream.chronicle.fact_compaction`` (internal stability).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import text

from khora.dream.exceptions import DreamForbiddenOpError
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord
from khora.telemetry import trace_span

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from khora.dream.config import DreamConfig
    from khora.storage.coordinator import StorageCoordinator


_PHASE = "compact"
_OP_SPAN = "khora.dream.chronicle.fact_compaction"

# Hard floor on fact-compaction retention enforced at apply time. Mirrors
# the :class:`DreamConfig` validator's floor (``_FACT_COMPACTION_RETENTION_FLOOR_DAYS``)
# so a tampered plan that bypassed config validation still cannot delete
# rows younger than this many days. Defense in depth.
_RETENTION_FLOOR_DAYS = 7


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
    writes from the planner. ``apply=True`` raises
    :class:`NotImplementedError` — the apply path now lives in
    :func:`apply_chronicle_fact_compaction` and is invoked by the
    orchestrator, not by this planner.
    """
    if apply:
        raise NotImplementedError(
            "plan_chronicle_fact_compaction does not execute apply mode (#669); "
            "call apply_chronicle_fact_compaction(plan_op, ...) from the orchestrator instead."
        )

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
        # ``retention_days`` is recorded on the op so the apply handler
        # can defense-in-depth re-check the floor even if a tampered
        # caller bypassed the DreamConfig validator.
        inputs: tuple[dict[str, Any], ...] = (
            {
                "fact_id": str(row["id"]),
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "age_days": age_days,
                "superseded_by": (str(row["superseded_by"]) if row["superseded_by"] else None),
                "retention_days": retention_days,
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


async def apply_chronicle_fact_compaction(
    plan_op: DreamOp,
    *,
    coordinator: StorageCoordinator | None = None,  # noqa: ARG001
    session: AsyncSession,
) -> UndoRecord:
    """Execute one planned ``chronicle_fact_compaction`` op.

    The handler issues one ``SELECT *`` per target row to capture the
    full content into ``UndoRecord.before["rows"]`` and **only then**
    deletes the rows. If the snapshot loop fails partway through, no
    DELETE has been issued — the caller's transaction rolls back to a
    clean state.

    Caller contract — the orchestrator owns:

    - the surrounding transaction (``coordinator.transaction()``);
    - commit / checkpoint / rollback;
    - logging and telemetry spans.

    This function MUST NOT log, emit telemetry, or commit. Returning
    an :class:`UndoRecord` is the only side-effect signal.

    Safety invariants (see module docstring):

    1. Snapshot before delete.
    2. ``retention_days >= 7`` defense-in-depth check rejects tampered
       ops with :class:`DreamForbiddenOpError`.
    3. Empty / already-deleted target set is a graceful no-op
       (idempotent).
    """
    if plan_op.op_type is not OpKind.CHRONICLE_FACT_COMPACTION:
        raise DreamForbiddenOpError(
            f"apply_chronicle_fact_compaction received op_type={plan_op.op_type!r}; expected CHRONICLE_FACT_COMPACTION."
        )

    retention_days = _extract_retention_days(plan_op)
    if retention_days < _RETENTION_FLOOR_DAYS:
        raise DreamForbiddenOpError(
            f"fact_compaction retention_days must be >= {_RETENTION_FLOOR_DAYS}, "
            f"got {retention_days}. The hard floor exists to prevent accidental "
            "data loss; refusing to apply even when DreamConfig validation was bypassed."
        )

    target_ids = _extract_target_ids(plan_op)

    # Snapshot every row BEFORE issuing a DELETE. If any SELECT fails
    # the surrounding transaction rolls back with zero deletes — the
    # caller (orchestrator) is responsible for that rollback.
    snapshots: list[dict[str, Any]] = []
    for fact_id in target_ids:
        row = await _snapshot_row(session, fact_id)
        if row is None:
            # Row already gone — idempotent re-apply. Skip.
            continue
        snapshots.append({"id": str(fact_id), "row": row})

    # Only AFTER all snapshots are captured do we run the DELETE.
    snapshotted_ids = [UUID(s["id"]) for s in snapshots]
    if snapshotted_ids:
        await _delete_rows(session, snapshotted_ids)

    return UndoRecord(
        op_id=plan_op.op_id,
        op_type=str(OpKind.CHRONICLE_FACT_COMPACTION),
        before={"rows": snapshots, "retention_days": retention_days},
        applied_at=datetime.now(UTC),
    )


def _extract_retention_days(plan_op: DreamOp) -> int:
    """Pull ``retention_days`` off the op's inputs, raise if absent.

    The planner always writes this field; a missing value means the op
    was hand-crafted by a caller that doesn't understand the contract
    and the handler refuses to guess a default.
    """
    for entry in plan_op.inputs:
        if isinstance(entry, dict) and "retention_days" in entry:
            try:
                return int(entry["retention_days"])
            except (TypeError, ValueError) as exc:
                raise DreamForbiddenOpError(
                    f"fact_compaction op {plan_op.op_id} has non-integer retention_days={entry['retention_days']!r}."
                ) from exc
    raise DreamForbiddenOpError(
        f"fact_compaction op {plan_op.op_id} is missing retention_days in inputs; "
        "the apply handler refuses to assume a default."
    )


def _extract_target_ids(plan_op: DreamOp) -> list[UUID]:
    """Pull every ``fact_id`` out of the op's inputs."""
    ids: list[UUID] = []
    for entry in plan_op.inputs:
        if isinstance(entry, dict) and "fact_id" in entry:
            try:
                ids.append(UUID(str(entry["fact_id"])))
            except (TypeError, ValueError) as exc:
                raise DreamForbiddenOpError(
                    f"fact_compaction op {plan_op.op_id} has malformed fact_id={entry['fact_id']!r}."
                ) from exc
    return ids


async def _snapshot_row(session: AsyncSession, fact_id: UUID) -> dict[str, Any] | None:
    """Return the full row content for ``fact_id`` or ``None`` if absent."""
    result = await session.execute(
        text("SELECT * FROM memory_facts WHERE id = :id"),
        {"id": _bind_uuid(session, fact_id)},
    )
    row = result.mappings().first()
    if row is None:
        return None
    return {k: _jsonable(v) for k, v in dict(row).items()}


async def _delete_rows(session: AsyncSession, ids: list[UUID]) -> None:
    """Hard-delete rows by id, dialect-aware.

    Postgres supports ``id = ANY(:ids)`` natively; SQLite needs an
    expanding bind. We use one bind-name per id on SQLite so the
    statement works regardless of dialect.
    """
    if not ids:
        return
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        await session.execute(
            text("DELETE FROM memory_facts WHERE id = ANY(:ids)"),
            {"ids": list(ids)},
        )
        return
    # SQLite / aiosqlite path — expanding bind. Bind the 32-char hex form
    # the sqlite_lance store wrote (see :func:`_bind_uuid`), not dashed str.
    placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
    params = {f"id_{i}": fid.hex for i, fid in enumerate(ids)}
    await session.execute(
        text(f"DELETE FROM memory_facts WHERE id IN ({placeholders})"),  # noqa: S608
        params,
    )


def _jsonable(value: Any) -> Any:
    """Coerce a row value to a JSON-serialisable form for the undo snapshot.

    ``datetime`` becomes ISO-8601 string, ``UUID`` becomes its string
    form, lists / tuples / dicts are walked recursively. Everything
    else passes through unchanged — the orchestrator's report sink
    handles final serialisation.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


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
    """Bind a UUID for raw ``text()`` SQL, per-dialect (#1067).

    Postgres asyncpg binds ``uuid.UUID`` natively. On SQLite the
    sqlite_lance store persists UUIDs as 32-char hex (no dashes) via
    SQLAlchemy ``Uuid(as_uuid=True)``, so a raw ``text()`` ``WHERE`` must
    bind ``value.hex`` - binding the dashed ``str(value)`` matched 0 rows.
    """
    dialect = session.bind.dialect.name if session.bind is not None else ""
    if dialect == "postgresql":
        return value
    return value.hex


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

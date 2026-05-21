"""Unit tests for ``apply_chronicle_fact_compaction`` (#669, Phase 4 apply).

The fact-compaction apply handler is the only Phase 4 op that
**hard-deletes** rows. The tests below pin the safety invariants:

* every deleted row's full content is snapshotted into the
  :class:`UndoRecord` *before* the DELETE runs;
* a programmatic bypass of the config validator still hits a
  defense-in-depth check that rejects ``retention_days < 7``;
* re-applying an op whose targets are already gone is a no-op;
* the undo round-trip — re-INSERT from the snapshot — reproduces the
  pre-delete row faithfully.

The handler does not commit; the orchestrator owns transaction
boundaries. These tests therefore call ``session.commit()`` themselves
after the handler returns.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.dream._undo_record import UndoRecord
from khora.dream.config import DreamConfig
from khora.dream.engines.chronicle import plan_chronicle_fact_compaction
from khora.dream.engines.chronicle.fact_compaction import (
    apply_chronicle_fact_compaction,
)
from khora.dream.exceptions import DreamForbiddenOpError
from khora.dream.plan import DreamOp, OpKind

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_alembic_config(url: str) -> Config:
    cfg = Config()
    migrations_dir = Path(__file__).resolve().parents[5] / "src" / "khora" / "db" / "migrations"
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    url = f"sqlite+aiosqlite:///{tmp_path / 'fact_compaction_apply.db'}"
    command.upgrade(_make_alembic_config(url), "head")
    return url


@pytest.fixture
async def session(sqlite_url: str) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(sqlite_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def _create_namespace(session: AsyncSession, ns_id: UUID) -> None:
    await session.execute(
        sa.text(
            "INSERT INTO memory_namespaces (id, namespace_id, tenancy_mode, version, is_active) "
            "VALUES (:id, :ns, 'shared', 1, 1)"
        ),
        {"id": str(ns_id), "ns": str(ns_id)},
    )
    await session.commit()


async def _insert_fact(
    session: AsyncSession,
    *,
    namespace_id: UUID,
    is_active: bool = False,
    updated_at: datetime | None = None,
    invalidated_at: datetime | None = None,
    superseded_by: UUID | None = None,
    subject: str = "Alice",
    predicate: str = "lives_in",
    object_: str = "Paris",
    confidence: float = 0.9,
) -> UUID:
    fact_id = uuid4()
    params: dict[str, object] = {
        "id": str(fact_id),
        "ns": str(namespace_id),
        "active": 1 if is_active else 0,
        "src": json.dumps([]),
        "subj": subject,
        "pred": predicate,
        "obj": object_,
        "conf": confidence,
    }
    if updated_at is not None:
        params["updated"] = updated_at.isoformat()
        params["created"] = updated_at.isoformat()
        updated_clause = ", created_at, updated_at"
        updated_values = ", :created, :updated"
    else:
        updated_clause = ""
        updated_values = ""
    if invalidated_at is not None:
        params["inv"] = invalidated_at.isoformat()
        inv_clause = ", invalidated_at"
        inv_values = ", :inv"
    else:
        inv_clause = ""
        inv_values = ""
    if superseded_by is not None:
        params["sup"] = str(superseded_by)
        sup_clause = ", superseded_by"
        sup_values = ", :sup"
    else:
        sup_clause = ""
        sup_values = ""

    stmt = (
        "INSERT INTO memory_facts "  # noqa: S608
        f"(id, namespace_id, subject, predicate, object, fact_text, confidence, is_active, source_chunk_ids"
        f"{updated_clause}{inv_clause}{sup_clause}) "
        f"VALUES (:id, :ns, :subj, :pred, :obj, 'fact text', :conf, :active, :src"
        f"{updated_values}{inv_values}{sup_values})"
    )
    await session.execute(sa.text(stmt), params)
    return fact_id


async def _count_rows(session: AsyncSession, namespace_id: UUID) -> int:
    result = await session.execute(
        sa.text("SELECT COUNT(*) FROM memory_facts WHERE namespace_id=:ns"),
        {"ns": str(namespace_id)},
    )
    return int(result.scalar_one())


async def _plan_one(session: AsyncSession, ns: UUID, *, retention_days: int = 365) -> DreamOp:
    """Return a single planned op (asserts the planner produced exactly one)."""
    cfg = DreamConfig(fact_compaction_retention_days=retention_days)
    ops = await plan_chronicle_fact_compaction(ns, session=session, config=cfg)
    assert len(ops) == 1, f"expected exactly 1 planned op, got {len(ops)}"
    return ops[0]


# ---------------------------------------------------------------------------
# Load-bearing safety: snapshot is captured before any DELETE runs
# ---------------------------------------------------------------------------


async def test_snapshot_captured_before_delete_executes(session: AsyncSession) -> None:
    """The handler MUST snapshot the row before the DELETE statement runs.

    Strategy: capture every SQL statement the handler issues against
    the session by wrapping :meth:`AsyncSession.execute`. The first
    statement targeting ``memory_facts`` MUST be a SELECT (the
    snapshot) — never a DELETE.

    This is the load-bearing safety property: if the DELETE runs first
    and the snapshot loop later fails, the row is gone with no undo
    record.
    """
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    fact_id = await _insert_fact(session, namespace_id=ns, is_active=False, updated_at=old)
    await session.commit()

    op = await _plan_one(session, ns)

    executed: list[str] = []
    real_execute = session.execute

    async def tracing_execute(stmt, *args, **kwargs):  # type: ignore[no-untyped-def]
        executed.append(str(stmt))
        return await real_execute(stmt, *args, **kwargs)

    session.execute = tracing_execute  # type: ignore[method-assign]
    try:
        undo = await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
    finally:
        session.execute = real_execute  # type: ignore[method-assign]
    await session.commit()

    facts_touches = [s for s in executed if "memory_facts" in s.lower()]
    assert facts_touches, "handler did not touch memory_facts at all"
    first = facts_touches[0].lower()
    assert "select" in first, f"first memory_facts statement was not a SELECT: {facts_touches[0]!r}"
    assert "delete" not in first, (
        f"first memory_facts statement was a DELETE — snapshot would be lost: {facts_touches[0]!r}"
    )

    assert undo.before["rows"], "undo.before['rows'] is empty"
    snapshotted_ids = {UUID(r["id"]) for r in undo.before["rows"]}
    assert fact_id in snapshotted_ids


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_deletes_and_returns_undo_record(session: AsyncSession) -> None:
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    fact_id = await _insert_fact(
        session,
        namespace_id=ns,
        is_active=False,
        updated_at=old,
        subject="Bob",
        predicate="works_at",
        object_="Acme",
        confidence=0.77,
    )
    await session.commit()

    op = await _plan_one(session, ns)
    undo = await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
    await session.commit()

    assert isinstance(undo, UndoRecord)
    assert undo.op_id == op.op_id
    assert undo.op_type == OpKind.CHRONICLE_FACT_COMPACTION.value
    assert undo.applied_at.tzinfo is not None
    assert undo.before["retention_days"] >= 7
    assert len(undo.before["rows"]) == 1
    row = undo.before["rows"][0]
    assert UUID(row["id"]) == fact_id
    assert row["row"]["subject"] == "Bob"
    assert row["row"]["predicate"] == "works_at"
    assert row["row"]["object"] == "Acme"
    assert float(row["row"]["confidence"]) == pytest.approx(0.77)

    assert await _count_rows(session, ns) == 0


# ---------------------------------------------------------------------------
# Defense-in-depth retention floor
# ---------------------------------------------------------------------------


async def test_retention_days_below_floor_raises(session: AsyncSession) -> None:
    """Programmatic bypass of the validator still hits the handler's floor."""
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    await _insert_fact(session, namespace_id=ns, is_active=False, updated_at=old)
    await session.commit()

    bad_op = DreamOp(
        op_id=uuid4(),
        phase="compact",
        op_type=OpKind.CHRONICLE_FACT_COMPACTION,
        inputs=(
            {
                "fact_id": str(uuid4()),
                "subject": "x",
                "predicate": "y",
                "object": "z",
                "age_days": 400.0,
                "superseded_by": None,
                "retention_days": 5,
            },
        ),
        outputs=(),
        decision="planned",
        rationale="tampered",
        namespace_id=ns,
    )

    with pytest.raises(DreamForbiddenOpError, match="retention_days"):
        await apply_chronicle_fact_compaction(bad_op, coordinator=None, session=session)

    assert await _count_rows(session, ns) == 1


async def test_missing_retention_days_raises(session: AsyncSession) -> None:
    """A planned op with no retention_days at all is rejected."""
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    await _insert_fact(session, namespace_id=ns, is_active=False, updated_at=old)
    await session.commit()

    bad_op = DreamOp(
        op_id=uuid4(),
        phase="compact",
        op_type=OpKind.CHRONICLE_FACT_COMPACTION,
        inputs=(
            {
                "fact_id": str(uuid4()),
                "subject": "x",
                "predicate": "y",
                "object": "z",
                "age_days": 400.0,
                "superseded_by": None,
            },
        ),
        outputs=(),
        decision="planned",
        rationale="tampered",
        namespace_id=ns,
    )

    with pytest.raises(DreamForbiddenOpError, match="retention_days"):
        await apply_chronicle_fact_compaction(bad_op, coordinator=None, session=session)

    assert await _count_rows(session, ns) == 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_apply_twice_is_no_op_on_second_run(session: AsyncSession) -> None:
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    await _insert_fact(session, namespace_id=ns, is_active=False, updated_at=old)
    await session.commit()

    op = await _plan_one(session, ns)

    first = await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
    await session.commit()
    assert len(first.before["rows"]) == 1
    assert await _count_rows(session, ns) == 0

    second = await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
    await session.commit()
    assert second.before["rows"] == []
    assert await _count_rows(session, ns) == 0


# ---------------------------------------------------------------------------
# Undo round-trip
# ---------------------------------------------------------------------------


async def test_undo_round_trip_reinserts_row_faithfully(session: AsyncSession) -> None:
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    fact_id = await _insert_fact(
        session,
        namespace_id=ns,
        is_active=False,
        updated_at=old,
        subject="Carol",
        predicate="speaks",
        object_="French",
        confidence=0.42,
    )
    await session.commit()

    op = await _plan_one(session, ns)
    undo = await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
    await session.commit()
    assert await _count_rows(session, ns) == 0

    # Restore from the snapshot.
    row = undo.before["rows"][0]["row"]
    cols = sorted(row.keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    col_list = ", ".join(cols)
    await session.execute(
        sa.text(f"INSERT INTO memory_facts ({col_list}) VALUES ({placeholders})"),  # noqa: S608
        row,
    )
    await session.commit()

    result = await session.execute(
        sa.text("SELECT subject, predicate, object, confidence FROM memory_facts WHERE id=:id"),
        {"id": str(fact_id)},
    )
    restored = result.mappings().one()
    assert restored["subject"] == "Carol"
    assert restored["predicate"] == "speaks"
    assert restored["object"] == "French"
    assert float(restored["confidence"]) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Handler does not commit
# ---------------------------------------------------------------------------


async def test_handler_does_not_commit(session: AsyncSession) -> None:
    """If the orchestrator rolls back, the handler's DELETE must be reverted."""
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    fact_id = await _insert_fact(session, namespace_id=ns, is_active=False, updated_at=old)
    await session.commit()

    op = await _plan_one(session, ns)
    await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
    await session.rollback()

    result = await session.execute(
        sa.text("SELECT COUNT(*) FROM memory_facts WHERE id=:id"),
        {"id": str(fact_id)},
    )
    assert int(result.scalar_one()) == 1


# ---------------------------------------------------------------------------
# Wrong op type is refused
# ---------------------------------------------------------------------------


async def test_wrong_op_type_is_refused(session: AsyncSession) -> None:
    """Defense in depth: handler refuses to process a non-compaction op."""
    ns = uuid4()
    await _create_namespace(session, ns)

    bad_op = DreamOp(
        op_id=uuid4(),
        phase="compact",
        op_type=OpKind.CHRONICLE_EVENT_CLUSTERING,
        inputs=(),
        outputs=(),
        decision="planned",
        rationale="",
        namespace_id=ns,
    )

    with pytest.raises(DreamForbiddenOpError, match="op_type"):
        await apply_chronicle_fact_compaction(bad_op, coordinator=None, session=session)


# ---------------------------------------------------------------------------
# Snapshot integrity under multi-row ops
# ---------------------------------------------------------------------------


async def test_multi_row_each_op_snapshots_its_own_row(session: AsyncSession) -> None:
    """When the planner emits one op per row, each handler call covers that row.

    The planner emits one DreamOp per stale fact; the orchestrator
    invokes the handler once per op. So each handler call is responsible
    for exactly the rows referenced in its op.
    """
    ns = uuid4()
    await _create_namespace(session, ns)
    old = datetime.now(UTC) - timedelta(days=400)
    ids = [
        await _insert_fact(
            session,
            namespace_id=ns,
            is_active=False,
            updated_at=old,
            subject=f"subj-{i}",
        )
        for i in range(3)
    ]
    await session.commit()

    cfg = DreamConfig(fact_compaction_retention_days=365)
    ops = await plan_chronicle_fact_compaction(ns, session=session, config=cfg)
    assert len(ops) == 3
    planned_ids = {UUID(op.inputs[0]["fact_id"]) for op in ops}
    assert planned_ids == set(ids)

    undos: list[UndoRecord] = []
    for op in ops:
        undo = await apply_chronicle_fact_compaction(op, coordinator=None, session=session)
        undos.append(undo)
    await session.commit()

    assert await _count_rows(session, ns) == 0
    snapshotted_ids = set()
    for undo in undos:
        assert len(undo.before["rows"]) == 1
        snapshotted_ids.add(UUID(undo.before["rows"][0]["id"]))
    assert snapshotted_ids == set(ids)

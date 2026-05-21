"""Unit tests for ``plan_chronicle_fact_compaction`` (#664, Phase 2.4 of #649).

Runs against an in-memory SQLite database with the full alembic chain
applied. The compaction planner is dry-run only in v0.14 — every test
here asserts zero writes; apply mode raises ``NotImplementedError``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from alembic import command
from khora.dream.config import DreamConfig
from khora.dream.engines.chronicle import plan_chronicle_fact_compaction
from khora.dream.plan import DreamOp, OpKind

pytestmark = pytest.mark.unit


def _make_alembic_config(url: str) -> Config:
    cfg = Config()
    migrations_dir = Path(__file__).resolve().parents[5] / "src" / "khora" / "db" / "migrations"
    cfg.set_main_option("script_location", str(migrations_dir))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["database_url"] = url
    return cfg


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    url = f"sqlite+aiosqlite:///{tmp_path / 'fact_compaction.db'}"
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
    is_active: bool,
    updated_at: datetime | None = None,
    invalidated_at: datetime | None = None,
    superseded_by: UUID | None = None,
    subject: str = "Alice",
    predicate: str = "lives_in",
    object_: str = "city",
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
        f"VALUES (:id, :ns, :subj, :pred, :obj, 'fact text', 0.9, :active, :src"
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_compaction_plans_legacy_tombstone(session: AsyncSession) -> None:
    """``is_active=False`` rows older than the retention window are planned."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    # 2 stale legacy tombstones (400 days old, > 365d default).
    stale_ids = {
        await _insert_fact(
            session,
            namespace_id=ns,
            is_active=False,
            updated_at=now - timedelta(days=400),
        )
        for _ in range(2)
    }
    # 1 active fact — never a candidate, regardless of age.
    await _insert_fact(session, namespace_id=ns, is_active=True, updated_at=now - timedelta(days=400))
    await session.commit()

    ops = await plan_chronicle_fact_compaction(ns, session=session, config=DreamConfig())

    assert len(ops) == 2
    planned_ids = {UUID(op.inputs[0]["fact_id"]) for op in ops}
    assert planned_ids == stale_ids
    for op in ops:
        assert isinstance(op, DreamOp)
        assert op.op_type is OpKind.CHRONICLE_FACT_COMPACTION
        assert op.decision == "planned"
        assert op.inputs[0]["age_days"] == pytest.approx(400.0, abs=1.0)


async def test_compaction_plans_bitemporal_tombstone(session: AsyncSession) -> None:
    """Rows with stale ``invalidated_at`` are planned even with ``is_active=True``.

    Covers the v0.14 dual-write window where the bi-temporal column may
    be set while the legacy flag still says active.
    """
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    target = await _insert_fact(
        session,
        namespace_id=ns,
        is_active=True,
        invalidated_at=now - timedelta(days=400),
    )
    await session.commit()

    ops = await plan_chronicle_fact_compaction(ns, session=session, config=DreamConfig())

    assert len(ops) == 1
    assert UUID(ops[0].inputs[0]["fact_id"]) == target


async def test_compaction_excludes_fresh_tombstones(session: AsyncSession) -> None:
    """Tombstones inside the retention window are never planned, even if inactive."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    # Fresh legacy tombstone (1 day old).
    await _insert_fact(
        session,
        namespace_id=ns,
        is_active=False,
        updated_at=now - timedelta(days=1),
    )
    # Fresh bi-temporal tombstone (10 days old).
    await _insert_fact(
        session,
        namespace_id=ns,
        is_active=False,
        invalidated_at=now - timedelta(days=10),
    )
    await session.commit()

    ops = await plan_chronicle_fact_compaction(ns, session=session, config=DreamConfig())
    assert ops == ()


@pytest.mark.parametrize("retention_days", [30, 365, 730])
async def test_compaction_respects_retention_threshold(session: AsyncSession, retention_days: int) -> None:
    """Retention threshold is configurable and respected at the row boundary."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    # Synthetic ages around the boundary.
    ages = [15, 60, 400, 800]
    for age in ages:
        await _insert_fact(
            session,
            namespace_id=ns,
            is_active=False,
            updated_at=now - timedelta(days=age),
        )
    await session.commit()

    cfg = DreamConfig(fact_compaction_retention_days=retention_days)
    ops = await plan_chronicle_fact_compaction(ns, session=session, config=cfg)

    expected = sum(1 for age in ages if age > retention_days)
    assert len(ops) == expected
    for op in ops:
        assert op.inputs[0]["age_days"] > retention_days


async def test_compaction_carries_superseded_by(session: AsyncSession) -> None:
    """The supersession pointer is surfaced verbatim in the op inputs."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    # Replacement fact (active, recent — itself never a candidate).
    replacement = await _insert_fact(session, namespace_id=ns, is_active=True)
    stale = await _insert_fact(
        session,
        namespace_id=ns,
        is_active=False,
        updated_at=now - timedelta(days=400),
        superseded_by=replacement,
    )
    await session.commit()

    ops = await plan_chronicle_fact_compaction(ns, session=session, config=DreamConfig())

    assert len(ops) == 1
    assert UUID(ops[0].inputs[0]["fact_id"]) == stale
    assert UUID(ops[0].inputs[0]["superseded_by"]) == replacement


async def test_compaction_no_writes(session: AsyncSession) -> None:
    """The planner must not insert / update / delete a single row."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    for _ in range(3):
        await _insert_fact(session, namespace_id=ns, is_active=True)
    for _ in range(4):
        await _insert_fact(
            session,
            namespace_id=ns,
            is_active=False,
            updated_at=now - timedelta(days=400),
        )
    await session.commit()

    before = await _count_rows(session, ns)
    await plan_chronicle_fact_compaction(ns, session=session, config=DreamConfig())
    after = await _count_rows(session, ns)
    assert before == after == 7


async def test_compaction_apply_raises_not_implemented(session: AsyncSession) -> None:
    """Apply mode is intentionally blocked in v0.14 — see #649 phase 4 / #669."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    await _insert_fact(
        session,
        namespace_id=ns,
        is_active=False,
        updated_at=now - timedelta(days=400),
    )
    await session.commit()

    with pytest.raises(NotImplementedError, match="#669"):
        await plan_chronicle_fact_compaction(ns, session=session, config=DreamConfig(), apply=True)
    # And no writes happened either.
    assert await _count_rows(session, ns) == 1


async def test_compaction_respects_namespace_isolation(session: AsyncSession) -> None:
    """Stale tombstones from other namespaces must not leak into the plan."""
    ns_a, ns_b = uuid4(), uuid4()
    await _create_namespace(session, ns_a)
    await _create_namespace(session, ns_b)
    now = datetime.now(UTC)
    await _insert_fact(
        session,
        namespace_id=ns_a,
        is_active=False,
        updated_at=now - timedelta(days=400),
    )
    for _ in range(3):
        await _insert_fact(
            session,
            namespace_id=ns_b,
            is_active=False,
            updated_at=now - timedelta(days=400),
        )
    await session.commit()

    ops_a = await plan_chronicle_fact_compaction(ns_a, session=session, config=DreamConfig())
    ops_b = await plan_chronicle_fact_compaction(ns_b, session=session, config=DreamConfig())

    assert len(ops_a) == 1
    assert len(ops_b) == 3
    assert ops_a[0].namespace_id == ns_a
    for op in ops_b:
        assert op.namespace_id == ns_b


def test_dream_op_shape_round_trips_json() -> None:
    """The DreamOp output must be JSON-serialisable for the report sinks."""
    op = asyncio.run(_one_op_for_serialisation())
    payload = {
        "decision": op.decision,
        "phase": op.phase,
        "op_type": op.op_type.value,
        "inputs": list(op.inputs),
        "namespace_id": str(op.namespace_id),
    }
    blob = json.dumps(payload)
    restored = json.loads(blob)
    assert restored["decision"] == "planned"
    assert restored["op_type"] == "chronicle_fact_compaction"
    assert restored["inputs"][0]["fact_id"]


async def _one_op_for_serialisation() -> DreamOp:
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with eng.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    sa.text(
                        "CREATE TABLE memory_facts ("
                        "id TEXT PRIMARY KEY, namespace_id TEXT, subject TEXT, "
                        "predicate TEXT, object TEXT, fact_text TEXT, "
                        "confidence REAL, is_active INTEGER, source_chunk_ids TEXT, "
                        "session_id TEXT, valid_to TEXT, invalidated_at TEXT, "
                        "invalidated_by TEXT, superseded_by TEXT, "
                        "created_at TEXT DEFAULT CURRENT_TIMESTAMP, "
                        "updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
                    )
                )
            )
        ns = uuid4()
        old = (datetime.now(UTC) - timedelta(days=400)).isoformat()
        async with eng.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO memory_facts "
                    "(id, namespace_id, subject, predicate, object, fact_text, "
                    "confidence, is_active, source_chunk_ids, created_at, updated_at) "
                    "VALUES (:id, :ns, 'Alice', 'lives_in', 'Paris', 'Alice lives in Paris', "
                    "0.9, 0, '[]', :ts, :ts)"
                ),
                {"id": str(uuid4()), "ns": str(ns), "ts": old},
            )
        factory = async_sessionmaker(eng, expire_on_commit=False)
        async with factory() as s:
            ops = await plan_chronicle_fact_compaction(ns, session=s, config=DreamConfig())
            assert ops, "expected at least one planned op"
            return ops[0]
    finally:
        await eng.dispose()

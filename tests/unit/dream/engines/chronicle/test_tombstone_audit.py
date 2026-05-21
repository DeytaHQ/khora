"""Unit tests for ``plan_chronicle_tombstone_audit`` (#654, Phase 1.2 of #649).

Runs against an in-memory SQLite database with the full alembic chain
applied, so the schema includes both the legacy ``is_active`` column and
the migration-033 bi-temporal columns. Zero LLM, zero mutations — the
audit is pure SELECT.
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
from khora.dream.engines.chronicle import plan_chronicle_tombstone_audit
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
    url = f"sqlite+aiosqlite:///{tmp_path / 'tombstone_audit.db'}"
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
) -> UUID:
    fact_id = uuid4()
    params: dict[str, object] = {
        "id": str(fact_id),
        "ns": str(namespace_id),
        "active": 1 if is_active else 0,
        "src": json.dumps([]),
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

    # Test helper; the `*_clause` / `*_values` fragments are local
    # literal strings — bound parameters carry every user value.
    stmt = (
        "INSERT INTO memory_facts "  # noqa: S608
        f"(id, namespace_id, subject, predicate, object, fact_text, confidence, is_active, source_chunk_ids{updated_clause}{inv_clause}) "
        f"VALUES (:id, :ns, 'Alice', 'lives_in', 'city', 'Alice lives in X', 0.9, :active, :src{updated_values}{inv_values})"
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


async def test_audit_counts_active_and_inactive(session: AsyncSession) -> None:
    """10 active + 5 inactive facts → counts match exactly."""
    ns = uuid4()
    await _create_namespace(session, ns)
    for _ in range(10):
        await _insert_fact(session, namespace_id=ns, is_active=True)
    for _ in range(5):
        await _insert_fact(session, namespace_id=ns, is_active=False)
    await session.commit()

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    outputs = op.outputs[0]
    assert isinstance(op, DreamOp)
    assert op.op_type is OpKind.CHRONICLE_TOMBSTONE_AUDIT
    assert op.decision == "audit_complete"
    assert outputs["active_count"] == 10
    assert outputs["inactive_count"] == 5
    assert outputs["total_count"] == 15


async def test_audit_computes_tombstone_ratio(session: AsyncSession) -> None:
    """tombstone_ratio = inactive / total, exactly."""
    ns = uuid4()
    await _create_namespace(session, ns)
    for _ in range(3):
        await _insert_fact(session, namespace_id=ns, is_active=True)
    for _ in range(7):
        await _insert_fact(session, namespace_id=ns, is_active=False)
    await session.commit()

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    assert op.outputs[0]["tombstone_ratio"] == pytest.approx(0.7)


async def test_audit_age_distribution(session: AsyncSession) -> None:
    """Synthetic ages 1, 5, 10, 30, 100 days → p50=10, p90=100, oldest=100."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    age_days = [1, 5, 10, 30, 100]
    for d in age_days:
        await _insert_fact(
            session,
            namespace_id=ns,
            is_active=False,
            updated_at=now - timedelta(days=d),
        )
    await _insert_fact(session, namespace_id=ns, is_active=True)
    await session.commit()

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    outputs = op.outputs[0]
    # Tolerances chosen wide enough to absorb sub-second wall-clock drift
    # between insert and audit (we recompute "now" inside the audit).
    assert outputs["oldest_tombstone_age_days"] == pytest.approx(100.0, abs=0.1)
    # percentile_disc(0.5) over [1,5,10,30,100] → rank ceil(2.5)=3 → 10.
    assert outputs["p50_age_days"] == pytest.approx(10.0, abs=0.1)
    # percentile_disc(0.9) over [1,5,10,30,100] → rank ceil(4.5)=5 → 100.
    assert outputs["p90_age_days"] == pytest.approx(100.0, abs=0.1)


async def test_audit_handles_empty_namespace(session: AsyncSession) -> None:
    """Empty namespace → decision = 'empty_namespace', no age stats."""
    ns = uuid4()
    await _create_namespace(session, ns)

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    outputs = op.outputs[0]
    assert op.decision == "empty_namespace"
    assert outputs["active_count"] == 0
    assert outputs["inactive_count"] == 0
    assert outputs["total_count"] == 0
    assert outputs["tombstone_ratio"] == 0.0
    assert outputs["oldest_tombstone_age_days"] is None
    assert outputs["p50_age_days"] is None
    assert outputs["p90_age_days"] is None


async def test_audit_no_writes(session: AsyncSession) -> None:
    """The audit must not insert / update / delete a single row."""
    ns = uuid4()
    await _create_namespace(session, ns)
    for _ in range(4):
        await _insert_fact(session, namespace_id=ns, is_active=True)
    for _ in range(3):
        await _insert_fact(session, namespace_id=ns, is_active=False)
    await session.commit()

    before = await _count_rows(session, ns)
    await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    after = await _count_rows(session, ns)
    assert before == after == 7


async def test_audit_counts_bitemporal_invalidated(session: AsyncSession) -> None:
    """invalidated_count tracks the migration-033 column independently of is_active."""
    ns = uuid4()
    await _create_namespace(session, ns)
    now = datetime.now(UTC)
    # 2 facts that are still active and not bi-temporally invalidated.
    for _ in range(2):
        await _insert_fact(session, namespace_id=ns, is_active=True)
    # 3 facts that are bi-temporally invalidated AND legacy-inactive.
    for _ in range(3):
        await _insert_fact(
            session,
            namespace_id=ns,
            is_active=False,
            invalidated_at=now - timedelta(days=1),
        )
    # 1 fact bi-temporally invalidated but still is_active=True (atypical
    # but legal during the v0.14 dual-write window).
    await _insert_fact(
        session,
        namespace_id=ns,
        is_active=True,
        invalidated_at=now - timedelta(days=1),
    )
    await session.commit()

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig())
    outputs = op.outputs[0]
    assert outputs["active_count"] == 3
    assert outputs["inactive_count"] == 3
    assert outputs["invalidated_count"] == 4


async def test_audit_respects_namespace_isolation(session: AsyncSession) -> None:
    """Facts from other namespaces must not bleed into the audit."""
    ns_a, ns_b = uuid4(), uuid4()
    await _create_namespace(session, ns_a)
    await _create_namespace(session, ns_b)
    # ns_a: 3 active, 1 inactive
    for _ in range(3):
        await _insert_fact(session, namespace_id=ns_a, is_active=True)
    await _insert_fact(session, namespace_id=ns_a, is_active=False)
    # ns_b: 1 active, 4 inactive — must not show up in ns_a's audit
    await _insert_fact(session, namespace_id=ns_b, is_active=True)
    for _ in range(4):
        await _insert_fact(session, namespace_id=ns_b, is_active=False)
    await session.commit()

    op_a = await plan_chronicle_tombstone_audit(ns_a, session=session, config=DreamConfig())
    op_b = await plan_chronicle_tombstone_audit(ns_b, session=session, config=DreamConfig())

    assert op_a.outputs[0]["active_count"] == 3
    assert op_a.outputs[0]["inactive_count"] == 1
    assert op_b.outputs[0]["active_count"] == 1
    assert op_b.outputs[0]["inactive_count"] == 4


async def test_audit_surfaces_recommended_retention(session: AsyncSession) -> None:
    """Caller-supplied retention threshold appears verbatim in outputs."""
    ns = uuid4()
    await _create_namespace(session, ns)
    await _insert_fact(session, namespace_id=ns, is_active=True)
    await session.commit()

    op = await plan_chronicle_tombstone_audit(ns, session=session, config=DreamConfig(), recommended_retention_days=42)
    assert op.outputs[0]["recommended_retention_days"] == 42


def test_dream_op_shape_round_trips_json() -> None:
    """The DreamOp output must be JSON-serialisable for the report sinks."""
    op = asyncio.run(_one_op_for_serialisation())
    payload = {
        "decision": op.decision,
        "phase": op.phase,
        "op_type": op.op_type.value,
        "outputs": list(op.outputs),
        "namespace_id": str(op.namespace_id),
    }
    blob = json.dumps(payload)
    restored = json.loads(blob)
    assert restored["decision"] in ("audit_complete", "empty_namespace")
    assert restored["op_type"] == "chronicle_tombstone_audit"


async def _one_op_for_serialisation() -> DreamOp:
    # Spin up a throwaway in-memory db just to round-trip the op shape.
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
        factory = async_sessionmaker(eng, expire_on_commit=False)
        async with factory() as s:
            return await plan_chronicle_tombstone_audit(uuid4(), session=s, config=DreamConfig())
    finally:
        await eng.dispose()

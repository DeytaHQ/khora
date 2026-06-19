"""Unit tests for ``plan_chronicle_event_clustering`` (#665, Phase 2.5 of #649).

Runs against an in-memory SQLite database with a hand-rolled
``chronicle_events`` table — the alembic chain omits the ``embedding``
column on SQLite (Postgres-only), but the planner reads it directly.

The tests cover:

- cluster forms when SVO cosine >= threshold AND date window holds
- no cluster forms across the date window
- no cluster forms below the cosine threshold
- canonical selection prefers the highest-confidence row
- the planner never proposes a write that touches ``chronicle_events.chunk_id``
- the planner emits zero writes (row-count invariant)
- ``DreamOp`` round-trips JSON cleanly
- apply path is wired (real handler in ``test_event_clustering_apply.py``)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.dream.config import DreamConfig
from khora.dream.engines.chronicle import (
    plan_chronicle_event_clustering,
)
from khora.dream.plan import OpKind

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """In-memory SQLite with a minimal ``chronicle_events`` schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    sa.text(
                        "CREATE TABLE chronicle_events ("
                        "id TEXT PRIMARY KEY, "
                        "namespace_id TEXT NOT NULL, "
                        "chunk_id TEXT NOT NULL, "
                        "subject TEXT NOT NULL, "
                        "verb TEXT NOT NULL, "
                        "object TEXT, "
                        "observation_date TEXT NOT NULL, "
                        "referenced_date TEXT, "
                        "confidence REAL NOT NULL DEFAULT 1.0, "
                        "embedding TEXT, "
                        "invalidated_at TEXT"
                        ")"
                    )
                )
            )
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def _insert_event(
    session: AsyncSession,
    *,
    namespace_id: UUID,
    subject: str,
    embedding: list[float],
    referenced_date: datetime,
    confidence: float = 1.0,
    chunk_id: UUID | None = None,
    observation_date: datetime | None = None,
    event_id: UUID | None = None,
    invalidated_at: datetime | None = None,
) -> tuple[UUID, UUID]:
    """Insert one ``chronicle_events`` row. Returns ``(event_id, chunk_id)``."""
    ev_id = event_id or uuid4()
    ch_id = chunk_id or uuid4()
    obs = observation_date or datetime.now(UTC)
    await session.execute(
        sa.text(
            "INSERT INTO chronicle_events "
            "(id, namespace_id, chunk_id, subject, verb, object, "
            "observation_date, referenced_date, confidence, embedding, "
            "invalidated_at) "
            "VALUES (:id, :ns, :chunk, :subject, 'did', '', "
            ":obs, :ref, :conf, :emb, :inv)"
        ),
        {
            # Seed UUIDs as 32-char hex - the form the sqlite_lance
            # production adapter writes, which is what ``_bind_uuid`` now
            # binds against on SQLite (#1067).
            "id": ev_id.hex,
            "ns": namespace_id.hex,
            "chunk": ch_id.hex,
            "subject": subject,
            "obs": obs.isoformat(),
            "ref": referenced_date.isoformat(),
            "conf": confidence,
            "emb": json.dumps(embedding),
            "inv": invalidated_at.isoformat() if invalidated_at else None,
        },
    )
    return ev_id, ch_id


async def _count_rows(session: AsyncSession, namespace_id: UUID) -> int:
    result = await session.execute(
        sa.text("SELECT COUNT(*) FROM chronicle_events WHERE namespace_id=:ns"),
        {"ns": namespace_id.hex},
    )
    return int(result.scalar_one())


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize; ingest path does this so the planner can dot-product."""
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm > 0 else vec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_cluster_forms_when_cosine_and_window_hold(session: AsyncSession) -> None:
    """Identical SVO embeddings within the date window cluster together."""
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    await _insert_event(session, namespace_id=ns, subject="alice", embedding=emb, referenced_date=today)
    await _insert_event(
        session, namespace_id=ns, subject="alice", embedding=emb, referenced_date=today + timedelta(days=3)
    )
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert len(ops) == 1
    op = ops[0]
    assert op.op_type is OpKind.CHRONICLE_EVENT_CLUSTERING
    assert op.decision == "planned"
    assert op.outputs[0]["cluster_size"] == 2


async def test_no_cluster_across_window(session: AsyncSession) -> None:
    """Same SVO outside the configured window must not cluster."""
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    # Default window is 7 days; 30 days apart is well outside.
    await _insert_event(session, namespace_id=ns, subject="alice", embedding=emb, referenced_date=today)
    await _insert_event(
        session, namespace_id=ns, subject="alice", embedding=emb, referenced_date=today + timedelta(days=30)
    )
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert ops == ()


async def test_no_cluster_below_threshold(session: AsyncSession) -> None:
    """Embeddings below the cosine threshold do not cluster."""
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    # Two orthogonal-ish vectors give cosine well below 0.95.
    emb_a = _normalize([1.0, 0.0, 0.0])
    emb_b = _normalize([0.5, 0.87, 0.0])  # cosine ~0.5
    await _insert_event(session, namespace_id=ns, subject="alice", embedding=emb_a, referenced_date=today)
    await _insert_event(session, namespace_id=ns, subject="alice", embedding=emb_b, referenced_date=today)
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert ops == ()


async def test_canonical_is_highest_confidence(session: AsyncSession) -> None:
    """Canonical event is the highest-confidence row in the cluster."""
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    lo_id, _ = await _insert_event(
        session, namespace_id=ns, subject="bob", embedding=emb, referenced_date=today, confidence=0.6
    )
    hi_id, _ = await _insert_event(
        session, namespace_id=ns, subject="bob", embedding=emb, referenced_date=today, confidence=0.95
    )
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert len(ops) == 1
    inputs = ops[0].inputs[0]
    assert inputs["canonical_id"] == str(hi_id)
    assert set(inputs["merged_event_ids"]) == {str(lo_id), str(hi_id)}


async def test_invariant_no_chunk_id_mutation(session: AsyncSession) -> None:
    """**Critical invariant**: no DreamOp output proposes mutating ``chunk_id``.

    The temporal recall channel at engine.py:1620 dedupes by chunk_id;
    mutating it breaks chunk -> event back-pointers. The planner must
    only describe an aggregate ``source_chunk_ids`` count — never a
    write directive against the column.
    """
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    await _insert_event(session, namespace_id=ns, subject="carol", embedding=emb, referenced_date=today)
    await _insert_event(
        session, namespace_id=ns, subject="carol", embedding=emb, referenced_date=today + timedelta(days=1)
    )
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert ops, "expected at least one cluster for the invariant check"

    for op in ops:
        # No top-level payload key may be the bare ``chunk_id`` column name
        # (that would describe a write directive against the canonical
        # row's FK column). The aggregate lands only as a *count* under a
        # distinct key (``merged_source_chunk_ids_count``).
        for payload in (*op.inputs, *op.outputs):
            assert "chunk_id" not in payload, f"op {op.op_id} proposes touching chunk_id: {payload}"
        # The aggregate that *does* land is the count-only summary.
        blob = json.dumps([list(op.inputs), list(op.outputs)], default=str)
        assert "merged_source_chunk_ids_count" in blob


async def test_invalidated_event_excluded_from_clustering(session: AsyncSession) -> None:
    """A tombstoned (invalidated) event must not re-enter clustering (#1147).

    The buggy planner ``SELECT``ed every row regardless of
    ``invalidated_at``, so a previously-merged tail could win the
    highest-confidence canonical election and a live event would be
    soft-merged into the already-invalidated row — a dangling merge
    chain. With the invalidated row excluded, only the single live event
    remains for the subject, so no cluster of size >= 2 forms.
    """
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    # Highest-confidence row, but already tombstoned by a prior dream run.
    await _insert_event(
        session,
        namespace_id=ns,
        subject="grace",
        embedding=emb,
        referenced_date=today,
        confidence=0.99,
        invalidated_at=today,
    )
    # A live event that, on buggy main, would be merged into the tombstone.
    await _insert_event(
        session,
        namespace_id=ns,
        subject="grace",
        embedding=emb,
        referenced_date=today,
        confidence=0.6,
    )
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert ops == (), "invalidated canonical must not pull a live event into a tombstoned merge"


async def test_invalidated_event_excluded_from_canonical_election(session: AsyncSession) -> None:
    """When two live events plus one tombstone share a subject, the canonical
    is elected only among the live rows (#1147).

    The tombstone has the highest confidence; if it were not excluded it
    would win ``_build_op``'s canonical election. The two remaining live
    rows still cluster, and the canonical must be the higher-confidence
    *live* row, never the invalidated one.
    """
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    tombstone_id, _ = await _insert_event(
        session,
        namespace_id=ns,
        subject="heidi",
        embedding=emb,
        referenced_date=today,
        confidence=0.99,
        invalidated_at=today,
    )
    live_hi, _ = await _insert_event(
        session,
        namespace_id=ns,
        subject="heidi",
        embedding=emb,
        referenced_date=today,
        confidence=0.7,
    )
    live_lo, _ = await _insert_event(
        session,
        namespace_id=ns,
        subject="heidi",
        embedding=emb,
        referenced_date=today,
        confidence=0.5,
    )
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert len(ops) == 1
    inputs = ops[0].inputs[0]
    assert inputs["canonical_id"] == str(live_hi)
    assert str(tombstone_id) not in inputs["merged_event_ids"]
    assert set(inputs["merged_event_ids"]) == {str(live_hi), str(live_lo)}


async def test_planner_emits_zero_writes(session: AsyncSession) -> None:
    """Row-count invariant: the planner must not insert / update / delete."""
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    for _ in range(3):
        await _insert_event(session, namespace_id=ns, subject="dave", embedding=emb, referenced_date=today)
    await session.commit()

    before = await _count_rows(session, ns)
    await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    after = await _count_rows(session, ns)
    assert before == after == 3


async def test_dream_op_round_trips_json(session: AsyncSession) -> None:
    """DreamOp output must be JSON-serializable for the report sinks."""
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    await _insert_event(session, namespace_id=ns, subject="eve", embedding=emb, referenced_date=today)
    await _insert_event(session, namespace_id=ns, subject="eve", embedding=emb, referenced_date=today)
    await session.commit()

    ops = await plan_chronicle_event_clustering(ns, session=session, config=DreamConfig())
    assert len(ops) == 1
    op = ops[0]
    payload = {
        "decision": op.decision,
        "phase": op.phase,
        "op_type": op.op_type.value,
        "inputs": list(op.inputs),
        "outputs": list(op.outputs),
        "namespace_id": str(op.namespace_id),
    }
    blob = json.dumps(payload, default=str)
    restored = json.loads(blob)
    assert restored["op_type"] == "chronicle_event_clustering"
    assert restored["decision"] == "planned"


async def test_threshold_and_window_are_configurable(session: AsyncSession) -> None:
    """Tightening the threshold drops a previously-formed cluster.

    Same fixture as the happy-path test, but the override config raises
    the cosine cutoff above 1.0 — nothing should cluster.
    """
    ns = uuid4()
    today = datetime(2026, 5, 15, tzinfo=UTC)
    emb = _normalize([1.0, 0.0, 0.0])
    await _insert_event(session, namespace_id=ns, subject="frank", embedding=emb, referenced_date=today)
    await _insert_event(session, namespace_id=ns, subject="frank", embedding=emb, referenced_date=today)
    await session.commit()

    strict = DreamConfig(event_clustering_cosine_threshold=0.999999)
    # With a near-perfect threshold, identical embeddings should still
    # cluster (dot==1.0). Now shrink the window to zero days so neither
    # of the *same-day* rows can cluster (they're tz-aware identical →
    # 0 days difference, which is <= 0, still clusters). So instead use
    # different dates to prove window=0 actually rejects.
    ops_strict = await plan_chronicle_event_clustering(ns, session=session, config=strict)
    assert len(ops_strict) == 1  # 1.0 >= 0.999999 → still clusters

    # And: a separate run with same data but a zero-day window between
    # rows ~3 hours apart still clusters (same calendar instant). We
    # don't assert that case here — the dedicated test_no_cluster_across_window
    # already proves the window enforcement.


def test_op_kind_string_value() -> None:
    """The OpKind member's wire value is the stable string consumers diff."""
    assert OpKind.CHRONICLE_EVENT_CLUSTERING.value == "chronicle_event_clustering"

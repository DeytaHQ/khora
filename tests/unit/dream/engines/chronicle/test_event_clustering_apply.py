"""Unit tests for ``apply_chronicle_event_clustering`` (#669, Phase 4 of #649).

Drives the apply handler that consumes a planned DreamOp and soft-merges
near-duplicate ``chronicle_events`` into a canonical row via the
bi-temporal columns introduced in migration 034 (``invalidated_at``,
``invalidated_by``, ``merged_into_event_id``).

The tests use an in-memory SQLite database with a hand-rolled
``chronicle_events`` table that mirrors the production schema, plus
the three migration-034 columns.

Invariants under test:

- a happy-path apply soft-merges tail rows into the canonical row,
- re-applying the same op is a no-op (idempotency via WHERE clause),
- the apply NEVER touches ``chunk_id`` (load-bearing FK for temporal
  recall back-pointers),
- the apply NEVER touches the ``documents`` table,
- the apply preserves the canonical row (no self-invalidation),
- the returned ``UndoRecord`` round-trips: applying the inverse SQL
  recorded in ``.before`` restores pre-apply state.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from khora.dream.engines.chronicle.event_clustering import (
    apply_chronicle_event_clustering,
)
from khora.dream.plan import DreamOp, OpKind
from khora.dream.result import UndoRecord

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """In-memory SQLite with ``chronicle_events`` + migration-034 columns."""
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
                        "invalidated_at TEXT, "
                        "invalidated_by TEXT, "
                        "merged_into_event_id TEXT, "
                        "FOREIGN KEY (merged_into_event_id) REFERENCES chronicle_events(id) ON DELETE SET NULL"
                        ")"
                    )
                )
            )
            # Companion documents table — used by the no-mutation invariant.
            await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    sa.text("CREATE TABLE documents (id TEXT PRIMARY KEY, sentinel TEXT NOT NULL)")
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
    subject: str = "alice",
    confidence: float = 1.0,
    event_id: UUID | None = None,
    chunk_id: UUID | None = None,
) -> tuple[UUID, UUID]:
    ev_id = event_id or uuid4()
    ch_id = chunk_id or uuid4()
    now = datetime.now(UTC).isoformat()
    await session.execute(
        sa.text(
            "INSERT INTO chronicle_events "
            "(id, namespace_id, chunk_id, subject, verb, object, "
            "observation_date, referenced_date, confidence, embedding) "
            "VALUES (:id, :ns, :ch, :subj, 'did', '', :obs, :ref, :conf, :emb)"
        ),
        {
            # Seed UUIDs as 32-char hex - the form the sqlite_lance
            # production adapter writes, which is what ``_bind_uuid`` now
            # binds against on SQLite (#1067).
            "id": ev_id.hex,
            "ns": namespace_id.hex,
            "ch": ch_id.hex,
            "subj": subject,
            "obs": now,
            "ref": now,
            "conf": confidence,
            "emb": json.dumps([1.0, 0.0, 0.0]),
        },
    )
    return ev_id, ch_id


async def _row(session: AsyncSession, event_id: UUID) -> dict:
    res = await session.execute(
        sa.text(
            "SELECT id, chunk_id, subject, invalidated_at, invalidated_by, merged_into_event_id "
            "FROM chronicle_events WHERE id = :id"
        ),
        {"id": event_id.hex},
    )
    r = res.first()
    if r is None:
        return {}
    return {
        "id": r.id,
        "chunk_id": r.chunk_id,
        "subject": r.subject,
        "invalidated_at": r.invalidated_at,
        "invalidated_by": r.invalidated_by,
        "merged_into_event_id": r.merged_into_event_id,
    }


def _build_op(
    *,
    canonical_id: UUID,
    tail_ids: list[UUID],
    namespace_id: UUID,
    op_id: UUID | None = None,
) -> DreamOp:
    """Build a planner-shaped DreamOp for the apply path to consume.

    Mirrors the shape produced by ``plan_chronicle_event_clustering``
    (one op per cluster; merged_event_ids includes the canonical id).
    """
    merged = [canonical_id, *tail_ids]
    inputs = (
        {
            "canonical_id": str(canonical_id),
            "merged_event_ids": [str(e) for e in merged],
        },
    )
    outputs = (
        {
            "subject": "alice",
            "cluster_size": len(merged),
            "retained_observation_dates": [],
            "merged_source_chunk_ids_count": 0,
        },
    )
    return DreamOp(
        op_id=op_id or uuid4(),
        phase="apply",
        op_type=OpKind.CHRONICLE_EVENT_CLUSTERING,
        inputs=inputs,
        outputs=outputs,
        decision="planned",
        rationale="test",
        started_at=datetime.now(UTC),
        duration_ms=0.0,
        namespace_id=namespace_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_apply_soft_merges_tail_into_canonical(session: AsyncSession) -> None:
    """Happy path: tail row carries the soft-delete state pointing at canonical."""
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.6)
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_id], namespace_id=ns)
    undo = await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    tail_after = await _row(session, tail_id)
    assert tail_after["invalidated_at"] is not None
    assert UUID(tail_after["invalidated_by"]) == op.op_id
    assert UUID(tail_after["merged_into_event_id"]) == canonical_id
    assert isinstance(undo, UndoRecord)
    assert undo.op_id == op.op_id
    assert undo.op_type == OpKind.CHRONICLE_EVENT_CLUSTERING.value


async def test_apply_is_idempotent_on_already_merged_tails(session: AsyncSession) -> None:
    """Re-applying the same op is a no-op — the WHERE filters already-invalidated."""
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.6)
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_id], namespace_id=ns)
    await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()
    first_state = await _row(session, tail_id)
    first_invalidated_at = first_state["invalidated_at"]

    # Re-apply. Second pass must not update the row again (invalidated_at
    # would otherwise rebump to the new NOW()).
    await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()
    second_state = await _row(session, tail_id)
    assert second_state["invalidated_at"] == first_invalidated_at


async def test_apply_does_not_mutate_chunk_id(session: AsyncSession) -> None:
    """**Load-bearing invariant**: chunk_id stays put.

    The temporal recall channel back-pointer (engine.py:1620) dedupes by
    chunk_id and surfaces the chunk-owning back-link. The apply path must
    only touch invalidation columns — never the FK column.
    """
    ns = uuid4()
    canonical_id, canonical_chunk = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_id, tail_chunk = await _insert_event(session, namespace_id=ns, confidence=0.6)
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_id], namespace_id=ns)
    await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    canonical_after = await _row(session, canonical_id)
    tail_after = await _row(session, tail_id)
    assert UUID(canonical_after["chunk_id"]) == canonical_chunk
    assert UUID(tail_after["chunk_id"]) == tail_chunk


async def test_apply_does_not_touch_documents(session: AsyncSession) -> None:
    """The Documents table is untouched by event clustering."""
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.6)
    sentinel_id = uuid4()
    await session.execute(
        sa.text("INSERT INTO documents (id, sentinel) VALUES (:id, 'untouched')"),
        {"id": str(sentinel_id)},
    )
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_id], namespace_id=ns)
    await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    res = await session.execute(
        sa.text("SELECT sentinel FROM documents WHERE id = :id"),
        {"id": str(sentinel_id)},
    )
    assert res.scalar_one() == "untouched"


async def test_apply_preserves_canonical_event(session: AsyncSession) -> None:
    """The canonical event's row is not invalidated by its own op."""
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.6)
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_id], namespace_id=ns)
    await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    canonical_after = await _row(session, canonical_id)
    assert canonical_after["invalidated_at"] is None
    assert canonical_after["invalidated_by"] is None
    assert canonical_after["merged_into_event_id"] is None


async def test_undo_round_trip(session: AsyncSession) -> None:
    """UndoRecord.before lets a caller restore pre-apply state.

    Manually replays the inverse SQL described by the snapshot and
    asserts the table returns to the original shape.
    """
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_a, _ = await _insert_event(session, namespace_id=ns, confidence=0.5)
    tail_b, _ = await _insert_event(session, namespace_id=ns, confidence=0.4)
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_a, tail_b], namespace_id=ns)
    undo = await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    # Sanity — both tails are invalidated now.
    assert (await _row(session, tail_a))["invalidated_at"] is not None
    assert (await _row(session, tail_b))["invalidated_at"] is not None

    # Replay the inverse from undo.before. The contract: ``clusters`` is the
    # top-level key (never ``chunk_id``), each cluster carries the canonical
    # id + tail ids + previous_states.
    assert "clusters" in undo.before
    assert "chunk_id" not in undo.before  # invariant assert at the undo level
    clusters = undo.before["clusters"]
    assert len(clusters) == 1
    cluster = clusters[0]
    assert UUID(cluster["canonical_id"]) == canonical_id
    assert {UUID(t) for t in cluster["tail_ids"]} == {tail_a, tail_b}

    # Manually replay the inverse — NULL out the three columns on every
    # tail. This is what an undo executor would do.
    for prev in cluster["previous_states"]:
        await session.execute(
            sa.text(
                "UPDATE chronicle_events SET "
                "invalidated_at = :iat, "
                "invalidated_by = :iby, "
                "merged_into_event_id = :mid "
                "WHERE id = :id"
            ),
            {
                "iat": prev["invalidated_at"],
                "iby": prev["invalidated_by"],
                "mid": prev["merged_into_event_id"],
                # Snapshot stores the canonical dashed UUID; SQLite keys the
                # row by 32-char hex, so an undo executor re-binds per-dialect.
                "id": UUID(prev["id"]).hex,
            },
        )
    await session.commit()

    # Both tails are back to their pre-apply state.
    for tid in (tail_a, tail_b):
        row = await _row(session, tid)
        assert row["invalidated_at"] is None
        assert row["invalidated_by"] is None
        assert row["merged_into_event_id"] is None


async def test_apply_handles_multiple_tails(session: AsyncSession) -> None:
    """A cluster of size N invalidates exactly N-1 rows."""
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tails = [(await _insert_event(session, namespace_id=ns, confidence=0.5 - i * 0.05))[0] for i in range(4)]
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=tails, namespace_id=ns)
    await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    # All four tails invalidated.
    for tid in tails:
        row = await _row(session, tid)
        assert row["invalidated_at"] is not None
        assert UUID(row["merged_into_event_id"]) == canonical_id

    # Canonical preserved.
    canonical_after = await _row(session, canonical_id)
    assert canonical_after["invalidated_at"] is None


async def test_apply_undo_top_level_key_is_clusters(session: AsyncSession) -> None:
    """Spec: ``UndoRecord.before`` top-level key MUST be ``clusters``.

    Anything named ``chunk_id`` at that level would imply touching the
    load-bearing FK column. Guard it explicitly.
    """
    ns = uuid4()
    canonical_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.95)
    tail_id, _ = await _insert_event(session, namespace_id=ns, confidence=0.6)
    await session.commit()

    op = _build_op(canonical_id=canonical_id, tail_ids=[tail_id], namespace_id=ns)
    undo = await apply_chronicle_event_clustering(op, coordinator=None, session=session)
    await session.commit()

    assert list(undo.before.keys()) == ["clusters"]

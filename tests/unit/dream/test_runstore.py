"""DreamRunStore abstraction: non-PG run-state + graph_mirror_pending (#1274).

Phase-2 foundation. Run-state (record / checkpoint / status / history /
resume) used to be PostgreSQL-only on the write side; the SQLite-backed
checkpoint and the SurrealDB-unified stack returned no run-state at all.
These tests drive the three :class:`DreamRunStore` impls directly:

  * the SQLite-sidecar store (default for any non-PG SQL stack),
  * the SurrealDB-relational store (unified stack),

and the per-op ``graph_mirror_pending`` accessors the #1272 reconciler
will use to re-attempt committed-but-unmirrored ops. The PostgreSQL impl
is exercised by the existing integration suite (byte-identical SQL).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from khora.dream.runstore import (
    GraphMirrorPending,
    SqliteDreamRunStore,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SQLite-sidecar store
# ---------------------------------------------------------------------------


async def _open_sqlite_store() -> tuple[SqliteDreamRunStore, str]:
    tmp = tempfile.mkdtemp(prefix="khora-runstore-")
    db_path = str(Path(tmp) / "runstore.db")
    store = SqliteDreamRunStore(db_path)
    await store.ensure_schema()
    return store, db_path


async def test_sqlite_record_status_history_roundtrip() -> None:
    store, _ = await _open_sqlite_store()
    ns = uuid4()
    run_id = uuid4()

    await store.record_run(run_id, ns, mode="apply", trigger="manual")
    await store.persist_plan(run_id, plan_hash="abc123", total_ops=3)

    info = await store.status(run_id)
    assert info is not None, "status returned None on sqlite-sidecar"
    assert info.run_id == run_id
    assert info.namespace_id == ns
    assert info.mode == "apply"
    assert info.started_at is not None

    history = await store.history(ns)
    assert len(history) == 1
    assert history[0].run_id == run_id


async def test_sqlite_checkpoint_advance_and_read() -> None:
    """Resume cursor: read_last_committed reflects advance_checkpoint."""
    store, _ = await _open_sqlite_store()
    ns = uuid4()
    run_id = uuid4()
    await store.record_run(run_id, ns, mode="apply")

    assert await store.read_last_committed(run_id) == -1

    await store.advance_checkpoint(run_id, 0)
    assert await store.read_last_committed(run_id) == 0

    await store.advance_checkpoint(run_id, 2)
    assert await store.read_last_committed(run_id) == 2


async def test_sqlite_finalize_sets_finished_at() -> None:
    store, _ = await _open_sqlite_store()
    ns = uuid4()
    run_id = uuid4()
    await store.record_run(run_id, ns, mode="apply")
    await store.finalize_run(run_id, state="completed", total_ops=2)

    info = await store.status(run_id)
    assert info is not None
    assert info.finished_at is not None
    assert info.duration_ms is not None


async def test_sqlite_graph_mirror_pending_set_get_clear() -> None:
    """graph_mirror_pending persists per op and is queryable / clearable."""
    store, _ = await _open_sqlite_store()
    ns = uuid4()
    run_id = uuid4()
    await store.record_run(run_id, ns, mode="apply")

    assert await store.get_graph_mirror_pending(run_id) == []

    op_a = uuid4()
    op_b = uuid4()
    await store.mark_graph_mirror_pending(
        run_id,
        GraphMirrorPending(op_seq=0, op_id=op_a, op_type="vectorcypher_prune_edges", payload={"edge_ids": [1, 2]}),
    )
    await store.mark_graph_mirror_pending(
        run_id,
        GraphMirrorPending(op_seq=1, op_id=op_b, op_type="vectorcypher_dedupe_entities", payload={"absorbed": "x"}),
    )

    pending = await store.get_graph_mirror_pending(run_id)
    assert {p.op_seq for p in pending} == {0, 1}
    by_seq = {p.op_seq: p for p in pending}
    assert by_seq[0].op_id == op_a
    assert by_seq[0].op_type == "vectorcypher_prune_edges"
    assert by_seq[0].payload == {"edge_ids": [1, 2]}
    assert by_seq[1].op_id == op_b

    await store.clear_graph_mirror_pending(run_id, 0)
    remaining = await store.get_graph_mirror_pending(run_id)
    assert {p.op_seq for p in remaining} == {1}


async def test_sqlite_mark_pending_is_idempotent_per_op_seq() -> None:
    """Re-marking the same op_seq replaces rather than duplicates."""
    store, _ = await _open_sqlite_store()
    ns = uuid4()
    run_id = uuid4()
    await store.record_run(run_id, ns, mode="apply")
    op = uuid4()
    entry = GraphMirrorPending(op_seq=0, op_id=op, op_type="t", payload={"v": 1})
    await store.mark_graph_mirror_pending(run_id, entry)
    await store.mark_graph_mirror_pending(run_id, GraphMirrorPending(op_seq=0, op_id=op, op_type="t", payload={"v": 2}))

    pending = await store.get_graph_mirror_pending(run_id)
    assert len(pending) == 1
    assert pending[0].payload == {"v": 2}


# ---------------------------------------------------------------------------
# SurrealDB-relational store
# ---------------------------------------------------------------------------


@pytest.mark.embedded
async def test_surreal_record_status_history_and_mirror_pending() -> None:
    surrealdb = pytest.importorskip("surrealdb")
    del surrealdb
    from khora.dream.runstore import SurrealDreamRunStore
    from khora.storage.backends.surrealdb.connection import SurrealDBConnection

    conn = SurrealDBConnection(mode="memory")
    await conn.connect()
    try:
        store = SurrealDreamRunStore(conn)
        await store.ensure_schema()

        ns = uuid4()
        run_id = uuid4()
        await store.record_run(run_id, ns, mode="apply", trigger="manual")
        await store.persist_plan(run_id, plan_hash="deadbeef", total_ops=2)

        info = await store.status(run_id)
        assert info is not None, "status returned None on surrealdb-unified"
        assert info.run_id == run_id
        assert info.namespace_id == ns
        assert info.mode == "apply"

        history = await store.history(ns)
        assert len(history) == 1
        assert history[0].run_id == run_id

        # Resume cursor round-trips.
        assert await store.read_last_committed(run_id) == -1
        await store.advance_checkpoint(run_id, 1)
        assert await store.read_last_committed(run_id) == 1

        # heartbeat_at must persist (SCHEMAFULL strips fields not DEFINEd in
        # the schema, so the SurrealQL write alone is not enough).
        hb = await conn.query_one(f"SELECT heartbeat_at FROM {store._record(run_id)}")  # noqa: S608 - record id is a UUID
        assert hb is not None and hb.get("heartbeat_at") is not None, "heartbeat_at not persisted on SurrealDB"

        # graph_mirror_pending per op.
        op = uuid4()
        await store.mark_graph_mirror_pending(
            run_id,
            GraphMirrorPending(op_seq=1, op_id=op, op_type="prune", payload={"ids": [9]}),
        )
        pending = await store.get_graph_mirror_pending(run_id)
        assert len(pending) == 1
        assert pending[0].op_seq == 1
        assert pending[0].op_id == op
        assert pending[0].payload == {"ids": [9]}

        await store.clear_graph_mirror_pending(run_id, 1)
        assert await store.get_graph_mirror_pending(run_id) == []
    finally:
        await conn.disconnect()

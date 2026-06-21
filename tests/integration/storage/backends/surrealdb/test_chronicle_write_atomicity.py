"""Regression test for issue #1228: chronicle write_events / write_facts must
be atomic on backend=surrealdb.

Before the fix, ``SurrealDBRelationalAdapter.write_events`` / ``write_facts``
iterated their rows in a plain Python ``for`` loop, issuing one ``CREATE`` per
row through a separate ``await self._conn.execute(...)`` with no transaction
around the loop. A row that failed partway through left the earlier rows
committed and the rest lost -- a partial write with no rollback.

These tests drive the REAL ``SurrealDBRelationalAdapter`` against an in-process
(``mode=memory``) SurrealDB engine and fault-inject a genuine backend failure
partway through the write (a row whose ``CREATE`` evaluates an invalid
SurrealQL expression). After the fix the table is unchanged (all-or-nothing).

No server required -- ``mode=memory`` runs surrealkv in-process.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.engines.chronicle.compression import MemoryFact  # noqa: E402
from khora.engines.chronicle.events import ChronicleEvent  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402

pytestmark = pytest.mark.integration

N_ROWS = 5
FAIL_AT = 3  # the 3rd CREATE is the one that blows up


@pytest.fixture
async def connection() -> SurrealDBConnection:
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="test")
    await conn.connect()
    try:
        yield conn
    finally:
        await conn.disconnect()


async def _count(conn: SurrealDBConnection, table: str) -> int:
    rows = await conn.query(f"SELECT count() AS cnt FROM {table} GROUP ALL")  # noqa: S608
    return int(rows[0].get("cnt", 0)) if rows else 0


async def test_write_events_is_atomic_on_failure(connection: SurrealDBConnection) -> None:
    """A mid-write failure leaves zero rows -- not a partial set (#1228)."""
    rel = SurrealDBRelationalAdapter(connection)
    ns_id = uuid4()
    events = [
        ChronicleEvent(
            id=uuid4(),
            namespace_id=ns_id,
            subject=f"service-{i}",
            verb="emitted",
            object="alert",
            confidence=0.9,
            source_text=f"event {i}",
            observation_date=datetime.now(UTC),
        )
        for i in range(N_ROWS)
    ]

    # Fault-inject a genuine backend failure: rewrite the FAIL_AT-th CREATE's
    # SET clause to evaluate an invalid SurrealQL expression. This raises
    # inside SurrealDB exactly like a real constraint violation would, and
    # exercises the production write path (transaction / execute_batch).
    real_execute = connection.execute
    real_execute_batch = connection.execute_batch
    n = {"create": 0}

    def _poison(sql: str) -> str:
        return sql.replace("confidence = $", "confidence = type::int('boom') + $", 1)

    async def flaky_execute(sql, bindings=None):
        if sql.lstrip().upper().startswith("CREATE"):
            n["create"] += 1
            if n["create"] == FAIL_AT:
                sql = _poison(sql)
        return await real_execute(sql, bindings)

    async def flaky_execute_batch(statements):
        poisoned = []
        for sql, b in statements:
            if sql.lstrip().upper().startswith("CREATE"):
                n["create"] += 1
                if n["create"] == FAIL_AT:
                    sql = _poison(sql)
            poisoned.append((sql, b))
        return await real_execute_batch(poisoned)

    connection.execute = flaky_execute  # type: ignore[method-assign]
    connection.execute_batch = flaky_execute_batch  # type: ignore[method-assign]

    raised = None
    try:
        await rel.write_events(events, namespace_id=ns_id)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        connection.execute = real_execute  # type: ignore[method-assign]
        connection.execute_batch = real_execute_batch  # type: ignore[method-assign]

    assert raised is not None, "the injected failure must surface to the caller"
    survived = await _count(connection, "chronicle_event")
    assert survived == 0, f"write_events must be all-or-nothing; {survived} partial rows survived"


async def test_write_facts_is_atomic_on_failure(connection: SurrealDBConnection) -> None:
    """A mid-write failure leaves zero rows -- not a partial set (#1228)."""
    rel = SurrealDBRelationalAdapter(connection)
    ns_id = uuid4()
    facts = [
        MemoryFact(
            id=uuid4(),
            namespace_id=ns_id,
            subject=f"subject-{i}",
            predicate="is",
            object_="value",
            fact_text=f"fact {i}",
            confidence=0.9,
        )
        for i in range(N_ROWS)
    ]

    real_execute = connection.execute
    real_execute_batch = connection.execute_batch
    n = {"create": 0}

    def _poison(sql: str) -> str:
        return sql.replace("confidence = $", "confidence = type::int('boom') + $", 1)

    async def flaky_execute(sql, bindings=None):
        if sql.lstrip().upper().startswith("CREATE"):
            n["create"] += 1
            if n["create"] == FAIL_AT:
                sql = _poison(sql)
        return await real_execute(sql, bindings)

    async def flaky_execute_batch(statements):
        poisoned = []
        for sql, b in statements:
            if sql.lstrip().upper().startswith("CREATE"):
                n["create"] += 1
                if n["create"] == FAIL_AT:
                    sql = _poison(sql)
            poisoned.append((sql, b))
        return await real_execute_batch(poisoned)

    connection.execute = flaky_execute  # type: ignore[method-assign]
    connection.execute_batch = flaky_execute_batch  # type: ignore[method-assign]

    raised = None
    try:
        await rel.write_facts(facts, namespace_id=ns_id)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        connection.execute = real_execute  # type: ignore[method-assign]
        connection.execute_batch = real_execute_batch  # type: ignore[method-assign]

    assert raised is not None, "the injected failure must surface to the caller"
    survived = await _count(connection, "memory_fact")
    assert survived == 0, f"write_facts must be all-or-nothing; {survived} partial rows survived"


async def test_write_events_happy_path_still_persists(connection: SurrealDBConnection) -> None:
    """Atomicity must not break the success path: all rows land, IDs returned in order."""
    rel = SurrealDBRelationalAdapter(connection)
    ns_id = uuid4()
    events = [
        ChronicleEvent(
            id=uuid4(),
            namespace_id=ns_id,
            subject=f"service-{i}",
            verb="emitted",
            object="alert",
            confidence=0.9,
            source_text=f"event {i}",
            observation_date=datetime.now(UTC),
        )
        for i in range(N_ROWS)
    ]

    ids = await rel.write_events(events, namespace_id=ns_id)

    assert ids == [ev.id for ev in events]
    assert await _count(connection, "chronicle_event") == N_ROWS


async def test_write_facts_happy_path_still_persists(connection: SurrealDBConnection) -> None:
    rel = SurrealDBRelationalAdapter(connection)
    ns_id = uuid4()
    facts = [
        MemoryFact(
            id=uuid4(),
            namespace_id=ns_id,
            subject=f"subject-{i}",
            predicate="is",
            object_="value",
            fact_text=f"fact {i}",
            confidence=0.9,
        )
        for i in range(N_ROWS)
    ]

    ids = await rel.write_facts(facts, namespace_id=ns_id)

    assert ids == [f.id for f in facts]
    assert await _count(connection, "memory_fact") == N_ROWS

"""Regression test for issue #712: chronicle silently drops events and facts on SurrealDB.

Before the fix, ``SurrealDBRelationalAdapter`` did not implement
``write_events`` / ``write_facts`` / ``query_active_facts_for_subject`` /
``query_events`` / ``supersede_fact``. The neither did the SurrealDB
vector adapter. ``StorageCoordinator._chronicle_backend`` then raised
``RuntimeError("No backend supports chronicle method ...")``, which the
chronicle engine caught as a generic ``Exception`` and downgraded to a
WARNING log — silently dropping every extracted event and fact.

This test exercises the storage-layer dispatch directly against an
in-memory SurrealDB. It mirrors the sqlite_lance fix in PR #528 (issue
#529) but for the SurrealDB unified backend.
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
from khora.storage.coordinator import StorageCoordinator  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
async def coordinator() -> StorageCoordinator:
    """A storage coordinator wired to an in-memory SurrealDB instance.

    The coordinator only needs the relational adapter populated for the
    chronicle dispatch path under test — that path picks ``self.vector``
    first, then ``self.relational``. The SurrealDB vector adapter does
    not implement chronicle methods, so dispatch falls through to the
    relational adapter (this is the path #712 fixes).
    """
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="test")
    await conn.connect()
    adapter = SurrealDBRelationalAdapter(conn)
    coord = StorageCoordinator(relational=adapter)
    try:
        yield coord
    finally:
        await conn.disconnect()


async def test_write_events_persists_on_surrealdb(coordinator: StorageCoordinator) -> None:
    """Issue #712 repro: ``coord.write_events`` must persist on SurrealDB.

    Before the fix this raised ``RuntimeError: No backend supports
    chronicle method 'write_events'`` from ``_chronicle_backend`` because
    neither the vector nor relational SurrealDB adapter exposed the
    method.
    """
    namespace_id = uuid4()
    chunk_id = uuid4()
    event = ChronicleEvent(
        id=uuid4(),
        chunk_id=chunk_id,
        namespace_id=namespace_id,
        subject="Marie Curie",
        verb="won",
        object="Nobel Prize",
        observation_date=datetime.now(UTC),
        referenced_date=datetime(1903, 12, 10, tzinfo=UTC),
        confidence=0.95,
        source_text="Marie Curie won the Nobel Prize in 1903.",
    )

    ids = await coordinator.write_events([event], namespace_id=namespace_id)

    assert ids == [event.id]


async def test_write_facts_persists_on_surrealdb(coordinator: StorageCoordinator) -> None:
    """Issue #712 repro for facts. Same dispatch failure as events."""
    namespace_id = uuid4()
    chunk_id = uuid4()
    fact = MemoryFact(
        id=uuid4(),
        namespace_id=namespace_id,
        subject="Marie Curie",
        predicate="won",
        object_="Nobel Prize",
        fact_text="Marie Curie won the Nobel Prize.",
        confidence=0.95,
        source_chunk_ids=[chunk_id],
    )

    ids = await coordinator.write_facts([fact], namespace_id=namespace_id)

    assert ids == [fact.id]


async def test_query_active_facts_for_subject_on_surrealdb(coordinator: StorageCoordinator) -> None:
    """Issue #712 repro for fact reconciliation reads.

    The chronicle engine's ``_reconcile_facts`` path calls this method to
    look up active facts for a subject before writing new ones; it was
    failing exactly the same way as write_events / write_facts.
    """
    namespace_id = uuid4()
    fact = MemoryFact(
        id=uuid4(),
        namespace_id=namespace_id,
        subject="Marie Curie",
        predicate="discovered",
        object_="Radium",
        fact_text="Marie Curie discovered radium.",
        confidence=0.95,
    )
    await coordinator.write_facts([fact], namespace_id=namespace_id)

    rows = await coordinator.query_active_facts_for_subject(namespace_id, "Marie Curie")

    assert any(getattr(r, "subject", None) == "Marie Curie" for r in rows)


async def test_supersede_fact_on_surrealdb(coordinator: StorageCoordinator) -> None:
    """Reconciliation also calls ``supersede_fact`` when a new fact replaces an old one."""
    namespace_id = uuid4()
    old = MemoryFact(
        id=uuid4(),
        namespace_id=namespace_id,
        subject="Pluto",
        predicate="is_a",
        object_="planet",
        fact_text="Pluto is a planet.",
    )
    new = MemoryFact(
        id=uuid4(),
        namespace_id=namespace_id,
        subject="Pluto",
        predicate="is_a",
        object_="dwarf planet",
        fact_text="Pluto is a dwarf planet.",
    )
    await coordinator.write_facts([old, new], namespace_id=namespace_id)

    await coordinator.supersede_fact(old.id, new.id, namespace_id=namespace_id)

    active = await coordinator.query_active_facts_for_subject(namespace_id, "Pluto")
    active_ids = {getattr(r, "id", None) for r in active}
    assert old.id not in active_ids, "superseded fact must not appear in active facts"
    assert new.id in active_ids

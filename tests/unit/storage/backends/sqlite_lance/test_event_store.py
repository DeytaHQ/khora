"""Tests for :class:`SQLiteLanceEventStoreAdapter` (DYT-2731)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

from khora.core.models.event import EventType, MemoryEvent

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="aiosqlite/lancedb not installed")

if _HAS_DEPS:
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.event_store import (
        SQLiteLanceEventStoreAdapter,
    )


# ---------------------------------------------------------------------------
# Inline DDL mirrors the ``memory_events`` table produced by Alembic
# migration ``000_initial_schema`` under the SQLite dialect gate.  DYT-2727
# lands the full migration; tests here keep the surface minimal to isolate
# the event-store adapter from migration churn.
# ---------------------------------------------------------------------------

_EVENTS_DDL = """\
CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    data TEXT,
    previous_data TEXT,
    actor_id TEXT,
    actor_type TEXT DEFAULT 'system',
    correlation_id TEXT,
    version INTEGER DEFAULT 1,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_resource ON memory_events(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS ix_events_namespace_timestamp ON memory_events(namespace_id, timestamp);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def handle(tmp_path: Path):
    cfg = EmbeddedStorageHandleConfig(
        db_path=str(tmp_path / "events.db"),
        lance_path=str(tmp_path / "events.lance"),
        embedding_dimension=8,
        use_halfvec=False,
    )
    h = EmbeddedStorageHandle(cfg)
    await h.connect()
    # FK checks would require a memory_namespaces row; skip for this adapter's
    # append-only tests (the real schema enforces FK at the DB layer).
    await h.sqlite.execute("PRAGMA foreign_keys=OFF")
    for stmt in _EVENTS_DDL.split(";"):
        s = stmt.strip()
        if s:
            await h.sqlite.execute(s)
    await h.sqlite.commit()
    try:
        yield h
    finally:
        await h.disconnect()


@pytest.fixture
def adapter(handle: EmbeddedStorageHandle) -> SQLiteLanceEventStoreAdapter:
    return SQLiteLanceEventStoreAdapter(handle)


@pytest.fixture
def ns() -> UUID:
    return uuid4()


def _make_event(
    *,
    namespace_id: UUID,
    event_type: EventType = EventType.DOCUMENT_CREATED,
    resource_id: UUID | None = None,
    timestamp: datetime | None = None,
    data: dict | None = None,
    previous_data: dict | None = None,
    metadata: dict | None = None,
    correlation_id: UUID | None = None,
    actor_id: str | None = None,
    actor_type: str = "system",
    version: int = 1,
) -> MemoryEvent:
    return MemoryEvent(
        id=uuid4(),
        namespace_id=namespace_id,
        event_type=event_type,
        timestamp=timestamp or datetime.now(UTC),
        resource_type=event_type.value.split(".")[0],
        resource_id=resource_id or uuid4(),
        data=data or {"k": "v"},
        previous_data=previous_data,
        actor_id=actor_id,
        actor_type=actor_type,
        correlation_id=correlation_id,
        version=version,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_append_event_then_list(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    evt = _make_event(namespace_id=ns, data={"title": "hello"})
    returned = await adapter.append_event(evt)
    assert returned.id == evt.id

    rows = await adapter.get_events(ns, limit=10)
    assert len(rows) == 1
    got = rows[0]
    assert got.id == evt.id
    assert got.namespace_id == ns
    assert got.event_type == EventType.DOCUMENT_CREATED
    assert got.resource_type == "document"
    assert got.data == {"title": "hello"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_append_events_batch_with_pagination(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    base = datetime.now(UTC)
    events = [
        _make_event(
            namespace_id=ns,
            timestamp=base + timedelta(seconds=i),
            data={"i": i},
        )
        for i in range(50)
    ]
    returned = await adapter.append_events_batch(events)
    assert len(returned) == 50

    total = await adapter.count_events(ns)
    assert total == 50

    first = await adapter.get_events(ns, limit=20, offset=0)
    second = await adapter.get_events(ns, limit=20, offset=20)
    third = await adapter.get_events(ns, limit=20, offset=40)
    assert len(first) == 20
    assert len(second) == 20
    assert len(third) == 10

    # DESC order by timestamp — newest first. i=49 should come first.
    assert first[0].data == {"i": 49}
    assert first[-1].data == {"i": 30}
    assert third[-1].data == {"i": 0}

    # No overlap.
    ids = {e.id for e in first} | {e.id for e in second} | {e.id for e in third}
    assert len(ids) == 50


@pytest.mark.unit
@pytest.mark.asyncio
async def test_append_events_batch_empty(adapter: SQLiteLanceEventStoreAdapter) -> None:
    assert await adapter.append_events_batch([]) == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_events_since_boundary(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    events = [_make_event(namespace_id=ns, timestamp=t0 + timedelta(hours=i)) for i in range(5)]
    await adapter.append_events_batch(events)

    # Strict > boundary — the exact boundary is excluded.
    cutoff = t0 + timedelta(hours=2)
    after = await adapter.get_events(ns, after=cutoff, limit=100)
    assert len(after) == 2  # hours 3, 4

    before = await adapter.get_events(ns, before=cutoff, limit=100)
    assert len(before) == 2  # hours 0, 1

    between = await adapter.get_events(
        ns,
        after=t0 + timedelta(minutes=30),
        before=t0 + timedelta(hours=3, minutes=30),
        limit=100,
    )
    # hours 1, 2, 3
    assert len(between) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_by_event_types(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    await adapter.append_events_batch(
        [
            _make_event(namespace_id=ns, event_type=EventType.DOCUMENT_CREATED),
            _make_event(namespace_id=ns, event_type=EventType.DOCUMENT_UPDATED),
            _make_event(namespace_id=ns, event_type=EventType.ENTITY_CREATED),
            _make_event(namespace_id=ns, event_type=EventType.ENTITY_CREATED),
            _make_event(namespace_id=ns, event_type=EventType.CHUNK_CREATED),
        ]
    )

    # Filter by a single type — passing strings and EventType both work.
    created = await adapter.get_events(ns, event_types=[EventType.ENTITY_CREATED], limit=100)
    assert len(created) == 2
    assert all(e.event_type == EventType.ENTITY_CREATED for e in created)

    # Filter by multiple types using raw string values.
    docs = await adapter.get_events(ns, event_types=["document.created", "document.updated"], limit=100)
    assert len(docs) == 2
    assert {e.event_type for e in docs} == {EventType.DOCUMENT_CREATED, EventType.DOCUMENT_UPDATED}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_count_by_namespace_and_type(adapter: SQLiteLanceEventStoreAdapter) -> None:
    ns_a = uuid4()
    ns_b = uuid4()
    await adapter.append_events_batch(
        [
            _make_event(namespace_id=ns_a, event_type=EventType.DOCUMENT_CREATED),
            _make_event(namespace_id=ns_a, event_type=EventType.DOCUMENT_CREATED),
            _make_event(namespace_id=ns_a, event_type=EventType.ENTITY_CREATED),
            _make_event(namespace_id=ns_b, event_type=EventType.DOCUMENT_CREATED),
        ]
    )

    assert await adapter.count_events(ns_a) == 3
    assert await adapter.count_events(ns_b) == 1
    assert await adapter.count_events(ns_a, event_types=[EventType.DOCUMENT_CREATED]) == 2
    assert await adapter.count_events(ns_b, event_types=[EventType.ENTITY_CREATED]) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_count_with_after(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    await adapter.append_events_batch(
        [_make_event(namespace_id=ns, timestamp=t0 + timedelta(minutes=m)) for m in range(10)]
    )
    # strict >
    assert await adapter.count_events(ns, after=t0 + timedelta(minutes=4)) == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_append_only_no_mutation_methods() -> None:
    """Adapter must not expose update/delete/mutate helpers."""
    forbidden = {"update_event", "delete_event", "upsert_event", "remove_event", "patch_event"}
    attrs = {name for name in dir(SQLiteLanceEventStoreAdapter) if not name.startswith("_")}
    assert forbidden.isdisjoint(attrs), f"unexpected mutation API: {forbidden & attrs}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_json_payload_roundtrip(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    payload = {
        "unicode": "héllo ✨ 世界",
        "nested": {"a": [1, 2, {"b": None}], "c": True},
        "null_field": None,
        "empty_list": [],
        "empty_dict": {},
    }
    prev = {"old": "state", "count": 0}
    meta = {"trace_id": "abc-123", "tags": ["t1", "t2"]}
    evt = _make_event(
        namespace_id=ns,
        data=payload,
        previous_data=prev,
        metadata=meta,
    )
    await adapter.append_event(evt)

    rows = await adapter.get_events(ns, limit=1)
    assert len(rows) == 1
    got = rows[0]
    assert got.data == payload
    assert got.previous_data == prev
    assert got.metadata == meta


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_events_for_resource(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    target = uuid4()
    other = uuid4()
    t0 = datetime(2026, 6, 1, tzinfo=UTC)
    events = [
        _make_event(
            namespace_id=ns,
            event_type=EventType.DOCUMENT_CREATED,
            resource_id=target,
            timestamp=t0,
        ),
        _make_event(
            namespace_id=ns,
            event_type=EventType.DOCUMENT_UPDATED,
            resource_id=target,
            timestamp=t0 + timedelta(hours=1),
        ),
        _make_event(
            namespace_id=ns,
            event_type=EventType.DOCUMENT_CREATED,
            resource_id=other,
            timestamp=t0 + timedelta(hours=2),
        ),
    ]
    await adapter.append_events_batch(events)

    got = await adapter.get_events_for_resource("document", target, limit=10)
    assert len(got) == 2
    assert got[0].event_type == EventType.DOCUMENT_UPDATED  # newest first
    assert got[1].event_type == EventType.DOCUMENT_CREATED
    assert all(e.resource_id == target for e in got)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_latest_event(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    target = uuid4()
    t0 = datetime(2026, 7, 1, tzinfo=UTC)
    await adapter.append_events_batch(
        [
            _make_event(
                namespace_id=ns,
                event_type=EventType.DOCUMENT_CREATED,
                resource_id=target,
                timestamp=t0,
            ),
            _make_event(
                namespace_id=ns,
                event_type=EventType.DOCUMENT_UPDATED,
                resource_id=target,
                timestamp=t0 + timedelta(hours=1),
            ),
        ]
    )

    latest = await adapter.get_latest_event("document", target)
    assert latest is not None
    assert latest.event_type == EventType.DOCUMENT_UPDATED

    # Unknown resource -> None.
    missing = await adapter.get_latest_event("document", uuid4())
    assert missing is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_correlation_id_and_nullable_fields(adapter: SQLiteLanceEventStoreAdapter, ns: UUID) -> None:
    cid = uuid4()
    evt = _make_event(
        namespace_id=ns,
        correlation_id=cid,
        actor_id="user-42",
        actor_type="user",
        version=7,
    )
    await adapter.append_event(evt)

    rows = await adapter.get_events(ns, limit=1)
    got = rows[0]
    assert got.correlation_id == cid
    assert got.actor_id == "user-42"
    assert got.actor_type == "user"
    assert got.version == 7
    # No previous_data supplied -> None round-trip.
    assert got.previous_data is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connection_lifecycle_delegated(
    tmp_path: Path,
) -> None:
    """connect/disconnect/is_healthy delegate to the shared handle."""
    cfg = EmbeddedStorageHandleConfig(
        db_path=str(tmp_path / "lc.db"),
        lance_path=str(tmp_path / "lc.lance"),
        embedding_dimension=4,
    )
    h = EmbeddedStorageHandle(cfg)
    adapter = SQLiteLanceEventStoreAdapter(h)

    assert await adapter.is_healthy() is False
    await adapter.connect()
    try:
        assert await adapter.is_healthy() is True
    finally:
        await adapter.disconnect()
    assert await adapter.is_healthy() is False

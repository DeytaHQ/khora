"""Coverage tests for khora.storage.event_store.

Exercises connect/disconnect lifecycle, URL normalisation, and the
session-bearing append/get/count operations using mocked async sessions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import MemoryEvent
from khora.core.models.event import EventType
from khora.storage.event_store import PostgreSQLEventStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Result:
    def __init__(
        self,
        *,
        scalar_one_or_none: Any = None,
        scalar_one: Any = None,
        scalars_all: list[Any] | None = None,
    ) -> None:
        self._scalar_one_or_none = scalar_one_or_none
        self._scalar_one = scalar_one
        self._scalars_all = scalars_all or []

    def scalar_one_or_none(self) -> Any:
        return self._scalar_one_or_none

    def scalar_one(self) -> Any:
        return self._scalar_one

    def scalars(self) -> Any:
        m = MagicMock()
        m.all = MagicMock(return_value=self._scalars_all)
        return m


class _Session:
    """Bare-minimum mock session implementing the methods used by event_store."""

    def __init__(self, results: list[_Result]) -> None:
        self._results = list(results)
        self.added: list[Any] = []
        self.added_all: list[list[Any]] = []
        self.commits = 0
        self.flushes = 0

    async def execute(self, *_args: Any, **_kwargs: Any) -> _Result:
        if not self._results:
            return _Result()
        return self._results.pop(0)

    def add(self, item: Any) -> None:
        self.added.append(item)

    def add_all(self, items: list[Any]) -> None:
        self.added_all.append(items)

    async def commit(self) -> None:
        self.commits += 1

    async def flush(self) -> None:
        self.flushes += 1

    async def refresh(self, model: Any) -> None:
        # Identity refresh: nothing to update.
        return None


class _SessionCtx:
    def __init__(self, session: _Session) -> None:
        self._session = session

    async def __aenter__(self) -> _Session:
        return self._session

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _attach_factory(store: PostgreSQLEventStore, session: _Session) -> None:
    """Wire the store's ``_session_factory`` so ``_get_session()`` returns a
    context that yields ``session``."""
    store._session_factory = MagicMock(return_value=_SessionCtx(session))  # type: ignore[assignment]


def _make_event(**overrides: Any) -> MemoryEvent:
    base = dict(
        id=uuid4(),
        namespace_id=uuid4(),
        event_type=EventType.DOCUMENT_CREATED,
        timestamp=datetime.now(UTC),
        resource_id=uuid4(),
        data={"k": "v"},
    )
    base.update(overrides)
    return MemoryEvent(**base)


# ---------------------------------------------------------------------------
# __init__ URL normalisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_postgresql_url_rewritten_to_asyncpg() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    assert s._database_url.startswith("postgresql+asyncpg://")


@pytest.mark.unit
def test_postgres_url_rewritten_to_asyncpg() -> None:
    s = PostgreSQLEventStore("postgres://x/y")
    assert s._database_url.startswith("postgresql+asyncpg://")


@pytest.mark.unit
def test_other_url_passes_through() -> None:
    s = PostgreSQLEventStore("postgresql+asyncpg://x/y")
    assert s._database_url == "postgresql+asyncpg://x/y"


@pytest.mark.unit
def test_engine_shared_flag_tracks_constructor_arg() -> None:
    engine = MagicMock()
    s = PostgreSQLEventStore("postgresql://x/y", engine=engine)
    assert s._engine_shared is True
    assert s._engine is engine


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connect_creates_engine_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = MagicMock()
    monkeypatch.setattr("khora.storage.event_store.create_async_engine", lambda *a, **kw: fake_engine)

    s = PostgreSQLEventStore("postgresql://x/y")
    await s.connect()
    assert s._engine is fake_engine
    assert s._session_factory is not None


@pytest.mark.unit
async def test_connect_is_noop_when_already_connected() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    s._session_factory = MagicMock()  # type: ignore[assignment]
    s._engine = MagicMock()
    before = s._engine
    await s.connect()
    assert s._engine is before  # unchanged


@pytest.mark.unit
async def test_disconnect_disposes_when_not_shared() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    s._engine = fake_engine
    s._engine_shared = False
    s._session_factory = MagicMock()  # type: ignore[assignment]

    await s.disconnect()

    fake_engine.dispose.assert_awaited()
    assert s._engine is None
    assert s._session_factory is None


@pytest.mark.unit
async def test_disconnect_does_not_dispose_shared_engine() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    s._engine = fake_engine
    s._engine_shared = True
    s._session_factory = MagicMock()  # type: ignore[assignment]

    await s.disconnect()

    fake_engine.dispose.assert_not_called()


@pytest.mark.unit
async def test_disconnect_when_already_disconnected_is_safe() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    # No engine attached.
    await s.disconnect()  # must not raise


# ---------------------------------------------------------------------------
# is_healthy
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_is_healthy_false_when_not_connected() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    assert await s.is_healthy() is False


@pytest.mark.unit
async def test_is_healthy_true_when_select_succeeds() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    s._engine = MagicMock()
    session = _Session(results=[_Result()])
    s._session_factory = MagicMock(return_value=_SessionCtx(session))  # type: ignore[assignment]
    assert await s.is_healthy() is True


@pytest.mark.unit
async def test_is_healthy_false_when_select_raises() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    s._engine = MagicMock()
    bad_session = _Session(results=[])
    bad_session.execute = AsyncMock(side_effect=RuntimeError("db down"))  # type: ignore[method-assign]
    s._session_factory = MagicMock(return_value=_SessionCtx(bad_session))  # type: ignore[assignment]
    assert await s.is_healthy() is False


# ---------------------------------------------------------------------------
# create_tables (deprecated path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_tables_raises_when_not_connected() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    with pytest.raises(RuntimeError, match="not connected"):
        with pytest.warns(DeprecationWarning):
            await s.create_tables()


# ---------------------------------------------------------------------------
# append_event / append_events_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_append_event_with_own_session() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    session = _Session(results=[])
    _attach_factory(s, session)

    event = _make_event()
    result = await s.append_event(event)

    assert session.commits == 1
    assert len(session.added) == 1
    assert result.id == event.id


@pytest.mark.unit
async def test_append_event_with_passed_session_does_not_commit() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    passed_session = _Session(results=[])

    event = _make_event()
    result = await s.append_event(event, session=passed_session)  # type: ignore[arg-type]

    # ``commit`` must NOT have been called — the caller owns the transaction.
    assert passed_session.commits == 0
    assert passed_session.flushes == 1
    assert result.id == event.id


@pytest.mark.unit
async def test_append_events_batch_empty_returns_empty() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    assert await s.append_events_batch([]) == []


@pytest.mark.unit
async def test_append_events_batch_with_own_session() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    session = _Session(results=[])
    _attach_factory(s, session)

    events = [_make_event(), _make_event()]
    out = await s.append_events_batch(events)

    assert out == events
    assert len(session.added_all) == 1
    assert len(session.added_all[0]) == 2
    assert session.commits == 1


@pytest.mark.unit
async def test_append_events_batch_with_passed_session_does_not_commit() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    passed_session = _Session(results=[])

    events = [_make_event()]
    out = await s.append_events_batch(events, session=passed_session)  # type: ignore[arg-type]

    assert out == events
    assert passed_session.commits == 0


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_events_applies_all_filters() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")

    model = MagicMock()
    model.id = uuid4()
    model.namespace_id = uuid4()
    model.event_type = "document.created"
    model.timestamp = datetime.now(UTC)
    model.resource_type = "document"
    model.resource_id = uuid4()
    model.data = {}
    model.previous_data = None
    model.actor_id = None
    model.actor_type = "system"
    model.correlation_id = None
    model.version = 1
    model.metadata_ = {}

    session = _Session(results=[_Result(scalars_all=[model])])
    _attach_factory(s, session)

    results = await s.get_events(
        uuid4(),
        event_types=["document.created"],
        resource_type="document",
        resource_id=uuid4(),
        after=datetime.now(UTC),
        before=datetime.now(UTC),
        limit=10,
        offset=5,
    )
    assert len(results) == 1
    assert results[0].event_type == EventType.DOCUMENT_CREATED


@pytest.mark.unit
async def test_get_events_for_resource() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    model = MagicMock()
    model.id = uuid4()
    model.namespace_id = uuid4()
    model.event_type = EventType.ENTITY_CREATED  # exercise enum-passthrough branch
    model.timestamp = datetime.now(UTC)
    model.resource_type = "entity"
    model.resource_id = uuid4()
    model.data = {}
    model.previous_data = None
    model.actor_id = None
    model.actor_type = "system"
    model.correlation_id = None
    model.version = 1
    model.metadata_ = {}

    session = _Session(results=[_Result(scalars_all=[model])])
    _attach_factory(s, session)

    out = await s.get_events_for_resource("entity", uuid4(), limit=50)
    assert len(out) == 1
    assert out[0].event_type == EventType.ENTITY_CREATED


@pytest.mark.unit
async def test_get_latest_event_returns_model_when_found() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    model = MagicMock()
    model.id = uuid4()
    model.namespace_id = uuid4()
    model.event_type = "chunk.created"
    model.timestamp = datetime.now(UTC)
    model.resource_type = "chunk"
    model.resource_id = uuid4()
    model.data = {}
    model.previous_data = None
    model.actor_id = None
    model.actor_type = "system"
    model.correlation_id = None
    model.version = 1
    model.metadata_ = {}

    session = _Session(results=[_Result(scalar_one_or_none=model)])
    _attach_factory(s, session)

    out = await s.get_latest_event("chunk", uuid4())
    assert out is not None
    assert out.event_type == EventType.CHUNK_CREATED


@pytest.mark.unit
async def test_get_latest_event_returns_none() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    session = _Session(results=[_Result(scalar_one_or_none=None)])
    _attach_factory(s, session)

    out = await s.get_latest_event("chunk", uuid4())
    assert out is None


# ---------------------------------------------------------------------------
# count_events
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_count_events_no_filters() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    session = _Session(results=[_Result(scalar_one=42)])
    _attach_factory(s, session)
    assert await s.count_events(uuid4()) == 42


@pytest.mark.unit
async def test_count_events_with_filters() -> None:
    s = PostgreSQLEventStore("postgresql://x/y")
    session = _Session(results=[_Result(scalar_one=3)])
    _attach_factory(s, session)
    assert (await s.count_events(uuid4(), event_types=["document.created"], after=datetime.now(UTC))) == 3

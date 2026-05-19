"""Coverage tests for khora.storage.expertise_store.

Exercises every CRUD path with a mocked SQLAlchemy AsyncSession.  No real DB.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from khora.extraction.skills import ExpertiseConfig
from khora.storage.expertise_store import ExpertiseStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(*, scalar_one_or_none: Any = None, scalars_all: list[Any] | None = None, rowcount: int = 0) -> Any:
    """Mock ``sqlalchemy.engine.Result`` with the methods used by the store."""
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=scalar_one_or_none)

    scalars = MagicMock()
    scalars.all = MagicMock(return_value=scalars_all or [])
    result.scalars = MagicMock(return_value=scalars)

    result.rowcount = rowcount
    return result


def _make_session(*, exec_results: list[Any]) -> Any:
    """Return a fake AsyncSession whose ``execute`` returns the scripted results in order."""
    session = MagicMock()
    queue = list(exec_results)

    async def _execute(*_args: Any, **_kwargs: Any) -> Any:
        if not queue:
            raise AssertionError("execute() called more times than scripted")
        return queue.pop(0)

    session.execute = _execute
    session.add = MagicMock()
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.close = AsyncMock()
    return session


class _SessionCtx:
    """Plain async-context-manager wrapper around a target session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *args: Any) -> None:
        return None


def _make_storage_with_session(session: Any) -> Any:
    """Build a fake storage coordinator whose ``relational._get_session()``
    returns an object whose ``__aenter__`` resolves to ``session``."""
    storage = MagicMock()
    storage.relational._get_session = MagicMock(return_value=_SessionCtx(session))
    return storage


def _make_storage_without_relational() -> Any:
    storage = MagicMock()
    storage.relational = None
    return storage


def _make_expertise(name: str = "expert_x") -> ExpertiseConfig:
    return ExpertiseConfig(name=name, version="1.0.0", description="desc")


# ---------------------------------------------------------------------------
# save() — both branches (existing + new)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_save_raises_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    with pytest.raises(RuntimeError, match="Relational storage not configured"):
        await store.save(uuid4(), _make_expertise())


@pytest.mark.unit
async def test_save_creates_new_when_missing() -> None:
    expertise = _make_expertise()
    new_id = uuid4()

    # First execute returns existing=None.
    first_result = _make_result(scalar_one_or_none=None)
    session = _make_session(exec_results=[first_result])

    async def _refresh(model: Any) -> None:
        model.id = str(new_id)

    session.refresh = _refresh

    store = ExpertiseStore(_make_storage_with_session(session))

    result_id = await store.save(uuid4(), expertise)

    assert result_id == new_id
    assert session.add.called
    session.commit.assert_awaited()


@pytest.mark.unit
async def test_save_updates_existing() -> None:
    expertise = _make_expertise()
    existing = MagicMock()
    existing.id = str(uuid4())

    result_obj = _make_result(scalar_one_or_none=existing)
    session = _make_session(exec_results=[result_obj])

    store = ExpertiseStore(_make_storage_with_session(session))

    returned = await store.save(uuid4(), expertise, set_active=False)

    assert returned == UUID(existing.id)
    assert existing.version == expertise.version
    assert existing.description == expertise.description
    assert existing.is_active is False
    session.commit.assert_awaited()


# ---------------------------------------------------------------------------
# get() — three return paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_returns_none_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    assert await store.get(uuid4(), "x") is None


@pytest.mark.unit
async def test_get_returns_none_when_not_found() -> None:
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=None)])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.get(uuid4(), "missing") is None


@pytest.mark.unit
async def test_get_returns_expertise_when_found() -> None:
    expertise = _make_expertise("found")
    model = MagicMock(config=expertise.to_dict())
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=model)])
    store = ExpertiseStore(_make_storage_with_session(session))

    got = await store.get(uuid4(), "found")
    assert got is not None
    assert got.name == "found"


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_by_id_returns_none_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    assert await store.get_by_id(uuid4()) is None


@pytest.mark.unit
async def test_get_by_id_returns_none_when_not_found() -> None:
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=None)])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.get_by_id(uuid4()) is None


@pytest.mark.unit
async def test_get_by_id_returns_expertise_when_found() -> None:
    expertise = _make_expertise("by_id")
    model = MagicMock(config=expertise.to_dict())
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=model)])
    store = ExpertiseStore(_make_storage_with_session(session))
    got = await store.get_by_id(uuid4())
    assert got is not None
    assert got.name == "by_id"


# ---------------------------------------------------------------------------
# get_active / get_by_namespace
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_active_returns_none_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    assert await store.get_active(uuid4()) is None


@pytest.mark.unit
async def test_get_active_returns_none_when_not_set() -> None:
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=None)])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.get_active(uuid4()) is None


@pytest.mark.unit
async def test_get_active_returns_expertise() -> None:
    expertise = _make_expertise("active")
    model = MagicMock(config=expertise.to_dict())
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=model)])
    store = ExpertiseStore(_make_storage_with_session(session))
    got = await store.get_active(uuid4())
    assert got is not None
    assert got.name == "active"


@pytest.mark.unit
async def test_get_by_namespace_is_alias_for_get_active() -> None:
    expertise = _make_expertise("ns_active")
    model = MagicMock(config=expertise.to_dict())
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=model)])
    store = ExpertiseStore(_make_storage_with_session(session))
    got = await store.get_by_namespace(uuid4())
    assert got is not None
    assert got.name == "ns_active"


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_list_returns_empty_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    assert await store.list(uuid4()) == []


@pytest.mark.unit
async def test_list_returns_models_as_dicts() -> None:
    m1 = MagicMock()
    m1.id, m1.name, m1.version, m1.description = "id1", "a", "1", "d"
    m1.is_active, m1.created_at, m1.updated_at = True, "t", "t"
    m2 = MagicMock()
    m2.id, m2.name, m2.version, m2.description = "id2", "b", "2", "e"
    m2.is_active, m2.created_at, m2.updated_at = True, "t", "t"
    session = _make_session(exec_results=[_make_result(scalars_all=[m1, m2])])
    store = ExpertiseStore(_make_storage_with_session(session))
    rows = await store.list(uuid4())
    assert len(rows) == 2
    assert rows[0]["name"] == "a"
    assert rows[1]["name"] == "b"


@pytest.mark.unit
async def test_list_include_inactive_does_not_filter_by_active() -> None:
    # Just exercise the include_inactive=True branch.
    session = _make_session(exec_results=[_make_result(scalars_all=[])])
    store = ExpertiseStore(_make_storage_with_session(session))
    rows = await store.list(uuid4(), include_inactive=True)
    assert rows == []


# ---------------------------------------------------------------------------
# set_active()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_set_active_returns_false_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    assert await store.set_active(uuid4(), "x") is False


@pytest.mark.unit
async def test_set_active_returns_true_when_row_updated() -> None:
    # Two execute calls: deactivate-all then activate-one.
    deact = _make_result(rowcount=2)
    act = _make_result(rowcount=1)
    session = _make_session(exec_results=[deact, act])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.set_active(uuid4(), "target") is True


@pytest.mark.unit
async def test_set_active_returns_false_when_no_row_updated() -> None:
    deact = _make_result(rowcount=0)
    act = _make_result(rowcount=0)
    session = _make_session(exec_results=[deact, act])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.set_active(uuid4(), "missing") is False


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_delete_returns_false_when_no_relational() -> None:
    store = ExpertiseStore(_make_storage_without_relational())
    assert await store.delete(uuid4(), "x") is False


@pytest.mark.unit
async def test_delete_returns_false_when_not_found() -> None:
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=None)])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.delete(uuid4(), "missing") is False


@pytest.mark.unit
async def test_delete_returns_true_when_deleted() -> None:
    model = MagicMock()
    session = _make_session(exec_results=[_make_result(scalar_one_or_none=model)])
    store = ExpertiseStore(_make_storage_with_session(session))
    assert await store.delete(uuid4(), "found") is True
    session.delete.assert_awaited_with(model)
    session.commit.assert_awaited()

"""Coverage tests for khora.storage.backends.postgresql.

Exercises URL normalisation, connect/disconnect lifecycle, and the various
CRUD/lookup methods using mocked SQLAlchemy AsyncSessions.  No real DB.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Document, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentStatus
from khora.db.models import DocumentModel, MemoryNamespaceModel
from khora.storage.backends.postgresql import PostgreSQLBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend_with_session(session_mock: Any) -> PostgreSQLBackend:
    """Build a backend whose ``_get_session()`` yields ``session_mock``."""
    backend = PostgreSQLBackend.__new__(PostgreSQLBackend)

    @asynccontextmanager
    async def _fake_session():  # type: ignore[no-untyped-def]
        yield session_mock

    backend._get_session = _fake_session  # type: ignore[method-assign,assignment]
    return backend


def _make_document_model(namespace_id: Any, *, checksum: str = "abc", status: Any = DocumentStatus.COMPLETED) -> Any:
    model = MagicMock(spec=DocumentModel)
    model.id = uuid4()
    model.namespace_id = namespace_id
    model.content = "content"
    model.status = status
    model.source = "test.txt"
    model.source_type = "file"
    model.content_type = "text/plain"
    model.title = "Test"
    model.author = "tester"
    model.language = "en"
    model.checksum = checksum
    model.size_bytes = 100
    model.metadata_ = {}
    model.chunk_count = 0
    model.entity_count = 0
    model.relationship_count = 0
    model.error_message = None
    model.extraction_config_hash = None
    model.extraction_params = None
    model.external_id = "ext"
    model.created_at = datetime.now(UTC)
    model.updated_at = datetime.now(UTC)
    model.processed_at = None
    model.source_timestamp = None
    model.session_id = None
    return model


def _make_namespace_model(*, tenancy_mode: str = "shared") -> Any:
    model = MagicMock(spec=MemoryNamespaceModel)
    model.id = uuid4()
    model.namespace_id = uuid4()
    model.tenancy_mode = tenancy_mode
    model.version = 1
    model.is_active = True
    model.config_overrides = {}
    model.sync_checkpoints = {}
    model.metadata_ = {}
    model.created_at = datetime.now(UTC)
    model.updated_at = datetime.now(UTC)
    return model


def _domain_namespace() -> MemoryNamespace:
    return MemoryNamespace(
        id=uuid4(),
        namespace_id=uuid4(),
        tenancy_mode=TenancyMode.SHARED,
        version=1,
        is_active=True,
    )


def _domain_document(namespace_id: Any) -> Document:
    return Document(
        id=uuid4(),
        namespace_id=namespace_id,
        content="hello",
        source="t.txt",
        source_type="file",
        checksum="ck",
        size_bytes=5,
    )


# ---------------------------------------------------------------------------
# __init__ URL normalisation + shared-engine flag
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_postgresql_url_rewritten() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    assert b._database_url.startswith("postgresql+asyncpg://")


@pytest.mark.unit
def test_postgres_url_rewritten() -> None:
    b = PostgreSQLBackend("postgres://x/y")
    assert b._database_url.startswith("postgresql+asyncpg://")


@pytest.mark.unit
def test_asyncpg_url_passthrough() -> None:
    url = "postgresql+asyncpg://x/y"
    b = PostgreSQLBackend(url)
    assert b._database_url == url


@pytest.mark.unit
def test_engine_shared_flag() -> None:
    engine = MagicMock()
    b = PostgreSQLBackend("postgresql://x/y", engine=engine)
    assert b._engine_shared is True
    assert b._engine is engine

    b2 = PostgreSQLBackend("postgresql://x/y")
    assert b2._engine_shared is False
    assert b2._engine is None


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_connect_creates_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = MagicMock()
    monkeypatch.setattr("khora.storage.backends.postgresql.create_async_engine", lambda *a, **k: fake_engine)
    b = PostgreSQLBackend("postgresql://x/y")
    await b.connect()
    assert b._engine is fake_engine
    assert b._session_factory is not None


@pytest.mark.unit
async def test_connect_idempotent() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    b._session_factory = MagicMock()  # type: ignore[assignment]
    b._engine = MagicMock()
    before = b._engine
    await b.connect()
    assert b._engine is before


@pytest.mark.unit
async def test_disconnect_disposes_when_not_shared() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    b._engine = fake_engine
    b._engine_shared = False
    b._session_factory = MagicMock()  # type: ignore[assignment]

    await b.disconnect()
    fake_engine.dispose.assert_awaited()
    assert b._engine is None


@pytest.mark.unit
async def test_disconnect_skips_dispose_when_shared() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()
    b._engine = fake_engine
    b._engine_shared = True
    b._session_factory = MagicMock()  # type: ignore[assignment]

    await b.disconnect()
    fake_engine.dispose.assert_not_called()


@pytest.mark.unit
async def test_disconnect_safe_when_already_disconnected() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    await b.disconnect()


# ---------------------------------------------------------------------------
# is_healthy
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_is_healthy_false_when_not_connected() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    assert await b.is_healthy() is False


@pytest.mark.unit
async def test_is_healthy_true_on_success() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    b._engine = MagicMock()
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())

    @asynccontextmanager
    async def _ctx():  # type: ignore[no-untyped-def]
        yield session

    b._session_factory = MagicMock(return_value=_ctx())  # type: ignore[assignment]
    assert await b.is_healthy() is True


@pytest.mark.unit
async def test_is_healthy_false_on_error() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    b._engine = MagicMock()
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=RuntimeError("db down"))

    @asynccontextmanager
    async def _ctx():  # type: ignore[no-untyped-def]
        yield session

    b._session_factory = MagicMock(return_value=_ctx())  # type: ignore[assignment]
    assert await b.is_healthy() is False


# ---------------------------------------------------------------------------
# create_tables (deprecated)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_tables_raises_when_not_connected() -> None:
    b = PostgreSQLBackend("postgresql://x/y")
    with pytest.raises(RuntimeError, match="not connected"):
        with pytest.warns(DeprecationWarning):
            await b.create_tables()


# ---------------------------------------------------------------------------
# Namespace operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_resolve_namespace_returns_row_id_when_found() -> None:
    expected = uuid4()
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=expected)
    session.execute = AsyncMock(return_value=result)

    b = _make_backend_with_session(session)
    got = await b.resolve_namespace(uuid4())
    assert got == expected


@pytest.mark.unit
async def test_resolve_namespace_raises_when_missing() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    with pytest.raises(ValueError, match="No active namespace"):
        await b.resolve_namespace(uuid4())


@pytest.mark.unit
async def test_get_namespace_returns_none_when_missing() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_namespace(uuid4()) is None


@pytest.mark.unit
async def test_get_namespace_returns_domain_model() -> None:
    model = _make_namespace_model()
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=model)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_namespace(uuid4())
    assert out is not None
    assert out.id == model.id
    assert out.tenancy_mode == TenancyMode.SHARED


@pytest.mark.unit
async def test_list_namespaces_paginated() -> None:
    m1 = _make_namespace_model()
    m2 = _make_namespace_model()

    count_result = MagicMock()
    count_result.scalar_one = MagicMock(return_value=2)
    list_result = MagicMock()
    list_scalars = MagicMock()
    list_scalars.all = MagicMock(return_value=[m1, m2])
    list_result.scalars = MagicMock(return_value=list_scalars)

    session = AsyncMock()
    queue = [count_result, list_result]

    async def _exec(*a: Any, **kw: Any) -> Any:
        return queue.pop(0)

    session.execute = _exec
    b = _make_backend_with_session(session)
    out = await b.list_namespaces(limit=10, offset=0)
    assert out.total == 2
    assert len(out.items) == 2


@pytest.mark.unit
async def test_list_namespaces_include_inactive() -> None:
    count_result = MagicMock()
    count_result.scalar_one = MagicMock(return_value=0)
    list_result = MagicMock()
    list_scalars = MagicMock()
    list_scalars.all = MagicMock(return_value=[])
    list_result.scalars = MagicMock(return_value=list_scalars)

    session = AsyncMock()
    queue = [count_result, list_result]

    async def _exec(*a: Any, **kw: Any) -> Any:
        return queue.pop(0)

    session.execute = _exec
    b = _make_backend_with_session(session)
    out = await b.list_namespaces(active_only=False)
    assert out.total == 0


@pytest.mark.unit
async def test_update_namespace_commits() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    b = _make_backend_with_session(session)
    ns = _domain_namespace()
    out = await b.update_namespace(ns)
    assert out is ns
    session.commit.assert_awaited()


@pytest.mark.unit
async def test_deactivate_namespace() -> None:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    b = _make_backend_with_session(session)
    await b.deactivate_namespace(uuid4())
    session.commit.assert_awaited()


@pytest.mark.unit
async def test_create_namespace_version_first_version() -> None:
    """No previous version -> creates v1 via create_namespace."""
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)

    captured: list[MemoryNamespace] = []

    async def _create_ns(ns: MemoryNamespace) -> MemoryNamespace:
        captured.append(ns)
        return ns

    b.create_namespace = _create_ns  # type: ignore[method-assign,assignment]

    out = await b.create_namespace_version(previous_version=None)
    assert out.version == 1
    assert out.is_active is True
    assert len(captured) == 1


@pytest.mark.unit
async def test_create_namespace_version_increments_and_deactivates_previous() -> None:
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)

    deactivated: list[Any] = []

    async def _deactivate(nid: Any) -> None:
        deactivated.append(nid)

    async def _create_ns(ns: MemoryNamespace) -> MemoryNamespace:
        return ns

    b.deactivate_namespace = _deactivate  # type: ignore[method-assign,assignment]
    b.create_namespace = _create_ns  # type: ignore[method-assign,assignment]

    prev = _domain_namespace()
    prev.version = 3
    out = await b.create_namespace_version(previous_version=prev)
    assert out.version == 4
    assert out.namespace_id == prev.namespace_id
    assert deactivated == [prev.id]


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_document_returns_none_when_missing() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_document(uuid4(), namespace_id=uuid4()) is None


@pytest.mark.unit
async def test_get_document_returns_domain_model() -> None:
    ns_id = uuid4()
    model = _make_document_model(ns_id)
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=model)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_document(uuid4(), namespace_id=ns_id)
    assert out is not None
    assert out.id == model.id


@pytest.mark.unit
async def test_get_document_requires_namespace_kwarg() -> None:
    """IDOR — IGR-221: missing ``namespace_id`` must raise TypeError."""
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    with pytest.raises(TypeError):
        await b.get_document(uuid4())  # type: ignore[call-arg]


@pytest.mark.unit
async def test_get_document_wrong_namespace_returns_none() -> None:
    """When the SQL ``namespace_id`` filter does not match, ``None`` is
    returned — verified by the mock yielding no row."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_document(uuid4(), namespace_id=uuid4()) is None


@pytest.mark.unit
async def test_list_documents_applies_filters() -> None:
    ns_id = uuid4()
    model = _make_document_model(ns_id)
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[model])
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.list_documents(ns_id, status="completed", updated_before=datetime.now(UTC))
    assert len(out) == 1


@pytest.mark.unit
async def test_count_documents_returns_int() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one = MagicMock(return_value=7)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.count_documents(uuid4()) == 7


@pytest.mark.unit
async def test_get_last_activity_at_returns_value() -> None:
    ts = datetime.now(UTC)
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=ts)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_last_activity_at(uuid4()) == ts


@pytest.mark.unit
async def test_get_document_stats_returns_tuple() -> None:
    ts = datetime.now(UTC)
    session = AsyncMock()
    result = MagicMock()
    result.one = MagicMock(return_value=(5, ts))
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    count, last = await b.get_document_stats(uuid4())
    assert count == 5
    assert last == ts


@pytest.mark.unit
async def test_get_document_by_checksum_returns_none() -> None:
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.first = MagicMock(return_value=None)
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_document_by_checksum(uuid4(), "ck") is None


@pytest.mark.unit
async def test_get_document_by_checksum_returns_model() -> None:
    ns_id = uuid4()
    model = _make_document_model(ns_id)
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.first = MagicMock(return_value=model)
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_document_by_checksum(ns_id, "ck")
    assert out is not None
    assert out.id == model.id


@pytest.mark.unit
async def test_get_document_by_external_id_short_circuits_on_none() -> None:
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    # No session needed because we short-circuit.
    assert await b.get_document_by_external_id(None, namespace_id=uuid4()) is None


@pytest.mark.unit
async def test_get_documents_batch_empty_returns_empty_dict() -> None:
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    assert await b.get_documents_batch([], namespace_id=uuid4()) == {}


@pytest.mark.unit
async def test_get_documents_batch_returns_keyed_dict() -> None:
    ns_id = uuid4()
    m1 = _make_document_model(ns_id)
    m2 = _make_document_model(ns_id)
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[m1, m2])
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_documents_batch([m1.id, m2.id], namespace_id=ns_id)
    assert set(out.keys()) == {m1.id, m2.id}


@pytest.mark.unit
async def test_get_documents_batch_requires_namespace_kwarg() -> None:
    """IDOR — IGR-221: missing ``namespace_id`` must raise TypeError."""
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    with pytest.raises(TypeError):
        await b.get_documents_batch([uuid4()])  # type: ignore[call-arg]


@pytest.mark.unit
async def test_get_documents_batch_wrong_namespace_drops_rows() -> None:
    """SQL filter drops cross-namespace rows; the mock returns no rows for
    a non-matching namespace_id, mirroring real DB behaviour."""
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[])
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_documents_batch([uuid4(), uuid4()], namespace_id=uuid4())
    assert out == {}


@pytest.mark.unit
async def test_get_documents_by_external_ids_filters_empty_strings() -> None:
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    # Filter is ``if e:`` so None and "" are dropped; whitespace strings would
    # still hit the DB.  This test only exercises the all-empty short-circuit.
    assert await b.get_documents_by_external_ids([None, ""], namespace_id=uuid4()) == {}  # type: ignore[list-item]


@pytest.mark.unit
async def test_get_documents_by_external_ids_returns_keyed_dict() -> None:
    ns_id = uuid4()
    m1 = _make_document_model(ns_id)
    m1.external_id = "ext1"
    m2 = _make_document_model(ns_id)
    m2.external_id = "ext2"
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[m1, m2])
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_documents_by_external_ids(["ext1", "ext2"], namespace_id=ns_id)
    assert set(out.keys()) == {"ext1", "ext2"}


@pytest.mark.unit
async def test_get_documents_by_checksums_empty_returns_empty() -> None:
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    assert await b.get_documents_by_checksums(uuid4(), []) == {}


@pytest.mark.unit
async def test_get_documents_by_checksums_keyed() -> None:
    ns_id = uuid4()
    m1 = _make_document_model(ns_id, checksum="ck1")
    m2 = _make_document_model(ns_id, checksum="ck2")
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all = MagicMock(return_value=[m1, m2])
    result.scalars = MagicMock(return_value=scalars)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_documents_by_checksums(ns_id, ["ck1", "ck2"])
    assert set(out.keys()) == {"ck1", "ck2"}


@pytest.mark.unit
async def test_get_document_sources_batch_empty_returns_empty() -> None:
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    assert await b.get_document_sources_batch([], namespace_id=uuid4()) == {}


@pytest.mark.unit
async def test_get_document_sources_batch_returns_keyed_dict() -> None:
    row1 = MagicMock()
    row1.id = uuid4()
    row1.title = "t1"
    row1.source = "s1"
    row1.source_type = "file"
    row1.created_at = datetime.now(UTC)
    row1.source_timestamp = None

    row2 = MagicMock()
    row2.id = uuid4()
    row2.title = "t2"
    row2.source = "s2"
    row2.source_type = "url"
    row2.created_at = datetime.now(UTC)
    row2.source_timestamp = None

    session = AsyncMock()
    result = MagicMock()
    result.all = MagicMock(return_value=[row1, row2])
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_document_sources_batch([row1.id, row2.id], namespace_id=uuid4())
    assert set(out.keys()) == {row1.id, row2.id}
    assert out[row1.id].title == "t1"


@pytest.mark.unit
async def test_get_document_sources_batch_requires_namespace_kwarg() -> None:
    """IDOR — IGR-221: missing ``namespace_id`` must raise TypeError."""
    b = PostgreSQLBackend.__new__(PostgreSQLBackend)
    with pytest.raises(TypeError):
        await b.get_document_sources_batch([uuid4()])  # type: ignore[call-arg]


@pytest.mark.unit
async def test_get_document_sources_batch_wrong_namespace_drops_rows() -> None:
    """SQL filter drops cross-namespace rows."""
    session = AsyncMock()
    result = MagicMock()
    result.all = MagicMock(return_value=[])
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    out = await b.get_document_sources_batch([uuid4()], namespace_id=uuid4())
    assert out == {}


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_delete_document_returns_false_when_missing() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    b = _make_backend_with_session(session)
    assert await b.delete_document(uuid4(), namespace_id=uuid4()) is False


@pytest.mark.unit
async def test_delete_document_returns_true_when_deleted() -> None:
    model = _make_document_model(uuid4())
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=model)
    session.execute = AsyncMock(return_value=result)
    session.delete = AsyncMock()
    session.commit = AsyncMock()
    b = _make_backend_with_session(session)
    assert await b.delete_document(uuid4(), namespace_id=uuid4()) is True
    session.delete.assert_awaited_with(model)


# ---------------------------------------------------------------------------
# Sync checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_sync_checkpoint_returns_value_when_present() -> None:
    cp = MagicMock()
    cp.checkpoint = "ckpt-1"
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=cp)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_sync_checkpoint(uuid4(), "src") == "ckpt-1"


@pytest.mark.unit
async def test_get_sync_checkpoint_returns_none_when_missing() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    b = _make_backend_with_session(session)
    assert await b.get_sync_checkpoint(uuid4(), "src") is None


@pytest.mark.unit
async def test_set_sync_checkpoint_updates_existing() -> None:
    cp = MagicMock()
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=cp)
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    b = _make_backend_with_session(session)
    await b.set_sync_checkpoint(uuid4(), "src", "new-ckpt")
    assert cp.checkpoint == "new-ckpt"
    session.commit.assert_awaited()


@pytest.mark.unit
async def test_set_sync_checkpoint_inserts_when_missing() -> None:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    b = _make_backend_with_session(session)
    await b.set_sync_checkpoint(uuid4(), "src", "first-ckpt")
    session.add.assert_called()
    session.commit.assert_awaited()

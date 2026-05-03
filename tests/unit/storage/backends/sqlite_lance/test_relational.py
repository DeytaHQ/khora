"""Tests for :class:`SQLiteLanceRelationalAdapter` (DYT-2728)."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Document, DocumentMetadata, MemoryNamespace, TenancyMode
from khora.core.models.document import DocumentStatus

pytestmark = pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed")

if _HAS_EMBEDDED:
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def adapter(migrated_sqlite_db, tmp_path):
    """Migrated SQLite DB + connected relational adapter.

    Uses the real Alembic migration chain (dialect-gated since DYT-2727)
    to validate that the adapter works against the production schema.
    """
    db_path = str(migrated_sqlite_db)
    lance_path = str(tmp_path / "khora.lance")

    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(db_path=db_path, lance_path=lance_path),
    )
    adapter = SQLiteLanceRelationalAdapter(handle)
    await adapter.connect()

    try:
        yield adapter
    finally:
        await adapter.disconnect()
        await handle.disconnect()


@pytest.fixture
async def namespace(adapter):
    """A freshly-created active namespace."""
    nid = uuid4()
    ns = MemoryNamespace(
        id=nid,
        namespace_id=nid,
        tenancy_mode=TenancyMode.SHARED,
    )
    return await adapter.create_namespace(ns)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_health_check_after_connect(adapter):
    assert await adapter.is_healthy() is True


async def test_session_factory_exposed(adapter):
    """``_session_factory`` must be publicly accessible for
    ``StorageCoordinator.transaction()`` to work."""
    assert adapter._session_factory is not None
    session = adapter._session_factory()
    await session.close()


# ---------------------------------------------------------------------------
# Namespace CRUD
# ---------------------------------------------------------------------------


async def test_create_and_get_namespace(adapter):
    nid = uuid4()
    ns = MemoryNamespace(id=nid, namespace_id=nid)
    created = await adapter.create_namespace(ns)
    assert created.id == nid

    fetched = await adapter.get_namespace(nid)
    assert fetched is not None
    assert fetched.namespace_id == nid
    assert fetched.is_active is True


async def test_resolve_namespace_by_stable_id(adapter, namespace):
    row_id = await adapter.resolve_namespace(namespace.namespace_id)
    assert row_id == namespace.id


async def test_resolve_namespace_by_row_id(adapter, namespace):
    row_id = await adapter.resolve_namespace(namespace.id)
    assert row_id == namespace.id


async def test_resolve_missing_namespace_raises(adapter):
    with pytest.raises(ValueError):
        await adapter.resolve_namespace(uuid4())


async def test_list_namespaces(adapter):
    for _ in range(3):
        nid = uuid4()
        await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid))

    result = await adapter.list_namespaces(limit=10)
    assert result.total == 3
    assert len(result.items) == 3
    assert all(ns.is_active for ns in result.items)


async def test_update_namespace(adapter, namespace):
    namespace.metadata = {"key": "value"}
    await adapter.update_namespace(namespace)
    fetched = await adapter.get_namespace(namespace.id)
    assert fetched is not None
    assert fetched.metadata == {"key": "value"}


async def test_deactivate_namespace(adapter, namespace):
    await adapter.deactivate_namespace(namespace.id)
    fetched = await adapter.get_namespace(namespace.id)
    assert fetched is not None
    assert fetched.is_active is False


async def test_namespace_versioning_roundtrip(adapter, namespace):
    """New version deactivates the prior version and preserves namespace_id."""
    v2 = await adapter.create_namespace_version(previous_version=namespace)
    assert v2.version == 2
    assert v2.is_active is True
    assert v2.namespace_id == namespace.namespace_id
    assert v2.id != namespace.id

    prior = await adapter.get_namespace(namespace.id)
    assert prior is not None
    assert prior.is_active is False

    resolved = await adapter.resolve_namespace(namespace.namespace_id)
    assert resolved == v2.id


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------


def _make_document(namespace_id, *, checksum: str = "abc", title: str = "Doc") -> Document:
    return Document(
        namespace_id=namespace_id,
        content="hello world",
        metadata=DocumentMetadata(
            source="file:///tmp/a.txt",
            source_type="file",
            title=title,
            checksum=checksum,
            size_bytes=11,
        ),
    )


async def test_create_and_get_document(adapter, namespace):
    doc = _make_document(namespace.id)
    created = await adapter.create_document(doc)
    assert created.id == doc.id

    fetched = await adapter.get_document(doc.id)
    assert fetched is not None
    assert fetched.content == "hello world"
    assert fetched.metadata.title == "Doc"


async def test_list_documents_filters_by_status(adapter, namespace):
    pending = _make_document(namespace.id, checksum="p", title="P")
    completed = _make_document(namespace.id, checksum="c", title="C")
    completed.status = DocumentStatus.COMPLETED
    await adapter.create_document(pending)
    await adapter.create_document(completed)

    all_docs = await adapter.list_documents(namespace.id)
    assert len(all_docs) == 2

    only_completed = await adapter.list_documents(namespace.id, status="completed")
    assert len(only_completed) == 1
    assert only_completed[0].metadata.title == "C"


async def test_update_document(adapter, namespace):
    doc = _make_document(namespace.id)
    await adapter.create_document(doc)
    doc.metadata.title = "Updated"
    doc.status = DocumentStatus.COMPLETED
    await adapter.update_document(doc)

    fetched = await adapter.get_document(doc.id)
    assert fetched is not None
    assert fetched.metadata.title == "Updated"
    assert fetched.status == DocumentStatus.COMPLETED


async def test_delete_document(adapter, namespace):
    doc = _make_document(namespace.id)
    await adapter.create_document(doc)
    assert await adapter.delete_document(doc.id) is True
    assert await adapter.get_document(doc.id) is None
    assert await adapter.delete_document(doc.id) is False


async def test_dedup_by_checksum(adapter, namespace):
    doc = _make_document(namespace.id, checksum="xyz")
    await adapter.create_document(doc)

    match = await adapter.get_document_by_checksum(namespace.id, "xyz")
    assert match is not None
    assert match.id == doc.id

    none = await adapter.get_document_by_checksum(namespace.id, "missing")
    assert none is None


async def test_dedup_excludes_failed_documents(adapter, namespace):
    """FAILED documents must not block re-ingestion of the same checksum."""
    failed = _make_document(namespace.id, checksum="same")
    failed.status = DocumentStatus.FAILED
    await adapter.create_document(failed)

    assert await adapter.get_document_by_checksum(namespace.id, "same") is None


# ---------------------------------------------------------------------------
# Aggregates / batch
# ---------------------------------------------------------------------------


async def test_count_documents_and_last_activity(adapter, namespace):
    assert await adapter.count_documents(namespace.id) == 0
    assert await adapter.get_last_activity_at(namespace.id) is None

    doc = _make_document(namespace.id)
    await adapter.create_document(doc)

    count, last = await adapter.get_document_stats(namespace.id)
    assert count == 1
    assert last is not None
    assert isinstance(last, datetime)


async def test_get_documents_batch(adapter, namespace):
    d1 = _make_document(namespace.id, checksum="1")
    d2 = _make_document(namespace.id, checksum="2")
    await adapter.create_document(d1)
    await adapter.create_document(d2)

    batch = await adapter.get_documents_batch([d1.id, d2.id, uuid4()])
    assert set(batch.keys()) == {d1.id, d2.id}
    assert batch[d1.id].metadata.checksum == "1"


async def test_get_documents_by_checksums(adapter, namespace):
    d1 = _make_document(namespace.id, checksum="aa")
    d2 = _make_document(namespace.id, checksum="bb")
    await adapter.create_document(d1)
    await adapter.create_document(d2)

    batch = await adapter.get_documents_by_checksums(namespace.id, ["aa", "bb", "missing"])
    assert set(batch.keys()) == {"aa", "bb"}


async def test_get_document_sources_batch(adapter, namespace):
    doc = _make_document(namespace.id, title="Attribution")
    await adapter.create_document(doc)

    sources = await adapter.get_document_sources_batch([doc.id])
    assert doc.id in sources
    src = sources[doc.id]
    assert src.title == "Attribution"
    assert src.source == "file:///tmp/a.txt"


async def test_empty_batch_is_noop(adapter):
    assert await adapter.get_documents_batch([]) == {}
    assert await adapter.get_documents_by_checksums(uuid4(), []) == {}
    assert await adapter.get_document_sources_batch([]) == {}


# ---------------------------------------------------------------------------
# Sync checkpoints
# ---------------------------------------------------------------------------


async def test_sync_checkpoint_upsert(adapter, namespace):
    assert await adapter.get_sync_checkpoint(namespace.id, "github") is None

    await adapter.set_sync_checkpoint(namespace.id, "github", "cursor-1")
    assert await adapter.get_sync_checkpoint(namespace.id, "github") == "cursor-1"

    # Update — must not duplicate the row.
    await adapter.set_sync_checkpoint(namespace.id, "github", "cursor-2")
    assert await adapter.get_sync_checkpoint(namespace.id, "github") == "cursor-2"


# ---------------------------------------------------------------------------
# Transaction support via shared session_factory
# ---------------------------------------------------------------------------


async def test_transaction_rollback_via_session_factory(adapter, namespace):
    """Simulate ``StorageCoordinator.transaction()`` — create in a session
    that is rolled back, and assert nothing persisted.
    """
    doc = _make_document(namespace.id, checksum="tx")

    assert adapter._session_factory is not None
    session = adapter._session_factory()
    try:
        await adapter.create_document(doc, session=session)
        # Sanity check: the insert is visible within the open transaction.
        fetched = await adapter.get_document(doc.id)
        # The outer ``get_document`` opens its own session and under SQLite
        # WAL may or may not see the uncommitted row depending on isolation.
        # Regardless, after rollback it must be gone.
        _ = fetched
        await session.rollback()
    finally:
        await session.close()

    assert await adapter.get_document(doc.id) is None


async def test_transaction_commit_via_session_factory(adapter, namespace):
    doc = _make_document(namespace.id, checksum="tx-commit")

    assert adapter._session_factory is not None
    session = adapter._session_factory()
    try:
        await adapter.create_document(doc, session=session)
        await session.commit()
    finally:
        await session.close()

    fetched = await adapter.get_document(doc.id)
    assert fetched is not None
    assert fetched.metadata.checksum == "tx-commit"

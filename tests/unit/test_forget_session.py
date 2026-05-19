"""Tests for ``Khora.forget_session`` and the session_id column plumbing (#620).

Validates that:
* The ``session_id`` column is present on the migrated SQLite schema for
  ``documents`` and ``chunks``.
* ``Document.session_id`` round-trips through the relational adapter.
* The (namespace_id, session_id) DELETE used by ``forget_session`` removes
  only matching rows without touching other sessions.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

try:
    import aiosqlite
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False
    aiosqlite = None  # type: ignore[assignment]

from khora.core.models import Document, MemoryNamespace

pytestmark = pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed")

if _HAS_EMBEDDED:
    from khora.db.session import run_migrations
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.relational import SQLiteLanceRelationalAdapter


@pytest.fixture(scope="module")
def _migrated_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Migrate a SQLite DB once per module — same trick as the sqlite_lance suite."""
    template_dir = tmp_path_factory.mktemp("forget_session_template")
    template_path = template_dir / "template.db"

    async def _migrate() -> None:
        result = await run_migrations(f"sqlite+aiosqlite:///{template_path}")
        if not result.success:
            raise RuntimeError(f"template migration failed: {result.error}")

    asyncio.run(_migrate())
    return template_path


@pytest.fixture
def migrated_db(_migrated_template: Path, tmp_path: Path) -> Path:
    target = tmp_path / "khora.db"
    shutil.copy(_migrated_template, target)
    return target


@pytest.fixture
async def adapter(migrated_db, tmp_path):
    db_path = str(migrated_db)
    lance_path = str(tmp_path / "khora.lance")

    handle = EmbeddedStorageHandle(
        EmbeddedStorageHandleConfig(db_path=db_path, lance_path=lance_path),
    )
    ad = SQLiteLanceRelationalAdapter(handle)
    await ad.connect()

    try:
        yield ad
    finally:
        await ad.disconnect()
        await handle.disconnect()


@pytest.fixture
async def namespace(adapter):
    nid = uuid4()
    return await adapter.create_namespace(MemoryNamespace(id=nid, namespace_id=nid))


def _make_document(namespace_id, *, checksum: str, title: str, session_id=None) -> Document:
    return Document(
        namespace_id=namespace_id,
        content=f"content for {title}",
        source="file:///tmp/a.txt",
        source_type="file",
        title=title,
        checksum=checksum,
        size_bytes=32,
        session_id=session_id,
    )


async def test_session_id_column_exists_on_documents(migrated_db):
    """Migration 030 adds the column on SQLite."""
    async with aiosqlite.connect(str(migrated_db)) as db:
        cur = await db.execute("PRAGMA table_info(documents)")
        rows = await cur.fetchall()
        col_names = {row[1] for row in rows}
    assert "session_id" in col_names


async def test_session_id_column_exists_on_chunks(migrated_db):
    """Migration 030 adds the column on chunks too."""
    async with aiosqlite.connect(str(migrated_db)) as db:
        cur = await db.execute("PRAGMA table_info(chunks)")
        rows = await cur.fetchall()
        col_names = {row[1] for row in rows}
    assert "session_id" in col_names


async def test_session_id_column_exists_on_downstream_tables(migrated_db):
    """Migration 030 also adds the column to the downstream tables."""
    expected = ("memory_events", "chronicle_events", "memory_facts")
    async with aiosqlite.connect(str(migrated_db)) as db:
        for table in expected:
            cur = await db.execute(f"PRAGMA table_info({table})")  # noqa: S608 — fixed list
            rows = await cur.fetchall()
            col_names = {row[1] for row in rows}
            assert "session_id" in col_names, f"missing session_id on {table}"


async def test_session_id_roundtrips_through_document(adapter, namespace):
    """Document.session_id is persisted and read back."""
    sid = uuid4()
    doc = _make_document(namespace.id, checksum="s1", title="With Session", session_id=sid)
    await adapter.create_document(doc)

    fetched = await adapter.get_document(doc.id, namespace_id=namespace.id)
    assert fetched is not None
    assert fetched.session_id == sid


async def test_session_id_none_when_not_set(adapter, namespace):
    """Default Document.session_id is None and persists as NULL."""
    doc = _make_document(namespace.id, checksum="s2", title="No Session")
    await adapter.create_document(doc)

    fetched = await adapter.get_document(doc.id, namespace_id=namespace.id)
    assert fetched is not None
    assert fetched.session_id is None


async def test_session_id_visible_via_list_documents(adapter, namespace):
    """list_documents exposes session_id per row so callers can filter client-side."""
    sid_a = uuid4()
    sid_b = uuid4()
    doc_a = _make_document(namespace.id, checksum="a1", title="A1", session_id=sid_a)
    doc_b1 = _make_document(namespace.id, checksum="b1", title="B1", session_id=sid_b)
    doc_b2 = _make_document(namespace.id, checksum="b2", title="B2", session_id=sid_b)
    await adapter.create_document(doc_a)
    await adapter.create_document(doc_b1)
    await adapter.create_document(doc_b2)

    docs = await adapter.list_documents(namespace.id, limit=100)
    by_sid: dict[object, int] = {}
    for d in docs:
        by_sid[d.session_id] = by_sid.get(d.session_id, 0) + 1
    assert by_sid[sid_a] == 1
    assert by_sid[sid_b] == 2


async def test_session_delete_isolates_target_session(adapter, namespace):
    """The SQL contract behind ``forget_session``: filter on (ns, session)."""
    sid_keep = uuid4()
    sid_drop = uuid4()
    keep1 = _make_document(namespace.id, checksum="k1", title="Keep1", session_id=sid_keep)
    keep2 = _make_document(namespace.id, checksum="k2", title="Keep2", session_id=sid_keep)
    drop1 = _make_document(namespace.id, checksum="d1", title="Drop1", session_id=sid_drop)
    drop2 = _make_document(namespace.id, checksum="d2", title="Drop2", session_id=sid_drop)
    for d in (keep1, keep2, drop1, drop2):
        await adapter.create_document(d)

    from sqlalchemy import delete

    from khora.db.models import DocumentModel

    async with adapter._get_session() as session:
        result = await session.execute(
            delete(DocumentModel).where(
                DocumentModel.namespace_id == namespace.id,
                DocumentModel.session_id == sid_drop,
            )
        )
        await session.commit()
        deleted_count = result.rowcount

    assert deleted_count == 2
    survivors = await adapter.list_documents(namespace.id, limit=100)
    surviving_ids = {d.id for d in survivors}
    assert keep1.id in surviving_ids
    assert keep2.id in surviving_ids
    assert drop1.id not in surviving_ids
    assert drop2.id not in surviving_ids


async def test_chunk_dataclass_propagates_session_id():
    """Chunk dataclass surfaces session_id as a public field (#620)."""
    from khora.core.models import Chunk

    sid = uuid4()
    chunk = Chunk(
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="x",
        session_id=sid,
    )
    assert chunk.session_id == sid


async def test_chunk_task_propagates_session_id_from_document():
    """chunk_document copies Document.session_id onto every emitted Chunk."""
    from khora.core.models import Document
    from khora.pipelines.tasks.chunk import chunk_document

    sid = uuid4()
    doc = Document(
        namespace_id=uuid4(),
        content="A short document of just a few sentences. Another sentence here.",
        session_id=sid,
    )
    chunks = await chunk_document(doc, strategy="fixed", chunk_size=64, chunk_overlap=0)
    assert chunks  # sanity: chunker emitted something
    assert all(c.session_id == sid for c in chunks)


async def test_ingest_coerce_session_id_helper():
    """The ingest helper accepts UUIDs and string UUIDs; returns None on garbage."""
    from khora.pipelines.flows.ingest import _coerce_session_id

    sid = uuid4()
    assert _coerce_session_id(sid) == sid
    assert _coerce_session_id(str(sid)) == sid
    assert _coerce_session_id(None) is None
    assert _coerce_session_id("") is None
    assert _coerce_session_id("not-a-uuid") is None
    assert _coerce_session_id(123) is None

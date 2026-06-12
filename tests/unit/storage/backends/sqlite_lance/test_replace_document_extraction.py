"""Regression tests for ``replace_document_extraction`` on the sqlite_lance stack.

Covers #1134 (the coordinator passed ``session=`` to the sqlite_lance vector
adapter's ``create_chunks_batch``, which has no such kwarg - TypeError) and
#1135 (the adapter's ``delete_chunks_by_document(session=...)`` left the
DELETE pending on the shared aiosqlite handle, so a later unrelated commit
silently destroyed the document's chunks outside any controlled transaction,
and the LanceDB compensation never ran).

The SQLAlchemy session handed out by ``coordinator.transaction()`` runs on
the relational adapter's separate engine - it can never cover the raw
aiosqlite handle the vector adapter writes through. The coordinator must
therefore only pass ``session=`` to vector backends that participate in the
SQLAlchemy session, and the adapter must always commit its own SQLite work.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Chunk, Document, MemoryNamespace

pytestmark = pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed")

if _HAS_EMBEDDED:
    from khora.config.schema import SQLiteLanceConfig
    from khora.storage.coordinator import StorageCoordinator
    from khora.storage.factory import StorageConfig, StorageFactory


_DIM = 8


@pytest.fixture
async def coordinator(migrated_sqlite_db: Path, tmp_path: Path):
    """Factory-built sqlite_lance coordinator over a migrated SQLite DB."""
    storage_config = StorageConfig(
        backend="sqlite_lance",
        sqlite_lance_config=SQLiteLanceConfig(
            db_path=str(migrated_sqlite_db),
            lance_path=str(tmp_path / "khora.lance"),
            embedding_dimension=_DIM,
        ),
    )
    coord = StorageFactory(config=storage_config).create_coordinator()
    await coord.connect()
    try:
        yield coord
    finally:
        await coord.disconnect()


@pytest.fixture
async def namespace(coordinator: StorageCoordinator) -> MemoryNamespace:
    nid = uuid4()
    return await coordinator.create_namespace(MemoryNamespace(id=nid, namespace_id=nid))


def _unit(idx: int) -> list[float]:
    vec = [0.0] * _DIM
    vec[idx % _DIM] = 1.0
    return vec


def _make_chunk(namespace_id: UUID, document_id: UUID, idx: int) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content=f"chunk {idx}",
        chunk_index=idx,
        embedding=_unit(idx),
        embedding_model="test-model",
    )


async def _seed_document(coordinator: StorageCoordinator, namespace: MemoryNamespace, n_chunks: int):
    doc = Document(
        namespace_id=namespace.id,
        content="original content",
        source="file:///tmp/a.txt",
        source_type="file",
        title="Doc",
        checksum="abc",
        size_bytes=16,
    )
    doc = await coordinator.create_document(doc)
    chunks = [_make_chunk(namespace.id, doc.id, i) for i in range(n_chunks)]
    await coordinator.create_chunks_batch(chunks)
    return doc, chunks


async def _chunk_ids_fresh_connection(db_path: Path, document_id: UUID) -> set[str]:
    """Read chunk ids for a document over a NEW connection.

    A fresh connection only sees committed state - reads through the shared
    handle would sit inside its pending implicit transaction and hide the
    #1135 time bomb.
    """
    import aiosqlite

    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?",
            (document_id.hex,),
        )
        rows = await cur.fetchall()
    return {row[0] for row in rows}


async def test_replace_swaps_chunks_without_typeerror(
    coordinator: StorageCoordinator,
    namespace: MemoryNamespace,
    migrated_sqlite_db: Path,
) -> None:
    """#1134: replace must not crash, and the chunk swap must be durable."""
    doc, old_chunks = await _seed_document(coordinator, namespace, 2)

    new_chunk = _make_chunk(namespace.id, doc.id, 7)
    doc.content = "replaced content"

    result = await coordinator.replace_document_extraction(
        namespace_id=namespace.id,
        old_document_id=doc.id,
        new_document=doc,
        new_chunks=[new_chunk],
        new_entities=[],
        new_relationships=[],
    )

    assert result.chunks_deleted == 2
    assert result.chunks_created == 1

    # Committed state: old chunks gone, new chunk present.
    ids = await _chunk_ids_fresh_connection(migrated_sqlite_db, doc.id)
    assert ids == {new_chunk.id.hex}

    # #1135: a later unrelated commit on the shared aiosqlite handle must
    # not change the chunk set (no pending DELETE left behind).
    await coordinator._vector._sqlite.commit()  # type: ignore[attr-defined]
    assert await _chunk_ids_fresh_connection(migrated_sqlite_db, doc.id) == ids

    # LanceDB compensation ran: vectors match the surviving SQLite rows.
    tbl = await coordinator._vector._chunks_table()  # type: ignore[attr-defined]
    assert await tbl.count_rows() == 1


async def test_failed_replace_leaves_no_pending_delete(
    coordinator: StorageCoordinator,
    namespace: MemoryNamespace,
    migrated_sqlite_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1135: a failed replace must not leave a pending DELETE that a later
    unrelated commit silently applies.

    The embedded path commits the vector-side delete + insert immediately
    (cross-store atomicity is partial on embedded - documented), so the
    state observed right after the failure must be IDENTICAL to the state
    after any later commit on the shared handle, and LanceDB must hold no
    orphaned vectors for the deleted chunks.
    """
    doc, old_chunks = await _seed_document(coordinator, namespace, 2)
    new_chunk = _make_chunk(namespace.id, doc.id, 7)

    async def boom(*args, **kwargs):
        raise RuntimeError("forced update_document failure")

    monkeypatch.setattr(coordinator._relational, "update_document", boom)

    with pytest.raises(Exception):  # noqa: B017 - on the buggy path this was TypeError, on the fixed path RuntimeError
        await coordinator.replace_document_extraction(
            namespace_id=namespace.id,
            old_document_id=doc.id,
            new_document=doc,
            new_chunks=[new_chunk],
            new_entities=[],
            new_relationships=[],
        )

    ids_after_failure = await _chunk_ids_fresh_connection(migrated_sqlite_db, doc.id)

    # The time bomb: an unrelated commit on the shared handle (a graph
    # upsert, event append, update_last_accessed... all end here) must not
    # change the committed chunk set.
    await coordinator._vector._sqlite.commit()  # type: ignore[attr-defined]
    ids_after_unrelated_commit = await _chunk_ids_fresh_connection(migrated_sqlite_db, doc.id)
    assert ids_after_unrelated_commit == ids_after_failure

    # LanceDB compensation ran for whatever was deleted: vector rows match
    # the surviving SQLite rows exactly (no orphaned vectors).
    tbl = await coordinator._vector._chunks_table()  # type: ignore[attr-defined]
    assert await tbl.count_rows() == len(ids_after_failure)

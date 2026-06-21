"""``delete_chunks_by_document`` actually deletes on backend=surrealdb (#1221).

Pre-#1221 the chunk-write path set only the ``namespace`` record-link and
never populated the scalar ``namespace_id`` column, while the delete/count
queries filtered on that never-populated scalar. The delete matched 0 rows,
so ``forget()`` and re-ingest were silent no-ops: old chunks survived and the
``chunk`` table grew unbounded.

Drives the real ``SurrealDBVectorAdapter`` against an in-process
(``mode="memory"``) SurrealDB engine - no docker required. Skipped when the
``surrealdb`` extra is absent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Chunk  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
async def vector():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="delchunks1221")
    await conn.connect()
    try:
        yield SurrealDBVectorAdapter(conn)
    finally:
        await conn.disconnect()


def _chunk(ns_id, doc_id, index: int = 0) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=f"quantum entanglement correlates measurement outcomes {index}",
        chunk_index=index,
        start_char=0,
        end_char=52,
        token_count=7,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="probe",
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_delete_chunks_by_document_deletes_single_chunk(vector) -> None:
    ns_id = uuid4()
    doc_id = uuid4()

    await vector.create_chunk(_chunk(ns_id, doc_id))

    before = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
    assert len(before) == 1

    deleted = await vector.delete_chunks_by_document(doc_id, namespace_id=ns_id)
    assert deleted == 1

    after = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
    assert after == []


@pytest.mark.asyncio
async def test_delete_chunks_by_document_deletes_batch_written_chunks(vector) -> None:
    ns_id = uuid4()
    doc_id = uuid4()

    await vector.create_chunks_batch([_chunk(ns_id, doc_id, i) for i in range(3)])

    before = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
    assert len(before) == 3

    deleted = await vector.delete_chunks_by_document(doc_id, namespace_id=ns_id)
    assert deleted == 3

    after = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
    assert after == []


@pytest.mark.asyncio
async def test_delete_chunks_scoped_to_namespace(vector) -> None:
    """A delete in one namespace must not touch another namespace's chunks."""
    ns_a = uuid4()
    ns_b = uuid4()
    doc_id = uuid4()  # same document id, different namespaces

    await vector.create_chunk(_chunk(ns_a, doc_id))
    await vector.create_chunk(_chunk(ns_b, doc_id))

    deleted = await vector.delete_chunks_by_document(doc_id, namespace_id=ns_a)
    assert deleted == 1

    assert await vector.get_chunks_by_document(doc_id, namespace_id=ns_a) == []
    assert len(await vector.get_chunks_by_document(doc_id, namespace_id=ns_b)) == 1


@pytest.mark.asyncio
async def test_reingest_does_not_accumulate_stale_chunks(vector) -> None:
    """Re-ingest (delete-then-write) leaves exactly the new chunk set."""
    ns_id = uuid4()
    doc_id = uuid4()

    await vector.create_chunk(_chunk(ns_id, doc_id, 0))

    # Re-ingest of the same document: the replace path deletes old chunks first.
    deleted = await vector.delete_chunks_by_document(doc_id, namespace_id=ns_id)
    assert deleted == 1
    await vector.create_chunks_batch([_chunk(ns_id, doc_id, i) for i in range(2)])

    after = await vector.get_chunks_by_document(doc_id, namespace_id=ns_id)
    assert len(after) == 2, "stale/duplicate chunks accumulated across re-ingest"

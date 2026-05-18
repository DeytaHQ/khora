"""Integration test for pgvector chunk-getter namespace isolation.

Exercises the real PostgreSQL backend to prove the namespace filter is
applied at SQL level (not just in the Python wrapper). Requires a
running PostgreSQL instance (``make dev``).

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_DATABASE_URL  (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Chunk, ChunkMetadata, Document, DocumentMetadata, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.db.session import run_migrations
from khora.storage.backends.pgvector import PgVectorBackend

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5432/khora",
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

pytestmark = [pytest.mark.integration]


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


skip_no_pg = pytest.mark.skipif(
    not _pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)


@pytest.fixture(scope="module")
async def _run_migrations_once():
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"


@pytest.fixture
async def backend(_run_migrations_once):
    be = PgVectorBackend(DATABASE_URL)
    await be.connect()
    yield be
    await be.disconnect()


def _make_chunk(namespace_id, document_id) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id,
        content="integration content",
        metadata=ChunkMetadata(document_id=document_id, chunk_index=0),
        embedding=None,
        embedding_model="test-model",
        created_at=datetime.now(UTC),
    )


def _make_doc(namespace_id) -> Document:
    return Document(
        id=uuid4(),
        namespace_id=namespace_id,
        content="doc content",
        status=DocumentStatus.PENDING,
        metadata=DocumentMetadata(
            source="test",
            source_type="file",
            content_type="text/plain",
            title="t",
            author="a",
            checksum="x",
        ),
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@skip_no_pg
class TestChunkNamespaceIsolationPg:
    async def test_cross_namespace_get_chunk_returns_none(self, backend: PgVectorBackend) -> None:
        ns_a = await backend.create_namespace(MemoryNamespace())
        ns_b = await backend.create_namespace(MemoryNamespace())

        doc = _make_doc(ns_a.id)
        await backend.create_document(doc)

        chunk = _make_chunk(ns_a.id, doc.id)
        await backend.create_chunk(chunk)

        # Same namespace returns the chunk.
        assert (await backend.get_chunk(chunk.id, namespace_id=ns_a.id)) is not None
        # Cross-namespace lookup must return None.
        assert (await backend.get_chunk(chunk.id, namespace_id=ns_b.id)) is None

    async def test_cross_namespace_get_chunks_batch_filters(self, backend: PgVectorBackend) -> None:
        ns_a = await backend.create_namespace(MemoryNamespace())
        ns_b = await backend.create_namespace(MemoryNamespace())

        doc_a = _make_doc(ns_a.id)
        doc_b = _make_doc(ns_b.id)
        await backend.create_document(doc_a)
        await backend.create_document(doc_b)

        c_a = _make_chunk(ns_a.id, doc_a.id)
        c_b = _make_chunk(ns_b.id, doc_b.id)
        await backend.create_chunks_batch([c_a, c_b])

        result = await backend.get_chunks_batch([c_a.id, c_b.id], namespace_id=ns_a.id)
        assert set(result.keys()) == {c_a.id}

    async def test_cross_namespace_get_chunks_by_document_returns_empty(self, backend: PgVectorBackend) -> None:
        ns_a = await backend.create_namespace(MemoryNamespace())
        ns_b = await backend.create_namespace(MemoryNamespace())

        doc = _make_doc(ns_a.id)
        await backend.create_document(doc)
        await backend.create_chunks_batch([_make_chunk(ns_a.id, doc.id) for _ in range(2)])

        # ns_B caller asks for doc — even guessing the doc id, namespace
        # filter short-circuits to [].
        assert await backend.get_chunks_by_document(doc.id, namespace_id=ns_b.id) == []
        # Same-namespace returns the rows.
        same_ns = await backend.get_chunks_by_document(doc.id, namespace_id=ns_a.id)
        assert len(same_ns) == 2

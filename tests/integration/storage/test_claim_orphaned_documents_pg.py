"""Integration tests for ``PostgreSQLBackend.claim_orphaned_documents`` (#885, #886).

Exercises:

- A PROCESSING document with an old ``updated_at`` is reclaimed (#885 -
  crashed-worker recovery).
- A fresh PROCESSING document is NOT reclaimed (cutoff respected).
- Two concurrent claims never return the same document (FOR UPDATE SKIP
  LOCKED, #886).

Requires a running PostgreSQL (``make dev``). Skipped automatically when
the configured ``KHORA_DATABASE_URL`` is unreachable.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.db.session import run_migrations
from khora.storage.backends.postgresql import PostgreSQLBackend

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
    be = PostgreSQLBackend(database_url=DATABASE_URL)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


def _doc(ns_id, *, status: DocumentStatus, updated_at: datetime) -> Document:
    return Document(
        namespace_id=ns_id,
        content="orphan content",
        checksum=uuid4().hex,
        status=status,
        updated_at=updated_at,
    )


@skip_no_pg
class TestClaimOrphanedDocumentsPg:
    async def test_stale_processing_doc_is_reclaimed(self, backend: PostgreSQLBackend) -> None:
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        stale = _doc(ns.id, status=DocumentStatus.PROCESSING, updated_at=now - timedelta(hours=1))
        await backend.create_document(stale)

        claimed = await backend.claim_orphaned_documents(
            ns.id,
            pending_before=now - timedelta(minutes=5),
            processing_before=now - timedelta(seconds=900),
            limit=100,
        )
        ids = {c.id for c in claimed}
        assert stale.id in ids
        reclaimed = next(c for c in claimed if c.id == stale.id)
        assert reclaimed.status == DocumentStatus.PROCESSING
        assert reclaimed.orphan_prior_status == "processing"

    async def test_fresh_processing_doc_not_reclaimed(self, backend: PostgreSQLBackend) -> None:
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        fresh = _doc(ns.id, status=DocumentStatus.PROCESSING, updated_at=now)
        await backend.create_document(fresh)

        claimed = await backend.claim_orphaned_documents(
            ns.id,
            pending_before=now - timedelta(minutes=5),
            processing_before=now - timedelta(seconds=900),
            limit=100,
        )
        assert fresh.id not in {c.id for c in claimed}

    async def test_concurrent_claims_do_not_overlap(self, backend: PostgreSQLBackend) -> None:
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        stale_ts = now - timedelta(hours=1)
        created = []
        for _ in range(20):
            d = _doc(ns.id, status=DocumentStatus.PROCESSING, updated_at=stale_ts)
            await backend.create_document(d)
            created.append(d.id)

        async def _claim():
            return await backend.claim_orphaned_documents(
                ns.id,
                pending_before=now - timedelta(minutes=5),
                processing_before=now - timedelta(seconds=900),
                limit=100,
            )

        a, b = await asyncio.gather(_claim(), _claim())
        ids_a = {d.id for d in a}
        ids_b = {d.id for d in b}
        # SKIP LOCKED: no document claimed by both concurrent callers.
        assert ids_a.isdisjoint(ids_b)
        # Between them they should claim all the stale docs from this namespace.
        assert ids_a | ids_b == set(created)

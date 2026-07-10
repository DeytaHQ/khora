"""Integration tests for checksum-dedup re-ingest semantics (#1464).

A crash mid-window in ``_remember_batch_impl`` half-ingests a document: chunks
commit at stage 3 but entities/relationships are never written and the document
row stays ``PENDING``. Before #1464 ``get_documents_by_checksums`` filtered only
``status != FAILED``, so the half-ingested ``PENDING`` row was returned as a
dedup hit and skipped forever on re-ingest - un-repairable without DB surgery.

The fix widens the dedup-skip exclusion: a checksum hit is re-ingestable when it
is FAILED **or** a stale PENDING (``updated_at`` older than the pending-processor
grace period). A *fresh* PENDING or a PROCESSING row is still a dedup hit, which
preserves the concurrent in-flight guard (two workers ingesting the same
checksum must not both full-ingest an in-flight document).

Exercises against real PostgreSQL:

- Every ``DocumentStatus`` seeded at a known ``updated_at`` and asserted for
  whether the checksum lookup returns it (dedup hit) or excludes it
  (re-ingestable), for both the batch (``get_documents_by_checksums``) and single
  (``get_document_by_checksum``) lookups.
- A faithful half-ingest simulation: a PENDING document with committed chunks but
  zero entities is a dedup hit while fresh and re-ingestable once stale.

Requires a running PostgreSQL (``make dev``). Skipped automatically when the
configured ``KHORA_DATABASE_URL`` is unreachable.
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from khora.core.models import Chunk, Document, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.db.session import run_migrations
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend

# This repo's compose puts Postgres on 5434 (see compose.yaml). Honor an explicit
# override but never fall back to the ambient dev DSN.
DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


pytestmark = [pytest.mark.integration]

# Matches the default pending_processor_grace_period_minutes (5 min): a PENDING
# doc older than this cutoff is a crash-abandoned half-ingest and re-ingests.
_GRACE = timedelta(minutes=5)


def _pg_reachable() -> bool:
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


@pytest.fixture
async def vector(_run_migrations_once):
    be = PgVectorBackend(database_url=DATABASE_URL, embedding_dimension=1536)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


def _doc(ns_id, *, checksum: str, status: DocumentStatus, updated_at: datetime) -> Document:
    return Document(
        namespace_id=ns_id,
        content="dedup content",
        checksum=checksum,
        status=status,
        updated_at=updated_at,
    )


@skip_no_pg
class TestChecksumDedupReingestPg:
    async def test_predicate_per_status_batch_lookup(self, backend: PostgreSQLBackend) -> None:
        """Only genuinely-ingested docs are dedup hits; FAILED and stale PENDING re-ingest."""
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        cutoff = now - _GRACE

        # One document per (status, freshness) at a distinct checksum.
        cases: dict[str, tuple[DocumentStatus, datetime, bool]] = {
            # checksum -> (status, updated_at, expect_dedup_hit)
            uuid4().hex: (DocumentStatus.COMPLETED, now, True),
            uuid4().hex: (DocumentStatus.PROCESSING, now, True),
            uuid4().hex: (DocumentStatus.PROCESSING, now - timedelta(hours=1), True),
            uuid4().hex: (DocumentStatus.ARCHIVED, now, True),
            uuid4().hex: (DocumentStatus.FAILED, now, False),
            uuid4().hex: (DocumentStatus.PENDING, now, True),  # fresh in-flight
            uuid4().hex: (DocumentStatus.PENDING, now - timedelta(hours=1), False),  # crash-abandoned
        }
        for cs, (status, updated_at, _) in cases.items():
            await backend.create_document(_doc(ns.id, checksum=cs, status=status, updated_at=updated_at))

        found = await backend.get_documents_by_checksums(ns.id, list(cases), pending_stale_before=cutoff)

        for cs, (status, updated_at, expect_hit) in cases.items():
            assert (cs in found) is expect_hit, (
                f"status={status.value} updated_at={'stale' if updated_at < cutoff else 'fresh'} "
                f"expected dedup_hit={expect_hit}, got in_result={cs in found}"
            )

    async def test_predicate_per_status_single_lookup(self, backend: PostgreSQLBackend) -> None:
        """Single-doc remember() dedup path shares the same predicate."""
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        cutoff = now - _GRACE

        stale_pending_cs = uuid4().hex
        fresh_pending_cs = uuid4().hex
        failed_cs = uuid4().hex
        completed_cs = uuid4().hex
        await backend.create_document(
            _doc(ns.id, checksum=stale_pending_cs, status=DocumentStatus.PENDING, updated_at=now - timedelta(hours=1))
        )
        await backend.create_document(
            _doc(ns.id, checksum=fresh_pending_cs, status=DocumentStatus.PENDING, updated_at=now)
        )
        await backend.create_document(_doc(ns.id, checksum=failed_cs, status=DocumentStatus.FAILED, updated_at=now))
        await backend.create_document(
            _doc(ns.id, checksum=completed_cs, status=DocumentStatus.COMPLETED, updated_at=now)
        )

        # Stale PENDING and FAILED are re-ingestable (lookup returns None).
        assert await backend.get_document_by_checksum(ns.id, stale_pending_cs, pending_stale_before=cutoff) is None
        assert await backend.get_document_by_checksum(ns.id, failed_cs, pending_stale_before=cutoff) is None
        # Fresh PENDING and COMPLETED are dedup hits (lookup returns the row).
        assert await backend.get_document_by_checksum(ns.id, fresh_pending_cs, pending_stale_before=cutoff) is not None
        assert await backend.get_document_by_checksum(ns.id, completed_cs, pending_stale_before=cutoff) is not None

    async def test_legacy_none_cutoff_only_excludes_failed(self, backend: PostgreSQLBackend) -> None:
        """Without a cutoff the predicate is backward-compatible (only FAILED excluded)."""
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        stale_pending_cs = uuid4().hex
        failed_cs = uuid4().hex
        await backend.create_document(
            _doc(ns.id, checksum=stale_pending_cs, status=DocumentStatus.PENDING, updated_at=now - timedelta(hours=1))
        )
        await backend.create_document(_doc(ns.id, checksum=failed_cs, status=DocumentStatus.FAILED, updated_at=now))

        found = await backend.get_documents_by_checksums(ns.id, [stale_pending_cs, failed_cs])
        # Legacy behavior: stale PENDING is still a dedup hit, FAILED still excluded.
        assert stale_pending_cs in found
        assert failed_cs not in found

    async def test_half_ingested_pending_reingests_when_stale(
        self, backend: PostgreSQLBackend, vector: PgVectorBackend
    ) -> None:
        """A crash mid-window leaves chunks but no entities; the stale PENDING doc re-ingests.

        Simulates the exact failure mode: create the document PENDING, commit its
        chunks (stage 3), never write entities (stage 6 never ran). Assert the
        half-ingest is visible (chunks present, no entities) and that the dedup
        lookup treats a *fresh* half-ingest as a hit (concurrent in-flight guard)
        but a *stale* one as re-ingestable (the #1464 repair path).
        """
        ns = await backend.create_namespace(MemoryNamespace())
        now = datetime.now(UTC)
        checksum = uuid4().hex

        # Stage 1: create the document PENDING (as the batch path does).
        doc = _doc(ns.id, checksum=checksum, status=DocumentStatus.PENDING, updated_at=now)
        doc = await backend.create_document(doc)

        # Stage 3: chunks commit to pgvector. Stage 6 (entities) never runs.
        await vector.create_chunks_batch(
            [
                Chunk(
                    namespace_id=ns.id,
                    document_id=doc.id,
                    content="half-ingested chunk",
                    chunk_index=0,
                    embedding=[0.1] * 1536,
                )
            ]
        )

        # The half-ingest is real: chunks exist, entities do not.
        chunks = await vector.get_chunks_by_document(doc.id, namespace_id=ns.id)
        assert len(chunks) == 1
        entities = await vector.list_entities(ns.id)
        assert entities == []

        # Fresh half-ingest: still a dedup hit (a concurrent worker may be mid-flight).
        fresh_cutoff = now - _GRACE
        assert checksum in await backend.get_documents_by_checksums(
            ns.id, [checksum], pending_stale_before=fresh_cutoff
        )

        # Stale half-ingest (worker presumed crashed): re-ingestable.
        stale_cutoff = now + _GRACE
        assert checksum not in await backend.get_documents_by_checksums(
            ns.id, [checksum], pending_stale_before=stale_cutoff
        )

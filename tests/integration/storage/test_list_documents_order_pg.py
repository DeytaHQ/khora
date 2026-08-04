"""``PostgreSQLBackend.list_documents`` total-order tests.

``list_documents`` sorts on ``(created_at DESC, id DESC)``. Bulk ingest stamps
many documents with the same ``created_at``; on that seed ``created_at`` alone
leaves the sort under-determined - Postgres is free to return tied rows in any
order, and its sort is not stable, so row positions are not addressable and
offset paging can serve the same document twice or skip one.

See :mod:`tests.test_helpers.document_order` for why the seed is non-vacuous.

Requires a running PostgreSQL (``make dev``). Skipped automatically when the
configured ``KHORA_DATABASE_URL`` is unreachable.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.db.session import run_migrations
from khora.storage.backends.postgresql import PostgreSQLBackend
from tests.test_helpers.document_order import id_ladder, seed_order, walk_pages

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5432/khora",
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


pytestmark = [pytest.mark.integration]

SEED_SIZE = 12


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


@pytest.fixture
async def seeded(backend: PostgreSQLBackend):
    """A fresh namespace holding ``SEED_SIZE`` documents that share ``created_at``.

    Rows are written in ``seed_order`` (non-monotonic by id) so that neither
    insertion order nor its reverse can coincide with the expected sequence.
    """
    ns = await backend.create_namespace(MemoryNamespace())
    shared_created_at = datetime.now(UTC)
    ids = id_ladder(SEED_SIZE)
    for i, doc_id in enumerate(seed_order(ids)):
        await backend.create_document(
            Document(
                id=doc_id,
                namespace_id=ns.id,
                content="tied content",
                checksum=f"sum-{i}",
                created_at=shared_created_at,
                updated_at=shared_created_at,
            )
        )
    return ns, ids


@skip_no_pg
class TestListDocumentsTotalOrderPg:
    async def test_ties_break_on_id_desc_and_repeat_identically(self, backend: PostgreSQLBackend, seeded) -> None:
        ns, ids = seeded

        docs = await backend.list_documents(ns.id)

        # The tie is real: if these differed, ``created_at`` alone would decide
        # the order and the id tie-break would never be exercised.
        assert len({d.created_at for d in docs}) == 1
        assert [d.id for d in docs] == sorted(ids, reverse=True)

        # Same query, same answer - the order is a property of the query, not
        # of whatever the scan happened to produce on the first call.
        for _ in range(2):
            assert [d.id for d in await backend.list_documents(ns.id)] == [d.id for d in docs]

    async def test_offset_pagination_is_exhaustive_and_non_overlapping(
        self, backend: PostgreSQLBackend, seeded
    ) -> None:
        ns, ids = seeded
        expected = sorted(ids, reverse=True)

        # Page size deliberately does not divide the seed size, so the final
        # page is short and an off-by-one at the boundary shows up.
        pages = await walk_pages(backend.list_documents, ns.id, page_size=5)
        seen = [d.id for page in pages for d in page]

        assert [len(p) for p in pages] == [5, 5, 2]
        assert len(seen) == len(set(seen))  # no document served twice
        assert set(seen) == set(expected)  # every document served
        assert seen == expected  # and in the same order as the unpaged read

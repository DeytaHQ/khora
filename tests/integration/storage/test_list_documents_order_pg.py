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

import pytest

from khora.core.models import Document, MemoryNamespace
from khora.db.session import run_migrations
from khora.storage.backends.postgresql import PostgreSQLBackend
from tests.test_helpers.document_order import order_seed, walk_pages

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    # This repo's compose puts Postgres on 5434 (see compose.yaml); defaulting to
    # 5432 would make the whole class silently skip on a local `make test`.
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
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
    """A fresh namespace holding ``SEED_SIZE`` documents pinning both sort keys.

    Rows are written in ``seed_order`` (non-monotonic by id) so that neither
    insertion order nor its reverse can coincide with the expected sequence.
    """
    ns = await backend.create_namespace(MemoryNamespace())
    seed = order_seed(SEED_SIZE)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await backend.create_document(
            Document(
                id=doc_id,
                namespace_id=ns.id,
                content="tied content",
                checksum=f"sum-{i}",
                created_at=created_at,
                updated_at=created_at,
            )
        )
    return ns, seed


@skip_no_pg
class TestListDocumentsTotalOrderPg:
    async def test_orders_by_created_at_then_id_desc_and_repeats_identically(
        self, backend: PostgreSQLBackend, seeded
    ) -> None:
        ns, seed = seeded

        docs = await backend.list_documents(ns.id)

        # The tie is real: if these differed, ``created_at`` alone would decide
        # the order and the id tie-break would never be exercised.
        assert len({d.created_at for d in docs if d.id in seed.tied_ids}) == 1

        # ``created_at`` leads: these two rows carry the id that would put them
        # at the opposite end, so only a leading ``created_at`` lands them here.
        assert docs[0].id == seed.newest_id
        assert docs[-1].id == seed.oldest_id

        assert [d.id for d in docs] == seed.expected

        # Same query, same answer - the order is a property of the query, not
        # of whatever the scan happened to produce on the first call.
        for _ in range(2):
            assert [d.id for d in await backend.list_documents(ns.id)] == [d.id for d in docs]

    async def test_offset_pagination_is_exhaustive_and_non_overlapping(
        self, backend: PostgreSQLBackend, seeded
    ) -> None:
        ns, seed = seeded
        expected = seed.expected

        # Page size deliberately does not divide the seed size, so the final
        # page is short and an off-by-one at the boundary shows up.
        pages = await walk_pages(backend.list_documents, ns.id, page_size=5)
        seen = [d.id for page in pages for d in page]

        assert [len(p) for p in pages] == [5, 5, 2]
        assert len(seen) == len(set(seen))  # no document served twice
        assert set(seen) == set(expected)  # every document served
        assert seen == expected  # and in the same order as the unpaged read

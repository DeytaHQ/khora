"""``SurrealDBRelationalAdapter.list_documents`` total-order tests.

``list_documents`` sorts on ``(created_at DESC, id DESC)``. Bulk ingest stamps
many documents with the same ``created_at``; on that seed ``created_at`` alone
leaves the sort under-determined and row positions are not addressable.

The SurrealDB leg is worth its own coverage because ``id`` is not a plain UUID
column here - it is a ``RecordID`` (``document:<uuid>``), so the SurrealQL
``ORDER BY id DESC`` sorts record ids rather than UUIDs. These tests pin that
the resulting sequence is a strict total order and that it agrees with
descending UUID order, which is what a later keyset cursor would rely on.

See :mod:`tests.test_helpers.document_order` for why the seed is non-vacuous.

Runs against an in-memory SurrealDB (``mode="memory"``) - no docker required.
Skipped when the ``surrealdb`` extra is not installed.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Document, MemoryNamespace, TenancyMode  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402
from tests.test_helpers.document_order import (  # noqa: E402
    OrderSeed,
    order_seed,
    walk_pages,
)

pytestmark = pytest.mark.integration

SEED_SIZE = 12


@pytest.fixture
async def adapter():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="doc_order")
    await conn.connect()
    adapter = SurrealDBRelationalAdapter(conn)
    try:
        yield adapter
    finally:
        await conn.disconnect()


@pytest.fixture
async def namespace(adapter):
    nid = uuid4()
    ns = MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED)
    return await adapter.create_namespace(ns)


async def _seed_ordered_documents(adapter, namespace) -> OrderSeed:
    """Seed ``SEED_SIZE`` documents pinning both sort keys.

    Rows are written in ``seed_order`` (non-monotonic by id) so that neither
    insertion order nor its reverse can coincide with the expected sequence.
    """
    seed = order_seed(SEED_SIZE)
    for i, (doc_id, created_at) in enumerate(seed.writes):
        await adapter.create_document(
            Document(
                id=doc_id,
                namespace_id=namespace.id,
                content="tied content",
                checksum=f"sum-{i}",
                created_at=created_at,
                updated_at=created_at,
            )
        )
    return seed


async def test_orders_by_created_at_then_id_desc_and_repeats_identically(adapter, namespace) -> None:
    seed = await _seed_ordered_documents(adapter, namespace)

    docs = await adapter.list_documents(namespace.id)

    # The tie is real: if these differed, ``created_at`` alone would decide the
    # order and the id tie-break would never be exercised.
    assert len({d.created_at for d in docs if d.id in seed.tied_ids}) == 1

    # ``created_at`` leads: these two rows carry the id that would put them at
    # the opposite end, so only a leading ``created_at`` lands them here.
    assert docs[0].id == seed.newest_id
    assert docs[-1].id == seed.oldest_id

    # RecordID ordering agrees with descending UUID order.
    assert [d.id for d in docs] == seed.expected

    # Same query, same answer - the order is a property of the query, not of
    # whatever the scan happened to produce on the first call.
    for _ in range(2):
        assert [d.id for d in await adapter.list_documents(namespace.id)] == [d.id for d in docs]


async def test_offset_pagination_is_exhaustive_and_non_overlapping(adapter, namespace) -> None:
    seed = await _seed_ordered_documents(adapter, namespace)
    expected = seed.expected

    # Page size deliberately does not divide the seed size, so the final page is
    # short and an off-by-one at the boundary shows up.
    pages = await walk_pages(adapter.list_documents, namespace.id, page_size=5)
    seen = [d.id for page in pages for d in page]

    assert [len(p) for p in pages] == [5, 5, 2]
    assert len(seen) == len(set(seen))  # no document served twice
    assert set(seen) == set(expected)  # every document served
    assert seen == expected  # and in the same order as the unpaged read

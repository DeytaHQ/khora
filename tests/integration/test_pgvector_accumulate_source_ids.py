"""Regression test for #1039: ``PgVectorBackend`` entity upsert must
*accumulate* source provenance (``source_document_ids`` /
``source_chunk_ids``) across re-upserts instead of clobbering it.

Bug: the ``ON CONFLICT DO UPDATE`` branch previously set both arrays to
``excluded.*`` (the incoming row's ids only), so re-extracting an entity
from a second document dropped the first document's provenance. forget()'s
survivor-strip then could not see which documents an entity came from, and
the pgvector half of a PG+Neo4j dual-write disagreed with the Neo4j half
(which uses ``ON MATCH SET e.source_*_ids = (existing + incoming)[-N..]``).

Fix: ``_accumulate_source_ids_sql`` unions existing+incoming uuid[] arrays,
dedups, and keeps each column's most-recent DISTINCT ids newest-last
(``_SOURCE_DOCUMENT_IDS_CAP`` =100 for documents, ``_SOURCE_CHUNK_IDS_CAP``
=250 for chunks, mirroring Neo4j). The dedup is what guards the core
regression here: re-extracting the SAME document many times must NOT evict a
prior *distinct* provenance id by burning through the cap.

Gated by ``KHORA_DATABASE_URL`` (defaults to the ``make dev`` Postgres on
port 5432). The test needs real Postgres — the accumulation is implemented
as a Postgres ``unnest ... WITH ORDINALITY`` / ``array_agg`` set-expression
that a SQLite mock cannot reproduce.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from khora.core.models import Entity
from khora.storage.backends.pgvector import _SOURCE_CHUNK_IDS_CAP, PgVectorBackend

# Match the schema's pgvector column dimension (Vector(1536)).
EMBED_DIM = 1536


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    url = os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5432/khora",
    )
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _pg_reachable(), reason="PostgreSQL not reachable (run `make dev`)"),
]


@pytest.fixture
async def backend() -> AsyncIterator[PgVectorBackend]:
    database_url = os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5432/khora",
    )
    be = PgVectorBackend(database_url=database_url, embedding_dimension=EMBED_DIM)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@pytest.fixture
async def namespace_id(backend: PgVectorBackend) -> AsyncIterator[UUID]:
    """Create a fresh namespace row directly so the test owns its tear-down
    surface and doesn't depend on the Khora façade."""
    ns_id = uuid4()
    engine = create_async_engine(backend._database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO memory_namespaces (id, namespace_id, version, "
                    "is_active, tenancy_mode, created_at, updated_at) "
                    "VALUES (:id, :nsid, 1, true, 'shared', NOW(), NOW())"
                ),
                {"id": ns_id, "nsid": ns_id},
            )
        yield ns_id
        async with engine.begin() as conn:
            await conn.execute(sa.text("DELETE FROM memory_namespaces WHERE id = :id"), {"id": ns_id})
    finally:
        await engine.dispose()


def _entity(
    ns_id: UUID,
    name: str,
    *,
    entity_type: str = "PERSON",
    source_document_ids: list[UUID] | None = None,
    source_chunk_ids: list[UUID] | None = None,
) -> Entity:
    """Build a fresh Entity with a new candidate UUID — mimics the
    LLM-ingestion shape where every extraction gets a new candidate id and
    storage dedupes by (namespace, name, type)."""
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        embedding=[0.1] * EMBED_DIM,
        source_document_ids=list(source_document_ids or []),
        source_chunk_ids=list(source_chunk_ids or []),
    )


async def _read_back(backend: PgVectorBackend, ns_id: UUID, name: str) -> Entity:
    """Resolve the persisted entity by name (identity = namespace+name+type).

    Read-by-name rather than read-by-id because re-upserts carry a fresh
    candidate id while the persisted row keeps the original id — name is the
    stable handle the public API exposes via ``get_entities_by_names_batch``.
    """
    by_name = await backend.get_entities_by_names_batch(ns_id, [name])
    assert name in by_name, f"entity {name!r} not found after upsert"
    return by_name[name]


# =========================================================================
# Batch path (the hot ingest path)
# =========================================================================


@pytest.mark.asyncio
async def test_batch_upsert_accumulates_and_dedups_source_ids(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Re-upserting the same identity through ``upsert_entities_batch`` with
    a *different* document/chunk must union the provenance (not clobber it),
    and a third upsert of an already-recorded id must not duplicate it."""
    name = f"acc-batch-{uuid4()}"
    doc_a, doc_b = uuid4(), uuid4()
    chunk_a, chunk_b = uuid4(), uuid4()

    await backend.upsert_entities_batch(
        namespace_id,
        [_entity(namespace_id, name, source_document_ids=[doc_a], source_chunk_ids=[chunk_a])],
    )
    await backend.upsert_entities_batch(
        namespace_id,
        [_entity(namespace_id, name, source_document_ids=[doc_b], source_chunk_ids=[chunk_b])],
    )

    persisted = await _read_back(backend, namespace_id, name)
    assert set(persisted.source_document_ids) == {doc_a, doc_b}
    assert set(persisted.source_chunk_ids) == {chunk_a, chunk_b}
    # Newest-at-tail: doc_b was upserted after doc_a, so it must land at the
    # tail — this locks the tail-cap contract (the cap evicts from the head).
    assert persisted.source_document_ids[-1] == doc_b

    # Re-upsert an already-recorded id — must not create a duplicate.
    await backend.upsert_entities_batch(
        namespace_id,
        [_entity(namespace_id, name, source_document_ids=[doc_b], source_chunk_ids=[chunk_b])],
    )

    persisted = await _read_back(backend, namespace_id, name)
    assert set(persisted.source_document_ids) == {doc_a, doc_b}
    assert len(persisted.source_document_ids) == 2, "doc_b must not be duplicated"
    assert set(persisted.source_chunk_ids) == {chunk_a, chunk_b}
    assert len(persisted.source_chunk_ids) == 2, "chunk_b must not be duplicated"


@pytest.mark.asyncio
async def test_batch_reextracting_same_doc_does_not_evict_prior_distinct_id(
    backend: PgVectorBackend, namespace_id: UUID
) -> None:
    """Core regression: re-extracting the SAME document/chunk many more times
    than the retention cap must NOT push a prior *distinct* provenance id out.

    Without the dedup pass, a flood of identical incoming ids would burn
    through the column's cap slots and evict ``first_doc`` / ``first_chunk``
    from the tail. The dedup collapses the repeats to a single retained id, so
    the prior distinct ids survive. Both columns share ``_accumulate_source_ids_sql``,
    so the chunk column is asserted the same way.
    """
    name = f"acc-evict-{uuid4()}"
    first_doc = uuid4()
    spammy_doc = uuid4()
    first_chunk = uuid4()
    spammy_chunk = uuid4()

    # Record distinct prior ids once.
    await backend.upsert_entities_batch(
        namespace_id,
        [_entity(namespace_id, name, source_document_ids=[first_doc], source_chunk_ids=[first_chunk])],
    )

    # Re-extract the same (spammy) doc/chunk far more times than either cap.
    # Separate upsert calls — a single batch cannot carry the same identity
    # twice (ON CONFLICT cannot affect a row a second time), and repeated calls
    # are the real re-ingest shape anyway.
    for _ in range(_SOURCE_CHUNK_IDS_CAP + 20):
        await backend.upsert_entities_batch(
            namespace_id,
            [_entity(namespace_id, name, source_document_ids=[spammy_doc], source_chunk_ids=[spammy_chunk])],
        )

    persisted = await _read_back(backend, namespace_id, name)
    assert first_doc in persisted.source_document_ids, "prior distinct doc id was evicted by repeats"
    assert set(persisted.source_document_ids) == {first_doc, spammy_doc}
    assert len(persisted.source_document_ids) == 2, "doc repeats must dedup to a single retained id"
    assert first_chunk in persisted.source_chunk_ids, "prior distinct chunk id was evicted by repeats"
    assert set(persisted.source_chunk_ids) == {first_chunk, spammy_chunk}
    assert len(persisted.source_chunk_ids) == 2, "chunk repeats must dedup to a single retained id"


# =========================================================================
# source_chunk_ids provenance filter on list_entities (#1448)
# =========================================================================


@pytest.mark.asyncio
async def test_list_entities_filters_by_source_chunk_ids(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """``list_entities(source_chunk_ids=...)`` filters entities by chunk
    provenance (any-overlap), per #1448.

    Seeds two entities — A sourced from chunks c1/c2, B from c3 — then pins
    the four contract cases: no filter returns both; a filter for one of A's
    chunks returns only A; an unknown chunk returns nothing; and an empty
    list matches nothing.
    """
    name_a = f"chunk-filter-A-{uuid4()}"
    name_b = f"chunk-filter-B-{uuid4()}"
    c1, c2, c3, c4 = uuid4(), uuid4(), uuid4(), uuid4()

    await backend.upsert_entities_batch(
        namespace_id,
        [
            _entity(namespace_id, name_a, source_chunk_ids=[c1, c2]),
            _entity(namespace_id, name_b, source_chunk_ids=[c3]),
        ],
    )

    # 1. No filter → both entities.
    all_names = {e.name for e in await backend.list_entities(namespace_id)}
    assert {name_a, name_b} <= all_names

    # 2. One of A's chunks → exactly A.
    only_a = await backend.list_entities(namespace_id, source_chunk_ids=[c1])
    assert {e.name for e in only_a} == {name_a}

    # 3. Unknown chunk id → nothing.
    assert await backend.list_entities(namespace_id, source_chunk_ids=[c4]) == []

    # 4. Empty list → matches nothing.
    assert await backend.list_entities(namespace_id, source_chunk_ids=[]) == []


# =========================================================================
# Single path (create_entity / _upsert_entity)
# =========================================================================


@pytest.mark.asyncio
async def test_single_upsert_accumulates_and_dedups_source_ids(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """The single-entity path (``create_entity`` → ``_upsert_entity``) must
    accumulate provenance on conflict identically to the batch path."""
    name = f"acc-single-{uuid4()}"
    doc_a, doc_b = uuid4(), uuid4()
    chunk_a, chunk_b = uuid4(), uuid4()

    await backend.create_entity(_entity(namespace_id, name, source_document_ids=[doc_a], source_chunk_ids=[chunk_a]))
    await backend.create_entity(_entity(namespace_id, name, source_document_ids=[doc_b], source_chunk_ids=[chunk_b]))

    persisted = await _read_back(backend, namespace_id, name)
    assert set(persisted.source_document_ids) == {doc_a, doc_b}
    assert set(persisted.source_chunk_ids) == {chunk_a, chunk_b}

    # Re-upsert an already-recorded id — must not duplicate.
    await backend.create_entity(_entity(namespace_id, name, source_document_ids=[doc_b], source_chunk_ids=[chunk_b]))

    persisted = await _read_back(backend, namespace_id, name)
    assert set(persisted.source_document_ids) == {doc_a, doc_b}
    assert len(persisted.source_document_ids) == 2, "doc_b must not be duplicated"
    assert len(persisted.source_chunk_ids) == 2, "chunk_b must not be duplicated"

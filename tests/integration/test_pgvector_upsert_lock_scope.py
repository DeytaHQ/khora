"""Regression test for #1471 (advisory-lock scope): the namespace-scoped
``pg_advisory_xact_lock`` in ``upsert_entities_batch`` /
``create_relationships_batch`` now commits per sub-batch, releasing the lock
between sub-batches instead of holding it across the whole write.

Two guarantees this test protects:

1. **Concurrency correctness.** Two concurrent same-namespace batch upserts
   that share hub entities must both complete without deadlocking and both
   persist their writes. The advisory lock still serialises each multi-row
   ``INSERT ... ON CONFLICT DO UPDATE`` statement, so the hub-node row-lock
   deadlock the lock was preventing cannot resurface even though the lock is
   now held for shorter windows.

2. **Multi-sub-batch upserts still work.** With ``batch_size`` small enough to
   split a single call into several sub-batches (each its own transaction), the
   canonical-id sync and per-entity ``is_new`` reporting must still be correct.

Gated by ``KHORA_DATABASE_URL`` (this repo's compose stack puts Postgres on
5434). Needs real Postgres - the advisory lock and ``xmax`` idiom are
Postgres-only, a SQLite mock cannot reproduce either.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from khora.core.models import Entity, Relationship
from khora.storage.backends.pgvector import PgVectorBackend

EMBED_DIM = 1536

# This repo's compose puts Postgres on 5434; honor an explicit override, else
# default to the compose port (mirrors tests/integration/conftest.py).
_DEFAULT_DATABASE_URL = "postgresql+asyncpg://khora:khora@localhost:5434/khora"


def _database_url() -> str:
    url = os.environ.get("KHORA_DATABASE_URL", _DEFAULT_DATABASE_URL)
    if "+asyncpg" not in url and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(_database_url().replace("+asyncpg", ""))
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
    be = PgVectorBackend(database_url=_database_url(), embedding_dimension=EMBED_DIM)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@pytest.fixture
async def namespace_id(backend: PgVectorBackend) -> AsyncIterator[UUID]:
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
            await conn.execute(sa.text("DELETE FROM relationships WHERE namespace_id = :id"), {"id": ns_id})
            await conn.execute(sa.text("DELETE FROM entities WHERE namespace_id = :id"), {"id": ns_id})
            await conn.execute(sa.text("DELETE FROM memory_namespaces WHERE id = :id"), {"id": ns_id})
    finally:
        await engine.dispose()


def _entity(ns_id: UUID, name: str, entity_type: str = "PERSON") -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        embedding=[0.1] * EMBED_DIM,
    )


@pytest.mark.asyncio
async def test_concurrent_same_namespace_upserts_no_deadlock(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Two concurrent batch upserts into the same namespace that share hub
    entities must both complete (no deadlock) and both persist. This is the
    scenario the advisory lock guards; the per-sub-batch commit must not let
    the hub-node deadlock resurface."""
    tag = uuid4().hex[:8]
    # Overlapping "hub" entities present in both batches (the deadlock surface)
    # plus batch-unique entities so each write touches shared and private rows.
    hubs = [f"hub-{tag}-{i}" for i in range(20)]
    batch_a = [_entity(namespace_id, n) for n in hubs] + [_entity(namespace_id, f"a-{tag}-{i}") for i in range(20)]
    batch_b = [_entity(namespace_id, n) for n in hubs] + [_entity(namespace_id, f"b-{tag}-{i}") for i in range(20)]

    # Small batch_size forces multiple sub-batches per call, so the lock is
    # acquired/released several times per call - maximising interleave and the
    # chance to trip a deadlock if the serialiser were dropped.
    results = await asyncio.gather(
        backend.upsert_entities_batch(namespace_id, batch_a, batch_size=8),
        backend.upsert_entities_batch(namespace_id, batch_b, batch_size=8),
    )

    assert len(results[0]) == len(batch_a)
    assert len(results[1]) == len(batch_b)

    # Both batches persisted: hub rows dedupe to one each, private rows unique.
    engine = create_async_engine(backend._database_url)
    try:
        async with engine.connect() as conn:
            total = (
                await conn.execute(
                    sa.text("SELECT count(*) FROM entities WHERE namespace_id = :ns"),
                    {"ns": namespace_id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    # 20 hubs (deduped) + 20 a-unique + 20 b-unique = 60.
    assert total == 60


@pytest.mark.asyncio
async def test_multi_sub_batch_upsert_syncs_ids_and_is_new(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """A single call split across several sub-batch transactions (small
    batch_size) must still canonicalise each input entity's id and report
    per-entity is_new correctly across the sub-batch boundary."""
    tag = uuid4().hex[:8]
    names = [f"e-{tag}-{i}" for i in range(10)]
    first = [_entity(namespace_id, n) for n in names]

    r1 = await backend.upsert_entities_batch(namespace_id, first, batch_size=3)
    assert len(r1) == 10
    assert all(is_new for _, is_new in r1), "all fresh entities should be is_new=True"
    # ids were synced to canonical stored rows (fresh insert keeps input id).
    ids_by_name = {e.name: e.id for e, _ in r1}

    # Re-upsert the same identities with brand-new candidate UUIDs; every one
    # must report is_new=False and be canonicalised back to the stored id.
    second = [_entity(namespace_id, n) for n in names]
    r2 = await backend.upsert_entities_batch(namespace_id, second, batch_size=3)
    assert len(r2) == 10
    assert all(not is_new for _, is_new in r2), "re-upsert of existing rows should be is_new=False"
    for e, _ in r2:
        assert e.id == ids_by_name[e.name], "id must be canonicalised to the stored row id"


@pytest.mark.asyncio
async def test_concurrent_relationship_inserts_no_deadlock(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Concurrent relationship-batch inserts into the same namespace complete
    without deadlock and both persist (same advisory-lock guard, now
    per-sub-batch commit)."""
    tag = uuid4().hex[:8]
    # Two entities to hang relationships off of.
    src = _entity(namespace_id, f"src-{tag}")
    tgt = _entity(namespace_id, f"tgt-{tag}")
    await backend.upsert_entities_batch(namespace_id, [src, tgt])

    def _rel() -> Relationship:
        return Relationship(
            id=uuid4(),
            namespace_id=namespace_id,
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relationship_type="RELATES_TO",
        )

    rels_a = [_rel() for _ in range(20)]
    rels_b = [_rel() for _ in range(20)]

    results = await asyncio.gather(
        backend.create_relationships_batch(rels_a, batch_size=8),
        backend.create_relationships_batch(rels_b, batch_size=8),
    )
    assert len(results[0]) == len(rels_a)
    assert len(results[1]) == len(rels_b)

    engine = create_async_engine(backend._database_url)
    try:
        async with engine.connect() as conn:
            total = (
                await conn.execute(
                    sa.text("SELECT count(*) FROM relationships WHERE namespace_id = :ns"),
                    {"ns": namespace_id},
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert total == 40

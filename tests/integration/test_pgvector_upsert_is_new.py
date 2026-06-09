"""Regression test for #719: ``PgVectorBackend.upsert_entities_batch`` must
return ``is_new=False`` for entities that already exist in the namespace.

Bug: prior to this fix the method hardcoded ``[(entity, True) for entity in
sorted_entities]`` regardless of whether the row was inserted or updated.
Downstream telemetry / coordinator counters that key off ``is_new=True``
were silently inflated, and the pgvector half of a dual-write disagreed
with the Neo4j half (which uses ``MERGE`` + ``ON CREATE``).

Fix: use Postgres's ``RETURNING (xmax = 0) AS is_new`` idiom — ``xmax`` is
``0`` on freshly inserted rows and a non-zero locking-tx id on rows that
took the ``ON CONFLICT DO UPDATE`` branch.

Gated by ``KHORA_DATABASE_URL`` (defaults to the ``make dev`` Postgres on
port 5432). The test needs real Postgres because the bug only manifests
with the ``xmax`` system column — a SQLite mock cannot reproduce it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from khora.core.models import Entity
from khora.storage.backends.pgvector import PgVectorBackend

# Match the schema's pgvector column dimension (Vector(1536)).
EMBED_DIM = 1536


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    url = os.environ.get(
        "KHORA_DATABASE_URL",
        "postgresql+asyncpg://khora:khora@localhost:5434/khora",
    )
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5434
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
        "postgresql+asyncpg://khora:khora@localhost:5434/khora",
    )
    be = PgVectorBackend(database_url=database_url, embedding_dimension=EMBED_DIM)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@pytest.fixture
async def namespace_id(backend: PgVectorBackend) -> UUID:
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


def _entity(ns_id: UUID, name: str, entity_type: str = "PERSON") -> Entity:
    """Build a fresh Entity with a new UUID — mimics the LLM-ingestion shape
    where every extraction gets a candidate id and storage dedupes by
    (namespace, name, type)."""
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        embedding=[0.1] * EMBED_DIM,
    )


@pytest.mark.asyncio
async def test_is_new_true_on_first_upsert_false_on_second(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """The same logical entity upserted twice (different candidate UUIDs)
    must report ``is_new=True`` on the first call and ``is_new=False`` on
    the second. Mirrors the issue's repro."""
    name = f"alice-{uuid4()}"

    result_1 = await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, name)])
    assert len(result_1) == 1
    assert result_1[0][1] is True, f"first upsert should report is_new=True, got {result_1[0][1]}"

    result_2 = await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, name)])
    assert len(result_2) == 1
    assert result_2[0][1] is False, f"second upsert should report is_new=False, got {result_2[0][1]}"


@pytest.mark.asyncio
async def test_mixed_batch_reports_per_entity_is_new(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """A batch containing both new and pre-existing entities must report
    ``is_new`` correctly for each — not a single uniform flag for the batch."""
    existing_name = f"existing-{uuid4()}"
    # Seed one entity so it's already in the table.
    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, existing_name)])

    new_name_a = f"new-a-{uuid4()}"
    new_name_b = f"new-b-{uuid4()}"
    batch = [
        _entity(namespace_id, new_name_a),
        _entity(namespace_id, existing_name),
        _entity(namespace_id, new_name_b),
    ]

    result = await backend.upsert_entities_batch(namespace_id, batch)
    is_new_by_name = {entity.name: is_new for entity, is_new in result}

    assert is_new_by_name[new_name_a] is True
    assert is_new_by_name[new_name_b] is True
    assert is_new_by_name[existing_name] is False


@pytest.mark.asyncio
async def test_same_name_different_type_treated_as_new(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """The unique constraint is on ``(namespace_id, name, entity_type)`` —
    two upserts with the same name but different types are independent
    entities and both must be ``is_new=True``."""
    name = f"polysemic-{uuid4()}"
    r1 = await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, name, "PERSON")])
    r2 = await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, name, "ORGANIZATION")])

    assert r1[0][1] is True
    assert r2[0][1] is True, "different entity_type must be a new row, not an update"

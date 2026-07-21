"""Integration test for the empty-attributes upsert guard on the pgvector
entity backend.

On entity upsert, an incoming EMPTY (``{}``) or NULL ``attributes`` must not
clobber a stored, populated ``attributes``. A non-empty incoming value
overwrites the stored dict wholesale (no key-union merge). The guard lives in
``PgVectorBackend._upsert_entity`` as a ``CASE`` on the
``on_conflict_do_update`` set-clause and is reached through the single-entity
``create_entity`` / ``update_entity`` path.

Gated by ``KHORA_DATABASE_URL`` (defaults to the ``make dev`` Postgres on port
5432). The guard is a SQL ``CASE`` over ``excluded.attributes`` vs the stored
column, so it can only be exercised against real Postgres — a SQLite mock
cannot reproduce the ``ON CONFLICT DO UPDATE`` semantics.
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
            await conn.execute(sa.text("DELETE FROM entities WHERE namespace_id = :id"), {"id": ns_id})
            await conn.execute(sa.text("DELETE FROM memory_namespaces WHERE id = :id"), {"id": ns_id})
    finally:
        await engine.dispose()


def _entity(ns_id: UUID, entity_id: UUID, name: str, attributes, entity_type: str = "PERSON") -> Entity:
    """Build an Entity with a caller-controlled ``id`` so re-upserts of the
    same logical entity read back deterministically by id.

    ``attributes`` is passed verbatim (including ``None``): the Entity
    dataclass does not coerce ``None`` -> ``{}`` in ``__post_init__``. The ORM
    column is ``JSONB`` with ``none_as_null=False``, so a ``None`` binds as
    ``'null'::jsonb`` (not a SQL NULL) and is caught by the guard's
    ``jsonb_typeof(excluded.attributes) = 'null'`` branch.
    """
    return Entity(
        id=entity_id,
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        attributes=attributes,
        embedding=[0.1] * EMBED_DIM,
    )


@pytest.mark.asyncio
async def test_empty_attributes_does_not_overwrite_populated(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Upsert ``{"a": 1}`` then the same key with ``{}`` — the stored
    attributes must stay ``{"a": 1}`` (empty incoming does not clobber)."""
    eid = uuid4()
    name = f"guard-empty-{uuid4()}"

    await backend.create_entity(_entity(namespace_id, eid, name, {"a": 1}))
    await backend.create_entity(_entity(namespace_id, eid, name, {}))

    got = await backend.get_entity(eid, namespace_id=namespace_id)
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_null_attributes_does_not_overwrite_populated(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Upsert ``{"a": 1}`` then the same key with NULL (``None``) attributes —
    the stored attributes must stay ``{"a": 1}``.

    Distinct from the empty-dict case: ``None`` binds as ``'null'::jsonb``
    (JSONB ``none_as_null=False``), so it is caught by the guard's
    ``jsonb_typeof(...) = 'null'`` branch rather than the ``== '{}'`` branch."""
    eid = uuid4()
    name = f"guard-null-{uuid4()}"

    await backend.create_entity(_entity(namespace_id, eid, name, {"a": 1}))
    await backend.create_entity(_entity(namespace_id, eid, name, None))

    got = await backend.get_entity(eid, namespace_id=namespace_id)
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_nonempty_attributes_overwrites_without_key_union(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Upsert ``{"a": 1}`` then the same key with ``{"b": 2}`` — the stored
    attributes must become EXACTLY ``{"b": 2}``.

    The guard only protects against empty/NULL incoming values; a non-empty
    incoming dict replaces the stored one wholesale. Key-union merge
    (``{"a": 1, "b": 2}``) is intentionally OUT OF SCOPE."""
    eid = uuid4()
    name = f"guard-overwrite-{uuid4()}"

    await backend.create_entity(_entity(namespace_id, eid, name, {"a": 1}))
    await backend.create_entity(_entity(namespace_id, eid, name, {"b": 2}))

    got = await backend.get_entity(eid, namespace_id=namespace_id)
    assert got is not None
    assert got.attributes == {"b": 2}
    assert "a" not in got.attributes


@pytest.mark.asyncio
async def test_batch_empty_attributes_does_not_overwrite_populated(
    backend: PgVectorBackend, namespace_id: UUID
) -> None:
    """Batch path (``upsert_entities_batch``): upsert ``{"a": 1}`` then the
    same key with ``{}`` — the stored attributes must stay ``{"a": 1}``.

    The batch multi-row ``INSERT ... ON CONFLICT DO UPDATE`` carries the same
    ``CASE`` guard as the single ``create_entity`` path; this exercises it via
    ``upsert_entities_batch`` rather than ``_upsert_entity``."""
    eid = uuid4()
    name = f"guard-batch-empty-{uuid4()}"

    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, eid, name, {"a": 1})])
    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, eid, name, {})])

    got = await backend.get_entity(eid, namespace_id=namespace_id)
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_batch_null_attributes_does_not_overwrite_populated(backend: PgVectorBackend, namespace_id: UUID) -> None:
    """Batch path: upsert ``{"a": 1}`` then the same key with NULL (``None``)
    attributes — the stored attributes must stay ``{"a": 1}``.

    Distinct from the empty-dict case: ``None`` binds as ``'null'::jsonb``
    (JSONB ``none_as_null=False``), so it is caught by the guard's
    ``jsonb_typeof(...) = 'null'`` branch rather than the ``== '{}'`` branch."""
    eid = uuid4()
    name = f"guard-batch-null-{uuid4()}"

    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, eid, name, {"a": 1})])
    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, eid, name, None)])

    got = await backend.get_entity(eid, namespace_id=namespace_id)
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_batch_nonempty_attributes_overwrites_without_key_union(
    backend: PgVectorBackend, namespace_id: UUID
) -> None:
    """Batch path: upsert ``{"a": 1}`` then the same key with ``{"b": 2}`` —
    the stored attributes must become EXACTLY ``{"b": 2}``.

    The guard only protects against empty/NULL incoming values; a non-empty
    incoming dict replaces the stored one wholesale. Key-union merge
    (``{"a": 1, "b": 2}``) is intentionally OUT OF SCOPE."""
    eid = uuid4()
    name = f"guard-batch-overwrite-{uuid4()}"

    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, eid, name, {"a": 1})])
    await backend.upsert_entities_batch(namespace_id, [_entity(namespace_id, eid, name, {"b": 2})])

    got = await backend.get_entity(eid, namespace_id=namespace_id)
    assert got is not None
    assert got.attributes == {"b": 2}
    assert "a" not in got.attributes

"""Integration test for `PgVectorBackend.upsert_entities_batch` is_new tracking.

Regression test for GitHub issue #719. Prior to the fix, the pgvector
backend returned ``is_new=True`` for every entity — even on the second
upsert of the same ``(namespace_id, name, entity_type)`` — because the
function hard-coded ``return [(entity, True) for entity in sorted_entities]``.

The Neo4j adapter's parametrized compliance test
(``tests/unit/storage/backends/test_protocol_compliance.py::test_upsert_entities_batch_new_vs_existing``)
runs against the ``graph_backend`` fixture only, so this gap on the
vector adapter went undetected. This test closes that gap by exercising
pgvector directly against a real PostgreSQL.

The fix uses ``RETURNING (xmax = 0) AS is_new`` to discriminate true
inserts from ``ON CONFLICT DO UPDATE`` paths.

Requires a running PostgreSQL instance from this repo's compose stack
(``make dev``); port ``5434`` per ``compose.yaml``. Set
``KHORA_DATABASE_URL`` to override.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Entity, MemoryNamespace
from khora.db.session import run_migrations
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
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
async def relational(_run_migrations_once):
    be = PostgreSQLBackend(database_url=DATABASE_URL)
    await be.connect()
    yield be
    await be.disconnect()


@pytest.fixture
async def vector_backend(_run_migrations_once):
    # Match the production embedding dimension; entity rows allow NULL embeddings,
    # so the actual vector dim does not affect this test.
    be = PgVectorBackend(database_url=DATABASE_URL, embedding_dimension=4)
    await be.connect()
    yield be
    await be.disconnect()


def _make_entity(namespace_id, *, name: str, entity_type: str = "PERSON") -> Entity:
    """Build a fresh Entity. Each call returns a new ``id``; the natural key
    ``(namespace_id, name, entity_type)`` is what drives the ON CONFLICT path."""
    return Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name=name,
        entity_type=entity_type,
        description="",
        attributes={},
        embedding=None,
        embedding_model="",
        mention_count=1,
        confidence=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@skip_no_pg
class TestPgVectorUpsertIsNew:
    """Regression coverage for issue #719 — pgvector ``is_new`` semantics."""

    async def test_upsert_entities_batch_new_vs_existing(
        self, relational: PostgreSQLBackend, vector_backend: PgVectorBackend
    ) -> None:
        ns = await relational.create_namespace(MemoryNamespace())

        # Seed 3 entities — all must report is_new=True.
        seed = [_make_entity(ns.id, name=f"S{i}") for i in range(3)]
        seeded = await vector_backend.upsert_entities_batch(ns.id, seed)
        assert all(is_new for _, is_new in seeded), f"expected all True on initial upsert, got {[f for _, f in seeded]}"

        # Mix 2 new + 3 colliding-by-(name, entity_type). Use fresh Entity
        # instances with new uuids — mirrors issue #719's repro where every
        # ingest pass mints fresh entity objects but the natural key collides.
        new = [_make_entity(ns.id, name=f"N{i}") for i in range(2)]
        collide = [_make_entity(ns.id, name=f"S{i}") for i in range(3)]
        results = await vector_backend.upsert_entities_batch(ns.id, new + collide)

        # The backend sorts internally by (namespace_id, name, entity_type),
        # so check the flag per-entity by name rather than positional order.
        flags = {e.name: is_new for e, is_new in results}
        assert flags == {
            "N0": True,
            "N1": True,
            "S0": False,
            "S1": False,
            "S2": False,
        }, f"is_new map wrong on second pass: {flags}"

        # Row count: 5 total (3 seeded + 2 net new). MERGE semantics preserved.
        assert await vector_backend.count_entities(ns.id) == 5

    async def test_upsert_entities_batch_empty(self, vector_backend: PgVectorBackend) -> None:
        assert await vector_backend.upsert_entities_batch(uuid4(), []) == []

    async def test_upsert_entities_batch_crosses_sub_batch_boundary(
        self, relational: PostgreSQLBackend, vector_backend: PgVectorBackend
    ) -> None:
        """``is_new`` must remain correct across multiple sub-batches.

        ``upsert_entities_batch`` chunks input by ``batch_size`` while holding
        the namespace advisory lock for the full transaction. The fix builds
        ``is_new_map`` incrementally across sub-batches; this test guards
        against bugs that would reset the map per sub-batch or otherwise
        lose track of seen rows across the chunk boundary.

        Uses a small ``batch_size=2`` so 5 entities span 3 sub-batches.
        """
        ns = await relational.create_namespace(MemoryNamespace())

        seed = [_make_entity(ns.id, name=f"X{i}") for i in range(5)]
        seeded = await vector_backend.upsert_entities_batch(ns.id, seed, batch_size=2)
        assert [is_new for _, is_new in seeded] == [True] * 5

        # All 5 collide across the chunk boundary on the second pass.
        collide = [_make_entity(ns.id, name=f"X{i}") for i in range(5)]
        results = await vector_backend.upsert_entities_batch(ns.id, collide, batch_size=2)
        flags = {e.name: is_new for e, is_new in results}
        assert flags == {f"X{i}": False for i in range(5)}, f"is_new across sub-batches wrong: {flags}"

        assert await vector_backend.count_entities(ns.id) == 5

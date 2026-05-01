"""Tests for :class:`SQLiteLanceGraphAdapter` (DYT-2729)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import lancedb  # noqa: F401

    _HAS_LANCEDB = True
except ImportError:
    _HAS_LANCEDB = False

from khora.core.models import Entity, Episode, Relationship

pytestmark = pytest.mark.skipif(not _HAS_LANCEDB, reason="lancedb not installed")

if _HAS_LANCEDB:
    from khora.db.session import run_migrations
    from khora.storage.backends.sqlite_lance.connection import (
        EmbeddedStorageHandle,
        EmbeddedStorageHandleConfig,
    )
    from khora.storage.backends.sqlite_lance.graph import SQLiteLanceGraphAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def handle(tmp_path: Path):
    """Connect a handle backed by the real Alembic-migrated schema.

    Unit tests here exercise graph semantics (CRUD, traversal, merge),
    not cross-table integrity; FKs are disabled so the tests don't have
    to seed ``memory_namespaces`` / ``entities`` parent rows for every
    relationship.  FK enforcement is exercised end-to-end in
    ``tests/integration/test_sqlite_lance_ingest.py``.
    """
    db_path = tmp_path / "graph.db"
    lance_path = tmp_path / "graph.lance"

    migration_result = await run_migrations(f"sqlite+aiosqlite:///{db_path}")
    assert migration_result.success, migration_result.error

    cfg = EmbeddedStorageHandleConfig(
        db_path=str(db_path),
        lance_path=str(lance_path),
        embedding_dimension=8,
        use_halfvec=False,
    )
    h = EmbeddedStorageHandle(cfg)
    await h.connect()
    await h.sqlite.execute("PRAGMA foreign_keys = OFF")
    await h.sqlite.commit()
    yield h
    await h.disconnect()


@pytest.fixture
async def adapter(handle):
    a = SQLiteLanceGraphAdapter(handle)
    await a.connect()
    return a


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    namespace_id: UUID,
    *,
    name: str = "alice",
    entity_type: str = "PERSON",
    attributes: dict | None = None,
) -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=namespace_id,
        name=name,
        entity_type=entity_type,
        description="",
        attributes=attributes or {},
        mention_count=1,
        confidence=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_relationship(
    namespace_id: UUID,
    src: UUID,
    tgt: UUID,
    *,
    rel_type: str = "RELATES_TO",
    description: str = "",
) -> Relationship:
    return Relationship(
        id=uuid4(),
        namespace_id=namespace_id,
        source_entity_id=src,
        target_entity_id=tgt,
        relationship_type=rel_type,
        description=description,
        confidence=1.0,
        weight=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Lifecycle + health
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_is_healthy(self, adapter: SQLiteLanceGraphAdapter):
        assert await adapter.is_healthy() is True

    async def test_disconnect_does_not_close_shared_handle(self, adapter: SQLiteLanceGraphAdapter):
        await adapter.disconnect()
        # Handle is shared — still healthy after the adapter disconnects.
        assert await adapter.is_healthy() is True


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------


class TestEntityCRUD:
    async def test_create_and_get(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        e = _make_entity(ns, name="Alice", attributes={"age": 30})
        await adapter.create_entity(e)

        fetched = await adapter.get_entity(e.id)
        assert fetched is not None
        assert fetched.id == e.id
        assert fetched.name == "Alice"
        assert fetched.attributes["age"] == 30

    async def test_get_entity_missing(self, adapter: SQLiteLanceGraphAdapter):
        assert await adapter.get_entity(uuid4()) is None

    async def test_get_entity_by_name_hit(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        e = _make_entity(ns, name="Bob", entity_type="PERSON")
        await adapter.create_entity(e)

        fetched = await adapter.get_entity_by_name(ns, "Bob", "PERSON")
        assert fetched is not None and fetched.id == e.id

    async def test_get_entity_by_name_miss(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        assert await adapter.get_entity_by_name(ns, "Nobody", "PERSON") is None

    async def test_update_entity(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        e = _make_entity(ns, name="Carol")
        await adapter.create_entity(e)

        e.description = "updated"
        e.mention_count = 42
        await adapter.update_entity(e)

        fetched = await adapter.get_entity(e.id)
        assert fetched is not None
        assert fetched.description == "updated"
        assert fetched.mention_count == 42

    async def test_delete_entity(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        e = _make_entity(ns, name="Dave")
        await adapter.create_entity(e)

        assert await adapter.delete_entity(e.id) is True
        assert await adapter.get_entity(e.id) is None
        assert await adapter.delete_entity(e.id) is False

    async def test_delete_entity_removes_relationships(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        r = _make_relationship(ns, a.id, b.id)
        await adapter.create_relationship(r)

        await adapter.delete_entity(a.id)
        assert await adapter.get_relationship(r.id) is None

    async def test_entity_exists(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        e = _make_entity(ns, name="Eve")
        await adapter.create_entity(e)

        assert await adapter.entity_exists(ns, "Eve", "PERSON") is True
        assert await adapter.entity_exists(ns, "Eve", "ORGANIZATION") is False
        assert await adapter.entity_exists(ns, "NotHere", "PERSON") is False

    async def test_list_entities(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        for i in range(5):
            await adapter.create_entity(_make_entity(ns, name=f"E{i}"))

        listed = await adapter.list_entities(ns, limit=10)
        assert len(listed) == 5

        filtered = await adapter.list_entities(ns, entity_type="PERSON", limit=10)
        assert len(filtered) == 5


# ---------------------------------------------------------------------------
# Batch upsert + entity-key gate concurrency
# ---------------------------------------------------------------------------


class TestUpsertEntitiesBatch:
    async def test_all_new(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        batch = [_make_entity(ns, name=f"N{i}") for i in range(10)]
        results = await adapter.upsert_entities_batch(ns, batch)

        assert len(results) == 10
        assert all(is_new for _, is_new in results)
        assert await adapter.count_entities(ns) == 10

    async def test_mixed_new_and_existing(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        # Seed 10 existing entities.
        existing = [_make_entity(ns, name=f"E{i}") for i in range(10)]
        await adapter.upsert_entities_batch(ns, existing)

        # Now upsert 10 fresh + 10 that collide on (name, type).
        fresh = [_make_entity(ns, name=f"F{i}") for i in range(10)]
        collide = [_make_entity(ns, name=f"E{i}") for i in range(10)]
        batch = fresh + collide

        results = await adapter.upsert_entities_batch(ns, batch)
        assert len(results) == 20

        # First 10 are new; last 10 re-use the existing rows.
        new_flags = [is_new for _, is_new in results]
        assert new_flags[:10] == [True] * 10
        assert new_flags[10:] == [False] * 10

        # Mention counts on the updated rows should have incremented.
        for ent, is_new in results[10:]:
            assert is_new is False
            assert ent.mention_count == 2  # seeded=1 + merged=1

        assert await adapter.count_entities(ns) == 20

    async def test_empty_batch(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        assert await adapter.upsert_entities_batch(ns, []) == []

    async def test_concurrent_overlapping_batches_are_serialized(self, adapter: SQLiteLanceGraphAdapter):
        """Five concurrent batches that all upsert the same 5 entity keys
        must produce exactly 5 entity rows (not 25), proving the gate
        serializes overlapping-key batches."""
        ns = uuid4()

        async def batch_task():
            entities = [_make_entity(ns, name=f"shared{i}") for i in range(5)]
            return await adapter.upsert_entities_batch(ns, entities)

        results = await asyncio.gather(*[batch_task() for _ in range(5)])

        # Each call returned 5 results.
        assert all(len(r) == 5 for r in results)

        # Only 5 rows total — entity key gate prevented duplicates.
        assert await adapter.count_entities(ns) == 5

        # Mention counts: first insert = 1, each subsequent merge adds 1.
        for name in [f"shared{i}" for i in range(5)]:
            ent = await adapter.get_entity_by_name(ns, name, "PERSON")
            assert ent is not None
            assert ent.mention_count == 5


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    async def test_create_and_get(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        r = _make_relationship(ns, a.id, b.id, rel_type="KNOWS")
        await adapter.create_relationship(r)

        fetched = await adapter.get_relationship(r.id)
        assert fetched is not None
        assert fetched.source_entity_id == a.id
        assert fetched.target_entity_id == b.id
        assert fetched.relationship_type == "KNOWS"

    async def test_delete_relationship(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        r = _make_relationship(ns, a.id, b.id)
        await adapter.create_relationship(r)

        assert await adapter.delete_relationship(r.id) is True
        assert await adapter.get_relationship(r.id) is None
        assert await adapter.delete_relationship(r.id) is False

    async def test_get_entity_relationships_directions(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        hub = _make_entity(ns, name="Hub")
        out1 = _make_entity(ns, name="Out1")
        out2 = _make_entity(ns, name="Out2")
        in1 = _make_entity(ns, name="In1")
        for e in (hub, out1, out2, in1):
            await adapter.create_entity(e)

        await adapter.create_relationship(_make_relationship(ns, hub.id, out1.id))
        await adapter.create_relationship(_make_relationship(ns, hub.id, out2.id))
        await adapter.create_relationship(_make_relationship(ns, in1.id, hub.id))

        out = await adapter.get_entity_relationships(hub.id, direction="outgoing")
        assert len(out) == 2
        assert all(r.source_entity_id == hub.id for r in out)

        incoming = await adapter.get_entity_relationships(hub.id, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].target_entity_id == hub.id

        both = await adapter.get_entity_relationships(hub.id, direction="both")
        assert len(both) == 3

    async def test_get_entity_relationships_type_filter(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        c = _make_entity(ns, name="C")
        for e in (a, b, c):
            await adapter.create_entity(e)
        await adapter.create_relationship(_make_relationship(ns, a.id, b.id, rel_type="KNOWS"))
        await adapter.create_relationship(_make_relationship(ns, a.id, c.id, rel_type="WORKS_WITH"))

        knows = await adapter.get_entity_relationships(a.id, direction="outgoing", relationship_types=["KNOWS"])
        assert len(knows) == 1
        assert knows[0].relationship_type == "KNOWS"

    async def test_list_relationships(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        for i in range(3):
            await adapter.create_relationship(_make_relationship(ns, a.id, b.id, description=f"r{i}"))

        listed = await adapter.list_relationships(ns, limit=10)
        assert len(listed) == 3

    async def test_create_relationships_batch(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)

        rels = [_make_relationship(ns, a.id, b.id) for _ in range(50)]
        created = await adapter.create_relationships_batch(rels)
        assert created == 50
        assert await adapter.count_relationships(ns) == 50

    async def test_relationship_type_is_sanitized(self, adapter: SQLiteLanceGraphAdapter):
        """User-supplied relationship types are UPPER_SNAKE-cased and
        stripped of non-identifier characters via sanitize_cypher_label,
        matching the shared graph-backend convention."""
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)

        r = _make_relationship(ns, a.id, b.id, rel_type="likes food; DROP TABLE")
        await adapter.create_relationship(r)

        fetched = await adapter.get_relationship(r.id)
        assert fetched is not None
        assert fetched.relationship_type == "LIKES_FOOD__DROP_TABLE"

    async def test_relationship_type_empty_falls_back_to_relates_to(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        # sanitize_cypher_label falls back to RELATES_TO only when the
        # sanitized result is empty (e.g. the input trims away to "").
        r = _make_relationship(ns, a.id, b.id, rel_type="   ")
        await adapter.create_relationship(r)

        fetched = await adapter.get_relationship(r.id)
        assert fetched is not None
        assert fetched.relationship_type == "RELATES_TO"


# ---------------------------------------------------------------------------
# Traversal — recursive CTEs
# ---------------------------------------------------------------------------


class TestTraversal:
    async def _chain(self, adapter: SQLiteLanceGraphAdapter, ns: UUID, length: int):
        """Create a → b → c → ... chain of ``length`` entities."""
        entities = [_make_entity(ns, name=f"n{i}") for i in range(length)]
        for e in entities:
            await adapter.create_entity(e)
        for i in range(length - 1):
            await adapter.create_relationship(_make_relationship(ns, entities[i].id, entities[i + 1].id))
        return entities

    async def test_find_paths_direct(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        entities = await self._chain(adapter, ns, 2)
        paths = await adapter.find_paths(ns, entities[0].id, entities[1].id, max_depth=1)
        assert len(paths) == 1
        assert len(paths[0]) == 1  # one hop

    async def test_find_paths_two_hops(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        entities = await self._chain(adapter, ns, 3)
        paths = await adapter.find_paths(ns, entities[0].id, entities[2].id, max_depth=3)
        # Exactly one path in a linear chain.
        assert len(paths) == 1
        assert len(paths[0]) == 2

    async def test_find_paths_three_hops(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        entities = await self._chain(adapter, ns, 4)
        paths = await adapter.find_paths(ns, entities[0].id, entities[3].id, max_depth=3)
        assert len(paths) == 1
        assert len(paths[0]) == 3
        # First edge starts at the source, last edge ends at the target.
        assert paths[0][0]["data"]["source_entity_id"] == str(entities[0].id)
        assert paths[0][-1]["data"]["target_entity_id"] == str(entities[3].id)

    async def test_find_paths_respects_max_depth(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        entities = await self._chain(adapter, ns, 5)
        # a -> b -> c -> d -> e, but we cap at 2.
        paths = await adapter.find_paths(ns, entities[0].id, entities[4].id, max_depth=2)
        assert paths == []

    async def test_find_paths_disconnected(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)
        paths = await adapter.find_paths(ns, a.id, b.id, max_depth=3)
        assert paths == []

    async def test_find_paths_rel_type_filter(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        c = _make_entity(ns, name="C")
        for e in (a, b, c):
            await adapter.create_entity(e)
        # Two different-typed edges forming two different candidate paths.
        await adapter.create_relationship(_make_relationship(ns, a.id, b.id, rel_type="KNOWS"))
        await adapter.create_relationship(_make_relationship(ns, b.id, c.id, rel_type="WORKS_WITH"))

        # Only KNOWS: can't reach c.
        paths = await adapter.find_paths(ns, a.id, c.id, max_depth=3, relationship_types=["KNOWS"])
        assert paths == []
        # Both allowed: one path.
        paths = await adapter.find_paths(ns, a.id, c.id, max_depth=3, relationship_types=["KNOWS", "WORKS_WITH"])
        assert len(paths) == 1

    async def test_get_neighborhood_depth_1(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        hub = _make_entity(ns, name="Hub")
        n1 = _make_entity(ns, name="N1")
        n2 = _make_entity(ns, name="N2")
        for e in (hub, n1, n2):
            await adapter.create_entity(e)
        await adapter.create_relationship(_make_relationship(ns, hub.id, n1.id))
        await adapter.create_relationship(_make_relationship(ns, n2.id, hub.id))

        nb = await adapter.get_neighborhood(hub.id, depth=1, limit=10)
        names = {e.name for e in nb["entities"]}
        assert names == {"N1", "N2"}
        assert len(nb["relationships"]) == 2

    async def test_get_neighborhood_depth_2(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        entities = await self._chain(adapter, ns, 4)
        # Chain: n0 -> n1 -> n2 -> n3. Depth-2 from n0 should include n1, n2.
        nb = await adapter.get_neighborhood(entities[0].id, depth=2, limit=10)
        names = {e.name for e in nb["entities"]}
        assert {"n1", "n2"}.issubset(names)
        assert "n3" not in names

    async def test_get_neighborhoods_batch_single_query(self, adapter: SQLiteLanceGraphAdapter):
        """``get_neighborhoods_batch`` must NOT fall through to the default
        N+1 implementation from ``GraphBackendBase``.  We count the SQL
        statements executed by the underlying aiosqlite connection — a
        single recursive CTE plus up to two follow-up resolves."""
        ns = uuid4()
        hub_a = _make_entity(ns, name="hub_a")
        hub_b = _make_entity(ns, name="hub_b")
        n1 = _make_entity(ns, name="n1")
        n2 = _make_entity(ns, name="n2")
        for e in (hub_a, hub_b, n1, n2):
            await adapter.create_entity(e)
        await adapter.create_relationship(_make_relationship(ns, hub_a.id, n1.id))
        await adapter.create_relationship(_make_relationship(ns, hub_b.id, n2.id))

        # Instrument the underlying aiosqlite.Connection.execute to count
        # statements that hit the CTE or the entity/rel resolvers.
        conn = adapter._handle.sqlite
        original_execute = conn.execute
        execute_calls: list[str] = []

        def _wrap(sql, *a, **k):
            execute_calls.append(str(sql))
            return original_execute(sql, *a, **k)

        conn.execute = _wrap  # type: ignore[method-assign]
        try:
            result = await adapter.get_neighborhoods_batch([hub_a.id, hub_b.id], depth=1, limit_per_entity=10)
        finally:
            conn.execute = original_execute  # type: ignore[method-assign]

        assert set(result.keys()) == {hub_a.id, hub_b.id}
        assert len(result[hub_a.id]["entities"]) == 1
        assert len(result[hub_b.id]["entities"]) == 1

        # Count the recursive CTEs (the walk). There must be EXACTLY ONE
        # regardless of how many seed entities we passed — that's the
        # whole point of the override.
        cte_calls = [c for c in execute_calls if "RECURSIVE walk" in c]
        assert len(cte_calls) == 1

    async def test_get_neighborhoods_batch_empty(self, adapter: SQLiteLanceGraphAdapter):
        assert await adapter.get_neighborhoods_batch([]) == {}

    async def test_find_paths_multigraph_back_and_forth(self, adapter: SQLiteLanceGraphAdapter):
        """A pair of nodes connected by two distinct directed edges
        (A→B via r1, B→A via r2) admits a legitimate 2-hop A→A path
        that traverses each edge exactly once.  Neo4j's ``MATCH [*1..N]``
        forbids reusing **edges**, not nodes — the pre-DYT-3548 CTE
        tracked visited *nodes*, which incorrectly blocked the return
        walk into A and silently dropped this path."""
        ns = uuid4()
        a = _make_entity(ns, name="mg_A")
        b = _make_entity(ns, name="mg_B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)

        r1 = _make_relationship(ns, a.id, b.id, rel_type="OUT")
        r2 = _make_relationship(ns, b.id, a.id, rel_type="BACK")
        await adapter.create_relationship(r1)
        await adapter.create_relationship(r2)

        paths = await adapter.find_paths(ns, a.id, a.id, max_depth=2)
        assert len(paths) == 1, f"expected the A→B→A round trip, got {len(paths)} paths"
        assert len(paths[0]) == 2
        edge_ids = [hop["data"]["id"] for hop in paths[0]]
        assert edge_ids == [str(r1.id), str(r2.id)]

    async def test_find_paths_parallel_edges_at_depth_two(self, adapter: SQLiteLanceGraphAdapter):
        """Two parallel A→B edges must yield the same number of paths
        at any depth — once both r1+r3 and r2+r3 reach C, the path
        count is independent of which parallel edge was taken first."""
        ns = uuid4()
        a = _make_entity(ns, name="pe_A")
        b = _make_entity(ns, name="pe_B")
        c = _make_entity(ns, name="pe_C")
        for e in (a, b, c):
            await adapter.create_entity(e)

        r1 = _make_relationship(ns, a.id, b.id, rel_type="ONE")
        r2 = _make_relationship(ns, a.id, b.id, rel_type="TWO")
        r3 = _make_relationship(ns, b.id, c.id, rel_type="THREE")
        for r in (r1, r2, r3):
            await adapter.create_relationship(r)

        paths = await adapter.find_paths(ns, a.id, c.id, max_depth=2)
        assert len(paths) == 2  # one per parallel A→B edge
        starting_edges = {p[0]["data"]["id"] for p in paths}
        assert starting_edges == {str(r1.id), str(r2.id)}
        assert all(p[-1]["data"]["id"] == str(r3.id) for p in paths)

    async def test_get_neighborhoods_batch_cyclic_walk_is_bounded(self, adapter: SQLiteLanceGraphAdapter):
        """Cycle A→B→A (2 parallel directed edges).  The pre-DYT-3548
        ``get_neighborhoods_batch`` CTE had NO visited set at all, so
        the recursive arms re-traversed the cycle ``2^depth`` times
        before the trailing ``DISTINCT`` collapsed the rows.  The fix
        tracks visited **edge ids** so each edge appears at most once
        per walk, bounding row count by ``num_edges`` rather than
        ``2^depth``.

        We compare the OLD-style CTE (no visited gate, kept inline as
        a test fixture) against the NEW-style (with the gate) on the
        same data.  The old structure must produce ``2^depth`` rows;
        the new must stay bounded.  This isolates the fix from
        unrelated downstream Python-side filtering.
        """
        ns = uuid4()
        a = _make_entity(ns, name="cyc_A")
        b = _make_entity(ns, name="cyc_B")
        await adapter.create_entity(a)
        await adapter.create_entity(b)

        # Two parallel directed edges forming a 2-cycle.
        edges = [
            _make_relationship(ns, a.id, b.id, rel_type="OUT"),
            _make_relationship(ns, b.id, a.id, rel_type="BACK"),
        ]
        for r in edges:
            await adapter.create_relationship(r)

        depth = 6

        # First: the public API returns a correct, bounded result.
        result = await adapter.get_neighborhoods_batch([a.id], depth=depth, limit_per_entity=1000)
        bucket = result[a.id]
        assert len(bucket["relationships"]) == 2
        assert {e.name for e in bucket["entities"]} == {"cyc_A", "cyc_B"}

        from khora.storage.backends.sqlite_lance._helpers import uuid_to_text

        seed = uuid_to_text(a.id)

        # OLD CTE (verbatim from the pre-fix code, no ``visited``).
        old_sql = """
            WITH RECURSIVE walk(seed, cur, depth, direction, edge_id) AS (
                SELECT r.source_entity_id, r.target_entity_id, 1, 'out', r.id
                FROM relationships r
                WHERE r.source_entity_id = ?
                UNION ALL
                SELECT r.target_entity_id, r.source_entity_id, 1, 'in', r.id
                FROM relationships r
                WHERE r.target_entity_id = ?
                UNION ALL
                SELECT walk.seed, r.target_entity_id, walk.depth + 1, 'out', r.id
                FROM walk
                JOIN relationships r ON r.source_entity_id = walk.cur
                WHERE walk.depth < ?
                UNION ALL
                SELECT walk.seed, r.source_entity_id, walk.depth + 1, 'in', r.id
                FROM walk
                JOIN relationships r ON r.target_entity_id = walk.cur
                WHERE walk.depth < ?
            )
            SELECT COUNT(*) AS c FROM walk
        """

        # NEW CTE (mirrors the post-fix production CTE).
        new_sql = """
            WITH RECURSIVE walk(seed, cur, depth, direction, edge_id, visited) AS (
                SELECT r.source_entity_id, r.target_entity_id, 1, 'out', r.id,
                       '|' || r.id || '|'
                FROM relationships r
                WHERE r.source_entity_id = ?
                UNION ALL
                SELECT r.target_entity_id, r.source_entity_id, 1, 'in', r.id,
                       '|' || r.id || '|'
                FROM relationships r
                WHERE r.target_entity_id = ?
                UNION ALL
                SELECT walk.seed, r.target_entity_id, walk.depth + 1, 'out', r.id,
                       walk.visited || r.id || '|'
                FROM walk
                JOIN relationships r ON r.source_entity_id = walk.cur
                WHERE walk.depth < ?
                  AND instr(walk.visited, '|' || r.id || '|') = 0
                UNION ALL
                SELECT walk.seed, r.source_entity_id, walk.depth + 1, 'in', r.id,
                       walk.visited || r.id || '|'
                FROM walk
                JOIN relationships r ON r.target_entity_id = walk.cur
                WHERE walk.depth < ?
                  AND instr(walk.visited, '|' || r.id || '|') = 0
            )
            SELECT COUNT(*) AS c FROM walk
        """

        async with adapter._handle.sqlite.execute(old_sql, [seed, seed, depth, depth]) as cur:
            old_row = await cur.fetchone()
        old_walk_rows = int(old_row["c"])

        async with adapter._handle.sqlite.execute(new_sql, [seed, seed, depth, depth]) as cur:
            new_row = await cur.fetchone()
        new_walk_rows = int(new_row["c"])

        # Old structure: cycle re-entered once per depth level, two
        # seed arms → ~2^(depth+1) rows.  Concretely, depth=6 yields
        # 126 rows; ascertain it's at least exponential-shaped (>= 50).
        assert old_walk_rows >= 50, (
            f"old (no-visited) CTE produced only {old_walk_rows} rows — test fixture is wrong, the cycle should explode"
        )
        # New structure: each edge at most once per walk.  With 2
        # edges and 2 seed arms, total walks are ≤ 4 (length-1) + ≤ 4
        # (length-2) = 8.
        assert new_walk_rows <= 16, (
            f"new (edge-visited) CTE emitted {new_walk_rows} rows — edge revisit is not properly gated"
        )
        # And the new bound must be a strict reduction from the old.
        assert new_walk_rows < old_walk_rows


# ---------------------------------------------------------------------------
# Attribute search
# ---------------------------------------------------------------------------


class TestAttributeSearch:
    async def test_search_by_attribute(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        alice = _make_entity(ns, name="Alice", attributes={"role": "eng"})
        bob = _make_entity(ns, name="Bob", attributes={"role": "pm"})
        await adapter.create_entity(alice)
        await adapter.create_entity(bob)

        results = await adapter.search_entities_by_attribute(ns, "role", "eng")
        assert len(results) == 1
        assert results[0].id == alice.id

    async def test_search_by_attribute_no_match(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        alice = _make_entity(ns, name="Alice", attributes={"role": "eng"})
        await adapter.create_entity(alice)

        results = await adapter.search_entities_by_attribute(ns, "role", "ceo")
        assert results == []


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class TestEpisodes:
    async def test_create_and_get(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        ep = Episode(
            id=uuid4(),
            namespace_id=ns,
            name="Kickoff",
            description="Project kickoff",
            occurred_at=datetime.now(UTC),
            duration_seconds=3600,
            entity_ids=[uuid4(), uuid4()],
        )
        await adapter.create_episode(ep)

        fetched = await adapter.get_episode(ep.id)
        assert fetched is not None
        assert fetched.name == "Kickoff"
        assert fetched.duration_seconds == 3600
        assert len(fetched.entity_ids) == 2

    async def test_list_episodes_time_range(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        now = datetime.now(UTC)

        past = Episode(id=uuid4(), namespace_id=ns, name="past", occurred_at=now - timedelta(days=5))
        mid = Episode(id=uuid4(), namespace_id=ns, name="mid", occurred_at=now - timedelta(days=2))
        future = Episode(id=uuid4(), namespace_id=ns, name="future", occurred_at=now + timedelta(days=1))
        for e in (past, mid, future):
            await adapter.create_episode(e)

        # Full listing
        all_eps = await adapter.list_episodes(ns, limit=10)
        assert len(all_eps) == 3

        # Windowed
        recent = await adapter.list_episodes(ns, start_time=now - timedelta(days=3), end_time=now, limit=10)
        names = {e.name for e in recent}
        assert names == {"mid"}


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


class TestCounts:
    async def test_count_entities_and_relationships(self, adapter: SQLiteLanceGraphAdapter):
        ns = uuid4()
        a = _make_entity(ns, name="A")
        b = _make_entity(ns, name="B")
        c = _make_entity(ns, name="C")
        for e in (a, b, c):
            await adapter.create_entity(e)
        await adapter.create_relationship(_make_relationship(ns, a.id, b.id))
        await adapter.create_relationship(_make_relationship(ns, b.id, c.id))

        assert await adapter.count_entities(ns) == 3
        assert await adapter.count_relationships(ns) == 2

    async def test_counts_namespace_isolation(self, adapter: SQLiteLanceGraphAdapter):
        ns_a, ns_b = uuid4(), uuid4()
        await adapter.create_entity(_make_entity(ns_a, name="a1"))
        await adapter.create_entity(_make_entity(ns_b, name="b1"))

        assert await adapter.count_entities(ns_a) == 1
        assert await adapter.count_entities(ns_b) == 1

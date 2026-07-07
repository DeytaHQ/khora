"""Graph traversal integration tests for sqlite_lance.

Builds a 3-hop entity graph directly via the graph adapter (bypasses LLM
extraction) and exercises ``find_paths`` / ``get_neighborhood`` /
``get_neighborhoods_batch`` under realistic multi-hop conditions.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

try:
    import aiosqlite  # noqa: F401
    import lancedb  # noqa: F401

    _HAS_EMBEDDED = True
except ImportError:
    _HAS_EMBEDDED = False

from khora.core.models import Entity, MemoryNamespace, Relationship
from tests.integration._sqlite_lance_fixtures import build_sqlite_lance_coordinator

pytestmark = [
    pytest.mark.embedded,
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_EMBEDDED, reason="aiosqlite/lancedb not installed"),
]


async def _seed_chain(coord, namespace_id, names):
    """Insert N entities connected in a directed chain A -> B -> C -> ... .

    Returns the list of persisted entities in chain order.
    """
    entities = [Entity(namespace_id=namespace_id, name=name, entity_type="PERSON") for name in names]
    await coord.upsert_entities_batch(namespace_id, entities)

    relationships = [
        Relationship(
            namespace_id=namespace_id,
            source_entity_id=entities[i].id,
            target_entity_id=entities[i + 1].id,
            relationship_type="KNOWS",
        )
        for i in range(len(entities) - 1)
    ]
    await coord.create_relationships_batch(relationships)
    return entities


class TestSQLiteLanceTraversal:
    """Recursive-CTE graph traversal through SQLiteLanceGraphAdapter."""

    async def test_get_neighborhood_depth_1(self, tmp_path: Path) -> None:
        """Depth-1 neighborhood of B on A→B→C yields both neighbors (both directions)."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            a, b, c = await _seed_chain(coord, ns.id, ["alice", "bob", "carol"])

            hood = await coord.graph.get_neighborhood(b.id, namespace_id=ns.id, depth=1, limit=50)  # type: ignore[union-attr]
            ids = {e.id for e in hood["entities"]}
            assert a.id in ids
            assert c.id in ids
            # Both adjacent edges show up.
            assert len(hood["relationships"]) == 2
        finally:
            await coord.disconnect()

    async def test_get_neighborhood_depth_2_reaches_further(self, tmp_path: Path) -> None:
        """Depth-2 on a 4-node chain surfaces the depth-2 neighbor."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            a, b, c, d = await _seed_chain(coord, ns.id, ["a", "b", "c", "d"])

            hood_1 = await coord.graph.get_neighborhood(a.id, namespace_id=ns.id, depth=1, limit=50)  # type: ignore[union-attr]
            hood_2 = await coord.graph.get_neighborhood(a.id, namespace_id=ns.id, depth=2, limit=50)  # type: ignore[union-attr]

            ids_1 = {e.id for e in hood_1["entities"]}
            ids_2 = {e.id for e in hood_2["entities"]}

            # Depth 1 from A reaches B.  Depth 2 also reaches C.
            assert b.id in ids_1
            assert c.id not in ids_1
            assert b.id in ids_2
            assert c.id in ids_2
            # D (depth 3) must not appear in depth-2 results.
            assert d.id not in ids_2
        finally:
            await coord.disconnect()

    async def test_get_neighborhood_depth_3_full_chain(self, tmp_path: Path) -> None:
        """Depth-3 expansion from the chain head reaches all downstream nodes."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            a, b, c, d = await _seed_chain(coord, ns.id, ["a", "b", "c", "d"])

            hood = await coord.graph.get_neighborhood(a.id, namespace_id=ns.id, depth=3, limit=50)  # type: ignore[union-attr]
            ids = {e.id for e in hood["entities"]}
            assert {b.id, c.id, d.id}.issubset(ids)
            # All 3 chain edges are reachable in the subgraph.
            assert len(hood["relationships"]) == 3
        finally:
            await coord.disconnect()

    async def test_find_paths_connected_pair(self, tmp_path: Path) -> None:
        """find_paths returns at least one path between a connected pair."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            a, b, c, d = await _seed_chain(coord, ns.id, ["a", "b", "c", "d"])

            paths = await coord.graph.find_paths(
                a.id,
                d.id,
                # type: ignore[union-attr]
                namespace_id=ns.id,
                max_depth=4,
            )
            assert paths, "expected at least one A->D path on a directed chain"
            # Chain A->B->C->D has exactly 3 edges.
            assert any(len(path) == 3 for path in paths)
        finally:
            await coord.disconnect()

    async def test_find_paths_disconnected_returns_empty(self, tmp_path: Path) -> None:
        """find_paths yields [] when no directed path exists between endpoints."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            # Two disjoint chains: A->B and C->D.  No A->D path exists.
            a, b = await _seed_chain(coord, ns.id, ["a", "b"])
            c, d = await _seed_chain(coord, ns.id, ["c", "d"])

            paths = await coord.graph.find_paths(
                a.id,
                d.id,
                # type: ignore[union-attr]
                namespace_id=ns.id,
                max_depth=5,
            )
            assert paths == []
        finally:
            await coord.disconnect()

    async def test_get_neighborhoods_batch_one_shot(self, tmp_path: Path) -> None:
        """Batched neighborhood expansion returns a result per seed entity."""
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            a, b, c, d = await _seed_chain(coord, ns.id, ["a", "b", "c", "d"])

            result = await coord.graph.get_neighborhoods_batch(  # type: ignore[union-attr]
                [a.id, d.id],
                namespace_id=ns.id,
                depth=1,
                limit_per_entity=10,
            )
            assert set(result.keys()) == {a.id, d.id}

            a_hood = result[a.id]
            d_hood = result[d.id]
            # A has only an outbound edge to B at depth 1.
            assert b.id in {e.id for e in a_hood["entities"]}
            # D has only an inbound edge from C at depth 1 (directed chain).
            assert c.id in {e.id for e in d_hood["entities"]}
        finally:
            await coord.disconnect()

    async def test_list_entities_filters_by_source_chunk_ids(self, tmp_path: Path) -> None:
        """``list_entities(source_chunk_ids=...)`` filters by chunk provenance (#1448).

        Seeds two entities — A sourced from chunks c1/c2, B from c3 — then pins
        the four contract cases: no filter returns both; a filter for one of A's
        chunks returns only A; an unknown chunk returns nothing; and an empty
        list matches nothing (any-overlap semantics).
        """
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            c1, c2, c3, c4 = uuid4(), uuid4(), uuid4(), uuid4()
            entity_a = Entity(namespace_id=ns.id, name="A", entity_type="PERSON", source_chunk_ids=[c1, c2])
            entity_b = Entity(namespace_id=ns.id, name="B", entity_type="PERSON", source_chunk_ids=[c3])
            await coord.upsert_entities_batch(ns.id, [entity_a, entity_b])

            # 1. No filter → both entities.
            all_names = {e.name for e in await coord.list_entities(ns.id)}
            assert all_names == {"A", "B"}

            # 2. One of A's chunks → exactly A.
            only_a = await coord.list_entities(ns.id, source_chunk_ids=[c1])
            assert {e.name for e in only_a} == {"A"}

            # 3. Unknown chunk id → nothing.
            assert await coord.list_entities(ns.id, source_chunk_ids=[c4]) == []

            # 4. Empty list → matches nothing.
            assert await coord.list_entities(ns.id, source_chunk_ids=[]) == []
        finally:
            await coord.disconnect()

    async def test_list_relationships_filters_by_between_entity_ids(self, tmp_path: Path) -> None:
        """``list_relationships(between_entity_ids=...)`` filters by endpoint membership (#1451).

        Seeds A→B and B→C, then pins the four contract cases: no filter returns
        both edges; ``[A, B]`` returns exactly A→B (B→C excluded — C outside the
        set); ``[A]`` returns [] (no self-loops seeded); and an empty list
        returns [] (BOTH-endpoints-in-set semantics).
        """
        coord = await build_sqlite_lance_coordinator(tmp_path)
        try:
            ns = await coord.create_namespace(MemoryNamespace())
            a, b, c = await _seed_chain(coord, ns.id, ["a", "b", "c"])

            def _edges(rels: list[Relationship]) -> set[tuple[UUID, UUID]]:
                return {(r.source_entity_id, r.target_entity_id) for r in rels}

            # 1. No filter → both edges.
            assert _edges(await coord.list_relationships(ns.id)) == {(a.id, b.id), (b.id, c.id)}

            # 2. [A, B] → exactly A→B (B→C excluded — C outside the set).
            filtered = await coord.list_relationships(ns.id, between_entity_ids=[a.id, b.id])
            assert _edges(filtered) == {(a.id, b.id)}

            # 3. [A] → nothing (no self-loops seeded).
            assert await coord.list_relationships(ns.id, between_entity_ids=[a.id]) == []

            # 4. Empty list → nothing.
            assert await coord.list_relationships(ns.id, between_entity_ids=[]) == []
        finally:
            await coord.disconnect()

"""Regression tests for VectorCypher.find_related_entities scoring (Issue #581).

Bug: on graph-only backends (sqlite_lance, surrealdb) the engine returned
``score=1.0`` for every neighborhood entity regardless of depth, instead
of the ``1 / (1 + distance)`` decay the Neo4j (dual-nodes) path uses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models import Entity, Relationship
from khora.engines.vectorcypher.engine import _bfs_distances_from


def _make_entity(name: str, ns_id) -> Entity:
    return Entity(id=uuid4(), namespace_id=ns_id, name=name, entity_type="MODULE")


@pytest.mark.unit
def test_bfs_distances_chain() -> None:
    """A→B→C: B is depth 1, C is depth 2 from A."""
    ns = uuid4()
    a, b, c = _make_entity("A", ns), _make_entity("B", ns), _make_entity("C", ns)
    rels = [
        Relationship(namespace_id=ns, source_entity_id=a.id, target_entity_id=b.id),
        Relationship(namespace_id=ns, source_entity_id=b.id, target_entity_id=c.id),
    ]
    distances = _bfs_distances_from(a.id, rels)
    assert distances[a.id] == 0
    assert distances[b.id] == 1
    assert distances[c.id] == 2


@pytest.mark.unit
def test_bfs_distances_handles_undirected_edges() -> None:
    """If C only appears as an inbound edge from B, BFS still finds it."""
    ns = uuid4()
    a, b, c = _make_entity("A", ns), _make_entity("B", ns), _make_entity("C", ns)
    rels = [
        # A -> B (outbound from A)
        Relationship(namespace_id=ns, source_entity_id=a.id, target_entity_id=b.id),
        # C -> B (inbound to B from C — must still place C at depth 2 from A)
        Relationship(namespace_id=ns, source_entity_id=c.id, target_entity_id=b.id),
    ]
    distances = _bfs_distances_from(a.id, rels)
    assert distances[b.id] == 1
    assert distances[c.id] == 2


@pytest.mark.unit
def test_bfs_distances_takes_shortest_path() -> None:
    """When multiple paths exist, BFS uses the shortest."""
    ns = uuid4()
    a, b, c = _make_entity("A", ns), _make_entity("B", ns), _make_entity("C", ns)
    rels = [
        Relationship(namespace_id=ns, source_entity_id=a.id, target_entity_id=b.id),
        Relationship(namespace_id=ns, source_entity_id=b.id, target_entity_id=c.id),
        # Long detour: A -> C via B -> A -> C is not shorter than direct A -> B -> C.
        # Add a direct A -> C — now C should drop to depth 1.
        Relationship(namespace_id=ns, source_entity_id=a.id, target_entity_id=c.id),
    ]
    distances = _bfs_distances_from(a.id, rels)
    assert distances[c.id] == 1


@pytest.mark.unit
def test_bfs_distances_empty_relationships() -> None:
    assert _bfs_distances_from(uuid4(), []) == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_find_related_entities_scores_decay_by_depth_on_graph_only_backend() -> None:
    """End-to-end: engine asks graph backend for neighborhood, BFS recovers
    depth, scores follow 1/(1+distance).
    """
    from khora.engines.vectorcypher.engine import VectorCypherEngine

    ns = uuid4()
    a = _make_entity("A", ns)
    b = _make_entity("B", ns)
    c = _make_entity("C", ns)
    rels = [
        Relationship(namespace_id=ns, source_entity_id=a.id, target_entity_id=b.id),
        Relationship(namespace_id=ns, source_entity_id=b.id, target_entity_id=c.id),
    ]

    graph = MagicMock()
    graph.get_neighborhood = AsyncMock(
        return_value={
            "entities": [a, b, c],
            "relationships": rels,
        }
    )
    storage = MagicMock()
    storage.graph = graph

    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine._get_dual_nodes = lambda: None  # type: ignore[method-assign]
    engine._get_storage = lambda: storage  # type: ignore[method-assign]

    results = await engine.find_related_entities(a.id, ns, max_depth=2, limit=10)

    # Seed must not be in results, and scores must decay with depth.
    scores_by_name = {entity.name: score for entity, score in results}
    assert "A" not in scores_by_name
    assert scores_by_name["B"] == pytest.approx(0.5)  # 1 / (1 + 1)
    assert scores_by_name["C"] == pytest.approx(1.0 / 3)  # 1 / (1 + 2)
    # Results sorted by score descending
    assert [name for name, _ in [(e.name, s) for e, s in results]] == ["B", "C"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_find_related_entities_no_graph_backend_returns_empty() -> None:
    """When the coordinator has no graph backend at all (e.g. chronicle on PG
    without Neo4j), find_related_entities must short-circuit to []."""
    from khora.engines.vectorcypher.engine import VectorCypherEngine

    storage = MagicMock()
    storage.graph = None
    engine = VectorCypherEngine.__new__(VectorCypherEngine)
    engine._get_dual_nodes = lambda: None  # type: ignore[method-assign]
    engine._get_storage = lambda: storage  # type: ignore[method-assign]

    results = await engine.find_related_entities(uuid4(), uuid4(), max_depth=2, limit=10)
    assert results == []

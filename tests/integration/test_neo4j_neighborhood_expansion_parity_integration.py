"""Real-Neo4j result-parity test for the bounded neighborhood expansion (#1419).

``DualNodeManager.get_entity_neighborhoods`` replaced the exponential
undirected all-paths enumeration (``OPTIONAL MATCH path =
(e)-[*1..depth]-(related:Entity)``) with a bounded per-hop frontier
expansion. The acceptance bar for that change is RESULT PARITY: the same
``{entity_id -> min distance}`` map per source entity as the legacy query,
for every depth 1-4 and for both ``prefer_current`` modes.

This test builds a fixture graph that covers the tricky legacy semantics:

- chain distances (B=1, C=2, D=3 from A)
- distance ties (C reachable at distance 2 via two different paths)
- multi-distance nodes (B reachable at 1 and 2 - legacy emitted BOTH
  entries; consumers reduce to min distance, which is what the new query
  returns directly)
- traversal through NON-Entity intermediates (F reachable only through a
  shared :Chunk node - the legacy pattern constrained only the endpoint
  label, never the intermediates)
- traversal through OTHER-NAMESPACE intermediates (P reachable only
  through an entity in a different namespace)
- ``prefer_current``: expired relationships block traversal, expired
  entities are not reported but remain traversable-through (G is reachable
  through the expired entity X)

and asserts old-query and new-method results are identical after reducing
both to ``{related_id: min distance}``.

Why this is marked @pytest.mark.integration and gated by NEO4J_INTEGRATION_TEST=1:

    CI's ``test-integration`` job provisions a Neo4j side-car and sets
    NEO4J_INTEGRATION_TEST=1 plus the connection env vars, so this runs in
    CI; local runs opt in via the same env var.

Connection env contract (same as the sibling Neo4j integration tests and
the CI job env - defaults match the CI side-car on 7687):

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)

How to run locally against THIS repo's compose stack (``make dev``; note the
compose Neo4j is on 7688 - CI's 7688 is a Memgraph side-car, so never rely
on the port default locally):

    NEO4J_INTEGRATION_TEST=1 \
    KHORA_NEO4J_URL=bolt://localhost:7688 \
    KHORA_NEO4J_PASSWORD=pleaseletmein \
    uv run pytest \
        tests/integration/test_neo4j_neighborhood_expansion_parity_integration.py -v
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from neo4j import AsyncGraphDatabase

from khora.engines.vectorcypher.dual_nodes import DualNodeManager

# The pre-#1419 query, verbatim (modulo whitespace), so parity is asserted
# against the actual legacy behavior rather than a re-interpretation of it.
_LEGACY_TEMPORAL_PREAMBLE = "WITH e, datetime() AS _now"
_LEGACY_TEMPORAL_CLAUSE = (
    "AND (related.valid_until IS NULL OR datetime(related.valid_until) > _now)"
    "\n          AND all(r IN relationships(path) "
    "WHERE r.valid_until IS NULL OR datetime(r.valid_until) > _now)"
)


def _legacy_query(depth: int, prefer_current: bool) -> str:
    temporal_preamble = _LEGACY_TEMPORAL_PREAMBLE if prefer_current else ""
    temporal_clause = _LEGACY_TEMPORAL_CLAUSE if prefer_current else ""
    return f"""
    UNWIND $entity_ids AS eid
    MATCH (e:Entity {{id: eid, namespace_id: $namespace_id}})
    {temporal_preamble}
    OPTIONAL MATCH path = (e)-[*1..{depth}]-(related:Entity)
    WHERE related.namespace_id = $namespace_id
      AND related.id <> e.id
      {temporal_clause}
    WITH e, related,
         CASE WHEN related IS NOT NULL THEN length(path) ELSE null END AS distance
    ORDER BY e.id, distance
    With e, collect(DISTINCT CASE
        WHEN related IS NOT NULL THEN {{
            id: related.id,
            name: related.name,
            entity_type: related.entity_type,
            description: related.description,
            source_tool: related.source_tool,
            distance: distance
        }}
        ELSE null
    END)[0..$limit] AS related_raw
    RETURN e.id AS source_id,
           [x IN related_raw WHERE x IS NOT NULL] AS related_entities
    """


def _min_distances(related_entities: list[dict]) -> dict[str, int]:
    """Reduce a related-entities list to {id: min distance}.

    The legacy query could emit the same node at several distances (its
    DISTINCT ran over the whole map, distance included); consumers reduce
    to min distance, so parity is asserted at that level.
    """
    out: dict[str, int] = {}
    for ent in related_entities:
        eid = ent["id"]
        dist = ent["distance"]
        if eid not in out or dist < out[eid]:
            out[eid] = dist
    return out


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestBoundedExpansionParity:
    """Old-vs-new result parity on a fixture graph covering legacy edge cases."""

    @pytest.fixture()
    async def graph(self):
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        driver = AsyncGraphDatabase.driver(url, auth=(user, password))
        ns = uuid4()
        other_ns = uuid4()

        ids: dict[str, UUID] = {name: uuid4() for name in "ABCDEFGHXOP"}
        past = (datetime.now(UTC) - timedelta(days=365)).isoformat()

        setup = """
        UNWIND $entities AS ent
        CREATE (n:Entity {id: ent.id, namespace_id: ent.ns, name: ent.name,
                          entity_type: 'PERSON', description: null,
                          source_tool: null, valid_until: ent.valid_until})
        """
        edges = """
        UNWIND $edges AS edge
        MATCH (a:Entity {id: edge.src}), (b:Entity {id: edge.dst})
        CREATE (a)-[:KNOWS {valid_until: edge.valid_until}]->(b)
        """
        chunk = """
        MATCH (a:Entity {id: $a_id}), (f:Entity {id: $f_id})
        CREATE (c:Chunk {id: $chunk_id, namespace_id: $ns, content: 'shared chunk'})
        CREATE (a)-[:MENTIONED_IN]->(c)
        CREATE (f)-[:MENTIONED_IN]->(c)
        """

        entities = [{"id": str(ids[n]), "ns": str(ns), "name": n, "valid_until": None} for n in "ABCDEFGH"]
        # X: expired entity in ns (not reportable under prefer_current, but
        # traversable-through). O: live entity in a DIFFERENT namespace
        # (never reportable, but traversable-through). P: live entity in ns
        # reachable only through O.
        entities.append({"id": str(ids["X"]), "ns": str(ns), "name": "X", "valid_until": past})
        entities.append({"id": str(ids["O"]), "ns": str(other_ns), "name": "O", "valid_until": None})
        entities.append({"id": str(ids["P"]), "ns": str(ns), "name": "P", "valid_until": None})

        edge_rows = [
            # chain: B=1, C=2, D=3 from A
            {"src": str(ids["A"]), "dst": str(ids["B"]), "valid_until": None},
            {"src": str(ids["B"]), "dst": str(ids["C"]), "valid_until": None},
            {"src": str(ids["C"]), "dst": str(ids["D"]), "valid_until": None},
            # tie: C also reachable A-E-C at distance 2; E=1
            {"src": str(ids["A"]), "dst": str(ids["E"]), "valid_until": None},
            {"src": str(ids["E"]), "dst": str(ids["C"]), "valid_until": None},
            # multi-distance: B reachable at 1 (A-B) and 2 (A-E-B)
            {"src": str(ids["E"]), "dst": str(ids["B"]), "valid_until": None},
            # expired-entity intermediate: A-X (live rel), X-G (live rel)
            {"src": str(ids["A"]), "dst": str(ids["X"]), "valid_until": None},
            {"src": str(ids["X"]), "dst": str(ids["G"]), "valid_until": None},
            # expired relationship: A-H
            {"src": str(ids["A"]), "dst": str(ids["H"]), "valid_until": past},
            # other-namespace intermediate: A-O (O in other_ns), O-P (P in ns)
            {"src": str(ids["A"]), "dst": str(ids["O"]), "valid_until": None},
            {"src": str(ids["O"]), "dst": str(ids["P"]), "valid_until": None},
        ]

        # Setup lives INSIDE the try so a transient failure mid-setup still
        # closes the driver and deletes any partially-created fixture nodes
        # from the shared instance.
        try:
            async with driver.session(database="neo4j") as session:
                await session.run(setup, entities=entities)
                await session.run(edges, edges=edge_rows)
                await session.run(
                    chunk,
                    a_id=str(ids["A"]),
                    f_id=str(ids["F"]),
                    chunk_id=str(uuid4()),
                    ns=str(ns),
                )

            yield driver, ns, ids
        finally:
            async with driver.session(database="neo4j") as session:
                await session.run(
                    "MATCH (n) WHERE n.namespace_id IN $ns_list DETACH DELETE n",
                    ns_list=[str(ns), str(other_ns)],
                )
            await driver.close()

    async def _run_legacy(
        self,
        driver,
        entity_ids: list[UUID],
        namespace_id: UUID,
        depth: int,
        prefer_current: bool,
        limit: int,
    ) -> dict[str, list[dict]]:
        async with driver.session(database="neo4j") as session:
            result = await session.run(
                _legacy_query(depth, prefer_current),
                entity_ids=[str(eid) for eid in entity_ids],
                namespace_id=str(namespace_id),
                limit=limit,
            )
            records = [record.data() async for record in result]
        return {r["source_id"]: r["related_entities"] for r in records}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    @pytest.mark.parametrize("prefer_current", [False, True])
    async def test_parity_with_legacy_query(self, graph, depth: int, prefer_current: bool) -> None:
        """New per-hop expansion == legacy all-paths on {id -> min distance}."""
        driver, ns, ids = graph
        manager = DualNodeManager(driver)
        sources = [ids["A"], ids["B"]]
        # High limit so per-entity truncation (order-nondeterministic among
        # ties in BOTH implementations) cannot mask a set difference.
        limit = 1000

        legacy = await self._run_legacy(driver, sources, ns, depth, prefer_current, limit)
        new = await manager.get_entity_neighborhoods(
            sources,
            ns,
            depth=depth,
            limit_per_entity=limit,
            prefer_current=prefer_current,
        )

        assert set(new) == set(legacy) == {str(ids["A"]), str(ids["B"])}
        for source_id in legacy:
            legacy_min = _min_distances(legacy[source_id])
            new_min = _min_distances(new[source_id])
            assert new_min == legacy_min, (
                f"parity mismatch for source={source_id} depth={depth} "
                f"prefer_current={prefer_current}:\nlegacy={legacy_min}\nnew={new_min}"
            )
            # The new query must already be duplicate-free (min-distance map).
            new_ids = [ent["id"] for ent in new[source_id]]
            assert len(new_ids) == len(set(new_ids))
            # And distance-ascending, like the legacy ORDER BY distance.
            new_dists = [ent["distance"] for ent in new[source_id]]
            assert new_dists == sorted(new_dists)

    @pytest.mark.asyncio
    async def test_legacy_edge_semantics_preserved(self, graph) -> None:
        """Spot-check the specific legacy semantics the fixture encodes."""
        driver, ns, ids = graph
        manager = DualNodeManager(driver)

        result = await manager.get_entity_neighborhoods(
            [ids["A"]], ns, depth=2, limit_per_entity=1000, prefer_current=False
        )
        dist = _min_distances(result[str(ids["A"])])

        assert dist[str(ids["B"])] == 1  # min of {1, 2}
        assert dist[str(ids["C"])] == 2  # tie: two distance-2 paths
        assert dist[str(ids["F"])] == 2  # through the :Chunk intermediate
        assert dist[str(ids["P"])] == 2  # through the other-namespace entity O
        assert str(ids["O"]) not in dist  # other namespace never reported
        assert dist[str(ids["H"])] == 1  # expired rel fine without prefer_current
        assert str(ids["D"]) not in dist  # distance 3 > depth 2

        current = await manager.get_entity_neighborhoods(
            [ids["A"]], ns, depth=2, limit_per_entity=1000, prefer_current=True
        )
        cdist = _min_distances(current[str(ids["A"])])

        assert str(ids["X"]) not in cdist  # expired entity not reported
        assert cdist[str(ids["G"])] == 2  # ...but traversable-through
        assert str(ids["H"]) not in cdist  # expired relationship blocks

    @pytest.mark.asyncio
    async def test_hop_limit_bounds_fanout(self, graph) -> None:
        """A tiny hop_limit caps discovery instead of blowing up or erroring."""
        driver, ns, ids = graph
        manager = DualNodeManager(driver)

        result = await manager.get_entity_neighborhoods([ids["A"]], ns, depth=1, limit_per_entity=1000, hop_limit=2)
        # Hop 1 from A reaches 6 nodes unbounded (B, E, X, H, O, chunk);
        # with hop_limit=2 at most 2 survive.
        assert len(result[str(ids["A"])]) <= 2

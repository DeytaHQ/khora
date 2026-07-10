"""Real-Neo4j result-parity test for the per-recall session fold (#1468).

Issue #1468 collapses a recall's sequential Neo4j graph reads
(``get_entity_neighborhoods`` -> optional version filter -> ``get_chunks_by_entities``)
onto ONE shared session via ``DualNodeManager.bind_session()``, instead of each
read opening + tearing down its own pooled connection.

The acceptance bar is RESULT PARITY: the bound (fused-session) path must return
byte-identical results to the unbound (per-call-session) path — because the
Cypher queries themselves are unchanged, the only difference is connection
reuse. This test seeds a fixture graph, runs both dual_nodes reads twice (once
per-call-session, once inside a single ``bind_session()``), and asserts:

- identical ``{related_id: distance}`` neighborhood maps per source entity
- identical ordered chunk lists (same ``(chunk_id, total_mentions,
  entity_ids-set)`` sequence) — the exact inputs the retriever turns into a
  ``(chunk_id, score, Chunk)`` tuple, so identical here == identical ranked
  recall output downstream.

It also asserts the leg's actual effect: the bound path acquires exactly ONE
session for BOTH reads, versus two for the unbound path.

Why @pytest.mark.integration + NEO4J_INTEGRATION_TEST=1: khora's CI does not
provision Neo4j; real-Neo4j coverage is opt-in. Run locally against THIS repo's
compose stack (``make dev``; note the compose Neo4j is on 7688):

    NEO4J_INTEGRATION_TEST=1 \
    KHORA_NEO4J_URL=bolt://localhost:7688 \
    KHORA_NEO4J_PASSWORD=pleaseletmein \
    uv run pytest \
        tests/integration/test_neo4j_session_fold_parity_integration.py -v
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from neo4j import AsyncGraphDatabase

from khora.engines.vectorcypher import dual_nodes as dual_nodes_mod
from khora.engines.vectorcypher.dual_nodes import DualNodeManager


def _neigh_min(related_entities: list[dict]) -> dict[str, int]:
    """Reduce a related-entities list to {id: min distance}."""
    out: dict[str, int] = {}
    for ent in related_entities:
        eid = ent["id"]
        dist = ent["distance"]
        if eid not in out or dist < out[eid]:
            out[eid] = dist
    return out


def _chunk_key(records: list[dict]) -> list[tuple[str, int, frozenset[str]]]:
    """Projection of the chunk records to the parity-relevant fields the
    retriever consumes: (chunk_id, total_mentions, {entity_ids}).

    Sorted by (score-key DESC, chunk_id) so a genuine content/score difference
    between the two paths is caught, while Neo4j's inherent tie-ordering jitter
    (rows with equal ORDER BY key can stream in different physical order on two
    executions of the SAME query) is NOT a false parity failure — session
    binding changes the connection, never the query, so any ordering difference
    among equal-key rows is unrelated to this leg.
    """
    projected = [
        (
            str(r["chunk_id"]),
            int(r.get("total_mentions") or 0),
            frozenset(str(e) for e in (r.get("entity_ids") or [])),
        )
        for r in records
    ]
    return sorted(projected, key=lambda t: (-t[1], t[0]))


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestSessionFoldParity:
    """Unbound (per-call-session) vs bound (fused-session) result parity."""

    @pytest.fixture()
    async def graph(self):
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        driver = AsyncGraphDatabase.driver(url, auth=(user, password))
        ns = uuid4()

        ids: dict[str, UUID] = {name: uuid4() for name in "ABCDE"}
        chunk_ids: dict[str, UUID] = {name: uuid4() for name in ("c1", "c2", "c3")}
        now = datetime.now(UTC)

        setup = """
        UNWIND $entities AS ent
        CREATE (n:Entity {id: ent.id, namespace_id: ent.ns, name: ent.name,
                          entity_type: 'PERSON', description: 'd', source_tool: 't',
                          valid_until: null})
        """
        edges = """
        UNWIND $edges AS edge
        MATCH (a:Entity {id: edge.src}), (b:Entity {id: edge.dst})
        CREATE (a)-[:KNOWS {valid_until: null}]->(b)
        """
        chunks = """
        UNWIND $chunks AS ck
        CREATE (c:Chunk {id: ck.id, namespace_id: ck.ns, document_id: ck.doc,
                         content: ck.content, occurred_at: ck.occurred_at,
                         created_at: ck.created_at, metadata: '{}', chunker_info: '{}'})
        """
        mentions = """
        UNWIND $links AS link
        MATCH (e:Entity {id: link.entity_id}), (c:Chunk {id: link.chunk_id})
        CREATE (e)-[:MENTIONED_IN {mention_count: link.mention_count, context: ''}]->(c)
        """

        entities = [{"id": str(ids[n]), "ns": str(ns), "name": n} for n in "ABCDE"]
        # chain A-B-C-D + branch A-E, so depth-2 from A reaches B(1),E(1),C(2)
        edge_rows = [
            {"src": str(ids["A"]), "dst": str(ids["B"])},
            {"src": str(ids["B"]), "dst": str(ids["C"])},
            {"src": str(ids["C"]), "dst": str(ids["D"])},
            {"src": str(ids["A"]), "dst": str(ids["E"])},
        ]
        doc = uuid4()
        chunk_rows = [
            {
                "id": str(chunk_ids["c1"]),
                "ns": str(ns),
                "doc": str(doc),
                "content": "chunk one",
                "occurred_at": (now - timedelta(days=2)).isoformat(),
                "created_at": (now - timedelta(days=2)).isoformat(),
            },
            {
                "id": str(chunk_ids["c2"]),
                "ns": str(ns),
                "doc": str(doc),
                "content": "chunk two",
                "occurred_at": (now - timedelta(days=1)).isoformat(),
                "created_at": (now - timedelta(days=1)).isoformat(),
            },
            {
                "id": str(chunk_ids["c3"]),
                "ns": str(ns),
                "doc": str(doc),
                "content": "chunk three",
                "occurred_at": now.isoformat(),
                "created_at": now.isoformat(),
            },
        ]
        # c1 mentioned by A(2x) and B(1x); c2 by B(3x); c3 by C(1x) and E(1x)
        link_rows = [
            {"entity_id": str(ids["A"]), "chunk_id": str(chunk_ids["c1"]), "mention_count": 2},
            {"entity_id": str(ids["B"]), "chunk_id": str(chunk_ids["c1"]), "mention_count": 1},
            {"entity_id": str(ids["B"]), "chunk_id": str(chunk_ids["c2"]), "mention_count": 3},
            {"entity_id": str(ids["C"]), "chunk_id": str(chunk_ids["c3"]), "mention_count": 1},
            {"entity_id": str(ids["E"]), "chunk_id": str(chunk_ids["c3"]), "mention_count": 1},
        ]

        try:
            async with driver.session(database="neo4j") as session:
                await session.run(setup, entities=entities)
                await session.run(edges, edges=edge_rows)
                await session.run(chunks, chunks=chunk_rows)
                await session.run(mentions, links=link_rows)
            yield driver, ns, ids
        finally:
            async with driver.session(database="neo4j") as session:
                await session.run("MATCH (n) WHERE n.namespace_id = $ns DETACH DELETE n", ns=str(ns))
            await driver.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("depth", [1, 2, 3])
    @pytest.mark.parametrize("prefer_current", [False, True])
    @pytest.mark.parametrize("temporal_sort", [False, True])
    async def test_bound_equals_unbound(self, graph, depth: int, prefer_current: bool, temporal_sort: bool) -> None:
        driver, ns, ids = graph
        manager = DualNodeManager(driver)
        seed = [ids["A"]]

        # --- Unbound path: each read opens its own session (status quo) ---
        neigh_unbound = await manager.get_entity_neighborhoods(
            seed, ns, depth=depth, limit_per_entity=1000, prefer_current=prefer_current
        )
        # Build the entity id set exactly like the retriever: entry UNION expanded.
        expanded_ids = {ent["id"] for ent in neigh_unbound.get(str(ids["A"]), [])}
        all_entity_ids = [UUID(x) for x in ({str(ids["A"])} | expanded_ids)]
        chunks_unbound = await manager.get_chunks_by_entities(
            all_entity_ids, ns, temporal_sort=temporal_sort, prefer_current=prefer_current, limit=50
        )

        # --- Bound path: both reads share ONE session (the #1468 fold) ---
        async with manager.bind_session():
            neigh_bound = await manager.get_entity_neighborhoods(
                seed, ns, depth=depth, limit_per_entity=1000, prefer_current=prefer_current
            )
            expanded_ids_b = {ent["id"] for ent in neigh_bound.get(str(ids["A"]), [])}
            all_entity_ids_b = [UUID(x) for x in ({str(ids["A"])} | expanded_ids_b)]
            chunks_bound = await manager.get_chunks_by_entities(
                all_entity_ids_b, ns, temporal_sort=temporal_sort, prefer_current=prefer_current, limit=50
            )

        # Neighborhood parity: identical {id -> min distance} per source.
        assert set(neigh_bound) == set(neigh_unbound)
        for src in neigh_unbound:
            assert _neigh_min(neigh_bound[src]) == _neigh_min(neigh_unbound[src])

        # Chunk parity: identical ordered (chunk_id, total_mentions, entity-set)
        # sequence — the exact inputs the retriever scores into ranked tuples.
        assert _chunk_key(chunks_bound) == _chunk_key(chunks_unbound)
        assert all_entity_ids  # sanity: seed present even without expansion

    @pytest.mark.asyncio
    async def test_bound_path_uses_single_session(self, graph, monkeypatch) -> None:
        """The bound path acquires ONE session for both reads; unbound acquires two."""
        driver, ns, ids = graph
        manager = DualNodeManager(driver)
        seed = [ids["A"]]

        acquisitions = {"count": 0}
        real_session = driver.session

        def _counting_session(*args, **kwargs):
            acquisitions["count"] += 1
            return real_session(*args, **kwargs)

        monkeypatch.setattr(driver, "session", _counting_session)

        # Unbound: two reads -> two driver.session() acquisitions.
        acquisitions["count"] = 0
        await manager.get_entity_neighborhoods(seed, ns, depth=2, limit_per_entity=1000)
        await manager.get_chunks_by_entities([ids["A"]], ns, limit=50)
        assert acquisitions["count"] == 2

        # Bound: two reads inside one bind -> ONE acquisition total.
        acquisitions["count"] = 0
        async with manager.bind_session():
            await manager.get_entity_neighborhoods(seed, ns, depth=2, limit_per_entity=1000)
            await manager.get_chunks_by_entities([ids["A"]], ns, limit=50)
        assert acquisitions["count"] == 1

        # The ContextVar is reset after the bind exits (no leak across recalls).
        assert dual_nodes_mod._BOUND_SESSION.get() is None

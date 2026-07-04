"""Before/after benchmark for issue #1419 - neighborhood expansion blowup.

Builds a dense synthetic entity graph (circulant graph: N entities, each
connected to its K nearest ring neighbors on both sides -> degree 2K) in a
throwaway namespace, then times:

  legacy: the pre-#1419 all-paths query
          (OPTIONAL MATCH path = (e)-[*1..depth]-(related:Entity))
  new:    DualNodeManager.get_entity_neighborhoods (bounded per-hop
          frontier expansion)

and verifies both produce the identical {related_id -> min distance} map.

The legacy query's cost is exponential in density (every undirected trail is
enumerated); the new query is linear in reachable nodes. On the default
shape (48 entities, degree 12, depth 4, 8 sources) the ring is fully
reachable within 2 hops, so the trail count - not the result size - is what
the legacy query pays for.

Usage (against this repo's compose stack; `make dev` first):

    uv run python scripts/bench_neighborhood_expansion.py
    uv run python scripts/bench_neighborhood_expansion.py --entities 64 --k 8 --depth 3

Env overrides: KHORA_NEO4J_TEST_URL / _USERNAME / _PASSWORD
(defaults bolt://localhost:7688, neo4j, pleaseletmein).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from uuid import uuid4

from neo4j import AsyncGraphDatabase

from khora.engines.vectorcypher.dual_nodes import DualNodeManager

LEGACY_QUERY_TEMPLATE = """
UNWIND $entity_ids AS eid
MATCH (e:Entity {{id: eid, namespace_id: $namespace_id}})
OPTIONAL MATCH path = (e)-[*1..{depth}]-(related:Entity)
WHERE related.namespace_id = $namespace_id
  AND related.id <> e.id
WITH e, related,
     CASE WHEN related IS NOT NULL THEN length(path) ELSE null END AS distance
ORDER BY e.id, distance
With e, collect(DISTINCT CASE
    WHEN related IS NOT NULL THEN {{
        id: related.id, name: related.name, entity_type: related.entity_type,
        description: related.description, source_tool: related.source_tool,
        distance: distance
    }}
    ELSE null
END)[0..$limit] AS related_raw
RETURN e.id AS source_id,
       [x IN related_raw WHERE x IS NOT NULL] AS related_entities
"""


def _min_distances(related: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ent in related:
        if ent["id"] not in out or ent["distance"] < out[ent["id"]]:
            out[ent["id"]] = ent["distance"]
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=48, help="ring size N")
    parser.add_argument("--k", type=int, default=6, help="ring reach (degree = 2k)")
    parser.add_argument("--depth", type=int, default=4, help="traversal depth (1-4)")
    parser.add_argument("--sources", type=int, default=8, help="number of entry entities")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per variant")
    args = parser.parse_args()

    url = os.environ.get("KHORA_NEO4J_TEST_URL", "bolt://localhost:7688")
    user = os.environ.get("KHORA_NEO4J_TEST_USERNAME", "neo4j")
    password = os.environ.get("KHORA_NEO4J_TEST_PASSWORD", "pleaseletmein")

    driver = AsyncGraphDatabase.driver(url, auth=(user, password))
    ns = uuid4()
    ids = [uuid4() for _ in range(args.entities)]
    sources = ids[: args.sources]
    # Parity must be checked over the FULL result set; a small limit would
    # make the comparison depend on tie-ordering inside the truncation.
    limit = 100_000

    entities = [{"id": str(eid), "ns": str(ns), "name": f"e{i}"} for i, eid in enumerate(ids)]
    edges = [
        {"src": str(ids[i]), "dst": str(ids[(i + off) % args.entities])}
        for i in range(args.entities)
        for off in range(1, args.k + 1)
    ]

    async with driver.session(database="neo4j") as session:
        await session.run(
            "UNWIND $entities AS ent "
            "CREATE (:Entity {id: ent.id, namespace_id: ent.ns, name: ent.name, "
            "entity_type: 'THING', description: null, source_tool: null})",
            entities=entities,
        )
        await session.run(
            "UNWIND $edges AS edge "
            "MATCH (a:Entity {id: edge.src}), (b:Entity {id: edge.dst}) "
            "CREATE (a)-[:RELATES_TO]->(b)",
            edges=edges,
        )

    manager = DualNodeManager(driver)
    params = {
        "entity_ids": [str(s) for s in sources],
        "namespace_id": str(ns),
        "limit": limit,
    }

    async def run_legacy() -> dict[str, dict[str, int]]:
        async with driver.session(database="neo4j") as session:
            result = await session.run(LEGACY_QUERY_TEMPLATE.format(depth=args.depth), **params)
            records = [record.data() async for record in result]
        return {r["source_id"]: _min_distances(r["related_entities"]) for r in records}

    async def run_new() -> dict[str, dict[str, int]]:
        result = await manager.get_entity_neighborhoods(sources, ns, depth=args.depth, limit_per_entity=limit)
        return {sid: _min_distances(rel) for sid, rel in result.items()}

    try:
        print(
            f"graph: {args.entities} entities, degree {2 * args.k}, "
            f"depth {args.depth}, {args.sources} sources, {args.runs} runs"
        )
        results = {}
        for name, fn in (("legacy (all-paths)", run_legacy), ("new (per-hop BFS)", run_new)):
            times = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                results[name] = await fn()
                times.append(time.perf_counter() - t0)
            best, mean = min(times), sum(times) / len(times)
            print(f"{name:20s} best {best * 1000:9.1f} ms   mean {mean * 1000:9.1f} ms")

        legacy_res = results["legacy (all-paths)"]
        new_res = results["new (per-hop BFS)"]
        assert legacy_res == new_res, "RESULT MISMATCH between legacy and new expansion"
        n_related = sum(len(v) for v in new_res.values())
        print(f"parity: OK ({len(new_res)} sources, {n_related} related entries, identical min-distance maps)")
    finally:
        async with driver.session(database="neo4j") as session:
            await session.run("MATCH (n:Entity {namespace_id: $ns}) DETACH DELETE n", ns=str(ns))
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())

"""Real-Neo4j integration test for ``list_relationships`` endpoint scoping (#1237).

Unlike the mock-based unit tests (which stub ``.data()`` and so can't detect a
regression in the Cypher shape), this test plants — via raw Cypher — the two
data shapes the public API can't create:

  1. a relationship whose endpoint ``:Entity`` node is missing its ``id``
     property (the ``UUID(None)`` crash trigger), and
  2. a cross-namespace endpoint edge,

then asserts ``list_relationships`` (a) does not raise and (b) returns only the
well-formed in-namespace edge — and that ``delete_malformed_orphan_relationships``
sweeps the malformed orphan that the enumerate-and-delete path can't reach.

Gated by ``NEO4J_INTEGRATION_TEST=1`` (khora CI does not provision Neo4j). Run:

    make dev
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_list_relationships_scoping_integration.py -v
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.storage.backends.neo4j import Neo4jBackend


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestNeo4jListRelationshipsScopingIntegration:
    @pytest.mark.asyncio
    async def test_excludes_malformed_and_cross_namespace_edges(self) -> None:
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        ns_a, ns_b = uuid4(), uuid4()
        doc_id = uuid4()
        marker = uuid4().hex
        alice = Entity(namespace_id=ns_a, name=f"alice-{marker}", entity_type="PERSON")
        carol = Entity(namespace_id=ns_a, name=f"carol-{marker}", entity_type="PERSON")
        bob = Entity(namespace_id=ns_b, name=f"bob-{marker}", entity_type="PERSON")
        good = Relationship(
            namespace_id=ns_a,
            source_entity_id=alice.id,
            target_entity_id=carol.id,
            relationship_type="KNOWS",
            description="well-formed in-namespace edge",
        )

        try:
            await backend.create_entity(alice)
            await backend.create_entity(carol)
            await backend.create_entity(bob)
            await backend.create_relationship(good)

            async with backend._session() as session:
                # (1) malformed endpoint: an :Entity in ns_a with NO id property,
                # plus an orphan edge (sole source = doc_id) pointing at it.
                await session.run(
                    "MATCH (a:Entity {id: $aid}) "
                    "CREATE (x:Entity {namespace_id: $ns, marker: $marker}) "
                    "CREATE (a)-[r:KNOWS {id: $rid, namespace_id: $ns, "
                    "source_document_ids: [$doc], created_at: $now}]->(x)",
                    aid=str(alice.id),
                    ns=str(ns_a),
                    marker=marker,
                    rid=str(uuid4()),
                    doc=str(doc_id),
                    now="2026-01-01T00:00:00+00:00",
                )
                # (2) cross-namespace endpoint: alice (ns_a) -> bob (ns_b),
                # edge stamped with ns_a.
                await session.run(
                    "MATCH (a:Entity {id: $aid}) MATCH (b:Entity {id: $bid}) "
                    "CREATE (a)-[r:KNOWS {id: $rid, namespace_id: $ns, created_at: $now}]->(b)",
                    aid=str(alice.id),
                    bid=str(bob.id),
                    rid=str(uuid4()),
                    ns=str(ns_a),
                    now="2026-01-01T00:00:00+00:00",
                )

            # Must not raise, and must return ONLY the well-formed in-ns edge.
            rels = await backend.list_relationships(ns_a, limit=100)
            assert [r.id for r in rels] == [good.id]

            # The malformed orphan edge survives the enumerate-and-delete path
            # (it can't be deserialized) but the sweep removes it.
            swept = await backend.delete_malformed_orphan_relationships(doc_id, namespace_id=ns_a)
            assert swept == 1
        finally:
            # DETACH DELETE removes the nodes and every edge incident to them
            # (the well-formed, cross-namespace, and malformed edges alike).
            async with backend._session() as session:
                await session.run(
                    "MATCH (n:Entity) WHERE n.id IN $ids OR n.marker = $marker DETACH DELETE n",
                    ids=[str(alice.id), str(carol.id), str(bob.id)],
                    marker=marker,
                )
            await backend.disconnect()

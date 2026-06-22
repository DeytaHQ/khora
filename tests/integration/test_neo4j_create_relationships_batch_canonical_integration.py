"""Real-Neo4j integration test for ``create_relationships_batch``'s canonical
per-edge results (#1320).

Verifies against a running Neo4j that:
  - a genuine create returns ``is_new=True`` and the stored id is the input id;
  - re-asserting the same (source, target, type, namespace) edge with a *fresh*
    submitted id MERGEs onto the existing edge: returns ``is_new=False`` and
    syncs the in-place ``rel.id`` to the FIRST edge's canonical id, not the
    second submitted one.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    KHORA_NEO4J_URL="bolt://localhost:7688" KHORA_NEO4J_PASSWORD="pleaseletmein" \
        NEO4J_INTEGRATION_TEST=1 UV_NO_SYNC=1 uv run pytest \
        tests/integration/test_neo4j_create_relationships_batch_canonical_integration.py \
        -o addopts="" -q
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
class TestNeo4jCreateRelationshipsBatchCanonical:
    @pytest.mark.asyncio
    async def test_create_then_merge_reports_canonical_id_and_is_new(self) -> None:
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        a = Entity(namespace_id=namespace_id, name=f"acme-{uuid4().hex[:8]}", entity_type="ORGANIZATION")
        b = Entity(namespace_id=namespace_id, name=f"beta-{uuid4().hex[:8]}", entity_type="ORGANIZATION")

        try:
            await backend.create_entity(a)
            await backend.create_entity(b)

            first = Relationship(
                namespace_id=namespace_id,
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="ACQUIRED",
                confidence=0.9,
            )
            first_id = first.id
            results = await backend.create_relationships_batch([first])
            assert len(results) == 1
            rel_out, is_new = results[0]
            # Genuine create: is_new=True, stored id is the submitted id.
            assert is_new is True
            assert rel_out.id == first_id

            # Re-assert the SAME endpoint pair + type with a fresh submitted id.
            second = Relationship(
                namespace_id=namespace_id,
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="ACQUIRED",
                confidence=0.95,
            )
            assert second.id != first_id
            results2 = await backend.create_relationships_batch([second])
            assert len(results2) == 1
            rel_out2, is_new2 = results2[0]
            # MERGE onto the existing edge: is_new=False, canonical id is the
            # FIRST edge's id (synced in place), never the second submitted id.
            assert is_new2 is False
            assert rel_out2.id == first_id
            assert second.id == first_id
        finally:
            async with backend._session() as session:
                await session.run(
                    "MATCH (n:Entity {namespace_id: $ns}) DETACH DELETE n",
                    ns=str(namespace_id),
                )
            await backend.disconnect()

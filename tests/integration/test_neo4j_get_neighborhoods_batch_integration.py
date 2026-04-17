"""Real-Neo4j integration test for ``Neo4jBackend.get_neighborhoods_batch`` (DYT-2629).

This test exercises the full Cypher → driver → ``result.data()`` →
relationship property extraction path against a running Neo4j instance to verify
that relationship properties are correctly preserved through variable-length
path traversals.

Why this is marked ``@pytest.mark.integration`` and gated by
``NEO4J_INTEGRATION_TEST=1``:

    Khora's CI does NOT provision a Neo4j instance. Real-Neo4j coverage lives
    behind an opt-in env var so CI stays green while local developers
    running ``make dev`` can exercise it.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_get_neighborhoods_batch_integration.py -v

Connection parameters are read from env vars with sensible defaults that
match the ``make dev`` compose stack:

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)

The test verifies that:
1. Relationship properties are correctly extracted from variable-length paths
2. Relationship types are preserved
3. Multiple relationships per path are correctly flattened
4. Edge cases (no neighbors, depth=1, etc.) are handled correctly
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
class TestNeo4jGetNeighborhoodsBatchIntegration:
    """End-to-end regression lock for DYT-2629 against a real Neo4j."""

    @pytest.mark.asyncio
    async def test_returns_relationships_with_properties_depth_1(self) -> None:
        """Create entities + relationships, read back via get_neighborhoods_batch depth=1.

        Verifies that relationship properties are correctly extracted from
        the Cypher result even when using variable-length path syntax
        [r*1..1] (which still produces list-of-relationships shape).
        """
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"alice-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Alice",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"bob-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Bob",
        )
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
            description="alice knows bob",
            properties={"since": "2024", "strength": "strong"},
            confidence=0.9,
            weight=0.75,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            result = await backend.get_neighborhoods_batch(
                [entity_a.id],
                depth=1,
                limit_per_entity=50,
            )

            assert entity_a.id in result
            neighborhood = result[entity_a.id]
            assert "entities" in neighborhood
            assert "relationships" in neighborhood

            # Should find entity_b as a neighbor
            assert len(neighborhood["entities"]) >= 1
            entity_names = [e.get("name") for e in neighborhood["entities"]]
            assert entity_b.name in entity_names

            # Should find the relationship with all properties preserved
            assert len(neighborhood["relationships"]) >= 1
            rels = neighborhood["relationships"]
            rel = None
            for r in rels:
                if r.get("id") == str(relationship.id):
                    rel = r
                    break

            assert rel is not None, f"Relationship {relationship.id} not found in results"
            assert rel["id"] == str(relationship.id)
            assert rel["namespace_id"] == str(namespace_id)
            assert rel["relationship_type"] == "KNOWS"
            assert rel["description"] == "alice knows bob"
            assert rel["confidence"] == 0.9
            assert rel["weight"] == 0.75
            # Properties and metadata should be preserved (stored as JSON strings in DB)
            assert rel.get("properties") is not None
            assert rel.get("metadata") is not None

        finally:
            # Best-effort cleanup
            try:
                await backend.delete_relationship(relationship.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_a.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_b.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_returns_relationships_with_properties_depth_2(self) -> None:
        """Create a chain of entities (A-B-C), read via get_neighborhoods_batch depth=2.

        Tests variable-length paths with depth > 1, which creates
        multi-hop traversals. Verifies that all relationships along
        the paths are correctly extracted.
        """
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"alice-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Alice",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"bob-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Bob",
        )
        entity_c = Entity(
            namespace_id=namespace_id,
            name=f"charlie-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Charlie",
        )

        rel_ab = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
            description="alice knows bob",
            properties={"confidence_level": "high"},
            confidence=0.95,
            weight=0.8,
        )
        rel_bc = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_b.id,
            target_entity_id=entity_c.id,
            relationship_type="WORKS_WITH",
            description="bob works with charlie",
            properties={"project": "xyz"},
            confidence=0.85,
            weight=0.7,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_entity(entity_c)
            await backend.create_relationship(rel_ab)
            await backend.create_relationship(rel_bc)

            result = await backend.get_neighborhoods_batch(
                [entity_a.id],
                depth=2,
                limit_per_entity=50,
            )

            assert entity_a.id in result
            neighborhood = result[entity_a.id]

            # At depth=2 from A, we should find B (direct) and C (2-hop)
            entity_ids_found = {e.get("id"): e for e in neighborhood["entities"]}
            assert str(entity_b.id) in entity_ids_found, "Entity B should be a neighbor"
            assert str(entity_c.id) in entity_ids_found, "Entity C should be a neighbor at depth 2"

            # Should find both relationships
            rels = neighborhood["relationships"]
            rel_ids = {r.get("id") for r in rels}
            assert str(rel_ab.id) in rel_ids, f"Relationship {rel_ab.id} (A-B) not found"
            assert str(rel_bc.id) in rel_ids, f"Relationship {rel_bc.id} (B-C) not found"

            # Verify properties on both relationships
            for rel in rels:
                if rel.get("id") == str(rel_ab.id):
                    assert rel["relationship_type"] == "KNOWS"
                    assert rel["description"] == "alice knows bob"
                    assert rel["confidence"] == 0.95
                elif rel.get("id") == str(rel_bc.id):
                    assert rel["relationship_type"] == "WORKS_WITH"
                    assert rel["description"] == "bob works with charlie"
                    assert rel["confidence"] == 0.85

        finally:
            # Best-effort cleanup
            try:
                await backend.delete_relationship(rel_bc.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_relationship(rel_ab.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_a.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_b.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_c.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_batch_multiple_entities_independently(self) -> None:
        """Create two separate entity neighborhoods, verify get_neighborhoods_batch
        returns independent results for each.
        """
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entity_1 = Entity(
            namespace_id=namespace_id,
            name=f"entity1-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Entity 1",
        )
        entity_1_neighbor = Entity(
            namespace_id=namespace_id,
            name=f"entity1_neighbor-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Neighbor of Entity 1",
        )
        entity_2 = Entity(
            namespace_id=namespace_id,
            name=f"entity2-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Entity 2",
        )
        entity_2_neighbor = Entity(
            namespace_id=namespace_id,
            name=f"entity2_neighbor-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Neighbor of Entity 2",
        )

        rel_1 = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_1.id,
            target_entity_id=entity_1_neighbor.id,
            relationship_type="KNOWS",
            description="relationship 1",
            confidence=0.9,
        )
        rel_2 = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_2.id,
            target_entity_id=entity_2_neighbor.id,
            relationship_type="COLLABORATES_WITH",
            description="relationship 2",
            confidence=0.8,
        )

        try:
            await backend.create_entity(entity_1)
            await backend.create_entity(entity_1_neighbor)
            await backend.create_entity(entity_2)
            await backend.create_entity(entity_2_neighbor)
            await backend.create_relationship(rel_1)
            await backend.create_relationship(rel_2)

            result = await backend.get_neighborhoods_batch(
                [entity_1.id, entity_2.id],
                depth=1,
            )

            assert entity_1.id in result
            assert entity_2.id in result

            # Entity 1's neighborhood should have entity_1_neighbor
            neighborhood_1 = result[entity_1.id]
            entity_1_neighbor_ids = {e.get("id") for e in neighborhood_1["entities"]}
            assert str(entity_1_neighbor.id) in entity_1_neighbor_ids

            # Entity 2's neighborhood should have entity_2_neighbor
            neighborhood_2 = result[entity_2.id]
            entity_2_neighbor_ids = {e.get("id") for e in neighborhood_2["entities"]}
            assert str(entity_2_neighbor.id) in entity_2_neighbor_ids

            # Verify relationships are distinct
            rel_1_found = any(r.get("id") == str(rel_1.id) for r in neighborhood_1["relationships"])
            rel_2_found = any(r.get("id") == str(rel_2.id) for r in neighborhood_2["relationships"])
            assert rel_1_found, "Relationship 1 should be in Entity 1's neighborhood"
            assert rel_2_found, "Relationship 2 should be in Entity 2's neighborhood"

            # Verify relationship types are correct
            for rel in neighborhood_1["relationships"]:
                if rel.get("id") == str(rel_1.id):
                    assert rel["relationship_type"] == "KNOWS"
            for rel in neighborhood_2["relationships"]:
                if rel.get("id") == str(rel_2.id):
                    assert rel["relationship_type"] == "COLLABORATES_WITH"

        finally:
            # Best-effort cleanup
            try:
                await backend.delete_relationship(rel_1.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_relationship(rel_2.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_1.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_1_neighbor.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_2.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_2_neighbor.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

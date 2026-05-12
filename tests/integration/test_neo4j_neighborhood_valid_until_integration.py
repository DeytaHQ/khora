"""Real-Neo4j integration test for relationship valid_until filtering in get_entity_neighborhoods.

This test exercises the valid_until filtering in DualNodeManager.get_entity_neighborhoods()
to verify that:
1. Relationships with valid_until in the future surface when prefer_current=True
2. Relationships with valid_until in the past do NOT surface when prefer_current=True
3. Expired relationships STILL surface when prefer_current=False

Why this is marked @pytest.mark.integration and gated by NEO4J_INTEGRATION_TEST=1:

    Khora's CI does NOT provision a Neo4j instance. Real-Neo4j coverage lives
    behind an opt-in env var so CI stays green while local developers
    running ``make dev`` can exercise it.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_neighborhood_valid_until_integration.py -v

Connection parameters are read from env vars with sensible defaults that
match the ``make dev`` compose stack:

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)

The test verifies that:
1. Future-dated valid_until relationships are included with prefer_current=True
2. Past-dated valid_until relationships are excluded with prefer_current=True
3. Past-dated valid_until relationships are included with prefer_current=False
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.engines.vectorcypher.dual_nodes import DualNodeManager
from khora.storage.backends.neo4j import Neo4jBackend


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestDualNodeManagerValidUntilFiltering:
    """End-to-end tests for valid_until filtering in get_entity_neighborhoods."""

    @pytest.mark.asyncio
    async def test_future_valid_until_surfaces_with_prefer_current_true(self) -> None:
        """Relationship with valid_until in future is included when prefer_current=True."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        # Create DualNodeManager from the same driver
        manager = DualNodeManager(backend._driver)

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

        # Relationship with valid_until far in the future
        future_date = datetime.now(UTC) + timedelta(days=365)
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
            description="alice knows bob",
            valid_until=future_date,
            confidence=0.9,
            weight=0.75,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            # Query with prefer_current=True
            # Should find entity_b as a neighbor because valid_until is in the future
            result = await manager.get_entity_neighborhoods(
                [entity_a.id],
                namespace_id,
                depth=1,
                prefer_current=True,
            )

            assert str(entity_a.id) in result
            neighborhood = result[str(entity_a.id)]

            # Should find entity_b in the neighborhood
            entity_b_ids = {ent.get("id") for ent in neighborhood}
            assert str(entity_b.id) in entity_b_ids, "entity_b should be found when rel has future valid_until"

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
    async def test_past_valid_until_excluded_with_prefer_current_true(self) -> None:
        """Relationship with valid_until in past is excluded when prefer_current=True."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        # Create DualNodeManager from the same driver
        manager = DualNodeManager(backend._driver)

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"carol-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Carol",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"dave-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Dave",
        )

        # Relationship with valid_until in the past
        past_date = datetime.now(UTC) - timedelta(days=365)
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="WORKS_WITH",
            description="carol works with dave",
            valid_until=past_date,
            confidence=0.85,
            weight=0.7,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            # Query with prefer_current=True
            # Should NOT find entity_b as a neighbor because valid_until is in the past
            result = await manager.get_entity_neighborhoods(
                [entity_a.id],
                namespace_id,
                depth=1,
                prefer_current=True,
            )

            assert str(entity_a.id) in result
            neighborhood = result[str(entity_a.id)]

            # Should NOT find entity_b in the neighborhood (filtered out)
            entity_b_ids = {ent.get("id") for ent in neighborhood}
            assert str(entity_b.id) not in entity_b_ids, "entity_b should NOT be found when rel has past valid_until"

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
    async def test_past_valid_until_surfaces_with_prefer_current_false(self) -> None:
        """Relationship with valid_until in past is included when prefer_current=False."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        # Create DualNodeManager from the same driver
        manager = DualNodeManager(backend._driver)

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"eve-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Eve",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"frank-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Frank",
        )

        # Relationship with valid_until in the past
        past_date = datetime.now(UTC) - timedelta(days=100)
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="COLLABORATES_WITH",
            description="eve collaborates with frank",
            valid_until=past_date,
            confidence=0.8,
            weight=0.6,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            # Query with prefer_current=False (default behavior)
            # Should find entity_b as a neighbor even though valid_until is in the past
            result = await manager.get_entity_neighborhoods(
                [entity_a.id],
                namespace_id,
                depth=1,
                prefer_current=False,
            )

            assert str(entity_a.id) in result
            neighborhood = result[str(entity_a.id)]

            # Should find entity_b in the neighborhood (no filtering applied)
            entity_b_ids = {ent.get("id") for ent in neighborhood}
            assert str(entity_b.id) in entity_b_ids, "entity_b should be found when prefer_current=False"

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
    async def test_null_valid_until_always_included(self) -> None:
        """Relationship with NULL valid_until is always included regardless of prefer_current."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        # Create DualNodeManager from the same driver
        manager = DualNodeManager(backend._driver)

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"grace-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Grace",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"henry-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Henry",
        )

        # Relationship with NO valid_until (NULL = no known end = always valid)
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="MANAGES",
            description="grace manages henry",
            confidence=0.95,
            weight=0.9,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            # Query with prefer_current=True
            # Should still find entity_b even though we're filtering for current relationships
            # because valid_until is NULL (no known end)
            result = await manager.get_entity_neighborhoods(
                [entity_a.id],
                namespace_id,
                depth=1,
                prefer_current=True,
            )

            assert str(entity_a.id) in result
            neighborhood = result[str(entity_a.id)]

            # Should find entity_b because NULL valid_until means still valid
            entity_b_ids = {ent.get("id") for ent in neighborhood}
            assert str(entity_b.id) in entity_b_ids, (
                "entity_b should be found when rel has NULL valid_until (always valid)"
            )

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
    async def test_depth2_expired_intermediate_relationship_excluded(self) -> None:
        """Depth>1: expired intermediate relationship blocks traversal to deeper nodes."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        manager = DualNodeManager(backend._driver)

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"alpha-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Alpha",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"bravo-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Bravo",
        )
        entity_c = Entity(
            namespace_id=namespace_id,
            name=f"charlie-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Charlie",
        )

        # A→B: current (no valid_until)
        rel_ab = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
            description="alpha knows bravo",
            confidence=0.9,
            weight=0.8,
        )

        # B→C: expired (valid_until in the past)
        past_date = datetime.now(UTC) - timedelta(days=365)
        rel_bc = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_b.id,
            target_entity_id=entity_c.id,
            relationship_type="KNOWS",
            description="bravo knows charlie",
            valid_until=past_date,
            confidence=0.9,
            weight=0.8,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_entity(entity_c)
            await backend.create_relationship(rel_ab)
            await backend.create_relationship(rel_bc)

            result = await manager.get_entity_neighborhoods(
                [entity_a.id],
                namespace_id,
                depth=2,
                prefer_current=True,
            )

            assert str(entity_a.id) in result
            neighborhood = result[str(entity_a.id)]

            neighbors_by_id = {ent.get("id"): ent for ent in neighborhood}

            # B should appear at distance 1 (direct neighbor via current rel_ab)
            assert str(entity_b.id) in neighbors_by_id, "entity_b should be reachable via current rel_ab"
            assert neighbors_by_id[str(entity_b.id)].get("distance") == 1

            # C should NOT appear (path A→B→C includes expired rel_bc)
            assert str(entity_c.id) not in neighbors_by_id, (
                "entity_c should NOT be reachable — intermediate rel_bc is expired"
            )

        finally:
            for rel_id in (rel_ab.id, rel_bc.id):
                try:
                    await backend.delete_relationship(rel_id)
                except Exception:  # noqa: BLE001
                    pass
            for ent_id in (entity_a.id, entity_b.id, entity_c.id):
                try:
                    await backend.delete_entity(ent_id)
                except Exception:  # noqa: BLE001
                    pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_multi_relationship_current_path_surfaces(self) -> None:
        """Multiple relationships between same nodes: current path surfaces despite expired one."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        manager = DualNodeManager(backend._driver)

        namespace_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"delta-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Delta",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"echo-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Echo",
        )

        # rel1: A→B expired
        past_date = datetime.now(UTC) - timedelta(days=365)
        rel1 = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="WORKED_WITH",
            description="delta worked with echo (past)",
            valid_until=past_date,
            confidence=0.8,
            weight=0.7,
        )

        # rel2: A→B current (no valid_until)
        rel2 = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="COLLABORATES_WITH",
            description="delta collaborates with echo (current)",
            confidence=0.9,
            weight=0.8,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(rel1)
            await backend.create_relationship(rel2)

            result = await manager.get_entity_neighborhoods(
                [entity_a.id],
                namespace_id,
                depth=1,
                prefer_current=True,
            )

            assert str(entity_a.id) in result
            neighborhood = result[str(entity_a.id)]

            entity_b_ids = {ent.get("id") for ent in neighborhood}
            assert str(entity_b.id) in entity_b_ids, (
                "entity_b should be reachable via current rel2 despite expired rel1"
            )

        finally:
            for rel_id in (rel1.id, rel2.id):
                try:
                    await backend.delete_relationship(rel_id)
                except Exception:  # noqa: BLE001
                    pass
            for ent_id in (entity_a.id, entity_b.id):
                try:
                    await backend.delete_entity(ent_id)
                except Exception:  # noqa: BLE001
                    pass
            await backend.disconnect()

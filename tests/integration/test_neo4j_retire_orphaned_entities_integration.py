"""Real-Neo4j integration tests for ``Neo4jBackend.retire_orphaned_entities_batch``.

These tests exercise the full retire → snapshot → temporal-query path against
a running Neo4j instance, verifying that :EntityVersion nodes, [:SUPERSEDES]
edges, and temporal property stamps work end-to-end through the driver.

Gated by ``NEO4J_INTEGRATION_TEST=1`` (set by the CI integration job).

How to run locally:

    make dev
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_retire_orphaned_entities_integration.py -v
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.storage.backends.neo4j import Neo4jBackend


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestNeo4jRetireOrphanedEntitiesBatchIntegration:
    """End-to-end tests for retire_orphaned_entities_batch against a real Neo4j."""

    @pytest.mark.asyncio
    async def test_retire_single_entity_creates_snapshot(self) -> None:
        """Retire one entity and verify the EntityVersion snapshot, SUPERSEDES edge,
        and stamps on the current entity."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"retire-test-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Test entity for retirement",
            attributes={"role": "engineer", "level": "senior"},
            source_document_ids=[uuid4()],
            confidence=0.95,
            metadata={"created_at": "2026-01-01T00:00:00+00:00"},
        )
        snapshot_id = uuid4()
        retired_at = datetime.now(UTC).isoformat()

        try:
            await backend.create_entity(entity)

            count = await backend.retire_orphaned_entities_batch(
                [
                    {
                        "current_id": str(entity.id),
                        "snapshot_id": str(snapshot_id),
                        "namespace_id": str(namespace_id),
                        "retired_at": retired_at,
                    }
                ]
            )
            assert count == 1

            # Verify via direct Cypher queries
            async with backend._session() as session:

                async def _verify(tx):
                    # 1. EntityVersion node
                    r = await tx.run(
                        "MATCH (ev:EntityVersion {id: $sid}) RETURN ev",
                        sid=str(snapshot_id),
                    )
                    ev_records = await r.data()
                    assert len(ev_records) == 1, "EntityVersion node not created"
                    ev = ev_records[0]["ev"]
                    assert ev["namespace_id"] == str(namespace_id)
                    assert ev["name"] == entity.name
                    assert ev["entity_type"] == "PERSON"
                    assert ev["description"] == "Test entity for retirement"
                    assert ev["confidence"] == 0.95
                    assert ev["retirement_reason"] == "document_replaced"
                    assert ev["version_valid_to"] == retired_at

                    # 2. SUPERSEDES edge
                    r = await tx.run(
                        """
                        MATCH (c:Entity {id: $eid})-[s:SUPERSEDES]->(ev:EntityVersion {id: $sid})
                        RETURN s.superseded_at AS superseded_at, s.reason AS reason
                        """,
                        eid=str(entity.id),
                        sid=str(snapshot_id),
                    )
                    edge_records = await r.data()
                    assert len(edge_records) == 1, "SUPERSEDES edge not created"
                    assert edge_records[0]["superseded_at"] == retired_at
                    assert edge_records[0]["reason"] == "document_replaced"

                    # 3. Current entity stamps
                    r = await tx.run(
                        """
                        MATCH (e:Entity {id: $eid})
                        RETURN e.valid_until AS valid_until,
                               e.version_valid_to AS version_valid_to,
                               e.updated_at AS updated_at
                        """,
                        eid=str(entity.id),
                    )
                    ent_records = await r.data()
                    assert len(ent_records) == 1
                    assert ent_records[0]["valid_until"] == retired_at
                    assert ent_records[0]["version_valid_to"] == retired_at
                    assert ent_records[0]["updated_at"] == retired_at

                await session.execute_read(_verify)

        finally:
            try:
                async with backend._session() as session:

                    async def _cleanup(tx):
                        await tx.run(
                            "MATCH (ev:EntityVersion {id: $sid}) DETACH DELETE ev",
                            sid=str(snapshot_id),
                        )
                        await tx.run(
                            "MATCH (e:Entity {id: $eid}) DETACH DELETE e",
                            eid=str(entity.id),
                        )

                    await session.execute_write(_cleanup)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_version_history_query_returns_snapshot(self) -> None:
        """Retire an entity, then run the _fetch_version_history Cypher pattern
        and verify the snapshot appears with correct fields."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"history-test-{uuid4().hex[:8]}",
            entity_type="ORGANIZATION",
            description="Org for version history test",
            attributes={"industry": "tech"},
            metadata={"created_at": "2026-01-01T00:00:00+00:00"},
        )
        snapshot_id = uuid4()
        retired_at = datetime.now(UTC).isoformat()

        try:
            await backend.create_entity(entity)
            await backend.retire_orphaned_entities_batch(
                [
                    {
                        "current_id": str(entity.id),
                        "snapshot_id": str(snapshot_id),
                        "namespace_id": str(namespace_id),
                        "retired_at": retired_at,
                    }
                ]
            )

            # Run the same Cypher as _fetch_version_history (retriever.py:1844-1859)
            query = """
            UNWIND $entity_ids AS eid
            MATCH (current:Entity {id: eid, namespace_id: $namespace_id})
            OPTIONAL MATCH (current)-[s:SUPERSEDES]->(prev:EntityVersion)
            RETURN current.id AS current_id,
                   current.name AS name,
                   current.entity_type AS entity_type,
                   current.attributes AS current_attributes,
                   current.version_valid_from AS current_valid_from,
                   current.version_valid_to AS current_valid_to,
                   prev.id AS previous_id,
                   prev.attributes AS previous_attributes,
                   prev.version_valid_from AS previous_valid_from,
                   prev.version_valid_to AS previous_valid_to,
                   s.superseded_at AS superseded_at
            ORDER BY current.name, s.superseded_at DESC
            """

            async with backend._session() as session:

                async def _work(tx):
                    result = await tx.run(
                        query,
                        entity_ids=[str(entity.id)],
                        namespace_id=str(namespace_id),
                    )
                    return [record.data() async for record in result]

                records = await session.execute_read(_work)

            assert len(records) == 1
            rec = records[0]
            assert rec["current_id"] == str(entity.id)
            assert rec["name"] == entity.name
            assert rec["entity_type"] == "ORGANIZATION"
            assert rec["previous_id"] == str(snapshot_id)
            assert rec["superseded_at"] == retired_at
            assert rec["previous_valid_to"] == retired_at

        finally:
            try:
                async with backend._session() as session:

                    async def _cleanup(tx):
                        await tx.run(
                            "MATCH (ev:EntityVersion {id: $sid}) DETACH DELETE ev",
                            sid=str(snapshot_id),
                        )
                        await tx.run(
                            "MATCH (e:Entity {id: $eid}) DETACH DELETE e",
                            eid=str(entity.id),
                        )

                    await session.execute_write(_cleanup)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_temporal_filtering_after_retirement(self) -> None:
        """Retire an entity, then run the _version_filter_entities Cypher pattern
        with target dates before and after retirement to verify correct filtering."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        now = datetime.now(UTC)
        t_creation = now - timedelta(hours=2)
        t_retirement = now - timedelta(hours=1)
        t_between = now - timedelta(minutes=90)  # Between creation and retirement
        t_after = now - timedelta(minutes=30)  # After retirement

        entity = Entity(
            namespace_id=namespace_id,
            name=f"temporal-test-{uuid4().hex[:8]}",
            entity_type="CONCEPT",
            description="Entity for temporal filtering test",
            metadata={"created_at": t_creation.isoformat()},
        )
        snapshot_id = uuid4()

        try:
            await backend.create_entity(entity)

            await backend.retire_orphaned_entities_batch(
                [
                    {
                        "current_id": str(entity.id),
                        "snapshot_id": str(snapshot_id),
                        "namespace_id": str(namespace_id),
                        "retired_at": t_retirement.isoformat(),
                    }
                ]
            )

            # The _version_filter_entities Cypher (retriever.py:1782-1801)
            filter_query = """
            UNWIND $entity_ids AS eid
            MATCH (e:Entity {id: eid, namespace_id: $namespace_id})
            OPTIONAL MATCH (e)-[:SUPERSEDES]->(ev:EntityVersion)
            WHERE ev.namespace_id = $namespace_id
              AND (ev.version_valid_from IS NULL OR ev.version_valid_from <= $target_date)
              AND (ev.version_valid_to IS NULL OR ev.version_valid_to > $target_date)
            WITH e, collect(ev.id) AS version_ids
            WITH e, version_ids,
                 CASE
                   WHEN e.version_valid_from IS NULL THEN true
                   WHEN e.version_valid_from <= $target_date
                        AND (e.version_valid_to IS NULL OR e.version_valid_to > $target_date)
                   THEN true
                   ELSE false
                 END AS current_valid
            WHERE current_valid OR size(version_ids) > 0
            RETURN CASE WHEN current_valid THEN e.id
                        ELSE version_ids[0]
                   END AS id
            """

            async with backend._session() as session:

                async def _query_at(tx, target_date: str):
                    result = await tx.run(
                        filter_query,
                        entity_ids=[str(entity.id)],
                        namespace_id=str(namespace_id),
                        target_date=target_date,
                    )
                    return [record.data() async for record in result]

                # T-90min: between creation and retirement — entity was valid
                # (current entity's version window [T-2h, T-1h) covers T-90min,
                #  so the filter returns the current entity, not the snapshot)
                records_between = await session.execute_read(lambda tx: _query_at(tx, t_between.isoformat()))
                assert len(records_between) == 1, f"Expected entity to be valid at T-90min, got {records_between}"
                assert records_between[0]["id"] == str(entity.id)

                # T-30min: after retirement — both current and snapshot are closed
                records_after = await session.execute_read(lambda tx: _query_at(tx, t_after.isoformat()))
                assert len(records_after) == 0, (
                    f"Expected no results at T-30min (after retirement), got {records_after}"
                )

        finally:
            try:
                async with backend._session() as session:

                    async def _cleanup(tx):
                        await tx.run(
                            "MATCH (ev:EntityVersion {id: $sid}) DETACH DELETE ev",
                            sid=str(snapshot_id),
                        )
                        await tx.run(
                            "MATCH (e:Entity {id: $eid}) DETACH DELETE e",
                            eid=str(entity.id),
                        )

                    await session.execute_write(_cleanup)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_retire_batch_multiple_entities(self) -> None:
        """Retire 3 entities in a single batch call and verify all snapshots
        and SUPERSEDES edges are created."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entities = [
            Entity(
                namespace_id=namespace_id,
                name=f"batch-test-{i}-{uuid4().hex[:8]}",
                entity_type="PERSON",
                description=f"Batch entity {i}",
                metadata={"created_at": "2026-01-01T00:00:00+00:00"},
            )
            for i in range(3)
        ]
        snapshot_ids = [uuid4() for _ in range(3)]
        retired_at = datetime.now(UTC).isoformat()

        try:
            for entity in entities:
                await backend.create_entity(entity)

            retirement_rows = [
                {
                    "current_id": str(entities[i].id),
                    "snapshot_id": str(snapshot_ids[i]),
                    "namespace_id": str(namespace_id),
                    "retired_at": retired_at,
                }
                for i in range(3)
            ]
            count = await backend.retire_orphaned_entities_batch(retirement_rows)
            assert count == 3

            # Verify all 3 EntityVersion nodes and SUPERSEDES edges
            async with backend._session() as session:

                async def _verify(tx):
                    for i in range(3):
                        r = await tx.run(
                            "MATCH (ev:EntityVersion {id: $sid}) RETURN ev",
                            sid=str(snapshot_ids[i]),
                        )
                        ev_records = await r.data()
                        assert len(ev_records) == 1, f"EntityVersion not created for entity {i}"

                        r = await tx.run(
                            """
                            MATCH (c:Entity {id: $eid})-[s:SUPERSEDES]->(ev:EntityVersion {id: $sid})
                            RETURN s
                            """,
                            eid=str(entities[i].id),
                            sid=str(snapshot_ids[i]),
                        )
                        edge_records = await r.data()
                        assert len(edge_records) == 1, f"SUPERSEDES edge not created for entity {i}"

                await session.execute_read(_verify)

        finally:
            try:
                async with backend._session() as session:

                    async def _cleanup(tx):
                        for i in range(3):
                            await tx.run(
                                "MATCH (ev:EntityVersion {id: $sid}) DETACH DELETE ev",
                                sid=str(snapshot_ids[i]),
                            )
                            await tx.run(
                                "MATCH (e:Entity {id: $eid}) DETACH DELETE e",
                                eid=str(entities[i].id),
                            )

                    await session.execute_write(_cleanup)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_list_entities_filters_by_source_chunk_ids(self) -> None:
        """``list_entities(source_chunk_ids=...)`` filters by chunk provenance (#1448).

        Seeds two entities — A sourced from chunks c1/c2, B from c3 — then pins
        the four contract cases: no filter returns both; a filter for one of A's
        chunks returns only A; an unknown chunk returns nothing; and an empty
        list matches nothing (any-overlap semantics).
        """
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        c1, c2, c3, c4 = uuid4(), uuid4(), uuid4(), uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"scid-A-{uuid4().hex[:8]}",
            entity_type="PERSON",
            source_chunk_ids=[c1, c2],
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"scid-B-{uuid4().hex[:8]}",
            entity_type="PERSON",
            source_chunk_ids=[c3],
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)

            # 1. No filter → both entities.
            names = {e.name for e in await backend.list_entities(namespace_id)}
            assert names == {entity_a.name, entity_b.name}

            # 2. One of A's chunks → exactly A.
            only_a = await backend.list_entities(namespace_id, source_chunk_ids=[c1])
            assert {e.name for e in only_a} == {entity_a.name}

            # 3. Unknown chunk id → nothing.
            assert await backend.list_entities(namespace_id, source_chunk_ids=[c4]) == []

            # 4. Empty list → matches nothing.
            assert await backend.list_entities(namespace_id, source_chunk_ids=[]) == []

        finally:
            try:
                async with backend._session() as session:

                    async def _cleanup(tx):
                        await tx.run(
                            "MATCH (e:Entity) WHERE e.id IN $ids DETACH DELETE e",
                            ids=[str(entity_a.id), str(entity_b.id)],
                        )

                    await session.execute_write(_cleanup)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_list_relationships_filters_by_between_entity_ids(self) -> None:
        """``list_relationships(between_entity_ids=...)`` filters by endpoint membership (#1451).

        Seeds A→B and B→C, then pins the four contract cases: no filter returns
        both edges; ``[A, B]`` returns exactly A→B (B→C excluded — C outside the
        set); ``[A]`` returns [] (no self-loops seeded); and an empty list
        returns [] (BOTH-endpoints-in-set semantics).
        """
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        entity_a = Entity(namespace_id=namespace_id, name=f"beid-A-{uuid4().hex[:8]}", entity_type="PERSON")
        entity_b = Entity(namespace_id=namespace_id, name=f"beid-B-{uuid4().hex[:8]}", entity_type="PERSON")
        entity_c = Entity(namespace_id=namespace_id, name=f"beid-C-{uuid4().hex[:8]}", entity_type="PERSON")
        rel_ab = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
        )
        rel_bc = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_b.id,
            target_entity_id=entity_c.id,
            relationship_type="KNOWS",
        )

        def _edges(rels):
            return {(r.source_entity_id, r.target_entity_id) for r in rels}

        try:
            await backend.upsert_entities_batch(namespace_id, [entity_a, entity_b, entity_c])
            await backend.create_relationships_batch([rel_ab, rel_bc])

            # 1. No filter → both edges.
            assert _edges(await backend.list_relationships(namespace_id)) == {
                (entity_a.id, entity_b.id),
                (entity_b.id, entity_c.id),
            }

            # 2. [A, B] → exactly A→B (B→C excluded — C outside the set).
            filtered = await backend.list_relationships(namespace_id, between_entity_ids=[entity_a.id, entity_b.id])
            assert _edges(filtered) == {(entity_a.id, entity_b.id)}

            # 3. [A] → nothing (no self-loops seeded).
            assert await backend.list_relationships(namespace_id, between_entity_ids=[entity_a.id]) == []

            # 4. Empty list → nothing.
            assert await backend.list_relationships(namespace_id, between_entity_ids=[]) == []

        finally:
            try:
                async with backend._session() as session:

                    async def _cleanup(tx):
                        await tx.run(
                            "MATCH (e:Entity) WHERE e.id IN $ids DETACH DELETE e",
                            ids=[str(entity_a.id), str(entity_b.id), str(entity_c.id)],
                        )

                    await session.execute_write(_cleanup)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_retire_empty_input_returns_zero(self) -> None:
        """Calling retire_orphaned_entities_batch with an empty list returns 0."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        try:
            count = await backend.retire_orphaned_entities_batch([])
            assert count == 0
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_retire_nonexistent_entity_returns_zero(self) -> None:
        """Retiring a non-existent entity ID returns 0 (MATCH skips silently)."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        try:
            count = await backend.retire_orphaned_entities_batch(
                [
                    {
                        "current_id": str(uuid4()),
                        "snapshot_id": str(uuid4()),
                        "namespace_id": str(uuid4()),
                        "retired_at": datetime.now(UTC).isoformat(),
                    }
                ]
            )
            assert count == 0
        finally:
            await backend.disconnect()

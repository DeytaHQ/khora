"""Real-Neo4j integration tests for ``Neo4jBackend.retire_orphaned_entities_batch`` (DYT-2668).

These tests exercise the full retire → snapshot → temporal-query path against
a running Neo4j instance, verifying that :EntityVersion nodes, [:SUPERSEDES]
edges, and temporal property stamps work end-to-end through the driver.

Gated by ``NEO4J_INTEGRATION_TEST=1`` (CI does not provision Neo4j).

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

from khora.core.models.entity import Entity
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

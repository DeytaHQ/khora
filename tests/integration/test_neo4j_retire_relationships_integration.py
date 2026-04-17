"""Real-Neo4j integration test for ``Neo4jBackend.retire_orphaned_relationships_batch`` (DYT-2669).

This test exercises the full Cypher retire path against a running Neo4j
instance, verifying that sole-sourced relationships are soft-retired
(valid_until stamped) while multi-sourced or wrong-doc relationships
are left untouched.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_retire_relationships_integration.py -v

Connection parameters are read from env vars with sensible defaults that
match the ``make dev`` compose stack:

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity, Relationship
from khora.storage.backends.neo4j import Neo4jBackend


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestNeo4jRetireOrphanedRelationshipsBatch:
    """End-to-end tests for retire_orphaned_relationships_batch against a real Neo4j."""

    @pytest.mark.asyncio
    async def test_sole_sourced_relationship_is_retired(self) -> None:
        """A relationship with exactly one source doc matching old_doc_id is retired."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_id = uuid4()
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
            source_document_ids=[doc_id],
            confidence=0.9,
            weight=0.75,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            retired_at = datetime.now(UTC)
            count = await backend.retire_orphaned_relationships_batch(
                [
                    {
                        "relationship_id": relationship.id,
                        "old_doc_id": doc_id,
                        "retired_at": retired_at,
                    }
                ]
            )

            assert count == 1

            rel = await backend.get_relationship(relationship.id)
            assert rel is not None
            assert rel.valid_until is not None
            assert rel.valid_until == retired_at
            assert rel.updated_at == retired_at
            # Other properties must be unchanged
            assert rel.description == "alice knows bob"
            assert rel.confidence == 0.9
            assert rel.weight == 0.75
            assert len(rel.source_document_ids) == 1
        finally:
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
    async def test_multi_sourced_relationship_not_retired(self) -> None:
        """A relationship with multiple source docs is NOT retired."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_id = uuid4()
        other_doc_id = uuid4()
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
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="WORKS_WITH",
            description="carol works with dave",
            source_document_ids=[doc_id, other_doc_id],
            confidence=0.8,
            weight=0.6,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            retired_at = datetime.now(UTC)
            count = await backend.retire_orphaned_relationships_batch(
                [
                    {
                        "relationship_id": relationship.id,
                        "old_doc_id": doc_id,
                        "retired_at": retired_at,
                    }
                ]
            )

            assert count == 0

            rel = await backend.get_relationship(relationship.id)
            assert rel is not None
            assert rel.valid_until is None
        finally:
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
    async def test_wrong_doc_id_not_retired(self) -> None:
        """A sole-sourced relationship is NOT retired when old_doc_id doesn't match."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_id = uuid4()
        wrong_doc_id = uuid4()
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
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="MANAGES",
            description="eve manages frank",
            source_document_ids=[doc_id],
            confidence=0.95,
            weight=0.5,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            retired_at = datetime.now(UTC)
            count = await backend.retire_orphaned_relationships_batch(
                [
                    {
                        "relationship_id": relationship.id,
                        "old_doc_id": wrong_doc_id,
                        "retired_at": retired_at,
                    }
                ]
            )

            assert count == 0

            rel = await backend.get_relationship(relationship.id)
            assert rel is not None
            assert rel.valid_until is None
        finally:
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
    async def test_empty_batch_returns_zero(self) -> None:
        """An empty batch should return 0 without touching Neo4j."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        try:
            count = await backend.retire_orphaned_relationships_batch([])
            assert count == 0
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_nonexistent_relationship_id_returns_zero(self) -> None:
        """A non-existent relationship ID should silently return 0."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        try:
            count = await backend.retire_orphaned_relationships_batch(
                [
                    {
                        "relationship_id": uuid4(),
                        "old_doc_id": uuid4(),
                        "retired_at": datetime.now(UTC),
                    }
                ]
            )
            assert count == 0
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_already_retired_relationship_is_restamped(self) -> None:
        """Re-retiring an already-retired relationship updates the timestamp."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_id = uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"grace-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Grace",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"heidi-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="Heidi",
        )
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="MENTORS",
            description="grace mentors heidi",
            source_document_ids=[doc_id],
            confidence=0.85,
            weight=0.7,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            # First retirement
            first_retired_at = datetime.now(UTC)
            count1 = await backend.retire_orphaned_relationships_batch(
                [
                    {
                        "relationship_id": relationship.id,
                        "old_doc_id": doc_id,
                        "retired_at": first_retired_at,
                    }
                ]
            )
            assert count1 == 1

            # Second retirement with later timestamp
            second_retired_at = datetime.now(UTC)
            count2 = await backend.retire_orphaned_relationships_batch(
                [
                    {
                        "relationship_id": relationship.id,
                        "old_doc_id": doc_id,
                        "retired_at": second_retired_at,
                    }
                ]
            )
            assert count2 == 1

            rel = await backend.get_relationship(relationship.id)
            assert rel is not None
            assert rel.valid_until == second_retired_at
        finally:
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

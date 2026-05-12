"""Real-Neo4j integration tests for ``Neo4jBackend.remap_source_document_ids_batch`` (DYT-2670).

Verifies that source_document_ids arrays on entities and relationships are
correctly remapped (old doc UUID swapped for new) against a running Neo4j
instance.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_remap_source_document_ids_integration.py -v

Connection parameters are read from env vars with sensible defaults that
match the ``make dev`` compose stack:

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)
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
class TestNeo4jRemapSourceDocumentIdsIntegration:
    """End-to-end tests for remap_source_document_ids_batch against a real Neo4j."""

    @pytest.mark.asyncio
    async def test_entity_remap_happy_path(self) -> None:
        """Remap replaces old_doc_id with new_doc_id in entity source_document_ids."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="test entity",
            source_document_ids=[doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[
                    {
                        "entity_id": str(entity.id),
                        "old_doc_id": str(doc_a),
                        "new_doc_id": str(doc_c),
                    }
                ],
                relationship_survivors=[],
            )

            got = await backend.get_entity(entity.id)
            assert got is not None
            doc_ids = got.source_document_ids
            assert len(doc_ids) == 2
            assert doc_b in doc_ids
            assert doc_c in doc_ids
            assert doc_a not in doc_ids
        finally:
            try:
                await backend.delete_entity(entity.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_entity_remap_old_doc_not_in_array(self) -> None:
        """Remap with old_doc_id absent from array leaves source_document_ids unchanged."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_x = uuid4(), uuid4(), uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="test entity",
            source_document_ids=[doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[
                    {
                        "entity_id": str(entity.id),
                        "old_doc_id": str(doc_x),
                        "new_doc_id": str(uuid4()),
                    }
                ],
                relationship_survivors=[],
            )

            got = await backend.get_entity(entity.id)
            assert got is not None
            doc_ids = got.source_document_ids
            assert len(doc_ids) == 2
            assert doc_a in doc_ids
            assert doc_b in doc_ids
        finally:
            try:
                await backend.delete_entity(entity.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_entity_remap_deduplicates_old_doc_id(self) -> None:
        """Remap removes all occurrences of old_doc_id and appends one new_doc_id."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="test entity",
            source_document_ids=[doc_a, doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[
                    {
                        "entity_id": str(entity.id),
                        "old_doc_id": str(doc_a),
                        "new_doc_id": str(doc_c),
                    }
                ],
                relationship_survivors=[],
            )

            got = await backend.get_entity(entity.id)
            assert got is not None
            doc_ids = got.source_document_ids
            assert len(doc_ids) == 2
            assert doc_b in doc_ids
            assert doc_c in doc_ids
            assert doc_a not in doc_ids
        finally:
            try:
                await backend.delete_entity(entity.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_entity_remap_multiple_survivors_in_batch(self) -> None:
        """Remap handles multiple survivors in a single batch call."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_c, doc_d, doc_e, doc_f = (
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
        )
        entity_1 = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="test entity 1",
            source_document_ids=[doc_a, doc_b],
        )
        entity_2 = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="test entity 2",
            source_document_ids=[doc_c, doc_d],
        )

        try:
            await backend.create_entity(entity_1)
            await backend.create_entity(entity_2)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[
                    {
                        "entity_id": str(entity_1.id),
                        "old_doc_id": str(doc_a),
                        "new_doc_id": str(doc_e),
                    },
                    {
                        "entity_id": str(entity_2.id),
                        "old_doc_id": str(doc_c),
                        "new_doc_id": str(doc_f),
                    },
                ],
                relationship_survivors=[],
            )

            got_1 = await backend.get_entity(entity_1.id)
            assert got_1 is not None
            doc_ids_1 = got_1.source_document_ids
            assert len(doc_ids_1) == 2
            assert doc_b in doc_ids_1
            assert doc_e in doc_ids_1

            got_2 = await backend.get_entity(entity_2.id)
            assert got_2 is not None
            doc_ids_2 = got_2.source_document_ids
            assert len(doc_ids_2) == 2
            assert doc_d in doc_ids_2
            assert doc_f in doc_ids_2
        finally:
            try:
                await backend.delete_entity(entity_1.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity_2.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_relationship_remap_happy_path(self) -> None:
        """Remap replaces old_doc_id with new_doc_id in relationship source_document_ids."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="entity a",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="entity b",
        )
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
            description="a knows b",
            source_document_ids=[doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[],
                relationship_survivors=[
                    {
                        "relationship_id": str(relationship.id),
                        "old_doc_id": str(doc_a),
                        "new_doc_id": str(doc_c),
                    }
                ],
            )

            got = await backend.get_relationship(relationship.id)
            assert got is not None
            doc_ids = got.source_document_ids
            assert len(doc_ids) == 2
            assert doc_b in doc_ids
            assert doc_c in doc_ids
            assert doc_a not in doc_ids
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
    async def test_relationship_remap_old_doc_not_in_array(self) -> None:
        """Remap with old_doc_id absent from array leaves relationship source_document_ids unchanged."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_x = uuid4(), uuid4(), uuid4()
        entity_a = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="entity a",
        )
        entity_b = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            description="entity b",
        )
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            relationship_type="KNOWS",
            description="a knows b",
            source_document_ids=[doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[],
                relationship_survivors=[
                    {
                        "relationship_id": str(relationship.id),
                        "old_doc_id": str(doc_x),
                        "new_doc_id": str(uuid4()),
                    }
                ],
            )

            got = await backend.get_relationship(relationship.id)
            assert got is not None
            doc_ids = got.source_document_ids
            assert len(doc_ids) == 2
            assert doc_a in doc_ids
            assert doc_b in doc_ids
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
    async def test_entity_remap_idempotent_on_retry(self) -> None:
        """Running the same remap twice must not duplicate new_doc_id.

        Regression: the self-heal path can retry the replace lifecycle,
        so the remap Cypher must be safe to re-apply.
        """
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            source_document_ids=[doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity)
            row = {
                "entity_id": str(entity.id),
                "old_doc_id": str(doc_a),
                "new_doc_id": str(doc_c),
            }

            # First application: swap doc_a → doc_c
            await backend.remap_source_document_ids_batch(entity_survivors=[row], relationship_survivors=[])
            # Retry the same row — caller payload is unchanged on self-heal
            await backend.remap_source_document_ids_batch(entity_survivors=[row], relationship_survivors=[])

            got = await backend.get_entity(entity.id)
            assert got is not None
            doc_ids = got.source_document_ids
            # No duplicates; old is gone; new appears exactly once
            assert len(doc_ids) == 2
            assert doc_ids.count(doc_c) == 1
            assert doc_b in doc_ids
            assert doc_a not in doc_ids
        finally:
            try:
                await backend.delete_entity(entity.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_entity_remap_tolerates_mixed_duplicates_in_array(self) -> None:
        """Array like [old, old, new] must collapse to [new], not [new, new]."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_c = uuid4(), uuid4()
        # Pre-existing mixed state: old_doc_id twice + new_doc_id once.
        entity = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
            source_document_ids=[doc_a, doc_a, doc_c],
        )

        try:
            await backend.create_entity(entity)
            await backend.remap_source_document_ids_batch(
                entity_survivors=[
                    {
                        "entity_id": str(entity.id),
                        "old_doc_id": str(doc_a),
                        "new_doc_id": str(doc_c),
                    }
                ],
                relationship_survivors=[],
            )

            got = await backend.get_entity(entity.id)
            assert got is not None
            doc_ids = got.source_document_ids
            # All occurrences of old removed, new_doc_id stays exactly once
            assert len(doc_ids) == 1
            assert doc_ids.count(doc_c) == 1
            assert doc_a not in doc_ids
        finally:
            try:
                await backend.delete_entity(entity.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_relationship_remap_self_loop_not_double_applied(self) -> None:
        """Self-loop relationships must only get remapped once despite undirected MATCH."""
        url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        backend = Neo4jBackend(url, user=user, password=password)
        await backend.connect()

        namespace_id = uuid4()
        doc_a, doc_b, doc_c = uuid4(), uuid4(), uuid4()
        entity = Entity(
            namespace_id=namespace_id,
            name=f"entity-{uuid4().hex[:8]}",
            entity_type="PERSON",
        )
        # Self-loop: src == tgt. The `()-[rel]-()` match would return the
        # same edge twice without DISTINCT.
        relationship = Relationship(
            namespace_id=namespace_id,
            source_entity_id=entity.id,
            target_entity_id=entity.id,
            relationship_type="REFERS_TO_SELF",
            source_document_ids=[doc_a, doc_b],
        )

        try:
            await backend.create_entity(entity)
            await backend.create_relationship(relationship)

            await backend.remap_source_document_ids_batch(
                entity_survivors=[],
                relationship_survivors=[
                    {
                        "relationship_id": str(relationship.id),
                        "old_doc_id": str(doc_a),
                        "new_doc_id": str(doc_c),
                    }
                ],
            )

            got = await backend.get_relationship(relationship.id)
            assert got is not None
            doc_ids = got.source_document_ids
            # No duplicate new_doc_id despite self-loop double-match
            assert len(doc_ids) == 2
            assert doc_ids.count(doc_c) == 1
            assert doc_b in doc_ids
            assert doc_a not in doc_ids
        finally:
            try:
                await backend.delete_relationship(relationship.id)
            except Exception:  # noqa: BLE001
                pass
            try:
                await backend.delete_entity(entity.id)
            except Exception:  # noqa: BLE001
                pass
            await backend.disconnect()

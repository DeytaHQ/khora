"""StorageCoordinator public read methods must accept the stable namespace_id.

``memory_namespaces`` carries two UUIDs: ``id`` (the row PK every child FK and
the Neo4j ``(:Entity {namespace_id})`` store reference) and ``namespace_id``
(the *stable* id ``create_namespace`` returns and ``MemoryNamespace`` documents
as the external identifier). ``kb.storage`` is a public property, and the
stable id is the documented value to use everywhere external.

The high-level facade resolves stable -> row internally before touching the
coordinator, but the coordinator's own public read methods forwarded the id
unchanged to the backend. Calling ``kb.storage.list_relationships(stable_id)``
therefore queried the graph on the stable id, matched zero nodes, and returned
an empty list (silent empty) while the row id worked. This test plants data
under the row id and asserts the stable id returns the *same* result.

Gated by ``NEO4J_INTEGRATION_TEST=1``; the CI integration job sets that flag.

How to run locally::

    make dev  # postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_coordinator_resolve_stable_namespace_integration.py -v

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_NEO4J_URL          (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME     (default: neo4j)
    KHORA_NEO4J_PASSWORD     (default: password)
    KHORA_DATABASE_URL       (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from khora.core.models import Entity, MemoryNamespace, Relationship
from khora.storage.backends.neo4j import Neo4jBackend
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator

EMBED_DIM = 1536


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real backends (requires make dev)",
)
class TestCoordinatorResolveStableNamespaceIntegration:
    """Coordinator read methods resolve the stable namespace_id internally."""

    @pytest.fixture
    async def coord(self) -> AsyncIterator[StorageCoordinator]:
        database_url = os.environ.get(
            "KHORA_DATABASE_URL",
            "postgresql+asyncpg://khora:khora@localhost:5432/khora",
        )
        neo4j_url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

        rel = PostgreSQLBackend(database_url=database_url)
        vec = PgVectorBackend(database_url=database_url, embedding_dimension=EMBED_DIM)
        graph = Neo4jBackend(neo4j_url, user=neo4j_user, password=neo4j_password)

        coord = StorageCoordinator(relational=rel, vector=vec, graph=graph)
        await coord.connect()
        try:
            yield coord
        finally:
            await coord.disconnect()

    @pytest.mark.asyncio
    async def test_reads_accept_stable_namespace_id(self, coord: StorageCoordinator) -> None:
        ns = await coord.create_namespace(MemoryNamespace())
        stable_id = ns.namespace_id
        row_id = await coord.resolve_namespace(stable_id)
        # The bug only exists when the two ids actually differ.
        assert stable_id != row_id

        marker = uuid4().hex[:6]
        alice = Entity(namespace_id=row_id, name=f"alice-{marker}", entity_type="PERSON")
        carol = Entity(namespace_id=row_id, name=f"carol-{marker}", entity_type="PERSON")
        await coord.upsert_entities_batch(row_id, [alice, carol])
        rel = Relationship(
            namespace_id=row_id,
            source_entity_id=alice.id,
            target_entity_id=carol.id,
            relationship_type="KNOWS",
            description="in-namespace edge",
        )
        await coord.create_relationships_batch([rel])

        try:
            # Baseline: the row id works (sanity check on the planted data).
            entities_by_row = await coord.list_entities(row_id)
            rels_by_row = await coord.list_relationships(row_id)
            assert {e.id for e in entities_by_row} >= {alice.id, carol.id}
            assert rel.id in {r.id for r in rels_by_row}

            # The fix: the *stable* id must return the same data, not [].
            entities_by_stable = await coord.list_entities(stable_id)
            rels_by_stable = await coord.list_relationships(stable_id)
            assert {e.id for e in entities_by_stable} == {e.id for e in entities_by_row}
            assert {r.id for r in rels_by_stable} == {r.id for r in rels_by_row}

            # Counts must match across the two ids too.
            assert await coord.count_entities(stable_id) == await coord.count_entities(row_id)
            assert await coord.count_relationships(stable_id) == await coord.count_relationships(row_id)

            # Per-entity / per-relationship helpers honour the stable id.
            got_entity = await coord.get_entity(alice.id, namespace_id=stable_id)
            assert got_entity is not None and got_entity.id == alice.id
            got_rel = await coord.get_relationship(rel.id, namespace_id=stable_id)
            assert got_rel is not None and got_rel.id == rel.id
            edges = await coord.get_entity_relationships(alice.id, namespace_id=stable_id)
            assert rel.id in {r.id for r in edges}
        finally:
            await coord.delete_entity(alice.id, namespace_id=row_id)
            await coord.delete_entity(carol.id, namespace_id=row_id)

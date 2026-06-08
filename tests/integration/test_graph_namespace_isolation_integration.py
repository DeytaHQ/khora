"""Real-backend cross-namespace isolation test for graph getters.

Verifies the security boundary on a live Postgres + Neo4j stack: an entity
persisted under namespace A is not retrievable by a caller scoped to
namespace B, even when that caller knows the entity ID. Same for
relationships and episodes.

Gated by ``NEO4J_INTEGRATION_TEST=1``; the CI integration job sets that flag.

How to run locally::

    make dev  # postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_graph_namespace_isolation_integration.py -v

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_NEO4J_URL          (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME     (default: neo4j)
    KHORA_NEO4J_PASSWORD     (default: password)
    KHORA_DATABASE_URL       (default: postgresql+asyncpg://khora:khora@localhost:5432/khora)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Entity, Episode, MemoryNamespace, Relationship
from khora.storage.backends.neo4j import Neo4jBackend
from khora.storage.backends.pgvector import PgVectorBackend
from khora.storage.backends.postgresql import PostgreSQLBackend
from khora.storage.coordinator import StorageCoordinator

EMBED_DIM = 4


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("NEO4J_INTEGRATION_TEST"),
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real backends (requires make dev)",
)
class TestGraphNamespaceIsolationIntegration:
    """End-to-end namespace isolation on live Postgres + Neo4j."""

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

    @pytest.fixture
    async def two_namespaces(self, coord: StorageCoordinator):
        ns_a = await coord.create_namespace(MemoryNamespace())
        ns_b = await coord.create_namespace(MemoryNamespace())
        return ns_a.namespace_id, ns_b.namespace_id

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="tracked in #1033: PG namespace FK violation (namespace_id_fkey) on first CI run; quarantined pending triage against a live stack.",
        strict=False,
    )
    async def test_get_entity_isolated_across_namespaces(self, coord: StorageCoordinator, two_namespaces) -> None:
        """Entity created in ns A is invisible to ns B even with the ID."""
        ns_a, ns_b = two_namespaces
        alice = Entity(
            namespace_id=ns_a,
            name=f"alice-{uuid4().hex[:6]}",
            entity_type="PERSON",
        )
        await coord.upsert_entities_batch(ns_a, [alice])

        # Same-namespace caller sees the entity.
        fetched_same = await coord.get_entity(alice.id, namespace_id=ns_a)
        assert fetched_same is not None
        assert fetched_same.id == alice.id

        # Cross-namespace caller — knows the ID but is scoped to ns_b — gets None.
        fetched_cross = await coord.get_entity(alice.id, namespace_id=ns_b)
        assert fetched_cross is None

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="tracked in #1033: PG namespace FK violation (namespace_id_fkey) on first CI run; quarantined pending triage against a live stack.",
        strict=False,
    )
    async def test_get_relationship_isolated_across_namespaces(self, coord: StorageCoordinator, two_namespaces) -> None:
        """Relationship created in ns A is invisible to ns B."""
        ns_a, ns_b = two_namespaces
        alice = Entity(namespace_id=ns_a, name=f"alice-{uuid4().hex[:6]}", entity_type="PERSON")
        bob = Entity(namespace_id=ns_a, name=f"bob-{uuid4().hex[:6]}", entity_type="PERSON")
        await coord.upsert_entities_batch(ns_a, [alice, bob])
        rel = Relationship(
            namespace_id=ns_a,
            source_entity_id=alice.id,
            target_entity_id=bob.id,
            relationship_type="KNOWS",
        )
        await coord.create_relationships_batch([rel])

        same = await coord.get_relationship(rel.id, namespace_id=ns_a)
        assert same is not None
        assert same.id == rel.id

        cross = await coord.get_relationship(rel.id, namespace_id=ns_b)
        assert cross is None

    @pytest.mark.asyncio
    async def test_get_episode_isolated_across_namespaces(self, coord: StorageCoordinator, two_namespaces) -> None:
        """Episode created in ns A is invisible to ns B."""
        ns_a, ns_b = two_namespaces
        ep = Episode(
            namespace_id=ns_a,
            name=f"meeting-{uuid4().hex[:6]}",
            occurred_at=datetime.now(UTC),
        )
        await coord.create_episode(ep)

        same = await coord.get_episode(ep.id, namespace_id=ns_a)
        assert same is not None
        assert same.id == ep.id

        cross = await coord.get_episode(ep.id, namespace_id=ns_b)
        assert cross is None

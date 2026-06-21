"""Live pg+Neo4j projection of materialized communities onto recall (#1308).

The query-time half of #1276: after the VectorCypher engine assembles a recall's
result entities, it fetches the materialized dream :Community summaries those
entities belong to via the live ``get_entity_communities`` reader and surfaces
them on ``RecallResult.communities`` (de-duped + capped).

This drives the engine's ``_project_communities`` against the REAL Neo4j reader
(real ``HAS_MEMBER`` Cypher), so it exercises engine -> coordinator -> Neo4j.

With materialized communities: a projection over a member entity surfaces the
community summary. Without: the projection is empty (zero added cost - the reader
returns [] for an unmaterialized namespace).

How to run locally::

    make dev   # starts postgres (5434) + neo4j (7688) via compose
    KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_USERNAME=neo4j KHORA_NEO4J_PASSWORD=pleaseletmein \\
    NEO4J_INTEGRATION_TEST=1 \\
        uv run pytest tests/integration/dream/test_neo4j_community_recall_projection.py -v
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import UUID, uuid4

import pytest

from khora.config.schema import KhoraConfig
from khora.core.models.entity import CommunityNode, Entity
from khora.core.models.recall import RecallEntity
from khora.core.models.tenancy import MemoryNamespace
from khora.engines.vectorcypher.engine import VectorCypherEngine

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")
NEO4J_USER = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")


def _reachable(url: str, default_port: int) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_reachable(DATABASE_URL, 5432) and _reachable(NEO4J_URL, 7687)),
        reason="pg+neo4j not reachable (run `make dev`)",
    ),
]

EMBED_DIM = 4


@pytest.fixture
async def engine() -> AsyncIterator[VectorCypherEngine]:
    config = KhoraConfig(database_url=DATABASE_URL, neo4j_url=NEO4J_URL)
    config.storage.neo4j_user = NEO4J_USER
    config.storage.neo4j_password = NEO4J_PASSWORD
    config.llm.embedding_dimension = EMBED_DIM
    config.storage.embedding_dimension = EMBED_DIM
    eng = VectorCypherEngine(config)
    await eng.connect()
    try:
        yield eng
    finally:
        await eng.disconnect()


def _graph_backend(engine: VectorCypherEngine):
    graph = engine._storage.graph
    return getattr(graph, "_backend", graph)


async def _seed_entity_both(engine: VectorCypherEngine, ns_row_id: UUID, name: str) -> UUID:
    """Seed one entity into PG + Neo4j with a matching id (vector-first then graph)."""
    ent = Entity(namespace_id=ns_row_id, name=name, entity_type="PERSON", description=name)
    await engine._storage.create_entity(ent)
    return ent.id


def _recall_entity(entity_id: UUID) -> RecallEntity:
    return RecallEntity(
        id=entity_id,
        name="member",
        entity_type="PERSON",
        description="",
        score=0.9,
        attributes={},
        mention_count=1,
        source_document_ids=[],
        source_chunk_ids=[],
    )


@pytest.mark.asyncio
async def test_recall_projects_materialized_community(engine: VectorCypherEngine) -> None:
    """A projection over a member entity surfaces the community summary (#1308)."""
    coordinator = engine._storage
    ns = await coordinator.create_namespace(MemoryNamespace())
    ns_row_id = await coordinator.resolve_namespace(ns.namespace_id)

    member = await _seed_entity_both(engine, ns_row_id, f"alice-{uuid4().hex[:8]}")
    other = await _seed_entity_both(engine, ns_row_id, f"bob-{uuid4().hex[:8]}")

    community = CommunityNode(
        id=uuid4(),
        namespace_id=ns_row_id,
        summary="alice and bob collaborate on the project",
        member_ids=[member, other],
        summary_depth=1,
    )
    count = await _graph_backend(engine).materialize_communities_batch(
        [community], namespace_id=ns_row_id, materialized_at=datetime.now(UTC)
    )
    assert count == 1

    # Drive the engine's projection over a recall hit touching the member entity.
    degradations: list = []
    communities = await engine._project_communities(
        [_recall_entity(member)], namespace_id=ns_row_id, degradations=degradations
    )

    assert [c.id for c in communities] == [community.id]
    assert communities[0].summary == "alice and bob collaborate on the project"
    assert set(communities[0].member_ids) >= {member, other}
    assert degradations == []


@pytest.mark.asyncio
async def test_recall_without_materialized_community_is_empty(engine: VectorCypherEngine) -> None:
    """A namespace with no materialized communities projects empty (zero added cost)."""
    coordinator = engine._storage
    ns = await coordinator.create_namespace(MemoryNamespace())
    ns_row_id = await coordinator.resolve_namespace(ns.namespace_id)

    member = await _seed_entity_both(engine, ns_row_id, f"carol-{uuid4().hex[:8]}")

    degradations: list = []
    communities = await engine._project_communities(
        [_recall_entity(member)], namespace_id=ns_row_id, degradations=degradations
    )

    assert communities == []
    assert degradations == []

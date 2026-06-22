"""Regression tests for #1329 — two cross-backend relationship MERGE bugs.

Part 1 — namespace-scoped endpoint matching:
  A relationship whose endpoint UUIDs belong to namespace A must NOT create an
  edge when submitted under namespace B, even when those UUIDs happen to match
  nodes that exist in namespace A.

Part 2 — intra-batch duplicate dedup:
  A batch with duplicate (namespace_id, source, target, type) entries must
  return exactly one tuple per stored edge and fire the hook once.

Backends tested:
  - Neo4j  (gated by NEO4J_INTEGRATION_TEST=1, bolt://localhost:7688)
  - Memgraph (gated by reachability at bolt://localhost:7689)

How to run locally::

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 \\
    KHORA_DATABASE_URL=postgresql+asyncpg://khora:khora@localhost:5434/khora \\
    KHORA_NEO4J_URL=bolt://localhost:7688 \\
    KHORA_NEO4J_PASSWORD=pleaseletmein \\
        uv run pytest tests/integration/test_1329_relationship_merge_namespace_scope.py -v

    # For memgraph: also start memgraph and ensure bolt://localhost:7689 is up
    docker compose up -d memgraph
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity, Relationship

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Neo4j gate
# ---------------------------------------------------------------------------
_NEO4J_INTEGRATION = bool(os.environ.get("NEO4J_INTEGRATION_TEST"))
_NEO4J_URL = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7688")
_NEO4J_USER = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
_NEO4J_PASSWORD = os.environ.get("KHORA_NEO4J_PASSWORD", "pleaseletmein")

# ---------------------------------------------------------------------------
# Memgraph gate (reachability probe — no env flag needed)
# ---------------------------------------------------------------------------
_MEMGRAPH_URL = os.environ.get("KHORA_MEMGRAPH_URL", "bolt://localhost:7689")
_MEMGRAPH_USER = os.environ.get("KHORA_MEMGRAPH_USERNAME", "memgraph")
_MEMGRAPH_PASSWORD = os.environ.get("KHORA_MEMGRAPH_PASSWORD", "")


def _reachable(bolt_url: str, default_port: int = 7687) -> bool:
    parsed = urlparse(bolt_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


_MEMGRAPH_UP = _reachable(_MEMGRAPH_URL)


# ===========================================================================
# Neo4j tests
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not _NEO4J_INTEGRATION,
    reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
)
class TestNeo4jRelationshipMergeNamespaceScope:
    """Part 1 and Part 2 regression tests on a live Neo4j backend."""

    @pytest.mark.asyncio
    async def test_cross_namespace_endpoint_does_not_create_edge(self) -> None:
        """Part 1: a relationship whose endpoint UUIDs belong to ns_a must NOT
        create an edge when the relationship row's namespace_id is ns_b.

        Before the fix the source/target MATCH ignored namespace_id, so the
        endpoint nodes were found by UUID alone. The edge was then written with
        namespace_id=ns_b attached to ns_a nodes — silent cross-tenant leak.
        After the fix the MATCH also filters namespace_id so no nodes are found
        and the MERGE never fires.
        """
        from khora.storage.backends.neo4j import Neo4jBackend

        backend = Neo4jBackend(_NEO4J_URL, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
        await backend.connect()

        ns_a = uuid4()
        ns_b = uuid4()

        # Two entities that live in ns_a.
        alice = Entity(namespace_id=ns_a, name=f"alice-{uuid4().hex[:6]}", entity_type="PERSON")
        bob = Entity(namespace_id=ns_a, name=f"bob-{uuid4().hex[:6]}", entity_type="PERSON")

        try:
            await backend.create_entity(alice)
            await backend.create_entity(bob)

            # Attempt to create a relationship between alice and bob but scoped
            # to ns_b.  The endpoint ids are from ns_a nodes that ns_b does not own.
            cross_rel = Relationship(
                namespace_id=ns_b,
                source_entity_id=alice.id,
                target_entity_id=bob.id,
                relationship_type="KNOWS",
                confidence=0.9,
            )
            results = await backend.create_relationships_batch([cross_rel])
            # The MATCH must fail (ns_b has no nodes with those ids), so the MERGE
            # never fires and the result list must be empty.
            assert results == [], (
                "Expected 0 results: the endpoint nodes belong to ns_a, not ns_b. "
                f"Got {len(results)} result(s) — cross-tenant edge was created."
            )
        finally:
            async with backend._session() as session:
                await session.run(
                    "MATCH (n:Entity {namespace_id: $ns}) DETACH DELETE n",
                    ns=str(ns_a),
                )
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_intra_batch_duplicate_relationships_single_result(self) -> None:
        """Part 2: submitting two identical (ns, source, target, type) relationships
        in one batch returns exactly one tuple and fires the hook once.

        Before the fix the return loop iterated over ALL input relationships,
        so a duplicate pair would yield two tuples pointing at the same stored
        edge, causing downstream hooks to fire twice.
        """
        from khora.storage.backends.neo4j import Neo4jBackend

        backend = Neo4jBackend(_NEO4J_URL, user=_NEO4J_USER, password=_NEO4J_PASSWORD)
        await backend.connect()

        ns = uuid4()
        a = Entity(namespace_id=ns, name=f"acme-{uuid4().hex[:6]}", entity_type="ORGANIZATION")
        b = Entity(namespace_id=ns, name=f"beta-{uuid4().hex[:6]}", entity_type="ORGANIZATION")

        try:
            await backend.create_entity(a)
            await backend.create_entity(b)

            rel1 = Relationship(
                namespace_id=ns,
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="PARTNER_OF",
                confidence=0.8,
            )
            rel2 = Relationship(
                namespace_id=ns,
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="PARTNER_OF",
                confidence=0.9,
            )
            # rel1 and rel2 share the same MERGE key: same ns/source/target/type.
            results = await backend.create_relationships_batch([rel1, rel2])
            assert len(results) == 1, (
                f"Expected 1 result for 2 duplicate edges; got {len(results)}. "
                "Hook would have fired twice before the fix."
            )
            _rel_out, is_new = results[0]
            assert is_new is True
        finally:
            async with backend._session() as session:
                await session.run(
                    "MATCH (n:Entity {namespace_id: $ns}) DETACH DELETE n",
                    ns=str(ns),
                )
            await backend.disconnect()


# ===========================================================================
# Memgraph tests
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not _MEMGRAPH_UP,
    reason="Memgraph not reachable at bolt://localhost:7689 (run `docker compose up -d memgraph`)",
)
class TestMemgraphRelationshipMergeNamespaceScope:
    """Part 1 and Part 2 regression tests on a live Memgraph backend."""

    @pytest.mark.asyncio
    async def test_cross_namespace_endpoint_does_not_create_edge(self) -> None:
        """Part 1: cross-namespace endpoint match must not produce an edge."""
        from khora.storage.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend(_MEMGRAPH_URL, user=_MEMGRAPH_USER, password=_MEMGRAPH_PASSWORD)
        await backend.connect()

        ns_a = uuid4()
        ns_b = uuid4()

        alice = Entity(namespace_id=ns_a, name=f"alice-{uuid4().hex[:6]}", entity_type="PERSON")
        bob = Entity(namespace_id=ns_a, name=f"bob-{uuid4().hex[:6]}", entity_type="PERSON")

        try:
            await backend.create_entity(alice)
            await backend.create_entity(bob)

            cross_rel = Relationship(
                namespace_id=ns_b,
                source_entity_id=alice.id,
                target_entity_id=bob.id,
                relationship_type="KNOWS",
                confidence=0.9,
            )
            results = await backend.create_relationships_batch([cross_rel])
            assert results == [], (
                "Expected 0 results: endpoint nodes belong to ns_a, not ns_b. "
                f"Got {len(results)} result(s) — cross-tenant edge was created."
            )
        finally:
            driver = backend._get_driver()
            async with driver.session() as session:
                await session.run(
                    "MATCH (n:Entity {namespace_id: $ns}) DETACH DELETE n",
                    ns=str(ns_a),
                )
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_intra_batch_duplicate_relationships_single_result(self) -> None:
        """Part 2: duplicate (ns, source, target, type) in one batch returns 1 tuple."""
        from khora.storage.backends.memgraph import MemgraphBackend

        backend = MemgraphBackend(_MEMGRAPH_URL, user=_MEMGRAPH_USER, password=_MEMGRAPH_PASSWORD)
        await backend.connect()

        ns = uuid4()
        a = Entity(namespace_id=ns, name=f"acme-{uuid4().hex[:6]}", entity_type="ORGANIZATION")
        b = Entity(namespace_id=ns, name=f"beta-{uuid4().hex[:6]}", entity_type="ORGANIZATION")

        try:
            await backend.create_entity(a)
            await backend.create_entity(b)

            rel1 = Relationship(
                namespace_id=ns,
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="PARTNER_OF",
                confidence=0.8,
            )
            rel2 = Relationship(
                namespace_id=ns,
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="PARTNER_OF",
                confidence=0.9,
            )
            results = await backend.create_relationships_batch([rel1, rel2])
            assert len(results) == 1, (
                f"Expected 1 result for 2 duplicate edges; got {len(results)}. "
                "Hook would have fired twice before the fix."
            )
            _rel_out, is_new = results[0]
            assert is_new is True
        finally:
            driver = backend._get_driver()
            async with driver.session() as session:
                await session.run(
                    "MATCH (n:Entity {namespace_id: $ns}) DETACH DELETE n",
                    ns=str(ns),
                )
            await backend.disconnect()

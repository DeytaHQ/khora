"""Real-Neo4j integration test for ``Neo4jBackend.get_entity_relationships``.

This test exercises the full Cypher → driver → ``result.data()`` →
``_record_to_relationship`` serialization path against a running Neo4j
instance. Unlike the mock-based unit tests in
``tests/unit/test_neo4j_backend.py`` — which stub ``.data()`` directly and
therefore can't detect a regression in the Cypher shape — this test pins
the boundary that actually matters: the driver's on-wire representation
of ``RETURN r`` (a ``neo4j.graph.Relationship`` that ``.data()``
serializes as a 3-tuple) vs ``RETURN properties(r) as r`` (a plain dict
that ``_record_to_relationship`` can index with string keys).

Why this is marked ``@pytest.mark.integration`` and gated by
``NEO4J_INTEGRATION_TEST=1``:

    Khora's CI does NOT provision a Neo4j instance. The existing
    "integration" tests in this repo are composition-level and still
    mock-based (see ``tests/integration/test_dual_nodes_timeout_integration.py``
    lines 23–33 for the full rationale). Real-Neo4j coverage lives
    behind an opt-in env var so CI stays green while local developers
    running ``make dev`` can exercise it.

How to run locally:

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_get_entity_relationships_integration.py -v

Connection parameters are read from env vars with sensible defaults that
match the ``make dev`` compose stack:

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)

The test would fail on the pre-fix code because the real driver's
``.data()`` method serializes a ``RETURN r`` value as a 3-tuple
``(start_dict, rel_type, end_dict)`` — handing
``_record_to_relationship`` a tuple instead of a dict and raising
``TypeError: tuple indices must be integers or slices, not str``. The
post-fix Cypher (``RETURN properties(r) as r``) returns a plain property
dict, which serializes cleanly.
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
class TestNeo4jGetEntityRelationshipsIntegration:
    """End-to-end regression lock for against a real Neo4j."""

    @pytest.mark.asyncio
    async def test_returns_relationship_through_real_driver(self) -> None:
        """Create 2 entities + 1 relationship, then read it back via the real driver.

        Would fail on pre-fix code because ``RETURN r`` serializes the
        ``neo4j.graph.Relationship`` as a 3-tuple through ``result.data()``,
        causing ``_record_to_relationship`` to raise ``TypeError`` when it
        indexes the tuple with a string key.
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
            properties={"since": "2024"},
            confidence=0.9,
            weight=0.75,
        )

        try:
            await backend.create_entity(entity_a)
            await backend.create_entity(entity_b)
            await backend.create_relationship(relationship)

            got = await backend.get_entity_relationships(
                entity_a.id,
                direction="outgoing",
            )

            assert isinstance(got, list)
            assert len(got) == 1
            rel = got[0]
            assert isinstance(rel, Relationship)
            assert rel.id == relationship.id
            assert rel.namespace_id == namespace_id
            assert rel.source_entity_id == entity_a.id
            assert rel.target_entity_id == entity_b.id
            assert rel.relationship_type == "KNOWS"
            assert rel.description == "alice knows bob"
            assert rel.properties == {"since": "2024"}
            assert rel.confidence == 0.9
            assert rel.weight == 0.75
        finally:
            # Best-effort cleanup; swallow errors so one failure doesn't
            # mask another.
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

"""Real-Neo4j integration test for the empty-attributes upsert guard.

On entity upsert, an incoming EMPTY (``{}``) or NULL ``attributes`` must not
clobber a stored, populated ``attributes``. A non-empty incoming value
overwrites the stored dict wholesale (no key-union merge). The guard lives in
``Neo4jBackend`` as the ``ON MATCH SET`` clause of ``_UPSERT_CYPHER``
(``upsert_entities_batch``):

    e.attributes = CASE WHEN row.attributes IS NULL OR row.attributes = '{}'
        OR row.attributes = '' THEN e.attributes ELSE row.attributes END

``attributes`` is persisted as a JSON *string* (``serialize_dict``), so ``{}``
serializes to ``'{}'`` and ``None`` serializes to a Cypher NULL — the two
branches the guard fences.

Why this is gated by ``NEO4J_INTEGRATION_TEST=1``:

    Khora's CI does NOT provision a Neo4j instance, so real-Neo4j coverage
    lives behind an opt-in env var (matching the sibling
    ``test_neo4j_get_entity_relationships_integration.py``). Local developers
    running ``make dev`` can exercise it.

How to run locally::

    make dev  # starts postgres + neo4j via docker compose
    NEO4J_INTEGRATION_TEST=1 uv run pytest \
        tests/integration/test_neo4j_entity_attributes_guard.py -v

Connection parameters are read from env vars with defaults matching the
``make dev`` compose stack::

    KHORA_NEO4J_URL       (default: bolt://localhost:7687)
    KHORA_NEO4J_USERNAME  (default: neo4j)
    KHORA_NEO4J_PASSWORD  (default: password)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from khora.core.models.entity import Entity
from khora.storage.backends.neo4j import Neo4jBackend

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("NEO4J_INTEGRATION_TEST"),
        reason="set NEO4J_INTEGRATION_TEST=1 to run against real Neo4j (requires make dev)",
    ),
]


@pytest.fixture
async def backend() -> AsyncIterator[Neo4jBackend]:
    url = os.environ.get("KHORA_NEO4J_URL", "bolt://localhost:7687")
    user = os.environ.get("KHORA_NEO4J_USERNAME", "neo4j")
    password = os.environ.get("KHORA_NEO4J_PASSWORD", "password")

    be = Neo4jBackend(url, user=user, password=password)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


def _entity(ns_id, name: str, attributes, entity_type: str = "PERSON") -> Entity:
    """Build a fresh Entity with a new candidate id — mirrors the LLM-ingestion
    shape where each extraction gets a throwaway id and storage dedupes by
    (namespace, name, entity_type).

    ``attributes`` is passed verbatim (including ``None``): the Entity
    dataclass does not coerce ``None`` -> ``{}``, so a ``None`` here serializes
    to a Cypher NULL and exercises the ``row.attributes IS NULL`` branch of the
    guard.
    """
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        attributes=attributes,
    )


@pytest.mark.asyncio
async def test_empty_attributes_does_not_overwrite_populated(backend: Neo4jBackend) -> None:
    """Upsert ``{"a": 1}`` then the same key with ``{}`` — the stored
    attributes must stay ``{"a": 1}`` (empty incoming does not clobber)."""
    ns = uuid4()
    name = f"guard-empty-{uuid4().hex[:8]}"

    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"a": 1})])
    await backend.upsert_entities_batch(ns, [_entity(ns, name, {})])

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_null_attributes_does_not_overwrite_populated(backend: Neo4jBackend) -> None:
    """Upsert ``{"a": 1}`` then the same key with NULL (``None``) attributes —
    the stored attributes must stay ``{"a": 1}``.

    Distinct from the empty-dict case: ``None`` serializes to a Cypher NULL and
    hits the ``row.attributes IS NULL`` branch rather than ``= '{}'``."""
    ns = uuid4()
    name = f"guard-null-{uuid4().hex[:8]}"

    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"a": 1})])
    await backend.upsert_entities_batch(ns, [_entity(ns, name, None)])

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_nonempty_attributes_overwrites_without_key_union(backend: Neo4jBackend) -> None:
    """Upsert ``{"a": 1}`` then the same key with ``{"b": 2}`` — the stored
    attributes must become EXACTLY ``{"b": 2}``.

    The guard only protects against empty/NULL incoming values; a non-empty
    incoming dict replaces the stored one wholesale. Key-union merge
    (``{"a": 1, "b": 2}``) is intentionally OUT OF SCOPE."""
    ns = uuid4()
    name = f"guard-overwrite-{uuid4().hex[:8]}"

    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"a": 1})])
    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"b": 2})])

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"b": 2}
    assert "a" not in got.attributes

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

This module also pins the version-chain side of the guard: a guard-preserved
(empty/NULL) re-upsert must NOT append a phantom ``:EntityVersion`` snapshot or
``[:SUPERSEDES]`` edge, while a genuine attribute change still creates exactly
one. ``upsert_entities_batch`` only snapshots when attributes actually change,
and the Phase-2 change-detector mirrors the ON MATCH SET guard by treating an
empty/NULL incoming value as "no change".

The single-entity ``update_entity`` query carries a twin of the guard; the
``test_update_entity_*`` cases below exercise that direct-update path (used by
the expansion/coordinator route), which the batch tests do not touch.

Why this is gated by ``NEO4J_INTEGRATION_TEST=1``:

    The integration job in ``.github/workflows/ci.yml`` provisions a Neo4j
    service and sets ``NEO4J_INTEGRATION_TEST=1``, so these tests run for real
    in CI. The env-var gate keeps them opt-in everywhere else: a local box
    without ``make dev`` (or any run that has not started Neo4j) skips them
    cleanly instead of erroring on an unreachable bolt port. It matches the
    sibling ``test_neo4j_get_entity_relationships_integration.py``.

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


async def _count_versions(backend: Neo4jBackend, ns_id, name: str, entity_type: str) -> tuple[int, int]:
    """Count ``:EntityVersion`` snapshots and ``[:SUPERSEDES]`` edges for one
    entity identity.

    The namespace is unique per test (fresh ``uuid4``), so matching on
    (namespace_id, name, entity_type) uniquely identifies this entity's version
    chain without needing the canonical node id. ``upsert_entities_batch``
    stamps those three properties onto every snapshot node it creates.
    """

    async def _query(tx) -> tuple[int, int]:
        ev_result = await tx.run(
            "MATCH (ev:EntityVersion {namespace_id: $ns, name: $name, entity_type: $etype}) RETURN count(ev) AS c",
            ns=str(ns_id),
            name=name,
            etype=entity_type,
        )
        ev_record = await ev_result.single()
        edge_result = await tx.run(
            "MATCH (:Entity {namespace_id: $ns, name: $name, entity_type: $etype})"
            "-[s:SUPERSEDES]->(:EntityVersion) RETURN count(s) AS c",
            ns=str(ns_id),
            name=name,
            etype=entity_type,
        )
        edge_record = await edge_result.single()
        return ev_record["c"], edge_record["c"]

    async with backend._session() as session:
        return await session.execute_read(_query)


@pytest.mark.asyncio
async def test_empty_reupsert_creates_no_phantom_version(backend: Neo4jBackend) -> None:
    """Guard-preserved re-upsert must not append a phantom version snapshot.

    Seeding ``{"a": 1}`` then re-upserting the same key with ``{}`` and ``None``
    leaves the stored attributes unchanged (the ON MATCH SET guard), so the
    Phase-2 change-detector must treat it as "no change" and create NO
    ``:EntityVersion`` node and NO ``[:SUPERSEDES]`` edge (H2 regression)."""
    ns = uuid4()
    name = f"guard-noversion-{uuid4().hex[:8]}"

    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"a": 1})])
    await backend.upsert_entities_batch(ns, [_entity(ns, name, {})])
    await backend.upsert_entities_batch(ns, [_entity(ns, name, None)])

    ev_count, supersedes_count = await _count_versions(backend, ns, name, "PERSON")
    assert ev_count == 0, f"empty/NULL re-upsert must not snapshot a version, got {ev_count}"
    assert supersedes_count == 0, f"empty/NULL re-upsert must not create a SUPERSEDES edge, got {supersedes_count}"

    # The current node still carries the seeded attributes (guard held).
    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_genuine_change_creates_single_version(backend: Neo4jBackend) -> None:
    """Positive control: a real attribute change still snapshots exactly one
    version.

    Seeding ``{"a": 1}`` then upserting ``{"b": 2}`` is a genuine change, so
    Phase-2 must create exactly one ``:EntityVersion`` node and one
    ``[:SUPERSEDES]`` edge — proving the H2 fix suppresses only the phantom
    (empty/NULL) case, not legitimate versioning."""
    ns = uuid4()
    name = f"guard-oneversion-{uuid4().hex[:8]}"

    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"a": 1})])
    await backend.upsert_entities_batch(ns, [_entity(ns, name, {"b": 2})])

    ev_count, supersedes_count = await _count_versions(backend, ns, name, "PERSON")
    assert ev_count == 1, f"genuine change must snapshot exactly one version, got {ev_count}"
    assert supersedes_count == 1, f"genuine change must create exactly one SUPERSEDES edge, got {supersedes_count}"

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"b": 2}


@pytest.mark.asyncio
async def test_update_entity_empty_attributes_does_not_overwrite_populated(backend: Neo4jBackend) -> None:
    """Single-entity path (``update_entity``): seed ``{"a": 1}`` then update the
    same node with ``{}`` — stored attributes must stay ``{"a": 1}``.

    ``update_entity`` is the direct single-node write (the expansion/coordinator
    update route) and carries its own guard, separate from ``_UPSERT_CYPHER``'s
    ``ON MATCH SET`` — so it needs coverage the batch tests don't provide."""
    ns = uuid4()
    name = f"guard-update-empty-{uuid4().hex[:8]}"

    seed = _entity(ns, name, {"a": 1})
    await backend.upsert_entities_batch(ns, [seed])

    # update_entity MATCHes by id, so target the stored node.
    update = _entity(ns, name, {})
    update.id = seed.id
    await backend.update_entity(update, namespace_id=ns)

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_update_entity_null_attributes_does_not_overwrite_populated(backend: Neo4jBackend) -> None:
    """Single-entity path: seed ``{"a": 1}`` then ``update_entity`` with NULL
    (``None``) attributes — stored attributes must stay ``{"a": 1}`` (hits the
    ``$attributes IS NULL`` branch)."""
    ns = uuid4()
    name = f"guard-update-null-{uuid4().hex[:8]}"

    seed = _entity(ns, name, {"a": 1})
    await backend.upsert_entities_batch(ns, [seed])

    update = _entity(ns, name, None)
    update.id = seed.id
    await backend.update_entity(update, namespace_id=ns)

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 1}


@pytest.mark.asyncio
async def test_update_entity_nonempty_attributes_overwrites_without_key_union(backend: Neo4jBackend) -> None:
    """Single-entity path: seed ``{"a": 1}`` then ``update_entity`` with
    ``{"b": 2}`` — stored attributes must become EXACTLY ``{"b": 2}``.

    A non-empty incoming value overwrites wholesale; key-union merge
    (``{"a": 1, "b": 2}``) is intentionally OUT OF SCOPE."""
    ns = uuid4()
    name = f"guard-update-overwrite-{uuid4().hex[:8]}"

    seed = _entity(ns, name, {"a": 1})
    await backend.upsert_entities_batch(ns, [seed])

    update = _entity(ns, name, {"b": 2})
    update.id = seed.id
    await backend.update_entity(update, namespace_id=ns)

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"b": 2}
    assert "a" not in got.attributes

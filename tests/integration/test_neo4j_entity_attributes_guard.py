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

import json
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


def _entity_with_stamp(ns_id, name: str, attributes, occurred_at: str, entity_type: str = "PERSON") -> Entity:
    """Like ``_entity`` but pins ``version_valid_from`` deterministically.

    ``_derive_version_valid_from`` resolves ``metadata["occurred_at"]`` first, so
    stamping it lets the version-chain tests assert an exact
    ``version_valid_from`` value instead of a clock-derived ``created_at`` that
    would make an "advances" assertion timing-flaky.
    """
    return Entity(
        id=uuid4(),
        namespace_id=ns_id,
        name=name,
        entity_type=entity_type,
        attributes=attributes,
        metadata={"occurred_at": occurred_at},
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


async def _current_version_valid_from(backend: Neo4jBackend, ns_id, name: str, entity_type: str) -> str | None:
    """Read ``version_valid_from`` off the current ``:Entity`` node.

    ``ON MATCH SET`` preserves the stamp via ``coalesce`` and only the Phase-2
    ``_VERSION_CYPHER`` advances it (``SET current.version_valid_from = ...``),
    so this is the property the version tests assert stays put (no-op re-upsert)
    or moves forward (genuine change).
    """

    async def _query(tx) -> str | None:
        result = await tx.run(
            "MATCH (e:Entity {namespace_id: $ns, name: $name, entity_type: $etype}) RETURN e.version_valid_from AS v",
            ns=str(ns_id),
            name=name,
            etype=entity_type,
        )
        record = await result.single()
        return record["v"] if record else None

    async with backend._session() as session:
        return await session.execute_read(_query)


@pytest.mark.asyncio
async def test_key_order_reupsert_creates_no_new_version(backend: Neo4jBackend) -> None:
    """Re-upserting the SAME attribute set in a DIFFERENT key order snapshots no version.

    ``attributes`` persist as a JSON *string* (``serialize_dict`` -> ``json.dumps``
    with no ``sort_keys``), so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    serialize to *different strings* but are equal as dicts. The Phase-2
    change-detector parses both sides to dicts before comparing, so the reorder
    is correctly seen as "no change": zero ``:EntityVersion`` nodes, zero
    ``[:SUPERSEDES]`` edges, and an unchanged ``version_valid_from``. A raw
    serialized-string compare (the pre-fix behavior) would treat the reorder as
    a change and append a phantom version."""
    ns = uuid4()
    name = f"order-noversion-{uuid4().hex[:8]}"

    seeded = {"a": 1, "b": 2}
    reordered = {"b": 2, "a": 1}
    # Precondition: same dict, different serialized string — otherwise this test
    # would not exercise the order-insensitive comparison at all.
    assert seeded == reordered
    assert json.dumps(seeded) != json.dumps(reordered)

    seed_results = await backend.upsert_entities_batch(ns, [_entity(ns, name, seeded)])
    seed_id = seed_results[0][0].id
    seeded_vvf = await _current_version_valid_from(backend, ns, name, "PERSON")

    # The re-upsert MUST land on the MATCH (existing-entity) path: the Phase-2
    # change-detector is skipped entirely for is_new nodes, so a re-upsert that
    # created a fresh node would make the zero-version assertion vacuous.
    reupsert_results = await backend.upsert_entities_batch(ns, [_entity(ns, name, reordered)])
    reupsert_entity, reupsert_is_new = reupsert_results[0]
    assert reupsert_is_new is False, "re-upsert must MATCH the seeded node (else the version assertion is vacuous)"
    assert reupsert_entity.id == seed_id, "re-upsert must resolve to the seeded entity id"

    ev_count, supersedes_count = await _count_versions(backend, ns, name, "PERSON")
    assert ev_count == 0, f"key-order-only re-upsert must not snapshot a version, got {ev_count}"
    assert supersedes_count == 0, f"key-order-only re-upsert must not create a SUPERSEDES edge, got {supersedes_count}"

    after_vvf = await _current_version_valid_from(backend, ns, name, "PERSON")
    assert after_vvf == seeded_vvf, (
        f"version_valid_from must not advance on a key-order-only re-upsert: {seeded_vvf!r} -> {after_vvf!r}"
    )

    # The stored attributes remain the same logical set.
    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_genuine_change_advances_version_valid_from(backend: Neo4jBackend) -> None:
    """Positive control: a genuine change snapshots one version AND advances ``version_valid_from``.

    Seeding ``{"a": 1}`` at one stamp then upserting ``{"a": 2}`` at a later
    stamp is a real change, so Phase-2 creates exactly one ``:EntityVersion`` +
    one ``[:SUPERSEDES]`` edge and moves the current node's
    ``version_valid_from`` forward to the new stamp — proving the
    order-insensitive detector suppresses only the reorder/empty phantom cases,
    not legitimate versioning. The stamps are pinned via ``occurred_at`` so the
    advance is deterministic, not clock-derived."""
    ns = uuid4()
    name = f"order-change-{uuid4().hex[:8]}"

    seeded_ts = "2026-01-01T00:00:00+00:00"
    changed_ts = "2026-02-01T00:00:00+00:00"

    seed_results = await backend.upsert_entities_batch(ns, [_entity_with_stamp(ns, name, {"a": 1}, seeded_ts)])
    seed_id = seed_results[0][0].id
    assert await _current_version_valid_from(backend, ns, name, "PERSON") == seeded_ts

    # The change must land on the MATCH path — otherwise Phase-2 never runs and
    # the one-version assertion would pass vacuously for the wrong reason.
    change_results = await backend.upsert_entities_batch(ns, [_entity_with_stamp(ns, name, {"a": 2}, changed_ts)])
    change_entity, change_is_new = change_results[0]
    assert change_is_new is False, "changed re-upsert must MATCH the seeded node"
    assert change_entity.id == seed_id, "changed re-upsert must resolve to the seeded entity id"

    ev_count, supersedes_count = await _count_versions(backend, ns, name, "PERSON")
    assert ev_count == 1, f"genuine change must snapshot exactly one version, got {ev_count}"
    assert supersedes_count == 1, f"genuine change must create exactly one SUPERSEDES edge, got {supersedes_count}"

    after_vvf = await _current_version_valid_from(backend, ns, name, "PERSON")
    assert after_vvf == changed_ts, f"version_valid_from must advance to the new stamp, got {after_vvf!r}"

    got = await backend.get_entity_by_name(ns, name, "PERSON")
    assert got is not None
    assert got.attributes == {"a": 2}


@pytest.mark.asyncio
async def test_int_float_value_reupsert_creates_no_new_version(backend: Neo4jBackend) -> None:
    """Documents intended semantics: an int/float-equivalent value is NOT a change.

    ``attributes`` round-trip through ``json.loads``, so a stored ``{"n": 1}`` and
    an incoming ``{"n": 1.0}`` parse to dicts that compare equal (``1 == 1.0``).
    The change-detector therefore snapshots no version — a deliberate, desirable
    normalization, locked here so a future switch to a type-strict compare can't
    silently reintroduce phantom versions."""
    ns = uuid4()
    name = f"order-intfloat-{uuid4().hex[:8]}"

    # Same dicts, different serialized string — an int vs float encoding.
    assert {"n": 1} == {"n": 1.0}
    assert json.dumps({"n": 1}) != json.dumps({"n": 1.0})

    seed_results = await backend.upsert_entities_batch(ns, [_entity(ns, name, {"n": 1})])
    seed_id = seed_results[0][0].id

    reupsert_results = await backend.upsert_entities_batch(ns, [_entity(ns, name, {"n": 1.0})])
    reupsert_entity, reupsert_is_new = reupsert_results[0]
    assert reupsert_is_new is False, "re-upsert must MATCH the seeded node"
    assert reupsert_entity.id == seed_id, "re-upsert must resolve to the seeded entity id"

    ev_count, supersedes_count = await _count_versions(backend, ns, name, "PERSON")
    assert ev_count == 0, f"int/float-equivalent value must not snapshot a version, got {ev_count}"
    assert supersedes_count == 0, (
        f"int/float-equivalent value must not create a SUPERSEDES edge, got {supersedes_count}"
    )


@pytest.mark.asyncio
async def test_list_value_reorder_reupsert_creates_a_version(backend: Neo4jBackend) -> None:
    """Documents the boundary of the fix: LIST element order is still significant.

    The order-insensitive fix normalizes MAPPING (dict-key) order only. A stored
    ``{"aliases": ["a", "b"]}`` vs an incoming ``{"aliases": ["b", "a"]}`` parse
    to unequal dicts (``["a", "b"] != ["b", "a"]``), so a genuine version IS
    snapshotted. This is intended/current behavior, not a bug — asserted here to
    pin the fix's scope."""
    ns = uuid4()
    name = f"order-listreorder-{uuid4().hex[:8]}"

    assert {"aliases": ["a", "b"]} != {"aliases": ["b", "a"]}

    seed_results = await backend.upsert_entities_batch(ns, [_entity(ns, name, {"aliases": ["a", "b"]})])
    seed_id = seed_results[0][0].id

    reupsert_results = await backend.upsert_entities_batch(ns, [_entity(ns, name, {"aliases": ["b", "a"]})])
    reupsert_entity, reupsert_is_new = reupsert_results[0]
    assert reupsert_is_new is False, "re-upsert must MATCH the seeded node"
    assert reupsert_entity.id == seed_id, "re-upsert must resolve to the seeded entity id"

    ev_count, supersedes_count = await _count_versions(backend, ns, name, "PERSON")
    assert ev_count == 1, f"list element reorder is still a change, expected one version, got {ev_count}"
    assert supersedes_count == 1, f"list element reorder must create one SUPERSEDES edge, got {supersedes_count}"


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

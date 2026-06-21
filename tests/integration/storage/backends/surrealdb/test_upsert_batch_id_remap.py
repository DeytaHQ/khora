"""Regression tests for issue #1151 against an in-memory SurrealDB instance.

``upsert_entities_batch`` must remap the CALLER's ``Entity.id`` to the
stored canonical id when the upsert matches an existing entity by
``(namespace_id, name, entity_type)`` - the #806 id-remap contract.  The
vectorcypher engine discards the return value and builds its pre-upsert ->
canonical id remap from the ids on its own input list, then issues
relationship writes whose endpoints were captured from those ids.  Before
the fix, the SurrealDB backend merged into the *fetched* row object and
never touched the input entity, so relationships for deduped entities
RELATEd to ``entity:<extraction-uuid>`` - a record that does not exist -
and were silently lost on every repeat ingest.

The Neo4j backend (``neo4j.py``, ``entity.id = UUID(neo4j_id)``) and
sqlite_lance (``graph.py``, ``entity.id = existing.id``) are the reference
implementations of the contract.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Entity, Relationship  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402

pytestmark = pytest.mark.integration

_NS_ID = UUID("11111111-2222-3333-4444-555555555555")


async def _connect() -> SurrealDBConnection:
    conn = SurrealDBConnection(mode="memory", namespace="test", database="test")
    await conn.connect()
    return conn


async def test_batch_upsert_remaps_caller_entity_id_on_match() -> None:
    """On a (ns, name, type) match the CALLER's entity.id becomes the stored id."""
    conn = await _connect()
    adapter = SurrealDBGraphAdapter(conn)
    try:
        first = Entity(namespace_id=_NS_ID, name="Alice", entity_type="PERSON", description="first ingest")
        await adapter.upsert_entities_batch(_NS_ID, [first])
        stored_id = first.id

        # Second ingest: a fresh Entity object with a new extraction-time
        # UUID but the same (namespace, name, entity_type) key.
        second = Entity(namespace_id=_NS_ID, name="Alice", entity_type="PERSON", description="second ingest")
        extraction_id = second.id
        assert extraction_id != stored_id

        results = await adapter.upsert_entities_batch(_NS_ID, [second])

        # The caller's input object must carry the canonical stored id -
        # the engine reads ids off its own input list, not the return value.
        assert second.id == stored_id
        assert len(results) == 1
        returned, is_new = results[0]
        assert is_new is False
        assert returned.id == stored_id
    finally:
        await conn.disconnect()


async def test_relationship_after_dedup_lands_on_stored_entity() -> None:
    """A relationship built from the caller's post-upsert ids resolves on read."""
    conn = await _connect()
    adapter = SurrealDBGraphAdapter(conn)
    try:
        alice_v1 = Entity(namespace_id=_NS_ID, name="Alice", entity_type="PERSON")
        await adapter.upsert_entities_batch(_NS_ID, [alice_v1])
        stored_alice_id = alice_v1.id

        # Repeat ingest: Alice deduped against the stored row, Bob is new.
        alice_v2 = Entity(namespace_id=_NS_ID, name="Alice", entity_type="PERSON")
        bob = Entity(namespace_id=_NS_ID, name="Bob", entity_type="PERSON")
        await adapter.upsert_entities_batch(_NS_ID, [alice_v2, bob])

        # The engine builds relationship endpoints from the ids on its own
        # input entities after the upsert (#806).
        rel = Relationship(
            id=uuid4(),
            namespace_id=_NS_ID,
            source_entity_id=alice_v2.id,
            target_entity_id=bob.id,
            relationship_type="KNOWS",
        )
        created = await adapter.create_relationships_batch([rel])
        # #1320: returns (relationship, is_new) per edge; SurrealDB RELATE
        # always creates, so is_new=True.
        assert created == [(rel, True)]

        # Before the fix alice_v2.id kept the extraction-time UUID, so the
        # RELATE targeted a nonexistent record and this read came back empty.
        rels = await adapter.get_entity_relationships(stored_alice_id, namespace_id=_NS_ID, direction="both")
        assert len(rels) == 1
        assert rels[0].relationship_type == "KNOWS"
        assert {rels[0].source_entity_id, rels[0].target_entity_id} == {stored_alice_id, bob.id}
    finally:
        await conn.disconnect()

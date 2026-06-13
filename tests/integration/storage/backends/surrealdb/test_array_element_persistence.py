"""Regression: SurrealDB persists array CONTENTS for every typed-array field (#1168).

Follow-up to #923 (entity / relates_to source arrays) and #1167
(memory_fact.source_chunk_ids). SurrealDB SCHEMAFULL tables silently coerce a
value bound to a bare ``TYPE option<array>`` field to ``[]`` unless the element
type is declared (either inline ``option<array<string>>`` or via the
``field[*]`` element-field idiom). #1168 audits every remaining bare
``option<array>`` declaration in the schema - the episode arrays
(``entity_ids`` / ``source_document_ids`` / ``source_chunk_ids``) were still
bare and silently dropped their contents.

Each test writes a non-empty array through the adapter and reads it back,
asserting the values survive (not silently ``[]``). Runs against
``mode="memory"`` - no docker required.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import (  # noqa: E402
    Entity,
    Episode,
    MemoryNamespace,
    Relationship,
    TenancyMode,
)
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_surrealdb_persists_entity_source_chunk_ids() -> None:
    """entity.source_chunk_ids round-trips (#923 element-field idiom)."""
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="arr_ent")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        ns_id = uuid4()
        await relational.create_namespace(
            MemoryNamespace(id=ns_id, namespace_id=ns_id, tenancy_mode=TenancyMode.SHARED)
        )
        chunk_a, chunk_b = uuid4(), uuid4()
        ent = Entity(id=uuid4(), namespace_id=ns_id, name="Shared", entity_type="CONCEPT")
        ent.source_chunk_ids = [chunk_a, chunk_b]
        await graph.upsert_entities_batch(ns_id, [ent])

        (got,) = await graph.list_entities(ns_id, limit=10)
        assert set(got.source_chunk_ids) == {chunk_a, chunk_b}, "SurrealDB dropped entity.source_chunk_ids contents"
    finally:
        await conn.disconnect()


@pytest.mark.asyncio
async def test_surrealdb_persists_relationship_source_arrays() -> None:
    """relates_to.source_document_ids / source_chunk_ids round-trip (#923)."""
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="arr_rel")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        ns_id = uuid4()
        await relational.create_namespace(
            MemoryNamespace(id=ns_id, namespace_id=ns_id, tenancy_mode=TenancyMode.SHARED)
        )
        src = Entity(id=uuid4(), namespace_id=ns_id, name="Src", entity_type="CONCEPT")
        tgt = Entity(id=uuid4(), namespace_id=ns_id, name="Tgt", entity_type="CONCEPT")
        await graph.upsert_entities_batch(ns_id, [src, tgt])

        doc_a, doc_b = uuid4(), uuid4()
        chunk_a, chunk_b = uuid4(), uuid4()
        rel = Relationship(
            id=uuid4(),
            namespace_id=ns_id,
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relationship_type="RELATES_TO",
            source_document_ids=[doc_a, doc_b],
            source_chunk_ids=[chunk_a, chunk_b],
        )
        await graph.create_relationships_batch([rel])

        (got,) = await graph.list_relationships(ns_id, limit=10)
        assert set(got.source_document_ids) == {doc_a, doc_b}, (
            "SurrealDB dropped relates_to.source_document_ids contents"
        )
        assert set(got.source_chunk_ids) == {chunk_a, chunk_b}, "SurrealDB dropped relates_to.source_chunk_ids contents"
    finally:
        await conn.disconnect()


@pytest.mark.asyncio
async def test_surrealdb_persists_episode_arrays() -> None:
    """episode.entity_ids / source_document_ids / source_chunk_ids round-trip (#1168).

    These three were still declared bare ``option<array>`` and silently
    dropped their contents to ``[]`` before #1168.
    """
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="arr_ep")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        ns_id = uuid4()
        await relational.create_namespace(
            MemoryNamespace(id=ns_id, namespace_id=ns_id, tenancy_mode=TenancyMode.SHARED)
        )
        ent_a = Entity(id=uuid4(), namespace_id=ns_id, name="A", entity_type="CONCEPT")
        ent_b = Entity(id=uuid4(), namespace_id=ns_id, name="B", entity_type="CONCEPT")
        await graph.upsert_entities_batch(ns_id, [ent_a, ent_b])

        doc_a, doc_b = uuid4(), uuid4()
        chunk_a, chunk_b = uuid4(), uuid4()
        episode = Episode(
            id=uuid4(),
            namespace_id=ns_id,
            name="meeting",
            entity_ids=[ent_a.id, ent_b.id],
            source_document_ids=[doc_a, doc_b],
            source_chunk_ids=[chunk_a, chunk_b],
        )
        await graph.create_episode(episode)

        got = await graph.get_episode(episode.id, namespace_id=ns_id)
        assert got is not None
        assert set(got.entity_ids) == {ent_a.id, ent_b.id}, "SurrealDB dropped episode.entity_ids contents"
        assert set(got.source_document_ids) == {doc_a, doc_b}, "SurrealDB dropped episode.source_document_ids contents"
        assert set(got.source_chunk_ids) == {chunk_a, chunk_b}, "SurrealDB dropped episode.source_chunk_ids contents"
    finally:
        await conn.disconnect()

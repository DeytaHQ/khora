"""Regression: SurrealDB persists entity ``source_document_ids`` (#923).

SurrealDB SCHEMAFULL tables silently strip array CONTENTS unless the array
element field (``field[*]``) is defined. Before #923 the entity / relates_to
``source_document_ids`` arrays were declared ``TYPE option<array>`` with no
element type, so every write came back ``[]`` - which made the
source-document refcount (and therefore the forget cascade) impossible on
SurrealDB. This guards that the element-field schema fix keeps the array
round-tripping.

Runs against ``mode="memory"`` - no docker required.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Entity, MemoryNamespace, TenancyMode  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_surrealdb_persists_entity_source_document_ids() -> None:
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="sdids923")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        ns_id = uuid4()
        await relational.create_namespace(
            MemoryNamespace(id=ns_id, namespace_id=ns_id, tenancy_mode=TenancyMode.SHARED)
        )
        doc_a = uuid4()
        doc_b = uuid4()

        ent = Entity(id=uuid4(), namespace_id=ns_id, name="Shared", entity_type="CONCEPT")
        ent.source_document_ids = [doc_a, doc_b]
        await graph.upsert_entities_batch(ns_id, [ent])

        (got,) = await graph.list_entities(ns_id, limit=10)
        assert set(got.source_document_ids) == {doc_a, doc_b}, (
            "SurrealDB dropped source_document_ids array contents (SCHEMAFULL element-field gap)"
        )
    finally:
        await conn.disconnect()


@pytest.mark.asyncio
async def test_surrealdb_list_entities_filters_by_source_chunk_ids() -> None:
    """``list_entities(source_chunk_ids=...)`` filters by chunk provenance (#1448).

    Seeds two entities — A sourced from chunks c1/c2, B from c3 — then pins
    the four contract cases: no filter returns both; a filter for one of A's
    chunks returns only A; an unknown chunk returns nothing; and an empty
    list matches nothing (CONTAINSANY / any-overlap semantics).
    """
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="scids1448")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        ns_id = uuid4()
        await relational.create_namespace(
            MemoryNamespace(id=ns_id, namespace_id=ns_id, tenancy_mode=TenancyMode.SHARED)
        )
        c1, c2, c3, c4 = uuid4(), uuid4(), uuid4(), uuid4()

        ent_a = Entity(id=uuid4(), namespace_id=ns_id, name="A", entity_type="PERSON", source_chunk_ids=[c1, c2])
        ent_b = Entity(id=uuid4(), namespace_id=ns_id, name="B", entity_type="PERSON", source_chunk_ids=[c3])
        await graph.upsert_entities_batch(ns_id, [ent_a, ent_b])

        # 1. No filter → both entities.
        assert {e.name for e in await graph.list_entities(ns_id, limit=10)} == {"A", "B"}

        # 2. One of A's chunks → exactly A.
        only_a = await graph.list_entities(ns_id, source_chunk_ids=[c1], limit=10)
        assert {e.name for e in only_a} == {"A"}

        # 3. Unknown chunk id → nothing.
        assert await graph.list_entities(ns_id, source_chunk_ids=[c4], limit=10) == []

        # 4. Empty list → matches nothing.
        assert await graph.list_entities(ns_id, source_chunk_ids=[], limit=10) == []
    finally:
        await conn.disconnect()

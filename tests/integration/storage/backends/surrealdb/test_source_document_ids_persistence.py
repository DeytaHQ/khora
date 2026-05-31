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

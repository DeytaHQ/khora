"""Forget-cascade cleanup against an in-memory SurrealDB stack (#923).

SurrealDB is a unified backend where entities/relationships live in the
graph adapter tables. Pre-#923 the engine cascade no-op'd here because
SurrealDB lacks Neo4j's ``fetch_document_extraction_state``. The fix gives
``SurrealDBGraphAdapter`` the cleanup primitives via ``GraphBackendBase`` and
re-anchors the cascade on ``source_document_ids`` refcounting.

Runs against ``mode="memory"`` - no docker required. Skipped when the
``surrealdb`` extra is absent.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import (  # noqa: E402
    Entity,
    MemoryNamespace,
    Relationship,
    TenancyMode,
)
from khora.engines._forget_cascade import cascade_forget_extraction  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
async def stack():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="forget923")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    vector = SurrealDBVectorAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        yield graph, vector, relational
    finally:
        await conn.disconnect()


@pytest.mark.asyncio
async def test_forget_cascade_cleans_entities_on_surrealdb(stack) -> None:
    graph, vector, relational = stack

    ns_id = uuid4()
    await relational.create_namespace(MemoryNamespace(id=ns_id, namespace_id=ns_id, tenancy_mode=TenancyMode.SHARED))

    doc_a = uuid4()
    doc_b = uuid4()

    shared = Entity(id=uuid4(), namespace_id=ns_id, name="Shared", entity_type="CONCEPT")
    shared.source_document_ids = [doc_a, doc_b]
    unique_a = Entity(id=uuid4(), namespace_id=ns_id, name="UniqueA", entity_type="CONCEPT")
    unique_a.source_document_ids = [doc_a]
    unique_b = Entity(id=uuid4(), namespace_id=ns_id, name="UniqueB", entity_type="CONCEPT")
    unique_b.source_document_ids = [doc_b]
    await graph.upsert_entities_batch(ns_id, [shared, unique_a, unique_b])

    orphan_edge = Relationship(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=shared.id,
        target_entity_id=unique_a.id,
        relationship_type="RELATES_TO",
    )
    orphan_edge.source_document_ids = [doc_a]
    shared_edge = Relationship(
        id=uuid4(),
        namespace_id=ns_id,
        source_entity_id=shared.id,
        target_entity_id=unique_b.id,
        relationship_type="RELATES_TO",
    )
    shared_edge.source_document_ids = [doc_a, doc_b]
    await graph.create_relationships_batch([orphan_edge, shared_edge])

    # ----- Act: forget doc A -----
    degradations = await cascade_forget_extraction(
        graph=graph,
        vector=vector,
        document_id=doc_a,
        namespace_id=ns_id,
        engine="test",
    )
    assert degradations == []

    names = {e.name: e for e in await graph.list_entities(ns_id, limit=1000)}
    assert "UniqueA" not in names
    assert "Shared" in names
    assert doc_a not in names["Shared"].source_document_ids
    assert doc_b in names["Shared"].source_document_ids
    assert "UniqueB" in names
    assert names["UniqueB"].source_document_ids == [doc_b]

    rels = await graph.list_relationships(ns_id, limit=1000)
    by_target = {r.target_entity_id: r for r in rels}
    assert unique_a.id not in by_target, "orphan relationship survived forget(doc_a)"
    assert unique_b.id in by_target, "shared relationship was wrongly deleted"
    survivor = by_target[unique_b.id]
    assert doc_a not in survivor.source_document_ids
    assert doc_b in survivor.source_document_ids

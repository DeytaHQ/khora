"""Round-trip tests for SurrealDB get-by-id methods (issue #750).

Covers the methods that silently returned ``None`` / empty before the
``table:⟨$var⟩`` interpolation fix:

- ``SurrealDBGraphAdapter.get_entity``
- ``SurrealDBGraphAdapter.get_relationship``
- ``SurrealDBVectorAdapter.get_chunk``
- ``SurrealDBVectorAdapter.get_chunks_by_document``
- ``SurrealDBVectorAdapter.get_chunks_batch``

Each get-by-id is exercised against two namespaces to make sure the
namespace filter still excludes cross-tenant lookups (IDOR).  We also
read the raw ``relates_to`` row to verify that ``in`` / ``out`` are
real ``entity:⟨<uuid>⟩`` ``RecordID`` objects (not literal ``$var``
strings) and that ``rel_id`` is populated (the SCHEMAFULL schema must
declare the field, otherwise SurrealDB silently drops the write).

Runs against an in-memory SurrealDB (``mode="memory"``) — no docker
required.  Skipped when the ``surrealdb`` extra is not installed.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from surrealdb.data.types.record_id import RecordID  # noqa: E402

from khora.core.models import (  # noqa: E402
    Chunk,
    Document,
    Entity,
    MemoryNamespace,
    Relationship,
    TenancyMode,
)
from khora.core.models.document import DocumentStatus  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.graph import SurrealDBGraphAdapter  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402
from khora.storage.backends.surrealdb.vector import SurrealDBVectorAdapter  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
async def stack():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="getbyid750")
    await conn.connect()
    graph = SurrealDBGraphAdapter(conn)
    vector = SurrealDBVectorAdapter(conn)
    relational = SurrealDBRelationalAdapter(conn)
    try:
        yield conn, graph, vector, relational
    finally:
        await conn.disconnect()


@pytest.fixture
async def seeded(stack):
    """Two namespaces, with entity / relationship / document / chunk in ns_a only."""
    conn, graph, vector, relational = stack

    ns_a_id = uuid4()
    ns_b_id = uuid4()
    ns_a = MemoryNamespace(id=ns_a_id, namespace_id=ns_a_id, tenancy_mode=TenancyMode.SHARED)
    ns_b = MemoryNamespace(id=ns_b_id, namespace_id=ns_b_id, tenancy_mode=TenancyMode.SHARED)
    await relational.create_namespace(ns_a)
    await relational.create_namespace(ns_b)

    e1 = Entity(id=uuid4(), namespace_id=ns_a_id, name="alice", entity_type="PERSON")
    e2 = Entity(id=uuid4(), namespace_id=ns_a_id, name="bob", entity_type="PERSON")
    await graph.upsert_entities_batch(ns_a_id, [e1, e2])

    r1 = Relationship(
        id=uuid4(),
        namespace_id=ns_a_id,
        source_entity_id=e1.id,
        target_entity_id=e2.id,
        relationship_type="KNOWS",
    )
    await graph.create_relationships_batch([r1])

    doc = Document(
        id=uuid4(),
        namespace_id=ns_a_id,
        content="hello",
        status=DocumentStatus.COMPLETED,
        checksum="r750",
    )
    await relational.create_document(doc)

    c1 = Chunk(
        id=uuid4(),
        namespace_id=ns_a_id,
        document_id=doc.id,
        content="chunk-0",
        chunk_index=0,
        embedding=[0.1] * 8,
        embedding_model="m",
    )
    c2 = Chunk(
        id=uuid4(),
        namespace_id=ns_a_id,
        document_id=doc.id,
        content="chunk-1",
        chunk_index=1,
        embedding=[0.2] * 8,
        embedding_model="m",
    )
    await vector.create_chunks_batch([c1, c2])

    return {
        "ns_a_id": ns_a_id,
        "ns_b_id": ns_b_id,
        "e1": e1,
        "e2": e2,
        "r1": r1,
        "doc": doc,
        "c1": c1,
        "c2": c2,
    }


async def test_get_entity_roundtrip(stack, seeded) -> None:
    """``get_entity`` returns the row in the owning namespace, ``None`` elsewhere."""
    _, graph, _, _ = stack
    got = await graph.get_entity(seeded["e1"].id, namespace_id=seeded["ns_a_id"])
    assert got is not None
    assert got.id == seeded["e1"].id
    assert got.name == "alice"

    leak = await graph.get_entity(seeded["e1"].id, namespace_id=seeded["ns_b_id"])
    assert leak is None


async def test_get_relationship_roundtrip(stack, seeded) -> None:
    """``get_relationship`` returns the row by ``rel_id``, IDOR-safe."""
    _, graph, _, _ = stack
    got = await graph.get_relationship(seeded["r1"].id, namespace_id=seeded["ns_a_id"])
    assert got is not None
    assert got.id == seeded["r1"].id
    assert got.source_entity_id == seeded["e1"].id
    assert got.target_entity_id == seeded["e2"].id
    assert got.relationship_type == "KNOWS"

    leak = await graph.get_relationship(seeded["r1"].id, namespace_id=seeded["ns_b_id"])
    assert leak is None


async def test_get_chunk_roundtrip(stack, seeded) -> None:
    """``get_chunk`` returns the row in the owning namespace, ``None`` elsewhere."""
    _, _, vector, _ = stack
    got = await vector.get_chunk(seeded["c1"].id, namespace_id=seeded["ns_a_id"])
    assert got is not None
    assert got.id == seeded["c1"].id
    assert got.content == "chunk-0"

    leak = await vector.get_chunk(seeded["c1"].id, namespace_id=seeded["ns_b_id"])
    assert leak is None


async def test_get_chunks_by_document_roundtrip(stack, seeded) -> None:
    """``get_chunks_by_document`` returns all chunks, IDOR-safe."""
    _, _, vector, _ = stack
    got = await vector.get_chunks_by_document(seeded["doc"].id, namespace_id=seeded["ns_a_id"])
    assert {c.id for c in got} == {seeded["c1"].id, seeded["c2"].id}
    # ORDER BY chunk_index ASC
    assert [c.chunk_index for c in got] == [0, 1]

    leak = await vector.get_chunks_by_document(seeded["doc"].id, namespace_id=seeded["ns_b_id"])
    assert leak == []


async def test_get_chunks_batch_roundtrip(stack, seeded) -> None:
    """``get_chunks_batch`` returns a dict keyed by chunk id, IDOR-safe."""
    _, _, vector, _ = stack
    got = await vector.get_chunks_batch(
        [seeded["c1"].id, seeded["c2"].id],
        namespace_id=seeded["ns_a_id"],
    )
    assert set(got.keys()) == {seeded["c1"].id, seeded["c2"].id}

    leak = await vector.get_chunks_batch(
        [seeded["c1"].id, seeded["c2"].id],
        namespace_id=seeded["ns_b_id"],
    )
    assert leak == {}


async def test_relates_to_row_has_real_record_id_endpoints(stack, seeded) -> None:
    """Raw row check: ``in`` / ``out`` / ``rel_id`` persisted correctly.

    Before #750 the ``FOR ... RELATE entity:⟨$rel.source_rid⟩...`` template
    bound ``$rel.source_rid`` as a literal string, so ``in`` / ``out`` were
    corrupt and ``rel_id`` was silently dropped by SCHEMAFULL.
    """
    conn, _, _, _ = stack
    rows = await conn.query(
        "SELECT in, out, rel_id FROM relates_to WHERE rel_id = $rid",
        {"rid": str(seeded["r1"].id)},
    )
    assert len(rows) == 1
    row = rows[0]

    # ``in`` and ``out`` must be real RecordIDs with UUID inner ids,
    # pointing at the seeded entity records.  The inner id attribute is
    # named ``id`` on ``surrealdb.RecordID``; the ``record_id=...`` form
    # is only the ``repr()`` rendering.
    assert isinstance(row["in"], RecordID)
    assert row["in"].table_name == "entity"
    assert isinstance(row["in"].id, UUID)
    assert row["in"].id == seeded["e1"].id

    assert isinstance(row["out"], RecordID)
    assert row["out"].table_name == "entity"
    assert isinstance(row["out"].id, UUID)
    assert row["out"].id == seeded["e2"].id

    # rel_id must round-trip the Khora UUID (not be dropped by SCHEMAFULL)
    assert row["rel_id"] == str(seeded["r1"].id)

"""Round-trip test for ``SurrealDBRelationalAdapter.get_document_projections_batch``.

Exercises the four risk surfaces flagged on the recall rewrite:

- RecordID → UUID coercion at the projection boundary (``_parse_uuid``
  on the ``id`` column, which Surreal hands back as a ``RecordID``).
- NULL / falsy ``source_type`` defaults to ``"library"`` on the
  projection (matches sqlite / postgres / sqlite_lance behaviour).
- ``metadata_`` column name is consistent between INSERT and SELECT
  (regression net for the schema-naming gotcha).
- Wider field set (``external_id`` / ``source_name`` / ``source_url`` /
  ``content_type`` + ``metadata``) survives the round trip.

Runs against an in-memory SurrealDB (``mode="memory"``) — no docker
required. Skipped when the ``surrealdb`` extra is not installed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

pytest.importorskip("surrealdb")

from khora.core.models import Document, DocumentProjection, MemoryNamespace, TenancyMode  # noqa: E402
from khora.storage.backends.surrealdb.connection import SurrealDBConnection  # noqa: E402
from khora.storage.backends.surrealdb.relational import SurrealDBRelationalAdapter  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture
async def adapter():
    conn = SurrealDBConnection(mode="memory", namespace="khora_test", database="projections")
    await conn.connect()
    adapter = SurrealDBRelationalAdapter(conn)
    try:
        yield adapter
    finally:
        await conn.disconnect()


@pytest.fixture
async def namespace(adapter):
    nid = uuid4()
    ns = MemoryNamespace(id=nid, namespace_id=nid, tenancy_mode=TenancyMode.SHARED)
    return await adapter.create_namespace(ns)


async def test_projections_batch_full_field_roundtrip(adapter, namespace) -> None:
    """Every projection column survives the SurrealDB INSERT → SELECT round-trip."""
    doc = Document(
        namespace_id=namespace.id,
        content="hello surreal",
        title="Surreal Title",
        external_id="surreal-ext-7",
        source="ws://example.com/x",
        source_name="Surreal Source",
        source_url="https://example.com/surreal",
        source_type="file",
        content_type="text/markdown",
        checksum="surreal-rt",
        size_bytes=12,
        metadata={"k": "v", "n": 1},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    await adapter.create_document(doc)

    projections = await adapter.get_document_projections_batch([doc.id], namespace_id=namespace.id)

    assert doc.id in projections
    proj = projections[doc.id]
    assert isinstance(proj, DocumentProjection)
    # RecordID → UUID coercion at the boundary.
    assert isinstance(proj.id, UUID)
    assert proj.id == doc.id
    assert proj.title == "Surreal Title"
    assert proj.external_id == "surreal-ext-7"
    assert proj.source == "ws://example.com/x"
    assert proj.source_name == "Surreal Source"
    assert proj.source_url == "https://example.com/surreal"
    assert proj.source_type == "file"
    assert proj.content_type == "text/markdown"
    assert proj.metadata == {"k": "v", "n": 1}
    assert proj.created_at is not None


async def test_projections_batch_normalizes_empty_strings_to_none(adapter, namespace) -> None:
    """Recall-response contract: unset optional strings surface as ``None`` for both
    ``""`` and ``NULL`` rows."""
    from khora.storage.backends.surrealdb.relational import _record_id

    # Row A: build with None at construction, then UPDATE the columns to ``""``
    # at the row level (the Document dataclass forbids blank ``external_id``).
    doc_a = Document(
        namespace_id=namespace.id,
        content="row-a",
        checksum="surreal-empty-a",
    )
    await adapter.create_document(doc_a)
    await adapter._conn.query(
        "UPDATE $rid SET title = $blank, external_id = $blank, source = $blank, "
        "source_name = $blank, source_url = $blank, content_type = $blank",
        {"rid": _record_id("document", doc_a.id), "blank": ""},
    )

    # Row B: all six fields stay None at construction time.
    doc_b = Document(
        namespace_id=namespace.id,
        content="row-b",
        checksum="surreal-empty-b",
    )
    await adapter.create_document(doc_b)

    projections = await adapter.get_document_projections_batch([doc_a.id, doc_b.id], namespace_id=namespace.id)

    assert set(projections.keys()) == {doc_a.id, doc_b.id}
    for doc_id in (doc_a.id, doc_b.id):
        proj = projections[doc_id]
        assert isinstance(proj, DocumentProjection)
        assert proj.title is None
        assert proj.external_id is None
        assert proj.source is None
        assert proj.source_name is None
        assert proj.source_url is None
        assert proj.content_type is None


async def test_projections_batch_null_source_type_defaults_to_library(adapter, namespace) -> None:
    """A document with falsy ``source_type`` projects as ``source_type="library"``.

    Mirrors the same coercion contract the other three backends honour.
    """
    doc = Document(
        namespace_id=namespace.id,
        content="x",
        title="no source_type",
        checksum="surreal-null-st",
        source_type="",  # falsy
    )
    await adapter.create_document(doc)

    projections = await adapter.get_document_projections_batch([doc.id], namespace_id=namespace.id)
    proj = projections[doc.id]
    assert proj.source_type == "library"


async def test_projections_batch_empty_input(adapter) -> None:
    """Empty input short-circuits to ``{}`` without a SurrealDB query."""
    assert await adapter.get_document_projections_batch([], namespace_id=uuid4()) == {}


async def test_projections_batch_unknown_id_omitted(adapter, namespace) -> None:
    """Unknown ids do not appear in the result — caller's job to handle missing."""
    doc = Document(namespace_id=namespace.id, content="present", checksum="surreal-present")
    await adapter.create_document(doc)
    missing = uuid4()

    projections = await adapter.get_document_projections_batch([doc.id, missing], namespace_id=namespace.id)
    assert set(projections.keys()) == {doc.id}

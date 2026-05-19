"""Round-trip test for ``PostgreSQLBackend.get_document_projections_batch``.

Exercises:

- Wider field set survives the SELECT (``external_id``, ``source_name``,
  ``source_url``, ``content_type``, ``metadata``).
- ``source_type`` NULL → ``"library"`` coercion at the projection
  boundary.
- ``metadata_`` JSON column round-trips as a dict.
- Empty input short-circuits to ``{}``.
- Unknown ids are silently omitted from the result.

Requires a running PostgreSQL (``make dev``). Skipped automatically when
the configured ``KHORA_DATABASE_URL`` is unreachable.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from khora.core.models import Document, DocumentProjection, MemoryNamespace
from khora.core.models.document import DocumentStatus
from khora.db.session import run_migrations
from khora.storage.backends.postgresql import PostgreSQLBackend

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5432/khora",
)

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


pytestmark = [pytest.mark.integration]


def _pg_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


skip_no_pg = pytest.mark.skipif(
    not _pg_reachable(),
    reason="PostgreSQL not reachable (run `make dev` first)",
)


@pytest.fixture(scope="module")
async def _run_migrations_once():
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"


@pytest.fixture
async def backend(_run_migrations_once):
    be = PostgreSQLBackend(database_url=DATABASE_URL)
    await be.connect()
    try:
        yield be
    finally:
        await be.disconnect()


@skip_no_pg
class TestGetDocumentProjectionsBatchPg:
    async def test_full_field_roundtrip(self, backend: PostgreSQLBackend) -> None:
        ns = await backend.create_namespace(MemoryNamespace())

        doc = Document(
            id=uuid4(),
            namespace_id=ns.id,
            content="hello pg",
            status=DocumentStatus.PENDING,
            title="PG Title",
            external_id=f"pg-ext-{uuid4().hex[:8]}",
            source="file:///tmp/pg.txt",
            source_name="PG Source",
            source_url="https://example.com/pg",
            source_type="file",
            content_type="text/plain",
            checksum=f"pg-rt-{uuid4().hex[:8]}",
            size_bytes=8,
            metadata={"k": "v", "n": 1, "nested": {"a": True}},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await backend.create_document(doc)

        projections = await backend.get_document_projections_batch([doc.id], namespace_id=ns.id)

        assert doc.id in projections
        proj = projections[doc.id]
        assert isinstance(proj, DocumentProjection)
        assert isinstance(proj.id, UUID)
        assert proj.id == doc.id
        assert proj.title == "PG Title"
        assert proj.external_id == doc.external_id
        assert proj.source == "file:///tmp/pg.txt"
        assert proj.source_name == "PG Source"
        assert proj.source_url == "https://example.com/pg"
        assert proj.source_type == "file"
        assert proj.content_type == "text/plain"
        assert proj.metadata == {"k": "v", "n": 1, "nested": {"a": True}}
        assert proj.created_at is not None

    async def test_null_source_type_defaults_to_library(self, backend: PostgreSQLBackend) -> None:
        """Document with falsy ``source_type`` projects as ``source_type="library"``."""
        ns = await backend.create_namespace(MemoryNamespace())
        doc = Document(
            id=uuid4(),
            namespace_id=ns.id,
            content="x",
            status=DocumentStatus.PENDING,
            title="no source_type",
            checksum=f"pg-null-st-{uuid4().hex[:8]}",
            source_type="",  # falsy → projection should coerce
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await backend.create_document(doc)

        projections = await backend.get_document_projections_batch([doc.id], namespace_id=ns.id)
        proj = projections[doc.id]
        assert proj.source_type == "library"

    async def test_empty_input(self, backend: PostgreSQLBackend) -> None:
        assert await backend.get_document_projections_batch([], namespace_id=uuid4()) == {}

    async def test_unknown_id_omitted(self, backend: PostgreSQLBackend) -> None:
        ns = await backend.create_namespace(MemoryNamespace())
        doc = Document(
            id=uuid4(),
            namespace_id=ns.id,
            content="present",
            status=DocumentStatus.PENDING,
            checksum=f"pg-present-{uuid4().hex[:8]}",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await backend.create_document(doc)

        projections = await backend.get_document_projections_batch([doc.id, uuid4()], namespace_id=ns.id)
        assert set(projections.keys()) == {doc.id}

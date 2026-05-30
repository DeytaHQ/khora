"""Regression tests for #895 - partial_update_document preserves unset columns.

The previous ``update_document`` path enumerated all 22 mutable columns on every
call. A caller that constructed a fresh ``Document`` to patch a single field
(say, status) would NULL-overwrite title, author, and every other field the
fresh dataclass left at its default. This file pins the contract for the new
``partial_update_document`` helper: it sends a SQL UPDATE that only mentions
the columns the caller passed in, and Postgres preserves everything else.

The PG backend reuses SQLAlchemy ORM models that work against SQLite too -
we lean on that to roundtrip the test without a Postgres process.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

# aiosqlite is the only external requirement (already pulled in by the
# sqlite_lance test stack). Skip cleanly when absent so the suite still
# runs in slimmer environments.
try:
    import aiosqlite  # noqa: F401

    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

from khora.core.models import MemoryNamespace, TenancyMode
from khora.core.models.document import Document, DocumentStatus
from khora.db.session import run_migrations

pytestmark = pytest.mark.skipif(not _HAS_SQLITE, reason="aiosqlite not installed")

if _HAS_SQLITE:
    from khora.storage.backends.postgresql import PostgreSQLBackend


@pytest.fixture(scope="module")
def _migrated_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the Alembic migration chain against a fresh SQLite DB once per module."""
    template_dir = tmp_path_factory.mktemp("partial_update_template")
    template_path = template_dir / "template.db"

    async def _migrate() -> None:
        result = await run_migrations(f"sqlite+aiosqlite:///{template_path}")
        if not result.success:
            raise RuntimeError(f"template migration failed: {result.error}")

    asyncio.run(_migrate())
    return template_path


@pytest.fixture
def fresh_db(_migrated_template: Path, tmp_path: Path) -> Path:
    """Each test gets its own copy of the migrated template."""
    target = tmp_path / "khora.db"
    shutil.copy(_migrated_template, target)
    return target


async def _make_backend(db_path: Path) -> PostgreSQLBackend:
    """Build a PostgreSQLBackend pointed at a SQLite file.

    PostgreSQLBackend is dialect-agnostic for the queries we exercise here
    (basic INSERT / UPDATE / SELECT via SQLAlchemy ORM), so we point it
    at SQLite to avoid spinning up Postgres for a unit test.
    """
    url = f"sqlite+aiosqlite:///{db_path}"
    backend = PostgreSQLBackend(url)
    await backend.connect()
    return backend


def _make_namespace() -> MemoryNamespace:
    return MemoryNamespace(
        id=uuid4(),
        namespace_id=uuid4(),
        tenancy_mode=TenancyMode.SHARED,
    )


@pytest.mark.asyncio
async def test_partial_update_document_preserves_unset_columns(fresh_db: Path) -> None:
    """Insert doc with title and author, patch only status - title/author must survive."""
    backend = await _make_backend(fresh_db)
    try:
        ns = _make_namespace()
        await backend.create_namespace(ns)

        original = Document(
            id=uuid4(),
            namespace_id=ns.id,
            content="hello world",
            status=DocumentStatus.PENDING,
            title="X",
            author="Y",
            source="src.txt",
            language="en",
            size_bytes=11,
        )
        await backend.create_document(original)

        # Sanity check: stored as inserted.
        before = await backend.get_document(original.id, namespace_id=ns.id)
        assert before is not None
        assert before.title == "X"
        assert before.author == "Y"
        assert before.status == DocumentStatus.PENDING

        # Patch a single column. Pass NOTHING for title/author/etc.
        rowcount = await backend.partial_update_document(
            original.id,
            namespace_id=ns.id,
            status=DocumentStatus.PROCESSING,
        )
        assert rowcount == 1

        # Roundtrip: title and author MUST be preserved.
        after = await backend.get_document(original.id, namespace_id=ns.id)
        assert after is not None
        assert after.title == "X", "title was NULL-overwritten by partial update"
        assert after.author == "Y", "author was NULL-overwritten by partial update"
        assert after.source == "src.txt"
        assert after.language == "en"
        assert after.size_bytes == 11
        assert after.status == DocumentStatus.PROCESSING
        # updated_at must move forward.
        assert after.updated_at >= before.updated_at
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_partial_update_document_rejects_unknown_columns(fresh_db: Path) -> None:
    """Unknown column names must raise rather than silently no-op."""
    backend = await _make_backend(fresh_db)
    try:
        ns = _make_namespace()
        await backend.create_namespace(ns)

        doc = Document(id=uuid4(), namespace_id=ns.id, content="x", title="T")
        await backend.create_document(doc)

        with pytest.raises(ValueError, match="unknown column"):
            await backend.partial_update_document(
                doc.id,
                namespace_id=ns.id,
                bogus_column="nope",
            )
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_partial_update_document_namespace_scoped(fresh_db: Path) -> None:
    """A document in namespace A must not be patchable from namespace B (IDOR guard)."""
    backend = await _make_backend(fresh_db)
    try:
        ns_a = _make_namespace()
        ns_b = _make_namespace()
        await backend.create_namespace(ns_a)
        await backend.create_namespace(ns_b)

        doc = Document(id=uuid4(), namespace_id=ns_a.id, content="x", title="orig")
        await backend.create_document(doc)

        # Wrong namespace - 0 rows touched, original title preserved.
        rowcount = await backend.partial_update_document(
            doc.id,
            namespace_id=ns_b.id,
            title="hijacked",
        )
        assert rowcount == 0

        after = await backend.get_document(doc.id, namespace_id=ns_a.id)
        assert after is not None
        assert after.title == "orig"
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_partial_update_document_metadata_alias(fresh_db: Path) -> None:
    """Caller passes 'metadata' but the ORM column is 'metadata_' - alias must work."""
    backend = await _make_backend(fresh_db)
    try:
        ns = _make_namespace()
        await backend.create_namespace(ns)

        doc = Document(
            id=uuid4(),
            namespace_id=ns.id,
            content="x",
            title="keep me",
            metadata={"a": 1},
        )
        await backend.create_document(doc)

        rowcount = await backend.partial_update_document(
            doc.id,
            namespace_id=ns.id,
            metadata={"a": 2, "b": 3},
        )
        assert rowcount == 1

        after = await backend.get_document(doc.id, namespace_id=ns.id)
        assert after is not None
        assert after.metadata == {"a": 2, "b": 3}
        # Title untouched.
        assert after.title == "keep me"
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_partial_update_document_no_fields_noop(fresh_db: Path) -> None:
    """Calling with no kwargs is a no-op (returns 0, no implicit updated_at touch)."""
    backend = await _make_backend(fresh_db)
    try:
        ns = _make_namespace()
        await backend.create_namespace(ns)

        doc = Document(id=uuid4(), namespace_id=ns.id, content="x", title="T")
        await backend.create_document(doc)
        before = await backend.get_document(doc.id, namespace_id=ns.id)
        assert before is not None
        before_updated = before.updated_at

        rowcount = await backend.partial_update_document(doc.id, namespace_id=ns.id)
        assert rowcount == 0

        after = await backend.get_document(doc.id, namespace_id=ns.id)
        assert after is not None
        assert after.updated_at == before_updated
    finally:
        await backend.disconnect()

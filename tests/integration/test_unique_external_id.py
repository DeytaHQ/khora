"""Integration tests for the unique (namespace_id, external_id) partial index.

Validates migration 022 which promotes the partial composite index on
(namespace_id, external_id) WHERE external_id IS NOT NULL to UNIQUE.

Requires a running PostgreSQL instance (``make dev``).

Connection parameters (env overrides, sensible ``make dev`` defaults)::

    KHORA_DATABASE_URL  (default: postgresql+asyncpg://khora:khora@localhost:5434/khora)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from khora.db.models import DocumentModel, MemoryNamespaceModel
from khora.db.session import run_migrations

DATABASE_URL = os.environ.get(
    "KHORA_DATABASE_URL",
    "postgresql+asyncpg://khora:khora@localhost:5434/khora",
)

# Normalize to asyncpg driver
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

pytestmark = [pytest.mark.integration]


def _pg_reachable() -> bool:
    """Quick TCP check to see if PostgreSQL is listening."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(DATABASE_URL.replace("+asyncpg", ""))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5434
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
    """Run Alembic migrations once for the entire module."""
    result = await run_migrations(DATABASE_URL)
    assert result.success, f"Migrations failed: {result.error}"


@pytest.fixture
async def session_factory(_run_migrations_once) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Create a disposable engine + session factory per test."""
    engine = create_async_engine(DATABASE_URL)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def _create_namespace(session: AsyncSession) -> MemoryNamespaceModel:
    """Insert a fresh namespace and return it."""
    ns_id = uuid4()
    ns = MemoryNamespaceModel(id=ns_id, namespace_id=ns_id)
    session.add(ns)
    await session.flush()
    return ns


async def _insert_document(
    session: AsyncSession,
    namespace_id,
    *,
    external_id: str | None = None,
) -> DocumentModel:
    """Insert a minimal document row."""
    doc = DocumentModel(
        id=uuid4(),
        namespace_id=namespace_id,
        content="test content",
        external_id=external_id,
    )
    session.add(doc)
    await session.flush()
    return doc


@skip_no_pg
class TestUniqueExternalId:
    """Tests for the UNIQUE partial index on (namespace_id, external_id)."""

    async def test_duplicate_external_id_raises_integrity_error(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Two documents with the same (namespace_id, external_id) must raise IntegrityError."""
        async with session_factory() as session:
            async with session.begin():
                ns = await _create_namespace(session)
                ext_id = f"dup-{uuid4().hex[:8]}"
                await _insert_document(session, ns.id, external_id=ext_id)
                with pytest.raises(IntegrityError, match=r"ix_documents_namespace_external_id_unique"):
                    await _insert_document(session, ns.id, external_id=ext_id)

    async def test_concurrent_insert_same_external_id_raises(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Concurrent inserts for the same (namespace_id, external_id) — at least one must fail."""
        # Pre-create namespace in its own transaction so both tasks can see it.
        async with session_factory() as session:
            async with session.begin():
                ns = await _create_namespace(session)

        ns_id = ns.id
        ext_id = f"conc-{uuid4().hex[:8]}"
        errors: list[Exception] = []

        async def _insert(factory: async_sessionmaker[AsyncSession]) -> None:
            async with factory() as sess:
                async with sess.begin():
                    doc = DocumentModel(
                        id=uuid4(),
                        namespace_id=ns_id,
                        content="concurrent",
                        external_id=ext_id,
                    )
                    sess.add(doc)

        tasks = [_insert(session_factory), _insert(session_factory)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]

        # At least one must fail with IntegrityError
        assert any(isinstance(e, IntegrityError) for e in errors), (
            f"Expected at least one IntegrityError, got: {errors}"
        )

    async def test_null_external_id_allows_duplicates(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """NULL external_id is excluded from the partial index — duplicates are fine."""
        async with session_factory() as session:
            async with session.begin():
                ns = await _create_namespace(session)
                doc1 = await _insert_document(session, ns.id, external_id=None)
                doc2 = await _insert_document(session, ns.id, external_id=None)

        # Both rows persisted with distinct IDs
        assert doc1.id != doc2.id

    async def test_empty_string_external_id_triggers_unique_constraint(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Empty string is NOT NULL — the partial index covers it, so duplicates must fail."""
        async with session_factory() as session:
            async with session.begin():
                ns = await _create_namespace(session)
                await _insert_document(session, ns.id, external_id="")
                with pytest.raises(IntegrityError, match=r"ix_documents_namespace_external_id_unique"):
                    await _insert_document(session, ns.id, external_id="")

    async def test_same_external_id_different_namespace_allowed(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The same external_id in different namespaces must be allowed."""
        async with session_factory() as session:
            async with session.begin():
                ns1 = await _create_namespace(session)
                ns2 = await _create_namespace(session)
                ext_id = f"cross-ns-{uuid4().hex[:8]}"
                doc1 = await _insert_document(session, ns1.id, external_id=ext_id)
                doc2 = await _insert_document(session, ns2.id, external_id=ext_id)

        assert doc1.id != doc2.id

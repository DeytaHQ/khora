"""Tests for DYT-2381: PostgreSQL backend excludes FAILED documents from checksum lookups.

Verifies that get_document_by_checksum() and get_documents_by_checksums() filter out
FAILED documents so they can be re-ingested.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.core.models.document import DocumentStatus
from khora.db.models import DocumentModel
from khora.storage.backends.postgresql import PostgreSQLBackend


def _mock_document_model(*, namespace_id, checksum="abc123", status=DocumentStatus.COMPLETED):
    """Create a mock DocumentModel."""
    model = MagicMock(spec=DocumentModel)
    model.id = uuid4()
    model.namespace_id = namespace_id
    model.content = "test content"
    model.status = status
    model.source = "test.txt"
    model.source_type = "file"
    model.content_type = "text/plain"
    model.title = "Test"
    model.author = "tester"
    model.language = None
    model.checksum = checksum
    model.size_bytes = 100
    model.metadata_ = {}
    model.chunk_count = 0
    model.entity_count = 0
    model.error_message = None
    model.extraction_config_hash = None
    model.created_at = datetime.now(UTC)
    model.updated_at = datetime.now(UTC)
    model.processed_at = None
    return model


def _make_backend(session_mock) -> PostgreSQLBackend:
    """Create a PostgreSQLBackend with a mocked session factory."""
    backend = PostgreSQLBackend.__new__(PostgreSQLBackend)

    @asynccontextmanager
    async def _fake_session():
        yield session_mock

    backend._get_session = _fake_session
    return backend


@pytest.mark.unit
class TestGetDocumentByChecksumExcludesFailed:
    """Verify get_document_by_checksum filters out FAILED documents (DYT-2381)."""

    @pytest.mark.asyncio
    async def test_returns_none_when_only_failed_exists(self) -> None:
        """When the only matching doc is FAILED, returns None."""
        ns_id = uuid4()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_document_by_checksum(ns_id, "abc123")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_includes_failed_filter(self) -> None:
        """The SQLAlchemy query includes DocumentModel.status != FAILED."""
        ns_id = uuid4()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        await backend.get_document_by_checksum(ns_id, "abc123")

        # Inspect the select() statement passed to session.execute
        call_args = session.execute.call_args
        stmt = call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "status" in compiled
        assert "FAILED" in compiled.upper() or "failed" in compiled

    @pytest.mark.asyncio
    async def test_returns_completed_document(self) -> None:
        """When a COMPLETED doc matches, it is returned."""
        ns_id = uuid4()
        model = _mock_document_model(namespace_id=ns_id, checksum="abc123")
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = model
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_document_by_checksum(ns_id, "abc123")

        assert result is not None
        assert result.id == model.id


@pytest.mark.unit
class TestGetDocumentsByChecksumsExcludesFailed:
    """Verify get_documents_by_checksums filters out FAILED documents (DYT-2381)."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_only_failed_exist(self) -> None:
        """When all matching docs are FAILED, returns empty dict."""
        ns_id = uuid4()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_documents_by_checksums(ns_id, ["abc123", "def456"])

        assert result == {}

    @pytest.mark.asyncio
    async def test_query_includes_failed_filter(self) -> None:
        """The SQLAlchemy query includes DocumentModel.status != FAILED."""
        ns_id = uuid4()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        await backend.get_documents_by_checksums(ns_id, ["abc123"])

        call_args = session.execute.call_args
        stmt = call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "status" in compiled
        assert "FAILED" in compiled.upper() or "failed" in compiled

    @pytest.mark.asyncio
    async def test_returns_completed_documents(self) -> None:
        """Completed docs are returned, keyed by checksum."""
        ns_id = uuid4()
        model1 = _mock_document_model(namespace_id=ns_id, checksum="abc123")
        model2 = _mock_document_model(namespace_id=ns_id, checksum="def456")
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [model1, model2]
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_documents_by_checksums(ns_id, ["abc123", "def456"])

        assert len(result) == 2
        assert "abc123" in result
        assert "def456" in result

    @pytest.mark.asyncio
    async def test_empty_checksums_returns_empty(self) -> None:
        """Empty checksums list returns empty dict without querying."""
        ns_id = uuid4()
        session = AsyncMock()
        backend = _make_backend(session)

        result = await backend.get_documents_by_checksums(ns_id, [])

        assert result == {}
        session.execute.assert_not_awaited()

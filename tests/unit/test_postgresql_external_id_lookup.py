"""Tests for DYT-2674: PostgreSQL backend get_document_by_external_id.

Verifies status-agnostic lookup (unlike get_document_by_checksum) and the
None short-circuit guard — both required for the replace dispatch.
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


def _mock_document_model(
    *,
    namespace_id,
    external_id="ext-1",
    status=DocumentStatus.COMPLETED,
):
    """Create a mock DocumentModel with an external_id."""
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
    model.checksum = "abc123"
    model.size_bytes = 100
    model.metadata_ = {}
    model.chunk_count = 0
    model.entity_count = 0
    model.error_message = None
    model.extraction_config_hash = None
    model.external_id = external_id
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
class TestGetDocumentByExternalId:
    """DYT-2674: lookup must return rows regardless of status."""

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self) -> None:
        """No row matching (namespace, external_id) returns None."""
        ns_id = uuid4()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_document_by_external_id(ns_id, "missing-ext")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_completed_document(self) -> None:
        """A COMPLETED row is returned."""
        ns_id = uuid4()
        model = _mock_document_model(namespace_id=ns_id, external_id="ext-42", status=DocumentStatus.COMPLETED)
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = model
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_document_by_external_id(ns_id, "ext-42")

        assert result is not None
        assert result.id == model.id

    @pytest.mark.asyncio
    async def test_returns_failed_document(self) -> None:
        """A FAILED row MUST be returned (enables self-heal)."""
        ns_id = uuid4()
        model = _mock_document_model(namespace_id=ns_id, external_id="ext-42", status=DocumentStatus.FAILED)
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = model
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_document_by_external_id(ns_id, "ext-42")

        assert result is not None
        assert result.status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_returns_processing_document(self) -> None:
        """A PROCESSING row MUST be returned."""
        ns_id = uuid4()
        model = _mock_document_model(namespace_id=ns_id, external_id="ext-42", status=DocumentStatus.PROCESSING)
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = model
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        result = await backend.get_document_by_external_id(ns_id, "ext-42")

        assert result is not None
        assert result.status == DocumentStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_query_does_not_filter_by_status(self) -> None:
        """The compiled SQL must NOT constrain status (unlike checksum lookup)."""
        ns_id = uuid4()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        backend = _make_backend(session)
        await backend.get_document_by_external_id(ns_id, "ext-1")

        call_args = session.execute.call_args
        stmt = call_args[0][0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        # Must filter on external_id in WHERE, but WHERE must NOT mention status.
        assert "external_id" in compiled
        where_clause = compiled.lower().split("where", 1)[1]
        assert "status" not in where_clause

    @pytest.mark.asyncio
    async def test_none_external_id_short_circuits(self) -> None:
        """external_id=None returns None without hitting the DB."""
        ns_id = uuid4()
        session = AsyncMock()
        session.execute = AsyncMock()

        backend = _make_backend(session)
        result = await backend.get_document_by_external_id(ns_id, None)

        assert result is None
        session.execute.assert_not_awaited()

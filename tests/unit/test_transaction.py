"""Unit tests for TransactionContext and StorageCoordinator.transaction()."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from khora.storage.coordinator import StorageCoordinator, TransactionContext


class TestTransactionContext:
    """Tests for TransactionContext dataclass."""

    def test_holds_session(self):
        session = MagicMock()
        ctx = TransactionContext(session=session)
        assert ctx.session is session

    @pytest.mark.asyncio
    async def test_savepoint_uses_begin_nested(self):
        """savepoint() calls session.begin_nested()."""
        mock_session = MagicMock()
        mock_nested = AsyncMock()
        mock_nested.__aenter__ = AsyncMock()
        mock_nested.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin_nested.return_value = mock_nested

        ctx = TransactionContext(session=mock_session)
        async with ctx.savepoint() as sp:
            assert sp is ctx  # same TransactionContext yielded
        mock_session.begin_nested.assert_called_once()


class TestCoordinatorTransaction:
    """Tests for StorageCoordinator.transaction()."""

    def _make_coordinator(self, *, with_relational=True):
        """Create coordinator with mock backends."""
        coord = StorageCoordinator()
        if with_relational:
            mock_backend = MagicMock()
            mock_session = AsyncMock()
            mock_factory = MagicMock(return_value=mock_session)
            mock_backend._session_factory = mock_factory
            coord.relational = mock_backend
        return coord

    @pytest.mark.asyncio
    async def test_transaction_commits_on_success(self):
        """Session is committed when context exits normally."""
        coord = self._make_coordinator()
        mock_session = coord._relational._session_factory.return_value

        async with coord.transaction() as txn:
            assert isinstance(txn, TransactionContext)
            assert txn.session is mock_session

        mock_session.commit.assert_awaited_once()
        mock_session.rollback.assert_not_awaited()
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_exception(self):
        """Session is rolled back when exception is raised."""
        coord = self._make_coordinator()
        mock_session = coord._relational._session_factory.return_value

        with pytest.raises(ValueError, match="boom"):
            async with coord.transaction():
                raise ValueError("boom")

        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_awaited()
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transaction_raises_without_backend(self):
        """transaction() raises RuntimeError when no SQL backend available."""
        coord = StorageCoordinator()
        with pytest.raises(RuntimeError, match="No SQL backend connected"):
            async with coord.transaction():
                pass

    @pytest.mark.asyncio
    async def test_transaction_falls_back_to_vector_backend(self):
        """transaction() uses vector backend when relational is missing."""
        coord = StorageCoordinator()
        mock_backend = MagicMock()
        mock_session = AsyncMock()
        mock_backend._session_factory = MagicMock(return_value=mock_session)
        coord.vector = mock_backend

        async with coord.transaction() as txn:
            assert txn.session is mock_session

        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transaction_falls_back_to_event_store(self):
        """transaction() uses event_store when relational and vector are missing."""
        coord = StorageCoordinator()
        mock_backend = MagicMock()
        mock_session = AsyncMock()
        mock_backend._session_factory = MagicMock(return_value=mock_session)
        coord.event_store = mock_backend

        async with coord.transaction() as txn:
            assert txn.session is mock_session

        mock_session.commit.assert_awaited_once()


class TestBackendSessionParam:
    """Tests that backend write methods accept an optional session parameter."""

    @pytest.mark.asyncio
    async def test_postgresql_create_document_with_session(self):
        """create_document uses provided session (no self-managed commit)."""
        from khora.storage.backends.postgresql import PostgreSQLBackend

        backend = PostgreSQLBackend("postgresql+asyncpg://test")
        mock_session = AsyncMock()
        mock_session.refresh = AsyncMock()

        # Mock the model returned from refresh
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.namespace_id = "ns-id"

        mock_doc = MagicMock()
        mock_doc.id = "test-id"
        mock_doc.namespace_id = "ns-id"
        mock_doc.content = "test"
        mock_doc.status = "pending"
        mock_doc.source = None
        mock_doc.source_type = None
        mock_doc.content_type = "text/plain"
        mock_doc.title = "test"
        mock_doc.author = None
        mock_doc.language = None
        mock_doc.checksum = "abc"
        mock_doc.size_bytes = 4
        mock_doc.metadata = {}
        mock_doc.chunk_count = 0
        mock_doc.entity_count = 0
        mock_doc.error_message = None
        mock_doc.created_at = None
        mock_doc.updated_at = None
        mock_doc.processed_at = None

        with patch.object(backend, "_document_model_to_domain", return_value=mock_doc):
            await backend.create_document(mock_doc, session=mock_session)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pgvector_create_chunks_batch_with_session(self):
        """create_chunks_batch uses provided session without committing."""
        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://test")
        mock_session = AsyncMock()

        mock_chunk = MagicMock()
        mock_chunk.id = "c-id"
        mock_chunk.namespace_id = "ns-id"
        mock_chunk.document_id = "d-id"
        mock_chunk.content = "test"
        mock_chunk.chunk_index = 0
        mock_chunk.start_char = 0
        mock_chunk.end_char = 4
        mock_chunk.token_count = 1
        mock_chunk.metadata = {}
        mock_chunk.chunker_info = {}
        mock_chunk.embedding = None
        mock_chunk.embedding_model = None
        mock_chunk.created_at = None

        result = await backend.create_chunks_batch([mock_chunk], session=mock_session)

        mock_session.add_all.assert_called_once()
        mock_session.commit.assert_not_awaited()
        assert result == [mock_chunk]

    @pytest.mark.asyncio
    async def test_event_store_append_event_with_session(self):
        """append_event uses provided session without committing."""
        from khora.storage.event_store import PostgreSQLEventStore

        store = PostgreSQLEventStore("postgresql+asyncpg://test")
        mock_session = AsyncMock()
        mock_session.refresh = AsyncMock()

        mock_event = MagicMock()
        mock_event.id = "e-id"
        mock_event.namespace_id = "ns-id"
        mock_event.event_type = "create"
        mock_event.timestamp = None
        mock_event.resource_type = "document"
        mock_event.resource_id = "r-id"
        mock_event.data = {}
        mock_event.previous_data = None
        mock_event.actor_id = None
        mock_event.actor_type = None
        mock_event.correlation_id = None
        mock_event.version = 1
        mock_event.metadata = {}

        with patch.object(store, "_model_to_domain", return_value=mock_event):
            await store.append_event(mock_event, session=mock_session)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pgvector_delete_chunks_by_document_with_session(self):
        """delete_chunks_by_document uses provided session without committing."""
        from uuid import uuid4

        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://test")
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 5
        mock_session.execute = AsyncMock(return_value=mock_result)

        doc_id = uuid4()
        count = await backend.delete_chunks_by_document(doc_id, session=mock_session)

        assert count == 5
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pgvector_delete_chunks_by_document_without_session(self):
        """delete_chunks_by_document opens own session and commits when no session provided."""
        from uuid import uuid4

        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://test")

        mock_result = MagicMock()
        mock_result.rowcount = 3

        mock_own_session = AsyncMock()
        mock_own_session.execute = AsyncMock(return_value=mock_result)
        mock_own_session.__aenter__ = AsyncMock(return_value=mock_own_session)
        mock_own_session.__aexit__ = AsyncMock(return_value=False)

        with patch.object(backend, "_get_session", return_value=mock_own_session):
            doc_id = uuid4()
            count = await backend.delete_chunks_by_document(doc_id)

        assert count == 3
        mock_own_session.execute.assert_awaited_once()
        mock_own_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pgvector_delete_chunks_by_document_no_chunks(self):
        """delete_chunks_by_document returns 0 when document has no chunks."""
        from uuid import uuid4

        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://test")
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute = AsyncMock(return_value=mock_result)

        count = await backend.delete_chunks_by_document(uuid4(), session=mock_session)
        assert count == 0

    @pytest.mark.asyncio
    async def test_pgvector_delete_chunks_by_document_error_propagates(self):
        """delete_chunks_by_document propagates execute errors."""
        from uuid import uuid4

        from khora.storage.backends.pgvector import PgVectorBackend

        backend = PgVectorBackend("postgresql+asyncpg://test")
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=RuntimeError("connection lost"))

        with pytest.raises(RuntimeError, match="connection lost"):
            await backend.delete_chunks_by_document(uuid4(), session=mock_session)

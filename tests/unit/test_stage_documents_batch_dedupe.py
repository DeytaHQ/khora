"""Unit tests for intra-batch dedup in stage_documents_batch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.pipelines.flows.ingest import compute_checksum, stage_documents_batch

NAMESPACE_ID = uuid4()


def _make_existing_doc(checksum: str) -> MagicMock:
    """Create a mock existing document (returned by get_documents_by_checksums)."""
    doc = MagicMock()
    doc.id = uuid4()
    doc.namespace_id = NAMESPACE_ID
    doc.status = "completed"
    doc.checksum = checksum
    return doc


def _make_storage(existing_checksums: list[str] | None = None):
    """Create a mock StorageCoordinator.

    Args:
        existing_checksums: checksums that already exist in DB.
    """
    storage = AsyncMock()
    existing = {}
    for cs in existing_checksums or []:
        existing[cs] = _make_existing_doc(cs)
    storage.get_documents_by_checksums = AsyncMock(return_value=existing)
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    return storage


def _doc_input(content: str) -> dict:
    return {"content": content, "source": "test"}


class TestStageDocumentsBatchDedupe:
    """Tests for intra-batch dedup in stage_documents_batch."""

    @pytest.mark.asyncio
    async def test_two_identical_docs(self) -> None:
        """Two identical docs -> create_document called once, both results same object."""
        storage = _make_storage()
        inputs = [_doc_input("same content"), _doc_input("same content")]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 2
        assert storage.create_document.call_count == 1
        assert results[0] is not None
        assert results[1] is not None
        assert results[0] is results[1]

        # L1: verify get_documents_by_checksums called with correct args
        checksum = compute_checksum("same content")
        storage.get_documents_by_checksums.assert_awaited_once_with(NAMESPACE_ID, [checksum])

    @pytest.mark.asyncio
    async def test_three_docs_two_identical_one_unique(self) -> None:
        """Three docs (two identical + one unique) -> create_document called twice."""
        storage = _make_storage()
        inputs = [
            _doc_input("duplicate"),
            _doc_input("unique"),
            _doc_input("duplicate"),
        ]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 3
        assert storage.create_document.call_count == 2
        assert all(r is not None for r in results)
        # First and third are the same object (intra-batch dup)
        assert results[0] is results[2]
        # Second is distinct
        assert results[1] is not results[0]

    @pytest.mark.asyncio
    async def test_db_dedup_returns_none(self) -> None:
        """Doc with checksum already in DB -> result is None, create_document not called."""
        content = "already exists"
        checksum = compute_checksum(content)
        storage = _make_storage(existing_checksums=[checksum])
        inputs = [_doc_input(content)]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 1
        assert results[0] is None
        assert storage.create_document.call_count == 0

    @pytest.mark.asyncio
    async def test_mixed_scenario(self) -> None:
        """Mix of DB-hit, intra-batch dup, and unique doc.

        - doc 0: matches DB -> None
        - doc 1 & 2: identical new content -> create once, both same object
        - doc 3: unique new content -> create once
        Total create_document calls: 2
        """
        db_content = "in database"
        db_checksum = compute_checksum(db_content)
        storage = _make_storage(existing_checksums=[db_checksum])
        inputs = [
            _doc_input(db_content),
            _doc_input("new duplicate"),
            _doc_input("new duplicate"),
            _doc_input("unique new"),
        ]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 4
        assert results[0] is None
        assert results[1] is not None
        assert results[2] is not None
        assert results[1] is results[2]
        assert results[3] is not None
        assert results[3] is not results[1]
        assert storage.create_document.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        """Empty batch -> empty list, no storage calls."""
        storage = _make_storage()

        results = await stage_documents_batch([], NAMESPACE_ID, storage)

        assert results == []
        storage.get_documents_by_checksums.assert_not_called()
        storage.create_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_docs_exist_in_db(self) -> None:
        """All docs already in DB -> all results None, create_document never called."""
        contents = ["aaa", "bbb", "ccc"]
        checksums = [compute_checksum(c) for c in contents]
        storage = _make_storage(existing_checksums=checksums)
        inputs = [_doc_input(c) for c in contents]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 3
        assert all(r is None for r in results)
        assert storage.create_document.call_count == 0

    @pytest.mark.asyncio
    async def test_large_batch_with_many_dups(self) -> None:
        """15 identical docs -> create_document called once, all results same object."""
        storage = _make_storage()
        inputs = [_doc_input("repeated") for _ in range(15)]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 15
        assert storage.create_document.call_count == 1
        assert all(r is not None for r in results)
        assert all(r is results[0] for r in results)

    @pytest.mark.asyncio
    async def test_failed_doc_not_treated_as_duplicate(self) -> None:
        """FAILED doc excluded by storage layer -> new doc created.

        The storage layer now filters out FAILED docs from checksum lookups,
        so get_documents_by_checksums returns empty. stage_documents_batch
        should then create a new document instead of skipping.
        """
        content = "previously failed"
        storage = _make_storage()  # empty = simulates FAILED doc filtered out
        inputs = [_doc_input(content)]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 1
        assert results[0] is not None
        assert storage.create_document.call_count == 1

    @pytest.mark.asyncio
    async def test_failed_and_completed_docs_mixed(self) -> None:
        """COMPLETED doc skipped, FAILED doc (filtered out) gets re-created.

        - doc 0: checksum matches a COMPLETED doc in DB -> None (skipped)
        - doc 1: checksum had a FAILED doc, storage filtered it out -> new doc created
        """
        completed_content = "completed doc"
        completed_checksum = compute_checksum(completed_content)
        failed_content = "previously failed doc"
        # Storage returns only the COMPLETED doc; FAILED doc is filtered out
        storage = _make_storage(existing_checksums=[completed_checksum])
        inputs = [_doc_input(completed_content), _doc_input(failed_content)]

        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 2
        assert results[0] is None  # COMPLETED doc skipped
        assert results[1] is not None  # FAILED doc re-created
        assert storage.create_document.call_count == 1


class TestStageDocumentsBatchIdentityScopedDedup:
    """#1171: checksum dedup in stage_documents_batch is scoped by identity."""

    def _make_existing_doc_with_identity(
        self,
        checksum: str,
        *,
        external_id: str | None = None,
        session_id=None,
    ) -> MagicMock:
        doc = MagicMock()
        doc.id = uuid4()
        doc.status = "completed"
        doc.checksum = checksum
        doc.external_id = external_id
        doc.session_id = session_id
        return doc

    def _make_storage_with_identity(self, existing_doc, checksum: str):
        storage = AsyncMock()
        storage.get_documents_by_checksums = AsyncMock(return_value={checksum: existing_doc})
        storage.create_document = AsyncMock(side_effect=lambda doc: doc)
        return storage

    @pytest.mark.asyncio
    async def test_same_content_new_session_id_creates_document(self) -> None:
        """#1171: same content + new session_id in batch MUST NOT be silently dropped."""
        from uuid import uuid4 as _uuid4

        session_a = _uuid4()
        session_b = _uuid4()
        content = "same content here"
        checksum = compute_checksum(content)

        existing = self._make_existing_doc_with_identity(checksum, external_id=None, session_id=session_a)
        storage = self._make_storage_with_identity(existing, checksum)

        inputs = [
            {"content": content, "metadata": {"session_id": str(session_b)}},
        ]
        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 1
        assert results[0] is not None, "New session_id must create a new document, not be dropped"
        assert storage.create_document.call_count == 1
        assert storage.create_document.call_args[0][0].session_id == session_b

    @pytest.mark.asyncio
    async def test_same_content_new_external_id_creates_document(self) -> None:
        """#1171: same content + new external_id in batch MUST NOT be silently dropped."""
        content = "same content here"
        checksum = compute_checksum(content)

        existing = self._make_existing_doc_with_identity(checksum, external_id="ext-a", session_id=None)
        storage = self._make_storage_with_identity(existing, checksum)

        inputs = [
            {"content": content, "external_id": "ext-b"},
        ]
        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 1
        assert results[0] is not None, "New external_id must create a new document, not be dropped"
        assert storage.create_document.call_count == 1
        assert storage.create_document.call_args[0][0].external_id == "ext-b"

    @pytest.mark.asyncio
    async def test_same_content_same_session_still_dedups(self) -> None:
        """#1171: same content + same session_id is still a legitimate duplicate."""
        from uuid import uuid4 as _uuid4

        session = _uuid4()
        content = "same content here"
        checksum = compute_checksum(content)

        existing = self._make_existing_doc_with_identity(checksum, external_id=None, session_id=session)
        storage = self._make_storage_with_identity(existing, checksum)

        inputs = [
            {"content": content, "metadata": {"session_id": str(session)}},
        ]
        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 1
        assert results[0] is None, "Same session_id + same checksum should still be a duplicate"
        assert storage.create_document.call_count == 0

    @pytest.mark.asyncio
    async def test_same_content_no_identity_still_dedups(self) -> None:
        """#1171: caller supplies no external_id/session_id -> checksum-only dedup."""
        from uuid import uuid4 as _uuid4

        content = "same content here"
        checksum = compute_checksum(content)

        existing = self._make_existing_doc_with_identity(checksum, external_id=None, session_id=_uuid4())
        storage = self._make_storage_with_identity(existing, checksum)

        inputs = [
            {"content": content},
        ]
        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 1
        assert results[0] is None, "No identity supplied -> checksum-only dedup applies"
        assert storage.create_document.call_count == 0

    @pytest.mark.asyncio
    async def test_intra_batch_same_content_different_session_ids_both_created(self) -> None:
        """#1171: two same-content docs with different session_ids in one batch are BOTH created."""
        from uuid import uuid4 as _uuid4

        session_a = _uuid4()
        session_b = _uuid4()
        content = "shared content"
        storage = AsyncMock()
        storage.get_documents_by_checksums = AsyncMock(return_value={})  # nothing in DB
        storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        inputs = [
            {"content": content, "metadata": {"session_id": str(session_a)}},
            {"content": content, "metadata": {"session_id": str(session_b)}},
        ]
        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 2
        assert results[0] is not None, "First doc (session_a) must be created"
        assert results[1] is not None, "Second doc (session_b) must be created"
        assert storage.create_document.call_count == 2, "Both docs must be created independently"

    @pytest.mark.asyncio
    async def test_intra_batch_same_content_different_external_ids_both_created(self) -> None:
        """#1171: two same-content docs with different external_ids in one batch are BOTH created."""
        content = "shared content"
        storage = AsyncMock()
        storage.get_documents_by_checksums = AsyncMock(return_value={})
        storage.create_document = AsyncMock(side_effect=lambda doc: doc)

        inputs = [
            {"content": content, "external_id": "ext-a"},
            {"content": content, "external_id": "ext-b"},
        ]
        results = await stage_documents_batch(inputs, NAMESPACE_ID, storage)

        assert len(results) == 2
        assert results[0] is not None, "First doc (ext-a) must be created"
        assert results[1] is not None, "Second doc (ext-b) must be created"
        assert storage.create_document.call_count == 2, "Both docs must be created independently"

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
    doc.metadata.checksum = checksum
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

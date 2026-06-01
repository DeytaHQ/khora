"""Regression: the pending-processor queue must not hold document content (#932).

``_ProcessorItem`` used to carry the full ``Document`` (including its entire
``content`` string), so a large batch - or a restart recovering many stale
PENDING rows - materialized the whole backlog's content in RAM at once
(``max_chunks_in_flight`` / ``max_concurrent`` only apply *after* dequeue).

The fix makes the queued item lean: it carries only a document identity
(``doc_id`` + ``namespace_id``) plus the batch registration. The worker
re-loads the full Document from storage at dequeue time. These tests assert
the fixed state:

1. The queued item holds no document content (no ``doc`` / ``content``).
2. The worker re-loads content by id and processes it.
3. A document that has gone missing between enqueue and dequeue is skipped
   gracefully (ADR-001) without crashing the worker and without leaving a
   waiting ``BatchHandle`` hung.

Storage is mocked; no external services.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.khora import _ProcessorItem

from .helpers import make_kb


def _kb_with_store(ns_id):
    """Build a connected kb whose storage persists docs in an in-memory map.

    ``create_document`` stores by id; ``get_document`` answers from the same
    map so the worker's #932 re-load finds the persisted row.
    """
    kb = make_kb(connected=True)
    store: dict = {}

    async def _create(doc):
        store[doc.id] = doc
        return doc

    async def _get(doc_id, *, namespace_id):
        return store.get(doc_id)

    kb._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
    kb._engine._storage.create_document = AsyncMock(side_effect=_create)
    kb._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)
    kb._engine._storage.get_document = AsyncMock(side_effect=_get)
    kb._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})
    return kb, store


@pytest.mark.unit
class TestProcessorItemLean:
    def test_item_carries_no_document_content(self) -> None:
        """A queued item holds an identity, not the Document or its content."""
        field_names = {f.name for f in fields(_ProcessorItem)}
        # The fix: identity only. No ``doc`` (and therefore no ``.content``).
        assert "doc" not in field_names
        assert "doc_data" not in field_names
        assert {"doc_id", "namespace_id", "batch_reg"} <= field_names

        big_content = "x" * 10_000_000  # 10 MB; would dominate RAM if held.
        item = _ProcessorItem(doc_id=uuid4(), namespace_id=uuid4(), batch_reg=None)
        # The item must not reference the content anywhere on it.
        for f in field_names:
            assert getattr(item, f) is not big_content
        assert not hasattr(item, "doc")
        assert not hasattr(item, "content")

    @pytest.mark.asyncio
    async def test_worker_reloads_content_by_id_and_processes(self) -> None:
        """The worker re-loads the Document by id, then processes it."""
        ns_id = uuid4()
        kb, _store = _kb_with_store(ns_id)

        seen_content: list[str] = []

        async def _process(doc, **kw):
            # The re-loaded Document must carry the full content.
            seen_content.append(doc.content)
            return (3, 2, 1)

        kb._engine.process_staged_document = AsyncMock(side_effect=_process)
        kb.start_pending_processor()

        results: list = []
        handle = await kb.submit_batch(
            [{"content": "hello world"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await asyncio.wait_for(handle.wait(), timeout=2.0)

        # get_document was used to re-load (not the queued item).
        kb._engine._storage.get_document.assert_awaited()
        assert seen_content == ["hello world"]
        assert handle.completed == 1
        assert handle.failed == 0
        assert results[0].success is True
        assert results[0].chunks_created == 3

    @pytest.mark.asyncio
    async def test_missing_document_at_dequeue_skipped_without_hang(self) -> None:
        """A row gone between enqueue and dequeue is skipped, not crashed.

        ADR-001 skip: the worker must not crash, the document must not be
        processed, and the waiting BatchHandle must still complete so
        ``wait()`` returns.
        """
        ns_id = uuid4()
        kb, store = _kb_with_store(ns_id)

        async def _create_then_forget(doc):
            # Persist so submit_batch's enqueue succeeds, then immediately
            # delete so the worker's re-load returns None at dequeue.
            store[doc.id] = doc
            return doc

        kb._engine._storage.create_document = AsyncMock(side_effect=_create_then_forget)
        process_mock = AsyncMock(side_effect=lambda doc, **kw: (1, 0, 0))
        kb._engine.process_staged_document = process_mock
        kb.start_pending_processor()

        results: list = []
        handle = await kb.submit_batch(
            [{"content": "doomed"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        # Forget the doc before the worker dequeues it.
        store.clear()

        # Must not hang despite the missing row.
        await asyncio.wait_for(handle.wait(), timeout=2.0)

        # The doc was never processed (it no longer existed).
        process_mock.assert_not_awaited()
        # The batch still completed so wait() returned; reported as skipped.
        assert handle.is_done is True
        assert handle.completed == 1
        assert handle.failed == 0
        assert len(results) == 1
        assert results[0].skipped is True
        assert results[0].success is True

        # The worker survives and keeps draining: a second, present doc still
        # processes after the skipped one.
        store2: dict = {}

        async def _create2(doc):
            store2[doc.id] = doc
            return doc

        async def _get2(doc_id, *, namespace_id):
            return store2.get(doc_id)

        kb._engine._storage.create_document = AsyncMock(side_effect=_create2)
        kb._engine._storage.get_document = AsyncMock(side_effect=_get2)
        process_mock.reset_mock()

        handle2 = await kb.submit_batch(
            [{"content": "present"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await asyncio.wait_for(handle2.wait(), timeout=2.0)
        process_mock.assert_awaited()
        assert handle2.failed == 0

    @pytest.mark.asyncio
    async def test_loader_raises_fires_result_exactly_once(self) -> None:
        """A load failure at dequeue must fire fire_result exactly once.

        Regression: the worker error path used to re-load on the failure
        branch (``else await self._load_pending_document(item)``). If the
        first load raised and the re-load then returned None, the loader
        fired a skip result *and* the error path fired a failure result -
        two results for one doc, overshooting BatchHandle._completed and
        double-firing on_result. The fix splits the load into its own try so
        a load failure fires exactly one failure result and never reaches the
        processing error handler.
        """
        ns_id = uuid4()
        kb, store = _kb_with_store(ns_id)

        # First get_document call raises (transient DB error); never returns.
        kb._engine._storage.get_document = AsyncMock(side_effect=RuntimeError("connection exhausted"))
        process_mock = AsyncMock(side_effect=lambda doc, **kw: (1, 0, 0))
        kb._engine.process_staged_document = process_mock
        kb.start_pending_processor()

        results: list = []
        handle = await kb.submit_batch(
            [{"content": "doc"}],
            on_result=lambda c, t, r: results.append(r),
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await asyncio.wait_for(handle.wait(), timeout=2.0)

        # The doc was never processed (load failed before processing).
        process_mock.assert_not_awaited()
        # Exactly one result, no overshoot past total.
        assert handle.total == 1
        assert handle.completed == 1
        assert handle.completed <= handle.total
        assert handle.failed == 1
        assert len(results) == 1
        assert results[0].success is False
        assert "connection exhausted" in (results[0].error or "")

        # The worker survives a load fault and keeps draining.
        store2: dict = {}

        async def _create2(doc):
            store2[doc.id] = doc
            return doc

        async def _get2(doc_id, *, namespace_id):
            return store2.get(doc_id)

        kb._engine._storage.create_document = AsyncMock(side_effect=_create2)
        kb._engine._storage.get_document = AsyncMock(side_effect=_get2)

        handle2 = await kb.submit_batch(
            [{"content": "present"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await asyncio.wait_for(handle2.wait(), timeout=2.0)
        process_mock.assert_awaited()
        assert handle2.failed == 0

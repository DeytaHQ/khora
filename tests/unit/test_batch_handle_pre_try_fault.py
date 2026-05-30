"""Regression: BatchHandle.wait() must not hang on a pre-try worker fault (#869).

``_process_pending_item_impl`` does meaningful work before its inner
``try`` block: engine resolution, ``getattr(process_fn)``, pre-FAILED state
cleanup, ``parse_dt``, and ``start_usage_collection``. If any of that
raises, the inner ``try`` never runs and
``batch_reg.fire_result(success=False)`` is never called. Pre-fix the
worker only logged the exception, so ``_BatchRegistration._remaining``
never decremented, ``BatchHandle._done_event`` never fired, and
``await handle.wait()`` blocked forever.

The fix adds a worker-level fallback that fires a failure result and
flips the doc to FAILED when ``item.batch_reg is not None``. This test
patches ``start_usage_collection`` to raise (a pre-try call site) and
asserts ``wait()`` returns within a short timeout.

Storage is mocked; no external services.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from khora.core.models.document import DocumentStatus

from .helpers import make_kb


def _kb_with_processor(ns_id):
    kb = make_kb(connected=True)
    kb._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
    kb._engine._storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    kb._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)
    kb._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})
    # process_staged_document would normally succeed; the pre-try fault
    # prevents this call site from being reached.
    kb._engine.process_staged_document = AsyncMock(side_effect=lambda doc, **kw: (1, 0, 0))
    kb.start_pending_processor()
    return kb


@pytest.mark.unit
class TestBatchHandlePreTryFault:
    @pytest.mark.asyncio
    async def test_wait_returns_when_pre_try_call_raises(self) -> None:
        """A pre-try fault must not hang ``BatchHandle.wait()``."""
        ns_id = uuid4()
        kb = _kb_with_processor(ns_id)

        # Patch the import target inside ``_process_pending_item_impl``.
        # ``start_usage_collection`` is called right before the inner
        # ``try`` block, so raising here exercises the pre-try fault path.
        def _boom() -> None:
            raise RuntimeError("simulated pre-try fault")

        results: list = []
        with patch("khora.telemetry.context.start_usage_collection", side_effect=_boom):
            handle = await kb.submit_batch(
                [{"content": "doc"}],
                on_result=lambda c, t, r: results.append(r),
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )
            # Pre-fix: this would block forever and raise TimeoutError.
            await asyncio.wait_for(handle.wait(), timeout=2.0)

        assert handle.is_done is True
        assert handle.completed == 1
        assert handle.failed == 1
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None
        assert "simulated pre-try fault" in results[0].error

    @pytest.mark.asyncio
    async def test_doc_flipped_to_failed_on_pre_try_fault(self) -> None:
        """The fallback must mark the doc FAILED, not leave it in PROCESSING."""
        ns_id = uuid4()
        kb = _kb_with_processor(ns_id)

        # Capture every Document passed to update_document so we can check
        # the final state.
        seen: list = []

        async def _capture_update(doc):
            seen.append((doc.id, doc.status))
            return doc

        kb._engine._storage.update_document = AsyncMock(side_effect=_capture_update)

        def _boom() -> None:
            raise RuntimeError("simulated pre-try fault")

        with patch("khora.telemetry.context.start_usage_collection", side_effect=_boom):
            handle = await kb.submit_batch(
                [{"content": "doc"}],
                on_result=lambda c, t, r: None,
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )
            await asyncio.wait_for(handle.wait(), timeout=2.0)

        # The last update_document call must reflect the FAILED flip from
        # the worker-level fallback (earlier calls may be the PROCESSING
        # transition issued by submit_batch itself).
        assert seen, "expected at least one update_document call"
        assert seen[-1][1] == DocumentStatus.FAILED

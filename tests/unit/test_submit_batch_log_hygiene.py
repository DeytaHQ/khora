"""submit_batch persistence-failure warnings must not leak the document payload.

When ``create_document`` / ``update_document`` raise, submit_batch logs a
``WARNING`` and records the doc as pre-failed. A bare ``{exc}`` on a SQLAlchemy
``DBAPIError`` interpolates the *full failed statement and its bind-parameter
tuple* — which for a document INSERT/UPDATE is the entire document content and
metadata. That is a content leak and a log-bloat hazard.

The fix routes both warnings through ``_safe_exc_summary``, which prefers the
driver's clean SQLSTATE message (``exc.orig`` / ``__cause__``) over the outer
SQLAlchemy repr and truncates. These tests drive a sentinel ``secret`` payload
through the bind-param-bearing outer repr and assert it never reaches the
captured WARNING record — on both the normal-insert path (``create_document``)
and the re-queue/update path (``update_document``).

Storage is mocked; no external services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from loguru import logger

from khora.core.models.document import Document, DocumentStatus

from .helpers import make_kb

# Sentinel standing in for the sensitive document payload (content / metadata /
# any credential a row carries) that a SQLAlchemy DBAPIError echoes back inside
# its bind-parameter tuple.
_SECRET = "SUPER-SECRET-DSN-postgres://user:p@ssw0rd@db/khora"


class _FakeDriverError(Exception):
    """Driver-level error: carries only the clean SQLSTATE line, no bind params."""

    sqlstate = "22007"  # invalid_datetime_format — the asyncpg-DataError family


class _FakeDBAPIError(Exception):
    """SQLAlchemy-shaped wrapper.

    Its ``str()`` embeds the failed statement and the bind-param tuple (the
    secret-bearing document payload), exactly like a real
    ``sqlalchemy.exc.DBAPIError``. ``.orig`` exposes the clean driver error
    that ``_safe_exc_summary`` is meant to prefer.
    """

    def __init__(self, orig: Exception) -> None:
        self.orig = orig
        # Mimic a sqlalchemy.exc.DBAPIError repr: driver line, then the failed
        # statement, then the bind-param tuple carrying the document payload.
        statement_repr = (
            f"({type(orig).__module__}.{type(orig).__name__}) {orig}\n"
            "[SQL: <documents upsert>]\n"
            f"[parameters: (content={_SECRET!r}, source_timestamp='not-a-real-ts')]"
        )
        super().__init__(statement_repr)


def _bind_param_leak_error() -> _FakeDBAPIError:
    return _FakeDBAPIError(_FakeDriverError("invalid input syntax for type timestamp with time zone"))


def _kb_with_processor(ns_id):
    kb = make_kb(connected=True)
    kb._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)
    kb._engine._storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    kb._engine._storage.update_document = AsyncMock(side_effect=lambda doc: doc)
    kb._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})
    kb._engine.process_staged_document = AsyncMock(side_effect=lambda doc, **kw: (1, 0, 0))
    kb.start_pending_processor()
    return kb


@pytest.mark.unit
class TestSubmitBatchWarningLogHygiene:
    @pytest.mark.asyncio
    async def test_create_document_failure_warning_omits_payload(self) -> None:
        """A create_document DBAPIError does not leak the bind-param payload into the WARNING."""
        ns_id = uuid4()
        kb = _kb_with_processor(ns_id)

        async def _raise(doc):
            raise _bind_param_leak_error()

        kb._engine._storage.create_document = AsyncMock(side_effect=_raise)

        captured: list[str] = []
        results: list = []
        handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            handle = await kb.submit_batch(
                [{"content": "doc"}],
                on_result=lambda c, t, r: results.append(r),
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )
            await handle.wait()
        finally:
            logger.remove(handler_id)

        # A warning about the create failure should still be emitted...
        assert any("could not create document" in m for m in captured), (
            f"expected a create-failure warning; got: {captured}"
        )
        # ...but it must not echo the secret-bearing bind-param payload.
        assert not any(_SECRET in m for m in captured), f"payload leaked into a log record: {captured}"
        # The DocumentResult.error surfaced to on_result is sanitized too.
        assert results and results[0].success is False
        assert _SECRET not in (results[0].error or ""), (
            f"payload leaked into DocumentResult.error: {results[0].error!r}"
        )

    @pytest.mark.asyncio
    async def test_update_document_failure_warning_omits_payload(self) -> None:
        """An update_document DBAPIError does not leak the bind-param payload into the WARNING."""
        ns_id = uuid4()
        kb = _kb_with_processor(ns_id)

        # Pre-existing FAILED doc → re-queue/update path.
        existing = Document(
            id=uuid4(),
            namespace_id=ns_id,
            content="old",
            status=DocumentStatus.FAILED,
            external_id="ext-1",
        )
        kb._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-1": existing})

        async def _raise(doc):
            raise _bind_param_leak_error()

        kb._engine._storage.update_document = AsyncMock(side_effect=_raise)

        captured: list[str] = []
        results: list = []
        handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            handle = await kb.submit_batch(
                [{"content": "new", "external_id": "ext-1"}],
                on_result=lambda c, t, r: results.append(r),
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )
            await handle.wait()
        finally:
            logger.remove(handler_id)

        assert any("could not update document" in m for m in captured), (
            f"expected an update-failure warning; got: {captured}"
        )
        assert not any(_SECRET in m for m in captured), f"payload leaked into a log record: {captured}"
        # The DocumentResult.error surfaced to on_result is sanitized too.
        assert results and results[0].success is False
        assert _SECRET not in (results[0].error or ""), (
            f"payload leaked into DocumentResult.error: {results[0].error!r}"
        )


@pytest.mark.unit
class TestProcessStagedDocumentLogHygiene:
    """The pending-processor failure path must not leak the document payload.

    When ``process_staged_document`` raises (and again when the subsequent
    ``update_document`` status flip raises), ``_process_pending_item`` logs the
    failure and surfaces ``DocumentResult.error`` to ``on_result``. A
    payload-bearing DBAPIError on either surface would otherwise leak the
    bind-param tuple. These exercise the wrapped ERROR log, the status-update
    WARNING, and the ``DocumentResult.error`` together.
    """

    @pytest.mark.asyncio
    async def test_process_failure_omits_payload_from_log_and_result(self) -> None:
        """A process_staged_document DBAPIError (plus a failing status update) leaks nothing."""
        ns_id = uuid4()
        kb = _kb_with_processor(ns_id)

        async def _raise_process(doc, **kw):
            raise _bind_param_leak_error()

        async def _raise_update(doc):
            raise _bind_param_leak_error()

        # process_staged_document raises → hits the wrapped ERROR log + the
        # sanitized DocumentResult.error. update_document also raises → hits the
        # wrapped status-update WARNING in the same except block.
        kb._engine.process_staged_document = AsyncMock(side_effect=_raise_process)
        kb._engine._storage.update_document = AsyncMock(side_effect=_raise_update)

        # Capture WARNING and above — loguru's level= is a minimum threshold, so
        # this catches both the ERROR (process failure) and WARNING (status update).
        captured: list[str] = []
        results: list = []
        handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
        try:
            handle = await kb.submit_batch(
                [{"content": "doc"}],
                on_result=lambda c, t, r: results.append(r),
                namespace=ns_id,
                entity_types=["PERSON"],
                relationship_types=["KNOWS"],
            )
            await handle.wait()
        finally:
            logger.remove(handler_id)

        # The two wrapped failure logs should still be emitted...
        assert any("failed to process document" in m for m in captured), (
            f"expected a process-failure log; got: {captured}"
        )
        assert any("could not update document status" in m for m in captured), (
            f"expected a status-update-failure warning; got: {captured}"
        )
        # ...but neither may echo the secret-bearing bind-param payload.
        assert not any(_SECRET in m for m in captured), f"payload leaked into a log record: {captured}"
        # The DocumentResult.error surfaced to on_result is sanitized too.
        assert results and results[0].success is False
        assert _SECRET not in (results[0].error or ""), (
            f"payload leaked into DocumentResult.error: {results[0].error!r}"
        )


@pytest.mark.unit
class TestSafeExcSummary:
    """Direct unit tests for the ``_safe_exc_summary`` log-scrubber helper."""

    def test_prefers_clean_orig_over_bind_param_repr(self) -> None:
        from khora.khora import _safe_exc_summary

        summary = _safe_exc_summary(_bind_param_leak_error())
        assert _SECRET not in summary
        assert "invalid input syntax for type timestamp" in summary
        # Class-name prefix keeps it diagnostic.
        assert summary.startswith("_FakeDBAPIError:")

    def test_strips_sql_and_param_markers_without_orig(self) -> None:
        """Defense in depth: a wrapper lacking .orig still has its [SQL:]/[parameters:] tail cut."""
        from khora.khora import _safe_exc_summary

        # No .orig and no __cause__ — forces the str(exc) fallback branch.
        exc = RuntimeError(f"boom\n[SQL: <documents upsert>]\n[parameters: (content={_SECRET!r})]")
        summary = _safe_exc_summary(exc)
        assert _SECRET not in summary
        assert "[SQL:" not in summary
        assert "[parameters:" not in summary
        assert summary == "RuntimeError: boom"

    def test_truncates_long_messages(self) -> None:
        from khora.khora import _safe_exc_summary

        exc = RuntimeError("x" * 500)
        summary = _safe_exc_summary(exc, max_len=50)
        # Prefix + 50 chars + ellipsis; never the full 500-char body.
        assert len(summary) < 120
        assert summary.endswith("…")

    def test_empty_message_returns_class_name(self) -> None:
        from khora.khora import _safe_exc_summary

        assert _safe_exc_summary(RuntimeError()) == "RuntimeError"

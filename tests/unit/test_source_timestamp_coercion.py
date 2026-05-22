"""String ``source_timestamp`` is coerced to ``datetime`` before persistence.

Upstream connectors hand ISO-8601 *strings* to the public
``source_timestamp`` kwarg / per-doc dict value. The public field is typed
``datetime``, but the strings slipped straight through ``submit_batch`` /
``remember`` / ``remember_batch`` and into the relational store. On Postgres
that surfaced as an asyncpg ``DataError`` (str where a timestamptz is
expected); on lenient backends it silently stored a string that broke
temporal filtering on read-back.

These tests pin the coercion at the public boundary:
  * ``submit_batch`` coerces on BOTH the normal-insert path and the
    re-queue/update path (FAILED doc reset to PENDING).
  * ``remember`` / ``remember_batch`` coerce the kwarg and per-doc value.
  * the document reaching storage carries a real ``datetime`` (the exact
    condition whose absence raised the original asyncpg ``DataError``).

Storage is mocked — these are unit tests, no external services. The mocked
``create_document`` / ``update_document`` capture the ``Document`` the facade
hands the relational store so we can assert on its ``source_timestamp`` type.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from khora.core.models.document import Document, DocumentStatus
from khora.khora import BatchResult, RememberResult

from .helpers import RESOLVE_ROW_ID, make_kb

# ---------------------------------------------------------------------------
# submit_batch — capture the Document handed to storage
# ---------------------------------------------------------------------------


def _kb_capturing_persisted(ns_id):
    """Build a connected kb whose storage captures persisted Documents.

    Returns ``(kb, created, updated)`` where ``created`` / ``updated`` are
    lists that receive each ``Document`` passed to ``create_document`` /
    ``update_document``. The pending processor is started so submit_batch
    does not raise "pending processor is not running".
    """
    kb = make_kb(connected=True)
    kb._engine._storage.resolve_namespace = AsyncMock(return_value=ns_id)

    created: list[Document] = []
    updated: list[Document] = []

    async def _create(doc):
        created.append(doc)
        return doc

    async def _update(doc):
        updated.append(doc)
        return doc

    kb._engine._storage.create_document = AsyncMock(side_effect=_create)
    kb._engine._storage.update_document = AsyncMock(side_effect=_update)
    kb._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={})

    async def _process(doc, **kwargs):
        return (1, 0, 0)

    kb._engine.process_staged_document = AsyncMock(side_effect=_process)
    kb.start_pending_processor()
    return kb, created, updated


@pytest.mark.unit
class TestSubmitBatchCoercesSourceTimestamp:
    """submit_batch coerces string source_timestamp on both persistence paths."""

    @pytest.mark.asyncio
    async def test_normal_insert_path_persists_datetime(self) -> None:
        """A string source_timestamp in the doc dict reaches create_document as a datetime."""
        ns_id = uuid4()
        kb, created, _updated = _kb_capturing_persisted(ns_id)

        handle = await kb.submit_batch(
            [{"content": "hello", "source_timestamp": "2025-01-15T10:30:00Z"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(created) == 1
        persisted = created[0]
        assert isinstance(persisted.source_timestamp, datetime), (
            f"expected datetime, got {type(persisted.source_timestamp).__name__}"
        )
        assert persisted.source_timestamp == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_top_level_kwarg_string_is_coerced(self) -> None:
        """A string top-level source_timestamp kwarg is coerced for docs that omit one."""
        ns_id = uuid4()
        kb, created, _updated = _kb_capturing_persisted(ns_id)

        handle = await kb.submit_batch(
            [{"content": "hello"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            source_timestamp="2025-04-01",  # date-only string at the kwarg level
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(created) == 1
        assert isinstance(created[0].source_timestamp, datetime)
        assert created[0].source_timestamp == datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_requeue_update_path_persists_datetime(self) -> None:
        """A FAILED doc re-queued via update_document gets a coerced datetime."""
        ns_id = uuid4()
        kb, _created, updated = _kb_capturing_persisted(ns_id)

        # Pre-existing FAILED document keyed by external_id triggers the
        # re-queue/update path rather than the normal-insert path.
        existing = Document(
            id=uuid4(),
            namespace_id=ns_id,
            content="old content",
            status=DocumentStatus.FAILED,
            external_id="ext-requeue",
        )
        kb._engine._storage.get_documents_by_external_ids = AsyncMock(return_value={"ext-requeue": existing})

        handle = await kb.submit_batch(
            [
                {
                    "content": "new content",
                    "external_id": "ext-requeue",
                    "source_timestamp": "2025-01-15T10:30:00+00:00",
                }
            ],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert len(updated) == 1
        persisted = updated[0]
        assert isinstance(persisted.source_timestamp, datetime), (
            f"expected datetime, got {type(persisted.source_timestamp).__name__}"
        )
        assert persisted.source_timestamp == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_datetime_passthrough_unchanged(self) -> None:
        """A real datetime is persisted unchanged (no double-coercion regression)."""
        ns_id = uuid4()
        kb, created, _updated = _kb_capturing_persisted(ns_id)

        when = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        handle = await kb.submit_batch(
            [{"content": "hello", "source_timestamp": when}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert created[0].source_timestamp == when

    @pytest.mark.asyncio
    async def test_none_stays_none(self) -> None:
        """Omitting source_timestamp persists None (backward compatible)."""
        ns_id = uuid4()
        kb, created, _updated = _kb_capturing_persisted(ns_id)

        handle = await kb.submit_batch(
            [{"content": "hello"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        assert created[0].source_timestamp is None


@pytest.mark.unit
class TestSubmitBatchAsyncpgDataErrorRegression:
    """Regression shaped like the original asyncpg DataError failure.

    asyncpg raises ``DataError`` when a ``str`` is bound to a ``timestamptz``
    column. The root cause was submit_batch persisting the *string* verbatim.
    Pin that the Document reaching storage now carries a ``datetime`` (the
    type asyncpg accepts) on both the insert and re-queue paths — so the bind
    that previously raised ``DataError`` now succeeds.
    """

    @pytest.mark.asyncio
    async def test_string_timestamp_no_longer_reaches_storage_as_str(self) -> None:
        ns_id = uuid4()
        kb, created, _updated = _kb_capturing_persisted(ns_id)

        # The exact failing shape: an ISO string handed to the public API.
        handle = await kb.submit_batch(
            [{"content": "slack msg", "source_timestamp": "2025-01-15T10:30:00Z"}],
            on_result=lambda c, t, r: None,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )
        await handle.wait()

        # No document failed during persistence, and the persisted value is a
        # datetime — asyncpg would accept this bind, where a str raised DataError.
        assert handle.failed == 0
        assert not isinstance(created[0].source_timestamp, str)
        assert isinstance(created[0].source_timestamp, datetime)


# ---------------------------------------------------------------------------
# remember / remember_batch — coercion at the facade before the engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRememberCoercesSourceTimestamp:
    """remember() coerces a string source_timestamp kwarg before the engine."""

    @pytest.mark.asyncio
    async def test_string_kwarg_coerced_to_datetime(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()

        mock_result = RememberResult(
            document_id=uuid4(),
            namespace_id=RESOLVE_ROW_ID,
            chunks_created=1,
            entities_extracted=0,
            relationships_created=0,
        )
        kb._engine.remember = AsyncMock(return_value=mock_result)

        await kb.remember(
            "Alice works for Acme",
            namespace=ns_id,
            source_timestamp="2025-01-15T10:30:00Z",
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        forwarded = kb._engine.remember.call_args.kwargs["source_timestamp"]
        assert isinstance(forwarded, datetime)
        assert forwarded == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_none_kwarg_stays_none(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()
        kb._engine.remember = AsyncMock(
            return_value=RememberResult(
                document_id=uuid4(),
                namespace_id=RESOLVE_ROW_ID,
                chunks_created=0,
                entities_extracted=0,
                relationships_created=0,
            )
        )

        await kb.remember(
            "content",
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        assert kb._engine.remember.call_args.kwargs["source_timestamp"] is None


@pytest.mark.unit
class TestRememberBatchCoercesSourceTimestamp:
    """remember_batch() coerces the kwarg and per-doc dict source_timestamp."""

    @pytest.mark.asyncio
    async def test_per_doc_string_value_coerced(self) -> None:
        kb = make_kb(connected=True)
        ns_id = uuid4()
        kb._engine.remember_batch = AsyncMock(
            return_value=BatchResult(total=1, processed=1, skipped=0, failed=0, chunks=1, entities=0, relationships=0)
        )

        docs = [{"content": "msg", "source_timestamp": "2025-01-15T10:30:00Z"}]
        await kb.remember_batch(
            docs,
            namespace=ns_id,
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        passed_docs = kb._engine.remember_batch.call_args.args[0]
        ts = passed_docs[0]["source_timestamp"]
        assert isinstance(ts, datetime)
        assert ts == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_kwarg_default_string_fills_and_coerces(self) -> None:
        """A string top-level kwarg fills docs that omit source_timestamp, coerced."""
        kb = make_kb(connected=True)
        ns_id = uuid4()
        kb._engine.remember_batch = AsyncMock(
            return_value=BatchResult(total=1, processed=1, skipped=0, failed=0, chunks=1, entities=0, relationships=0)
        )

        docs = [{"content": "msg"}]  # no per-doc source_timestamp
        await kb.remember_batch(
            docs,
            namespace=ns_id,
            source_timestamp="2025-04-01",
            entity_types=["PERSON"],
            relationship_types=["KNOWS"],
        )

        passed_docs = kb._engine.remember_batch.call_args.args[0]
        ts = passed_docs[0]["source_timestamp"]
        assert isinstance(ts, datetime)
        assert ts == datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC)

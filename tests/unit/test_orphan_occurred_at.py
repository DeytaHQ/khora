"""Orphan-recovery occurred_at resolution parity (#1121).

`_process_pending_item_impl` resolves the chunk event time. On the normal
(batch) path it consults ``doc.metadata['occurred_at']`` via the engine's
``_parse_datetime`` before falling back to ``source_timestamp``. The
crash-recovery orphan path (``batch_reg is None``) must resolve it
identically - a document recovered after a crash with a persisted
``metadata['occurred_at']`` and no ``source_timestamp`` must stamp its
chunks with the event time, not the ingest time, so temporal recall does
not silently diverge between the two paths.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora import Khora, KhoraConfig
from khora.core.models.document import Document, DocumentStatus
from khora.khora import _ProcessorItem


class _FakeStorage:
    """Minimal storage stub - the orphan path never touches it on success."""

    vector = None

    async def update_document(self, doc: Document) -> None:  # pragma: no cover
        return None


class _FakeEngine:
    """Engine stub that captures the ``occurred_at`` passed to processing."""

    def __init__(self) -> None:
        self._storage = _FakeStorage()
        self.captured_occurred_at: datetime | None = None

    def _parse_datetime(self, value):
        # Mirror the real engine: ISO string -> aware datetime.
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    async def process_staged_document(self, doc, *, occurred_at, **kwargs):
        self.captured_occurred_at = occurred_at
        return (0, 0, 0)


def _make_kb_with_fake_engine() -> tuple[Khora, _FakeEngine]:
    kb = Khora(KhoraConfig())
    engine = _FakeEngine()
    kb._engine = engine
    return kb, engine


@pytest.mark.unit
async def test_orphan_path_uses_metadata_occurred_at() -> None:
    """Orphan recovery (batch_reg=None) honours metadata['occurred_at']."""
    event_time = datetime(2024, 3, 14, 9, 0, tzinfo=UTC)
    ingest_time = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)

    doc = Document(
        id=uuid4(),
        namespace_id=uuid4(),
        content="recovered after crash",
        status=DocumentStatus.PENDING,
        metadata={"occurred_at": event_time.isoformat()},
        source_timestamp=None,
        created_at=ingest_time,
    )

    kb, engine = _make_kb_with_fake_engine()
    item = _ProcessorItem(doc_id=doc.id, namespace_id=doc.namespace_id, batch_reg=None)

    await kb._process_pending_item_impl(item, doc)

    # Pre-fix: orphan path used doc.created_at (ingest time), ignoring
    # metadata['occurred_at']. After the fix it matches the normal path.
    assert engine.captured_occurred_at == event_time


@pytest.mark.unit
async def test_orphan_path_falls_back_to_created_at_without_metadata() -> None:
    """Without metadata occurred_at or source_timestamp, orphan uses created_at."""
    ingest_time = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
    doc = Document(
        id=uuid4(),
        namespace_id=uuid4(),
        content="no event time",
        status=DocumentStatus.PENDING,
        metadata={},
        source_timestamp=None,
        created_at=ingest_time,
    )

    kb, engine = _make_kb_with_fake_engine()
    item = _ProcessorItem(doc_id=doc.id, namespace_id=doc.namespace_id, batch_reg=None)

    await kb._process_pending_item_impl(item, doc)

    assert engine.captured_occurred_at == ingest_time


@pytest.mark.unit
async def test_orphan_path_prefers_source_timestamp_over_created_at() -> None:
    """source_timestamp still wins over created_at when no metadata event time."""
    src_time = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    ingest_time = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)
    doc = Document(
        id=uuid4(),
        namespace_id=uuid4(),
        content="has source timestamp",
        status=DocumentStatus.PENDING,
        metadata={},
        source_timestamp=src_time,
        created_at=ingest_time,
    )

    kb, engine = _make_kb_with_fake_engine()
    item = _ProcessorItem(doc_id=doc.id, namespace_id=doc.namespace_id, batch_reg=None)

    await kb._process_pending_item_impl(item, doc)

    assert engine.captured_occurred_at == src_time

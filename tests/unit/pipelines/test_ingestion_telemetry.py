"""Telemetry on the silent ingestion-fallback path (issue #568 Phase B).

Verifies that ``process_document`` increments the
``khora.ingest.source_timestamp.fallback_count`` counter and emits a
WARN log when a connector fails to provide a source-system timestamp.

The counter is mocked via the ``record_ingestion_fallback`` helper; the
log is captured through a temporary loguru sink because khora uses
loguru (not stdlib logging), so pytest's ``caplog`` would not see it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from loguru import logger


def _make_storage_mock() -> MagicMock:
    storage = MagicMock()
    storage.get_document_by_checksum = AsyncMock(return_value=None)
    storage.create_document = AsyncMock(side_effect=lambda doc: doc)
    storage.update_document = AsyncMock()
    storage.create_chunks_batch = AsyncMock()
    storage.upsert_entities_batch = AsyncMock(return_value=[])
    storage.update_entity_embeddings_batch = AsyncMock()
    storage.create_relationships_batch = AsyncMock(return_value=0)
    storage.list_entities = AsyncMock(return_value=[])
    storage.list_relationships = AsyncMock(return_value=[])
    return storage


def _make_chunk(ns_id, doc_id, content="test content"):
    from khora.core.models import Chunk, ChunkMetadata

    return Chunk(
        id=uuid4(),
        namespace_id=ns_id,
        document_id=doc_id,
        content=content,
        metadata=ChunkMetadata(),
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.now(UTC),
    )


def _make_document_mock(doc_id, ns_id, content, metadata_custom=None):
    doc = MagicMock()
    doc.id = doc_id
    doc.namespace_id = ns_id
    doc.content = content
    doc.metadata = MagicMock(custom=metadata_custom or {}, title="")
    doc.created_at = datetime.now(UTC)
    doc.mark_processing = MagicMock()
    doc.mark_completed = MagicMock()
    doc.mark_failed = MagicMock()
    doc.status = "pending"
    return doc


@pytest.fixture
def loguru_messages():
    """Capture loguru WARNING messages via a temporary sink."""
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    yield captured
    logger.remove(sink_id)


async def _run_process_document(document, storage, temporal_store):
    from khora.pipelines.flows.ingest import process_document

    chunks = [_make_chunk(document.namespace_id, document.id, content=document.content)]
    with (
        patch("khora.pipelines.tasks.chunk_document", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.embed_chunks", new=AsyncMock(return_value=chunks)),
        patch("khora.pipelines.tasks.extract_entities", new=AsyncMock(return_value=([], []))),
    ):
        await process_document(
            document,
            storage,
            temporal_store=temporal_store,
            entity_types=["PERSON"],
            relationship_types=["WORKS_FOR"],
        )
    return chunks


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_timestamp_present_no_fallback_no_warning(loguru_messages):
    """A connector that populates ``sent_at`` produces no fallback signal."""
    ns_id, doc_id = uuid4(), uuid4()
    document = _make_document_mock(
        doc_id,
        ns_id,
        "Alice met Bob.",
        metadata_custom={
            "source_system": "slack",
            "sent_at": "2026-05-13T14:00:00+00:00",
        },
    )
    storage = _make_storage_mock()
    temporal_store = MagicMock(create_chunks_batch=AsyncMock(return_value=[]))

    with patch("khora.telemetry.temporal_metrics.record_ingestion_fallback") as record_mock:
        await _run_process_document(document, storage, temporal_store)

    record_mock.assert_not_called()
    assert not any("fell back to ingest time" in m for m in loguru_messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slack_without_timestamp_fires_counter_and_warning(loguru_messages):
    """Slack connector that forgets ``sent_at`` produces both signals."""
    ns_id, doc_id = uuid4(), uuid4()
    document = _make_document_mock(
        doc_id,
        ns_id,
        "Alice met Bob.",
        metadata_custom={"source_system": "slack", "source_id": "S-12345"},
    )
    storage = _make_storage_mock()
    temporal_store = MagicMock(create_chunks_batch=AsyncMock(return_value=[]))

    with patch("khora.telemetry.temporal_metrics.record_ingestion_fallback") as record_mock:
        chunks = await _run_process_document(document, storage, temporal_store)

    # Counter fires once per chunk that fell back.
    assert record_mock.call_count == len(chunks)
    for call in record_mock.call_args_list:
        assert call.args == ("slack",) or call.kwargs == {"source_type": "slack"} or call.args[0] == "slack"

    # WARN log fires exactly once per document (throttled, not per-chunk).
    fallback_warns = [m for m in loguru_messages if "fell back to ingest time" in m]
    assert len(fallback_warns) == 1, fallback_warns
    assert "slack" in fallback_warns[0]
    assert f"document_id={doc_id}" in fallback_warns[0]
    assert "S-12345" in fallback_warns[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_manual_source_without_timestamp_increments_counter_but_no_warning(
    loguru_messages,
):
    """``manual`` is a legitimate timestamp-free source — counter only, no log."""
    ns_id, doc_id = uuid4(), uuid4()
    document = _make_document_mock(
        doc_id,
        ns_id,
        "Hand-uploaded note.",
        metadata_custom={"source_system": "manual"},
    )
    storage = _make_storage_mock()
    temporal_store = MagicMock(create_chunks_batch=AsyncMock(return_value=[]))

    with patch("khora.telemetry.temporal_metrics.record_ingestion_fallback") as record_mock:
        await _run_process_document(document, storage, temporal_store)

    record_mock.assert_called()
    # Counter receives the literal "manual" string — temporal_metrics buckets it.
    record_mock.assert_called_with("manual")
    assert not any("fell back to ingest time" in m for m in loguru_messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_source_system_increments_counter_with_none_no_warning(
    loguru_messages,
):
    """No ``source_system`` field means we can't blame the connector — log skipped."""
    ns_id, doc_id = uuid4(), uuid4()
    document = _make_document_mock(doc_id, ns_id, "Bare content.", metadata_custom={})
    storage = _make_storage_mock()
    temporal_store = MagicMock(create_chunks_batch=AsyncMock(return_value=[]))

    with patch("khora.telemetry.temporal_metrics.record_ingestion_fallback") as record_mock:
        await _run_process_document(document, storage, temporal_store)

    record_mock.assert_called()
    # source_type is None; ``record_ingestion_fallback`` will bucket it to "unknown".
    record_mock.assert_called_with(None)
    assert not any("fell back to ingest time" in m for m in loguru_messages)


@pytest.mark.unit
def test_record_ingestion_fallback_buckets_off_enum_to_unknown():
    """Direct unit test for the metric helper's label bounding."""
    from khora.telemetry import temporal_metrics

    fake_counter = MagicMock()
    with patch.object(temporal_metrics, "_get_ingest_fallback", return_value=fake_counter):
        temporal_metrics.record_ingestion_fallback("slack")
        temporal_metrics.record_ingestion_fallback("totally-made-up-source")
        temporal_metrics.record_ingestion_fallback(None)
        temporal_metrics.record_ingestion_fallback("manual")

    calls = fake_counter.add.call_args_list
    assert len(calls) == 4
    labels = [c.kwargs["attributes"]["source_type"] for c in calls]
    assert labels == ["slack", "unknown", "unknown", "manual"]

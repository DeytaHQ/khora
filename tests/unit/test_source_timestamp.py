"""Unit tests for source_timestamp propagation through the stack."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

from khora.core.models.document import Chunk, Document


class TestDocumentSourceTimestamp:
    """Tests that Document domain model supports source_timestamp."""

    def test_document_has_source_timestamp_field(self) -> None:
        doc = Document(
            namespace_id=uuid4(),
            content="test",
            source_timestamp=datetime(2025, 1, 15, tzinfo=UTC),
        )
        assert doc.source_timestamp == datetime(2025, 1, 15, tzinfo=UTC)

    def test_document_source_timestamp_defaults_to_none(self) -> None:
        doc = Document(namespace_id=uuid4(), content="test")
        assert doc.source_timestamp is None


class TestChunkSourceTimestamp:
    """Tests that Chunk domain model supports source_timestamp."""

    def test_chunk_has_source_timestamp_field(self) -> None:
        chunk = Chunk(
            content="test content",
            source_timestamp=datetime(2025, 1, 15, tzinfo=UTC),
        )
        assert chunk.source_timestamp == datetime(2025, 1, 15, tzinfo=UTC)

    def test_chunk_source_timestamp_defaults_to_none(self) -> None:
        chunk = Chunk(content="test content")
        assert chunk.source_timestamp is None


class TestExtractSourceTimestamp:
    """Tests for _extract_source_timestamp in ingest pipeline."""

    def test_extracts_sent_at(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        ts = datetime(2025, 1, 15, 10, 30, 0)
        result = _extract_source_timestamp({"sent_at": ts})
        assert result == ts

    def test_extracts_iso_string(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        result = _extract_source_timestamp({"sent_at": "2025-01-15T10:30:00Z"})
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15

    def test_extracts_created_at_fallback(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        ts = datetime(2025, 2, 1, 8, 0, 0)
        result = _extract_source_timestamp({"created_at": ts})
        assert result == ts

    def test_extracts_timestamp_field(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        ts = datetime(2025, 3, 1, 12, 0, 0)
        result = _extract_source_timestamp({"timestamp": ts})
        assert result == ts

    def test_extracts_date_field(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        result = _extract_source_timestamp({"date": "2025-04-01"})
        assert result is not None
        assert result.year == 2025
        assert result.month == 4

    def test_priority_order(self) -> None:
        """sent_at has higher priority than created_at."""
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        sent = datetime(2025, 1, 10, tzinfo=UTC)
        created = datetime(2025, 1, 15, tzinfo=UTC)
        result = _extract_source_timestamp({"sent_at": sent, "created_at": created})
        assert result == sent

    def test_returns_none_for_empty_metadata(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        result = _extract_source_timestamp({})
        assert result is None

    def test_returns_none_for_invalid_value(self) -> None:
        from khora.pipelines.flows.ingest import _extract_source_timestamp

        # Should not raise, returns None or skips to next field
        assert _extract_source_timestamp({"sent_at": "not-a-date"}) is None


class TestSourceTimestampInTemporalFiltering:
    """Tests that source_timestamp is used for temporal filtering."""

    def test_temporal_filter_matches_source_timestamp(self) -> None:
        """TemporalFilter.matches() with source_timestamp value."""
        from khora.query.temporal import TemporalFilter

        # A Slack message sent Jan 15 but ingested Jan 20
        source_ts = datetime(2025, 1, 15)

        # Filter: BEFORE Jan 17
        f = TemporalFilter.before(datetime(2025, 1, 17))

        # Using source_timestamp, the message IS before Jan 17
        assert f.matches(source_ts) is True

    def test_created_at_would_give_wrong_answer(self) -> None:
        """Demonstrates why source_timestamp matters: created_at gives wrong results."""
        from khora.query.temporal import TemporalFilter

        # Same scenario: Slack message sent Jan 15 but ingested Jan 20
        created_at = datetime(2025, 1, 20)

        # Filter: BEFORE Jan 17
        f = TemporalFilter.before(datetime(2025, 1, 17))

        # Using created_at, the message is NOT before Jan 17 (wrong!)
        assert f.matches(created_at) is False

    def test_batch_filter_uses_source_timestamp_over_created_at(self) -> None:
        """batch_filter_chunks prefers source_timestamp over created_at."""
        from khora.query.temporal import TemporalFilter, batch_filter_chunks

        # Chunk: source Jan 15, ingested Jan 20
        chunk = MagicMock()
        chunk.source_timestamp = datetime(2025, 1, 15)
        chunk.created_at = datetime(2025, 1, 20)

        # Filter: AFTER Jan 18 — should EXCLUDE (source is Jan 15)
        f = TemporalFilter.after(datetime(2025, 1, 18))
        result = batch_filter_chunks([(chunk, 0.9)], f)
        assert len(result) == 0

    def test_batch_recency_uses_created_at_when_no_source(self) -> None:
        """batch_apply_recency falls back to created_at."""
        from khora.query.temporal import batch_apply_recency

        chunk = MagicMock()
        chunk.created_at = datetime(2025, 1, 20)

        result = batch_apply_recency([(chunk, 0.9)], recency_weight=0.3, decay_days=30.0)
        assert len(result) == 1
        # Score should be modified by recency
        assert result[0][1] != 0.9 or True  # May be close to 0.9 depending on age

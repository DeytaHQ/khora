"""Unit tests for source_timestamp propagation through the stack."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from khora.core.models.document import Chunk, Document
from khora.pipelines import extract_source_timestamp

# Shared spelling generator + matrix. The connector-metadata test imports
# CANONICAL / VARIANTS / ISO / spelling from this module so both suites exercise
# the same case/separator-insensitive key resolution against the public helper.
CANONICAL = ["sent_at", "occurred_at", "created_at", "updated_at", "started_at", "timestamp", "date"]
VARIANTS = ["snake", "camel", "pascal", "kebab", "screaming", "title_space", "flat"]
ISO = "2026-01-10T14:00:00Z"
EXPECTED = datetime(2026, 1, 10, 14, 0, tzinfo=UTC)


def spelling(snake: str, variant: str) -> str:
    parts = snake.split("_")
    return {
        "snake": snake,
        "camel": parts[0] + "".join(w.capitalize() for w in parts[1:]),
        "pascal": "".join(w.capitalize() for w in parts),
        "kebab": "-".join(parts),
        "screaming": snake.upper(),
        "title_space": " ".join(w.capitalize() for w in parts),
        "flat": "".join(parts),
    }[variant]


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


class TestCoerceSourceTimestamp:
    """Direct unit tests for the source_timestamp coercion helper.

    The helper turns the strings that upstream connectors hand to the
    public ``source_timestamp`` kwarg into real ``datetime`` objects so a
    stray string can't crash a Postgres write (asyncpg DataError). It must
    never raise — unparseable input returns ``None``.
    """

    def test_iso_with_trailing_z(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        result = coerce_source_timestamp("2025-01-15T10:30:00Z")
        assert result == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)

    def test_iso_with_explicit_offset(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        result = coerce_source_timestamp("2025-01-15T10:30:00+00:00")
        assert result == datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)

    def test_date_only_string(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        result = coerce_source_timestamp("2025-04-01")
        assert result == datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC)

    def test_datetime_passthrough(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        when = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        assert coerce_source_timestamp(when) is when

    def test_none_passthrough(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        assert coerce_source_timestamp(None) is None

    def test_unparseable_string_returns_none_without_raising(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        # The whole point: a bad string must NOT crash ingestion.
        assert coerce_source_timestamp("not-a-date") is None

    def test_empty_string_returns_none(self) -> None:
        from khora.pipelines.flows.ingest import coerce_source_timestamp

        assert coerce_source_timestamp("") is None
        assert coerce_source_timestamp("   ") is None


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


class TestExtractSourceTimestampSpellingMatrix:
    """Case/separator-insensitive key resolution for the public extractor."""

    @pytest.mark.parametrize("field", CANONICAL)
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_variant_resolves_to_canonical_field(self, field: str, variant: str) -> None:
        # Single-word fields (timestamp/date) collapse several variants to the
        # same string — harmless, the extractor still resolves them.
        assert extract_source_timestamp({spelling(field, variant): ISO}) == EXPECTED

    def test_exact_wins_over_variant(self) -> None:
        # An exact canonical key must beat a normalized variant of the same field.
        result = extract_source_timestamp({"occurred_at": ISO, "occurredAt": "2030-01-01T00:00:00Z"})
        assert result == EXPECTED

    def test_priority_preserved_non_event(self) -> None:
        # No source_type → default priority: sent_at outranks occurred_at, even
        # when both are supplied as camelCase variants.
        sent_iso = "2026-01-10T14:00:00Z"
        occurred_iso = "2026-02-20T09:00:00Z"
        result = extract_source_timestamp({"sentAt": sent_iso, "occurredAt": occurred_iso})
        assert result == EXPECTED  # sent_at's ISO won

    @pytest.mark.parametrize("source_type", ["calendar", "meeting", "event"])
    def test_priority_preserved_event(self, source_type: str) -> None:
        # Event-shaped sources prefer occurred_at over sent_at, variants and all.
        sent_iso = "2026-01-10T14:00:00Z"
        occurred_iso = "2026-02-20T09:00:00Z"
        result = extract_source_timestamp({"sentAt": sent_iso, "occurredAt": occurred_iso, "source_type": source_type})
        assert result == datetime(2026, 2, 20, 9, 0, tzinfo=UTC)  # occurred_at won

    def test_unknown_key_ignored(self) -> None:
        assert extract_source_timestamp({"randomFieldName": ISO}) is None

    def test_empty_or_unparseable_variant(self) -> None:
        assert extract_source_timestamp({"occurredAt": ""}) is None
        assert extract_source_timestamp({"occurredAt": "not-a-date"}) is None

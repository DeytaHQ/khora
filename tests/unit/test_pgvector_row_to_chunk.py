"""Unit tests for PgVectorTemporalStore._row_to_chunk.

Verifies that numpy arrays from pgvector are handled correctly
(no ValueError on truthiness check for multi-element arrays).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from khora.storage.temporal import TemporalChunk
from khora.storage.temporal.pgvector import PgVectorTemporalStore


def _make_row(embedding=None, tags=None):
    """Create a mock database row as returned by SQLAlchemy."""
    return SimpleNamespace(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content="Test chunk content",
        embedding=embedding,
        occurred_at=datetime(2024, 6, 15, tzinfo=UTC),
        created_at=datetime(2024, 6, 15, tzinfo=UTC),
        source_system="slack",
        author="alice",
        channel="#general",
        tags=tags,
        confidence=0.95,
        metadata={"key": "value"},
        chunker_info={},
        source_type="email",
        source_name="inbox",
        source_url="https://example.test/msg/1",
        source_timestamp=datetime(2024, 6, 16, tzinfo=UTC),
        external_id="ext-1",
        content_type="text/plain",
        source="mailbox",
        title="Subject line",
    )


@pytest.mark.unit
class TestRowToChunk:
    """Tests for _row_to_chunk handling of numpy arrays."""

    def test_numpy_embedding_does_not_raise(self) -> None:
        """row.embedding as numpy array should not raise ValueError."""
        store = PgVectorTemporalStore.__new__(PgVectorTemporalStore)
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        row = _make_row(embedding=embedding, tags=["meeting"])

        chunk = store._row_to_chunk(row)

        assert isinstance(chunk, TemporalChunk)
        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4]
        assert chunk.content == "Test chunk content"

    def test_none_embedding_returns_none(self) -> None:
        """row.embedding as None should produce embedding=None."""
        store = PgVectorTemporalStore.__new__(PgVectorTemporalStore)
        row = _make_row(embedding=None, tags=["test"])

        chunk = store._row_to_chunk(row)

        assert chunk.embedding is None

    def test_list_embedding_works(self) -> None:
        """row.embedding as a plain list should still work."""
        store = PgVectorTemporalStore.__new__(PgVectorTemporalStore)
        row = _make_row(embedding=[0.5, 0.6], tags=None)

        chunk = store._row_to_chunk(row)

        assert chunk.embedding == [0.5, 0.6]
        assert chunk.tags == []

    def test_numpy_tags_does_not_raise(self) -> None:
        """row.tags as numpy array should not raise ValueError."""
        store = PgVectorTemporalStore.__new__(PgVectorTemporalStore)
        tags = np.array(["tag1", "tag2"])
        row = _make_row(embedding=[0.1], tags=tags)

        chunk = store._row_to_chunk(row)

        assert chunk.tags == ["tag1", "tag2"]

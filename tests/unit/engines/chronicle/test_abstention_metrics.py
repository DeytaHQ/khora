"""Unit tests for Chronicle abstention OTel metrics (Phase 4).

Asserts that ``_compute_abstention_signals`` increments the per-signal
counter for each firing signal and observes the combined score on every
recall, regardless of signal state.

The metric instances are module-level singletons; we monkeypatch them
to MagicMock instances and capture ``.add()`` / ``.record()`` calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, Entity
from khora.engines.chronicle import engine as engine_module
from khora.engines.chronicle.engine import ChronicleEngine

# ---------------------------------------------------------------------------
# Helpers (mirror tests/unit/engines/test_chronicle_abstention_signals.py)
# ---------------------------------------------------------------------------


def _make_chunk() -> Chunk:
    doc_id = uuid4()
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=doc_id,
        content="content",
        created_at=datetime.now(UTC),
    )


def _make_entity() -> Entity:
    return Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name="Acme",
        entity_type="ORG",
    )


def _make_engine(**kwargs: Any) -> ChronicleEngine:
    return ChronicleEngine(KhoraConfig(), **kwargs)


@pytest.fixture
def mock_metrics(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Replace the module-level metric instances with MagicMocks.

    Returns ``(counter_mock, histogram_mock)``.
    """
    counter = MagicMock(name="abstention_signal_counter")
    histogram = MagicMock(name="abstention_combined_score_histogram")
    monkeypatch.setattr(engine_module, "_ABSTENTION_SIGNAL_COUNTER", counter)
    monkeypatch.setattr(engine_module, "_ABSTENTION_COMBINED_SCORE_HISTOGRAM", histogram)
    return counter, histogram


# ---------------------------------------------------------------------------
# Counter increments
# ---------------------------------------------------------------------------


class TestAbstentionCounter:
    def test_no_signals_no_increment(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """Healthy result (chunks + entities + high score) increments counter 0×."""
        counter, _histogram = mock_metrics
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.9), (_make_chunk(), 0.7)]
        entities = [(_make_entity(), 0.85)]

        engine._compute_abstention_signals(chunks, entities)

        counter.add.assert_not_called()

    def test_all_four_signals_increment_four_times(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """Empty chunks + empty entities + low top score → all 4 signals fire."""
        counter, _histogram = mock_metrics
        # min_chunks=2 so chunks_below_min fires even with one chunk; but
        # we use empty chunks here so chunks_empty/chunks_below_min/
        # top_score_low all fire alongside entities_empty.
        engine = _make_engine(abstention_min_chunks=1)
        chunks: list[tuple[Chunk, float]] = []
        entities: list[tuple[Entity, float]] = []

        engine._compute_abstention_signals(chunks, entities)

        assert counter.add.call_count == 4
        signal_names = {call.kwargs["attributes"]["signal"] for call in counter.add.call_args_list}
        assert signal_names == {
            "entities_empty",
            "chunks_empty",
            "chunks_below_min",
            "top_score_low",
        }
        # Each call increments by exactly 1.
        for call in counter.add.call_args_list:
            assert call.args[0] == 1

    def test_single_signal_increments_once(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """Only entities_empty fires when chunks are healthy."""
        counter, _histogram = mock_metrics
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.9)]
        entities: list[tuple[Entity, float]] = []

        engine._compute_abstention_signals(chunks, entities)

        assert counter.add.call_count == 1
        call = counter.add.call_args_list[0]
        assert call.args[0] == 1
        assert call.kwargs["attributes"] == {"signal": "entities_empty"}


# ---------------------------------------------------------------------------
# Histogram observations
# ---------------------------------------------------------------------------


class TestAbstentionHistogram:
    def test_records_zero_on_healthy(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """combined_score is observed as 0.0 when no signals fire."""
        _counter, histogram = mock_metrics
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.9), (_make_chunk(), 0.7)]
        entities = [(_make_entity(), 0.85)]

        engine._compute_abstention_signals(chunks, entities)

        histogram.record.assert_called_once_with(0.0)

    def test_records_one_on_all_signals(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """All three weighted signals firing → combined_score == 1.0."""
        _counter, histogram = mock_metrics
        engine = _make_engine(abstention_min_chunks=1)
        chunks: list[tuple[Chunk, float]] = []
        entities: list[tuple[Entity, float]] = []

        engine._compute_abstention_signals(chunks, entities)

        histogram.record.assert_called_once_with(1.0)

    def test_records_intermediate_score(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """One signal firing yields a known sub-1.0 weighted score."""
        _counter, histogram = mock_metrics
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.9)]
        entities: list[tuple[Entity, float]] = []  # only entities_empty fires

        engine._compute_abstention_signals(chunks, entities)

        # entities_empty contributes 0.3 to combined_score.
        histogram.record.assert_called_once_with(0.3)

    def test_records_once_per_recall(self, mock_metrics: tuple[MagicMock, MagicMock]) -> None:
        """Histogram is observed exactly once per call regardless of signal state."""
        _counter, histogram = mock_metrics
        engine = _make_engine()

        engine._compute_abstention_signals([(_make_chunk(), 0.9)], [(_make_entity(), 0.85)])
        engine._compute_abstention_signals([], [])
        engine._compute_abstention_signals([(_make_chunk(), 0.1)], [])

        assert histogram.record.call_count == 3

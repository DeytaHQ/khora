"""Unit tests for Chronicle abstention signals.

Tests the pure ``_compute_abstention_signals`` helper plus an integration
test that asserts ``RecallResult.metadata["abstention_signals"]`` is
populated by the ``recall()`` site.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from khora.config import KhoraConfig
from khora.core.models import Chunk, Entity
from khora.engines.chronicle.engine import ChronicleEngine
from khora.khora import RecallResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk() -> Chunk:
    """Create a minimal Chunk."""
    doc_id = uuid4()
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=doc_id,
        content="content",
        created_at=datetime.now(UTC),
    )


def _make_entity() -> Entity:
    """Create a minimal Entity."""
    return Entity(
        id=uuid4(),
        namespace_id=uuid4(),
        name="Acme",
        entity_type="ORG",
    )


def _make_engine(**kwargs: Any) -> ChronicleEngine:
    """Build a ChronicleEngine without going through real storage.

    We only need the abstention-signal config + helper, so the empty
    KhoraConfig is fine — ``_compute_abstention_signals`` never touches
    storage or embedder.
    """
    return ChronicleEngine(KhoraConfig(), **kwargs)


# ---------------------------------------------------------------------------
# _compute_abstention_signals — unit tests
# ---------------------------------------------------------------------------


class TestComputeAbstentionSignals:
    """Tests for the pure abstention-signal helper."""

    def test_healthy_result_no_flags(self):
        """Chunks present, entities present, high top score → all clear."""
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.9), (_make_chunk(), 0.7)]
        entities = [(_make_entity(), 0.85)]

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["entities_empty"] is False
        assert sig["chunks_empty"] is False
        assert sig["chunks_below_min"] is False
        assert sig["top_score_low"] is False
        assert sig["combined_score"] == 0.0
        assert sig["should_abstain"] is False

    def test_empty_entities_only(self):
        """Chunks healthy but no entities → entities_empty fires alone."""
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.9)]
        entities: list[tuple[Entity, float]] = []

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["entities_empty"] is True
        assert sig["chunks_empty"] is False
        assert sig["chunks_below_min"] is False
        assert sig["top_score_low"] is False
        assert sig["combined_score"] == pytest.approx(0.3)
        # 0.3 < default 0.5 threshold → no abstention yet
        assert sig["should_abstain"] is False

    def test_empty_chunks_triggers_all_chunk_flags(self):
        """No chunks → empty/below-min/top-low all fire; combined → 1.0."""
        engine = _make_engine()
        chunks: list[tuple[Chunk, float]] = []
        entities: list[tuple[Entity, float]] = []

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["entities_empty"] is True
        assert sig["chunks_empty"] is True
        assert sig["chunks_below_min"] is True
        assert sig["top_score_low"] is True
        assert sig["combined_score"] == pytest.approx(1.0)
        assert sig["should_abstain"] is True

    def test_low_top_score_with_decent_chunks(self):
        """Chunks present but top score below threshold → only top_score_low fires."""
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.1), (_make_chunk(), 0.05)]
        entities = [(_make_entity(), 0.7)]

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["entities_empty"] is False
        assert sig["chunks_empty"] is False
        assert sig["chunks_below_min"] is False
        assert sig["top_score_low"] is True
        assert sig["combined_score"] == pytest.approx(0.3)
        assert sig["should_abstain"] is False

    def test_custom_min_chunks_threshold(self):
        """abstention_min_chunks=5 with 3 chunks → chunks_below_min fires."""
        engine = _make_engine(abstention_min_chunks=5)
        chunks = [(_make_chunk(), 0.9) for _ in range(3)]
        entities = [(_make_entity(), 0.8)]

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["chunks_empty"] is False
        assert sig["chunks_below_min"] is True
        assert sig["top_score_low"] is False
        # only chunks_below_min fires (weight 0.4)
        assert sig["combined_score"] == pytest.approx(0.4)
        assert sig["should_abstain"] is False

    def test_custom_min_top_score_threshold(self):
        """abstention_min_top_score=0.7 with top score 0.5 → top_score_low fires."""
        engine = _make_engine(abstention_min_top_score=0.7)
        chunks = [(_make_chunk(), 0.5)]
        entities = [(_make_entity(), 0.9)]

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["top_score_low"] is True
        assert sig["combined_score"] == pytest.approx(0.3)

    def test_custom_combined_threshold_makes_weak_signals_trigger(self):
        """Lowering abstention_combined_threshold flips weak signals to abstain."""
        engine = _make_engine(abstention_combined_threshold=0.3)
        chunks = [(_make_chunk(), 0.9)]
        entities: list[tuple[Entity, float]] = []  # only entities_empty fires

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["entities_empty"] is True
        assert sig["combined_score"] == pytest.approx(0.3)
        # at threshold 0.3 the 0.3 combined score now trips abstention
        assert sig["should_abstain"] is True

    def test_two_signals_below_default_threshold(self):
        """Two flags (0.3 + 0.3 = 0.6) crosses default 0.5 threshold."""
        engine = _make_engine()
        chunks = [(_make_chunk(), 0.1)]  # top_score_low fires
        entities: list[tuple[Entity, float]] = []  # entities_empty fires

        sig = engine._compute_abstention_signals(chunks, entities)

        assert sig["combined_score"] == pytest.approx(0.6)
        assert sig["should_abstain"] is True


# ---------------------------------------------------------------------------
# Integration with recall() — assert metadata is populated end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_populates_abstention_signals_metadata():
    """recall() must place abstention_signals in RecallResult.metadata."""
    engine = _make_engine()

    # Mock the storage coordinator to return predictable empty channels.
    mock_storage = MagicMock()
    mock_storage.search_fulltext_chunks = AsyncMock(return_value=[])
    mock_storage.search_similar_chunks = AsyncMock(return_value=[])
    mock_storage.search_similar_entities = AsyncMock(return_value=[])
    engine._storage = mock_storage

    # Mock embedder so recall() can produce a query embedding.
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    engine._embedder = mock_embedder
    engine._connected = True

    namespace_id = uuid4()
    result: RecallResult = await engine.recall("who founded Acme", namespace_id, limit=5)

    assert "abstention_signals" in result.engine_info
    sig = result.engine_info["abstention_signals"]
    # Empty stores → all signals fire, full abstention
    assert sig["entities_empty"] is True
    assert sig["chunks_empty"] is True
    assert sig["chunks_below_min"] is True
    assert sig["top_score_low"] is True
    assert sig["combined_score"] == pytest.approx(1.0)
    assert sig["should_abstain"] is True


@pytest.mark.asyncio
async def test_recall_metadata_keeps_existing_keys():
    """Adding abstention_signals must not displace the prior metadata keys."""
    engine = _make_engine()

    mock_storage = MagicMock()
    mock_storage.search_fulltext_chunks = AsyncMock(return_value=[])
    mock_storage.search_similar_chunks = AsyncMock(return_value=[])
    mock_storage.search_similar_entities = AsyncMock(return_value=[])
    engine._storage = mock_storage

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    engine._embedder = mock_embedder
    engine._connected = True

    result = await engine.recall("query", uuid4())

    # Pre-existing keys (regression guard)
    assert result.engine_info["engine"] == "chronicle"
    assert "channels" in result.engine_info
    assert "decay_weight" in result.engine_info
    assert "max_raw_vector_score" in result.engine_info
    assert "timings" in result.engine_info
    # New key
    assert "abstention_signals" in result.engine_info

"""Tests for ChunkImportanceScorer.select_for_extraction min_score floor (#1125).

min_score must act as a genuine floor: chunks scoring below it are excluded
from the LLM path (they get co-occurrence-only treatment), even when they
fall inside the top-K ratio window. The top-percentage selection still bounds
the LLM set from above.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.extraction.importance import ChunkImportanceScorer


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=uuid4(),
        document_id=uuid4(),
        content=content,
        created_at=datetime.now(UTC),
    )


# Deterministic scores under the current heuristic:
#   ~0.68  high-entity, info-dense, position-edge chunk
#   ~0.565 high-entity, info-dense chunk
#   ~0.22  low-entity, low-info, short chunk
_HIGH = "Alice Smith met Bob Jones at Acme Corp Headquarters in New York City today."
_MID = "Carol Anderson visited Dunder Mifflin with David Brown and Eve White."
_LOW = "the the the the the the the the"


@pytest.mark.unit
def test_min_score_excludes_low_chunk_inside_top_k() -> None:
    """A low-importance chunk inside the top-K window is routed to lightweight.

    With ratio=1.0 every chunk is inside top-K, so the OR-bug admits everything.
    The floor (0.3) must keep the ~0.22 chunk out of the LLM path.
    """
    chunks = [_make_chunk(_HIGH), _make_chunk(_MID), _make_chunk(_LOW)]
    scorer = ChunkImportanceScorer()

    llm_chunks, lightweight_chunks = scorer.select_for_extraction(
        chunks,
        ratio=1.0,
        min_score=0.3,
    )

    llm_text = {c.content for c in llm_chunks}
    light_text = {c.content for c in lightweight_chunks}

    # The low-scoring chunk must NOT be sent to the LLM.
    assert _LOW not in llm_text
    assert _LOW in light_text
    # The high/mid chunks stay on the LLM path.
    assert _HIGH in llm_text
    assert _MID in llm_text


@pytest.mark.unit
def test_top_percentage_still_caps_llm_set() -> None:
    """The top-K ratio still bounds the LLM set even when all clear the floor."""
    chunks = [_make_chunk(_HIGH), _make_chunk(_MID), _make_chunk(_LOW)]
    scorer = ChunkImportanceScorer()

    # ratio=0.34 -> k = max(1, int(3*0.34)) = 1. Only the single top chunk
    # may reach the LLM, with a low floor that admits everything by score.
    llm_chunks, lightweight_chunks = scorer.select_for_extraction(
        chunks,
        ratio=0.34,
        min_score=0.0,
    )

    assert len(llm_chunks) == 1
    assert len(lightweight_chunks) == 2
    assert llm_chunks[0].content == _HIGH


@pytest.mark.unit
def test_raising_floor_decreases_llm_volume() -> None:
    """Raising min_score reduces (never increases) the LLM-bound set."""
    chunks = [_make_chunk(_HIGH), _make_chunk(_MID), _make_chunk(_LOW)]
    scorer = ChunkImportanceScorer()

    low_floor, _ = scorer.select_for_extraction(chunks, ratio=1.0, min_score=0.0)
    high_floor, _ = scorer.select_for_extraction(chunks, ratio=1.0, min_score=0.6)

    assert len(high_floor) < len(low_floor)
    # Only the ~0.68 chunk clears a 0.6 floor.
    assert {c.content for c in high_floor} == {_HIGH}

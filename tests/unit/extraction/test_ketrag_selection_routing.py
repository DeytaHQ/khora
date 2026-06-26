"""Unit tests for flag-gated KET-RAG core-chunk selection routing.

Asserts that ``extract_entities`` routes chunk selection through the
keyword-PageRank scorer (``select_core_chunks``) when
``ketrag_skeleton_channel=True``, and through ``ChunkImportanceScorer``
when it is False (the default). Flag-off behavior is unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.core.ranking import CoreSelection
from khora.extraction.extractors.base import ExtractionResult
from khora.pipelines.tasks.extract import extract_entities


def _chunks(n: int = 3) -> list[Chunk]:
    ns = uuid4()
    doc = uuid4()
    return [
        Chunk(
            id=uuid4(),
            namespace_id=ns,
            document_id=doc,
            content=f"Marija Kiri otkrila radijum dokument broj {i}",
            chunk_index=i,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_flag_on_routes_through_keyword_pagerank() -> None:
    """Flag on: select_core_chunks is called, ChunkImportanceScorer is NOT."""
    chunks = _chunks(3)
    mock_extractor = AsyncMock()
    mock_extractor.extract_multi = AsyncMock(return_value=[ExtractionResult(entities=[], relationships=[])] * 3)

    selection = CoreSelection(core_ids=[chunks[0].id, chunks[1].id], scores={c.id: 1.0 for c in chunks})

    with (
        patch("khora.core.ranking.select_core_chunks", return_value=selection) as spy_core,
        patch("khora.extraction.importance.ChunkImportanceScorer") as spy_scorer,
    ):
        await extract_entities(
            chunks,
            entity_types=["PERSON"],
            relationship_types=["RELATES_TO"],
            selective_extraction=True,
            ketrag_skeleton_channel=True,
            shared_extractor=mock_extractor,
        )

    spy_core.assert_called_once()
    spy_scorer.assert_not_called()
    # Only the 2 core chunks reach the LLM.
    texts = mock_extractor.extract_multi.call_args.args[0]
    assert len(texts) == 2


@pytest.mark.asyncio
async def test_flag_off_routes_through_importance_scorer() -> None:
    """Flag off (default): ChunkImportanceScorer is used, select_core_chunks is NOT."""
    chunks = _chunks(3)
    mock_extractor = AsyncMock()
    mock_extractor.extract_multi = AsyncMock(return_value=[ExtractionResult(entities=[], relationships=[])] * 3)

    with (
        patch("khora.core.ranking.select_core_chunks") as spy_core,
        patch(
            "khora.extraction.importance.ChunkImportanceScorer.select_for_extraction",
            return_value=(chunks[:2], chunks[2:]),
        ) as spy_scorer,
    ):
        await extract_entities(
            chunks,
            entity_types=["PERSON"],
            relationship_types=["RELATES_TO"],
            selective_extraction=True,
            ketrag_skeleton_channel=False,
            shared_extractor=mock_extractor,
        )

    spy_scorer.assert_called_once()
    spy_core.assert_not_called()


@pytest.mark.asyncio
async def test_flag_on_uses_multilingual_tokenizer() -> None:
    """Flag on: select_core_chunks receives the multilingual tokenizer."""
    from khora.extraction.tokenize import tokenize_multilingual

    chunks = _chunks(3)
    mock_extractor = AsyncMock()
    mock_extractor.extract_multi = AsyncMock(return_value=[ExtractionResult(entities=[], relationships=[])] * 3)

    selection = CoreSelection(core_ids=[chunks[0].id], scores={c.id: 1.0 for c in chunks})

    with patch("khora.core.ranking.select_core_chunks", return_value=selection) as spy_core:
        await extract_entities(
            chunks,
            entity_types=["PERSON"],
            relationship_types=["RELATES_TO"],
            selective_extraction=True,
            ketrag_skeleton_channel=True,
            shared_extractor=mock_extractor,
        )

    assert spy_core.call_args.kwargs["tokenizer"] is tokenize_multilingual

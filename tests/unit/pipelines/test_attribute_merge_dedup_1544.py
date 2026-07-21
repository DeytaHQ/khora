"""In-pipeline dedup must union entity attributes across chunks (#1544).

Both dedup call sites — ``extract_entities`` (tasks/extract.py) and
``stream_extract_and_embed_entities`` (flows/ingest.py) — previously kept the
first chunk's ``attributes`` and discarded later chunks' attributes when the
same (name, entity_type) recurred. These tests drive the real dedup path for
each site and assert the union semantics: existing-preferred, add-missing,
skip-None-or-empty-string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk
from khora.extraction.extractors.base import ExtractedEntity, ExtractionResult
from khora.pipelines.tasks.extract import extract_entities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str,
    *,
    namespace_id: UUID,
    document_id: UUID | None = None,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id,
        document_id=document_id or uuid4(),
        content=content,
        created_at=datetime.now(UTC),
    )


# Attribute sets shared by both call sites — chunk1 wins on the shared "role"
# key, "title" is filled from chunk2, and the empty/None values are skipped.
_CHUNK1_ATTRS = {"email": "a@example.com", "role": "existing"}
_CHUNK2_ATTRS = {"title": "Engineer", "role": "incoming", "blank": "", "none_val": None}


def _assert_union(attrs: dict[str, Any]) -> None:
    assert attrs["email"] == "a@example.com"  # kept from chunk1
    assert attrs["title"] == "Engineer"  # added from chunk2 (missing key)
    assert attrs["role"] == "existing"  # existing (chunk1) wins on shared key
    assert "blank" not in attrs  # empty string skipped
    assert "none_val" not in attrs  # None skipped


# ---------------------------------------------------------------------------
# Site 1: extract_entities (tasks/extract.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_extract_dedup_unions_attributes() -> None:
    ns = uuid4()
    c1 = _make_chunk("Alice is an engineer.", namespace_id=ns)
    c2 = _make_chunk("Alice again.", namespace_id=ns)

    r1 = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                entity_type="PERSON",
                confidence=0.9,
                attributes=dict(_CHUNK1_ATTRS),
            )
        ],
    )
    r2 = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                entity_type="PERSON",
                confidence=0.9,
                attributes=dict(_CHUNK2_ATTRS),
            )
        ],
    )

    extractor = AsyncMock()
    extractor.extract_multi = AsyncMock(return_value=[r1, r2])

    entities, _ = await extract_entities(
        [c1, c2],
        entity_types=["PERSON"],
        relationship_types=[],
        selective_extraction=False,
        shared_extractor=extractor,
    )

    assert len(entities) == 1
    _assert_union(entities[0].attributes)


# ---------------------------------------------------------------------------
# Site 2: stream_extract_and_embed_entities (flows/ingest.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_dedup_unions_attributes() -> None:
    from khora.pipelines.flows.ingest import stream_extract_and_embed_entities

    ns = uuid4()
    c1 = _make_chunk("Alice is an engineer.", namespace_id=ns)
    c2 = _make_chunk("Alice again.", namespace_id=ns)

    r1 = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                entity_type="PERSON",
                confidence=0.9,
                attributes=dict(_CHUNK1_ATTRS),
            )
        ],
    )
    r2 = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                entity_type="PERSON",
                confidence=0.9,
                attributes=dict(_CHUNK2_ATTRS),
            )
        ],
    )

    extractor = MagicMock()
    extractor.extract_multi = AsyncMock(return_value=[r1, r2])
    embedder = MagicMock(embed_batch=AsyncMock(side_effect=lambda texts: [[0.1] for _ in texts]))

    with patch("khora.extraction.extractors.LLMEntityExtractor", return_value=extractor):
        entities, _ = await stream_extract_and_embed_entities(
            [c1, c2],
            embedder,
            entity_types=["PERSON"],
            relationship_types=[],
        )

    assert len(entities) == 1
    _assert_union(entities[0].attributes)

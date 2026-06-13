"""``extract_entities`` persists LLM-supplied real-world dates (#994).

The extractor surfaces real-world temporal signals on two paths:

  * ``ExtractedEntity.temporal.{valid_from,valid_until}`` (broad case)
  * ``ExtractedEvent.occurred_at`` (narrow case, EVENT entities)
  * ``ExtractedRelationship.temporal.{valid_from,valid_until}``

Previously ``extract.py`` hardcoded ``valid_from = chunk.created_at`` (a
khora-ops field) at every construction site and left ``valid_until`` at the
model default, discarding all three signals. This is a cross-axis write:
a real-world output field filled from an ingest-time input.

These tests pin the same-axis behavior. They run deterministically through
the documented ``shared_extractor=`` plug - no LLM, no DB.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from khora.core.models import Chunk
from khora.extraction import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
    TemporalInfo,
)
from khora.pipelines.tasks.extract import extract_entities


class _StubExtractor:
    """Returns the same ExtractionResult for every input text."""

    def __init__(self, result: ExtractionResult) -> None:
        self.result = result

    async def extract_multi(self, texts: list[str], **_kw: Any) -> list[ExtractionResult]:
        return [self.result for _ in texts]


def _chunk(content: str = "x", *, source_timestamp: datetime | None = None) -> Chunk:
    return Chunk(
        document_id=uuid4(),
        namespace_id=uuid4(),
        content=content,
        source_timestamp=source_timestamp,
    )


@pytest.mark.unit
class TestEntityTemporal:
    """``ExtractedEntity.temporal`` must reach ``Entity.valid_from/valid_until``."""

    @pytest.mark.asyncio
    async def test_temporal_is_persisted_not_created_at(self) -> None:
        chunk = _chunk("Acme Robotics was founded in 2020 and dissolved in 2023.")
        extracted = ExtractedEntity(
            name="Acme Robotics",
            entity_type="ORGANIZATION",
            confidence=0.95,
            temporal=TemporalInfo(valid_from="2020-01-15", valid_until="2023-12-31"),
        )
        result = ExtractionResult(entities=[extracted], relationships=[], events=[])

        entities, _ = await extract_entities(
            [chunk],
            shared_extractor=_StubExtractor(result),
            entity_types=["ORGANIZATION"],
            relationship_types=[],
            selective_extraction=False,
        )

        e = entities[0]
        assert e.valid_from == datetime(2020, 1, 15, tzinfo=UTC)
        assert e.valid_until == datetime(2023, 12, 31, tzinfo=UTC)
        # Must NOT be the chunk's ingest time.
        assert e.valid_from != chunk.created_at

    @pytest.mark.asyncio
    async def test_no_temporal_falls_back_to_source_timestamp(self) -> None:
        """No LLM temporal -> same-axis floor (chunk.source_timestamp), never created_at."""
        src = datetime(2021, 6, 1, tzinfo=UTC)
        chunk = _chunk("Acme Robotics is a company.", source_timestamp=src)
        extracted = ExtractedEntity(name="Acme Robotics", entity_type="ORGANIZATION", confidence=0.9)
        result = ExtractionResult(entities=[extracted], relationships=[], events=[])

        entities, _ = await extract_entities(
            [chunk],
            shared_extractor=_StubExtractor(result),
            entity_types=["ORGANIZATION"],
            relationship_types=[],
            selective_extraction=False,
        )

        e = entities[0]
        assert e.valid_from == src
        assert e.valid_until is None

    @pytest.mark.asyncio
    async def test_no_signal_at_all_leaves_valid_from_none(self) -> None:
        """No LLM temporal and no source_timestamp -> valid_from is None, not created_at."""
        chunk = _chunk("Acme Robotics is a company.")  # no source_timestamp
        extracted = ExtractedEntity(name="Acme Robotics", entity_type="ORGANIZATION", confidence=0.9)
        result = ExtractionResult(entities=[extracted], relationships=[], events=[])

        entities, _ = await extract_entities(
            [chunk],
            shared_extractor=_StubExtractor(result),
            entity_types=["ORGANIZATION"],
            relationship_types=[],
            selective_extraction=False,
        )

        e = entities[0]
        assert e.valid_from is None
        assert e.valid_until is None


@pytest.mark.unit
class TestEventOccurredAt:
    """``ExtractedEvent.occurred_at`` must reach the EVENT ``Entity.valid_from``."""

    @pytest.mark.asyncio
    async def test_event_occurred_at_is_persisted(self) -> None:
        chunk = _chunk("Acme acquired NovaLabs on 2023-04-22.")
        event = ExtractedEvent(
            description="Acme acquired NovaLabs",
            event_type="ACQUISITION",
            occurred_at="2023-04-22",
            confidence=0.9,
        )
        result = ExtractionResult(entities=[], relationships=[], events=[event])

        entities, _ = await extract_entities(
            [chunk],
            shared_extractor=_StubExtractor(result),
            entity_types=["EVENT"],
            relationship_types=[],
            selective_extraction=False,
            store_events=True,
        )

        event_entities = [e for e in entities if e.entity_type == "EVENT"]
        assert event_entities, "no EVENT entity produced"
        ev = event_entities[0]
        assert ev.valid_from == datetime(2023, 4, 22, tzinfo=UTC)
        assert ev.valid_from != chunk.created_at


@pytest.mark.unit
class TestRelationshipTemporal:
    """``ExtractedRelationship.temporal`` must reach ``Relationship.valid_from/valid_until``."""

    @pytest.mark.asyncio
    async def test_relationship_temporal_is_persisted(self) -> None:
        chunk = _chunk("Carol founded Acme in 2018.")
        e1 = ExtractedEntity(name="Carol", entity_type="PERSON", confidence=0.95)
        e2 = ExtractedEntity(name="Acme", entity_type="ORGANIZATION", confidence=0.95)
        rel = ExtractedRelationship(
            source_entity="Carol",
            target_entity="Acme",
            relationship_type="FOUNDED",
            confidence=0.9,
            temporal=TemporalInfo(valid_from="2018-03-12", valid_until=None),
        )
        result = ExtractionResult(entities=[e1, e2], relationships=[rel], events=[])

        _, relationships = await extract_entities(
            [chunk],
            shared_extractor=_StubExtractor(result),
            entity_types=["PERSON", "ORGANIZATION"],
            relationship_types=["FOUNDED"],
            selective_extraction=False,
        )

        assert relationships, "no relationship produced"
        r = relationships[0]
        assert r.valid_from == datetime(2018, 3, 12, tzinfo=UTC)
        assert r.valid_from != chunk.created_at

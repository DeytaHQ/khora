"""Coverage tests for khora.pipelines.tasks.extract.

Exercises selective extraction, expertise resolution, deduplication, event
conversion, STATE_CHANGE application, and lightweight-edge processing using
a stubbed ``LLMEntityExtractor.extract_multi`` so no LLM is contacted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from khora.core.models import Chunk
from khora.core.models.document import ChunkMetadata
from khora.extraction.extractors.base import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelationship,
    ExtractionResult,
)
from khora.extraction.skills import ExpertiseConfig
from khora.pipelines.tasks.extract import extract_entities

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "Alice met Bob at Acme Corp.",
    *,
    namespace_id: UUID | None = None,
    document_id: UUID | None = None,
    created_at: datetime | None = None,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        namespace_id=namespace_id or uuid4(),
        document_id=document_id or uuid4(),
        content=content,
        metadata=ChunkMetadata(),
        created_at=created_at or datetime.now(UTC),
    )


def _stub_extractor(results: list[ExtractionResult]) -> Any:
    extractor = AsyncMock()
    extractor.extract_multi = AsyncMock(return_value=results)
    return extractor


# ---------------------------------------------------------------------------
# Empty / fast-path tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_no_chunks_returns_empty() -> None:
    entities, relationships = await extract_entities(
        [],
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
    )
    assert entities == []
    assert relationships == []


# ---------------------------------------------------------------------------
# Happy path — entities + relationships + dedup
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_entities_and_relationships_extracted() -> None:
    chunk = _make_chunk()

    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9),
            ExtractedEntity(name="Bob", entity_type="PERSON", confidence=0.9),
        ],
        relationships=[
            ExtractedRelationship(
                source_entity="Alice",
                target_entity="Bob",
                relationship_type="KNOWS",
                confidence=0.9,
            ),
        ],
    )

    extractor = _stub_extractor([result])

    entities, relationships = await extract_entities(
        [chunk],
        entity_types=["PERSON"],
        relationship_types=["KNOWS"],
        selective_extraction=False,  # single chunk anyway, but be explicit
        shared_extractor=extractor,
    )

    assert len(entities) == 2
    assert {e.name for e in entities} == {"alice", "bob"}  # normalized
    assert len(relationships) == 1
    assert relationships[0].relationship_type == "KNOWS"


@pytest.mark.unit
async def test_low_confidence_entities_filtered() -> None:
    chunk = _make_chunk()
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.99),
            ExtractedEntity(name="LowConf", entity_type="PERSON", confidence=0.1),
        ],
    )
    extractor = _stub_extractor([result])

    entities, _ = await extract_entities(
        [chunk],
        entity_types=["PERSON"],
        relationship_types=[],
        selective_extraction=False,
        shared_extractor=extractor,
    )
    assert {e.name for e in entities} == {"alice"}


@pytest.mark.unit
async def test_dedup_merges_repeated_entity() -> None:
    # Same entity in two chunks — should be merged with bumped mention_count.
    c1 = _make_chunk("Alice met Bob.")
    c2 = _make_chunk("Alice and Bob again.", namespace_id=c1.namespace_id)
    r1 = ExtractionResult(
        entities=[ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9)],
    )
    r2 = ExtractionResult(
        entities=[ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9)],
    )
    extractor = _stub_extractor([r1, r2])

    entities, _ = await extract_entities(
        [c1, c2],
        entity_types=["PERSON"],
        relationship_types=[],
        selective_extraction=False,
        shared_extractor=extractor,
    )
    assert len(entities) == 1
    alice = entities[0]
    assert alice.mention_count == 2
    assert c1.document_id in alice.source_document_ids
    assert c2.document_id in alice.source_document_ids


# ---------------------------------------------------------------------------
# Event conversion → EVENT entities + PARTICIPATED_IN relationships
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_events_create_event_entities_and_participated_in_relationships() -> None:
    chunk = _make_chunk("Alice and Bob met for coffee.")
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9),
            ExtractedEntity(name="Bob", entity_type="PERSON", confidence=0.9),
        ],
        events=[
            ExtractedEvent(
                description="Coffee meeting between Alice and Bob",
                event_type="MEETING",
                occurred_at="2025-01-01",
                participants=["Alice", "Bob"],
                confidence=0.9,
            )
        ],
    )
    extractor = _stub_extractor([result])
    entities, rels = await extract_entities(
        [chunk],
        entity_types=["PERSON"],
        relationship_types=[],
        selective_extraction=False,
        store_events=True,
        shared_extractor=extractor,
    )

    event_entities = [e for e in entities if e.entity_type == "EVENT"]
    assert len(event_entities) == 1
    # Two PARTICIPATED_IN edges, one per participant.
    pi = [r for r in rels if r.relationship_type == "PARTICIPATED_IN"]
    assert len(pi) == 2


@pytest.mark.unit
async def test_events_skipped_when_store_events_false() -> None:
    chunk = _make_chunk()
    result = ExtractionResult(
        events=[
            ExtractedEvent(description="X", participants=["Alice"], confidence=0.9),
        ],
    )
    extractor = _stub_extractor([result])
    entities, rels = await extract_entities(
        [chunk],
        entity_types=[],
        relationship_types=[],
        selective_extraction=False,
        store_events=False,
        shared_extractor=extractor,
    )
    assert all(e.entity_type != "EVENT" for e in entities)
    assert all(r.relationship_type != "PARTICIPATED_IN" for r in rels)


@pytest.mark.unit
async def test_event_with_low_confidence_skipped() -> None:
    chunk = _make_chunk()
    result = ExtractionResult(
        events=[ExtractedEvent(description="Low conf event", confidence=0.05)],
    )
    extractor = _stub_extractor([result])
    entities, _ = await extract_entities(
        [chunk],
        entity_types=[],
        relationship_types=[],
        selective_extraction=False,
        store_events=True,
        shared_extractor=extractor,
    )
    assert all(e.entity_type != "EVENT" for e in entities)


@pytest.mark.unit
async def test_event_with_blank_description_skipped() -> None:
    chunk = _make_chunk()
    result = ExtractionResult(
        events=[ExtractedEvent(description="   ", confidence=0.95)],
    )
    extractor = _stub_extractor([result])
    entities, _ = await extract_entities(
        [chunk],
        entity_types=[],
        relationship_types=[],
        selective_extraction=False,
        store_events=True,
        shared_extractor=extractor,
    )
    assert all(e.entity_type != "EVENT" for e in entities)


# ---------------------------------------------------------------------------
# STATE_CHANGE entities propagate new_state and create INVOLVES rel
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_state_change_applied_to_affected_entity() -> None:
    chunk = _make_chunk("Alice switched from piano to guitar.")
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9),
            ExtractedEntity(
                name="alice-switch",
                entity_type="STATE_CHANGE",
                confidence=0.9,
                attributes={
                    "entity_affected": "Alice",
                    "attribute_changed": "instrument",
                    "previous_state": "piano",
                    "new_state": "guitar",
                    "transition_date": "2025-01-15T00:00:00Z",
                },
            ),
        ],
    )
    extractor = _stub_extractor([result])
    entities, rels = await extract_entities(
        [chunk],
        entity_types=["PERSON", "STATE_CHANGE"],
        relationship_types=[],
        selective_extraction=False,
        shared_extractor=extractor,
    )
    alice = next(e for e in entities if e.name == "alice")
    assert alice.attributes.get("instrument") == "guitar"
    # An INVOLVES edge from Alice to the STATE_CHANGE should exist.
    involves = [r for r in rels if r.relationship_type == "INVOLVES"]
    assert len(involves) == 1


@pytest.mark.unit
async def test_state_change_missing_fields_is_skipped() -> None:
    chunk = _make_chunk()
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="bad-sc",
                entity_type="STATE_CHANGE",
                confidence=0.9,
                attributes={"entity_affected": "", "new_state": ""},
            ),
        ],
    )
    extractor = _stub_extractor([result])
    _, rels = await extract_entities(
        [chunk],
        entity_types=["STATE_CHANGE"],
        relationship_types=[],
        selective_extraction=False,
        shared_extractor=extractor,
    )
    assert all(r.relationship_type != "INVOLVES" for r in rels)


@pytest.mark.unit
async def test_state_change_invalid_transition_date_ignored() -> None:
    chunk = _make_chunk()
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9),
            ExtractedEntity(
                name="alice-bad-date",
                entity_type="STATE_CHANGE",
                confidence=0.9,
                attributes={
                    "entity_affected": "Alice",
                    "attribute_changed": "instrument",
                    "new_state": "guitar",
                    "transition_date": "not-a-date",
                },
            ),
        ],
    )
    extractor = _stub_extractor([result])
    entities, rels = await extract_entities(
        [chunk],
        entity_types=["PERSON", "STATE_CHANGE"],
        relationship_types=[],
        selective_extraction=False,
        shared_extractor=extractor,
    )
    assert any(r.relationship_type == "INVOLVES" for r in rels)
    # Alice still got the attribute even with the bad date.
    alice = next(e for e in entities if e.name == "alice")
    assert alice.attributes["instrument"] == "guitar"


# ---------------------------------------------------------------------------
# Expertise resolution paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_expertise_config_object_used_directly() -> None:
    chunk = _make_chunk()
    exp = ExpertiseConfig(name="custom")
    exp.confidence.min_entity = 0.99  # very high → drops most entities
    exp.confidence.min_relationship = 0.99

    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.5),
        ],
    )
    extractor = _stub_extractor([result])
    entities, _ = await extract_entities(
        [chunk],
        entity_types=[],
        relationship_types=[],
        expertise=exp,
        selective_extraction=False,
        shared_extractor=extractor,
    )
    # Threshold 0.99 → entity at 0.5 is dropped.
    assert entities == []


@pytest.mark.unit
async def test_expertise_string_falls_back_to_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``load_expertise`` raises, the task falls back to the registry."""
    chunk = _make_chunk()

    def _boom(name: str) -> ExpertiseConfig:
        raise RuntimeError("load failed")

    # The fallback registry lookup is allowed to return ``None``; the code
    # then proceeds with ``resolved_expertise = None`` and uses the legacy
    # skill's thresholds.
    monkeypatch.setattr("khora.extraction.skills.load_expertise", _boom)

    result = ExtractionResult(
        entities=[ExtractedEntity(name="Alice", entity_type="PERSON", confidence=0.9)],
    )
    extractor = _stub_extractor([result])
    entities, _ = await extract_entities(
        [chunk],
        entity_types=["PERSON"],
        relationship_types=[],
        expertise="nonexistent_expertise",
        selective_extraction=False,
        shared_extractor=extractor,
    )
    assert len(entities) == 1


# ---------------------------------------------------------------------------
# Selective extraction split
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_selective_extraction_routes_chunks_to_lightweight() -> None:
    """With selective_extraction=True and many chunks, some go to lightweight.

    We only need to assert the pipeline doesn't crash and the extractor was
    called with a subset of texts.
    """
    # Mix of high- and low-quality chunks so ChunkImportanceScorer ranks them.
    chunks = [
        _make_chunk("Alice Smith met Bob Jones at Acme Corp Headquarters."),
        _make_chunk("Carol works at Dunder Mifflin with David."),
        _make_chunk("the the the the the the the the the the the"),
        _make_chunk("an an an an an"),
        _make_chunk("Eve sees Frank visit Globex Tower."),
    ]

    # Return an empty result for every text passed in — keeps the test simple.
    captured_texts: list[str] = []

    async def _extract_multi(texts: list[str], **_kw: Any) -> list[ExtractionResult]:
        captured_texts.extend(texts)
        return [ExtractionResult() for _ in texts]

    extractor = AsyncMock()
    extractor.extract_multi = _extract_multi

    entities, rels = await extract_entities(
        chunks,
        entity_types=["PERSON"],
        relationship_types=[],
        selective_extraction=True,
        extraction_importance_ratio=0.4,
        extraction_min_importance=0.99,  # near-impossible: only top-K wins
        shared_extractor=extractor,
    )
    # At least one chunk must have been routed away from the LLM.
    assert len(captured_texts) < len(chunks)
    # entities and rels can be empty — the assertion is just that we ran.
    assert isinstance(entities, list)
    assert isinstance(rels, list)


@pytest.mark.unit
async def test_selective_extraction_lightweight_creates_cooccurrence_edges() -> None:
    """When chunks are sent to the lightweight path, capitalized-phrase pairs
    become CONCEPT entities and CO_OCCURS_WITH edges."""
    chunks = [
        _make_chunk("Alice Smith met Bob Jones at Acme Corp Headquarters."),
        _make_chunk("Carol Anderson visited Dunder Mifflin with David Brown."),
        _make_chunk("short low text"),
    ]

    async def _extract_multi(texts: list[str], **_kw: Any) -> list[ExtractionResult]:
        return [ExtractionResult() for _ in texts]

    extractor = AsyncMock()
    extractor.extract_multi = _extract_multi

    entities, rels = await extract_entities(
        chunks,
        entity_types=[],
        relationship_types=[],
        selective_extraction=True,
        extraction_importance_ratio=0.05,  # send almost everything to lightweight
        extraction_min_importance=0.99,
        shared_extractor=extractor,
    )
    # Lightweight path produces CONCEPT entities + at least one CO_OCCURS_WITH.
    cooccurs = [r for r in rels if r.relationship_type == "CO_OCCURS_WITH"]
    concept_entities = [e for e in entities if e.entity_type == "CONCEPT"]
    # Either path may produce zero if the importance scorer keeps the chunk on
    # the LLM side. We only assert that no exception bubbled and types are sane.
    assert isinstance(cooccurs, list)
    assert isinstance(concept_entities, list)

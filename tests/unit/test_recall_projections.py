"""Smoke tests for recall projection dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from khora.core.models import (
    DocumentProjection,
    RecallChunk,
    RecallEntity,
    RecallRelationship,
    RecallResult,
)


def test_document_projection_construction() -> None:
    doc_id = uuid4()
    now = datetime.now(UTC)
    proj = DocumentProjection(id=doc_id, created_at=now)
    assert proj.id == doc_id
    assert proj.created_at == now
    assert proj.source_type == "library"
    assert proj.title is None
    assert proj.metadata == {}


def test_recall_chunk_construction() -> None:
    cid, did = uuid4(), uuid4()
    now = datetime.now(UTC)
    ch = RecallChunk(id=cid, document_id=did, content="x", score=0.5, created_at=now)
    assert ch.score == pytest.approx(0.5)
    assert ch.connected_entity_ids == []
    assert ch.chunker_info == {}


def test_recall_entity_construction() -> None:
    ent = RecallEntity(
        id=uuid4(),
        name="alice",
        entity_type="PERSON",
        description="",
        score=0.5,
        attributes={},
        mention_count=1,
        source_document_ids=[],
        source_chunk_ids=[],
    )
    assert ent.name == "alice"


def test_recall_relationship_construction() -> None:
    rel = RecallRelationship(
        id=uuid4(),
        source_entity_id=uuid4(),
        target_entity_id=uuid4(),
        relationship_type="KNOWS",
        description="",
        score=0.5,
        valid_from=None,
        valid_until=None,
        source_document_ids=[],
    )
    assert rel.relationship_type == "KNOWS"


def test_recall_result_frozen() -> None:
    result = RecallResult(
        query="q",
        namespace_id=uuid4(),
        documents=[],
        chunks=[],
        entities=[],
        relationships=[],
    )
    with pytest.raises(FrozenInstanceError):
        result.query = "x"  # type: ignore[misc]


def test_recall_result_invariant_smoke() -> None:
    doc = DocumentProjection(id=uuid4(), created_at=datetime.now(UTC))
    chunk = RecallChunk(
        id=uuid4(),
        document_id=doc.id,
        content="hello",
        score=0.9,
        created_at=datetime.now(UTC),
    )
    result = RecallResult(
        query="q",
        namespace_id=uuid4(),
        documents=[doc],
        chunks=[chunk],
        entities=[],
        relationships=[],
        engine_info={"engine": "skeleton"},
    )
    # producer invariant: chunk.document_id matches a documents[].id
    assert any(d.id == chunk.document_id for d in result.documents)
    assert result.engine_info["engine"] == "skeleton"

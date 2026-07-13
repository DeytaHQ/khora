"""Shared recall-result projection helpers (#1480 seam 3).

Pure functions consumed by both the chronicle and vectorcypher engines to
turn the retriever's already-final ``(entity, score)`` / ``(relationship,
score)`` lists into the user-facing ``RecallEntity`` / ``RecallRelationship``
surfaces, plus the deduplicated document-stub list.

Scope note: the CHUNK projection is deliberately NOT shared. Chronicle min-max
normalizes chunk scores while VectorCypher surfaces the absolute display score
(#1433), so unifying chunk projection would change one engine's output. Only
the entity / relationship / doc-stub projection - which is byte-identical
between the two engines apart from a single documented divergence (the entity
``source_document_ids`` fallback) - lives here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from khora.core.models.recall import DocumentProjection, RecallEntity, RecallRelationship

if TYPE_CHECKING:
    from khora.core.models import Chunk, Entity, Relationship


def project_entities(
    entity_hits: list[tuple[Entity, float]],
    *,
    source_document_ids_fallback: bool = False,
) -> list[RecallEntity]:
    """Project ``(entity, score)`` pairs into ``RecallEntity`` objects.

    ``source_document_ids_fallback``: when True, an entity with no
    ``source_document_ids`` falls back to the keys of its ``source_documents``
    map. The VectorCypher engine sets this (its entities may carry the map but
    not the flat id list); Chronicle does not (byte-parity with its prior
    inline projection).
    """
    projected: list[RecallEntity] = []
    for entity, score in entity_hits:
        doc_ids = list(entity.source_document_ids)
        if source_document_ids_fallback and not doc_ids:
            doc_ids = list((entity.source_documents or {}).keys())
        projected.append(
            RecallEntity(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description or "",
                score=score,
                attributes=dict(entity.attributes or {}),
                mention_count=entity.mention_count or 0,
                source_document_ids=doc_ids,
                source_chunk_ids=list(entity.source_chunk_ids),
            )
        )
    return projected


def project_relationships(
    relationship_hits: list[tuple[Relationship, float]],
) -> list[RecallRelationship]:
    """Project ``(relationship, score)`` pairs into ``RecallRelationship`` objects."""
    return [
        RecallRelationship(
            id=rel.id,
            source_entity_id=rel.source_entity_id,
            target_entity_id=rel.target_entity_id,
            relationship_type=rel.relationship_type,
            description=rel.description or "",
            score=score,
            valid_from=rel.valid_from,
            valid_until=rel.valid_until,
            source_document_ids=list(rel.source_document_ids),
        )
        for rel, score in relationship_hits
    ]


def project_document_stubs(
    chunk_hits: list[tuple[Chunk, float]],
    entities: list[RecallEntity],
    relationships: list[RecallRelationship] | None = None,
) -> list[DocumentProjection]:
    """Build the deduplicated ``DocumentProjection`` list.

    Append order (part of the byte-parity contract): documents referenced by
    the returned chunks first (with their full projection), then stubs for
    documents referenced only by entities, then stubs for documents referenced
    only by relationships. A ``None`` / empty ``relationships`` list makes the
    third loop a no-op, matching Chronicle (no relationship surface) exactly.

    The producer invariant is that every id in
    ``entities[i].source_document_ids`` / ``relationships[i].source_document_ids``
    appears in the returned list; the entity / relationship loops add stubs for
    any not already covered by a chunk.
    """
    seen_doc_ids: set[UUID] = set()
    documents: list[DocumentProjection] = []
    for chunk, _ in chunk_hits:
        if chunk.document_id in seen_doc_ids:
            continue
        seen_doc_ids.add(chunk.document_id)
        src = chunk.source_document
        documents.append(
            DocumentProjection(
                id=chunk.document_id,
                created_at=chunk.created_at,
                source_type=(src.source_type if src and src.source_type else "library"),
                title=(src.title if src and src.title else None),
                source=(src.source if src and src.source else None),
                source_timestamp=(src.source_timestamp if src else None),
                metadata=dict(chunk.metadata or {}),
            )
        )
    for entity in entities:
        for did in entity.source_document_ids:
            if did in seen_doc_ids:
                continue
            seen_doc_ids.add(did)
            documents.append(DocumentProjection(id=did, created_at=datetime.now(UTC), source_type="library"))
    for rel in relationships or []:
        for did in rel.source_document_ids:
            if did in seen_doc_ids:
                continue
            seen_doc_ids.add(did)
            documents.append(DocumentProjection(id=did, created_at=datetime.now(UTC), source_type="library"))
    return documents


__all__: list[str] = [
    "project_document_stubs",
    "project_entities",
    "project_relationships",
]

"""Public projection dataclasses returned by recall operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from khora.core.models.entity import CommunityNode

if TYPE_CHECKING:
    from khora.khora import LLMUsage


@dataclass(slots=True, frozen=True)
class DocumentProjection:
    """User-facing document metadata projection surfaced in recall responses.

    Only ``id``, ``created_at``, and ``source_type`` are guaranteed; all
    other fields default to ``None`` (or ``{}`` for ``metadata``) when the
    caller didn't supply them at remember-time.

    ``source_type`` and ``source_name`` are free-form strings — Khora does
    not validate or enumerate them. Downstream applications may impose
    their own taxonomies if they need closed-list semantics.
    """

    id: UUID
    created_at: datetime
    source_type: str = "library"
    title: str | None = None
    external_id: str | None = None
    source: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    content_type: str | None = None
    source_timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RecallChunk:
    """A chunk in a recall response.

    Score is a typed field, not a tuple position.

    ``document_id`` is a foreign key into the top-level
    ``RecallResult.documents`` list — the full ``DocumentProjection`` lives
    there to avoid duplicating the same document across many chunks.

    ``chunker_info`` is the chunker's per-chunk output dict, stored in its
    own dedicated column. The chunker self-identification contract
    requires at minimum ``{"chunker": "<name>"}``.

    ``occurred_at`` is engine-populated.

    ``connected_entity_ids`` is the set of entities the recall pipeline
    linked to this chunk, derived by inverting
    ``RecallEntity.source_chunk_ids`` after the engine returns. Empty
    list semantics: **unknown**, not "no edges" — engines that return no
    entities (skeleton, chronicle without entity hits, graph-less
    stacks) leave this empty even when the underlying graph backend
    might know edges to entities that simply weren't returned in this
    result.
    """

    id: UUID
    document_id: UUID
    content: str
    score: float
    created_at: datetime
    occurred_at: datetime | None = None
    connected_entity_ids: list[UUID] = field(default_factory=list)
    chunker_info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RecallEntity:
    """An entity in a recall response, with provenance back to documents and chunks."""

    id: UUID
    name: str
    entity_type: str
    description: str
    score: float
    attributes: dict[str, Any]
    mention_count: int
    source_document_ids: list[UUID]
    source_chunk_ids: list[UUID]


@dataclass(slots=True, frozen=True)
class RecallRelationship:
    """A relationship in a recall response, with temporal validity bounds."""

    id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: str
    score: float
    valid_from: datetime | None
    valid_until: datetime | None
    source_document_ids: list[UUID]


@dataclass(slots=True, frozen=True)
class RecallResult:
    """Result of a recall operation — a JSON-serializable response projection.

    ``documents`` is the deduplicated set of source documents referenced
    by any chunk, entity, or relationship in the result. Every
    ``chunks[i].document_id`` and every id in
    ``entities[i].source_document_ids`` /
    ``relationships[i].source_document_ids`` is guaranteed to match some
    ``documents[j].id`` (producer-enforced invariant).

    ``relationships`` is always present (possibly empty) — engines without
    a graph backend return ``[]``.

    ``communities`` is the deduplicated set of materialized dream community
    summaries (#1276) the result's matched entities belong to (the GraphRAG
    query-time payoff, #1308). Always present (possibly empty) — empty on a
    stack without materialized communities or a backend lacking the
    community reader. Deduplicated by community id and capped.

    ``engine_info`` is a free-form dict of engine-specific telemetry.
    Every engine MUST emit the key ``"engine": <strategy-name>`` so
    consumers can route on producer identity.
    """

    query: str
    namespace_id: UUID
    documents: list[DocumentProjection]
    chunks: list[RecallChunk]
    entities: list[RecallEntity]
    relationships: list[RecallRelationship]
    usage: list[LLMUsage] = field(default_factory=list)
    engine_info: dict[str, Any] = field(default_factory=dict)
    communities: list[CommunityNode] = field(default_factory=list)


__all__ = [
    "CommunityNode",
    "DocumentProjection",
    "RecallChunk",
    "RecallEntity",
    "RecallRelationship",
    "RecallResult",
]

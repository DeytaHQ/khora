"""Entity and relationship models for Khora Memory Lake.

Entities represent extracted knowledge (people, organizations, concepts, etc.)
and relationships connect them in a knowledge graph stored in Neo4j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True)
class Entity:
    """An extracted entity from a document.

    Entities are nodes in the knowledge graph stored in Neo4j.
    They represent people, organizations, concepts, and other
    knowledge extracted from documents.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    name: str = ""
    entity_type: str = "CONCEPT"
    description: str = ""

    # Attributes from extraction
    attributes: dict[str, Any] = field(default_factory=dict)

    # Source provenance — canonical SaaS tool that produced this entity
    source_tool: str = ""

    # Source tracking
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)
    mention_count: int = 1

    # Embedding for entity similarity search
    embedding: list[float] | None = None
    embedding_model: str = ""

    # Temporal validity
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Confidence score from extraction
    confidence: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Sanitize fields that must not be None (LLM sometimes returns null)."""
        if self.name is None:
            self.name = ""
        if len(self.name) > 512:
            self.name = self.name[:509] + "..."
        if self.description is None:
            self.description = ""
        if self.source_tool is None:
            self.source_tool = ""

    def validate(self) -> None:
        """Validate and clean attributes using the registered schema for this entity type."""
        from khora.core.models.schemas import validate_attributes

        self.attributes = validate_attributes(self.entity_type, self.attributes)

    def merge_with(self, other: Entity) -> None:
        """Merge another entity into this one (deduplication)."""
        # Combine source references
        for doc_id in other.source_document_ids:
            if doc_id not in self.source_document_ids:
                self.source_document_ids.append(doc_id)
        for chunk_id in other.source_chunk_ids:
            if chunk_id not in self.source_chunk_ids:
                self.source_chunk_ids.append(chunk_id)

        # Update mention count
        self.mention_count += other.mention_count

        # Merge attributes (prefer existing)
        # Handle case where attributes might be a list instead of dict (defensive)
        other_attrs = other.attributes if isinstance(other.attributes, dict) else {}
        for key, value in other_attrs.items():
            if key not in self.attributes:
                self.attributes[key] = value

        # Update confidence (take max)
        self.confidence = max(self.confidence, other.confidence)

        # Update description if empty
        if not self.description and other.description:
            self.description = other.description

        self.updated_at = datetime.now(UTC)


@dataclass(slots=True)
class Relationship:
    """A relationship between two entities.

    Relationships are edges in the knowledge graph stored in Neo4j.
    They connect entities and describe how they relate to each other.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    source_entity_id: UUID = field(default_factory=uuid4)
    target_entity_id: UUID = field(default_factory=uuid4)
    relationship_type: str = "RELATES_TO"
    description: str = ""

    # Additional properties
    properties: dict[str, Any] = field(default_factory=dict)

    # Source tracking
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)

    # Temporal validity
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    # Confidence and weight
    confidence: float = 1.0
    weight: float = 1.0

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class Episode:
    """An episodic memory representing a temporal event or experience.

    Episodes capture time-bound events with associated entities,
    supporting temporal queries and event-based recall.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""

    # Temporal bounds
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_seconds: int | None = None

    # Associated entities
    entity_ids: list[UUID] = field(default_factory=list)

    # Source tracking
    source_document_ids: list[UUID] = field(default_factory=list)
    source_chunk_ids: list[UUID] = field(default_factory=list)

    # Episode embedding for similarity search
    embedding: list[float] | None = None
    embedding_model: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def end_time(self) -> datetime | None:
        """Calculate the end time of the episode."""
        if self.duration_seconds is not None:
            from datetime import timedelta

            return self.occurred_at + timedelta(seconds=self.duration_seconds)
        return None

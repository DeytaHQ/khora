"""Entity and relationship models for Khora Memory Lake.

Entities represent extracted knowledge (people, organizations, concepts, etc.)
and relationships connect them in a knowledge graph stored in Neo4j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EntityType(str, Enum):
    """Standard entity types for knowledge extraction."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    DATE = "DATE"
    CUSTOM = "CUSTOM"


class RelationshipType(str, Enum):
    """Standard relationship types for knowledge graphs."""

    # Person relationships
    WORKS_FOR = "WORKS_FOR"
    KNOWS = "KNOWS"
    MANAGES = "MANAGES"
    REPORTS_TO = "REPORTS_TO"
    COLLABORATES_WITH = "COLLABORATES_WITH"

    # Organization relationships
    OWNS = "OWNS"
    PART_OF = "PART_OF"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"

    # Location relationships
    LOCATED_IN = "LOCATED_IN"
    HEADQUARTERED_IN = "HEADQUARTERED_IN"

    # Concept relationships
    RELATES_TO = "RELATES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    DERIVED_FROM = "DERIVED_FROM"

    # Temporal relationships
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    CONCURRENT_WITH = "CONCURRENT_WITH"

    # Generic
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CUSTOM = "CUSTOM"


@dataclass
class Entity:
    """An extracted entity from a document.

    Entities are nodes in the knowledge graph stored in Neo4j.
    They represent people, organizations, concepts, and other
    knowledge extracted from documents.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    name: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    description: str = ""

    # Attributes from extraction
    attributes: dict[str, Any] = field(default_factory=dict)

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
        for key, value in other.attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value

        # Update confidence (take max)
        self.confidence = max(self.confidence, other.confidence)

        # Update description if empty
        if not self.description and other.description:
            self.description = other.description

        self.updated_at = datetime.now(UTC)


@dataclass
class Relationship:
    """A relationship between two entities.

    Relationships are edges in the knowledge graph stored in Neo4j.
    They connect entities and describe how they relate to each other.
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    source_entity_id: UUID = field(default_factory=uuid4)
    target_entity_id: UUID = field(default_factory=uuid4)
    relationship_type: RelationshipType = RelationshipType.RELATES_TO
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


@dataclass
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

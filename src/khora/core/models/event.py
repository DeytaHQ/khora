"""Event sourcing models for Khora.

All changes to Khora are captured as immutable events,
enabling perfect audit trails, temporal queries, and replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class EventType(str, Enum):
    """Types of events in Khora."""

    # Document events
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"

    # Chunk events
    CHUNK_CREATED = "chunk.created"
    CHUNK_EMBEDDED = "chunk.embedded"
    CHUNK_DELETED = "chunk.deleted"
    # Fired after all entity events for a single chunk have been dispatched.
    # Carries the per-chunk entity set so subscribers can express
    # co-occurrence filters like "alert when X and Y appear in the same
    # chunk" — single-entity events cannot (Issue #579 Phase 2 Item B).
    CHUNK_ENTITIES_RESOLVED = "chunk.entities_resolved"

    # Entity events
    ENTITY_CREATED = "entity.created"
    ENTITY_UPDATED = "entity.updated"
    ENTITY_MERGED = "entity.merged"
    ENTITY_DELETED = "entity.deleted"

    # Relationship events
    RELATIONSHIP_CREATED = "relationship.created"
    RELATIONSHIP_UPDATED = "relationship.updated"
    RELATIONSHIP_DELETED = "relationship.deleted"

    # Episode events
    EPISODE_CREATED = "episode.created"
    EPISODE_UPDATED = "episode.updated"
    EPISODE_DELETED = "episode.deleted"

    # Namespace events
    NAMESPACE_CREATED = "namespace.created"
    NAMESPACE_UPDATED = "namespace.updated"
    NAMESPACE_DELETED = "namespace.deleted"

    # Sync events
    SYNC_STARTED = "sync.started"
    SYNC_COMPLETED = "sync.completed"
    SYNC_FAILED = "sync.failed"
    SYNC_CHECKPOINT = "sync.checkpoint"

    # Query events (for analytics)
    QUERY_EXECUTED = "query.executed"

    # Recall/search events — fired by Khora.recall() so operators can
    # subscribe to query-level signals without adding query-side filter
    # mechanics (Phase 1 of #576).
    RECALL_REQUESTED = "recall.requested"
    RECALL_RESULTS_READY = "recall.results_ready"
    RECALL_COMPLETED = "recall.completed"


@dataclass
class MemoryEvent:
    """An immutable event in Khora.

    Events form an append-only log of all changes, enabling:
    - Complete audit trail
    - Temporal queries (state at any point in time)
    - Event replay for recovery or migration
    - Change data capture for external systems
    """

    id: UUID = field(default_factory=uuid4)
    namespace_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.DOCUMENT_CREATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Reference to the affected resource
    resource_type: str = ""  # document, chunk, entity, relationship, episode
    resource_id: UUID = field(default_factory=uuid4)

    # Event data
    data: dict[str, Any] = field(default_factory=dict)

    # Previous state for update events (enables undo)
    previous_data: dict[str, Any] | None = None

    # Actor who triggered the event
    actor_id: str | None = None
    actor_type: str = "system"  # system, user, api, pipeline

    # Correlation ID for tracking related events
    correlation_id: UUID | None = None

    # Version for optimistic locking
    version: int = 1

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Extract resource type from event type if not set."""
        if not self.resource_type and self.event_type:
            # Extract from event type (e.g., "document.created" -> "document")
            self.resource_type = self.event_type.value.split(".")[0]

    @classmethod
    def document_created(cls, namespace_id: UUID, document_id: UUID, data: dict[str, Any], **kwargs) -> MemoryEvent:
        """Create a document created event."""
        return cls(
            namespace_id=namespace_id,
            event_type=EventType.DOCUMENT_CREATED,
            resource_type="document",
            resource_id=document_id,
            data=data,
            **kwargs,
        )

    @classmethod
    def entity_created(cls, namespace_id: UUID, entity_id: UUID, data: dict[str, Any], **kwargs) -> MemoryEvent:
        """Create an entity created event."""
        return cls(
            namespace_id=namespace_id,
            event_type=EventType.ENTITY_CREATED,
            resource_type="entity",
            resource_id=entity_id,
            data=data,
            **kwargs,
        )

    @classmethod
    def chunk_embedded(cls, namespace_id: UUID, chunk_id: UUID, data: dict[str, Any], **kwargs) -> MemoryEvent:
        """Create a chunk embedded event."""
        return cls(
            namespace_id=namespace_id,
            event_type=EventType.CHUNK_EMBEDDED,
            resource_type="chunk",
            resource_id=chunk_id,
            data=data,
            **kwargs,
        )

    @classmethod
    def relationship_created(
        cls, namespace_id: UUID, relationship_id: UUID, data: dict[str, Any], **kwargs
    ) -> MemoryEvent:
        """Create a relationship created event."""
        return cls(
            namespace_id=namespace_id,
            event_type=EventType.RELATIONSHIP_CREATED,
            resource_type="relationship",
            resource_id=relationship_id,
            data=data,
            **kwargs,
        )

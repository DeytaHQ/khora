# Memory Events

Khora uses event sourcing to capture all changes to Khora. Every create, update, and delete operation is recorded as an immutable event.

## MemoryEvent Model

Located at `src/khora/core/models/event.py`.

```python
@dataclass
class MemoryEvent:
    id: UUID                        # Unique event ID
    namespace_id: UUID              # Namespace scope
    event_type: EventType           # Type of event
    timestamp: datetime             # When event occurred
    resource_type: str              # "document", "entity", etc.
    resource_id: UUID               # ID of affected resource

    # Payload
    data: dict[str, Any] = field(default_factory=dict)
    previous_data: dict[str, Any] | None = None  # For updates

    # Actor information
    actor_id: str | None = None     # Who triggered the event
    actor_type: str = "system"      # user, system, api, pipeline

    # Transaction correlation
    correlation_id: UUID | None = None

    # Versioning
    version: int = 1

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
```

## Event Types

```python
class EventType(str, Enum):
    # Document lifecycle
    DOCUMENT_CREATED = "document.created"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"

    # Chunk operations
    CHUNK_CREATED = "chunk.created"
    CHUNK_EMBEDDED = "chunk.embedded"
    CHUNK_DELETED = "chunk.deleted"

    # Entity operations
    ENTITY_CREATED = "entity.created"
    ENTITY_UPDATED = "entity.updated"
    ENTITY_MERGED = "entity.merged"
    ENTITY_DELETED = "entity.deleted"

    # Relationship operations
    RELATIONSHIP_CREATED = "relationship.created"
    RELATIONSHIP_UPDATED = "relationship.updated"
    RELATIONSHIP_DELETED = "relationship.deleted"

    # Episode operations
    EPISODE_CREATED = "episode.created"
    EPISODE_UPDATED = "episode.updated"
    EPISODE_DELETED = "episode.deleted"

    # Namespace operations
    NAMESPACE_CREATED = "namespace.created"
    NAMESPACE_UPDATED = "namespace.updated"
    NAMESPACE_DELETED = "namespace.deleted"

    # Sync operations
    SYNC_STARTED = "sync.started"
    SYNC_COMPLETED = "sync.completed"
    SYNC_FAILED = "sync.failed"
    SYNC_CHECKPOINT = "sync.checkpoint"

    # Query tracking
    QUERY_EXECUTED = "query.executed"
```

## Resource Types

| Resource Type | Description | Example Events |
|---------------|-------------|----------------|
| `document` | Source documents | created, processed, deleted |
| `chunk` | Document chunks | created, embedded |
| `entity` | Knowledge graph nodes | created, merged, updated |
| `relationship` | Knowledge graph edges | created, updated |
| `episode` | Temporal events | created, updated |
| `namespace` | Memory namespaces | created, updated |
| `sync` | Sync operations | started, completed, failed |
| `query` | Search queries | executed |

## Actor Types

| Actor Type | Description | Example |
|------------|-------------|---------|
| `system` | Internal system operations | Background cleanup |
| `user` | User-initiated via UI | Manual document upload |
| `api` | Programmatic API call | `kb.remember()` from application code |
| `pipeline` | Prefect pipeline | Ingestion pipeline |

## Factory Methods

`MemoryEvent` provides four factory classmethods that wrap the common
event shapes for the four most-frequent resource types. Each takes
`(namespace_id, <resource>_id, data, **kwargs)`; the `**kwargs`
forward to the dataclass constructor (`actor_id`, `actor_type`,
`correlation_id`, `causation_id`, …).

```python
# Document created
event = MemoryEvent.document_created(
    namespace_id=ns_id,
    document_id=doc.id,
    data={
        "title": doc.title,
        "source": doc.source,
        "checksum": doc.checksum,
    },
    actor_id="user:123",
    actor_type="user",
)

# Entity created
event = MemoryEvent.entity_created(
    namespace_id=ns_id,
    entity_id=entity.id,
    data={
        "name": entity.name,
        "type": entity.entity_type.value,
        "confidence": entity.confidence,
    },
)

# Chunk embedded
event = MemoryEvent.chunk_embedded(
    namespace_id=ns_id,
    chunk_id=chunk.id,
    data={
        "model": "text-embedding-3-small",
        "dimension": 1536,
    },
)

# Relationship created
event = MemoryEvent.relationship_created(
    namespace_id=ns_id,
    relationship_id=rel.id,
    data={
        "source_id": str(rel.source_entity_id),
        "target_id": str(rel.target_entity_id),
        "type": rel.relationship_type,
    },
)
```

For event types without a factory (document updates, entity merges,
sync events, deletes, anything custom), construct `MemoryEvent`
directly with `event_type=EventType.<KIND>`:

```python
from khora.core.models.event import MemoryEvent, EventType

event = MemoryEvent(
    event_type=EventType.DOCUMENT_DELETED,
    namespace_id=ns_id,
    resource_id=doc.id,
    resource_type="document",
    data={"reason": "manual_purge"},
)
```

See [Event Types](#event-types) above for the full enum.

## Correlation IDs

Related events share a correlation ID for transaction tracing:

```python
import uuid

# Generate correlation ID for ingestion transaction
correlation_id = uuid.uuid4()

# All events in this ingestion share the same correlation ID
events = [
    MemoryEvent.document_created(
        ...,
        correlation_id=correlation_id,
    ),
    MemoryEvent.chunk_embedded(
        ...,
        correlation_id=correlation_id,
    ),
    MemoryEvent.entity_created(
        ...,
        correlation_id=correlation_id,
    ),
    MemoryEvent.relationship_created(
        ...,
        correlation_id=correlation_id,
    ),
]
```

This enables:
- Tracing all events from a single operation
- Understanding causality chains
- Debugging pipeline issues

## Previous Data

Update events capture the previous state for rollback and auditing:

```python
# Entity update with previous state
event = MemoryEvent(
    event_type=EventType.ENTITY_UPDATED,
    resource_type="entity",
    resource_id=entity.id,
    data={
        "confidence": 0.95,
        "mention_count": 10,
    },
    previous_data={
        "confidence": 0.85,
        "mention_count": 8,
    },
)

# Can compute diff
changes = {
    key: (event.previous_data.get(key), event.data.get(key))
    for key in set(event.data) | set(event.previous_data or {})
    if event.previous_data.get(key) != event.data.get(key)
}
# changes = {"confidence": (0.85, 0.95), "mention_count": (8, 10)}
```

## Event Storage

Events are stored in PostgreSQL:

```sql
CREATE TABLE memory_events (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID NOT NULL,
    data JSONB NOT NULL,
    previous_data JSONB,
    actor_id VARCHAR(255),
    actor_type VARCHAR(50) DEFAULT 'system',
    correlation_id UUID,
    version INTEGER DEFAULT 1,
    metadata JSONB
);

-- Query indexes
CREATE INDEX idx_events_namespace_time
    ON memory_events(namespace_id, timestamp DESC);

CREATE INDEX idx_events_resource
    ON memory_events(resource_type, resource_id);

CREATE INDEX idx_events_correlation
    ON memory_events(correlation_id);

CREATE INDEX idx_events_type
    ON memory_events(event_type);
```

## Querying Events

```python
# Get recent events
events = await kb.storage.get_events(
    namespace_id,
    limit=50,
)

# Filter by event type
entity_events = await kb.storage.get_events(
    namespace_id,
    event_types=["entity.created", "entity.merged"],
)

# Filter by resource
doc_history = await kb.storage.get_events(
    namespace_id,
    resource_type="document",
    resource_id=document_id,
)

# Time range query
recent = await kb.storage.get_events(
    namespace_id,
    after=datetime.utcnow() - timedelta(hours=24),
)

# Get correlated events
transaction_events = await kb.storage.get_events(
    namespace_id,
    correlation_id=correlation_id,
)
```

## Use Cases

### Audit Trail

```python
# Who made changes to this entity?
events = await kb.storage.get_events(
    namespace_id,
    resource_type="entity",
    resource_id=entity_id,
)

for event in events:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  Actor: {event.actor_type}:{event.actor_id}")
    print(f"  Changes: {event.data}")
```

### Change History

```python
# Show entity change history
events = await kb.storage.get_events(
    namespace_id,
    resource_type="entity",
    resource_id=entity_id,
    event_types=["entity.created", "entity.updated", "entity.merged"],
)

for event in sorted(events, key=lambda e: e.timestamp):
    if event.event_type == EventType.ENTITY_CREATED:
        print(f"Created: {event.data}")
    elif event.event_type == EventType.ENTITY_UPDATED:
        print(f"Updated: {event.previous_data} → {event.data}")
    elif event.event_type == EventType.ENTITY_MERGED:
        print(f"Merged with: {event.data['merged_entity_name']}")
```

### Pipeline Debugging

```python
# Get all events from a specific pipeline run
events = await kb.storage.get_events(
    namespace_id,
    correlation_id=pipeline_correlation_id,
)

# Analyze pipeline execution
for event in sorted(events, key=lambda e: e.timestamp):
    print(f"{event.timestamp}: {event.event_type} - {event.resource_type}")
```

## Next Steps

- [Event Sourcing](../architecture/event-sourcing.md) - Architecture details
- [Documents & Chunks](documents-chunks.md) - Content models
- [Knowledge Graph](knowledge-graph.md) - Entity models

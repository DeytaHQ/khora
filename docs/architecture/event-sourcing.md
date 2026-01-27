# Event Sourcing

Khora implements event sourcing as a core pattern, capturing all state changes as immutable events. This enables complete audit trails, temporal queries, and potential event replay for recovery or migration.

## Overview

Every change to the memory lake is recorded as an immutable event in an append-only log:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Event Store                                   │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ Event 1      │ document.created   { id, content, ... }      │
│  ├──────────────┤                                               │
│  │ Event 2      │ chunk.created      { id, document_id, ... }  │
│  ├──────────────┤                                               │
│  │ Event 3      │ entity.created     { id, name, type, ... }   │
│  ├──────────────┤                                               │
│  │ Event 4      │ entity.merged      { merged_id, into_id }    │
│  ├──────────────┤                                               │
│  │ Event 5      │ document.deleted   { id }                    │
│  ├──────────────┤                                               │
│  │    ...       │                                               │
│  └──────────────┘                                               │
│                                                                  │
│  (Append-only, immutable, ordered by timestamp)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Event Structure

Each event is a `MemoryEvent` instance:

```python
@dataclass
class MemoryEvent:
    id: UUID                    # Unique event ID
    namespace_id: UUID          # Namespace scope
    event_type: EventType       # Type of event
    timestamp: datetime         # When event occurred
    resource_type: str          # "document", "entity", etc.
    resource_id: UUID           # ID of affected resource
    data: dict[str, Any]        # Event payload
    previous_data: dict | None  # Previous state (for updates)
    actor_id: str | None        # Who triggered the event
    actor_type: str             # "user", "system", "api", "pipeline"
    correlation_id: UUID | None # Link related events
    version: int                # For optimistic locking
    metadata: dict[str, Any]    # Additional context
```

## Event Types

Khora defines 25+ event types organized by resource:

### Document Events

| Event Type | Description |
|------------|-------------|
| `document.created` | New document added |
| `document.updated` | Document metadata changed |
| `document.deleted` | Document removed |
| `document.processed` | Document successfully processed |
| `document.failed` | Document processing failed |

### Chunk Events

| Event Type | Description |
|------------|-------------|
| `chunk.created` | New chunk created |
| `chunk.embedded` | Embedding generated |
| `chunk.deleted` | Chunk removed |

### Entity Events

| Event Type | Description |
|------------|-------------|
| `entity.created` | New entity extracted |
| `entity.updated` | Entity attributes changed |
| `entity.merged` | Entity merged with duplicate |
| `entity.deleted` | Entity removed |

### Relationship Events

| Event Type | Description |
|------------|-------------|
| `relationship.created` | New relationship created |
| `relationship.updated` | Relationship changed |
| `relationship.deleted` | Relationship removed |

### Episode Events

| Event Type | Description |
|------------|-------------|
| `episode.created` | New temporal episode |
| `episode.updated` | Episode modified |
| `episode.deleted` | Episode removed |

### Namespace Events

| Event Type | Description |
|------------|-------------|
| `namespace.created` | New namespace created |
| `namespace.updated` | Namespace settings changed |
| `namespace.deleted` | Namespace removed |

### Sync Events

| Event Type | Description |
|------------|-------------|
| `sync.started` | Sync pipeline started |
| `sync.completed` | Sync pipeline finished |
| `sync.failed` | Sync pipeline failed |
| `sync.checkpoint` | Checkpoint updated |

### Query Events

| Event Type | Description |
|------------|-------------|
| `query.executed` | Query executed (for analytics) |

## Actor Types

Events track who triggered them:

```python
actor_type = "system"   # Internal system operations
actor_type = "user"     # User-initiated via UI
actor_type = "api"      # API call
actor_type = "pipeline" # Prefect pipeline
```

## Correlation IDs

Related events share a `correlation_id` for transaction tracing:

```python
# All events from a single ingestion share correlation_id
correlation_id = uuid4()

# Stage document
event1 = MemoryEvent(
    event_type=EventType.DOCUMENT_CREATED,
    correlation_id=correlation_id,
    ...
)

# Create chunks
event2 = MemoryEvent(
    event_type=EventType.CHUNK_CREATED,
    correlation_id=correlation_id,  # Same correlation ID
    ...
)

# Extract entities
event3 = MemoryEvent(
    event_type=EventType.ENTITY_CREATED,
    correlation_id=correlation_id,  # Same correlation ID
    ...
)
```

## Factory Methods

`MemoryEvent` provides factory methods for common events:

```python
# Document created
event = MemoryEvent.document_created(
    namespace_id=namespace_id,
    document_id=doc.id,
    data={"title": doc.title, "source": doc.source},
)

# Entity created
event = MemoryEvent.entity_created(
    namespace_id=namespace_id,
    entity_id=entity.id,
    data={"name": entity.name, "type": entity.entity_type},
)

# Chunk embedded
event = MemoryEvent.chunk_embedded(
    namespace_id=namespace_id,
    chunk_id=chunk.id,
    data={"model": "text-embedding-3-small"},
)
```

## Event Store Implementation

The event store is PostgreSQL-based for reliability:

```python
class PostgreSQLEventStore:
    """Append-only event store."""

    async def append_event(self, event: MemoryEvent) -> MemoryEvent:
        """Append a single event."""
        ...

    async def append_events_batch(
        self, events: list[MemoryEvent]
    ) -> list[MemoryEvent]:
        """Append multiple events in a transaction."""
        ...

    async def get_events(
        self,
        namespace_id: UUID,
        *,
        event_types: list[str] | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEvent]:
        """Query events with filters."""
        ...
```

### Database Schema

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

-- Indexes for common queries
CREATE INDEX idx_events_namespace_type
    ON memory_events(namespace_id, event_type);

CREATE INDEX idx_events_resource
    ON memory_events(resource_type, resource_id);

CREATE INDEX idx_events_timestamp
    ON memory_events(timestamp DESC);

CREATE INDEX idx_events_correlation
    ON memory_events(correlation_id);
```

## Use Cases

### Audit Trail

Query all events for compliance:

```python
events = await lake.storage.get_events(
    namespace_id,
    resource_type="document",
    resource_id=document_id,
)

for event in events:
    print(f"{event.timestamp}: {event.event_type} by {event.actor_id}")
```

### Temporal Queries

Reconstruct state at a point in time:

```python
# Get all events before a specific date
events = await lake.storage.get_events(
    namespace_id,
    before=datetime(2024, 1, 1),
)

# Filter to only see what existed at that time
created = {e.resource_id for e in events if "created" in e.event_type}
deleted = {e.resource_id for e in events if "deleted" in e.event_type}
active_at_date = created - deleted
```

### Change Data Capture

Stream events to external systems:

```python
async def sync_to_external_system(namespace_id: UUID):
    last_sync = await get_last_sync_time()

    events = await lake.storage.get_events(
        namespace_id,
        after=last_sync,
    )

    for event in events:
        await external_system.process(event)

    await set_last_sync_time(datetime.utcnow())
```

### Event Replay

Rebuild state from events (disaster recovery):

```python
async def rebuild_namespace(namespace_id: UUID):
    events = await lake.storage.get_events(
        namespace_id,
        event_types=["document.created", "entity.created"],
    )

    for event in sorted(events, key=lambda e: e.timestamp):
        if event.event_type == EventType.DOCUMENT_CREATED:
            await recreate_document(event.data)
        elif event.event_type == EventType.ENTITY_CREATED:
            await recreate_entity(event.data)
```

### Analytics

Track query patterns:

```python
# Get query events for analytics
query_events = await lake.storage.get_events(
    namespace_id,
    event_types=["query.executed"],
    after=datetime.utcnow() - timedelta(days=7),
)

# Analyze query patterns
query_counts = Counter(e.data.get("query_type") for e in query_events)
```

## Event Sourcing with Previous Data

Update events capture the previous state:

```python
# Entity update event with rollback capability
event = MemoryEvent(
    event_type=EventType.ENTITY_UPDATED,
    resource_id=entity.id,
    data={"confidence": 0.95},
    previous_data={"confidence": 0.85},  # Original value
)
```

This enables:
- Undo operations
- Diff visualization
- Compliance auditing of changes

## Best Practices

1. **Always include correlation_id** for related operations
2. **Batch event writes** for performance during ingestion
3. **Include meaningful actor_id** for audit trails
4. **Keep data payloads minimal** (store IDs, not full objects)
5. **Query with limits** to avoid memory issues

## API Examples

### Appending Events

```python
# Single event
event = await lake.storage.append_event(
    MemoryEvent.entity_created(
        namespace_id=ns_id,
        entity_id=entity.id,
        data={"name": entity.name},
        actor_id="pipeline:ingestion",
        actor_type="pipeline",
    )
)

# Batch (more efficient)
events = await lake.storage.append_events_batch([
    MemoryEvent.chunk_created(...),
    MemoryEvent.chunk_embedded(...),
    MemoryEvent.entity_created(...),
])
```

### Querying Events

```python
# Get recent events
recent = await lake.storage.get_events(
    namespace_id,
    limit=50,
)

# Filter by type
entity_events = await lake.storage.get_events(
    namespace_id,
    resource_type="entity",
)

# Get events for specific resource
doc_history = await lake.storage.get_events(
    namespace_id,
    resource_type="document",
    resource_id=document_id,
)

# Time range query
last_week = await lake.storage.get_events(
    namespace_id,
    after=datetime.utcnow() - timedelta(days=7),
)
```

## Next Steps

- [Architecture Overview](overview.md) - System design
- [Storage Backends](storage-backends.md) - PostgreSQL, pgvector, Neo4j

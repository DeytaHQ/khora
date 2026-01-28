# Event Sourcing

Every change to Khora is recorded. When you create a document, merge an entity, or delete a chunk - each action becomes an immutable event in an append-only log. This isn't just for auditing (though it's great for that) - it enables time travel through your data's history.

## The Idea

Instead of just storing the current state, we store every change that led to that state:

```
Traditional Database:
+------------------+
| Document: "abc"  |  <- Only current state
| Status: complete |
+------------------+

Event-Sourced:
+------------------+
| Event 1: Created |  "abc" created with status "pending"
+------------------+
| Event 2: Updated |  status changed to "processing"
+------------------+
| Event 3: Updated |  status changed to "completed"
+------------------+
         |
         v
    Current state can be derived by replaying events
```

The event log is the source of truth. Current state is just a view computed from that log.

## What Gets Recorded

**Every meaningful action:**

- Document created, updated, deleted, processed, failed
- Chunk created, embedded, deleted
- Entity created, updated, merged, deleted
- Relationship created, updated, deleted, inferred
- Namespace created, updated, deleted
- Sync started, completed, failed, checkpointed
- Query executed (for analytics)

## Event Structure

Each event captures what happened, when, and who did it:

```python
MemoryEvent(
    id=uuid4(),
    namespace_id=namespace_id,

    # What happened
    event_type="entity.merged",
    resource_type="entity",
    resource_id=merged_entity_id,

    # The details
    data={
        "merged_into": target_entity_id,
        "reason": "duplicate_name",
        "attributes_merged": ["description", "confidence"]
    },
    previous_data={
        "name": "Einstein",
        "description": "physicist"
    },

    # When and who
    timestamp=datetime.now(UTC),
    actor_id="pipeline:ingestion",
    actor_type="pipeline",

    # Link related events
    correlation_id=ingestion_batch_id,

    # For optimistic concurrency
    version=1
)
```

### The Fields Explained

**`event_type`** - What happened. Examples:
- `document.created` - New document added
- `entity.merged` - Two entities unified
- `relationship.inferred` - Relationship created by inference
- `sync.checkpoint` - Sync progress saved

**`resource_type` / `resource_id`** - What was affected. Lets you query "show me everything that happened to this entity."

**`data`** - Event payload. Contains the new state or action details.

**`previous_data`** - For updates, the old state. Enables undo and diff views.

**`actor_id` / `actor_type`** - Who triggered this:
- `user:alice@example.com` / `user`
- `pipeline:ingestion` / `pipeline`
- `api:client-123` / `api`
- `system:expander` / `system`

**`correlation_id`** - Links related events. When you call `remember()`, all resulting events (document created, chunks created, entities extracted) share one correlation ID. Essential for understanding "what happened as a result of X?"

## Using the Event Store

### Recording Events

Events are recorded automatically by Khora during operations. But you can also record explicitly:

```python
# Single event
event = await storage.append_event(
    MemoryEvent.entity_created(
        namespace_id=namespace_id,
        entity_id=entity.id,
        data={"name": entity.name, "type": entity.entity_type},
        actor_id="user:alice",
        actor_type="user"
    )
)

# Batch (more efficient for pipelines)
events = await storage.append_events_batch([
    MemoryEvent.chunk_created(...),
    MemoryEvent.chunk_embedded(...),
    MemoryEvent.entity_created(...)
])
```

### Querying Events

Find out what happened:

```python
# Recent events in a namespace
recent = await storage.get_events(
    namespace_id,
    limit=50
)

# History of a specific document
doc_history = await storage.get_events(
    namespace_id,
    resource_type="document",
    resource_id=document_id
)

# All entity merges last week
merges = await storage.get_events(
    namespace_id,
    event_types=["entity.merged"],
    after=datetime.now() - timedelta(days=7)
)

# Everything from a specific ingestion
batch_events = await storage.get_events(
    namespace_id,
    correlation_id=ingestion_correlation_id
)
```

## What You Can Do With This

### Audit Trails

"Who changed this and when?"

```python
# Get full history of an entity
history = await storage.get_events(
    namespace_id,
    resource_type="entity",
    resource_id=entity_id
)

for event in history:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  By: {event.actor_id}")
    if event.previous_data:
        print(f"  Changed from: {event.previous_data}")
    print(f"  Changed to: {event.data}")
```

### Time Travel

"What did we know on January 1st?"

```python
# Get all events before that date
events = await storage.get_events(
    namespace_id,
    before=datetime(2024, 1, 1)
)

# Figure out what existed then
created = {e.resource_id for e in events if "created" in e.event_type}
deleted = {e.resource_id for e in events if "deleted" in e.event_type}
existed_on_jan1 = created - deleted

# You could even reconstruct the full state by replaying events
```

### Change Data Capture

Stream changes to external systems:

```python
async def sync_to_warehouse():
    last_sync = await get_last_sync_timestamp()

    new_events = await storage.get_events(
        namespace_id,
        after=last_sync
    )

    for event in new_events:
        await send_to_data_warehouse(event)

    await save_last_sync_timestamp(datetime.now())
```

### Disaster Recovery

Rebuild from events if something goes wrong:

```python
async def rebuild_namespace(namespace_id):
    # Clear current state
    await clear_namespace_data(namespace_id)

    # Get all events
    events = await storage.get_events(namespace_id)

    # Replay them in order
    for event in sorted(events, key=lambda e: e.timestamp):
        await replay_event(event)
```

### Analytics

Understand usage patterns:

```python
# What queries are people running?
query_events = await storage.get_events(
    namespace_id,
    event_types=["query.executed"],
    after=datetime.now() - timedelta(days=30)
)

# Most common query types
from collections import Counter
query_types = Counter(e.data.get("mode") for e in query_events)
print(query_types.most_common(5))

# Peak usage times
hours = Counter(e.timestamp.hour for e in query_events)
```

## Event Types Reference

### Document Events

| Type | When | Data |
|------|------|------|
| `document.created` | New document | title, source, checksum |
| `document.updated` | Metadata changed | changed fields |
| `document.processed` | Processing complete | chunk_count, entity_count |
| `document.failed` | Processing error | error message |
| `document.deleted` | Removed | - |

### Entity Events

| Type | When | Data |
|------|------|------|
| `entity.created` | New entity extracted | name, type, confidence |
| `entity.updated` | Attributes changed | changed fields |
| `entity.merged` | Unified with duplicate | merged_into, reason |
| `entity.deleted` | Removed | - |

### Relationship Events

| Type | When | Data |
|------|------|------|
| `relationship.created` | New relationship | source, target, type |
| `relationship.inferred` | Created by inference | rule, confidence |
| `relationship.deleted` | Removed | - |

### Chunk Events

| Type | When | Data |
|------|------|------|
| `chunk.created` | Document chunked | index, token_count |
| `chunk.embedded` | Embedding generated | model |
| `chunk.deleted` | Removed | - |

### Sync Events

| Type | When | Data |
|------|------|------|
| `sync.started` | Pipeline begins | source, config |
| `sync.completed` | Pipeline finishes | stats |
| `sync.failed` | Pipeline error | error |
| `sync.checkpoint` | Progress saved | checkpoint value |

## Best Practices

1. **Always set correlation_id** for related operations. Makes debugging much easier.

2. **Batch event writes** during ingestion. One DB transaction for 100 events beats 100 transactions.

3. **Include meaningful actor_id**. "user:alice@example.com" is better than "user".

4. **Keep data payloads small**. Store IDs and changed fields, not entire objects.

5. **Query with limits**. Event logs grow large. Always paginate.

6. **Use previous_data for updates**. It enables undo and makes diffs possible.

## Database Schema

Events live in PostgreSQL for reliability:

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

-- Fast queries by namespace and type
CREATE INDEX ON memory_events(namespace_id, event_type);

-- Fast queries by resource
CREATE INDEX ON memory_events(resource_type, resource_id);

-- Time-based queries
CREATE INDEX ON memory_events(timestamp DESC);

-- Correlation tracking
CREATE INDEX ON memory_events(correlation_id);
```

## What's Next?

- **[Storage Backends](storage-backends.md)** - How data is stored
- **[Multi-Tenancy](multi-tenancy.md)** - Namespace organization
- **[Temporal Queries](../query-engine/temporal-queries.md)** - Using timestamps in search

# Semantic Hooks

Semantic hooks let you subscribe to events during document ingestion and receive callbacks when extracted entities or relationships match your criteria. Use them to build real-time notifications, dashboards, or downstream processing pipelines.

## Quick Example

```python
from khora import MemoryLake
from khora.hooks import SemanticFilter

async with MemoryLake(db_url) as lake:
    # Simple: get notified on every new entity
    async def on_entity(event):
        print(f"New entity: {event.data.get('name')} ({event.data.get('entity_type')})")

    lake.subscribe("entity.created", on_entity)

    # Advanced: semantic filter for specific entity types
    filter = SemanticFilter(
        name="competitor_mentions",
        description="Any mention of a competitor company",
        entity_types=["ORGANIZATION"],
    )
    lake.subscribe("entity.created", on_entity, filter=filter)

    # Ingestion triggers callbacks
    await lake.remember("Acme Corp announced a partnership with Globex.", namespace=ns_id)
```

## 3-Level Filter Cascade

Hooks use a cost-efficient cascade — each level is only evaluated if the previous level passes:

| Level | Method | Cost | Purpose |
|-------|--------|------|---------|
| 0 | Type pre-filter | Free | Match by `entity_types` or `relationship_types` |
| 1 | Embedding similarity | Sub-millisecond | Binary-quantized cosine similarity gate |
| 2 | LLM evaluation | ~0.001 USD/call | Nano-model yes/no for ambiguous cases |

### Level 0: Type Filter

Filters by entity type or relationship type. No computation required:

```python
SemanticFilter(
    name="people_only",
    entity_types=["PERSON"],
)
```

### Level 1: Embedding Pre-Screen

Compares the filter's description embedding against the entity/relationship embedding using binary-quantized Hamming distance, then full cosine on survivors:

```python
SemanticFilter(
    name="ai_research",
    description="Research related to artificial intelligence and machine learning",
    similarity_threshold=0.7,  # cosine similarity cutoff
)
```

### Level 2: LLM Evaluation

For ambiguous cases, a nano-LLM (configurable) makes a yes/no decision:

```python
SemanticFilter(
    name="strategic_mentions",
    description="Any mention of strategic business decisions",
    entity_types=["EVENT", "CONCEPT"],
    # LLM evaluates when embedding similarity is borderline
)
```

## Event Types

Subscribe to any of these event types:

| Event | Fires When |
|-------|------------|
| `entity.created` | New entity extracted during ingestion |
| `entity.updated` | Existing entity updated (e.g., merged) |
| `relationship.created` | New relationship extracted |
| `chunk.embedded` | Chunk embedding generated |
| `document.created` | Document stored |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `KHORA_HOOKS_ENABLED` | Enable the hook system | `true` |
| `KHORA_HOOKS_FILTER_MODEL` | LLM for Level 2 evaluation | `gpt-4o-mini` |
| `KHORA_HOOKS_DEFAULT_SIMILARITY_THRESHOLD` | Level 1 cosine threshold | `0.7` |
| `KHORA_HOOKS_MAX_CONCURRENT` | Max concurrent callback executions | `10` |

Per-filter model override:

```python
SemanticFilter(
    name="custom",
    description="...",
    filter_model="claude-haiku-4-5-20251001",  # override global model
)
```

## API

### `lake.subscribe(event_type, callback, filter=None)`

Register a callback for an event type. Returns a subscription ID.

### `lake.unsubscribe(subscription_id)`

Remove a subscription.

### `lake.hooks`

Access the underlying `HookDispatcher` for advanced usage (e.g., `lake.hooks.clear()`).

## Related Documentation

- [Extraction Pipeline](../extraction/ingestion-pipeline.md) — where hooks fire during ingestion
- [Event Sourcing](../architecture/event-sourcing.md) — the event model hooks build on

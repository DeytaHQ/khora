# Multi-Tenancy

> **Status:** The `TenancyMode` enum exists in the codebase but is **not wired at runtime**. Currently only namespace-level row filtering is active.

Khora isolates data through **namespaces**. Every document, chunk, entity, and relationship belongs to exactly one namespace. This is the sole unit of isolation — there is no organization or workspace hierarchy within khora. Higher-level grouping (organizations, teams, etc.) is the responsibility of the consuming service.

## Namespaces: Where Data Lives

All your data is scoped to namespaces. When you call `remember()` or `recall()`, you must always specify a namespace:

```python
from khora import MemoryLake

async with MemoryLake() as lake:
    # Store in a specific namespace (required)
    await lake.remember(
        "Important document content...",
        namespace=namespace_id
    )

    # Search within that namespace (required)
    results = await lake.recall(
        "What's in those documents?",
        namespace=namespace_id
    )
```

The `namespace` parameter is **required** — there is no default namespace. Omitting it raises a `ValueError`.

This isolation is enforced at the database level:

```sql
-- Every query includes namespace filtering
SELECT * FROM chunks
WHERE namespace_id = '...'
  AND similarity > 0.3
ORDER BY similarity DESC;
```

You can't accidentally see another namespace's data.

## Creating Namespaces

```python
from khora import MemoryLake
from khora.core.models import MemoryNamespace

async with MemoryLake() as lake:
    # Create a namespace (UUID-identified, no name/description)
    namespace = await lake.storage.create_namespace(
        MemoryNamespace()
    )

    # Now store data
    await lake.remember(
        "Important content...",
        namespace=namespace.id
    )
```

## Finding Namespaces

```python
# By UUID
ns = await lake.storage.get_namespace(namespace_id)

# List all (active only by default)
result = await lake.storage.list_namespaces()
namespaces = result.items
```

## Namespace Versioning

Namespaces can be versioned. This solves a common problem - how do you replace all your data without downtime?

### The Problem

Say you have a "production" namespace with 100,000 documents. You want to rebuild everything from scratch (maybe your extraction logic improved). You can't just delete and re-import - your users would see empty results during the rebuild.

### The Solution: Version and Swap

```
Before:
  production (v1) <- active, serving queries

During rebuild:
  production (v1) <- still active, serving queries
  production (v2) <- inactive, being populated

After:
  production (v1) <- now inactive
  production (v2) <- now active, serving queries
```

The swap is atomic. One moment users see v1, the next they see v2. No downtime, no partial data.

### How to Do It

```python
async with MemoryLake() as lake:
    # 1. Get current namespace by UUID
    current = await lake.storage.get_namespace(current_namespace_id)

    # 2. Create new version
    new_version = await lake.storage.create_namespace_version(
        previous_version=current
    )
    # new_version.version = 2
    # new_version.is_active = False
    # new_version.previous_version_id = current.id

    # 3. Populate the new version
    for doc in all_your_documents:
        await lake.remember(
            doc.content,
            namespace=new_version.id
        )

    # 4. Verify everything looks good
    test_results = await lake.recall("test query", namespace=new_version.id)
    assert len(test_results.chunks) > 0

    # 5. Swap! Activate new, deactivate old
    new_version.is_active = True
    await lake.storage.update_namespace(new_version)
    await lake.storage.deactivate_namespace(current.id)
```

### Rollback

Made a mistake? Swap back:

```python
# Reactivate the old version
old_version.is_active = True
await lake.storage.update_namespace(old_version)
await lake.storage.deactivate_namespace(new_version.id)
```

Keep old versions around until you're confident the new one is working.

## Configuration Overrides

Each namespace can override global settings. Maybe one namespace needs a different embedding model, or stricter similarity thresholds:

```python
namespace = MemoryNamespace(
    config_overrides={
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 3072,
        "min_chunk_similarity": 0.5,  # Higher threshold
        "extraction_model": "claude-sonnet-4-20250514"
    }
)
```

Configuration resolves:
1. Namespace overrides (highest priority)
2. Global config (lowest priority)

## Sync Checkpoints

Namespaces track where they are in syncing from external sources:

```python
namespace.sync_checkpoints = {
    "slack": "1706140800",           # Unix timestamp
    "linear": "2024-01-25T00:00:00Z" # ISO 8601
}
```

This enables incremental sync:

```python
# Get last checkpoint
checkpoint = await lake.storage.get_sync_checkpoint(namespace_id, "slack")

# Fetch only new messages
new_messages = await slack_client.get_messages(since=checkpoint)

# Process them...
for msg in new_messages:
    await lake.remember(msg.content, namespace=namespace_id)

# Update checkpoint
await lake.storage.set_sync_checkpoint(
    namespace_id,
    "slack",
    str(new_messages[-1].timestamp)
)
```

## Cross-Namespace Queries

Need to search multiple namespaces? (Note: this bypasses isolation - make sure you have permission)

```python
all_results = []
result = await lake.storage.list_namespaces()
for namespace in result.items:
    if namespace.is_active and user_can_access(namespace):
        results = await lake.recall(query, namespace=namespace.id)
        all_results.extend(results.chunks)

# Re-rank the combined results
all_results.sort(key=lambda x: x[1], reverse=True)
```

## What's Next?

- **[Event Sourcing](event-sourcing.md)** - The audit trail of all changes
- **[Storage Backends](storage-backends.md)** - How data is actually stored
- **[Overview](overview.md)** - High-level architecture

# Multi-Tenancy

> **Status:** The `TenancyMode` enum exists in the codebase but is **not wired at runtime**. Currently only namespace-level row filtering is active.

Khora isolates data through **namespaces**. Every document, chunk, entity, and relationship belongs to exactly one namespace. This is the sole unit of isolation — there is no organization or workspace hierarchy within khora. Higher-level grouping (organizations, teams, etc.) is the responsibility of the consuming service.

## Namespaces: Where Data Lives

All your data is scoped to namespaces. When you call `remember()` or `recall()`, you must always specify a namespace:

```python
from khora import Khora

async with Khora() as kb:
    # Store in a specific namespace (required)
    await kb.remember(
        "Important document content...",
        namespace=namespace_id
    )

    # Search within that namespace (required)
    results = await kb.recall(
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

## Protocol-level isolation contract (v0.16.0)

Namespace scoping is part of the storage Protocol contract itself. Every read, exists-check, AND mutation method on every backend (`RelationalBackend`, `VectorBackend`, `GraphBackend`, `EventStore`) declares `*, namespace_id: UUID` as a required keyword-only parameter and filters at the **query layer** (SQL `WHERE`, Cypher `MATCH (... {namespace_id: $ns})`, SurrealQL filter) — the namespace check happens in the database, never as a Python comparison after the row has already been fetched.

Looking up a row whose id belongs to a different namespace returns `None` / `False` / an empty list / dict straight from the query. The caller's `namespace_id` is the authority; the row's stored `namespace_id` only matters as the value the filter matches against.

```python
# Pattern (v0.16.0+):
doc = await coordinator.get_document(document_id, namespace_id=caller_ns)
# `doc` is None if the id does not exist OR belongs to a different namespace.

entity = await coordinator.get_entity(entity_id, namespace_id=caller_ns)
neighbourhood = await coordinator.get_neighborhood(
    entity_id, namespace_id=caller_ns, depth=2,
)
# Traversal does NOT cross into other namespaces; if the seed entity is
# outside `caller_ns`, the result is the empty neighbourhood.

ok = await coordinator.delete_document(document_id, namespace_id=caller_ns)
# `ok` is False when the id is in a different namespace — delete is a no-op.
```

The surface that received this treatment in PRs #761 / #765 / #766 / #769 covers:

- **Reads** (16 methods): `get_document(_s_batch)`, `get_document_sources_batch`, `get_document_projections_batch`, `get_document_by_external_id`, `get_documents_by_external_ids`, `entity_exists`, vector `get_entity` / `get_entities_batch`, graph `get_entity(_ies_batch)`, `get_relationship`, `get_episode`, `get_entity_relationships`, `get_neighborhood(_s_batch)`, `find_paths`, `get_temporal_neighbors`, event store `get_events_for_resource`, `get_latest_event`.
- **Writes** (12+ methods): `delete_document`, `delete_chunks_by_document`, `update_entity`, `update_entity_embedding(_s_batch)`, `delete_entities_batch`, `delete_relationships_batch`, `supersede_fact`, `delete_entity`, `delete_relationship`, Neo4j-specific `retire_orphaned_relationships_batch` / `remap_source_document_ids_batch`.

### Structural signature gate

The invariant is enforced statically. `tests/security/test_cross_namespace_idor_signatures.py` walks every concrete backend class at test-collection time, enumerates every method matching the read/write naming patterns (`get_*`, `entity_exists`, `find_paths`, `get_neighborhood*`, `delete_*`, `update_entity*`, `supersede_*`) with a required id-typed parameter, and asserts:

1. The signature includes `namespace_id`.
2. `namespace_id` is keyword-only (kind `KEYWORD_ONLY` or `VAR_KEYWORD`).

A new backend method without `namespace_id=` fails CI at test collection — the regression cannot land without a security review. Genuine namespace-resolver helpers (e.g. `get_namespace_by_name`) are explicitly allow-listed in `_EXEMPT_METHODS` with a one-line justification.

### Coordinator facade is the supported API

`StorageCoordinator.{relational,vector,graph,event_store}` is now wrapped in a `NamespaceRequiredProxy` (see `src/khora/storage/_namespace_proxy.py`). Direct backend access through these public attributes:

1. Emits a one-shot `DeprecationWarning` per role per process.
2. Refuses the namespace-scoped read methods unless `namespace_id=` is passed (raises `TypeError`).
3. Does not forward access to underscore-prefixed attributes — backend internals such as `_engine`, `_handle`, `_conn`, `_session_factory` are reachable only via the private `coord._{role}` accessors used by coordinator internals.

The public attributes are scheduled for removal in v0.17 — call the coordinator's facade methods (`coordinator.get_document(...)`, `coordinator.get_entity(...)`, `coordinator.get_neighborhood(...)`) instead.

## Creating Namespaces

```python
from khora import Khora

async with Khora() as kb:
    # Create a namespace via the public API
    namespace = await kb.create_namespace()

    # Now store data using the stable namespace_id
    await kb.remember(
        "Important content...",
        namespace=namespace.namespace_id
    )
```

## Namespace Dual-ID Scheme

Each namespace has two IDs:

| ID | Purpose | Changes? |
|----|---------|----------|
| **`namespace_id`** | Stable identifier across all versions | Never — use this in your application |
| **`id`** | Row-level UUID | Changes per version (v1, v2, etc.) |

Public API methods accept `namespace_id` and resolve to the active version's `id` automatically. Resolution via `resolve_namespace()` is idempotent — it accepts either ID type. This adds one indexed lookup per API call (sub-ms).

Child table foreign keys (documents, chunks, entities) reference `id`, not `namespace_id`.

## Finding Namespaces

```python
# By stable namespace_id (recommended)
ns = await kb.get_namespace_by_stable_id(stable_namespace_id)

# By row-level UUID
ns = await kb.storage.get_namespace(namespace_id)

# List all (active only by default)
result = await kb.storage.list_namespaces()
namespaces = result.items
```

## Namespace Versioning

Namespaces can be versioned. This solves a common problem - how do you replace all your data without downtime?

### The Problem

Say you have a "production" namespace with 100,000 documents. You want to rebuild everything from scratch (maybe your extraction logic improved). You can't just delete and re-import - your users would see empty results during the rebuild.

### The Solution: Version and Swap

```text
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
async with Khora() as kb:
    # 1. Get current namespace by UUID
    current = await kb.storage.get_namespace(current_namespace_id)

    # 2. Create new version
    new_version = await kb.storage.create_namespace_version(
        previous_version=current
    )
    # new_version.version = 2
    # new_version.is_active = False

    # 3. Populate the new version
    for doc in all_your_documents:
        await kb.remember(
            doc.content,
            namespace=new_version.id
        )

    # 4. Verify everything looks good
    test_results = await kb.recall("test query", namespace=new_version.id)
    assert len(test_results.chunks) > 0

    # 5. Swap! Activate new, deactivate old
    new_version.is_active = True
    await kb.storage.update_namespace(new_version)
    await kb.storage.deactivate_namespace(current.id)
```

### Rollback

Made a mistake? Swap back:

```python
# Reactivate the old version
old_version.is_active = True
await kb.storage.update_namespace(old_version)
await kb.storage.deactivate_namespace(new_version.id)
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
checkpoint = await kb.storage.get_sync_checkpoint(namespace_id, "slack")

# Fetch only new messages
new_messages = await slack_client.get_messages(since=checkpoint)

# Process them...
for msg in new_messages:
    await kb.remember(msg.content, namespace=namespace_id)

# Update checkpoint
await kb.storage.set_sync_checkpoint(
    namespace_id,
    "slack",
    str(new_messages[-1].timestamp)
)
```

## Cross-Namespace Queries

Need to search multiple namespaces? (Note: this bypasses isolation - make sure you have permission)

```python
all_results = []
result = await kb.storage.list_namespaces()
for namespace in result.items:
    if namespace.is_active and user_can_access(namespace):
        results = await kb.recall(query, namespace=namespace.id)
        all_results.extend(results.chunks)

# Re-rank the combined results
all_results.sort(key=lambda x: x[1], reverse=True)
```

## What's Next?

- **[Event Sourcing](event-sourcing.md)** - The audit trail of all changes
- **[Storage Backends](storage-backends.md)** - How data is actually stored
- **[Overview](overview.md)** - High-level architecture

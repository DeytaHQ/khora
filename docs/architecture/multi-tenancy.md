# Multi-Tenancy

> **Status (v0.2.3):** The `TenancyMode` enum and ACL enforcement code described below exist in the codebase but are **not wired at runtime**. Currently only namespace-level row filtering is active. See `docs/design/namespace-optimization-plan.md` for the implementation roadmap.

Different teams need different data. Different projects shouldn't mix. Sometimes you need complete isolation for compliance. Khora's multi-tenancy model handles all of this through a simple hierarchy: Organizations contain Workspaces contain Namespaces.

## The Hierarchy

```
Acme Corporation (Organization)
│
├── Engineering (Workspace)
│   ├── production (Namespace)    ← Live data
│   ├── production-v2 (Namespace) ← New version being staged
│   └── sandbox (Namespace)       ← Experiments
│
└── Product (Workspace)
    ├── research (Namespace)
    └── competitor-analysis (Namespace)
```

Each level serves a purpose:

**Organization** - Your company. Handles billing, sets global policies, defines whether data is shared or isolated at the infrastructure level.

**Workspace** - A team or project. Groups related namespaces, provides a boundary for access control.

**Namespace** - Where data actually lives. Every document, chunk, entity, and relationship belongs to exactly one namespace. This is your unit of isolation.

## Namespaces: Where Data Lives

All your data is scoped to namespaces. When you call `remember()` or `recall()`, you're always operating within a namespace:

```python
from khora import MemoryLake

async with MemoryLake() as lake:
    # Store in a specific namespace
    await lake.remember(
        "Important document content...",
        namespace=namespace_id
    )

    # Search within that namespace
    results = await lake.recall(
        "What's in those documents?",
        namespace=namespace_id
    )
```

This isolation is enforced at the database level:

```sql
-- Every query includes namespace filtering
SELECT * FROM chunks
WHERE namespace_id = '...'
  AND similarity > 0.3
ORDER BY similarity DESC;
```

You can't accidentally see another namespace's data.

## Namespace Versioning

Here's a powerful feature: namespaces can be versioned. This solves a common problem - how do you replace all your data without downtime?

### The Problem

Say you have a "production" namespace with 100,000 documents. You want to rebuild everything from scratch (maybe your extraction logic improved). You can't just delete and re-import - your users would see empty results during the rebuild.

### The Solution: Version and Swap

```
Before:
  production (v1) ← active, serving queries

During rebuild:
  production (v1) ← still active, serving queries
  production (v2) ← inactive, being populated

After:
  production (v1) ← now inactive
  production (v2) ← now active, serving queries
```

The swap is atomic. One moment users see v1, the next they see v2. No downtime, no partial data.

### How to Do It

```python
async with MemoryLake() as lake:
    # 1. Get current namespace
    current = await lake.storage.get_namespace_by_slug(
        workspace_id, "production"
    )

    # 2. Create new version
    new_version = await lake.storage.create_namespace_version(
        workspace_id,
        slug="production",
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

## Tenancy Modes

### Shared Mode (Default)

All organizations share the same database infrastructure. Isolation is achieved through row-level filtering on `namespace_id`.

```python
Organization(
    name="Acme Corp",
    tenancy_mode=TenancyMode.SHARED
)
```

**Pros:**
- Lower cost (shared infrastructure)
- Simpler operations
- Good enough for most use cases

**Cons:**
- Data is logically separate but physically together
- May not satisfy strict compliance requirements

### Isolated Mode

Each organization gets its own database instances. Complete physical separation.

```python
Organization(
    name="SecureCorp",
    tenancy_mode=TenancyMode.ISOLATED
)
```

**Pros:**
- Complete data isolation
- Meets HIPAA, SOC2, and similar requirements
- Independent scaling and maintenance

**Cons:**
- Higher cost (dedicated infrastructure)
- More operational complexity

**Configuration:**
```python
# Each isolated org needs its own connection strings
org_config = {
    "postgresql_url": "postgresql://securecorp:pass@securecorp-db:5432/khora",
    "neo4j_url": "bolt://securecorp-graph:7687"
}
```

## Configuration Overrides

Each namespace can override global settings. Maybe one namespace needs a different embedding model, or stricter similarity thresholds:

```python
namespace = MemoryNamespace(
    workspace_id=workspace_id,
    name="High-Precision Research",
    slug="research",
    config_overrides={
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 3072,
        "min_chunk_similarity": 0.5,  # Higher threshold
        "extraction_model": "claude-sonnet-4-20250514"
    }
)
```

Configuration resolves top-down:
1. Namespace overrides (highest priority)
2. Workspace defaults
3. Organization defaults
4. Global config (lowest priority)

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

## Access Control

Permissions flow down the hierarchy:

```
Organization (admin role)
    │
    └── Workspace (member role)
            │
            └── Namespace (read/write permissions)
```

If you can access an organization, you can access its workspaces (subject to workspace permissions). If you can access a workspace, you can access its namespaces (subject to namespace permissions).

```python
from khora.acl import ACLEnforcer, ACLContext

enforcer = ACLEnforcer(storage)

# Check before accessing
context = ACLContext(
    user_id="user-123",
    namespace_id=namespace_id,
    operation="read"
)

if await enforcer.check_permission(context):
    results = await lake.recall(query, namespace=namespace_id)
else:
    raise PermissionDenied()
```

## Setting Up the Hierarchy

### Create Everything

```python
from khora import MemoryLake
from khora.core.models import Organization, Workspace, MemoryNamespace

async with MemoryLake() as lake:
    # Organization
    org = await lake.storage.create_organization(
        Organization(name="Acme Corp", slug="acme")
    )

    # Workspace
    workspace = await lake.storage.create_workspace(
        Workspace(
            organization_id=org.id,
            name="Engineering",
            slug="engineering"
        )
    )

    # Namespace
    namespace = await lake.storage.create_namespace(
        MemoryNamespace(
            workspace_id=workspace.id,
            name="Production",
            slug="production"
        )
    )

    # Now store data
    await lake.remember(
        "Important content...",
        namespace=namespace.id
    )
```

### Find Existing Namespaces

```python
# By slug
ns = await lake.storage.get_namespace_by_slug(workspace_id, "production")

# List all in a workspace
namespaces = await lake.storage.list_namespaces(workspace_id)

# Filter to active only
active = [ns for ns in namespaces if ns.is_active]
```

### Cross-Namespace Queries

Need to search multiple namespaces? (Note: this bypasses isolation - make sure you have permission)

```python
all_results = []
for namespace in await lake.storage.list_namespaces(workspace_id):
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

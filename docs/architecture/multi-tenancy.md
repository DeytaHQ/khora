# Multi-Tenancy

Khora provides a hierarchical multi-tenancy model that supports both shared infrastructure and complete tenant isolation. This document describes the tenancy hierarchy, isolation modes, and versioning capabilities.

## Tenancy Hierarchy

```
Organization
    │
    ├── Workspace 1
    │       │
    │       ├── Namespace A (v1) ← active
    │       ├── Namespace A (v2) ← inactive (previous version)
    │       └── Namespace B
    │
    └── Workspace 2
            │
            └── Namespace C
```

### Organization

The top-level tenant container representing a company or billing entity.

```python
@dataclass
class Organization:
    id: UUID
    name: str              # "Acme Corporation"
    slug: str              # "acme" (URL-friendly)
    tenancy_mode: TenancyMode  # SHARED or ISOLATED
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**Use Cases:**
- Billing and subscription management
- Cross-workspace analytics
- Organization-wide settings

### Workspace

A logical grouping within an organization, typically representing a team or project.

```python
@dataclass
class Workspace:
    id: UUID
    organization_id: UUID  # Parent organization
    name: str              # "Engineering Team"
    slug: str              # "engineering"
    description: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**Use Cases:**
- Team-level isolation
- Project boundaries
- Access control groups

### Namespace (MemoryNamespace)

The primary unit of memory isolation. All documents, chunks, entities, and relationships are scoped to a namespace.

```python
@dataclass
class MemoryNamespace:
    id: UUID
    workspace_id: UUID     # Parent workspace
    name: str              # "Production Data"
    slug: str              # "production"
    description: str

    # Versioning
    version: int = 1
    is_active: bool = True
    previous_version_id: UUID | None = None

    # Configuration
    config_overrides: dict[str, Any] = {}
    sync_checkpoints: dict[str, str] = {}

    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

**Use Cases:**
- Data isolation per use case
- Environment separation (dev/staging/prod)
- Data versioning for replacement workflows

## Tenancy Modes

### Shared Mode (Default)

All tenants share the same database with row-level filtering by `namespace_id`.

```python
class TenancyMode(str, Enum):
    SHARED = "shared"  # Row-level security with namespace_id filtering
```

**Characteristics:**
- Single database instance
- All queries filtered by `namespace_id`
- Lower infrastructure cost
- Suitable for most use cases

**Query Pattern:**
```sql
-- All queries include namespace filter
SELECT * FROM documents
WHERE namespace_id = $1
  AND status = 'completed';
```

### Isolated Mode

Each tenant gets dedicated database instances.

```python
class TenancyMode(str, Enum):
    ISOLATED = "isolated"  # Separate database instances per tenant
```

**Characteristics:**
- Separate PostgreSQL/Neo4j instances per organization
- Complete data isolation
- Higher infrastructure cost
- Required for regulated industries (HIPAA, SOC2)

**Configuration:**
```python
# Isolated mode requires per-org connection strings
org_config = {
    "postgresql_url": "postgresql://acme:pass@acme-db:5432/khora",
    "neo4j_url": "bolt://acme-graph:7687",
}
```

## Namespace Versioning

Namespaces support versioning for data replacement workflows, enabling:
- Full data replacement without downtime
- Rollback to previous versions
- A/B testing with different data sets

### Version Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Version Workflow                              │
│                                                                  │
│   1. Create new version                                         │
│      ┌─────────────┐                                            │
│      │ Namespace   │ version=1, is_active=True                  │
│      │ (v1)        │                                            │
│      └─────────────┘                                            │
│             │                                                    │
│             ▼                                                    │
│   2. Ingest new data into v2                                    │
│      ┌─────────────┐     ┌─────────────┐                        │
│      │ Namespace   │     │ Namespace   │ version=2              │
│      │ (v1)        │     │ (v2)        │ is_active=False        │
│      │ active      │     │ staging     │ previous=v1.id         │
│      └─────────────┘     └─────────────┘                        │
│                                 │                                │
│                                 ▼                                │
│   3. Activate v2, deactivate v1                                 │
│      ┌─────────────┐     ┌─────────────┐                        │
│      │ Namespace   │     │ Namespace   │ version=2              │
│      │ (v1)        │     │ (v2)        │ is_active=True         │
│      │ inactive    │     │ active      │                        │
│      └─────────────┘     └─────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### API Usage

```python
from khora import MemoryLake

async with MemoryLake() as lake:
    # Get current active namespace
    current_ns = await lake.storage.get_namespace_by_slug(
        workspace_id, "production"
    )

    # Create new version
    new_ns = await lake.storage.create_namespace_version(
        workspace_id,
        "production",
        previous_version=current_ns,
    )

    # Ingest new data into v2
    for doc in new_documents:
        await lake.remember(doc.content, namespace=new_ns.id)

    # Verify new data is correct...

    # Activate new version (deactivates previous)
    new_ns.is_active = True
    await lake.storage.update_namespace(new_ns)
    await lake.storage.deactivate_namespace(current_ns.id)
```

## Configuration Overrides

Each namespace can override global configuration:

```python
namespace = MemoryNamespace(
    workspace_id=workspace.id,
    name="Custom Config",
    slug="custom",
    config_overrides={
        # Override embedding model
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 3072,

        # Override extraction settings
        "extraction_model": "claude-sonnet-4-20250514",

        # Override query settings
        "min_chunk_similarity": 0.4,
    },
)
```

**Resolution Order:**
1. Namespace `config_overrides` (highest priority)
2. Workspace-level settings
3. Organization-level settings
4. Global `KhoraConfig` (lowest priority)

## Sync Checkpoints

Namespaces track incremental sync state per source:

```python
@dataclass
class MemoryNamespace:
    # ...
    sync_checkpoints: dict[str, str] = field(default_factory=dict)
    # Example: {"slack": "1706140800", "linear": "2024-01-25T00:00:00Z"}
```

**Usage:**
```python
# Get last sync checkpoint
checkpoint = await lake.storage.get_sync_checkpoint(
    namespace_id, "slack"
)

# Fetch only new messages since checkpoint
new_messages = await slack_client.get_messages(since=checkpoint)

# After sync, update checkpoint
await lake.storage.set_sync_checkpoint(
    namespace_id, "slack", str(latest_timestamp)
)
```

## Access Control

Khora provides ACL enforcement at the namespace level:

```python
from khora.acl import ACLEnforcer, ACLContext

enforcer = ACLEnforcer(storage)

# Check permission before operation
context = ACLContext(
    user_id="user-123",
    namespace_id=namespace_id,
    operation="read",
)

if await enforcer.check_permission(context):
    results = await lake.recall(query, namespace=namespace_id)
```

**Permission Inheritance:**
```
Organization permissions
        │
        ▼
    Workspace permissions (inherit from org)
        │
        ▼
    Namespace permissions (inherit from workspace)
```

## API Examples

### Creating Tenancy Hierarchy

```python
from khora import MemoryLake
from khora.core.models import Organization, Workspace, MemoryNamespace

async with MemoryLake() as lake:
    # Create organization
    org = await lake.storage.create_organization(
        Organization(name="Acme Corp", slug="acme")
    )

    # Create workspace
    workspace = await lake.storage.create_workspace(
        Workspace(
            organization_id=org.id,
            name="Engineering",
            slug="engineering",
        )
    )

    # Create namespace
    namespace = await lake.storage.create_namespace(
        MemoryNamespace(
            workspace_id=workspace.id,
            name="Production",
            slug="production",
        )
    )

    # Store memories in namespace
    await lake.remember(
        "Important engineering document...",
        namespace=namespace.id,
    )
```

### Listing Namespaces

```python
# List all namespaces in a workspace
namespaces = await lake.storage.list_namespaces(workspace.id)

# Get specific namespace by slug
ns = await lake.storage.get_namespace_by_slug(
    workspace.id, "production"
)
```

### Cross-Namespace Queries

By default, queries are scoped to a single namespace. For cross-namespace queries, iterate over namespaces:

```python
all_results = []
for namespace in await lake.storage.list_namespaces(workspace.id):
    if namespace.is_active:
        results = await lake.recall(query, namespace=namespace.id)
        all_results.extend(results.chunks)

# Deduplicate and re-rank as needed
```

## Next Steps

- [Event Sourcing](event-sourcing.md) - Audit trails and temporal queries
- [Architecture Overview](overview.md) - System design

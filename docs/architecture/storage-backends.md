# Storage Backends

Khora uses three complementary storage systems, each optimized for specific access patterns. This document details the purpose, configuration, and implementation of each backend.

## Overview

| Backend | Technology | Purpose | Data Stored |
|---------|------------|---------|-------------|
| Relational | PostgreSQL | Structured data, ACID transactions | Documents, tenancy, permissions, metadata |
| Vector | pgvector | Similarity search | Chunk embeddings, entity embeddings |
| Graph | Neo4j | Relationship traversal | Entity nodes, relationship edges |
| Event Store | PostgreSQL | Audit trail, temporal queries | Immutable events |

## PostgreSQL (Relational Backend)

The relational backend handles all structured data with full ACID guarantees.

### Data Stored

- **Documents** - Source content with metadata, status, and processing info
- **Organizations** - Top-level tenant containers
- **Workspaces** - Project/team isolation within organizations
- **Namespaces** - Memory isolation with versioning support
- **Sync Checkpoints** - Incremental sync state per source

### Schema Highlights

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL REFERENCES namespaces(id),
    content TEXT,
    status VARCHAR(20),  -- pending, processing, completed, failed
    metadata JSONB,
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ
);

-- Tenancy hierarchy
CREATE TABLE organizations (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(100) UNIQUE,
    tenancy_mode VARCHAR(20),  -- shared, isolated
    metadata JSONB
);

CREATE TABLE workspaces (
    id UUID PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id),
    name VARCHAR(255),
    slug VARCHAR(100)
);

CREATE TABLE namespaces (
    id UUID PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id),
    name VARCHAR(255),
    slug VARCHAR(100),
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    previous_version_id UUID,
    config_overrides JSONB
);
```

### Connection Configuration

```python
from khora.storage import StorageConfig

config = StorageConfig(
    postgresql_url="postgresql://user:pass@localhost:5432/khora",
    # Connection pool settings
    pool_size=5,
    max_overflow=10,
)
```

### Key Operations

- `create_document()` / `update_document()` / `delete_document()`
- `create_organization()` / `create_workspace()` / `create_namespace()`
- `get_document_by_checksum()` - Deduplication check
- `create_namespace_version()` - Version management
- `get_sync_checkpoint()` / `set_sync_checkpoint()` - Incremental sync

## pgvector (Vector Backend)

The vector backend enables semantic similarity search using PostgreSQL's pgvector extension.

### Data Stored

- **Chunk Embeddings** - Vector representations of document chunks
- **Entity Embeddings** - Vector representations of extracted entities

### Index Configuration

Khora uses IVFFlat indexing for approximate nearest neighbor search:

```sql
-- Chunk embeddings with IVFFlat index
CREATE TABLE chunk_embeddings (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT,
    embedding vector(1536),  -- Dimension matches embedding model
    metadata JSONB,
    created_at TIMESTAMPTZ
);

CREATE INDEX idx_chunk_embeddings_vector
ON chunk_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Entity embeddings
CREATE TABLE entity_embeddings (
    entity_id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    embedding vector(1536),
    model VARCHAR(100)
);
```

### Similarity Search

```python
# Internal implementation
async def search_similar_chunks(
    namespace_id: UUID,
    query_embedding: list[float],
    limit: int = 10,
    min_similarity: float = 0.3,
) -> list[tuple[Chunk, float]]:
    """
    Uses pgvector's <=> operator for cosine distance.
    Score = 1 - distance (so higher = more similar)
    """
    query = """
        SELECT *, 1 - (embedding <=> $1) as similarity
        FROM chunk_embeddings
        WHERE namespace_id = $2
          AND 1 - (embedding <=> $1) >= $3
        ORDER BY embedding <=> $1
        LIMIT $4
    """
```

### Embedding Models

Khora uses LiteLLM for embedding generation, defaulting to:
- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimension**: 1536

Configure in code:
```python
from khora.config import LiteLLMConfig

config = LiteLLMConfig(
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
)
```

## Neo4j (Graph Backend)

The graph backend stores entities and relationships for traversal queries.

### Data Stored

- **Entity Nodes** - People, organizations, concepts, etc.
- **Relationship Edges** - Typed connections between entities
- **Episode Nodes** - Temporal events with associated entities

### Node Labels

```cypher
// Entity node structure
(:Entity {
    id: "uuid",
    namespace_id: "uuid",
    name: "Einstein",
    entity_type: "PERSON",
    description: "Theoretical physicist",
    attributes: {role: "Professor"},
    mention_count: 5,
    confidence: 0.95,
    valid_from: datetime,
    valid_until: datetime
})

// Episode node structure
(:Episode {
    id: "uuid",
    namespace_id: "uuid",
    name: "Nobel Prize Award",
    description: "Awarded Nobel Prize in Physics",
    occurred_at: datetime,
    duration_seconds: 3600
})
```

### Relationship Types

Standard relationship types defined in `EntityType` enum:

| Type | Description |
|------|-------------|
| `WORKS_FOR` | Employment relationship |
| `KNOWS` | Personal connection |
| `MANAGES` / `REPORTS_TO` | Organizational hierarchy |
| `COLLABORATES_WITH` | Professional collaboration |
| `OWNS` / `PART_OF` | Ownership and composition |
| `LOCATED_IN` | Geographic association |
| `RELATES_TO` | Generic relationship |
| `DEPENDS_ON` / `IMPLEMENTS` | Technical dependencies |
| `PRECEDES` / `FOLLOWS` | Temporal ordering |

### Graph Queries

```cypher
// Find entity neighborhood
MATCH (e:Entity {id: $entity_id})-[r*1..2]-(related:Entity)
WHERE related.namespace_id = $namespace_id
RETURN DISTINCT related, r
LIMIT $limit

// Path finding between entities
MATCH path = shortestPath(
    (source:Entity {id: $source_id})-[*..3]-(target:Entity {id: $target_id})
)
RETURN path
```

### Connection Configuration

```python
from khora.storage import StorageConfig

config = StorageConfig(
    neo4j_url="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    neo4j_database="neo4j",  # Optional, uses default
)

# Or via environment variables:
# KHORA_NEO4J_URL=bolt://localhost:7687
# KHORA_NEO4J_USER=neo4j
# KHORA_NEO4J_PASSWORD=password
```

## Event Store

The event store provides an append-only log for all changes, enabling event sourcing patterns.

See [Event Sourcing](event-sourcing.md) for detailed documentation.

## Protocol-Based Design

All backends implement protocols defined in `src/khora/storage/backends/base.py`:

```python
from typing import Protocol

class VectorBackendProtocol(Protocol):
    """Protocol for vector storage backends."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_healthy(self) -> bool: ...

    async def create_chunk(self, chunk: Chunk) -> Chunk: ...
    async def create_chunks_batch(self, chunks: list[Chunk]) -> list[Chunk]: ...
    async def get_chunk(self, chunk_id: UUID) -> Chunk | None: ...
    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[Chunk, float]]: ...


class GraphBackendProtocol(Protocol):
    """Protocol for graph storage backends."""

    async def create_entity(self, entity: Entity) -> Entity: ...
    async def get_entity(self, entity_id: UUID) -> Entity | None: ...
    async def get_entity_by_name(
        self, namespace_id: UUID, name: str, entity_type: str
    ) -> Entity | None: ...
    async def get_neighborhood(
        self,
        entity_id: UUID,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]: ...
```

This design enables:
- **Testing**: Mock implementations for unit tests
- **Swappability**: Replace backends without changing core logic
- **Extensibility**: Add new backend types (e.g., Qdrant, Milvus)

## Backend Initialization

The `StorageCoordinator` is created via factory function:

```python
from khora.storage import StorageConfig, create_storage_coordinator

config = StorageConfig(
    postgresql_url="postgresql://...",
    pgvector_url="postgresql://...",  # Usually same as postgresql_url
    neo4j_url="bolt://...",
)

coordinator = create_storage_coordinator(config)
await coordinator.connect()

# Use coordinator...

await coordinator.disconnect()
```

## Health Checking

Each backend provides health checking:

```python
health = await coordinator.health_check()
# Returns StorageHealth with:
# - relational: bool
# - vector: bool
# - graph: bool
# - event_store: bool
# - is_healthy: bool (True if relational + vector are healthy)
```

## Next Steps

- [Multi-Tenancy](multi-tenancy.md) - Namespace isolation and versioning
- [Event Sourcing](event-sourcing.md) - Immutable event log

# Storage Backends

Khora doesn't use one database - it uses three, each chosen for what it does best. This might seem like overkill, but each backend excels at queries the others struggle with.

## The Three Musketeers

```
+--------------------+    +--------------------+    +--------------------+
|    PostgreSQL      |    |      pgvector      |    |       Neo4j        |
|                    |    |                    |    |                    |
|  The Record Keeper |    |  The Meaning       |    |  The Connector     |
|                    |    |  Finder            |    |                    |
|  Documents         |    |  "What's similar   |    |  "Who knows whom?" |
|  Who owns what     |    |  to this?"         |    |  "What's related   |
|  What happened     |    |                    |    |  to what?"         |
|  when              |    |  Embedding         |    |                    |
|                    |    |  similarity        |    |  Entity nodes      |
|  ACID guarantees   |    |  search            |    |  Relationship      |
|  Transactions      |    |                    |    |  edges             |
+--------------------+    +--------------------+    +--------------------+
```

**PostgreSQL** is your source of truth. It stores documents, tracks ownership through the tenant hierarchy, and maintains an immutable event log. When you need to know "what exactly is stored" or "who has access to what", PostgreSQL answers.

**pgvector** enables semantic search. It stores embeddings - those 1536-dimensional vectors that capture meaning - and finds similar content using cosine similarity. When you ask "what do we know about machine learning?", pgvector finds conceptually related content even if it uses different words.

**Neo4j** captures relationships. It stores entities (people, organizations, concepts) as nodes and their relationships as edges. When you ask "who works with Alice?" or "what projects is this technology used in?", Neo4j traverses the graph to find answers.

## PostgreSQL: The Foundation

Everything starts and ends here. PostgreSQL handles all structured data with full ACID guarantees.

### What It Stores

**Documents** - Your actual content:
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    content TEXT,
    status VARCHAR(20),  -- pending → processing → completed/failed
    metadata JSONB,      -- title, source, checksum, etc.
    chunk_count INTEGER,
    entity_count INTEGER,
    created_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ
);
```

**Tenant Hierarchy** - Who owns what:
```sql
-- Organization (top level, e.g., your company)
-- └── Workspace (team or project)
--     └── Namespace (isolated data container)

CREATE TABLE organizations (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    slug VARCHAR(100) UNIQUE,
    tenancy_mode VARCHAR(20)  -- shared or isolated
);

CREATE TABLE workspaces (
    id UUID PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id),
    name VARCHAR(255)
);

CREATE TABLE namespaces (
    id UUID PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id),
    name VARCHAR(255),
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);
```

**Events** - Everything that happens:
```sql
CREATE TABLE memory_events (
    id UUID PRIMARY KEY,
    namespace_id UUID,
    event_type VARCHAR(100),    -- document.created, entity.merged, etc.
    resource_type VARCHAR(50),  -- document, chunk, entity
    resource_id UUID,
    data JSONB,
    actor_id VARCHAR(255),
    correlation_id UUID,
    timestamp TIMESTAMPTZ
);
```

### Connection Setup

```python
from khora.storage import StorageConfig

config = StorageConfig(
    postgresql_url="postgresql://user:pass@localhost:5432/khora",
    pool_size=5,
    max_overflow=10
)
```

Or via environment:
```bash
export KHORA_DATABASE_URL="postgresql://user:pass@localhost:5432/khora"
```

## pgvector: Semantic Search

The pgvector extension turns PostgreSQL into a vector database. It stores embeddings and enables similarity search.

### What It Stores

**Chunk Embeddings** - Vector representations of document pieces:
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT,
    embedding vector(1536),  -- The magic: 1536 floats
    index INTEGER,           -- Position in document
    token_count INTEGER,
    created_at TIMESTAMPTZ
);

-- IVFFlat index for fast approximate search
CREATE INDEX ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Entity Embeddings** - For finding similar entities:
```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    name VARCHAR(255),
    entity_type VARCHAR(50),
    embedding vector(1536),
    -- ... other fields
);
```

### How Similarity Search Works

When you search, Khora:
1. Embeds your query using the same model as the stored content
2. Finds nearest neighbors using cosine similarity
3. Returns chunks sorted by how similar they are

```sql
-- The actual query (simplified)
SELECT id, content,
       1 - (embedding <=> $query_embedding) as similarity
FROM chunks
WHERE namespace_id = $namespace_id
ORDER BY embedding <=> $query_embedding
LIMIT 10;
```

The `<=>` operator computes cosine distance. Subtracting from 1 converts to similarity (higher = more similar).

### Embedding Configuration

Default model: `text-embedding-3-small` (OpenAI, 1536 dimensions)

```python
from khora.config import LiteLLMConfig

config = LiteLLMConfig(
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536
)
```

## Neo4j: The Knowledge Graph

Neo4j stores entities and relationships as a property graph - nodes connected by edges.

### What It Stores

**Entity Nodes**:
```cypher
(:Entity {
    id: "uuid-here",
    namespace_id: "uuid-here",
    name: "Albert Einstein",
    entity_type: "PERSON",
    description: "Theoretical physicist",
    confidence: 0.95,
    mention_count: 15,
    attributes: {
        birth_year: 1879,
        known_for: ["relativity", "E=mc²"]
    }
})
```

**Relationship Edges**:
```cypher
(:Entity)-[:WORKS_FOR {
    weight: 0.9,
    since: "1905-01-01"
}]->(:Entity)
```

### Relationship Types

Built-in types cover common cases:

| Type | Meaning |
|------|---------|
| `WORKS_FOR` | Employment |
| `KNOWS` | Personal connection |
| `MANAGES` / `REPORTS_TO` | Hierarchy |
| `COLLABORATES_WITH` | Professional relationship |
| `OWNS` / `PART_OF` | Ownership, membership |
| `LOCATED_IN` | Geographic |
| `DEPENDS_ON` | Technical dependency |
| `PRECEDES` / `FOLLOWS` | Temporal order |

You can also define custom types through the expertise system.

### Graph Queries

Finding an entity's neighborhood:
```cypher
MATCH (e:Entity {id: $entity_id})-[r*1..2]-(related:Entity)
WHERE related.namespace_id = $namespace_id
RETURN DISTINCT related, r
LIMIT 50
```

Finding paths between entities:
```cypher
MATCH path = shortestPath(
    (a:Entity {id: $source})-[*..3]-(b:Entity {id: $target})
)
RETURN path
```

### Connection Setup

```python
config = StorageConfig(
    neo4j_url="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your-password"
)
```

Or via environment:
```bash
export KHORA_NEO4J_URL="bolt://neo4j:password@localhost:7687"
```

## Dual Entity Storage

Here's something important: **entities live in both Neo4j AND pgvector**.

Why? Different query patterns:

- **Neo4j**: "Who works with Einstein?" → Graph traversal
- **pgvector**: "Find entities similar to this description" → Embedding similarity

When you create an entity, the `StorageCoordinator` stores it in both places:

```python
async def create_entity(self, entity: Entity) -> Entity:
    # Store in Neo4j for graph queries
    if self.graph:
        entity = await self.graph.create_entity(entity)

    # Store in pgvector for similarity search
    if self.vector:
        await self.vector.create_entity(entity)

    return entity
```

This redundancy is intentional - each backend serves different access patterns.

## The StorageCoordinator

You don't interact with backends directly. The `StorageCoordinator` orchestrates everything:

```python
from khora.storage import create_storage_coordinator, StorageConfig

config = StorageConfig(
    postgresql_url="postgresql://...",
    neo4j_url="bolt://..."
)

coordinator = create_storage_coordinator(config)
await coordinator.connect()

# Store a document
await coordinator.create_document(document)

# Search for similar chunks
results = await coordinator.search_similar_chunks(
    namespace_id,
    query_embedding,
    limit=10
)

# Get entity neighborhood
neighborhood = await coordinator.get_neighborhood(
    entity_id,
    depth=2
)

await coordinator.disconnect()
```

### Health Checking

Each backend reports its health:

```python
health = await coordinator.health_check()

print(f"PostgreSQL: {'OK' if health.relational else 'DOWN'}")
print(f"pgvector: {'OK' if health.vector else 'DOWN'}")
print(f"Neo4j: {'OK' if health.graph else 'DOWN'}")
print(f"Overall: {'healthy' if health.is_healthy else 'degraded'}")
```

## Protocol-Based Design

Each backend implements a protocol (Python's version of an interface):

```python
class VectorBackendProtocol(Protocol):
    async def connect(self) -> None: ...
    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        limit: int
    ) -> list[tuple[Chunk, float]]: ...

class GraphBackendProtocol(Protocol):
    async def create_entity(self, entity: Entity) -> Entity: ...
    async def get_neighborhood(
        self,
        entity_id: UUID,
        depth: int
    ) -> dict[str, Any]: ...
```

This means you could theoretically swap pgvector for Qdrant, or Neo4j for Memgraph, without changing the rest of the system. The protocols define the contract; implementations fulfill it.

## What's Next?

- **[Multi-Tenancy](multi-tenancy.md)** - Organizations, workspaces, namespaces
- **[Event Sourcing](event-sourcing.md)** - The immutable event log
- **[Overview](overview.md)** - High-level architecture

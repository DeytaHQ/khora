# Storage Backends

Khora doesn't use one database - it uses three, each chosen for what it does best. This might seem like overkill, but each backend excels at queries the others struggle with.

## The Three Musketeers

```
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│    PostgreSQL      │    │      pgvector      │    │       Neo4j        │
│                    │    │                    │    │                    │
│  The Record Keeper │    │  The Meaning       │    │  The Connector     │
│                    │    │  Finder            │    │                    │
│  Documents         │    │  "What's similar   │    │  "Who knows whom?" │
│  Who owns what     │    │  to this?"         │    │  "What's related   │
│  What happened     │    │                    │    │  to what?"         │
│  when              │    │  Embedding         │    │                    │
│                    │    │  similarity        │    │  Entity nodes      │
│  ACID guarantees   │    │  search            │    │  Relationship      │
│  Transactions      │    │                    │    │  edges             │
└────────────────────┘    └────────────────────┘    └────────────────────┘
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

**Namespaces** - The sole isolation boundary:
```sql
CREATE TABLE memory_namespaces (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    tenancy_mode VARCHAR(20),  -- shared or isolated
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    config_overrides JSONB DEFAULT '{}',
    sync_checkpoints JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
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
    max_overflow=10,
    pool_pre_ping=True,  # Check connection health before use
)
```

Or via environment:
```bash
export KHORA_DATABASE_URL="postgresql://user:pass@localhost:5432/khora"
export KHORA_STORAGE__POOL_PRE_PING=true
```

**`pool_pre_ping`** issues a lightweight `SELECT 1` before handing out a connection from the pool. This detects stale connections (from network interruptions, DB restarts, or idle timeouts) and transparently replaces them. Adds ~1ms per connection checkout but prevents `connection reset by peer` errors in long-running processes.

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
    embedding vector(1536),  -- The magic: 1536 floats (or halfvec for float16)
    index INTEGER,           -- Position in document
    token_count INTEGER,
    created_at TIMESTAMPTZ
);

-- HNSW index for fast approximate search (replaced IVFFlat)
CREATE INDEX ix_khora_chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (ef_construction = 128);
```

> **Note:** The HNSW index replaced the earlier IVFFlat index. HNSW provides better recall and doesn't require retraining when data grows. The `ef_construction=128` parameter (up from the default 64) improves index quality at build time. Query-time recall can be tuned separately with `SET hnsw.ef_search = N` (default 200 at query time).
>
> **halfvec support:** pgvector's `halfvec` type stores embeddings as float16 instead of float32, halving storage and memory usage with minimal quality loss. Khora supports halfvec via the `embedding_type` storage configuration.

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

When you create an entity, the `StorageCoordinator` stores it in both places. For updates and batch upserts, writes to graph and vector backends run in parallel via `asyncio.gather` — since neither backend depends on the other's result:

```python
async def update_entity(self, entity: Entity) -> Entity:
    # Graph and vector writes happen concurrently
    if self.graph and self.vector:
        graph_result, _ = await asyncio.gather(
            self.graph.update_entity(entity),
            self.vector.update_entity(entity),
        )
        return graph_result
```

This redundancy is intentional - each backend serves different access patterns. The parallel writes mean you don't pay a latency penalty for it.

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

## Shared Connection Pools

When multiple backends point at the same database — the common case where PostgreSQL, pgvector, and the event store all share one URL — `StorageFactory` avoids creating redundant connection pools.

`StorageFactory.get_or_create_engine()` normalizes the database URL (stripping query parameters and trailing slashes) and caches `AsyncEngine` instances by the normalized key. The first backend to request an engine for a given URL creates it; subsequent backends receive the same engine instance.

```python
# Three backends, one pool
factory = StorageFactory(config)

# All three calls return the same AsyncEngine:
pg_engine = factory.get_or_create_engine(config.postgresql_url)
vec_engine = factory.get_or_create_engine(config.postgresql_url)
event_engine = factory.get_or_create_engine(config.postgresql_url)

assert pg_engine is vec_engine is event_engine  # True
```

This reduces the total number of database connections from 3× pool size to 1× pool size. Backends that share an engine skip `dispose()` on disconnect to avoid pulling the pool out from under siblings — only the last backend to disconnect disposes the engine.

## Transactions

`StorageCoordinator.transaction()` provides atomic multi-backend writes through a single database transaction:

```python
async with coordinator.transaction() as txn:
    # All writes share one database session
    await coordinator.create_document(document, session=txn.session)
    await coordinator.upsert_entities_batch(
        namespace_id, entities, session=txn.session
    )

    # Savepoints for partial rollback
    async with txn.savepoint():
        await coordinator.create_relationships_batch(
            relationships, session=txn.session
        )
        # If this block raises, only the savepoint rolls back

    # Commit happens automatically when the outer context exits
    # Rollback happens automatically on exception
```

The `TransactionContext` returned by `transaction()` exposes:
- `txn.session` — the active `AsyncSession` to pass to backend write methods
- `txn.savepoint()` — create a nested savepoint for partial rollback

Backend write methods accept an optional `session` parameter. When provided, they join the existing transaction instead of creating their own.

## Batch Operations

For bulk ingestion, batch methods significantly reduce database round-trips. These are used by smart mode's post-ingestion resolution but are available for any bulk workflow.

### Entity Batch Upsert

```python
# Upsert up to 50 entities per batch
results = await coordinator.upsert_entities_batch(
    namespace_id,
    entities,
    batch_size=50,
)
# Returns: list of (entity, is_new) tuples
```

**Neo4j implementation** uses `UNWIND + MERGE`:
```cypher
UNWIND $entities AS e
MERGE (n:Entity {namespace_id: e.namespace_id, name: e.name, entity_type: e.entity_type})
ON CREATE SET n.id = e.id, n.description = e.description, ...
ON MATCH SET n.description = e.description, n.updated_at = e.updated_at, ...
```

**Concurrent batch coordination**: Non-overlapping entity batches run concurrently (up to `entity_write_concurrency`, default 12), but batches that share entity keys are automatically serialized by `_EntityKeyGate` to avoid Neo4j lock contention. A key is the `(namespace_id, name, entity_type)` triple — the same triple used in the `MERGE` clause. This means two batches touching completely different entities proceed in parallel, while two batches that both contain "Microsoft/ORGANIZATION" are queued so only one MERGE transaction runs at a time for that key.

**How `_EntityKeyGate` works**: The gate maintains a `dict[tuple, asyncio.Lock]` mapping entity keys to locks. Before a batch write, the coordinator extracts all entity keys from the batch, acquires the lock for each key (in sorted order to prevent deadlocks), and releases them after the write completes. Locks for keys that are no longer in use are pruned periodically to prevent unbounded memory growth.

**PostgreSQL implementation** uses a single multi-row `INSERT ... ON CONFLICT DO UPDATE` — all entities in one SQL statement rather than individual inserts.

### Relationship Batch Create

```python
# Create relationships in batches, grouped by type
count = await coordinator.create_relationships_batch(
    relationships,
    batch_size=50,
)
# Returns: number of relationships created
```

**Neo4j implementation** groups relationships by type and uses `UNWIND + CREATE` with dynamic relationship types.

### When Batch Operations Are Used

| Context | Method | Why |
|---------|--------|-----|
| Per-document ingestion | `create_relationships_batch()` | Store all extracted relationships in one transaction |
| Per-document ingestion | `update_entity_embeddings_batch()` | Store all entity embeddings in one transaction |
| Smart mode post-resolution | `upsert_entities_batch()` | Write resolved entities after cross-document unification |
| Smart mode post-inference | `create_relationships_batch()` | Write inferred relationships in bulk |
| Any bulk workflow | All batch methods | Reduce N+1 query patterns to ceil(N/batch_size) queries |

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
    # Batch operations
    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        batch_size: int = 50,
    ) -> list[tuple[Entity, bool]]: ...
    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        batch_size: int = 50,
    ) -> int: ...
```

This means you can swap pgvector for ArcadeDB, or Neo4j for Kùzu/Memgraph, without changing the rest of the system. The protocols define the contract; implementations fulfill it.

## Alternative Graph Backends

Beyond Neo4j, Khora supports three additional graph backends:

| Backend | Type | Protocol | Best For |
|---------|------|----------|----------|
| **Neo4j** (default) | Server | Bolt/Cypher | Production, multi-user, large graphs |
| **Kùzu** | Embedded | Cypher | Single-process, CI/testing, edge devices |
| **Memgraph** | Server | Bolt/Cypher | In-memory, low-latency, streaming |
| **ArcadeDB** | Server | HTTP/Cypher+SQL | Multi-model (graph + vector in one DB) |
| **SurrealDB** | Server | WebSocket/HTTP | Unified multi-model (graph + vector + relational in one DB) |

### Kùzu (Embedded)

Kùzu runs in-process — no server needed. Ideal for testing and small deployments:

```yaml
storage:
  graph:
    backend: kuzu
    database_path: ./kuzu_db
```

```python
from khora.config.schema import KuzuConfig, StorageSettings

settings = StorageSettings(graph=KuzuConfig(database_path="./kuzu_db"))
```

Note: Kùzu's Python API is synchronous; all calls are wrapped in `asyncio.to_thread()`.

### Memgraph

Memgraph speaks the Bolt protocol using the same `neo4j` Python driver. Key differences from Neo4j: no APOC, different index syntax, no multi-database support, in-memory by default.

```yaml
storage:
  graph:
    backend: memgraph
    url: bolt://localhost:7687
    user: memgraph
```

### ArcadeDB (Multi-Model)

ArcadeDB can serve as **both** graph and vector backend. When both configs point to the same instance, Khora creates one backend object for both roles:

```yaml
storage:
  graph:
    backend: arcadedb
    url: http://localhost:2480
    database: khora
  vector:
    backend: arcadedb
    url: http://localhost:2480
    database: khora
```

## SurrealDB: The Unified Backend

SurrealDB is an alternative that serves as **all three backends** — relational, vector, and graph — in a single database. This simplifies deployment dramatically: one database instead of PostgreSQL + pgvector + Neo4j.

### What It Provides

| Role | SurrealDB Implementation |
|------|-------------------------|
| **Relational** | Document/namespace storage with SurrealQL queries |
| **Vector** | Native vector similarity search with HNSW indexes |
| **Graph** | Native graph traversal via SurrealQL graph queries |
| **Event Store** | Append-only event log with SurrealQL |

### Configuration

```python
from khora.config.schema import SurrealDBConfig, StorageSettings

settings = StorageSettings(
    surrealdb=SurrealDBConfig(
        url="ws://localhost:8000/rpc",
        namespace="khora",
        database="main",
    )
)
```

Or via environment:
```bash
export KHORA_STORAGE__SURREALDB__URL="ws://localhost:8000/rpc"
export KHORA_STORAGE__SURREALDB__NAMESPACE="khora"
export KHORA_STORAGE__SURREALDB__DATABASE="main"
```

### Architecture

```
src/khora/storage/backends/surrealdb/
├── __init__.py       # Package exports
├── connection.py     # WebSocket/HTTP connection management
├── relational.py     # Document, namespace, chunk storage
├── vector.py         # Embedding storage and similarity search
├── graph.py          # Entity nodes, relationship edges, graph traversal
├── event_store.py    # Immutable event log
├── schema.py         # SurrealQL schema definitions (tables, indexes)
└── _helpers.py       # Shared utilities (UUID conversion, etc.)
```

### When to Use SurrealDB

| Scenario | Recommendation |
|----------|---------------|
| Simplest deployment | SurrealDB — one database to manage |
| Maximum performance | PostgreSQL + pgvector + Neo4j — each optimized for its role |
| Development/testing | SurrealDB — easy setup, no multi-DB coordination |
| Production at scale | PostgreSQL + Neo4j — mature, battle-tested |

> **Status:** SurrealDB support is Phase 1 — the foundation is implemented and functional, but may lack some advanced features available in the PostgreSQL + Neo4j stack.

## Alternative Vector Backends

| Backend | Type | Best For |
|---------|------|----------|
| **pgvector** (default) | PostgreSQL extension | Most deployments, colocated with relational data |
| **SurrealDB** | WebSocket/HTTP | Unified single-server setup |
| **ArcadeDB** | HTTP/REST | Multi-model single-server setup |

## Bulk Mode

For initial data loading, `bulk_mode=True` applies write optimizations across all backends:

```python
config = StorageSettings(bulk_mode=True)
```

| Backend | Optimization |
|---------|-------------|
| **pgvector** | Defers HNSW index creation until after bulk load. Call `ensure_hnsw_indexes()` afterward. |
| **Neo4j** | Larger batch sizes, deferred constraints, reduced per-write validation |
| **PostgreSQL** | Standard behavior (already batch-optimized) |

After bulk loading completes:

```python
from khora.storage.optimize import ensure_hnsw_indexes

# Rebuild deferred indexes (idempotent)
await ensure_hnsw_indexes(engine, schema="public")

# Re-enable Neo4j constraints
await neo4j_backend.ensure_constraints()
```

> **Note:** Bulk mode is for initial data loading, not production use. It trades consistency guarantees for throughput.

## Configuration

### New-Style (Recommended)

Use discriminated union configs for explicit backend selection:

```python
from khora.config.schema import KhoraConfig, StorageSettings, KuzuConfig

config = KhoraConfig(
    storage=StorageSettings(
        postgresql_url="postgresql://localhost:5432/khora",
        graph=KuzuConfig(database_path="./kuzu_db"),
    )
)
```

### Legacy (Backwards Compatible)

Flat fields continue to work and are automatically migrated:

```bash
export KHORA_DATABASE_URL="postgresql://localhost:5432/khora"
export KHORA_NEO4J_URL="bolt://neo4j:password@localhost:7687"
```

### Install Optional Dependencies

```bash
# Install specific backend
pip install khora[kuzu]
pip install khora[memgraph]
pip install khora[arcadedb]

# Install all graph backends
pip install khora[graph-all]

# Install everything
pip install khora[all-backends]
```

## What's Next?

- **[Multi-Tenancy](multi-tenancy.md)** - Namespace isolation
- **[Event Sourcing](event-sourcing.md)** - The immutable event log
- **[Overview](overview.md)** - High-level architecture

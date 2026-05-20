# Storage Backends

Khora doesn't use one database - it uses three, each chosen for what it does best. This might seem like overkill, but each backend excels at queries the others struggle with.

## The Three Musketeers

```text
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
export KHORA_STORAGE_POOL_PRE_PING=true
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

**New-style graph config (recommended):**

```python
from khora.config.schema import KhoraConfig, Neo4jConfig, StorageSettings

# Embedded credentials
cfg = KhoraConfig(
    database_url="postgresql://user:pass@localhost:5432/khora",
    storage=StorageSettings(
        graph=Neo4jConfig(url="bolt://neo4j:password@localhost:7687"),
    ),
)

# Split credentials (URL and password managed separately - e.g. password
# sourced from a secrets manager, URL from service discovery)
cfg = KhoraConfig(
    database_url="postgresql://user:pass@localhost:5432/khora",
    storage=StorageSettings(
        graph=Neo4jConfig(
            url="bolt://localhost:7687",
            user="neo4j",
            password="from-secrets-manager",
        ),
    ),
)
```

Embedded credentials in the URL take precedence; the explicit `password` field is used as the fallback when the URL has none.

## Dual Entity Storage

Here's something important: **entities live in both Neo4j AND pgvector**.

Why? Different query patterns:

- **Neo4j**: "Who works with Einstein?" → Graph traversal
- **pgvector**: "Find entities similar to this description" → Embedding similarity

When you create an entity, the `StorageCoordinator` stores it in both places. For updates and batch upserts, writes to graph and vector backends run in parallel via `asyncio.gather` - since neither backend depends on the other's result. Both calls carry the caller's `namespace_id` so the underlying SQL / Cypher filters the row at the query layer:

```python
async def update_entity(self, entity: Entity, *, namespace_id: UUID) -> Entity:
    # Graph and vector writes happen concurrently
    if self._graph and self._vector:
        graph_result, _ = await asyncio.gather(
            self._graph.update_entity(entity, namespace_id=namespace_id),
            self._vector.update_entity(entity, namespace_id=namespace_id),
        )
        return graph_result
```

This redundancy is intentional - each backend serves different access patterns. The parallel writes mean you don't pay a latency penalty for it.

## The StorageCoordinator

You don't interact with backends directly. The `StorageCoordinator` orchestrates everything, and every read or mutation it routes carries the caller's `namespace_id` so the underlying backend can filter at the query layer:

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

# Search for similar chunks (namespace scope is implicit in the call)
results = await coordinator.search_similar_chunks(
    namespace_id,
    query_embedding,
    limit=10,
)

# Get entity neighborhood - namespace_id is required, kwarg-only
neighborhood = await coordinator.get_neighborhood(
    entity_id,
    namespace_id=namespace_id,
    depth=2,
)

await coordinator.disconnect()
```

### Namespace-scoped reads and writes (v0.16.0)

Every read, exists-check, and mutation method on each storage backend Protocol takes `*, namespace_id: UUID` as a required keyword-only parameter and filters at the SQL / Cypher / SurrealQL layer. The namespace is **never** post-checked against a returned row - when an id belongs to a different namespace, the backend returns `None` / an empty result / `False` straight from the query. The full surface tightened in PRs #761, #765, #766, #769:

- **Reads** - `get_document(_s_batch)`, `get_document_sources_batch`, `get_document_projections_batch`, `get_document_by_external_id`, `get_documents_by_external_ids`, `get_chunk(_s_batch)`, `get_chunks_by_document`, `entity_exists`, vector `get_entity` / `get_entities_batch`, graph `get_entity(_ies_batch)`, `get_relationship`, `get_episode`, `get_entity_relationships`, `get_neighborhood(_s_batch)`, `find_paths`, `get_temporal_neighbors`, event store `get_events_for_resource`, `get_latest_event`.
- **Writes** - `delete_document`, `delete_chunks_by_document`, `update_entity`, `update_entity_embedding(_s_batch)`, `delete_entities_batch`, `delete_relationships_batch`, `supersede_fact`, `delete_entity`, `delete_relationship`. Neo4j additionally hardens `retire_orphaned_relationships_batch` and `remap_source_document_ids_batch` against cross-namespace effect.

The coordinator's public `coordinator.{relational,vector,graph,event_store}` attributes are now wrapped in a `NamespaceRequiredProxy` (see `src/khora/storage/_namespace_proxy.py`) that:

1. Emits one `DeprecationWarning` per role per process on first access - direct backend access is deprecated; use the coordinator's facade methods.
2. Refuses to dispatch any of the namespace-scoped read methods listed above unless the caller passes `namespace_id=…` (raises `TypeError`).
3. Does not forward access to underscore-prefixed attributes - backend internals such as `_engine`, `_handle`, `_conn`, `_session_factory` are only reachable via the private `coord._{role}` accessors used by coordinator internals.

The public `coordinator.{relational,vector,graph,event_store}` attributes are scheduled for removal in v0.17. Internal coordinator code uses `self._{role}` directly.

A structural signature gate (`tests/security/test_cross_namespace_idor_signatures.py`) walks every concrete backend at collection time and asserts that every `get_*` / `entity_exists` / `find_paths` / `get_neighborhood*` / `delete_*` / `update_entity*` / `supersede_*` method with a required id-typed parameter declares `*, namespace_id: UUID` kwarg-only. A new method that violates this contract fails CI at collection time. See `docs/architecture/multi-tenancy.md` for the structural invariant rationale.

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

When multiple backends point at the same database - the common case where PostgreSQL, pgvector, and the event store all share one URL - `StorageFactory` avoids creating redundant connection pools.

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

This reduces the total number of database connections from 3× pool size to 1× pool size. Backends that share an engine skip `dispose()` on disconnect to avoid pulling the pool out from under siblings - only the last backend to disconnect disposes the engine.

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
- `txn.session` - the active `AsyncSession` to pass to backend write methods
- `txn.savepoint()` - create a nested savepoint for partial rollback

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

**Concurrent batch coordination**: Non-overlapping entity batches run concurrently (up to `entity_write_concurrency`, default 12), but batches that share entity keys are automatically serialized by `_EntityKeyGate` to avoid Neo4j lock contention. A key is the `(namespace_id, name, entity_type)` triple - the same triple used in the `MERGE` clause. This means two batches touching completely different entities proceed in parallel, while two batches that both contain "Microsoft/ORGANIZATION" are queued so only one MERGE transaction runs at a time for that key.

**How `_EntityKeyGate` works**: The gate maintains a `dict[tuple, asyncio.Lock]` mapping entity keys to locks. Before a batch write, the coordinator extracts all entity keys from the batch, acquires the lock for each key (in sorted order to prevent deadlocks), and releases them after the write completes. Locks for keys that are no longer in use are pruned periodically to prevent unbounded memory growth.

**PostgreSQL implementation** uses a single multi-row `INSERT ... ON CONFLICT DO UPDATE` - all entities in one SQL statement rather than individual inserts.

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

Each backend implements a protocol (Python's version of an interface). The full set of read, exists, and mutation methods declares `*, namespace_id: UUID` as a required keyword-only parameter (PRs #761, #765, #766, #769):

```python
class VectorBackendProtocol(Protocol):
    async def connect(self) -> None: ...
    async def search_similar(
        self,
        namespace_id: UUID,
        query_embedding: list[float],
        *,
        limit: int = 10,
    ) -> list[tuple[Chunk, float]]: ...
    async def get_chunk(self, chunk_id: UUID, *, namespace_id: UUID) -> Chunk | None: ...
    async def entity_exists(self, entity_id: UUID, *, namespace_id: UUID) -> bool: ...

class GraphBackendProtocol(Protocol):
    async def create_entity(self, entity: Entity) -> Entity: ...
    async def get_entity(self, entity_id: UUID, *, namespace_id: UUID) -> Entity | None: ...
    async def get_neighborhood(
        self,
        entity_id: UUID,
        *,
        namespace_id: UUID,
        depth: int = 1,
    ) -> dict[str, Any]: ...
    async def delete_entity(self, entity_id: UUID, *, namespace_id: UUID) -> bool: ...
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

This means you can swap pgvector for SurrealDB, or Neo4j for Memgraph, without changing the rest of the system. The protocols define the contract; implementations fulfill it.

Backend internals - connection handles, session factories, raw drivers - are reachable only through underscore-prefixed names (`_engine`, `_handle`, `_conn`, `_session_factory`); the deprecation proxy on `StorageCoordinator` does not forward them to external callers.

## Alternative Graph Backends

Beyond Neo4j, Khora supports the following additional graph backends:

| Backend | Type | Protocol | Best For |
|---------|------|----------|----------|
| **Neo4j** (default) | Server | Bolt/Cypher | Production, multi-user, large graphs |
| **Memgraph** | Server | Bolt/Cypher | In-memory, low-latency, streaming |
| **SurrealDB** | Server/Embedded | WebSocket/HTTP | Unified multi-model (graph + vector + relational in one DB) |
| **Neptune** | Managed (AWS) | Bolt/OpenCypher | AWS-native, managed, no operational overhead |
| **AGE** | PostgreSQL extension | Cypher-in-SQL | Graph queries without extra infrastructure |

### Memgraph

Memgraph speaks the Bolt protocol using the same `neo4j` Python driver. Key differences from Neo4j: no APOC, different index syntax, no multi-database support, in-memory by default.

```yaml
storage:
  graph:
    backend: memgraph
    url: bolt://localhost:7687
    user: memgraph
```

## AWS Neptune

Neptune is Amazon's managed graph database. Khora connects via the Bolt protocol (OpenCypher) and optionally supports IAM SigV4 auth for secure, password-less access.

```bash
pip install khora[neptune]          # Bolt protocol
pip install khora[neptune-iam]      # Bolt + IAM SigV4
```

```bash
export KHORA_STORAGE_GRAPH_BACKEND=neptune
export KHORA_STORAGE_GRAPH_URL="bolt://your-cluster.neptune.amazonaws.com:8182"
```

**When to use:** AWS-native deployments where you want a managed graph service with no operational overhead. Neptune handles backups, patching, and scaling automatically.

## PostgreSQL AGE

Apache AGE adds Cypher query support to PostgreSQL via an extension. Khora's AGE backend runs Cypher-in-SQL, sharing the same connection pool as the relational backend - no separate graph server required.

```bash
pip install khora[age]
```

```bash
export KHORA_STORAGE_GRAPH_BACKEND=age
export KHORA_STORAGE_GRAPH_AGE_GRAPH_NAME=khora  # default: khora
```

**When to use:** When you want graph queries without adding another database to your stack. AGE runs inside PostgreSQL, so there is no extra infrastructure to manage.

**UUID interpolation hardening (v0.16.0).** AGE's `cypher()` SQL function rejects `$param` placeholders, so UUIDs are interpolated directly into the Cypher source. `AgeBackend._uuid_lit(value)` routes every UUID interpolation site (in `storage/backends/age.py`) through `UUID(...)` validation before string conversion. This is type-safe today - every caller passes `uuid.UUID` - and injection-resistant against future duck-typed callers that might pass a `str`. Invalid UUIDs fail fast at the boundary with a `ValueError`, never reaching the graph store.

## SurrealDB: The Unified Backend

SurrealDB is an alternative that serves as **all three backends** - relational, vector, and graph - in a single database. This simplifies deployment dramatically: one database instead of PostgreSQL + pgvector + Neo4j.

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
export KHORA_STORAGE_SURREALDB_URL="ws://localhost:8000/rpc"
export KHORA_STORAGE_SURREALDB_NAMESPACE="khora"
export KHORA_STORAGE_SURREALDB_DATABASE="main"
```

### Architecture

```text
src/khora/storage/backends/surrealdb/
├── __init__.py       # Package exports
├── connection.py     # WebSocket/HTTP connection management
├── relational.py     # Document, namespace, chunk storage
├── vector.py         # Embedding storage and similarity search
├── graph.py          # Entity nodes, relationship edges, graph traversal
├── event_store.py    # Immutable event log
├── schema.py         # SurrealQL schema definitions (tables, indexes)
└── _helpers.py       # Shared utilities (UUID conversion, _rid binding, etc.)
```

### `RELATES_TO` schema (v0.16.0)

The `relates_to` RELATION table now carries a Khora-owned identity column:

```surql
DEFINE TABLE IF NOT EXISTS relates_to TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS rel_id ON relates_to TYPE string;
-- ... namespace_id, relationship_type, weight, properties, ...
DEFINE INDEX IF NOT EXISTS idx_relates_to_rel_id ON relates_to FIELDS rel_id UNIQUE;
```

`rel_id` stores the Khora `Relationship.id` UUID. Until v0.16.0 the graph adapter relied on SurrealDB's auto-generated record id, but `SCHEMAFULL` plus the parameterized-id bug below meant some `RELATE` writes silently dropped their endpoints and Khora's `Relationship.id` was not round-tripped at all. The `UNIQUE` index on `rel_id` lets `get_relationship(...)` look up edges by the Khora id without colliding with adjacent rows.

### `table:⟨$var⟩` interpolation bug fix (v0.16.0, PR #770)

SurrealDB does not substitute parameters inside the RecordID-literal shorthand `table:⟨$var⟩` - the `$var` reaches the engine as a literal string. 19 sites across `storage/backends/surrealdb/{graph,vector}.py` were silently broken before v0.16.0: `graph.get_relationship`, `vector.get_chunk`, and `vector.get_chunks_by_document` always returned `None` / empty, and every relationship row stored by `RELATE` had corrupted `in` / `out` endpoints with a `rel_id` of `None` (the `SCHEMAFULL` table dropped the writes without raising). The fix is two-fold:

- Bind via the `_rid()` parameter helper (in `surrealdb/_helpers.py`), which constructs a real `RecordID` object Surreal honours as a parameter.
- Use `(type::thing($var))` in `FOR` loops where a parameter-substituted record-id is needed inside SurrealQL.

Combined with the new `rel_id` field + `UNIQUE` index, the graph adapter now writes and reads relationships correctly under embedded, memory, and remote modes.

### SurrealDB transactions and batching

SurrealDB supports SurrealQL-level transactions (`BEGIN` / `COMMIT` / `CANCEL`) on the WebSocket connection. The Python client surfaces these as ordinary queries in remote mode. Embedded (`surrealkv://`) and memory (`memory://`) modes raise `UnsupportedFeatureError` on `BEGIN`, so khora preserves the per-statement-atomicity contract there and exposes a batched alternative.

`SurrealDBConnection` (v0.12.0) ships three primitives:

| Method / property | Remote (`ws://`) | Embedded / memory |
|---|---|---|
| `supports_transactions` (property) | `True` | `False` |
| `async with conn.transaction():` | Wraps body in `BEGIN TRANSACTION` / `COMMIT TRANSACTION`. On exception, issues `CANCEL TRANSACTION` and re-raises. If `CANCEL` itself fails the original exception is preserved. | No-op context manager - body runs as ordinary per-statement-atomic writes. |
| `await conn.execute_batch([(sql, bindings), ...])` | Same semantics as embedded - concatenates with `;` and runs as one round-trip; for true atomicity prefer `transaction()`. | Multiple statements in one round-trip. Parameter-name collisions across statements raise rather than silently overwrite. |

Example:

```python
async with conn.transaction():
    await conn.execute("CREATE entity SET name = $n", {"n": "Acme"})
    await conn.execute("RELATE $a->relates_to->$b", {"a": acme_id, "b": deal_id})
# Remote: COMMIT issued automatically on clean exit.
# Embedded/memory: each statement runs as its own surrealkv tx (existing contract preserved).
```

The coordinator's `StorageCoordinator.transaction()` remains session-shaped (SQLAlchemy `AsyncSession`) and does not yet route to SurrealDB transactions - that integration is a follow-up. For atomic SurrealDB-only multi-statement operations on remote stacks today, use the `conn.transaction()` primitive directly.

### When to Use SurrealDB

| Scenario | Recommendation |
|----------|---------------|
| Simplest deployment | SurrealDB - one database to manage |
| Maximum performance | PostgreSQL + pgvector + Neo4j - each optimized for its role |
| Development/testing | SurrealDB - easy setup, no multi-DB coordination |
| Production at scale | PostgreSQL + Neo4j - mature, battle-tested |

## Alternative Vector Backends

| Backend | Type | Best For |
|---------|------|----------|
| **pgvector** (default) | PostgreSQL extension | Most deployments, colocated with relational data |
| **SurrealDB** | WebSocket/HTTP | Unified single-server setup |
| **LanceDB** | Embedded (file-backed) | Chronicle engine, zero-infrastructure deployments |

### `sqlite_lance` defense-in-depth (v0.16.0, PR #769)

The `sqlite_lance` adapter pairs SQLite (chunk metadata, source of truth) with LanceDB (vector index). After the LanceDB nearest-neighbour query returns a list of chunk ids, the SQLite re-fetch step now adds `AND namespace_id = ?` to the row lookup rather than trusting LanceDB's filter alone. If LanceDB's where-clause ever regresses, the SQLite filter still keeps cross-namespace rows out of the result.

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
from khora.config.schema import KhoraConfig, StorageSettings, MemgraphConfig

config = KhoraConfig(
    storage=StorageSettings(
        postgresql_url="postgresql://localhost:5432/khora",
        graph=MemgraphConfig(url="bolt://localhost:7687"),
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
pip install khora[memgraph]
# Install all graph backends
pip install khora[graph-all]

# Install everything
pip install khora[all-backends]
```

## What's Next?

- **[Multi-Tenancy](multi-tenancy.md)** - Namespace isolation
- **[Event Sourcing](event-sourcing.md)** - The immutable event log
- **[Overview](overview.md)** - High-level architecture

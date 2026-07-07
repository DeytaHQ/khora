# Storage Backends

Khora doesn't use one database - it uses three, each chosen for what it does best. Each backend is chosen because it performs queries the others cannot do or handle less well.

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
export KHORA_STORAGE_POSTGRESQL_POOL_PRE_PING=true
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
    embedding vector(1536),  -- 1536 floats (or halfvec for float16)
    chunk_index INTEGER,     -- Position in document
    token_count INTEGER,
    created_at TIMESTAMPTZ
);

-- HNSW index for fast approximate search (replaced IVFFlat)
CREATE INDEX ix_khora_chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (ef_construction = 128);
```

> **Note:** `ix_khora_chunks_embedding_hnsw` is the HNSW index on the vectorcypher runtime table `khora_chunks`. The ORM/Alembic `chunks` table has its own HNSW index named `ix_chunks_embedding_hnsw`. The `chunks` label in the CREATE statement above is the runtime-table shape.

> **Note:** The HNSW index replaced the earlier IVFFlat index. HNSW provides better recall and doesn't require retraining when data grows. The `ef_construction=128` parameter (up from the default 64) improves index quality at build time. Query-time recall can be tuned separately with `SET hnsw.ef_search = N` (config default 100 at query time).
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

Ordering by the raw `<=>` operator ascending matters: it is the only form pgvector's HNSW index can serve. Ordering by the wrapped similarity (`1 - distance DESC`) forces a full sequential scan (#1407). Similarity is computed in the projection only, and `min_similarity` is applied as a post-filter on the returned distance for the same reason. Because khora always ANDs a namespace filter (which pgvector applies *after* the index scan), queries also set `SET LOCAL hnsw.iterative_scan = relaxed_order` (pgvector >= 0.8) so a selective namespace cannot starve the result set below `limit`; `SET LOCAL` scopes the setting to the transaction, so nothing leaks across pooled connections. Query-time accuracy is tuned via `KHORA_STORAGE_HNSW_EF_SEARCH` (default 100).

### HNSW at Scale (1-10M chunks)

**RAM sizing.** HNSW queries are only fast while the index stays in memory (shared_buffers + OS page cache). Each element stores its vector plus ~`2 * m` link slots per layer. With the default halfvec (float16) expression indexes from migration 018 (`m = 24`, 1536 dims), budget roughly **3.5 KB per row**: ~3.5 GB at 1M chunks, ~35 GB at 10M. Full-precision indexes (migration 007) double the vector portion: ~6.5 KB per row, ~65 GB at 10M. Size `shared_buffers` (or the machine) so the hot index fits; a spilled HNSW index degrades to random I/O per graph hop.

**Bulk-load pattern.** Inserting rows into a table that already has an HNSW index builds the graph row-by-row - by far the slowest path. For an initial 1M+ ingest:

1. Drop (or defer creating) the HNSW indexes.
2. Bulk-insert the data (`COPY` or large batched inserts).
3. Build the indexes afterwards with `CREATE INDEX CONCURRENTLY`, after raising `maintenance_work_mem` (ideally >= the index size, e.g. `8GB`) and `max_parallel_maintenance_workers` (e.g. `7`) for the session. The graph is built in memory and written once.

Khora's migrations already use `CREATE INDEX CONCURRENTLY` for all HNSW indexes, so running migrations *after* a bulk restore naturally follows this pattern.

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

### Namespace-scoped reads and writes

Every read, exists-check, and mutation method on each storage backend Protocol takes `*, namespace_id: UUID` as a required keyword-only parameter and filters at the SQL / Cypher / SurrealQL layer. The namespace is **never** post-checked against a returned row - when an id belongs to a different namespace, the backend returns `None` / an empty result / `False` straight from the query. The full surface tightened in PRs #761, #765, #766, #769:

- **Reads** - `get_document(_s_batch)`, `get_document_sources_batch`, `get_document_projections_batch`, `get_document_by_external_id`, `get_documents_by_external_ids`, `get_chunk(_s_batch)`, `get_chunks_by_document`, `entity_exists`, vector `get_entity` / `get_entities_batch`, graph `get_entity(_ies_batch)`, `get_relationship`, `get_episode`, `get_entity_relationships`, `get_neighborhood(_s_batch)`, `find_paths`, `get_temporal_neighbors`, event store `get_events_for_resource`, `get_latest_event`.
- **Writes** - `delete_document`, `delete_chunks_by_document`, `update_entity`, `update_entity_embedding(_s_batch)`, `delete_entities_batch`, `delete_relationships_batch`, `supersede_fact`, `delete_entity`, `delete_relationship`. Neo4j additionally hardens `retire_orphaned_relationships_batch` and `remap_source_document_ids_batch` against cross-namespace effect.

The coordinator's public `coordinator.{relational,vector,graph,event_store}` attributes are now wrapped in a `NamespaceRequiredProxy` (see `src/khora/storage/_namespace_proxy.py`) that:

1. Emits one `DeprecationWarning` per role per process on first access - direct backend access is deprecated; use the coordinator's facade methods.
2. Refuses to dispatch any of the namespace-scoped read methods listed above unless the caller passes `namespace_id=…` (raises `TypeError`).
3. Does not forward access to underscore-prefixed attributes - backend internals such as `_engine`, `_handle`, `_conn`, `_session_factory` are only reachable via the private `coord._{role}` accessors used by coordinator internals.

The public `coordinator.{relational,vector,graph,event_store}` attributes are deprecated; call the coordinator's facade methods instead. Internal coordinator code uses `self._{role}` directly.

### Graph-less fallback for list_entities / list_relationships

When no graph backend is configured (PostgreSQL-only chronicle stacks), `StorageCoordinator.list_entities` and `list_relationships` fall back to the pgvector backend (`PgvectorBackend.list_entities` / `list_relationships`) rather than raising. This path was introduced in #587 to prevent `AttributeError` crashes in the expansion pipeline on graph-less stacks.

Since #1066, extracted relationships are persisted to the pgvector relationships mirror on the graph-less chronicle+PG path (previously they were silently dropped: `remember()` reported success while `stats().relationships` stayed `0`), so the `list_relationships` fallback now returns the stored edges rather than `[]`. The #1429 canonical-id sync (above) is what keeps that write FK-safe when an entity is re-mentioned across documents.

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

`StorageCoordinator.transaction()` provides atomic writes scoped to the shared PostgreSQL backends (relational, pgvector, event store) through a single `AsyncSession`. Neo4j and embedded LanceDB are NOT participants in this session - writes to those backends during the same block are not rolled back if the SQL transaction rolls back:

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
# batch_size is keyword-only (default 200 on pgvector / coordinator, 100 on Neo4j / base)
results = await coordinator.upsert_entities_batch(
    namespace_id,
    entities,
    batch_size=200,
)
# Returns: list of (entity, is_new) tuples, one per unique MERGE key.
# Duplicate entities in the caller-supplied list that share the same
# (namespace_id, name, entity_type) MERGE key collapse to a single
# result tuple - hooks fire once, not once per input (#1329).
#
# Canonical-id sync (#1429 / #806): on a re-mention, an entity arrives
# with a fresh extraction-time UUID. ON CONFLICT DO UPDATE / MERGE keeps
# the *existing* row id, and the input entity.id is synced in place to
# that canonical stored id before return - pgvector derives it from the
# RETURNING row, Neo4j and sqlite_lance apply the same remap. Callers
# build relationship endpoints from entity.id, so without this the FK
# would point at a throwaway UUID and abort ingest on graph-less
# chronicle+PG stacks.
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
# Create relationships in batches, grouped by type (batch_size keyword-only)
results = await coordinator.create_relationships_batch(
    relationships,
    batch_size=50,
)
# Returns: list[(relationship, is_new)] - one tuple per unique MERGE key,
# mirroring upsert_entities_batch's (entity, is_new) contract (#1320).
# Each relationship's `id` is synced in place to the canonical stored
# edge id; `is_new` is True for a genuine create, False for a dedup-merge.
# Duplicate relationships in the caller-supplied list that share the same
# (namespace_id, source, target, type) MERGE key collapse to a single
# result tuple - hooks fire once, not once per input (#1329 Part 2).
# len(results) may be less than len(relationships) when duplicates are present.
```

**Neo4j implementation** groups relationships by type and uses `UNWIND + MERGE` (matched on source/target + namespace) with dynamic relationship types.

**Created-vs-merged accuracy (#1320).** The split is *exact* on the MERGE-by-endpoint backends (Neo4j, Memgraph), where an `OPTIONAL MATCH` of the pre-MERGE edge reports `is_new`. The ON-CONFLICT(id) fallbacks (pgvector, sqlite_lance) report it from the relationship-`id` collision they key on (so a fresh-id edge between an already-related pair reads `is_new=True`). SurrealDB's bare `RELATE` and the per-record `GraphBackendBase` default (Neptune, AGE) cannot distinguish create from merge and report a best-effort `is_new=True` with the input id. The `relationship.created` / `relationship.updated` semantic hooks are dispatched from these results on both engines (Chronicle ingest flow + VectorCypher remember path), carrying the canonical stored id.

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
    # Batch operations (params are keyword-only)
    async def upsert_entities_batch(
        self,
        namespace_id: UUID,
        entities: list[Entity],
        *,
        batch_size: int = 100,       # pgvector / coordinator default 200
        bulk_mode: bool = False,
    ) -> list[tuple[Entity, bool]]: ...
    async def create_relationships_batch(
        self,
        relationships: list[Relationship],
        *,
        batch_size: int = 100,       # neo4j default 200, coordinator 50
    ) -> list[tuple[Relationship, bool]]: ...
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

**Dream graph mirror is flat soft-delete only (#1278).** Memgraph has no versioning primitives and no APOC, and stores `valid_until` as a plain string property. So unlike the Neo4j mirror (which snapshots the retired node into a `:EntityVersion` chain), `MemgraphBackend.soft_retire_entities_batch` simply SETs `valid_until` by id; endpoint rewrite (`rewrite_relationship_endpoints_batch`) and relabel (`rename_types_batch`) are re-create + delete in plain Cypher (no APOC). The reverse verbs flat-restore by clearing `valid_until`. `supports_dream_mirror()` advertises `prune_edges`, `dedupe_entities`, and `normalize_schema` (the flat-capable kinds); `community_summary` (the GraphRAG `:Community` materialization) is **not** advertised. The `list_entities` / `list_relationships` read paths filter `valid_until` unconditionally so a mirrored soft-delete is hidden from recall in lockstep with the PG read filter. The cross-store live-set invariant is guarded by `tests/integration/dream/test_memgraph_dream_mirror_integration.py` against a docker-compose Memgraph stack (3.x; `docker compose up -d memgraph`, opt-in `memgraph` profile). Note: the engine routes Memgraph through the storage-factory-built backend and skips the Neo4j-only `DualNodeManager`, so the `:Chunk` dual-node temporal recall path is Neo4j/SurrealDB/sqlite_lance only.

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
export KHORA_STORAGE_GRAPH_GRAPH_NAME=khora  # default: khora_graph
```

The doubled `GRAPH` in `KHORA_STORAGE_GRAPH_GRAPH_NAME` is not a typo - the first `GRAPH` comes from the sub-object name (`storage.graph`), and the second `GRAPH_NAME` is the field name on `AGEConfig`.

**When to use:** When you want graph queries without adding another database to your stack. AGE runs inside PostgreSQL, so there is no extra infrastructure to manage.

**UUID interpolation hardening.** AGE's `cypher()` SQL function rejects `$param` placeholders, so UUIDs are interpolated directly into the Cypher source. `AgeBackend._uuid_lit(value)` routes every UUID interpolation site (in `storage/backends/age.py`) through `UUID(...)` validation before string conversion. This is type-safe today - every caller passes `uuid.UUID` - and injection-resistant against future duck-typed callers that might pass a `str`. Invalid UUIDs fail fast at the boundary with a `ValueError`, never reaching the graph store.

## SurrealDB: The Unified Backend

SurrealDB is an alternative that serves as **all three backends** - relational, vector, and graph - in a single database. This simplifies deployment dramatically: one database instead of PostgreSQL + pgvector + Neo4j.

### What It Provides

| Role | SurrealDB Implementation |
|------|-------------------------|
| **Relational** | Document/namespace storage with SurrealQL queries |
| **Vector** | Vector similarity search; the embedded path uses brute-force cosine (KNN `<|K|>` is unreliable in embedded mode), HNSW on the remote path. The remote HNSW DDL is sized from the configured `llm.embedding_dimension` (default 1536) and `hnsw_m` = 24 / `hnsw_ef_construction` = 128 (mirroring the pgvector defaults) - not a fixed 1536 literal. Switching embedders (changed dimension) on a live SurrealDB deployment requires a **manual re-index**: `DEFINE INDEX IF NOT EXISTS` will not re-shape an existing index. |
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

### `RELATES_TO` schema

The `relates_to` RELATION table carries a Khora-owned identity column:

```surql
DEFINE TABLE IF NOT EXISTS relates_to TYPE RELATION SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS rel_id ON relates_to TYPE string;
-- ... namespace_id, relationship_type, weight, properties, ...
DEFINE INDEX IF NOT EXISTS idx_relates_to_rel_id ON relates_to FIELDS rel_id UNIQUE;
```

`rel_id` stores the Khora `Relationship.id` UUID. The `UNIQUE` index on `rel_id` lets `get_relationship(...)` look up edges by the Khora id without colliding with adjacent rows.

### `table:⟨$var⟩` interpolation (PR #770)

SurrealDB does not substitute parameters inside the RecordID-literal shorthand `table:⟨$var⟩` - the `$var` reaches the engine as a literal string. Khora avoids this in two ways:

- Bind via the `_rid()` parameter helper (in `surrealdb/_helpers.py`), which constructs a real `RecordID` object Surreal honours as a parameter.
- Use `(type::thing($var))` in `FOR` loops where a parameter-substituted record-id is needed inside SurrealQL.

Combined with the `rel_id` field + `UNIQUE` index, the graph adapter writes and reads relationships correctly under embedded, memory, and remote modes.

### SurrealDB transactions and batching

SurrealDB supports SurrealQL-level transactions (`BEGIN` / `COMMIT` / `CANCEL`) on the WebSocket connection. The Python client surfaces these as ordinary queries in remote mode. Embedded (`surrealkv://`) and memory (`memory://`) modes raise `UnsupportedFeatureError` on `BEGIN`, so khora preserves the per-statement-atomicity contract there and exposes a batched alternative.

`SurrealDBConnection` ships three primitives:

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

## Replace graph-mirror reconciler (#884 -> #1430)

Re-ingesting a document under an existing `external_id` runs a replace: `StorageCoordinator.replace_document_extraction` commits Postgres first (new chunks + document `COMPLETED`, one transaction), then mirrors the change to the graph backend (entity retire / remap / strip / upsert) outside any transaction. A graph failure in that window leaves PG holding the new document while the graph still holds the previous extraction. #884 made that divergence observable (`khora.storage.replace_document.partial_failure` + a `graph_mirror_failed_after_pg_commit` degradation on `RememberResult.metadata`); #1430 makes it durable-recoverable, following the dream reconciler's shape (#1272):

- **Pending marker.** On a post-commit graph failure the coordinator persists the exact computed graph plan (retire / remap / strip rows plus the net-new entities and relationships, which are not durably stored anywhere else once the mirror fails) as a JSON payload on `documents.graph_mirror_pending` (migration 051; Postgres-only partial index `ix_documents_graph_mirror_pending`). Payload shape: `khora.storage.replace_mirror.build_replace_mirror_payload`.
- **Status stays COMPLETED, flagged.** Per #887, PG data is durable and consistent, so the row is NOT restamped `FAILED` (that would contradict fully-written data and re-trigger self-heal paths) and no new status enum value is introduced (least API breakage). The non-NULL `graph_mirror_pending` column is the "flagged-completed" signal that the graph is known-diverged for this document; `GraphMirrorFailedAfterPGCommitError.pending_persisted` and the degradation's `pending_persisted` key report whether the marker write itself succeeded.
- **Reconciler.** `StorageCoordinator.reconcile_replace_graph_mirror(namespace_id)` replays each pending payload (idempotent: retire verbs match by id, remap/strip are no-ops on re-run, entity upsert MERGEs, relationship create dedup-merges per #1320) and clears the marker on success. It is drained at the start of every `replace_document_extraction` call in the namespace - the same trigger shape as the dream reconciler's `_drain_graph_mirror_pending` (run at apply start) - and can be invoked directly for operator-driven repair. A still-failing marker stays queued, increments the shared `khora.storage.replace_document.partial_failure` counter, and surfaces an ADR-001 degradation on `ReplaceResult.degradations` (merged into `RememberResult.metadata["degradations"]`).
- **Supersession.** A later successful replace of the same document clears any stale marker inside the PG transaction - the old payload describes a superseded extraction and must not be replayed over the newer state.

Markers are only written on graph-backed stacks whose relational backend supports them (PostgreSQL); embedded/unified stacks have no post-commit mirror window and skip the drain.

## Bi-temporal columns are ACTIVE on read (#888 -> #970 -> #1272)

Migrations 033 and 034 added bi-temporal columns to the PostgreSQL schema. Migration 033 adds `valid_to`, `invalidated_at`, `invalidated_by` to `relationships` and `memory_facts` (not `entities`) plus partial indexes `WHERE invalidated_at IS NULL`. Migration 034 adds the same three columns to `chronicle_events`. The `entities` table carries only `valid_from` / `valid_until` (no `invalidated_at` / `invalidated_by`); entities are live-filtered via `_entity_live_filter()` on the `valid_until` window alone. Through #888 these columns were **reserved scaffolding** (written by dream-apply, never filtered on read) because there was no Neo4j tombstone-mirror; a pg-only read filter would have diverged PostgreSQL reads from graph reads of the same data.

The Neo4j tombstone-mirror landed in **#1272 (the #970 definition-of-done)**, so the read filter is now **live on both stores in lockstep**:

- **Written by dream-apply, pg-side, then mirrored to the graph.** Dedupe, prune, fact compaction, and event clustering set the PG soft-delete columns (`valid_to` on a pruned edge, `invalidated_at` on a dedupe self-loop, `valid_until` on the absorbed entity). A post-commit step (`DreamOrchestrator._mirror_dream_op`, modeled on `replace_document_extraction`) then folds those three PG columns onto the graph's single `valid_until` via the #1271 capability-gated verbs (`soft_invalidate_relationships_batch`, `soft_retire_entities_batch`, `rewrite_relationship_endpoints_batch` for incident-edge re-point, #1273). The mirror runs OUTSIDE the apply transaction (eventual consistency); a failure after the PG commit increments `khora.dream.graph_mirror.partial_failure`, records a `Degradation` on `DreamResult.metadata`, and queues the op in `khora_dream_runs.graph_mirror_pending` for the reconciler to re-attempt. The checkpoint advances inside the PG commit before the mirror runs, so resume alone cannot heal a failed mirror - the reconciler (`_drain_graph_mirror_pending`, run at apply start) is the only path that converges the two stores.
- **Filtered on read, both stores.** The pgvector recall list paths apply `_entity_live_filter()` (`valid_until IS NULL OR valid_until > now()`) and `_relationship_live_filter()` (`valid_to IS NULL AND invalidated_at IS NULL AND valid_until window open`); the Neo4j `list_entities` / `list_relationships` paths filter `valid_until` unconditionally. Pruned / merged-self-loop rows are now invisible to recall on a pg+Neo4j stack, byte-identically across both stores.

The cross-store live-set invariant is guarded by `tests/integration/dream/test_neo4j_dream_mirror_integration.py` (a real pg+Neo4j stack); the old reserved-columns tripwire (`tests/unit/test_bitemporal_columns_reserved.py`) was widened to cover `valid_to` + the Neo4j filter, then inverted to assert the filters are now present.

### Community materialization - the GraphRAG payoff (#1276)

The same post-commit mirror path also materializes dream **community summaries** into the graph. The `community_summary` op computes LLM-grounded per-community summaries and persists them to the PG `khora_dream_communities` table; before #1276 that table had **zero readers** (the summaries were computed and discarded for retrieval on every backend). On apply, `DreamOrchestrator._mirror_dream_op` now also dispatches the `vectorcypher_community_summary` op kind through the #1271 capability seam (`supports_dream_mirror()` advertises it) to a new graph verb `materialize_communities_batch`, which MERGEs `:Community` nodes (carrying `summary` + `member_ids` + optional `embedding`) and `[:HAS_MEMBER]` edges from each `:Community` to its member `:Entity` nodes. MERGE keys on `(id, namespace_id)`, so a re-run / reconciler replay never duplicates (idempotent on community id). This leg is **additive** (no soft-deletes), so a mirror failure follows the same `graph_mirror_pending` + `khora.dream.graph_mirror.partial_failure` reconciler path as the soft-delete legs.

The summaries are then queryable at recall via the read-only readers `get_communities` (per namespace) and `get_entity_communities` (anchored to a recall hit's entity set), exposed on `GraphBackendProtocol`, the `StorageCoordinator`, and the top-level `Khora.get_communities` / `Khora.get_entity_communities` accessors. Backends without native support advertise nothing and record a structured skip (no silent divergence); the readers default to an empty list (read-only, never raise). Guarded by `tests/unit/dream/test_dream_community_mirror.py`, `tests/unit/storage/backends/test_graph_dream_verbs.py`, and the live pg+Neo4j cases in `test_neo4j_dream_mirror_integration.py`.

### Graph-side undo - no silent half-revert (#1275)

`dream_undo` reverses an applied dream op from its `undo.json` snapshot. Before #1275 it reversed only the PG soft-deletes; the forward graph mirror (#1272/#1273) was never undone, so undo was a **half-revert** that re-diverged the two stores - PG returned to live, the graph kept the merged shape. #1275 adds a graph-side reverse so undo restores PG and graph to **identical pre-apply live sets**.

Each forward mirror verb has a matching reverse on `GraphBackendProtocol` (Neo4j native impl; other backends inherit the same capability-gated `DreamBackendUnsupported` default as the forward verbs, so a backend without a native reverse records a skip rather than diverging):

- `soft_invalidate_relationships_batch` → `restore_relationships_batch` (clears graph `valid_until` on pruned edges / dedupe self-loops)
- `soft_retire_entities_batch` → `restore_entities_batch` (clears the absorbed node's `valid_until` / `version_valid_to` **and** detach-deletes the `:EntityVersion` / `[:SUPERSEDES]` snapshot the forward mirror created)
- `rewrite_relationship_endpoints_batch` → `restore_relationship_endpoints_batch` (re-points incident edges back from canonical → absorbed, using the PRE-rewrite endpoints recorded in `previous_relationships`)

After the PG reverse commits, `dream_undo` runs `_unmirror_dream_op` (post-commit, eventual-consistency, the same shape as the forward `_mirror_dream_op`): it reuses `extract_mirror_targets` to know exactly which entities / self-loops / edges the forward mirror touched, then inverts each leg via `unmirror_targets`. The reverse verbs are idempotent by id (a second undo matches nothing and does not re-diverge the graph). A reverse-mirror failure does **not** roll back the committed PG reverse; it increments `khora.dream.graph_unmirror.partial_failure` and logs an ADR-001 degradation so the divergence is observable. Guarded by `tests/integration/dream/test_neo4j_dream_undo_integration.py` (live pg+Neo4j) plus the reverse-extraction / verb unit cases in `tests/unit/dream/test_dream_graph_mirror.py` and `tests/unit/dream/test_dream_undo.py`.

### Backend dream-apply notes

Dream-apply behavior varies by graph backend:

- **Neo4j / Memgraph (default stack).** All PG soft-delete outcomes are mirrored post-commit via `DreamOrchestrator._mirror_dream_op` (eventual-consistency). A failed mirror queues the op in `khora_dream_runs.graph_mirror_pending`; the reconciler (`_drain_graph_mirror_pending`, run at apply start) retries. Mirror verbs: `soft_invalidate_relationships_batch` (prune/reconcile), `soft_retire_entities_batch` (dedupe), `rewrite_relationship_endpoints_batch` (incident-edge re-point, #1293), `materialize_communities_batch` (community summary). Undo mirrors via `_unmirror_dream_op` (same eventual-consistency shape).

- **AGE (PostgreSQL extension).** Same capability set as Neo4j for flat soft-delete ops (`prune_edges`, `contradiction_reconcile`) - see `age.supports_dream_mirror()`. Crucially, AGE lives inside PostgreSQL, so since #1307/#1310 the orchestrator folds the mirror into the same apply transaction (`mirror_in_transaction()` returns `True`): no `graph_mirror_pending` reconciler lag, no eventual-consistency window. Entity-version ops (`dedupe_entities`) and relabel (`normalize_schema`) are not supported on AGE (no versioning primitive); those op kinds are recorded as a structured skip.

- **Neptune.** Capability-gated mirror for `prune_edges` and `contradiction_reconcile` via `soft_invalidate_relationships_batch` (Bolt protocol SET by id, #1279/#1298). Entity-version ops (`dedupe_entities`, `normalize_schema`) are not supported (no versioning primitive); structured skip recorded. Mirror is post-commit eventual-consistency (same reconciler path as Neo4j).

- **SurrealDB-unified.** Dream-apply runs via `_apply_one_op_surrealdb` in the orchestrator (#1280/#1300). The unified store is its own mirror - no separate post-commit graph mirror is dispatched. Incident-edge re-point (`rewrite_relationship_endpoints_batch`) is handled within the SurrealDB apply path (#1304).

- **sqlite_lance (embedded / test stack).** Single-store: the graph layer reads the same SQLite file, so no mirror is needed. `vectorcypher_dedupe_entities` and `vectorcypher_prune_edges` run on this path as of #1277/#1299. Postgres-only ops (`centroid_recompute`, `source_chunk_ids_gc`, `contradiction_reconcile`) are skipped via `_POSTGRES_ONLY_OP_KINDS` gate.

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

### `sqlite_lance` defense-in-depth (PR #769)

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
await ensure_hnsw_indexes(engine)

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

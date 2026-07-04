# Skeleton Construction Engine

The **Skeleton Construction engine** is a temporal-first memory engine optimized for event streams, chat histories, and time-sensitive data. Unlike the VectorCypher engine which focuses on knowledge graph construction, Skeleton Construction prioritizes temporal relationships and cost-efficient retrieval through skeleton-based indexing.

## When to Use Skeleton Construction

Choose the Skeleton Construction engine when:

- **Time matters most**: Chat logs, event streams, meeting transcripts, logs
- **Cost is a concern**: 5-10x fewer LLM calls via skeleton indexing
- **Infrastructure is limited**: PostgreSQL-only (no graph database required)
- **Freshness is critical**: Bi-temporal model tracks both event time and ingestion time
- **Structured filters needed**: Filter by author, channel, tags, time ranges

Choose VectorCypher instead when:

- Building long-term knowledge bases with rich entity relationships
- Graph traversal and entity exploration are primary use cases
- Upfront extraction cost is acceptable for better retrieval quality (use `engine_kwargs={"vectorcypher_config": VectorCypherConfig(skeleton_core_ratio=1.0)}` for full 100% extraction)

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       SkeletonConstructionEngine                             │
│                      remember() / recall() / forget()                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│  │  SkeletonIndexer │  │ TimeHierarchy    │  │ TemporalEdge     │            │
│  │  (PageRank core) │  │ Builder          │  │ Storage          │            │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘            │
│           │                     │                     │                      │
├───────────┴─────────────────────┴─────────────────────┴──────────────────────┤
│                          TemporalVectorStore                                 │
│         (pgvector | weaviate | turbopuffer | sqlite_lance | surrealdb)       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                      ┌──────────────────┐              │
│  │   PostgreSQL     │                      │   Weaviate       │              │
│  │   + pgvector     │         OR           │   (optional)     │              │
│  │   + BRIN indexes │                      │                  │              │
│  └──────────────────┘                      └──────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

> **Note:** `TemporalEdge Storage` and `TimeHierarchy Builder` shown above exist as code modules but are not yet wired into the engine's ingest/recall pipeline. Temporal filtering via `occurred_at` on chunks works through the pgvector backend directly.

### Core Design Principles

1. **Bi-Temporal Model**: Every piece of data has two timestamps:
   - `occurred_at`: When the event actually happened
   - `ingested_at`: When we learned about it

2. **Hierarchical Time Graph**: Time organized as Year → Quarter → Month → Week → Day for efficient range queries

3. **Skeleton-Based Indexing**: PageRank identifies ~10% "core" chunks for LLM extraction; others use keyword-based retrieval

4. **Lazy Entity Expansion**: Non-core chunks are expanded on-demand during retrieval, not upfront

## Key Components

### SkeletonConstructionEngine (`src/khora/engines/skeleton/engine.py`)

The main engine class implementing `MemoryEngineProtocol`:

```python
from khora import Khora

# Use Skeleton Construction engine explicitly
async with Khora("postgresql://...", engine="skeleton") as kb:
    ns = await kb.create_namespace()

    # Store with temporal context
    result = await kb.remember(
        "Meeting notes from quarterly review",
        title="Q1 Review",
        namespace=ns.namespace_id,
        metadata={
            "author": "alice@company.com",
            "channel": "leadership",
            "occurred_at": "2024-01-15T10:00:00Z"
        },
        entity_types=["PERSON", "EVENT"],
        relationship_types=["PARTICIPATES_IN"],
    )

    # Recall with temporal filters
    results = await kb.recall(
        "What decisions were made?",
        namespace=ns.namespace_id,
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 3, 31),
    )
    # Note: structured filters (author, channel) and per-call hybrid
    # alpha aren't exposed on the public facade today.
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `remember()` | Store content with deduplication and checksum tracking |
| `recall()` | Retrieve memories with temporal filtering and hybrid search |
| `forget()` | Remove a memory from the engine |
| `remember_batch()` | Batch ingestion with parallel processing |
| `stats()` | Get document/chunk/entity counts |

### TemporalEdgeStorage (`src/khora/engines/skeleton/temporal_edges.py`)

Manages bi-temporal edges with conflict detection, inspired by [Graphiti](https://github.com/getzep/graphiti):

```python
@dataclass
class TemporalEdge:
    id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str

    # Bi-temporal fields
    occurred_at: datetime      # When the fact happened
    ingested_at: datetime      # When we learned about it
    valid_from: datetime       # Validity window start
    valid_until: datetime      # Validity window end

    # Conflict tracking
    is_valid: bool = True
    invalidated_by_id: UUID | None = None
    confidence: float = 1.0
```

**Conflict Resolution:**

For exclusive relationships (WORKS_FOR, REPORTS_TO, MARRIED_TO), new edges automatically invalidate older conflicting edges:

```python
# Alice works for Acme (Jan 2024)
edge1 = await storage.create_edge(
    alice_id, acme_id, "WORKS_FOR",
    namespace_id=ns_id,
    occurred_at=jan_2024,
)

# Alice now works for Beta (Mar 2024) - edge1 is automatically invalidated
edge2 = await storage.create_edge(
    alice_id, beta_id, "WORKS_FOR",
    namespace_id=ns_id,
    occurred_at=mar_2024,
)
```

<!-- TODO(docs): TemporalEdgeStorage is not wired into the skeleton engine's ingest/recall path (see status note above). Verify the create_edge/get_valid_at signatures shown here against current code if this module is ever revived. -->


### TimeHierarchyBuilder (`src/khora/engines/skeleton/time_hierarchy.py`)

Implements TG-RAG-inspired hierarchical time navigation:

```
2024 (year)
├── Q1 2024 (quarter)
│   ├── January 2024 (month)
│   │   ├── Week 1 (week)
│   │   │   ├── 2024-01-01 (day)
│   │   │   ├── 2024-01-02 (day)
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

**Benefits:**

- Fast range queries ("What happened in Q1 2024?")
- Drill-down from coarse to fine granularity
- Automatic ancestor creation on demand
- Edge/entity counts aggregated at each level

### SkeletonIndexer (`src/khora/engines/skeleton/skeleton.py`)

KET-RAG-inspired PageRank-based core chunk selection:

```python
# core_ratio is a constructor parameter (default 0.1 = top 10% as "core")
indexer = SkeletonIndexer(core_ratio=0.1)

# Add chunks (fast, no LLM) - add_chunk takes a chunk object
for chunk in chunks:
    indexer.add_chunk(chunk)

# Build skeleton - identifies core chunks, returns their IDs
core_chunk_ids = indexer.build_skeleton()

# Only core chunks get LLM extraction
for chunk_id in core_chunk_ids:
    await extract_entities(chunk_id)  # LLM call

# Non-core chunks use keyword-based retrieval
# Expanded lazily on-demand during recall()
```

**How It Works:**

1. Extract keywords from all chunks (TF-based, no LLM)
2. Build keyword-chunk bipartite graph
3. Calculate IDF scores for keywords
4. Build chunk-to-chunk edges via shared keywords
5. Run PageRank to identify central chunks
6. Select top N% (default 10%) as "core"

**Cost Savings:**

| Approach | LLM Calls per 1000 docs |
|----------|-------------------------|
| Full extraction (VectorCypher with `skeleton_core_ratio=1.0`) | ~1000 |
| Default VectorCypher (selective, 50%) | ~500 |
| Skeleton indexing (Skeleton) | ~100 |

### LazyEntityExpander (`src/khora/engines/skeleton/skeleton.py`)

On-demand entity extraction for non-core chunks:

```python
expander = LazyEntityExpander(skeleton_indexer)

# During recall(), if a non-core chunk is highly relevant
if not skeleton_indexer.is_core_chunk(chunk_id):
    entities = await expander.maybe_expand(chunk_id, chunk_content)
    # Returns keyword-based pseudo-entities or triggers full extraction
```

## Backend Options

### PostgreSQL + pgvector (Default)

Single-infrastructure deployment using PostgreSQL extensions:

```python
engine = SkeletonConstructionEngine(config, backend="pgvector")
```

**Features:**

- HNSW index for vector similarity search
- GIN index for BM25 full-text search (tsvector)
- BRIN index for temporal range queries (99% space savings)
- Native hybrid search via SQL

**Schema (`khora_chunks` table):**

```sql
CREATE TABLE khora_chunks (
    id UUID PRIMARY KEY,
    namespace_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    occurred_at TIMESTAMP WITH TIME ZONE,  -- BRIN indexed
    created_at TIMESTAMP WITH TIME ZONE,
    source_system VARCHAR(255),
    author VARCHAR(255),
    channel VARCHAR(255),
    tags TEXT[],
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB,
    content_tsv TSVECTOR  -- GIN indexed, auto-generated
);

-- BRIN index for time-series data (very compact)
CREATE INDEX idx_chunks_occurred_at_brin ON khora_chunks USING BRIN (occurred_at);

-- HNSW for vector similarity
CREATE INDEX idx_chunks_embedding_hnsw ON khora_chunks
    USING hnsw (embedding vector_cosine_ops);

-- GIN for full-text search
CREATE INDEX idx_chunks_content_tsv ON khora_chunks USING GIN (content_tsv);
```

### Weaviate (Advanced)

For horizontal scaling and native multi-tenancy:

```python
# Self-hosted (compose.yaml `weaviate` profile)
engine = SkeletonConstructionEngine(
    config,
    backend="weaviate",
    weaviate_url="http://localhost:8090",
)
```

**Auth and Weaviate Cloud** (issue #783). The
backend accepts a `WeaviateBackendConfig` in place of the URL string
for cloud, authenticated, or custom-port deployments:

```python
from khora.storage.temporal.weaviate import WeaviateBackendConfig

# Weaviate Cloud
cloud_config = WeaviateBackendConfig(
    cluster_url="https://my-cluster.weaviate.network",
    api_key="...",  # SecretStr also accepted
)
engine = SkeletonConstructionEngine(config, backend="weaviate", weaviate_url=cloud_config)

# Self-hosted with API-key auth + non-default gRPC port
local_auth = WeaviateBackendConfig(
    url="http://localhost:8090",
    api_key="local-key",
    grpc_port=50061,            # compose.yaml offset
)
engine = SkeletonConstructionEngine(config, backend="weaviate", weaviate_url=local_auth)
```

`url` and `cluster_url` are mutually exclusive; `cluster_url` requires
`api_key`. The `weaviate_url` constructor argument keeps the legacy
string-only contract for back-compat (wraps into a default
`WeaviateBackendConfig(url=...)`).

**Async client.** The backend uses `weaviate.use_async_with_local /
use_async_with_custom / use_async_with_weaviate_cloud` under the hood
so the Skeleton event loop does not block on Weaviate I/O.

**Features:**

- Native hybrid search (alpha blending)
- Multi-tenant isolation (namespace = tenant)
- Horizontal scaling
- Built-in BM25 + vector fusion
- API-key auth + Weaviate Cloud

**Tests.** Unit tests exercise the async client via mocks. Integration
tests against a real cluster live in
`tests/integration/test_weaviate_async_integration.py` and are gated
behind `WEAVIATE_INTEGRATION_TEST=1`; CI provisions a Weaviate
service-container side-car for the `weaviate-integration` job.

### Turbopuffer (Serverless / Large-Scale)

Pay-per-query serverless vector + BM25 store backed by object storage.
Pitched at scale tiers above what pgvector / self-hosted Weaviate
make economical (2.5T vectors in production at Cursor, Notion):

```python
from khora.storage.temporal.turbopuffer import TurbopufferBackendConfig

cfg = TurbopufferBackendConfig(
    api_key="tpuf_...",            # str or SecretStr
    region="gcp-us-central1",      # see https://turbopuffer.com/docs/regions
)
engine = SkeletonConstructionEngine(
    config,
    backend="turbopuffer",
    turbopuffer_config=cfg,
)
```

The constructor also accepts a bare API-key string for back-compat with
the other backends' shape.

**Mapping:** one turbopuffer namespace per khora `namespace_id`, named
`f"khora_{namespace_id.hex}"`. The `khora_` prefix lets
`GET /v2/namespaces?prefix=khora_` enumerate only this deployment's
tenants (useful for cross-tenant audit + GDPR delete-by-tenant).

**Hybrid search is client-side RRF.** turbopuffer doesn't expose a
server-blended linear alpha score; it offers a multi-query batch that
the client fuses with Reciprocal Rank Fusion. Consequence: the
`hybrid_alpha` parameter is a **no-op** on this backend - the fusion is
rank-based, not score-weighted. If you need true server-blended linear
alpha scores, stay on Weaviate.

**Filter expressibility:** every `ChunkTemporalFilter` predicate compiles
to turbopuffer's filter DSL (`Eq`, `Gte`, `Lte`, `Lt`, `In`, `Contains`,
`ContainsAny`). The one workaround: ALL-tags semantics fold into an
`And` of N `Contains` clauses (turbopuffer has no native `ContainsAll`).

**Recall filters are unsupported.** The deterministic recall-filter API
(the `filter_ast` argument to `search`) is **not** implemented on
turbopuffer. Passing a non-None filter RAISES
`RecallFilterUnsupportedError` rather than silently returning unfiltered
results - the backend fails loud instead of dropping the filter on the
floor. Native server-side filter pushdown is a possible future
enhancement.

**Features:**

- Multi-region serverless (no infrastructure to run)
- Native BM25 + vector via multi-query batch
- Pay-per-write (per-WU) + pay-per-read (per-RU) + storage at object-store rates
- Apache-2.0 SDK; SaaS backend (closed-source)

**When NOT to pick it:**

- You need an embedded / offline path. turbopuffer is SaaS-only; sqlite_lance is the right answer.
- You need server-blended alpha hybrid. Weaviate.
- Corpora under ~10M chunks with modest query volume. pgvector is cheaper and faster at that tier.

**Crossover guidance.** At a 1 GB namespace and 1M queries/month
turbopuffer runs roughly ~$8 (queries) + ~$0.33 (storage) + write
amortisation - call it ~$25/month. The same workload on
self-hosted Postgres+pgvector is ~$30-50/month in VM + storage cost
plus ops overhead. Crossover lands somewhere around 50M chunks
with sustained query volume; below that pgvector wins, above that
turbopuffer wins. Numbers as of 2026-05-24 - re-check
[pricing](https://turbopuffer.com/pricing) before adopting.

**Install:** `pip install khora[turbopuffer]`.

**Tests.** A suite of unit tests exercises the SDK via a fake `turbopuffer`
module injected into `sys.modules` (no real network or API key needed).
Real-cluster integration tests are gated behind
`TURBOPUFFER_INTEGRATION_TEST=1` + a `TURBOPUFFER_API_KEY` env var; not
wired into CI yet (would need a repo secret for the sandbox key).

## Query Capabilities

### Temporal Filtering

```python
from khora.core.temporal import ChunkTemporalFilter

# By time range
results = await engine.recall(
    "project updates",
    namespace_id=namespace_id,
    temporal_filter=ChunkTemporalFilter(
        occurred_after=datetime(2024, 1, 1),
        occurred_before=datetime(2024, 3, 31),
    )
)

# By structured fields
results = await engine.recall(
    "decisions",
    namespace_id=namespace_id,
    temporal_filter=ChunkTemporalFilter(
        author="alice@company.com",
        channel="leadership",
        tags=["important", "decision"]
    )
)

# Combined
results = await engine.recall(
    "Q1 decisions",
    namespace_id=namespace_id,
    temporal_filter=ChunkTemporalFilter(
        occurred_after=datetime(2024, 1, 1),
        occurred_before=datetime(2024, 3, 31),
        author="alice@company.com",
        channel="leadership"
    )
)
```

### Hybrid Search

Combines vector similarity with BM25 keyword matching using Reciprocal Rank Fusion (RRF):

```python
# Adjust the blend
results = await engine.recall(
    query,
    namespace_id=namespace_id,
    hybrid_alpha=0.7,  # 0.7 * vector + 0.3 * BM25
)
```

| `hybrid_alpha` | Behavior |
|----------------|----------|
| `1.0` | Pure vector search (semantic) |
| `0.7` | Balanced, slightly favor semantic |
| `0.5` | Equal weight |
| `0.3` | Balanced, slightly favor keywords |
| `0.0` | Pure BM25 (keyword) |

### Search Modes

```python
from khora import SearchMode

# Vector-only (semantic similarity)
results = await engine.recall(query, namespace_id=ns_id, mode=SearchMode.VECTOR)

# Hybrid (vector + BM25 with RRF)
results = await engine.recall(query, namespace_id=ns_id, mode=SearchMode.HYBRID)

# Keyword-only (pure BM25, supported on all backends including weaviate/turbopuffer)
results = await engine.recall(query, namespace_id=ns_id, mode=SearchMode.KEYWORD)

# Note: SearchMode.GRAPH is not supported in Skeleton Construction engine
# Use VectorCypher engine for graph-based queries
```

### Deterministic recall filters

The deterministic recall-filter API (the `filter_ast` argument to the backend
`search`) is supported on the SurrealDB backend, but only over the system keys the
`temporal_chunk` table backs with a real column: the two datetime keys
`occurred_at` and `created_at`, plus any `metadata.<path>`. The other eight system
keys (`source_name`, `source_type`, `source_url`, `source_timestamp`,
`external_id`, `content_type`, `source`, `title`) are denormalized document fields
that are **not** columns on `temporal_chunk`. A filter on one of them RAISES
`RecallFilterUnsupportedError` rather than silently returning nothing: on the
SCHEMAFULL table the missing field reads as absent, and because SurrealQL's
absent-compare is total-false (`NONE = x` → `false`) the predicate would drop every
row. Failing loud surfaces the unsupported key instead of returning a quietly empty
result set.

## Configuration

### Via KhoraConfig

```python
from khora.config import KhoraConfig, QueryConfig

config = KhoraConfig(
    database_url="postgresql://localhost/khora",
    query=QueryConfig(
        recency_decay_days=7,
    ),
)
# Engine selection is a Khora() constructor kwarg, not a config field:
#     kb = Khora(config, engine="skeleton")
# `hybrid_alpha` is a per-call argument to kb.recall(...) only; there is
# no global env-var or config-field equivalent.
```

### Via Environment Variables

```bash
KHORA_DATABASE_URL=postgresql://localhost/khora
KHORA_QUERY_RECENCY_DECAY_DAYS=30
```

Engine and backend selection are constructor-only - pass
`engine="skeleton"` and `engine_kwargs={"backend": "pgvector"}` to
`Khora(...)`. The `hybrid_alpha` blend weight is a per-call argument
to `kb.recall(...)` and has no env-var equivalent.

### Via YAML

```yaml
# config/skeleton/khora.yaml
query:
  recency_decay_days: 7

temporal:
  default_lookback_days: 90
  hierarchy_enabled: true
```

## Performance Characteristics

| Metric | Skeleton Construction | VectorCypher |
|--------|----------------------|--------------|
| LLM calls per 1000 docs | ~100 | ~500 (default) / ~1000 (`skeleton_core_ratio=1.0`) |
| Ingestion latency | Lower | Higher |
| Infrastructure | PostgreSQL only | PostgreSQL + Neo4j |
| Temporal queries | Native (bi-temporal) | Per-category |
| Entity relationships | On-demand | Pre-computed |
| Graph traversal | Limited | Full support |

## Related Documentation

- [Engine Comparison](engine-comparison.md) - Detailed VectorCypher vs Skeleton comparison
- [Temporal Model](temporal-model.md) - Deep dive into bi-temporal design
- [Skeleton Indexing](skeleton-indexing.md) - PageRank-based core selection
- [Hybrid Search](hybrid-search.md) - Vector + BM25 fusion details

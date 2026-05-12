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
- Upfront extraction cost is acceptable for better retrieval quality (use `engine_kwargs={"skeleton_core_ratio": 1.0}` for full 100% extraction)

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
│                    (pgvector backend | weaviate backend)                     │
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

> **Note (v0.2.3):** `TemporalEdge Storage` and `TimeHierarchy Builder` shown above exist as code modules but are not yet wired into the engine's ingest/recall pipeline. Temporal filtering via `occurred_at` on chunks works through the pgvector backend directly.

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
    # Store with temporal context
    result = await kb.remember(
        "Meeting notes from quarterly review",
        title="Q1 Review",
        metadata={
            "author": "alice@company.com",
            "channel": "leadership",
            "occurred_at": "2024-01-15T10:00:00Z"
        }
    )

    # Recall with temporal and structured filters
    results = await kb.recall(
        "What decisions were made?",
        temporal_filter={
            "occurred_after": "2024-01-01",
            "occurred_before": "2024-03-31",
            "author": "alice@company.com"
        },
        hybrid_alpha=0.7  # 70% vector, 30% BM25
    )
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
edge1 = await storage.create_edge(alice_id, acme_id, "WORKS_FOR", occurred_at=jan_2024)

# Alice now works for Beta (Mar 2024) - edge1 is automatically invalidated
edge2 = await storage.create_edge(alice_id, beta_id, "WORKS_FOR", occurred_at=mar_2024)
```

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
# Add chunks (fast, no LLM)
for chunk in chunks:
    indexer.add_chunk(chunk.id, chunk.content)

# Build skeleton - identifies top 10% as "core"
core_chunk_ids = indexer.build_skeleton(core_ratio=0.1)

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
| Default VectorCypher (selective, 70%) | ~700 |
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
engine = SkeletonConstructionEngine(
    config,
    backend="weaviate",
    weaviate_url="http://localhost:8080"
)
```

**Features:**

- Native hybrid search (alpha blending)
- Multi-tenant isolation (namespace = tenant)
- Horizontal scaling
- Built-in BM25 + vector fusion

## Query Capabilities

### Temporal Filtering

```python
from khora.engines.skeleton.backends import TemporalFilter

# By time range
results = await engine.recall(
    "project updates",
    namespace_id,
    temporal_filter=TemporalFilter(
        occurred_after=datetime(2024, 1, 1),
        occurred_before=datetime(2024, 3, 31),
    )
)

# By structured fields
results = await engine.recall(
    "decisions",
    namespace_id,
    temporal_filter=TemporalFilter(
        author="alice@company.com",
        channel="leadership",
        tags=["important", "decision"]
    )
)

# Combined
results = await engine.recall(
    "Q1 decisions",
    namespace_id,
    temporal_filter=TemporalFilter(
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
    namespace_id,
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
results = await engine.recall(query, ns_id, mode=SearchMode.VECTOR)

# Hybrid (vector + BM25 with RRF)
results = await engine.recall(query, ns_id, mode=SearchMode.HYBRID)

# Note: SearchMode.GRAPH is not supported in Skeleton Construction engine
# Use VectorCypher engine for graph-based queries
```

## Configuration

### Via KhoraConfig

```python
from khora.config import KhoraConfig

config = KhoraConfig(
    database_url="postgresql://localhost/khora",
    engine=EngineConfig(
        name="skeleton",
        backend="pgvector",  # or "weaviate"
    ),
    query=QueryConfig(
        hybrid_alpha=0.7,
        recency_decay_days=30,
    ),
)
```

### Via Environment Variables

```bash
KHORA_DATABASE_URL=postgresql://localhost/khora
KHORA_ENGINE_NAME=skeleton
KHORA_ENGINE_BACKEND=pgvector
KHORA_QUERY_HYBRID_ALPHA=0.7
KHORA_QUERY_RECENCY_DECAY_DAYS=30
```

### Via YAML

```yaml
# config/skeleton/khora.yaml
engine:
  name: skeleton
  backend: pgvector

query:
  hybrid_alpha: 0.7
  recency_decay_days: 30

temporal:
  default_lookback_days: 90
  hierarchy_enabled: true
```

## Performance Characteristics

| Metric | Skeleton Construction | VectorCypher |
|--------|----------------------|--------------|
| LLM calls per 1000 docs | ~100 | ~700 (default) / ~1000 (`skeleton_core_ratio=1.0`) |
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
- [References](../REFERENCES.md) - Research papers and inspirations

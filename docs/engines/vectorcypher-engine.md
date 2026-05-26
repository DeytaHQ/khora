# VectorCypher Engine

The **VectorCypher engine** is a hybrid retrieval engine that combines vector similarity search (pgvector) with Cypher graph traversal (Neo4j). Inspired by Graph RAG 2026 and HippoRAG 2, it excels at complex multi-hop queries while maintaining efficient simple lookups through intelligent query routing.

## When to Use VectorCypher

Choose the VectorCypher engine when:

- **Multi-hop queries matter**: "Who works on deals with companies that Alex mentioned?"
- **Graph traversal is essential**: Navigate organizational hierarchies, deal chains, team structures
- **Relationship discovery is key**: Find implicit connections across data sources
- **Neo4j is available**: VectorCypher requires Neo4j (not optional)

Choose VectorCypher especially when:

- **Temporal reasoning is needed**: "What is she currently working on?", "What happened most recently?"
- **Mixed query complexity**: Automatic routing + temporal detection adapts retrieval per query

Choose Skeleton Construction instead when:

- **Cost is the primary concern**: Skeleton uses 5-10x fewer LLM calls
- **No Neo4j available**: VectorCypher requires Neo4j
- **Simple infrastructure preferred**: Skeleton works with PostgreSQL only

For comprehensive extraction over 100% of chunks, pass `engine_kwargs={"skeleton_core_ratio": 1.0}` - VectorCypher's KET-RAG selectivity defaults to the top 70% of chunks but accepts a 1.0 override that extracts from every chunk.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          VectorCypherEngine                                  │
│                      remember() / recall() / forget()                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐      │
│  │ QueryComplexity    │  │ VectorCypher       │  │ DualNodeManager    │      │
│  │ Router             │  │ Retriever          │  │ (HippoRAG 2)       │      │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘      │
│            │                       │                       │                 │
│  ┌─────────┴────────────┐  ┌──────┴───────────────────────┴────────────┐     │
│  │   Query Analysis     │  │              Retrieval Pipeline            │    │
│  │  SIMPLE → Vector     │  │   Vector Search → Cypher Expand → Fusion  │     │
│  │  MODERATE → Shallow  │  │                                           │     │
│  │  COMPLEX → Deep      │  │                                           │     │
│  └──────────────────────┘  └───────────────────────────────────────────┘     │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                             Storage Layer                                    │
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐    │
│  │   PostgreSQL     │      │    pgvector      │      │     Neo4j        │    │
│  │   (Documents,    │      │   (Embeddings,   │      │  (Entity nodes,  │    │
│  │    Metadata)     │      │  Chunk vectors)  │      │   Chunk nodes,   │    │
│  │                  │      │                  │      │   MENTIONED_IN)  │    │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘    │
│         REQUIRED                 REQUIRED                 REQUIRED           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Dual-Node Architecture**: Inspired by HippoRAG 2, maintains both Chunk and Entity nodes in Neo4j, linked via `MENTIONED_IN` relationships

2. **Query Routing**: Intelligent classification routes queries to optimal search paths (vector-only for simple queries, full VectorCypher for complex)

3. **Skeleton-Based Extraction**: Only core chunks (identified via PageRank, default 70%) get full LLM entity extraction, balancing cost and quality

4. **RRF Fusion**: Reciprocal Rank Fusion combines vector and graph results with configurable weights

5. **Bi-Temporal Support**: Inherits temporal model from Skeleton Construction (`occurred_at` vs `ingested_at`)

## Key Components

### VectorCypherEngine (`src/khora/engines/vectorcypher/engine.py`)

The main engine class implementing `MemoryEngineProtocol`:

```python
from khora import Khora

# Use VectorCypher engine explicitly
async with Khora("postgresql://...", engine="vectorcypher") as kb:
    ns = await kb.create_namespace()

    # Store with temporal context
    result = await kb.remember(
        "Meeting notes from Q1 planning with John",
        namespace=ns.namespace_id,
        title="Q1 Planning",
        metadata={
            "author": "alice@company.com",
            "occurred_at": "2024-01-15T10:00:00Z"
        },
        entity_types=["PERSON", "EVENT"],
        relationship_types=["PARTICIPATES_IN"],
    )

    # Recall with graph-enhanced retrieval. Per-call graph depth isn't
    # exposed on kb.recall(); configure globally via
    # `KhoraConfig.query.max_graph_depth`.
    results = await kb.recall(
        "What did we discuss with John about planning?",
        namespace=ns.namespace_id,
    )
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `remember()` | Store content with deduplication and skeleton-based extraction |
| `recall()` | Retrieve memories with VectorCypher hybrid search |
| `forget()` | Remove a memory (cleans both pgvector and Neo4j) |
| `remember_batch()` | Batch ingestion with parallel processing |
| `find_related_entities()` | Graph traversal to find related entities |
| `stats()` | Get document/chunk/entity counts |

### RecallResult Context

`recall()` returns `RecallResult` objects whose typed projections expose:
- **`chunks`** - Matching text passages as `RecallChunk` entries (`chunk.content`)
- **`entities`** - Entities mentioned in matching chunks
- **`relationships`** - Connections between entities in the result set
- **`documents`** - Full `DocumentProjection` rows for every document referenced by a chunk, entity, or relationship (always populated; see [Source Document Population](#source-document-population))

Callers that need a flat context string for an LLM render one with the
public `khora.context_text(result, max_chunks=...)` helper:

```python
from khora import Khora, context_text

result = await kb.recall("query", namespace=ns_id)
prompt_context = context_text(result, max_chunks=5)
```

### Source Document Population

`recall()` always returns a `RecallResult` whose `documents` list holds full `DocumentProjection` rows for every document referenced by a chunk, entity, or relationship in the result - this is a producer-enforced invariant (see #761). Khora batch-fetches `DocumentSource` metadata after the engine returns (chunked at 1,000 IDs) and replaces the engine's lightweight stubs in place. The engine itself uses the namespace-scoped coordinator facade for that lookup, so cross-namespace ids never leak through.

```python
result = await kb.recall("query", namespace=ns_id)
docs_by_id = {d.id: d for d in result.documents}
for chunk in result.chunks:
    print(docs_by_id[chunk.document_id].title)
```

Entity-read methods (`get_entity()`, `list_entities()`, `find_related_entities()`, `search_entities()`) accept `include_sources: bool = False` to opt-in to per-entity `source_documents` population. All four require `namespace_id=` (kwarg-only) on every call - the IDOR close-out (#769) enforces this at the Protocol level on every storage backend.

### VectorCypherRetriever (`src/khora/engines/vectorcypher/retriever.py`)

Implements the hybrid retrieval pipeline:

```python
@dataclass
class RetrieverConfig:
    # Graph traversal settings
    default_depth: int = 2
    max_depth: int = 4
    max_entry_entities: int = 10

    # Adaptive depth settings
    adaptive_depth_enabled: bool = True
    adaptive_depth_high_entity_threshold: int = 10  # Shallow if >= 10 entities
    adaptive_depth_low_entity_threshold: int = 2    # Deeper if <= 2 entities

    # Fusion settings
    rrf_k: int = 60
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    # Per-complexity fusion overrides
    simple_vector_weight: float = 0.8
    simple_graph_weight: float = 0.2
    complex_vector_weight: float = 0.4
    complex_graph_weight: float = 0.6

    # Temporal settings
    recency_weight: float = 0.2
    recency_decay_days: int = 30
    recency_decay_type: str = "exponential"  # "exponential" or "linear"

    # Search thresholds
    min_entity_similarity: float = 0.3
    hybrid_alpha: float = 0.7
    coherence_weight: float = 0.0  # Weight for cross-chunk coherence scoring

    # Entity expansion
    lazy_entity_expansion: bool = False  # Defer entity expansion until needed

    # Limits
    max_chunks: int = 50
    max_entities: int = 30
```

**Retrieval Pipeline:**

1. **Route Query**: Classify as SIMPLE, MODERATE, or COMPLEX
2. **Detect Temporal Signal**: Classify query into a temporal category (see [Temporal Detection](#temporal-detection))
3. **Embed Query**: Generate query embedding via LiteLLM
4. **Vector Search**: Find entry entities via pgvector similarity (with `hnsw.ef_search = 200`)
5. **Cypher Expand**: Traverse graph to find related entities (if complex)
6. **Fetch Chunks**: Get chunks via `MENTIONED_IN` relationships, with optional temporal sort
7. **RRF Fusion**: Combine vector and graph results
8. **Recency Boost**: Apply temporal boosting with category-specific weights and decay

### QueryComplexityRouter (`src/khora/engines/vectorcypher/router.py`)

Routes queries to optimal search paths:

```python
class QueryComplexity(Enum):
    SIMPLE = "simple"    # Vector-only search (fastest)
    MODERATE = "moderate" # Shallow graph (depth=1)
    COMPLEX = "complex"  # Full VectorCypher (depth=2-3)
```

**Routing Heuristics:**

| Pattern | Complexity | Examples |
|---------|------------|----------|
| Simple questions | SIMPLE | "What is X?", "Who is Y?" |
| Relationship keywords | MODERATE+ | "related to", "connected with" |
| Comparison keywords | COMPLEX | "compare", "difference between" |
| Multi-hop keywords | COMPLEX | "through", "chain", "path" |
| Multiple entities | COMPLEX | Queries mentioning 2+ entities |

```python
# Examples of query routing
"What is the company policy?" → SIMPLE (vector-only)
"Who works with Alice?" → MODERATE (shallow graph)
"How are Alice and Bob connected through projects?" → COMPLEX (deep graph)
```

### DualNodeManager (`src/khora/engines/vectorcypher/dual_nodes.py`)

Manages HippoRAG 2 dual-node structure in Neo4j:

```
(:Entity)-[:MENTIONED_IN]->(:Chunk)
(:Chunk)-[:AT_TIME]->(:TimeNode)
```

**Key Operations:**

```python
# Create chunk nodes in batch
await dual_nodes.create_chunk_nodes_batch(chunks, namespace_id)

# Link entities to chunks
await dual_nodes.link_entities_to_chunks_batch(entity_chunk_links)

# Get chunks by entities (via MENTIONED_IN)
chunks = await dual_nodes.get_chunks_by_entities(
    entity_ids=[...],
    namespace_id=namespace_id,
    temporal_filter=temporal_filter,
    temporal_sort=True,  # ORDER BY c.occurred_at DESC, total_mentions DESC
)

# Get entity neighborhoods (graph expansion)
neighborhoods = await dual_nodes.get_entity_neighborhoods(
    entity_ids=[...],
    namespace_id=namespace_id,
    depth=2,
)
```

The `temporal_sort` parameter controls Cypher ordering:

| `temporal_sort` | Cypher `ORDER BY` | When Used |
|-----------------|-------------------|-----------|
| `False` (default) | `total_mentions DESC` | Non-temporal queries - rank by relevance |
| `True` | `c.occurred_at DESC, total_mentions DESC` | Temporal queries - most recent chunks first, tiebreak by relevance |

Neo4j already has an index on `Chunk.occurred_at`, so the temporal sort adds negligible overhead.

### RRF Fusion (`src/khora/engines/vectorcypher/fusion.py`)

Combines vector and graph results using Reciprocal Rank Fusion:

```python
# Weighted RRF fusion
fused_results = weighted_rrf(
    vector_results=vector_chunks,
    graph_results=graph_chunks,
    k=60,              # RRF constant
    vector_weight=0.6, # Emphasize vector
    graph_weight=0.4,  # Graph contribution
)

# Apply recency boost
fused_results = apply_recency_boost(
    fused_results,
    recency_scores,
    recency_weight=0.2,
)
```

**RRF Formula:**
```
score = sum(weight_i / (k + rank_i)) for each source
```

## Query Routing

VectorCypher uses intelligent query routing to balance performance and quality:

### SIMPLE Queries (Vector-Only)

**Characteristics:**
- Simple factual questions
- Single entity mentions
- Direct lookups

**Path:** Query → Embed → pgvector Search → Results

**Latency:** Sub-200ms P95

```python
# Routed as SIMPLE
"What is the company policy on remote work?"
"Who is the CEO?"
"When was the product launched?"
```

### MODERATE Queries (Shallow Graph)

**Characteristics:**
- Single relationship exploration
- Moderate entity complexity
- One-hop connections

**Path:** Query → Embed → Entry Entities → Shallow Expand (depth=1) → Fusion

**Latency:** Sub-400ms P95

```python
# Routed as MODERATE
"Who works with Alice?"
"What projects is the engineering team on?"
"Show me deals related to Acme Corp"
```

### COMPLEX Queries (Full VectorCypher)

**Characteristics:**
- Multi-hop relationships
- Comparisons across entities
- Aggregations over graph structure

**Path:** Query → Embed → Entry Entities → Deep Expand (depth=2-3) → Fusion → Recency

**Latency:** Sub-800ms P95

```python
# Routed as COMPLEX
"How are Alice and Bob connected through projects?"
"Compare the deals that John and Sarah worked on"
"What's the chain of approvals for this budget?"
```

## Temporal Detection

The VectorCypher engine includes a `TemporalDetector` (`src/khora/engines/vectorcypher/temporal_detection.py`) that classifies every query into a temporal category before retrieval begins. This replaces the previous regex-based `_detect_temporal_filter()` with a richer signal that drives multiple retrieval parameters simultaneously.

### How It Works

In `engine.py`'s `recall()` method:

```python
temporal_signal = TemporalDetector().detect(query)

# EXPLICIT category still produces a TemporalFilter for date-range pushdown to pgvector
if temporal_signal.temporal_filter is not None:
    temporal_filter = temporal_signal.temporal_filter
```

The detector classifies the query, and the resulting `TemporalSignal.category` maps to a `RetrievalParams` tuple that controls recency weight, Neo4j temporal sort, and decay override:

| Category | `recency_weight` | `temporal_sort` | `decay_days_override` | Example Query |
|----------|------------------|-----------------|-----------------------|---------------|
| `NONE` | 0.2 | No | - | "What is the capital of France?" |
| `EXPLICIT` | 0.3 | No | - | "What happened before April 2024?" |
| `STATE_QUERY` | 0.5 | Yes | - | "What instrument is she currently playing?" |
| `ORDINAL` | 0.1 | Yes | - | "Which event happened first?" |
| `AGGREGATE` | 0.0 | No | - | "How many projects in total?" |
| `RECENCY` | 0.5 | Yes | 7 | "What's the most recent update?" |
| `CHANGE` | 0.3 | Yes | - | "Does she still work at Google?" |

### Effect on the Retrieval Pipeline

- **`recency_weight`** - Passed to `apply_recency_boost()` after RRF fusion. Higher values amplify the temporal signal; `0.0` (AGGREGATE) disables recency entirely.
- **`temporal_sort`** - When `True`, `DualNodeManager.get_chunks_by_entities()` uses `ORDER BY c.occurred_at DESC, total_mentions DESC` in its Cypher query, surfacing the most recent chunks first.
- **`decay_days_override`** - Overrides the default `recency_decay_days` (30) for categories like RECENCY where a tighter window (7 days) is more appropriate.

### Simple Path Recency

The SIMPLE retrieval path (vector-only, no graph traversal) applies recency boosting when the temporal category calls for it. `_simple_retrieve()` wraps results in `FusedResult` objects, calls `_calculate_recency_scores()`, and applies `apply_recency_boost()` when the effective recency weight is greater than zero.

### Relative Recency Reference

`_calculate_recency_scores()` uses `max(occurred_at)` from the result set as the reference point instead of `datetime.now(UTC)`. This means benchmark data or historical data produces meaningful recency discrimination regardless of when the query is executed - a result from "2 days before the newest result" always gets the same score, whether the data is from 2024 or 2026.

## Configuration

### VectorCypherConfig

```python
from khora.engines.vectorcypher import VectorCypherConfig

config = VectorCypherConfig(
    # Routing
    routing_enabled=True,
    routing_use_llm=False,  # Heuristic routing (faster)

    # Skeleton indexing
    skeleton_core_ratio=0.70,  # 70% get full KG extraction

    # Graph traversal
    graph_default_depth=2,
    graph_max_depth=4,
    graph_max_entry_entities=10,

    # Fusion (default weights for MODERATE queries)
    fusion_rrf_k=60,
    fusion_vector_weight=0.6,
    fusion_graph_weight=0.4,

    # Per-complexity fusion overrides
    fusion_simple_vector_weight=0.8,   # SIMPLE: vector-heavy
    fusion_simple_graph_weight=0.2,
    fusion_complex_vector_weight=0.4,  # COMPLEX: graph-heavy
    fusion_complex_graph_weight=0.6,

    # Temporal
    temporal_recency_weight=0.2,
    temporal_recency_decay_days=30,

    # Search thresholds
    fusion_hybrid_alpha=0.7,
    retriever_min_entity_similarity=0.3,
)
```

### Via `engine_kwargs` (Khora Constructor)

The recommended way to pass `VectorCypherConfig` is through the `engine_kwargs` parameter on `Khora`:

```python
from khora import Khora
from khora.engines.vectorcypher import VectorCypherConfig

async with Khora(
    "postgresql://localhost/khora",
    engine="vectorcypher",
    engine_kwargs={"vectorcypher_config": VectorCypherConfig(
        skeleton_core_ratio=0.50,
        fusion_complex_vector_weight=0.3,
        fusion_complex_graph_weight=0.7,
        retriever_min_entity_similarity=0.25,
    )},
) as kb:
    ns = await kb.create_namespace()
    result = await kb.remember(
        "content",
        namespace=ns.namespace_id,
        entity_types=["PERSON", "ORG"],
        relationship_types=["WORKS_AT"],
    )
    results = await kb.recall("query", namespace=ns.namespace_id)
```

The `engine_kwargs` dict is forwarded directly to the `VectorCypherEngine` constructor, which accepts `vectorcypher_config` as a keyword argument.

### Via Environment Variables

```bash
# Storage
KHORA_DATABASE_URL=postgresql://localhost/khora
KHORA_NEO4J_URL=bolt://localhost:7687
KHORA_NEO4J_USER=neo4j
KHORA_NEO4J_PASSWORD=password

# Engine
KHORA_ENGINE_NAME=vectorcypher
```

### Via YAML

```yaml
# config/vectorcypher/khora.yaml
engine:
  name: vectorcypher

vectorcypher:
  routing:
    enabled: true
    use_llm: false
  skeleton:
    core_ratio: 0.70
  graph:
    default_depth: 2
    max_depth: 4
  fusion:
    rrf_k: 60
    vector_weight: 0.6
    graph_weight: 0.4

query:
  hybrid_alpha: 0.7
  apply_recency_bias: true
```

## Requirements

**Required:**
- PostgreSQL with pgvector extension
- Neo4j (required, not optional)

**Recommended:**
- Neo4j GDS library (for efficient entity vector search)
- Neo4j 5.x+ for best performance

```bash
# Environment setup
KHORA_DATABASE_URL=postgresql://localhost/khora
KHORA_NEO4J_URL=bolt://localhost:7687
KHORA_NEO4J_USER=neo4j
KHORA_NEO4J_PASSWORD=password
```

## Performance Characteristics

| Metric | SIMPLE | MODERATE | COMPLEX |
|--------|--------|----------|---------|
| P95 Latency | <200ms | <400ms | <800ms |
| Graph Depth | 0 | 1 | 2-3 |
| Entry Entities | 5 | 10 | 15 |
| Use Graph | No | Yes | Yes |

### Comparison with Other Engines

| Metric | VectorCypher | Skeleton |
|--------|--------------|----------|
| LLM calls per 1000 docs | ~700 (default) / ~1000 (`skeleton_core_ratio=1.0`) | ~100 |
| Core chunk ratio | 70% (configurable 0.0–1.0) | 10% |
| Multi-hop queries | Native | Limited |
| Graph database | Required | Not required |
| Query routing | Yes | No |
| RRF fusion | Yes | No |

## Tuning Guide

### core_ratio

Controls what percentage of chunks get full knowledge graph extraction:

| Value | LLM Calls | Graph Density | Use When |
|-------|-----------|---------------|----------|
| 0.90 | Most | Very dense graph | Maximum recall, cost not a concern |
| 0.70 | Default | Dense graph | Most cases (good quality/cost balance) |
| 0.50 | Moderate | Moderate graph | Cost-conscious with decent coverage |
| 0.25 | Fewer | Sparse graph | Cost-sensitive, simple queries |

### graph_depth

Controls Cypher traversal depth for complex queries:

| Depth | Hops | Latency | Use When |
|-------|------|---------|----------|
| 1 | Direct connections | Fast | Simple lookups |
| 2 | Friends-of-friends | Default | Most queries |
| 3 | 3-hop paths | Slower | Complex relationships |
| 4 | Maximum | Slowest | Deep exploration |

### Fusion Weights

Controls blending of vector and graph results:

| vector_weight | graph_weight | Behavior |
|---------------|--------------|----------|
| 0.8 | 0.2 | Mostly semantic similarity |
| 0.6 | 0.4 | Balanced (default) |
| 0.4 | 0.6 | Graph-heavy (relationship queries) |
| 0.2 | 0.8 | Mostly graph traversal |

## Adaptive Depth

When `adaptive_depth_enabled=True` (the default), the retriever dynamically adjusts graph traversal depth based on how many entry entities the vector search returns:

| Entry Entities | Depth Adjustment | Reason |
|----------------|------------------|--------|
| ≥ 10 (high threshold) | Reduce to depth 1 | Many entities → deep traversal explodes candidates without adding signal |
| 3–9 | Use configured depth | Normal range, default behavior |
| ≤ 2 (low threshold) | Increase depth by 1 | Few entities → deeper traversal compensates for sparse entry points |

The thresholds are configurable:

```python
VectorCypherConfig(
    # ... or via RetrieverConfig directly
)

# RetrieverConfig fields:
#   adaptive_depth_enabled: bool = True
#   adaptive_depth_high_entity_threshold: int = 10
#   adaptive_depth_low_entity_threshold: int = 2
```

This prevents two failure modes: (1) candidate explosion when many entities each fan out at depth 2+, and (2) under-retrieval when very few entities match and a shallow traversal misses relevant connections.

## Score Normalization

The fusion function `weighted_rrf_normalized` normalizes vector and graph scores to [0, 1] via min-max normalization before computing Reciprocal Rank Fusion. This matters when the two sources produce scores on very different scales - for example, cosine similarity scores in [0.3, 0.9] vs graph proximity scores in [0.01, 0.5]. Without normalization, the source with larger absolute scores dominates the fusion.

Both the SIMPLE and COMPLEX retrieval paths normalize final scores to [0,1] using min-max normalization.

```
RRF score = vector_weight / (k + vector_rank) + graph_weight / (k + graph_rank)
Tiebreaker = normalized_score from the dominant source
```

## Search Index Improvements

Migration 005 adds three PostgreSQL indexes that improve query-time performance:

| Index | Type | Target | Purpose |
|-------|------|--------|---------|
| `ix_khora_chunks_tags_gin` | GIN | `khora_chunks.tags` | Fast array-containment queries (`tags @> ARRAY['topic']`) |
| `ix_khora_chunks_ns_occurred` | B-tree (composite) | `(namespace_id, occurred_at)` | Temporal filtering within a namespace |
| `ix_khora_chunks_embedding_hnsw` | HNSW | `khora_chunks.embedding` | Vector similarity with `ef_construction=128` (up from 64) |

The HNSW index rebuild with higher `ef_construction` improves recall at index-build time - more candidates are considered during graph construction, producing a higher-quality approximate nearest neighbor index. Query-time `ef_search` can be tuned separately via PostgreSQL's `SET hnsw.ef_search = N`.

Run the migration with:

```bash
uv run alembic upgrade head
```

## Recent Improvements

**Cross-encoder reranking.** After the initial vector + Cypher retrieval, an optional cross-encoder model rescores the top candidates for precision. The model is cached across queries to avoid reload overhead, and inference runs in `asyncio.to_thread` to keep the event loop free. Enable/disable via `KHORA_QUERY_ENABLE_RERANKING`.

**Independent BM25 channel.** VectorCypher now runs BM25 full-text search as a separate retrieval channel alongside vector and Cypher graph traversal. Results are fused via RRF, giving keyword-exact matches a dedicated signal path rather than relying solely on embedding similarity.

**Temporal SQL pushdown.** Relative date expressions in queries ("last 7 days", "since March") are detected by the temporal classifier and translated into SQL WHERE clauses that filter at the database level before vector search. This reduces the candidate set and improves both latency and relevance for time-scoped queries. Controlled by `KHORA_QUERY_TEMPORAL_SQL_PUSHDOWN`.

**VectorCypher is now the default engine** when creating a `Khora` without an explicit `engine=` argument.

## Related Documentation

- [Engine Comparison](engine-comparison.md) - Detailed comparison of all engines
- [Skeleton Construction Engine](skeleton-engine.md) - Cost-optimized temporal engine
- [Temporal Model](temporal-model.md) - Bi-temporal design deep dive
- [Hybrid Search](hybrid-search.md) - Vector + BM25 fusion details
- [Temporal Queries](../query-engine/temporal-queries.md) - Temporal detection, recency bias, and temporal filters

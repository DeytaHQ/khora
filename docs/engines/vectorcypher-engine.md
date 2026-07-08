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

For comprehensive extraction over 100% of chunks, pass `engine_kwargs={"vectorcypher_config": VectorCypherConfig(skeleton_core_ratio=1.0)}` - VectorCypher's KET-RAG selectivity defaults to the top 50% of chunks (cost-parity default since #1420; 0.7+ is the quality opt-in) but accepts a 1.0 override that extracts from every chunk.

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

3. **Skeleton-Based Extraction**: Only core chunks (identified via PageRank, default 50%) get full LLM entity extraction, balancing cost and quality

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
    recency_weight: float = 0.35
    recency_decay_days: int = 30
    recency_decay_type: str = "exponential"  # "exponential" or "linear"

    # Search thresholds
    min_entity_similarity: float = 0.3
    hybrid_alpha: float = 0.7
    coherence_weight: float = 0.1  # Weight for cross-chunk coherence scoring

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
4. **Vector Search**: Find entry entities via pgvector similarity (with `hnsw.ef_search = 100`)
5. **Cypher Expand**: Traverse graph to find related entities (if complex)
6. **Fetch Chunks**: Get chunks via `MENTIONED_IN` relationships, with optional temporal sort
7. **RRF Fusion**: Combine vector and graph results
8. **Recency Boost**: Apply temporal boosting with category-specific weights and decay
9. **Coherence Boost**: Blend a bigram-coherence signal into fused scores to demote word-shuffled confounders
10. **Cross-Encoder Rerank**: Optional cross-encoder rescoring of the top candidates
11. **LLM Rerank**: Optional listwise LLM rerank (default off; decisive-winner skip)
12. **Version-Aware Scoring**: Adjust scores for entity-version validity on point-in-time queries
13. **Attach Absolute Scores**: Replace the reported score value with the raw query-chunk cosine (order unchanged, #1433 - see [Score Reporting](#score-normalization))
14. **MMR Diversity**: Select the final top-`limit` set with Maximal Marginal Relevance

### QueryComplexityRouter (`khora.query.router`)

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

**Bounded per-hop expansion (#1419).** `get_entity_neighborhoods` runs an unrolled per-hop BFS that caps each hop's new frontier at `hop_limit=200` nodes per source entity, replacing the old exponential all-paths `[*1..depth]` enumeration - hub nodes can no longer blow up a hop.

**Timeout degradation (#1419, ADR-001).** On a Neo4j timeout the graph channel is no longer a silent `{}`: a structured `Degradation` (`component="vectorcypher.cypher_expand"`, `reason="neo4j_timeout"`) is recorded on `RecallResult.engine_info["degradations"]` and the `khora.vectorcypher.cypher_expand.degraded_total` counter increments.

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
    recency_weight=0.35,
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

The VectorCypher engine includes a `TemporalDetector` (canonical location `khora.query.temporal_detection`; `src/khora/engines/vectorcypher/temporal_detection.py` remains as a back-compat re-export shim) that classifies every query into a temporal category before retrieval begins. This replaces the previous regex-based `_detect_temporal_filter()` with a richer signal that drives multiple retrieval parameters simultaneously.

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
| `NONE` | 0.0 | No | - | "What is the capital of France?" |
| `EXPLICIT` | 0.3 | No | - | "What happened before April 2024?" |
| `STATE_QUERY` | 0.5 | Yes | - | "What instrument is she currently playing?" |
| `ORDINAL` | 0.3 | Yes | - | "Which event happened first?" |
| `AGGREGATE` | 0.0 | No | - | "How many projects in total?" |
| `RECENCY` | 0.5 | Yes | 3 | "What's the most recent update?" |
| `CHANGE` | 0.4 | Yes | 14 | "Does she still work at Google?" |

### Effect on the Retrieval Pipeline

- **`recency_weight`** - Passed to `apply_recency_boost()` after RRF fusion. Higher values amplify the temporal signal; `0.0` (AGGREGATE) disables recency entirely.
- **`temporal_sort`** - When `True`, `DualNodeManager.get_chunks_by_entities()` uses `ORDER BY c.occurred_at DESC, total_mentions DESC` in its Cypher query, surfacing the most recent chunks first.
- **`decay_days_override`** - Overrides the default `recency_decay_days` (30) for categories like RECENCY where a tighter window (3 days) is more appropriate.

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
    skeleton_core_ratio=0.50,  # 50% get full KG extraction (default; 0.7+ = quality opt-in)

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
    temporal_recency_weight=0.35,
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
KHORA_STORAGE_NEO4J_USER=neo4j
KHORA_STORAGE_NEO4J_PASSWORD=password
```

Engine selection is constructor-only - pass `engine="vectorcypher"` to
`Khora(...)` (it is also the default).

### Via YAML

```yaml
# config/vectorcypher/khora.yaml
vectorcypher:
  routing:
    enabled: true
    use_llm: false
  skeleton:
    core_ratio: 0.50
  graph:
    default_depth: 2
    max_depth: 4
  fusion:
    rrf_k: 60
    vector_weight: 0.6
    graph_weight: 0.4

query:
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
KHORA_STORAGE_NEO4J_USER=neo4j
KHORA_STORAGE_NEO4J_PASSWORD=password
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
| LLM calls per 1000 docs | ~500 (default) / ~1000 (`skeleton_core_ratio=1.0`) | ~100 |
| Core chunk ratio | 50% (configurable 0.0–1.0) | 10% |
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
| 0.70 | More | Dense graph | Quality opt-in (denser graph, ~40% more extraction calls) |
| 0.50 | Moderate | Moderate graph | Default (cost parity with pre-#1408 behavior) |
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

**Adaptive graph-empty fallback.** On temporal queries the weights adapt at runtime to stop a sparse graph from diluting good vector hits: if the graph channel returns 0 chunks the temporal weights (0.3/0.7) flip to vector-heavy **0.85/0.15**; if it returns fewer than 3 chunks they fall back to the canonical **0.6/0.4**.

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

Min-max normalization is an **internal** step only (it feeds the coherence blend on the COMPLEX path). The score reported to callers is not a min-max [0,1] value: per the score-vs-order contract (#811/#1433/#1441), the exit of the pipeline overwrites the reported value with the **absolute raw query-chunk cosine** (`attach_relevance_scores`), so an off-topic top result reads low (e.g. ~0.1) instead of being forced to 1.0. Ordering is still decided by the internal `rrf_score` (fusion + boosts + rerank); the list is never re-sorted by the reported value. The SIMPLE path reports cosines directly from the vector store. See [Recall semantics](../query-engine/recall-semantics.md).

```
RRF score = vector_weight / (k + vector_rank) + graph_weight / (k + graph_rank)   # internal ranking
Reported chunk.score = raw query-to-chunk cosine (0.0 when no vector measurement)
```

## Abstention & Confidence

VectorCypher emits abstention and confidence signals on every recall via `result.engine_info` (not `result.metadata`):

```python
result = await kb.recall("What is Alice currently working on?", namespace=ns_id)

signals = result.engine_info["abstention_signals"]
# Four boolean flags:
print(signals["entities_empty"])       # No entities found
print(signals["chunks_empty"])         # No chunks returned
print(signals["chunks_below_min"])     # Fewer than minimum chunks threshold
print(signals["top_score_low"])        # Top chunk score below minimum similarity

print(signals["combined_score"])       # 0.0–1.0 weighted abstention-risk signal (higher = riskier)
print(signals["should_abstain"])       # Convenience bool

confidence = result.engine_info["confidence"]  # 0.0–1.0 calibrated score
```

The `should_abstain` flag is passive - VectorCypher still returns chunks even when it trips. Use it to suppress LLM answer generation when retrieval quality is too low. In the default `cosine_floor` mode, `should_abstain` fires when `top_score_low` trips on its own OR retrieval came back genuinely empty (`chunks_empty AND entities_empty`); it is NOT thresholded from `combined_score`. The `combined_score` is a weighted risk indicator that only directly drives the decision in the legacy `weighted` mode.

The confidence score formula: `0.8 * clip01(top_cosine / target_cosine) + 0.2 * clip01(top_score_gap / target_gap)`.

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

**Independent BM25 channel (opt-in).** When `KHORA_QUERY_ENABLE_BM25_CHANNEL=true` is set, VectorCypher runs BM25 full-text search as a separate retrieval channel alongside vector and Cypher graph traversal. Results are fused via RRF, giving keyword-exact matches a dedicated signal path rather than relying solely on embedding similarity. Default is OFF; keyword matching in HYBRID mode uses the `enable_keyword_search` path inside `HybridQueryEngine`, not this channel.

**Lexical-channel selector (#1391).** `KhoraConfig.query.lexical_channel` (`Literal["bm25", "keyword_ppr"]`, default `"bm25"`) picks which retriever fills the lexical recall slot. `"keyword_ppr"` swaps in an experimental keyword-chunk PageRank channel (per-query personalized PageRank over the namespace's keyword-to-chunk bipartite graph) that feeds the **same** fusion slot as BM25, weighted by `bm25_weight` - fusion is otherwise unchanged. The selector is itself the opt-in for `keyword_ppr`; it does NOT require `enable_bm25_channel=true`. Tunables: `keyword_ppr_damping` (0.85), `keyword_ppr_max_edges` (50000). Switching to `keyword_ppr` requires a re-ingest to populate the `keyword_chunks` table.

**`min_similarity` cosine floor (#1404/#1406/#1425/#1438/#1445).** A per-call `recall(..., min_similarity=...)` floors the vector channel at the storage layer; `0.0` (the default) falls back to the configured `min_chunk_similarity` / `KHORA_QUERY_MIN_CHUNK_SIMILARITY` (default `0.0` = no floor). An explicit (> 0) floor also bounds the lexical channel: lexical-only chunks are excluded from the fused set (their BM25 / keyword-PPR scores are not cosines), and KEYWORD mode gates hits against a floored vector search - mirroring the temporal stores' #1404 semantics. See [Recall semantics](../query-engine/recall-semantics.md).

**Query-time Personalized PageRank (opt-in, #542).** `KHORA_QUERY_ENABLE_PPR_RETRIEVAL=true` (default OFF) replaces the BFS+RRF graph-expansion channel with the HippoRAG-2-style query-time PPR: PageRank is seeded from the entry entities, and chunks are scored by the PR-weighted mass over their mentioned entities blended with the query cosine (`mass * (1 + sim)`). It degrades to vector-only when the entity graph is empty or no entry entities are found - it never crashes the query. Knobs: `ppr_damping` (0.85), `ppr_max_iter` (50), `ppr_tol` (1e-5), `ppr_top_entities` (30), plus the #1373 seed-neighborhood augmentation caps (`ppr_neighborhood_per_seed_limit`, `ppr_max_neighborhood_entities`).

**Temporal SQL pushdown.** Relative date expressions in queries ("last 7 days", "since March") are detected by the temporal classifier and translated into SQL WHERE clauses that filter at the database level before vector search. This reduces the candidate set and improves both latency and relevance for time-scoped queries. Controlled by `KHORA_QUERY_TEMPORAL_SQL_PUSHDOWN`.

**VectorCypher is now the default engine** when creating a `Khora` without an explicit `engine=` argument.

## Related Documentation

- [Engine Comparison](engine-comparison.md) - Detailed comparison of all engines
- [Skeleton Construction Engine](skeleton-engine.md) - Cost-optimized temporal engine
- [Temporal Model](temporal-model.md) - Bi-temporal design deep dive
- [Hybrid Search](hybrid-search.md) - Vector + BM25 fusion details
- [Temporal Queries](../query-engine/temporal-queries.md) - Temporal detection, recency bias, and temporal filters

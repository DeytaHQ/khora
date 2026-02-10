# VectorCypher Engine

The **VectorCypher engine** is a hybrid retrieval engine that combines vector similarity search (pgvector) with Cypher graph traversal (Neo4j). Inspired by Graph RAG 2026 and HippoRAG 2, it excels at complex multi-hop queries while maintaining efficient simple lookups through intelligent query routing.

## When to Use VectorCypher

Choose the VectorCypher engine when:

- **Multi-hop queries matter**: "Who works on deals with companies that Alex mentioned?"
- **Graph traversal is essential**: Navigate organizational hierarchies, deal chains, team structures
- **Relationship discovery is key**: Find implicit connections across data sources
- **Neo4j is available**: VectorCypher requires Neo4j (not optional)

Choose Skeleton Construction instead when:

- **Cost is the primary concern**: Skeleton uses 5-10x fewer LLM calls
- **No Neo4j available**: VectorCypher requires Neo4j
- **Time-based queries dominate**: Skeleton's bi-temporal model is more optimized
- **Simple infrastructure preferred**: Skeleton works with PostgreSQL only

Choose GraphRAG instead when:

- **Full knowledge graph needed**: Complete entity/relationship extraction upfront
- **Highest extraction quality required**: GraphRAG extracts from 100% of documents

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VectorCypherEngine                                  │
│                      remember() / recall() / forget()                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐    │
│  │ QueryComplexity    │  │ VectorCypher       │  │ DualNodeManager    │    │
│  │ Router             │  │ Retriever          │  │ (HippoRAG 2)       │    │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘    │
│            │                       │                       │                │
│  ┌─────────┴────────────┐  ┌──────┴───────────────────────┴────────────┐   │
│  │   Query Analysis     │  │              Retrieval Pipeline            │   │
│  │  SIMPLE → Vector     │  │   Vector Search → Cypher Expand → Fusion  │   │
│  │  MODERATE → Shallow  │  │                                           │   │
│  │  COMPLEX → Deep      │  │                                           │   │
│  └──────────────────────┘  └───────────────────────────────────────────┘   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Storage Layer                                    │
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │   PostgreSQL     │      │    pgvector      │      │     Neo4j        │  │
│  │   (Documents,    │      │   (Embeddings,   │      │  (Entity nodes,  │  │
│  │    Metadata)     │      │  Chunk vectors)  │      │   Chunk nodes,   │  │
│  │                  │      │                  │      │   MENTIONED_IN)  │  │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘  │
│         REQUIRED                 REQUIRED                 REQUIRED          │
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
from khora import MemoryLake

# Use VectorCypher engine explicitly
async with MemoryLake("postgresql://...", engine="vectorcypher") as lake:
    # Store with temporal context
    result = await lake.remember(
        "Meeting notes from Q1 planning with John",
        title="Q1 Planning",
        metadata={
            "author": "alice@company.com",
            "occurred_at": "2024-01-15T10:00:00Z"
        }
    )

    # Recall with graph-enhanced retrieval
    results = await lake.recall(
        "What did we discuss with John about planning?",
        graph_depth=2,  # Expand 2 hops in graph
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

### VectorCypherRetriever (`src/khora/engines/vectorcypher/retriever.py`)

Implements the hybrid retrieval pipeline:

```python
@dataclass
class RetrieverConfig:
    # Graph traversal settings
    default_depth: int = 2
    max_depth: int = 4
    max_entry_entities: int = 10

    # Fusion settings
    rrf_k: int = 60
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    # Temporal settings
    recency_weight: float = 0.2
    recency_decay_days: int = 30
```

**Retrieval Pipeline:**

1. **Route Query**: Classify as SIMPLE, MODERATE, or COMPLEX
2. **Embed Query**: Generate query embedding via LiteLLM
3. **Vector Search**: Find entry entities via pgvector similarity
4. **Cypher Expand**: Traverse graph to find related entities (if complex)
5. **Fetch Chunks**: Get chunks via `MENTIONED_IN` relationships
6. **RRF Fusion**: Combine vector and graph results
7. **Recency Boost**: Apply temporal boosting for recent content

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
)

# Get entity neighborhoods (graph expansion)
neighborhoods = await dual_nodes.get_entity_neighborhoods(
    entity_ids=[...],
    namespace_id=namespace_id,
    depth=2,
)
```

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

    # Fusion
    fusion_rrf_k=60,
    fusion_vector_weight=0.6,
    fusion_graph_weight=0.4,

    # Temporal
    temporal_recency_weight=0.2,
    temporal_recency_decay_days=30,
)
```

### Via Environment Variables

```bash
# Storage
KHORA_DATABASE_URL=postgresql://localhost/khora
KHORA_NEO4J_URL=bolt://localhost:7687
KHORA_NEO4J_USER=neo4j
KHORA_NEO4J_PASSWORD=password

# Engine
KHORA_ENGINE__NAME=vectorcypher
```

### Via Genesis YAML

```yaml
# config/vectorcypher/genesis.yaml
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

| Metric | VectorCypher | Skeleton | GraphRAG |
|--------|--------------|----------|----------|
| LLM calls per 1000 docs | ~250 | ~100 | ~1000 |
| Core chunk ratio | 70% (default) | 10% | 100% |
| Multi-hop queries | Native | Limited | Full |
| Graph database | Required | Not required | Required |
| Query routing | Yes | No | No |
| RRF fusion | Yes | No | Yes |

## Tuning Guide

### core_ratio

Controls what percentage of chunks get full knowledge graph extraction:

| Value | LLM Calls | Graph Density | Use When |
|-------|-----------|---------------|----------|
| 0.35 | More | Dense graph | Multi-hop queries critical |
| 0.25 | Default | Good density | Most cases |
| 0.15 | Fewer | Sparse graph | Cost-sensitive |

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

## Related Documentation

- [Engine Comparison](engine-comparison.md) - Detailed comparison of all engines
- [Skeleton Construction Engine](skeleton-engine.md) - Cost-optimized temporal engine
- [Temporal Model](temporal-model.md) - Bi-temporal design deep dive
- [Hybrid Search](hybrid-search.md) - Vector + BM25 fusion details
- [References](../REFERENCES.md) - Research papers and inspirations (HippoRAG 2, Graph RAG 2026)

# Query Engine Overview

Khora's HybridQueryEngine combines multiple search methods to provide comprehensive retrieval. This document describes the query engine architecture and configuration.

## Query Pipeline

```
Query Input
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 1: Query Understanding                     │
│                                                                  │
│  Single LLM call extracts:                                      │
│  - Intent classification                                         │
│  - Entity mentions with confidence                               │
│  - Temporal references (ISO 8601)                                │
│  - Query expansion terms                                         │
│  - Source priority weights                                       │
│  - Search strategy recommendations                               │
│  - Pre-computed follow-up queries                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 2: Entity Linking                          │
│                                                                  │
│  Link query entity mentions to stored entities:                 │
│  - Exact name matching                                           │
│  - Fuzzy matching (Levenshtein)                                  │
│  - Embedding similarity                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 3: Multi-Source Search                     │
│                                                                  │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │   Vector    │   │    Graph    │   │   Keyword   │          │
│   │   Search    │   │   Search    │   │   (BM25)    │          │
│   │             │   │             │   │             │          │
│   │  pgvector   │   │   Neo4j     │   │  In-memory  │          │
│   │  cosine     │   │  traversal  │   │   index     │          │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │
│          │                 │                 │                   │
│          └────────────────┬┘─────────────────┘                  │
│                           │                                      │
│            (Parallel execution)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 4: RRF Fusion                              │
│                                                                  │
│  Reciprocal Rank Fusion:                                        │
│                                                                  │
│    score(d) = Σ (weight_source / (k + rank_source(d)))          │
│                                                                  │
│  Default weights:                                                │
│    - Vector: 0.5                                                 │
│    - Graph: 0.3                                                  │
│    - Keyword: 0.2                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 5: Temporal Filtering                      │
│                                                                  │
│  Apply time constraints:                                         │
│  - BEFORE / AFTER / BETWEEN                                      │
│  - Recency bias (exponential decay)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 6: Reranking (Optional)                    │
│                                                                  │
│  Neural reranking for improved relevance                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Step 7: Result Limiting                         │
│                                                                  │
│  Return top-k results                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
   QueryResult
```

## HybridQueryEngine

Located at `src/khora/query/engine.py`.

```python
from khora.query import HybridQueryEngine, QueryConfig, SearchMode

engine = HybridQueryEngine(
    storage=storage_coordinator,
    llm_config=llm_config,
)

result = await engine.query(
    "Who developed relativity?",
    namespace_id=namespace_id,
    config=QueryConfig(
        mode=SearchMode.HYBRID,
        limit=10,
        min_similarity=0.3,
    ),
)
```

## QueryConfig

```python
@dataclass
class QueryConfig:
    # Search mode
    mode: SearchMode = SearchMode.HYBRID

    # Result limits
    limit: int = 10
    min_similarity: float = 0.0

    # RRF weights
    vector_weight: float = 0.5
    graph_weight: float = 0.3
    keyword_weight: float = 0.2
    rrf_k: int = 60

    # Temporal
    temporal_filter: TemporalFilter | None = None
    recency_bias: float = 0.0  # 0 = disabled, 0.1-1.0 = strength

    # Graph options
    graph_depth: int = 2
    graph_relationship_types: list[str] | None = None

    # Features
    enable_understanding: bool = True
    enable_reranking: bool = False
    enable_agentic: bool = False  # Multi-step exploration
```

## QueryResult

```python
@dataclass
class QueryResult:
    # Primary results
    chunks: list[tuple[Chunk, float]]
    entities: list[tuple[Entity, float]]

    # Graph context
    graph_info: GraphInfo | None = None

    # Search contributions
    search_contributions: SearchContributions | None = None

    # Temporal info
    temporal_info: TemporalInfo | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
```

### SearchContributions

```python
@dataclass
class SearchContributions:
    vector: int   # Chunks from vector search
    graph: int    # Chunks from graph search
    keyword: int  # Chunks from keyword search
```

### GraphInfo

```python
@dataclass
class GraphInfo:
    entities_linked: list[str]      # Entity names matched
    relationships_traversed: list[tuple[str, str, str]]  # (from, type, to)
```

## Search Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `VECTOR` | Semantic similarity only | Conceptual queries |
| `GRAPH` | Entity/relationship traversal | "Related to X" queries |
| `KEYWORD` | BM25 keyword matching | Exact term search |
| `HYBRID` | Vector + Graph combined | Default, balanced |
| `ALL` | All three methods | Comprehensive search |

See [Search Modes](search-modes.md) for details.

## API Usage

### Via MemoryLake

```python
from khora import MemoryLake, SearchMode

async with MemoryLake() as lake:
    results = await lake.recall(
        "quantum physics discoveries",
        mode=SearchMode.HYBRID,
        limit=10,
    )

    for chunk, score in results.chunks:
        print(f"[{score:.2f}] {chunk.content[:100]}...")
```

### Direct Engine Usage

```python
from khora.query import HybridQueryEngine, QueryConfig

engine = HybridQueryEngine(storage=storage)

result = await engine.query(
    "Einstein collaborators",
    namespace_id,
    config=QueryConfig(
        mode=SearchMode.GRAPH,
        graph_depth=2,
    ),
)
```

### With Temporal Filter

```python
from khora.query.temporal import TemporalFilter

results = await lake.recall(
    "product updates",
    temporal_filter=TemporalFilter.last_days(7),
    recency_bias=0.3,
)
```

See [Temporal Queries](temporal-queries.md) for details.

### Agentic Search

```python
from khora.query.agentic import AgenticSearchAgent

agent = AgenticSearchAgent(engine=engine)

result = await agent.search(
    "What is our product strategy?",
    namespace_id,
    max_steps=3,
)

# Access full trace
print(result.trace.to_dict())
```

See [Agentic Search](agentic-search.md) for details.

## Query Understanding

The engine uses LLM-based query understanding to extract:

- Intent classification
- Entity mentions
- Temporal references
- Query expansions
- Search strategy recommendations

See [Query Understanding](query-understanding.md) for details.

## Fusion

Results from multiple sources are combined using Reciprocal Rank Fusion:

```
score(d) = Σ (weight / (k + rank))
```

See [Fusion](fusion.md) for details.

## Next Steps

- [Search Modes](search-modes.md) - Vector, Graph, Keyword, Hybrid
- [Query Understanding](query-understanding.md) - LLM-based analysis
- [Fusion](fusion.md) - Reciprocal Rank Fusion
- [Temporal Queries](temporal-queries.md) - Time filtering
- [Agentic Search](agentic-search.md) - Multi-step exploration

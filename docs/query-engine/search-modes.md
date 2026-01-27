# Search Modes

Khora supports multiple search modes, each optimized for different query types. This document describes each mode and when to use it.

## Available Modes

```python
from khora import SearchMode

class SearchMode(str, Enum):
    VECTOR = "vector"     # Semantic similarity search
    GRAPH = "graph"       # Entity relationship traversal
    KEYWORD = "keyword"   # BM25 keyword matching
    HYBRID = "hybrid"     # Vector + Graph combined with RRF
    ALL = "all"           # All three methods
```

## Vector Search

Semantic similarity search using embeddings.

### How It Works

1. Embed the query using the same embedding model as chunks
2. Search pgvector for nearest neighbors (cosine similarity)
3. Return chunks sorted by similarity score

```python
# Internal implementation
query_embedding = await embedder.embed(query)

results = await storage.search_similar_chunks(
    namespace_id,
    query_embedding,
    limit=limit,
    min_similarity=min_similarity,
)
```

### Best For

- Conceptual queries ("explain quantum entanglement")
- Finding related content even without exact keywords
- Questions where meaning matters more than specific terms

### Example

```python
results = await lake.recall(
    "machine learning applications in healthcare",
    mode=SearchMode.VECTOR,
)
```

### Configuration

```python
QueryConfig(
    mode=SearchMode.VECTOR,
    limit=10,
    min_similarity=0.3,  # Minimum cosine similarity
)
```

## Graph Search

Entity and relationship traversal in Neo4j.

### How It Works

1. Extract entity mentions from query (via understanding or simple matching)
2. Link mentions to stored entities
3. Traverse the knowledge graph from linked entities
4. Retrieve chunks associated with discovered entities

```python
# Find query entities
linked_entities = await link_query_entities(query, namespace_id)

# Traverse graph
neighborhood = await storage.get_neighborhood(
    entity_id=linked_entity.id,
    depth=graph_depth,
    relationship_types=relationship_types,
)

# Get chunks from related entities
chunks = await get_entity_chunks(neighborhood.entities)
```

### Best For

- Relationship queries ("who works with Einstein?")
- Entity exploration ("what is related to Project X?")
- When you know specific entity names

### Example

```python
results = await lake.recall(
    "Acme Corp partnerships",
    mode=SearchMode.GRAPH,
    config=QueryConfig(
        graph_depth=2,
        graph_relationship_types=["PARTNERS_WITH", "COLLABORATES_WITH"],
    ),
)
```

### Configuration

```python
QueryConfig(
    mode=SearchMode.GRAPH,
    graph_depth=2,                # Traversal depth
    graph_relationship_types=[    # Filter by relationship type
        "WORKS_FOR",
        "MANAGES",
    ],
)
```

## Keyword Search

Traditional BM25 keyword matching.

### How It Works

1. Parse query into keywords (tokenization, stemming)
2. Search using BM25 ranking algorithm
3. Score based on term frequency and document frequency

### Best For

- Exact phrase matching
- Technical terms or product names
- When you need specific keywords to appear

### Example

```python
results = await lake.recall(
    "error: connection refused port 5432",
    mode=SearchMode.KEYWORD,
)
```

### Configuration

```python
QueryConfig(
    mode=SearchMode.KEYWORD,
    limit=10,
)
```

## Hybrid Search (Default)

Combines Vector and Graph search with Reciprocal Rank Fusion.

### How It Works

1. Execute Vector search
2. Execute Graph search
3. Combine results using RRF

```python
# Parallel execution
vector_results, graph_results = await asyncio.gather(
    vector_search(query, namespace_id, limit),
    graph_search(query, namespace_id, depth),
)

# Fuse with RRF
fused = reciprocal_rank_fusion(
    [vector_results, graph_results],
    weights=[0.5, 0.3],
    k=60,
)
```

### Best For

- General queries (default mode)
- When you want both semantic and relationship-based results
- Balanced retrieval

### Example

```python
results = await lake.recall(
    "quarterly planning discussion with engineering team",
    mode=SearchMode.HYBRID,
)
```

### Configuration

```python
QueryConfig(
    mode=SearchMode.HYBRID,
    vector_weight=0.5,
    graph_weight=0.3,
)
```

## All Sources

Executes all three search methods.

### How It Works

1. Execute Vector, Graph, and Keyword searches in parallel
2. Combine all results using RRF
3. Return combined rankings

### Best For

- Comprehensive search
- When unsure which method will work best
- Research/exploration queries

### Example

```python
results = await lake.recall(
    "product launch timeline Q4",
    mode=SearchMode.ALL,
)

# See which sources contributed
print(f"Vector hits: {results.search_contributions.vector}")
print(f"Graph hits: {results.search_contributions.graph}")
print(f"Keyword hits: {results.search_contributions.keyword}")
```

### Configuration

```python
QueryConfig(
    mode=SearchMode.ALL,
    vector_weight=0.5,
    graph_weight=0.3,
    keyword_weight=0.2,
)
```

## Mode Selection Guide

| Query Type | Recommended Mode |
|------------|------------------|
| Conceptual questions | `VECTOR` |
| "Related to X" queries | `GRAPH` |
| Exact phrase search | `KEYWORD` |
| General queries | `HYBRID` |
| Uncertain/exploratory | `ALL` |

## Search Contributions

Query results include contribution tracking:

```python
result = await lake.recall(query, mode=SearchMode.ALL)

contributions = result.search_contributions
print(f"Vector: {contributions.vector} results")
print(f"Graph: {contributions.graph} results")
print(f"Keyword: {contributions.keyword} results")
```

## Combining with Other Options

### With Temporal Filtering

```python
results = await lake.recall(
    "engineering updates",
    mode=SearchMode.HYBRID,
    temporal_filter=TemporalFilter.last_days(30),
)
```

### With Agentic Exploration

```python
results = await lake.recall(
    "product strategy",
    mode=SearchMode.HYBRID,
    config=QueryConfig(
        enable_agentic=True,
    ),
)
```

### With Specific Graph Traversal

```python
results = await lake.recall(
    "team structure",
    mode=SearchMode.GRAPH,
    config=QueryConfig(
        graph_depth=3,
        graph_relationship_types=["REPORTS_TO", "MANAGES"],
    ),
)
```

## Next Steps

- [Query Understanding](query-understanding.md) - LLM analysis
- [Fusion](fusion.md) - How results are combined
- [Overview](overview.md) - Full query pipeline

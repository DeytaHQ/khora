# Reciprocal Rank Fusion

Khora uses Reciprocal Rank Fusion (RRF) to combine results from multiple search methods. This calibration-free algorithm produces stable rankings without score normalization.

## Overview

RRF combines ranked lists by giving each document a score based on its rank in each list:

```
score(d) = Σ (weight / (k + rank(d)))
```

Where:
- `weight` = source-specific weight (e.g., 0.5 for vector)
- `k` = smoothing constant (default 60)
- `rank(d)` = position of document in that source's results (1-indexed)

## Why RRF?

**Calibration-free**: No need to normalize scores across sources
- Vector similarity: 0.0-1.0
- BM25 keyword: unbounded positive
- Graph traversal: various metrics

**Rank-based**: Position matters more than absolute score
- Stable across different scoring systems
- Resistant to outlier scores

**Simple**: No training required
- Works well out of the box
- Tunable via weights and k

## Formula

For a document appearing in multiple sources:

```
RRF_score(d) = Σ_s (weight_s / (k + rank_s(d)))
```

### Example

Document "D" appears:
- Rank 3 in vector search (weight 0.5)
- Rank 7 in graph search (weight 0.3)
- Rank 1 in keyword search (weight 0.2)

With k=60:
```
score = 0.5/(60+3) + 0.3/(60+7) + 0.2/(60+1)
      = 0.5/63 + 0.3/67 + 0.2/61
      = 0.00794 + 0.00448 + 0.00328
      = 0.0157
```

## Default Configuration

```python
QueryConfig(
    # Source weights (must sum to 1.0 for interpretability)
    vector_weight=0.5,
    graph_weight=0.3,
    keyword_weight=0.2,

    # Smoothing constant
    rrf_k=60,
)
```

## Weight Selection

### Default (Balanced)

```python
vector_weight = 0.5  # Semantic similarity prioritized
graph_weight = 0.3   # Relationships considered
keyword_weight = 0.2 # Exact matches as backup
```

### Semantic Focus

```python
vector_weight = 0.7
graph_weight = 0.2
keyword_weight = 0.1
```

### Relationship Focus

```python
vector_weight = 0.3
graph_weight = 0.5
keyword_weight = 0.2
```

### Keyword Focus

```python
vector_weight = 0.2
graph_weight = 0.2
keyword_weight = 0.6
```

## K Parameter

The `k` parameter controls rank smoothness:

### k=60 (Default)

Standard smoothing. Top ranks matter significantly more than lower ranks.

```
Rank 1: 1/(60+1) = 0.0164
Rank 10: 1/(60+10) = 0.0143
Rank 100: 1/(60+100) = 0.0063

Ratio (rank 1 vs rank 10): 1.15x
```

### k=1 (Aggressive)

Top ranks dominate. Massive difference between positions.

```
Rank 1: 1/(1+1) = 0.5
Rank 10: 1/(1+10) = 0.091

Ratio: 5.5x
```

### k=100 (Smooth)

More equal treatment of ranks. Top positions less dominant.

```
Rank 1: 1/(100+1) = 0.0099
Rank 10: 1/(100+10) = 0.0091

Ratio: 1.09x
```

## Implementation

Located at `src/khora/query/fusion.py`.

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Any, float]]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: List of (item, score) tuples per source
        weights: Weight per source (default: equal weights)
        k: Smoothing constant

    Returns:
        Fused list of (item, score) tuples
    """
    if weights is None:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)

    scores: dict[Any, float] = {}

    for source_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[source_idx]

        for rank, (item, _original_score) in enumerate(ranked_list, start=1):
            item_key = get_item_key(item)  # e.g., chunk.id

            rrf_contribution = weight / (k + rank)
            scores[item_key] = scores.get(item_key, 0) + rrf_contribution

    # Sort by RRF score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

## Query Engine Integration

```python
class HybridQueryEngine:
    async def query(self, query: str, namespace_id: UUID, config: QueryConfig):
        # Execute searches in parallel
        vector_results, graph_results, keyword_results = await asyncio.gather(
            self._vector_search(query, namespace_id, config),
            self._graph_search(query, namespace_id, config),
            self._keyword_search(query, namespace_id, config),
        )

        # Determine which sources to fuse based on mode
        ranked_lists = []
        weights = []

        if config.mode in (SearchMode.VECTOR, SearchMode.HYBRID, SearchMode.ALL):
            ranked_lists.append(vector_results)
            weights.append(config.vector_weight)

        if config.mode in (SearchMode.GRAPH, SearchMode.HYBRID, SearchMode.ALL):
            ranked_lists.append(graph_results)
            weights.append(config.graph_weight)

        if config.mode in (SearchMode.KEYWORD, SearchMode.ALL):
            ranked_lists.append(keyword_results)
            weights.append(config.keyword_weight)

        # Apply RRF
        fused_results = reciprocal_rank_fusion(
            ranked_lists,
            weights=weights,
            k=config.rrf_k,
        )

        return fused_results[:config.limit]
```

## Search Contributions Tracking

Track how many results came from each source:

```python
@dataclass
class SearchContributions:
    vector: int   # Results originating from vector search
    graph: int    # Results originating from graph search
    keyword: int  # Results originating from keyword search

# In query result
result = await engine.query(query, namespace_id)
print(f"Vector contributed: {result.search_contributions.vector}")
print(f"Graph contributed: {result.search_contributions.graph}")
print(f"Keyword contributed: {result.search_contributions.keyword}")
```

## Adaptive Weights (Query Understanding)

Query understanding can recommend weights:

```python
# Query understanding extracts source priority
understanding = await understand_query("Who manages engineering?")

# relationship-focused → boost graph
recommended_weights = understanding.source_priority
# {"graph": 0.6, "vector": 0.3, "keyword": 0.1}

# Apply recommended weights
result = await engine.query(
    query,
    namespace_id,
    config=QueryConfig(
        vector_weight=recommended_weights["vector"],
        graph_weight=recommended_weights["graph"],
        keyword_weight=recommended_weights["keyword"],
    ),
)
```

## Handling Duplicates

Documents appearing in multiple sources get contributions from all:

```python
# Document D in both vector (rank 2) and graph (rank 5)
# k=60, equal weights 0.5

vector_contribution = 0.5 / (60 + 2) = 0.00806
graph_contribution = 0.5 / (60 + 5) = 0.00769

total_score = 0.00806 + 0.00769 = 0.01575
```

This naturally boosts documents that appear in multiple sources.

## API Example

```python
from khora import MemoryLake, SearchMode

async with MemoryLake() as lake:
    results = await lake.recall(
        "Einstein contributions",
        mode=SearchMode.ALL,
        config=QueryConfig(
            vector_weight=0.6,
            graph_weight=0.3,
            keyword_weight=0.1,
            rrf_k=60,
        ),
    )

    # Results are RRF-fused from all three sources
    for chunk, score in results.chunks:
        print(f"[{score:.4f}] {chunk.content[:80]}...")
```

## Next Steps

- [Search Modes](search-modes.md) - When to use each mode
- [Query Understanding](query-understanding.md) - Adaptive weight selection
- [Overview](overview.md) - Full query pipeline

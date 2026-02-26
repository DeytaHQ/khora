# Reciprocal Rank Fusion

When you search Khora, you get results from multiple sources: vector similarity, graph traversal, and keyword matching. Each returns a ranked list. But how do you combine them into a single, coherent ranking?

This is where Reciprocal Rank Fusion (RRF) comes in.

## The Problem

Different search methods return different kinds of scores:

```
Vector search:   0.92, 0.87, 0.81, 0.76, ...    (similarity: 0-1)
Keyword search:  12.4, 8.7, 6.2, 4.1, ...       (BM25: unbounded)
Graph search:    3, 2, 2, 1, ...                (hop count, degree)
```

You can't just add these scores - they're on completely different scales. Normalizing them is tricky and sensitive to outliers.

## The Solution: Rank-Based Fusion

RRF ignores the scores entirely and focuses on *ranks*. The intuition: if a document appears near the top of multiple lists, it's probably relevant.

The formula:

```
RRF_score(doc) = Σ weight / (k + rank)
```

For each source where the document appears, add `weight / (k + rank)`.

## How It Works

Say document D appears in three search results:

| Source | Rank | Weight |
|--------|------|--------|
| Vector | 2 | 0.5 |
| Graph | 5 | 0.3 |
| Keyword | 1 | 0.2 |

With k=60 (the default smoothing constant):

```
Vector contribution:  0.5 / (60 + 2) = 0.5 / 62 = 0.00806
Graph contribution:   0.3 / (60 + 5) = 0.3 / 65 = 0.00462
Keyword contribution: 0.2 / (60 + 1) = 0.2 / 61 = 0.00328

Total RRF score: 0.00806 + 0.00462 + 0.00328 = 0.01596
```

Now compare to document E, which only appears in vector search at rank 1:

```
Vector contribution:  0.5 / (60 + 1) = 0.5 / 61 = 0.00820

Total RRF score: 0.00820
```

Document D scores higher (0.01596 vs 0.00820) because it appears in multiple sources, even though E ranked #1 in vector search.

This is exactly what we want: **documents found by multiple methods get boosted**.

## The K Parameter

The `k` parameter controls how much top ranks matter versus lower ranks:

### k = 60 (Default)

Balanced. Top ranks are better, but not dramatically:

```
Rank 1:   1/(60+1)  = 0.0164
Rank 10:  1/(60+10) = 0.0143
Rank 50:  1/(60+50) = 0.0091

Ratio (rank 1 vs 10): 1.15x
```

### k = 1 (Aggressive)

Top ranks dominate:

```
Rank 1:   1/(1+1)  = 0.500
Rank 10:  1/(1+10) = 0.091
Rank 50:  1/(1+50) = 0.020

Ratio (rank 1 vs 10): 5.5x
```

Use this when you're very confident in your top results.

### k = 100 (Smooth)

Ranks are more equal:

```
Rank 1:   1/(100+1)  = 0.0099
Rank 10:  1/(100+10) = 0.0091
Rank 50:  1/(100+50) = 0.0067

Ratio (rank 1 vs 10): 1.09x
```

Use this when you want to give lower-ranked results more of a chance.

## Default Weights

Khora's defaults prioritize semantic similarity while still benefiting from other methods:

```python
vector_weight = 0.5   # Semantic similarity is usually most valuable
graph_weight = 0.3    # Relationships add important context
keyword_weight = 0.2  # Catches exact matches that embeddings miss
```

## Tuning Weights for Your Use Case

### Semantic Focus

When conceptual similarity matters most:

```python
QueryConfig(
    vector_weight=0.7,
    graph_weight=0.2,
    keyword_weight=0.1
)
```

Good for: Research queries, "what's similar to X", exploratory search

### Relationship Focus

When connections between things matter:

```python
QueryConfig(
    vector_weight=0.3,
    graph_weight=0.5,
    keyword_weight=0.2
)
```

Good for: "Who works with X?", "What's connected to Y?", knowledge graphs

### Keyword Focus

When exact terms are critical:

```python
QueryConfig(
    vector_weight=0.2,
    graph_weight=0.2,
    keyword_weight=0.6
)
```

Good for: Technical searches, product names, acronyms, code

### Balanced

When you're not sure:

```python
QueryConfig(
    vector_weight=0.4,
    graph_weight=0.3,
    keyword_weight=0.3
)
```

## Adaptive Weights

Khora's query understanding can recommend weights based on your query:

```python
# Query: "Who manages the engineering team?"
# Understanding: This is a relationship query

# Automatically suggested:
source_priority = {
    "graph": 0.6,    # Boost graph for relationship queries
    "vector": 0.3,
    "keyword": 0.1
}
```

To use these recommendations:

```python
from khora.query import HybridQueryEngine, QueryConfig

# The engine can apply recommendations automatically
result = await engine.query(
    "Who manages engineering?",
    namespace_id,
    config=QueryConfig(use_adaptive_weights=True)
)
```

## Implementation Details

The HybridQueryEngine fusion happens in `src/khora/query/fusion.py`. The VectorCypher engine has its own fusion in `src/khora/engines/vectorcypher/fusion.py`:

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]],
    weights: list[float],
    k: int = 60
) -> list[tuple[Chunk, float]]:
    """Combine ranked lists using RRF."""

    scores: dict[UUID, float] = {}

    for source_idx, ranked_list in enumerate(ranked_lists):
        weight = weights[source_idx]

        for rank, (chunk, _original_score) in enumerate(ranked_list, start=1):
            rrf_contribution = weight / (k + rank)
            scores[chunk.id] = scores.get(chunk.id, 0) + rrf_contribution

    # Sort by RRF score, highest first
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items
```

## Tracking Contributions

After fusion, you can see where results came from:

```python
result = await lake.recall("machine learning applications")

print(f"Vector contributed: {result.search_contributions.vector} results")
print(f"Graph contributed: {result.search_contributions.graph} results")
print(f"Keyword contributed: {result.search_contributions.keyword} results")
```

This helps you understand which search methods are working for your queries.

## Why RRF Works

RRF has several nice properties:

1. **Calibration-free** - No need to normalize scores across different scales

2. **Robust to outliers** - A very high score in one source doesn't dominate

3. **Favors agreement** - Documents found by multiple methods get boosted

4. **Simple** - No machine learning, no training, just math

5. **Tunable** - Weights and k let you adjust behavior

Research has shown RRF performs comparably to learned fusion methods while being much simpler to implement and understand.

## Coherence Scoring (v0.3.5)

After RRF fusion, the VectorCypher retriever applies a lightweight coherence signal to penalize word-shuffled confounders — documents that share the same vocabulary as a relevant chunk but in a nonsensical order. This avoids the cost of an LLM reranking call for obvious confounders.

### How It Works

`bigram_coherence_score()` evaluates text by checking function-word transitions: articles should precede content words, prepositions should precede noun phrases, and so on. Genuine text has predictable bigram patterns; word-shuffled text does not.

### Integration

`apply_coherence_boost()` blends the coherence score into the RRF score:

```
final_score = (1 - coherence_weight) * rrf_score + coherence_weight * coherence_score
```

The default `coherence_weight=0.1` applies a gentle adjustment — enough to demote obvious confounders without overriding the RRF ranking for legitimate results.

### Configuration

```python
from khora.engines.vectorcypher import RetrieverConfig

config = RetrieverConfig(
    coherence_weight=0.1,  # default; set to 0.0 to disable
)
```

> **Note:** Coherence scoring only applies to the VectorCypher retriever pipeline. The HybridQueryEngine's RRF fusion (in `query/fusion.py`) does not include this stage.

## Practical Example

```python
from khora import MemoryLake, SearchMode
from khora.query import QueryConfig

async with MemoryLake() as lake:
    # Search with custom fusion settings
    results = await lake.recall(
        "Einstein's contributions to physics",
        mode=SearchMode.ALL,
        config=QueryConfig(
            vector_weight=0.5,
            graph_weight=0.3,
            keyword_weight=0.2,
            rrf_k=60,
            limit=10
        )
    )

    for chunk, score in results.chunks:
        print(f"[{score:.4f}] {chunk.content[:80]}...")
```

## What's Next?

- **[Search Modes](search-modes.md)** - When to use each search method
- **[Query Understanding](query-understanding.md)** - How adaptive weights are determined
- **[Overview](overview.md)** - The complete query pipeline

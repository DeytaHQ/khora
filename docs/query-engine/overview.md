# Query Engine Overview

When you ask Khora a question, a lot happens behind the scenes. The query engine doesn't just do a simple search - it *understands* your question, searches multiple backends in parallel, and intelligently combines the results.

## The Seven-Step Pipeline

Every query goes through these steps:

```
"Who worked with Einstein on relativity?"
                    |
                    v
         +--------------------+
         |  1. UNDERSTAND     |  What are you really asking?
         +--------------------+
                    |
                    v
         +--------------------+
         |  2. LINK           |  Match "Einstein" to stored entities
         +--------------------+
                    |
         +---------+---------+
         |         |         |
         v         v         v
     +-------+ +-------+ +-------+
     |VECTOR | |GRAPH  | |KEYWORD|   3. SEARCH (parallel)
     +-------+ +-------+ +-------+
         |         |         |
         +---------+---------+
                   |
                   v
         +--------------------+
         |  4. FUSE           |  Combine rankings with RRF
         +--------------------+
                   |
                   v
         +--------------------+
         |  5. FILTER         |  Apply time constraints
         +--------------------+
                   |
                   v
         +--------------------+
         |  6. RERANK         |  Optional: neural reranking
         +--------------------+
                   |
                   v
         +--------------------+
         |  7. LIMIT          |  Return top results
         +--------------------+
                   |
                   v
           Your Results
```

Let's walk through what happens at each step.

## Step 1: Query Understanding

Before searching, Khora uses an LLM to understand what you're actually asking. A single call extracts:

**Intent** - What type of question is this?
- `SEARCH` - Find relevant information
- `QUESTION` - Answer a specific question
- `TEMPORAL` - Time-based query ("last week's updates")
- `RELATIONSHIP` - Connection query ("who knows whom")
- `EXPLORATION` - Open-ended browsing

**Entity Mentions** - People, organizations, concepts referenced:
```python
# Query: "What did Microsoft announce about Azure?"
entities = [
    {"name": "Microsoft", "type": "ORGANIZATION", "confidence": 0.95},
    {"name": "Azure", "type": "PRODUCT", "confidence": 0.90}
]
```

**Temporal References** - Any time expressions, converted to ISO 8601:
```python
# Query: "Updates from last week"
temporal = {
    "original": "last week",
    "start": "2024-01-08T00:00:00Z",
    "end": "2024-01-14T23:59:59Z"
}
```

**Search Strategy** - Which search methods will work best:
```python
# Relationship query -> boost graph search
source_priority = {"graph": 0.6, "vector": 0.3, "keyword": 0.1}
```

**Follow-up Queries** - Pre-computed queries for agentic exploration:
```python
follow_ups = [
    "What are Einstein's most famous publications?",
    "Who else contributed to special relativity?"
]
```

This understanding step is crucial - it shapes everything that follows.

## Step 2: Entity Linking

The query mentioned "Einstein" - but which Einstein? We need to connect query mentions to actual stored entities.

Three matching strategies are used:

1. **Exact match**: "Einstein" → entity named "Einstein"
2. **Fuzzy match**: "Einstien" (typo) → entity "Einstein" (Levenshtein distance)
3. **Embedding similarity**: "the famous physicist" → entity "Albert Einstein"

Linked entities become starting points for graph traversal.

## Step 3: Multi-Source Search

Now we search, hitting all three backends in parallel:

### Vector Search (pgvector)

Converts your query to a vector and finds semantically similar chunks:

```sql
-- Under the hood (simplified)
SELECT chunk_id, content,
       1 - (embedding <=> query_embedding) as similarity
FROM chunks
WHERE namespace_id = $1
ORDER BY embedding <=> query_embedding
LIMIT 50;
```

Great for: Conceptual similarity, paraphrased content, "what's related to X"

### Graph Search (Neo4j)

Starts from linked entities and explores outward:

```cypher
// Find content connected to Einstein
MATCH (e:Entity {name: "Einstein"})-[r*1..2]-(related)
MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
RETURN chunk, r
```

Great for: Relationship queries, "who worked with", "what's connected to"

### Keyword Search (BM25)

Classic text matching with term frequency weighting:

```python
# Tokenize, stem, match
query_terms = ["einstein", "relativ"]  # stemmed
scores = bm25.score(query_terms, all_chunks)
```

Great for: Exact phrases, technical terms, names, acronyms

## Step 4: Reciprocal Rank Fusion

Each search method returns a ranked list. How do we combine them?

**The problem**: Scores aren't comparable. Vector similarity is 0-1, BM25 can be anything, graph metrics vary.

**The solution**: Reciprocal Rank Fusion (RRF) ignores scores entirely and uses *ranks*:

```
RRF_score(chunk) = sum of (weight / (60 + rank)) for each source
```

A chunk ranked #1 in vector and #3 in graph:
```
score = 0.5/(60+1) + 0.3/(60+3) = 0.0082 + 0.0048 = 0.013
```

Chunks appearing in multiple sources get boosted. The `k=60` constant smooths out differences between top ranks.

Default weights:
- Vector: 0.5 (semantic similarity is usually most valuable)
- Graph: 0.3 (relationships add crucial context)
- Keyword: 0.2 (catches exact matches others might miss)

## Step 5: Temporal Filtering

If the query had a time component, we filter results:

```python
TemporalFilter.between("2024-01-01", "2024-01-31")
TemporalFilter.last_days(7)
TemporalFilter.after("2023-06-01")
```

Recency bias can also be applied - recent content scores higher.
Recency weighting is configured globally via `QueryConfig.apply_recency_bias`
and `QueryConfig.recency_weight`; the legacy `recency_bias=` knob isn't
exposed on the current public surface.

## Step 5b: MMR Diversity Selection (Optional)

When `enable_diversity=True` (the default in `QuerySettings`), Maximal Marginal Relevance (MMR) selection ensures result diversity after fusion. MMR iteratively picks candidates that maximize relevance while minimizing similarity to already-selected results:

```
selected = argmax(lambda * relevance - (1 - lambda) * max_sim_to_selected)
```

The MMR stage runs in Rust via `_accel.mmr_diversity_select` with NumPy and pure-Python fallbacks. This prevents returning redundant near-duplicate chunks when the same information appears in multiple documents.

## Step 6: Reranking (Optional)

For higher precision, a neural reranker can reorder the top results:

```python
config = QueryConfig(enable_reranking=True)
```

This uses a cross-encoder model that looks at query-document pairs together, catching nuances that initial retrieval might miss. On the default VectorCypher engine reranking is **on by default** (model `BAAI/bge-reranker-v2-m3`) and configured via `KHORA_QUERY_RERANKING_*` / `config.query.reranking_*` — or a `VectorCypherConfig` for per-engine overrides. See [Reranking](retrieval-tuning.md#reranking) for the knobs, model selection, and how to disable or pick a lighter model.

**Optional date-prefix experiment (opt-in).** `CrossEncoderReranker(include_date_prefix=True)` (or the `include_date_prefix=True` kwarg on `create_reranker`) prepends `[YYYY-MM-DD] ` to each candidate's content before scoring. Off-the-shelf cross-encoders tokenize ISO dates fine, so this gives them an explicit recency signal at negligible token cost. Date source priority: `metadata.custom.occurred_at` → `metadata.custom.sent_at` → `metadata.created_at`. Default **OFF** pending an A/B run on the corporate-shape benchmark (Issue #594, Phase D5).

## Step 7: Result Limiting

Finally, we return the top-k results with all the metadata you need:

```python
QueryResult(
    chunks=[(chunk1, 0.85), (chunk2, 0.72), ...],
    entities=[(einstein_entity, 0.95), ...],
    graph_info=GraphInfo(
        entities_linked=["Albert Einstein"],
        relationships_traversed=[("Einstein", "AUTHORED", "Relativity Paper")]
    ),
    search_contributions=SearchContributions(vector=4, graph=3, keyword=1)
)
```

> **Two retrieval surfaces, two result shapes.** Khora exposes two independent retrieval paths and they intentionally use different result types:
>
> - **`Khora.recall()`** - the top-level public API. Returns a typed `RecallResult` from `khora.core.models.recall` (re-exported as `from khora import RecallResult`) with `chunks: list[RecallChunk]`, `entities: list[RecallEntity]`, `relationships: list[RecallRelationship]`, and a producer-enforced `documents: list[DocumentProjection]` invariant - every chunk's `document_id` and every entity / relationship `source_document_ids` entry is guaranteed to appear in `documents[]`. Use `khora.context_text(result)` to render a formatted context string.
>
> - **`HybridQueryEngine.query()`** (this doc) - the in-package retrieval surface used by `khora.query.agentic.AgenticSearchAgent`. Returns the tuple-shaped `QueryResult` shown above. Carries richer per-method telemetry (`search_contributions`, `graph_info`, `temporal_info`) that doesn't fit the recall projection. Not consumed by `Khora.recall()` or by any of the typed engines (vectorcypher / chronicle / skeleton).
>
> The two paths do not intersect. If you're a library consumer, use `Khora.recall()`. The `HybridQueryEngine` surface is internal-shape and may change between minor releases.


## Using the Query Engine

### Simple Usage via Khora

```python
from khora import Khora, SearchMode

async with Khora() as kb:
    ns = await kb.create_namespace()
    results = await kb.recall(
        "machine learning applications",
        namespace=ns.namespace_id,
        mode=SearchMode.HYBRID,
        limit=10,
    )

    for chunk in results.chunks:
        print(f"[{chunk.score:.2f}] {chunk.content[:100]}...")
```

### With Full Configuration

```python
from datetime import datetime, timedelta, timezone

results = await kb.recall(
    "product updates",
    namespace=ns_id,
    mode=SearchMode.HYBRID,
    limit=20,
    min_similarity=0.1,
    start_time=datetime.now(timezone.utc) - timedelta(days=30),
)
```

`kb.recall()` accepts `mode`, `limit`, `min_similarity`, `start_time`,
and `end_time` directly. Fusion weights, reranking, and recency knobs
are global - configure them via `KhoraConfig.query` (`QueryConfig`) at
construction time or via `KHORA_QUERY_*` environment variables; there
is no `config=` kwarg on `kb.recall()`.

### Agentic Search

Per-query agentic search isn't exposed on `kb.recall()`. Use
`khora.query.agentic.AgenticSearchAgent` directly (see
[agentic-search.md](agentic-search.md)).

## Search Mode Quick Reference

| Mode | What It Does | Best For |
|------|-------------|----------|
| `VECTOR` | Semantic similarity only | "What's similar to X?" |
| `GRAPH` | Entity relationships only | "Who works with X?" |
| `KEYWORD` | Exact term matching only | Technical terms, names |
| `HYBRID` | Vector + Graph + Keyword | Default, balanced |
| `ALL` | All three methods | Same as HYBRID (legacy distinction) |

## What's Next?

- **[Search Modes](search-modes.md)** - When to use each mode
- **[Retrieval Tuning](retrieval-tuning.md)** - Threshold changes, fallbacks, and benchmark-driven improvements
- **[Query Understanding](query-understanding.md)** - How the LLM analyzes queries
- **[Fusion](fusion.md)** - Deep dive into RRF
- **[Temporal Queries](temporal-queries.md)** - Time-based filtering
- **[Agentic Search](agentic-search.md)** - Multi-step exploration

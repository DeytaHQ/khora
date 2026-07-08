# Search Modes

Not all questions are the same. "What's machine learning?" is different from "Who manages the engineering team?" Khora offers different search modes to match different query types.

## The Five Modes

```python
from khora import SearchMode

SearchMode.VECTOR    # Semantic similarity - "What's conceptually related?"
SearchMode.GRAPH     # Relationship traversal - "What's connected to what?"
SearchMode.KEYWORD   # Exact matching - "Where do these words appear?"
SearchMode.HYBRID    # Vector + Graph combined (default); keyword is engine-gated
SearchMode.ALL       # Vector + Graph + Keyword
```

## Vector Search

**What it does**: Finds content that's *semantically similar* to your query, even if the exact words are different.

**How it works**:
1. Your query gets converted to a vector (embedding)
2. pgvector finds chunks with similar vectors
3. Results ranked by cosine similarity

**When to use it**:
- Conceptual questions ("explain quantum computing")
- Finding related content ("what else discusses this topic?")
- When exact keywords don't matter

**Example**:
```python
# "AI ethics" will find content about
# "artificial intelligence morality" or "responsible machine learning"
results = await kb.recall(
    "AI ethics in healthcare",
    namespace=ns_id,
    mode=SearchMode.VECTOR,
)
```

**Why it works**: Vector search understands that "dog" and "puppy" are related, that "king" minus "man" plus "woman" roughly equals "queen", and that a question about "revenue" might be answered by content about "sales figures".

## Graph Search

**What it does**: Starts from entities mentioned in your query and explores their relationships.

**How it works**:
1. Identifies entities in your query ("Einstein", "Acme Corp")
2. Links them to stored entities (exact match, fuzzy match, or embedding similarity)
3. Traverses relationships in Neo4j
4. Returns content connected to discovered entities

**When to use it**:
- Relationship queries ("who works with Alice?")
- Entity exploration ("what's connected to Project X?")
- Organizational questions ("what teams report to Bob?")

**Example**:
```python
# Find everything connected to "Machine Learning Team"
results = await kb.recall(
    "Machine Learning Team projects and members",
    namespace=ns_id,
    mode=SearchMode.GRAPH,
)
# Note: VectorCypher uses adaptive depth automatically. There is no
# per-call graph depth knob; depth is driven by entry-entity count.
```

**Why it works**: If Alice works at Acme and Bob also works at Acme, graph search can infer they're colleagues - even if no document explicitly says so.

**Note**: Graph search now also uses entity embeddings. When your query mentions concepts rather than exact names, pgvector finds entities with similar descriptions, then Neo4j explores their relationships.

## Keyword Search

**What it does**: Classic text search - finds content containing your exact terms.

**How it works**:
1. Tokenizes and stems your query
2. Applies BM25 (Best Match 25) scoring
3. Ranks by term frequency / inverse document frequency

**When to use it**:
- Exact phrase search ("error: connection refused")
- Technical terms ("KHORA_DATABASE_URL")
- Product names, acronyms, identifiers
- When you need those specific words to appear

**Example**:
```python
# Find content with exactly this error message
results = await kb.recall(
    '"NullPointerException in UserService.java"',
    namespace=ns_id,
    mode=SearchMode.KEYWORD,
)
```

**Why it works**: Sometimes you don't want "conceptually similar" - you want exactly what you typed. Keyword search delivers precision.

## Hybrid Search (Default)

**What it does**: Runs Vector, Graph, and Keyword search in parallel, then intelligently combines results.

**How it works**:
1. Execute vector search → ranked list A
2. Execute graph search → ranked list B
3. Execute keyword/full-text search → ranked list C
4. Combine using Reciprocal Rank Fusion (RRF)
5. Documents appearing in multiple lists get boosted

**When to use it**:
- General queries (this is the default for a reason)
- When you want semantic, relationship, and keyword coverage
- When you're not sure which mode would work best

**Example**:
```python
# Gets the best of all three methods
results = await kb.recall(
    "quarterly planning with the product team",
    namespace=ns_id,
    mode=SearchMode.HYBRID,
)
```

**Why it works**: A document ranked #5 in vector and #3 in graph will beat one ranked #1 in vector alone. Consensus across methods signals relevance. Keyword search catches exact terms that vector search might miss.

Default weights:
- Vector: 0.6 (semantic similarity usually matters most)
- Graph: 0.4 (relationships add important context)
- Keyword: 0.3 (catches exact terms, proper nouns, dates; only fuses when the lexical channel is enabled)

On the default `kb.recall()` (VectorCypher) engine only vector + graph fuse by default - the keyword channel is OFF unless `KHORA_QUERY_ENABLE_BM25_CHANNEL=true` (see the note below).

> **Note**: HYBRID includes keyword search alongside vector + graph. [Benchmark analysis](retrieval-tuning.md) showed that without the keyword fallback, 25% of descriptive queries returned zero results. `enable_keyword_search` gates keyword search inside `HybridQueryEngine`. On the default `kb.recall()` (VectorCypher) path it is inert; use `SearchMode.KEYWORD` for pure keyword recall or `KHORA_QUERY_ENABLE_BM25_CHANNEL=true` to add a BM25 channel alongside vector+graph in VectorCypher.

## All Sources

**What it does**: Runs all three search methods and fuses the results.

**How it works**:
1. Execute vector, graph, AND keyword searches in parallel
2. Combine all three lists using RRF
3. Track which sources contributed each result

**When to use it**:
- Comprehensive search where nothing should be missed
- Exploratory queries when you're not sure what you're looking for
- When you want to compare what each method finds

**Example**:
```python
results = await kb.recall(
    "authentication security issues Q4",
    namespace=ns_id,
    mode=SearchMode.ALL,
)

# See what each method contributed (exposed on engine_info, not an attribute)
by_method = results.engine_info.get("search_methods", {}).get("by_method", {})
print(f"Vector: {by_method.get('vector', {}).get('count', 0)} chunks")
print(f"Graph: {by_method.get('graph', {}).get('count', 0)} chunks")
```

**Why it works**: Some queries benefit from semantic understanding, others from relationships, others from exact terms. ALL mode lets each method contribute what it's good at.

Default weights:
- Vector: 0.6
- Graph: 0.4
- Keyword: 0.3

## Quick Reference

| Query Type | Best Mode | Why |
|------------|-----------|-----|
| "What is X?" | `VECTOR` | Conceptual understanding |
| "Who works with X?" | `GRAPH` | Relationship traversal |
| "Error: connection refused" | `KEYWORD` | Exact phrase matching |
| "Project updates" | `HYBRID` | Balanced, all three methods (default) |
| "Everything about the merger" | `ALL` | Comprehensive |

## Tuning Weights

You can adjust how much each method contributes:

```python
# Boost semantic similarity
QueryConfig(
    mode=SearchMode.ALL,
    vector_weight=0.7,
    graph_weight=0.2,
    keyword_weight=0.1
)

# Boost relationships
QueryConfig(
    mode=SearchMode.HYBRID,
    vector_weight=0.4,
    graph_weight=0.6
)
```

## Combining with Other Features

### With Temporal Filtering

Limit results to a time window:

```python
from datetime import datetime, timedelta, timezone

results = await kb.recall(
    "product decisions",
    namespace=ns_id,
    mode=SearchMode.HYBRID,
    start_time=datetime.now(timezone.utc) - timedelta(days=30),
)
```

### With Graph Constraints

VectorCypher uses adaptive graph depth automatically - there is no
`max_graph_depth` knob on `KhoraConfig.query` or `kb.recall()` for the
default VectorCypher path. Depth is driven by entry-entity count (see
[Adaptive Depth](../engines/vectorcypher-engine.md#adaptive-depth)):

```python
results = await kb.recall(
    "engineering org structure",
    namespace=ns_id,
    mode=SearchMode.GRAPH,
)
```

## What's Next?

- **[Fusion](fusion.md)** - How RRF combines results
- **[Query Understanding](query-understanding.md)** - How queries get analyzed
- **[Temporal Queries](temporal-queries.md)** - Time-based filtering
- **[Agentic Search](agentic-search.md)** - Multi-step exploration

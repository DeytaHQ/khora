# Search Modes

Not all questions are the same. "What's machine learning?" is different from "Who manages the engineering team?" Khora offers different search modes to match different query types.

## The Five Modes

```python
from khora import SearchMode

SearchMode.VECTOR    # Semantic similarity - "What's conceptually related?"
SearchMode.GRAPH     # Relationship traversal - "What's connected to what?"
SearchMode.KEYWORD   # Exact matching - "Where do these words appear?"
SearchMode.HYBRID    # Vector + Graph combined (default)
SearchMode.ALL       # All three methods
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
results = await lake.recall(
    "AI ethics in healthcare",
    mode=SearchMode.VECTOR
)
```

**The magic**: Vector search understands that "dog" and "puppy" are related, that "king" minus "man" plus "woman" roughly equals "queen", and that a question about "revenue" might be answered by content about "sales figures".

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
results = await lake.recall(
    "Machine Learning Team projects and members",
    mode=SearchMode.GRAPH,
    config=QueryConfig(
        graph_depth=2,  # Go 2 hops out
        graph_relationship_types=["WORKS_ON", "MEMBER_OF", "MANAGES"]
    )
)
```

**The magic**: If Alice works at Acme and Bob also works at Acme, graph search can infer they're colleagues - even if no document explicitly says so.

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
results = await lake.recall(
    '"NullPointerException in UserService.java"',
    mode=SearchMode.KEYWORD
)
```

**The magic**: Sometimes you don't want "conceptually similar" - you want exactly what you typed. Keyword search delivers precision.

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
results = await lake.recall(
    "quarterly planning with the product team",
    mode=SearchMode.HYBRID
)
```

**The magic**: A document ranked #5 in vector and #3 in graph will beat one ranked #1 in vector alone. Consensus across methods signals relevance. Keyword search catches exact terms that vector search might miss.

Default weights:
- Vector: 50% (semantic similarity usually matters most)
- Graph: 30% (relationships add important context)
- Keyword: 20% (catches exact terms, proper nouns, dates)

> **Note**: HYBRID previously only ran vector + graph. Keyword search was added to HYBRID after [benchmark analysis](retrieval-tuning.md) showed that the missing keyword fallback caused 25% of descriptive queries to return zero results. You can disable it with `enable_keyword_search=False` if you want the old behavior.

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
results = await lake.recall(
    "authentication security issues Q4",
    mode=SearchMode.ALL
)

# See what each method contributed
print(f"Vector: {results.search_contributions.vector} results")
print(f"Graph: {results.search_contributions.graph} results")
print(f"Keyword: {results.search_contributions.keyword} results")
```

**The magic**: Some queries benefit from semantic understanding, others from relationships, others from exact terms. ALL mode lets each method contribute what it's good at.

Default weights:
- Vector: 50%
- Graph: 30%
- Keyword: 20%

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
results = await lake.recall(
    "product decisions",
    mode=SearchMode.HYBRID,
    temporal_filter=TemporalFilter.last_days(30)
)
```

### With Graph Constraints

Control how deep and what relationships to explore:

```python
results = await lake.recall(
    "engineering org structure",
    mode=SearchMode.GRAPH,
    config=QueryConfig(
        graph_depth=3,
        graph_relationship_types=["REPORTS_TO", "MANAGES", "MEMBER_OF"]
    )
)
```

### With Agentic Exploration

Let the query engine follow up on initial results:

```python
results = await lake.recall(
    "competitive landscape",
    mode=SearchMode.HYBRID,
    config=QueryConfig(enable_agentic=True)
)
```

## What's Next?

- **[Fusion](fusion.md)** - How RRF combines results
- **[Query Understanding](query-understanding.md)** - How queries get analyzed
- **[Temporal Queries](temporal-queries.md)** - Time-based filtering
- **[Agentic Search](agentic-search.md)** - Multi-step exploration

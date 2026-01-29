# Performance Optimization

After profiling the query pipeline, the main bottlenecks turned out to be **sequential database operations**, not LLM calls. The query understanding system already uses a single comprehensive LLM call — but several sequential `await` loops were causing 5–15 second delays that batch operations reduced to milliseconds.

**Overall impact**: 40–60% reduction in query latency for typical searches.

## Performance Profile

### Before Optimization

| Phase | Typical Latency | Bottleneck |
|-------|----------------|------------|
| Query Understanding | 2–3s | 1 LLM call (already optimal) |
| Entity Linking | 1–2s | Sequential matching strategies |
| Vector Search | 1–2s | Sequential entity fetches |
| Graph Search | 2–3s | Sequential neighborhood queries |
| Keyword Search | 0.5–1s | OK |
| Reranking | 3–5s | Many LLM calls |
| **Total** | **10–20s** | |

Agentic search (3 steps) added another 2–5 seconds per step from sequential chunk source lookups.

### After Optimization

| Query type | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Simple query | 10–20s | 2–5s | ~75% |
| Agentic search (3 steps) | 20–40s | 6–12s | ~65% |
| Repeated query (cached) | 10–20s | <1s | ~95% |

## Phase 1: Batch Database Operations

The highest-impact changes replaced sequential `await` loops with single batch queries.

### Batch Entity Fetches in Vector Search

Vector search was fetching entities one at a time:

```python
# Before: N sequential awaits
for entity_id, score in entity_ids_scores:
    entity = await self._storage.get_entity(entity_id)

# After: single batch query
entity_ids = [eid for eid, _ in entity_ids_scores]
entities_map = await self._storage.get_entities_batch(entity_ids)
```

Impact: 8–15% overall latency reduction. The `get_entities_batch()` method uses a single SQL `IN` clause instead of N individual queries.

### Parallel Graph Neighborhood Lookups

Graph search was fetching entity data and neighborhoods sequentially. Now both are fetched in parallel via `asyncio.gather`:

```python
entities_map, neighborhoods = await asyncio.gather(
    self._storage.get_entities_batch(entity_ids),
    self._storage.get_neighborhoods_batch(entity_ids, depth=max_depth),
)
```

Impact: 5–10% improvement. The `get_neighborhoods_batch()` method fetches all neighborhoods in a single graph query.

### Batch Chunk Source Lookups in Agentic Search

Agentic search was looking up the source document for each chunk individually:

```python
# Before
for chunk, score in step_result.chunks:
    source = await self._get_chunk_source(chunk, namespace_id)

# After
doc_ids = list({chunk.document_id for chunk, _ in step_result.chunks})
docs = await self._storage.get_documents_batch(doc_ids)
```

Impact: 3–5% improvement per agentic step.

## Phase 2: Parallelized Operations

### Parallel Entity Linking Strategies

Entity linking runs exact match first (fast, often hits). If no exact match, fuzzy and embedding strategies now run in parallel instead of sequentially:

```python
# Exact match first (early exit if found)
exact = await self._exact_name_match(mention, namespace_id)
if exact:
    return LinkedEntity(entity=exact, confidence=1.0, method="exact")

# Fuzzy + embedding in parallel
tasks = {}
if self._fuzzy_match:
    tasks["fuzzy"] = self._fuzzy_name_match(mention, namespace_id)
if self._embedding_match:
    tasks["embedding"] = self._embedding_match_entities(mention, namespace_id)
results = await asyncio.gather(*tasks.values())
```

Impact: 2–3x faster entity linking when exact match misses.

### Batch LLM Reranking

Instead of one LLM call per candidate, multiple candidates are scored in a single prompt:

```python
# Before: 50 LLM calls for 50 candidates
# After: 5-10 LLM calls (batch of 5-10 candidates each)
```

Impact: 5–10x fewer LLM calls during reranking.

### Multi-Chunk Entity Extraction

Entity extraction during ingestion now processes multiple chunks per LLM call using structured output:

```python
# Before: 1 LLM call per chunk
# After: 1 LLM call per 5-10 chunks
```

Impact: 5–10x fewer extraction calls during ingestion.

## Phase 3: Structural Optimizations

### Storage Batch Methods

New batch methods on `StorageCoordinator`:

| Method | Description |
|--------|-------------|
| `get_entities_batch(ids)` | Fetch multiple entities in one SQL query |
| `get_documents_batch(ids)` | Fetch multiple documents in one SQL query |
| `get_neighborhoods_batch(ids, depth)` | Fetch multiple graph neighborhoods |

These use `WHERE id IN (...)` clauses (PostgreSQL) and batched Cypher queries (Neo4j) to avoid N+1 query patterns.

### Query Result Caching

An LRU cache with TTL avoids re-executing identical queries:

```python
class QueryCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300): ...
    async def get(self, query, namespace_id, mode) -> QueryResult | None: ...
    async def set(self, query, namespace_id, mode, result): ...
```

Cache key is `sha256(query + namespace_id + mode)`. TTL defaults to 5 minutes. The cache is checked at the start of `HybridQueryEngine.query()` and populated on completion.

### Streaming Results for Agentic Search

Agentic search supports streaming results as each step completes:

```python
async def search_stream(query, namespace_id, max_steps=3):
    """Yield AgenticSearchStep as each step completes."""
```

This allows consumers to display partial results while follow-up queries are still executing.

## Phase 4: Advanced Optimizations

### Speculative Execution

Agentic search pre-executes likely follow-up queries in parallel with the main search:

```python
# Main search and follow-up queries start concurrently
main_task = self._execute_search(query, namespace_id, understanding)
followup_tasks = [
    self._execute_search(fq, namespace_id, None)
    for fq in understanding.follow_up_queries[:2]
]
main_result = await main_task
followup_results = await asyncio.gather(*followup_tasks)
```

Impact: 20–30% improvement for multi-step agentic searches.

### Connection Pooling

Database connection pools are sized to match concurrent operation patterns:

| Setting | Value |
|---------|-------|
| PostgreSQL pool size | 20 |
| PostgreSQL max overflow | 30 |
| Neo4j max connection pool | 50 |

## Search Pipeline Timing

Every query now includes per-phase timing in its metadata, exposed via `SearchMetrics`:

```python
result = await engine.query("find documents about X", namespace_id)
metrics = result.metadata["metrics"]
# {
#   "understanding_ms": 2100,
#   "linking_ms": 350,
#   "search_ms": 800,
#   "fusion_ms": 12,
#   "reranking_ms": 1500,
#   "total_ms": 4762
# }
```

Use these metrics to identify regressions and verify optimization impact.

## What's Next

- **[Query Engine Overview](../query-engine/overview.md)** — how the full search pipeline works
- **[Agentic Search](../query-engine/agentic-search.md)** — multi-step search with follow-ups
- **[Storage Backends](storage-backends.md)** — PostgreSQL, pgvector, and Neo4j details

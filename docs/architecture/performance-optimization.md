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

### Reranking Skip for Small Result Sets

Neural reranking (cross-encoder) is skipped when fewer than 5 candidate chunks are available. This saves several seconds of latency for queries that return few results, particularly after zero-result fallback recovery.

```python
# Before: always rerank if enabled
if cfg.enable_reranking and fused_chunks:

# After: only rerank when there are enough candidates to meaningfully reorder
if cfg.enable_reranking and len(fused_chunks) >= 5:
```

Impact: saves 3–8 seconds on queries with few results.

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

## Phase 5: Ingestion Performance -- Entity Resolution

The query pipeline isn't the only place where performance matters. The ingestion pipeline had a critical bottleneck: entity resolution during expansion was O(n^2) and degraded severely as the knowledge graph grew. At around 12,000 documents the pipeline would stall for over an hour.

### The Problem

In `incremental` inference mode (the previous default), every document triggered:

1. **Full namespace reload** -- `list_entities(limit=1000)` + `list_relationships(limit=5000)` from storage
2. **O(n^2) pairwise matching** -- Cosine similarity on all entity embeddings + Levenshtein on all same-type entity names
3. **N+1 entity writes** -- Each entity did individual `get_entity_by_name()` + create/update
4. **Context rebuild** -- `RuleEvaluationContext.from_data()` rebuilt all indices every inference pass

For 16,000 documents with 5,000 entities, this meant ~16,000 database reloads and ~16,000 O(n^2) comparisons.

### The Solution: Smart Mode

Smart mode (`inference_mode="smart"`, now the default) separates entity resolution into two phases:

| Phase | When | What | Complexity |
|-------|------|------|-----------|
| **Phase 1** | Per document | Extract, within-doc exact dedup via `EntityIndex` | O(1) per entity lookup |
| **Phase 2** | After all docs | Cross-document resolution via token blocking, relationship inference on full graph | O(n * k), k ~ 10-20 |

### Complexity Comparison

| Operation | Before (incremental) | After (smart) |
|-----------|---------------------|---------------|
| Per-doc entity dedup | Load 1K entities from DB + O(n^2) unify | O(1) index lookup |
| Per-doc relationships | Load 5K rels from DB | Skip entirely |
| Embedding matching (total) | O(n^2) * num_docs | O(n * k) * 1 pass |
| Fuzzy matching (total) | O(n^2) * num_docs | O(n * k) * 1 pass |
| Entity storage | 2 queries per entity | 1 batch per 50 entities |
| Relationship inference | Per-doc context rebuild * 2 passes | 1 context build * 2 passes total |

Where k is the token-blocked candidate set size per entity, typically 10-20.

### Token Blocking

The key technique behind the improvement. Instead of comparing every entity to every other entity, token blocking requires at least one shared name token before running expensive similarity computations:

```
"Microsoft Corporation"  ->  tokens: {microsoft, corporation}
"Microsoft Corp"         ->  tokens: {microsoft, corp}
"Apple Inc"              ->  tokens: {apple, inc}

Candidates for "Microsoft Corporation":
  Token "microsoft" -> {Microsoft Corp}    (shared token)
  "Apple Inc" never enters the candidate set.
```

This reduces the number of Levenshtein distance computations from O(n^2) to O(n * k), where k is the number of entities sharing at least one token. For typical knowledge graphs this means 10-20 candidates per entity instead of thousands.

For embedding-based matching, the candidate set also includes all same-type entities (since embeddings can match semantically similar entities with completely different names). This is still a significant reduction from full pairwise comparison.

### Batch Storage Operations

Smart mode also reduces database round-trips through batch operations:

| Method | Backend | Implementation |
|--------|---------|---------------|
| `upsert_entities_batch()` | Neo4j | `UNWIND + MERGE` with ON CREATE SET / ON MATCH SET |
| `upsert_entities_batch()` | PostgreSQL | `INSERT ... ON CONFLICT DO UPDATE` |
| `create_relationships_batch()` | Neo4j | `UNWIND + CREATE` grouped by relationship type |

Default batch size is 50 entities per batch, configurable via `ExpansionConfig.batch_storage_size`.

### Configuration

```yaml
expansion:
  inference_mode: smart        # "smart", "incremental", "batch", "none"
  preload_existing: true       # Load existing entities into index before processing
  batch_storage_size: 50       # Entities per batch upsert
```

See [Semantic Expansion](../extraction/semantic-expansion.md) for full details on the `EntityIndex` and resolution pipeline.

### References

- Papadakis, G., et al. "Blocking and Filtering Techniques for Entity Resolution." *ACM Computing Surveys*, 2020. [doi:10.1145/3377455](https://dl.acm.org/doi/abs/10.1145/3377455)
- Microsoft GraphRAG. "Default Dataflow." [Documentation](https://microsoft.github.io/graphrag/index/default_dataflow/)

## What's Next

- **[Query Engine Overview](../query-engine/overview.md)** -- How the full search pipeline works
- **[Agentic Search](../query-engine/agentic-search.md)** -- Multi-step search with follow-ups
- **[Storage Backends](storage-backends.md)** -- PostgreSQL, pgvector, and Neo4j details
- **[Semantic Expansion](../extraction/semantic-expansion.md)** -- Entity resolution and inference details

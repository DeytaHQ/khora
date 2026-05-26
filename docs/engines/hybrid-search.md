# Hybrid Search

The Skeleton Construction engine implements hybrid search combining vector similarity (semantic) with BM25 full-text search (keyword), merged using Reciprocal Rank Fusion (RRF). This document explains the search pipeline and tuning options.

## Search Pipeline

```
Query
  │
  ├──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
┌─────────────────┐         ┌─────────────────┐
│  Vector Search  │         │   BM25 Search   │
│  (Embeddings)   │         │  (Full-text)    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
    Ranked List 1               Ranked List 2
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Reciprocal Rank    │
          │  Fusion (RRF)       │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Temporal Filter    │
          │  (if specified)     │
          └──────────┬──────────┘
                     │
                     ▼
              Final Results
```

## Vector Search

Semantic similarity using embeddings (pgvector or Weaviate).

### How It Works

1. Query is embedded using the same model as documents
2. Cosine similarity computed against stored embeddings
3. Results ranked by similarity score

```python
# PostgreSQL + pgvector
SELECT
    id, content,
    1 - (embedding <=> $query_embedding) AS similarity
FROM khora_chunks
WHERE namespace_id = $namespace_id
ORDER BY embedding <=> $query_embedding
LIMIT $limit;
```

### Strengths

- Semantic understanding ("car" matches "automobile")
- Handles synonyms and paraphrases
- Good for natural language queries
- Works well with long, contextual queries

### Weaknesses

- May miss exact keyword matches
- Embeddings can conflate unrelated concepts
- Computationally expensive for large corpora
- Requires embedding model at query time

## BM25 Search

Full-text keyword search using PostgreSQL tsvector or Weaviate.

### How It Works

1. Query tokenized into terms
2. BM25 scoring based on term frequency and document length
3. Results ranked by relevance score

```python
# PostgreSQL
SELECT
    id, content,
    ts_rank_cd(content_tsv, plainto_tsquery($query)) AS bm25_score
FROM khora_chunks
WHERE namespace_id = $namespace_id
  AND content_tsv @@ plainto_tsquery($query)
ORDER BY bm25_score DESC
LIMIT $limit;
```

### Strengths

- Exact keyword matching
- Fast (inverted index)
- Good for technical terms, proper nouns
- Predictable behavior

### Weaknesses

- No semantic understanding
- Misses synonyms
- Sensitive to query phrasing
- Struggles with natural language queries

## Reciprocal Rank Fusion (RRF)

RRF combines multiple ranked lists without requiring score normalization:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where:
- `k` is a constant (default: 60)
- `rank_i(d)` is the rank of document `d` in list `i`

### Algorithm

```python
def rrf_fusion(
    vector_results: list[tuple[UUID, float]],
    bm25_results: list[tuple[UUID, float]],
    alpha: float = 0.5,
    k: int = 60,
) -> list[tuple[UUID, float]]:
    """Combine vector and BM25 results using RRF."""
    scores = defaultdict(float)

    # Vector contribution
    for rank, (doc_id, _) in enumerate(vector_results, start=1):
        scores[doc_id] += alpha * (1 / (k + rank))

    # BM25 contribution
    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores[doc_id] += (1 - alpha) * (1 / (k + rank))

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: -x[1])
```

### Why RRF?

| Method | Pros | Cons |
|--------|------|------|
| Score normalization | Preserves score magnitude | Different score scales |
| Weighted sum | Simple | Requires calibration |
| **RRF** | Scale-invariant, robust | Ignores score magnitude |

RRF is preferred because:
1. Vector similarity (0-1) and BM25 scores (unbounded) have different scales
2. Rank-based fusion is more stable across queries
3. No calibration needed

## hybrid_alpha Parameter

Controls the balance between vector and BM25:

```python
results = await engine.recall(
    query,
    namespace_id=namespace_id,
    hybrid_alpha=0.7,  # 70% vector, 30% BM25
)
```

| `hybrid_alpha` | Vector Weight | BM25 Weight | Best For |
|----------------|---------------|-------------|----------|
| 1.0 | 100% | 0% | Pure semantic search |
| 0.8 | 80% | 20% | Natural language, conceptual queries |
| 0.7 | 70% | 30% | **Default** - balanced |
| 0.5 | 50% | 50% | Equal weight |
| 0.3 | 30% | 70% | Technical terms, exact phrases |
| 0.0 | 0% | 100% | Pure keyword search |

### Tuning Guidelines

**Increase alpha (more vector):**
- Natural language questions
- Conceptual queries ("how does X work?")
- Synonym-heavy content
- Long, contextual queries

**Decrease alpha (more BM25):**
- Technical documentation
- Proper nouns, product names
- Exact phrase matching needed
- Short, keyword-style queries

## Temporal Filtering

After fusion, results are filtered by temporal constraints:

```python
from khora.engines.skeleton.backends import TemporalFilter

results = await engine.recall(
    "project updates",
    namespace_id=namespace_id,
    temporal_filter=TemporalFilter(
        occurred_after=datetime(2024, 1, 1),
        occurred_before=datetime(2024, 6, 30),
        author="alice@company.com",
        channel="engineering",
        tags=["important"],
    )
)
```

### Filter Fields

| Field | Type | Description |
|-------|------|-------------|
| `occurred_after` | datetime | Event happened after this time |
| `occurred_before` | datetime | Event happened before this time |
| `created_after` | datetime | Ingested after this time |
| `created_before` | datetime | Ingested before this time |
| `source_system` | str | Source system (e.g., "slack", "linear") |
| `author` | str | Author identifier |
| `channel` | str | Channel/category |
| `tags` | list[str] | All tags must match |
| `additional` | dict | Custom metadata filters |

### SQL Generation

Filters are converted to SQL WHERE clauses:

```python
def _build_filter_conditions(
    self,
    tf: TemporalFilter,
) -> tuple[str, dict]:
    """Build SQL conditions from TemporalFilter."""
    conditions = []
    params = {}

    if tf.occurred_after:
        conditions.append("occurred_at >= :occurred_after")
        params["occurred_after"] = tf.occurred_after

    if tf.occurred_before:
        conditions.append("occurred_at < :occurred_before")
        params["occurred_before"] = tf.occurred_before

    if tf.author:
        conditions.append("author = :author")
        params["author"] = tf.author

    if tf.tags:
        conditions.append("tags @> :tags")
        params["tags"] = tf.tags

    return " AND ".join(conditions), params
```

## Backend Implementations

### pgvector Backend

```python
async def search(
    self,
    query_embedding: list[float],
    *,
    namespace_id: UUID,
    limit: int = 10,
    hybrid_alpha: float | None = None,
    query_text: str | None = None,
    temporal_filter: TemporalFilter | None = None,
) -> list[TemporalSearchResult]:
    """Search with optional hybrid mode."""

    if hybrid_alpha is not None and query_text:
        # Hybrid search
        vector_results = await self._vector_search(
            query_embedding, namespace_id=namespace_id, limit=limit * 2,
        )
        bm25_results = await self._bm25_search(
            query_text, namespace_id=namespace_id, limit=limit * 2,
        )
        fused = self._rrf_fusion(
            vector_results, bm25_results, alpha=hybrid_alpha
        )
        results = fused[:limit]
    else:
        # Vector-only
        results = await self._vector_search(
            query_embedding, namespace_id=namespace_id, limit=limit,
        )

    # Apply temporal filter
    if temporal_filter:
        results = self._apply_temporal_filter(results, temporal_filter)

    return results
```

> `namespace_id` is a required kwarg on every storage read/write and is filtered at the SQL/SurrealQL layer (#769). The `sqlite_lance` backend's `search_similar` additionally re-filters by `namespace_id` on the SQLite re-fetch step as defense-in-depth.

### Weaviate Backend

Weaviate has native hybrid search with alpha blending:

```python
async def search(
    self,
    query_embedding: list[float],
    *,
    namespace_id: UUID,
    limit: int = 10,
    hybrid_alpha: float | None = None,
    query_text: str | None = None,
    temporal_filter: TemporalFilter | None = None,
) -> list[TemporalSearchResult]:
    """Search using Weaviate's native hybrid."""

    collection = self.client.collections.get("KhoraChunk")

    if hybrid_alpha is not None and query_text:
        # Native hybrid search
        response = await collection.query.hybrid(
            query=query_text,
            vector=query_embedding,
            alpha=hybrid_alpha,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            limit=limit,
            filters=self._build_weaviate_filter(temporal_filter),
            return_metadata=MetadataQuery(score=True),
        )
    else:
        # Vector-only
        response = await collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            filters=self._build_weaviate_filter(temporal_filter),
            return_metadata=MetadataQuery(certainty=True),
        )

    return [self._object_to_result(obj) for obj in response.objects]
```

## Performance Optimization

### Index Strategy

```sql
-- HNSW for vector similarity (pgvector)
CREATE INDEX idx_chunks_embedding_hnsw ON khora_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 128);

-- GIN for full-text search
CREATE INDEX idx_chunks_content_tsv ON khora_chunks
    USING GIN (content_tsv);

-- BRIN for temporal filtering (compact)
CREATE INDEX idx_chunks_occurred_at_brin ON khora_chunks
    USING BRIN (occurred_at);

-- B-tree for structured fields
CREATE INDEX idx_chunks_author ON khora_chunks (author);
CREATE INDEX idx_chunks_channel ON khora_chunks (channel);
```

### Query Planning

Hybrid search issues two queries in parallel:

```python
async def _hybrid_search(self, ...):
    # Run vector and BM25 in parallel
    vector_task = asyncio.create_task(
        self._vector_search(query_embedding, namespace_id=namespace_id, limit=limit * 2)
    )
    bm25_task = asyncio.create_task(
        self._bm25_search(query_text, namespace_id=namespace_id, limit=limit * 2)
    )

    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
    return self._rrf_fusion(vector_results, bm25_results, alpha=hybrid_alpha)
```

### Caching

Embedding the query is the slowest part. Consider caching:

```python
@lru_cache(maxsize=1000)
async def embed_query(query: str) -> list[float]:
    return await embedder.embed(query)
```

## Benchmarks

*On 100,000 chunks, PostgreSQL + pgvector:*

| Search Type | Latency (p50) | Latency (p99) |
|-------------|---------------|---------------|
| Vector only | 45ms | 120ms |
| BM25 only | 15ms | 50ms |
| Hybrid | 60ms | 150ms |
| Hybrid + temporal filter | 70ms | 180ms |

*Retrieval quality (NDCG@10):*

| Search Type | NDCG@10 |
|-------------|---------|
| Vector only | 0.72 |
| BM25 only | 0.58 |
| Hybrid (α=0.7) | 0.78 |
| Hybrid (α=0.5) | 0.75 |

## Configuration

### Via Code

```python
from khora import Khora

async with Khora(db_url, engine="skeleton") as kb:
    results = await kb.recall(
        query,
        namespace=ns_id,
        mode=SearchMode.HYBRID,
        start_time=datetime(2024, 1, 1),
    )
# Note: per-call hybrid alpha / author filters aren't exposed on the
# public facade. Configure global weighting via environment variables
# (see below) or `KhoraConfig.query`.
```

### Via Environment

```bash
KHORA_QUERY_VECTOR_WEIGHT=0.7
KHORA_QUERY_KEYWORD_WEIGHT=0.3
KHORA_QUERY_MIN_CHUNK_SIMILARITY=0.0
KHORA_QUERY_MIN_ENTITY_SIMILARITY=0.0
```

`hybrid_alpha` is a per-call argument to `kb.recall(...)` only — there
is no global env-var equivalent. The vector/keyword channel weights
above (`KHORA_QUERY_VECTOR_WEIGHT` / `KHORA_QUERY_KEYWORD_WEIGHT`) are
the closest tunable defaults. The per-call `limit` argument likewise
has no env-var default; see `KHORA_QUERY_STAGE1_RECALL_LIMIT`,
`KHORA_QUERY_STAGE3_FILTER_LIMIT`, and `KHORA_QUERY_STAGE4_RERANK_LIMIT`
for per-stage caps.

### Via YAML

```yaml
query:
  vector_weight: 0.7
  keyword_weight: 0.3
  min_chunk_similarity: 0.0
  min_entity_similarity: 0.0
  recency_decay_days: 30
```

## Related Documentation

- [Skeleton Construction Engine](skeleton-engine.md) - Overview of the Skeleton Construction engine
- [Query Engine](../query-engine/search-modes.md) - Search modes overview
- [Fusion](../query-engine/fusion.md) - RRF implementation details
- [Temporal Queries](../query-engine/temporal-queries.md) - Time-based filtering

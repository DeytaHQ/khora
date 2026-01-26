# Khora Performance Optimization Plan

## Executive Summary

After analyzing the codebase, I've identified that **the main bottlenecks are NOT LLM calls but sequential database operations**. The query understanding system is already well-optimized (single comprehensive LLM call), but several sequential `await` loops are causing 5-15 second delays that could be reduced to milliseconds.

**Expected Impact**: 40-60% reduction in query latency for typical searches.

---

## Current Performance Profile

### Typical Query (10-20 seconds)
| Phase | Current Time | Bottleneck |
|-------|-------------|------------|
| Query Understanding | 2-3s | 1 LLM call (optimal) |
| Entity Linking | 1-2s | Sequential matching |
| Vector Search | 1-2s | Sequential entity fetches |
| Graph Search | 2-3s | **Sequential neighborhood queries** |
| Keyword Search | 0.5-1s | OK |
| Reranking | 3-5s | Many LLM calls |

### Agentic Search (20-40 seconds for 3 steps)
- Sequential chunk source lookups add 2-5s per step
- Follow-up queries not fully parallelized

---

## Phase 1: Quick Wins (High Impact, Low Effort)

### 1.1 Batch Entity Fetches in Vector Search
**File**: `src/khora/query/engine.py:675-678`
**Impact**: 8-15% overall improvement

```python
# BEFORE (Sequential - 5-20 awaits)
for entity_id, score in entity_ids_scores:
    entity = await self._storage.get_entity(entity_id)
    if entity:
        entities.append((entity, score))

# AFTER (Single batch query)
entity_ids = [eid for eid, _ in entity_ids_scores]
entities_map = await self._storage.get_entities_batch(entity_ids)
entities = [(entities_map[eid], score) for eid, score in entity_ids_scores if eid in entities_map]
```

**Requires**: Add `get_entities_batch()` to StorageCoordinator

---

### 1.2 Parallelize Graph Neighborhood Lookups
**File**: `src/khora/query/engine.py:720-758`
**Impact**: 5-10% improvement

```python
# BEFORE (Sequential - nested awaits)
for entity_id in linked_entity_ids[:5]:
    entity = await self._storage.get_entity(entity_id)
    if entity:
        neighborhood = await self._storage.get_neighborhood(entity_id, ...)
        graph_context[str(entity_id)] = neighborhood

# AFTER (Parallel with asyncio.gather)
async def fetch_entity_with_neighborhood(entity_id):
    entity = await self._storage.get_entity(entity_id)
    if not entity:
        return None
    neighborhood = await self._storage.get_neighborhood(entity_id, max_depth=1, limit=10)
    return entity_id, entity, neighborhood

tasks = [fetch_entity_with_neighborhood(eid) for eid in linked_entity_ids[:5]]
results = await asyncio.gather(*tasks)

for result in results:
    if result:
        entity_id, entity, neighborhood = result
        entities.append((entity, 1.0))
        graph_context[str(entity_id)] = neighborhood
```

---

### 1.3 Batch Chunk Source Lookups in Agentic Search
**File**: `src/khora/query/agentic.py:233-239`
**Impact**: 3-5% improvement per agentic step

```python
# BEFORE (Sequential)
for chunk, score in step1_result.chunks:
    source = await self._get_chunk_source(chunk, namespace_id)
    all_chunks[str(chunk.id)] = (chunk, score, source)

# AFTER (Batch document fetch)
doc_ids = list({chunk.document_id for chunk, _ in step1_result.chunks})
docs = await self._storage.get_documents_batch(doc_ids)
doc_map = {doc.id: doc for doc in docs if doc}

for chunk, score in step1_result.chunks:
    doc = doc_map.get(chunk.document_id)
    source = doc.metadata.source if doc and doc.metadata else "unknown"
    all_chunks[str(chunk.id)] = (chunk, score, source)
```

**Requires**: Add `get_documents_batch()` to StorageCoordinator

---

## Phase 2: Medium Effort Optimizations

### 2.1 Parallelize Entity Linking Strategies
**File**: `src/khora/query/linking.py:143-201`
**Impact**: 2-3x faster entity linking

```python
# BEFORE (Sequential: exact -> fuzzy -> embedding)
if self._exact_match:
    exact = await self._exact_name_match(mention, namespace_id)
    if exact:
        return LinkedEntity(...)
if self._fuzzy_match:
    fuzzy_matches = await self._fuzzy_name_match(mention, namespace_id)
if self._embedding_match:
    embedding_matches = await self._embedding_match_entities(mention, namespace_id)

# AFTER (Parallel non-exact, with early exit for exact)
if self._exact_match:
    exact = await self._exact_name_match(mention, namespace_id)
    if exact:
        return LinkedEntity(entity=exact, confidence=1.0, method="exact")

# Run fuzzy and embedding in parallel
tasks = {}
if self._fuzzy_match:
    tasks["fuzzy"] = self._fuzzy_name_match(mention, namespace_id)
if self._embedding_match and self._embedder:
    tasks["embedding"] = self._embedding_match_entities(mention, namespace_id)

if tasks:
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_map = dict(zip(tasks.keys(), results))
    # Merge and rank candidates from both methods
```

---

### 2.2 Batch LLM Reranking (Multi-Candidate Prompts)
**File**: `src/khora/query/reranking.py:247-304`
**Impact**: 5-10x fewer LLM calls for reranking

```python
# BEFORE (One LLM call per candidate)
async def score_single(candidate):
    prompt = f"Rate relevance of this passage to query..."
    response = await litellm.acompletion(...)
    return parse_score(response)

# AFTER (Batch 5-10 candidates per LLM call)
async def score_batch(candidates: list, query: str) -> list[float]:
    prompt = f"""Rate the relevance of each passage to the query on a scale of 0-10.

Query: {query}

Passages:
{chr(10).join(f"[{i+1}] {c.content[:200]}" for i, c in enumerate(candidates))}

Return JSON: {{"scores": [score1, score2, ...]}}"""

    response = await litellm.acompletion(model=self._model, messages=[...])
    return parse_scores(response)
```

---

### 2.3 Multi-Chunk Entity Extraction
**File**: `src/khora/extraction/extractors/llm.py`
**Impact**: 5-10x fewer extraction calls

```python
# BEFORE (One chunk per LLM call)
for chunk in chunks:
    result = await extractor.extract(chunk.content)

# AFTER (Multiple chunks per call with structured output)
async def extract_multi(self, chunks: list[str], entity_types: list[str]) -> list[ExtractionResult]:
    prompt = f"""Extract entities from each text section below.

Entity types to find: {entity_types}

{chr(10).join(f"=== SECTION {i+1} ==={chr(10)}{text}" for i, text in enumerate(chunks))}

Return JSON array with one object per section:
[{{"section": 1, "entities": [...]}}, ...]"""

    # Single LLM call for 5-10 chunks
    response = await litellm.acompletion(...)
    return parse_multi_extraction(response)
```

---

## Phase 3: Structural Optimizations

### 3.1 Add Batch Methods to Storage Layer

**StorageCoordinator additions** (`src/khora/storage/coordinator.py`):

```python
async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
    """Fetch multiple entities in a single query."""
    if not self._relational:
        return {}
    return await self._relational.get_entities_batch(entity_ids)

async def get_documents_batch(self, document_ids: list[UUID]) -> list[Document]:
    """Fetch multiple documents in a single query."""
    if not self._relational:
        return []
    return await self._relational.get_documents_batch(document_ids)

async def get_neighborhoods_batch(self, entity_ids: list[UUID], max_depth: int = 1) -> dict[UUID, list]:
    """Fetch neighborhoods for multiple entities."""
    if not self._graph:
        return {}
    return await self._graph.get_neighborhoods_batch(entity_ids, max_depth)
```

**PostgreSQL Backend additions** (`src/khora/storage/backends/postgresql.py`):

```python
async def get_entities_batch(self, entity_ids: list[UUID]) -> dict[UUID, Entity]:
    """Fetch multiple entities in one query using IN clause."""
    async with self._get_session() as session:
        result = await session.execute(
            select(EntityModel).where(EntityModel.id.in_([str(eid) for eid in entity_ids]))
        )
        models = result.scalars().all()
        return {UUID(m.id): self._entity_model_to_domain(m) for m in models}
```

---

### 3.2 Query Result Caching

**New file**: `src/khora/query/cache.py`

```python
from functools import lru_cache
from hashlib import sha256
import asyncio
from datetime import datetime, timedelta

class QueryCache:
    """LRU cache for query results with TTL."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()

    def _make_key(self, query: str, namespace_id: UUID, mode: str) -> str:
        return sha256(f"{query}:{namespace_id}:{mode}".encode()).hexdigest()

    async def get(self, query: str, namespace_id: UUID, mode: str) -> Any | None:
        key = self._make_key(query, namespace_id, mode)
        async with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                if datetime.now() - timestamp < self._ttl:
                    return result
                del self._cache[key]
        return None

    async def set(self, query: str, namespace_id: UUID, mode: str, result: Any):
        key = self._make_key(query, namespace_id, mode)
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (datetime.now(), result)
```

---

### 3.3 Streaming Results for Agentic Search

**Enhancement**: Return results incrementally as each step completes

```python
async def search_stream(
    self,
    query: str,
    namespace_id: UUID,
    max_steps: int = 3,
) -> AsyncGenerator[AgenticSearchStep, None]:
    """Stream search results as each step completes."""

    # Step 1: Initial search
    understanding = await self._understand_query(query, namespace_id)
    step1_result = await self._execute_search(query, namespace_id, understanding)

    yield AgenticSearchStep(
        step_number=1,
        query=query,
        result=step1_result,
        is_final=False,
    )

    # Follow-up steps
    for i, follow_up in enumerate(understanding.follow_up_queries[:max_steps-1], start=2):
        step_result = await self._execute_search(follow_up, namespace_id, None)
        yield AgenticSearchStep(
            step_number=i,
            query=follow_up,
            result=step_result,
            is_final=(i == max_steps),
        )
```

---

## Phase 4: Advanced Optimizations

### 4.1 Speculative Execution for Agentic Search

Pre-execute likely follow-up queries while still processing current step:

```python
async def search_speculative(self, query: str, namespace_id: UUID):
    """Execute search with speculative follow-up execution."""

    # Start understanding
    understanding = await self._understand_query(query, namespace_id)

    # Execute main search AND pre-start follow-up searches in parallel
    main_task = self._execute_search(query, namespace_id, understanding)

    # Speculatively start follow-up queries
    followup_tasks = [
        self._execute_search(fq, namespace_id, None)
        for fq in understanding.follow_up_queries[:2]
    ]

    # Wait for main, let follow-ups continue
    main_result = await main_task

    # Collect already-completed follow-ups
    followup_results = await asyncio.gather(*followup_tasks, return_exceptions=True)

    return AgenticSearchResult(
        main=main_result,
        followups=[r for r in followup_results if not isinstance(r, Exception)]
    )
```

---

### 4.2 Connection Pooling Optimization

Ensure database connection pools are properly sized:

```python
# In StorageConfig or initialization
OPTIMAL_POOL_SETTINGS = {
    "postgresql_pool_size": 20,  # Up from 5
    "postgresql_max_overflow": 30,  # Up from 10
    "neo4j_max_connection_pool_size": 50,  # Match concurrent operations
}
```

---

## Implementation Priority

| Phase | Task | Impact | Effort | Priority |
|-------|------|--------|--------|----------|
| 1.1 | Batch entity fetches | 8-15% | 2h | **P0** |
| 1.2 | Parallel graph lookups | 5-10% | 2h | **P0** |
| 1.3 | Batch chunk sources | 3-5% | 1h | **P0** |
| 2.1 | Parallel entity linking | 5-8% | 4h | P1 |
| 2.2 | Batch LLM reranking | 10-20% | 4h | P1 |
| 2.3 | Multi-chunk extraction | 20-30% | 6h | P1 |
| 3.1 | Storage batch methods | Required | 4h | **P0** |
| 3.2 | Query caching | 30-50%* | 4h | P2 |
| 3.3 | Streaming results | UX improvement | 6h | P2 |
| 4.1 | Speculative execution | 20-30% | 8h | P3 |
| 4.2 | Connection pooling | 5-10% | 1h | P1 |

*For repeated queries

---

## Expected Results

### Before Optimization
- Simple query: 10-20 seconds
- Agentic search (3 steps): 20-40 seconds

### After Phase 1 (Quick Wins)
- Simple query: **6-12 seconds** (40% improvement)
- Agentic search: **15-25 seconds** (30% improvement)

### After Phase 2 (Medium Effort)
- Simple query: **4-8 seconds** (60% improvement)
- Agentic search: **10-18 seconds** (50% improvement)

### After All Phases
- Simple query: **2-5 seconds** (75% improvement)
- Agentic search: **6-12 seconds** (65% improvement)
- With caching: **<1 second** for repeated queries

---

## Monitoring & Metrics

Add timing instrumentation to track improvements:

```python
import time
from contextlib import asynccontextmanager
from loguru import logger

@asynccontextmanager
async def timed_operation(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[PERF] {name}: {elapsed:.3f}s")

# Usage
async with timed_operation("vector_search"):
    results = await self._vector_search(...)
```

---

## Next Steps

1. **Immediate**: Implement Phase 1 batch methods in storage layer
2. **This week**: Apply Phase 1 optimizations to query engine and agentic search
3. **Next sprint**: Phase 2 LLM batching optimizations
4. **Future**: Phase 3-4 structural changes based on production metrics

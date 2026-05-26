# Engine Comparison

Khora supports three pluggable engines with different strengths. This guide helps you choose the right engine for your use case.

## Production-readiness by stack

Production-readiness is **per (engine × stack)**, not per engine. The same engine can be production-ready on one storage stack and experimental on another.

| Engine        | PostgreSQL + pgvector + Neo4j  | PostgreSQL + pgvector (no graph) | SQLite + LanceDB (embedded) | SurrealDB (unified)        |
|---------------|--------------------------------|----------------------------------|-----------------------------|----------------------------|
| VectorCypher  | **Production-ready**           | n/a (graph required)             | Experimental     | Experimental               |
| Chronicle     | n/a (graph not required)       | **Production-ready**             | Experimental                | Experimental               |
| Skeleton      | n/a (graph not required)       | Available                        | Experimental     | Experimental               |

- **Production-ready** - qualified for production deployment; covered by integration and e2e tests; documented gotchas have known mitigations.
- **Available** - supported, exercised in tests, but not stamped production-ready. Equivalent retrieval semantics; less load-tested.
- **Experimental** - feature-complete enough for demos, evaluation, and tests on small corpora. Not a deployment story. See the [embedded backend caveats](../configuration.md#embedded-backends-experimental) for the full list of gaps.

The embedded path (SQLite + LanceDB) has a documented scale ceiling: **~1M chunks, ~100k entities, ~500k edges, traversal depth ≤3**. SurrealDB is experimental on multiple fronts (Python SDK on alpha track, KNN unreliable in embedded mode).

## Quick Comparison

| Aspect | VectorCypher (default) | Skeleton Construction | Chronicle |
|--------|------------------------|----------------------|-----------|
| **Primary Focus** | Hybrid retrieval | Temporal events | Conversational memory |
| **Entity Extraction** | Selective (70% default, configurable 0.0–1.0) | Lazy (on-demand) | Full extraction |
| **Core Data Model** | Dual nodes (Entity + Chunk) | Chunks with temporal metadata | SVO events + facts |
| **Time Model** | Bi-temporal + temporal detection (7 categories) | Bi-temporal (`occurred_at` + `ingested_at`) | Triple timestamps + Ebbinghaus decay |
| **LLM Cost** | Medium (~700 calls/1000 docs) at default; ~1000 at `skeleton_core_ratio=1.0` | Lower (~100 calls/1000 docs) | Medium (~700 calls/1000 docs) |
| **Graph Backend** | Required (Neo4j/Neptune/AGE) | Not required | Not required |
| **Search Modes** | Vector + Cypher + BM25 + RRF | Vector + BM25 Hybrid | 4-channel: Semantic + BM25 + Temporal + Entity |
| **Point-in-time queries** | Production-only (PG+Neo4j); not supported on the embedded `sqlite_lance` backend | n/a | n/a |
| **Best For** | Complex multi-hop queries, knowledge bases | Chat history, logs, events | Temporal queries, long conversations |

## Detailed Comparison

### Entity Extraction

**VectorCypher:**
- Skeleton-based selective extraction (KET-RAG): top N% chunks by importance get full LLM extraction; remaining chunks get keyword + co-occurrence edges.
- `skeleton_core_ratio` defaults to 0.70 (70% core). Set to `1.0` for full extraction (legacy GraphRAG behavior).
- Entities stored in Neo4j (or alternate graph backend) for Cypher traversal.

```python
# VectorCypher: skeleton-selective extraction
async with Khora(db_url, engine="vectorcypher") as kb:
    ns = await kb.create_namespace("default")
    result = await kb.remember(content, namespace=ns.namespace_id)
    print(f"Extracted {result.entities_extracted} entities")

# For 100% extraction (legacy GraphRAG behavior):
async with Khora(db_url, engine="vectorcypher",
                 engine_kwargs={"skeleton_core_ratio": 1.0}) as kb:
    ns = await kb.create_namespace("default")
    result = await kb.remember(content, namespace=ns.namespace_id)
```

**Skeleton Construction:**
- Uses skeleton indexing to identify ~10% "core" chunks.
- Only core chunks get LLM extraction.
- Non-core chunks use keyword-based pseudo-entities.
- Lazy expansion during retrieval if needed.

```python
# Skeleton Construction: minimal extraction, skeleton-based
async with Khora(db_url, engine="skeleton") as kb:
    ns = await kb.create_namespace("default")
    result = await kb.remember(content, namespace=ns.namespace_id)
    # Entities only extracted for "core" chunks (high PageRank)
```

**Chronicle:**
- Full extraction on every document - runs the same shared ingest pipeline VectorCypher uses, no skeleton selectivity.
- Extracts SVO events (subject-verb-object tuples) in addition to entities/relationships.
- Stores in PostgreSQL only; no graph backend involvement.

### Time Model

**VectorCypher:**
- Per-category temporal detection (7 categories) on the query side.
- Bi-temporal storage when `occurred_at` is provided via metadata.
- Recency bias in scoring (configurable).

**Skeleton Construction:**
- Bi-temporal model:
  - `occurred_at`: When the event actually happened
  - `ingested_at`: When we learned about it
- Hierarchical time navigation (Year → Quarter → Month → Week → Day).
- Native temporal filtering in queries.

```python
# Skeleton Construction: store event with occurrence time
await kb.remember(
    content,
    namespace=ns_id,
    metadata={"occurred_at": "2024-01-15T00:00:00Z"},
)

# Query: "What happened in January?"
results = await kb.recall(
    "January events",
    namespace=ns_id,
    time_range=("2024-01-01", "2024-01-31"),
)
```

**Chronicle:**
- Triple timestamps: `valid_from`, `valid_to`, `recorded_at`.
- Ebbinghaus forgetting-curve decay applied to relevance scores.
- 4-channel parallel retrieval (semantic + BM25 + temporal + entity).

### Search Capabilities

**VectorCypher:**
- Vector similarity + Cypher graph traversal + BM25 keyword + RRF fusion.
- Query routing determines the search path automatically.

```python
# VectorCypher: query routing determines the search path automatically
results = await kb.recall("Who founded Acme Corp?", namespace=ns_id)  # Multi-hop entity query
results = await kb.recall("CEO recent news", namespace=ns_id)  # Hybrid vector + temporal
```

**Skeleton Construction:**
- Vector + BM25 hybrid only (no graph traversal).
- Time-filtered hybrid search with skeleton expansion.

```python
# Skeleton Construction: time-filtered hybrid search
results = await kb.recall(
    "deployment errors",
    namespace=ns_id,
    time_range=("2024-01-01", "2024-01-31"),
    mode=SearchMode.HYBRID,
)
```

**Chronicle:**
- 4 parallel channels: semantic vector + BM25 keyword + temporal + entity.
- Abstention signals (4 flags + weighted score) to detect low-confidence results.

### Infrastructure Requirements

| Component | VectorCypher | Skeleton | Chronicle |
|-----------|--------------|----------|-----------|
| PostgreSQL + pgvector | Required | Required | Required |
| Graph database (Neo4j/Memgraph/Neptune/AGE) | Required | Not required | Not required |
| LLM API | Required (extraction + embeddings) | Required (embeddings + lazy extraction) | Required (extraction + embeddings) |
| Embedded option (sqlite-lance) | Experimental | Experimental | Experimental |

### Cost Analysis

For 1000 documents averaging 5KB each:

| Operation | VectorCypher (default 70%) | VectorCypher (`skeleton_core_ratio=1.0`) | Skeleton Construction | Chronicle |
|-----------|---------------------------|------------------------------------------|----------------------|-----------|
| Entity extraction | ~700 LLM calls | ~1000 LLM calls | ~100 LLM calls (core only) | ~1000 LLM calls |
| Embedding generation | ~1000 embedding calls | ~1000 embedding calls | ~1000 embedding calls | ~1000 embedding calls |
| Graph operations | Cypher writes | Cypher writes | None | None |
| **Total LLM cost** | **~$0.10–0.20** | **~$0.15–0.30** | **~$0.02–0.05** | **~$0.15–0.30** |

## Use Case Guide

### Choose VectorCypher When...

- **Multi-hop knowledge queries**: "Who works with engineers who worked on the auth project?"
- **Graph-shaped data**: Documents reference entities that reference other entities.
- **Balanced cost/quality**: Default 70% extraction is cheaper than full GraphRAG-style; 100% available via `skeleton_core_ratio=1.0`.
- **Production-grade**: The default-recommended engine; well-tested on PostgreSQL + Neo4j.

### Choose Skeleton Construction When...

- **Time matters most**: Chat logs, event streams, meeting transcripts.
- **Cost is a primary concern**: 5–10× fewer LLM calls.
- **PostgreSQL-only infrastructure**: No graph database available.
- **Freshness is critical**: Bi-temporal model tracks both event time and ingestion time.

### Choose Chronicle When...

- **Conversational memory**: Chat history, support tickets, meeting transcripts.
- **Benchmark-optimized recall**: LongMemEval, LoCoMo, BEAM.
- **Temporal queries**: "What did Alice say last week about the budget?"
- **No graph DB**: Runs on PostgreSQL + pgvector only.

## Switching engines

Switching between VectorCypher and Skeleton Construction is a re-ingest, not a query-side switch - they store data differently (Skeleton has no graph backend). Pick based on your workload:

- Cost-sensitive + PG-only → Skeleton
- Multi-hop entity queries + graph DB available → VectorCypher

For migration from the retired `graphrag` engine to `vectorcypher`, see [migrations.md](../migrations.md#replacing-the-removed-graphrag-engine).

## Hybrid Approach

For complex use cases, consider running multiple engines against the same data store:

```python
# Example: dual-engine setup (conceptual)
async def hybrid_query(query: str, ns_id):
    async with Khora(db_url, engine="vectorcypher") as kg_kb:
        async with Khora(db_url, engine="skeleton") as temporal_kb:
            # Route entity queries to VectorCypher
            if has_entity_intent(query):
                return await kg_kb.recall(query, namespace=ns_id)
            # Route temporal queries to Skeleton
            elif has_temporal_intent(query):
                return await temporal_kb.recall(query, namespace=ns_id)
```

## Performance Benchmarks

| Metric | VectorCypher (default) | Skeleton Construction | Notes |
|--------|------------------------|----------------------|-------|
| Ingestion (1000 docs) | ~30 minutes | ~3 minutes | Selective vs lazy extraction |
| Retrieval (single query) | ~200ms | ~150ms | Vector + Cypher + BM25 vs Vector + BM25 |
| Entity lookup | ~50ms | ~100ms | Direct graph vs lazy expansion |
| Multi-hop traversal | ~100ms | Limited | Graph backend native |
| Time-range filtering | ~200ms | ~75ms | Skeleton optimized |
| Storage (1000 docs) | ~120MB | ~80MB | VectorCypher adds graph state |

## Related Documentation

- [VectorCypher Engine](vectorcypher-engine.md) - default, hybrid retrieval.
- [Skeleton Construction Engine](skeleton-engine.md) - temporal-first, no graph DB.
- [Chronicle Engine](chronicle-engine.md) - conversational + temporal.
- [Temporal Model](temporal-model.md) - bi-temporal design details.
- [Hybrid Search](hybrid-search.md) - vector + BM25 fusion primitive.
- [References](../REFERENCES.md) - research papers and inspirations.

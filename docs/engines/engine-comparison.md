# Engine Comparison

Khora supports four pluggable engines with different strengths. This guide helps you choose the right engine for your use case.

## Quick Comparison

| Aspect | VectorCypher (default) | GraphRAG | Skeleton Construction | Chronicle |
|--------|------------------------|----------|----------------------|-----------|
| **Primary Focus** | Hybrid retrieval | Knowledge graphs | Temporal events | Conversational memory |
| **Entity Extraction** | Skeleton (70%) | Upfront (all documents) | Lazy (on-demand) | Full extraction |
| **Core Data Model** | Dual nodes (Entity + Chunk) | Entities & relationships | Chunks with temporal metadata | SVO events + facts |
| **Time Model** | Bi-temporal + temporal detection (7 categories) | Single (`created_at`) | Bi-temporal (`occurred_at` + `ingested_at`) | Triple timestamps + Ebbinghaus decay |
| **LLM Cost** | Medium (~700 calls/1000 docs) | Higher (~1000 calls/1000 docs) | Lower (~100 calls/1000 docs) | Medium (~700 calls/1000 docs) |
| **Graph Backend** | Required (Neo4j/Neptune/AGE) | Required (Neo4j/Memgraph) | Not required | Not required |
| **Search Modes** | Vector + Cypher + BM25 + RRF | Vector + Graph + Keyword | Vector + BM25 Hybrid | 4-channel: Semantic + BM25 + Temporal + Entity |
| **Point-in-time queries** | Production-only (PG+Neo4j); not supported on the embedded `sqlite_lance` backend | n/a | n/a | n/a |
| **Best For** | Complex multi-hop queries | Knowledge bases | Chat history, logs, events | Temporal queries, long conversations |

## Detailed Comparison

### Entity Extraction

**GraphRAG:**
- Extracts entities from every document during ingestion
- Rich relationship extraction with typed edges
- Cross-document entity deduplication and resolution
- Full knowledge graph construction

```python
# GraphRAG: Entity extraction happens during remember()
result = await lake.remember(content)
print(f"Extracted {result.entities_extracted} entities")
print(f"Created {result.relationships_created} relationships")
```

**Skeleton Construction:**
- Uses skeleton indexing to identify ~10% "core" chunks
- Only core chunks get LLM extraction
- Non-core chunks use keyword-based pseudo-entities
- Lazy expansion during retrieval if needed

```python
# Skeleton Construction: Minimal extraction, skeleton-based
result = await lake.remember(content)
# Entities only extracted for "core" chunks (high PageRank)
```

### Time Model

**GraphRAG:**
- Single timestamp: `created_at` (when document was ingested)
- Recency bias in search (configurable decay)
- No distinction between event time and ingestion time

**Skeleton Construction:**
- Bi-temporal model:
  - `occurred_at`: When the event actually happened
  - `ingested_at`: When we learned about it
- Hierarchical time navigation (Year → Quarter → Month → Week → Day)
- Native temporal filtering in queries

```python
# Skeleton Construction: Store event with occurrence time
await lake.remember(
    "Alice joined the team",
    metadata={"occurred_at": "2024-01-15T09:00:00Z"}
)

# Query: "What happened in January?"
results = await lake.recall(
    "team changes",
    temporal_filter={"occurred_after": "2024-01-01", "occurred_before": "2024-02-01"}
)
```

### Search Capabilities

**GraphRAG:**

| Mode | Description |
|------|-------------|
| `VECTOR` | Semantic similarity via embeddings |
| `GRAPH` | Entity-centric traversal, relationship exploration |
| `KEYWORD` | BM25 full-text search |
| `HYBRID` | All three combined with RRF |

```python
# GraphRAG: Entity exploration
entities = await lake.list_entities(entity_type="PERSON")
related = await lake.find_related_entities(entity_id, max_depth=2)
```

**Skeleton Construction:**

| Mode | Description |
|------|-------------|
| `VECTOR` | Semantic similarity via embeddings |
| `HYBRID` | Vector + BM25 with configurable alpha |

```python
# Skeleton Construction: Time-filtered hybrid search
results = await lake.recall(
    query,
    hybrid_alpha=0.7,  # 70% vector, 30% BM25
    temporal_filter={"author": "alice", "channel": "engineering"}
)
```

### Entity Extraction (VectorCypher)

**VectorCypher:**
- Uses skeleton indexing with a higher core ratio (default 70%)
- Core chunks get full LLM entity extraction; non-core use keywords
- Entities and chunks stored as dual nodes in Neo4j (`MENTIONED_IN` links)
- Per-complexity fusion weights tune how much graph signal to blend in

```python
# VectorCypher: Skeleton-based extraction with graph storage
result = await lake.remember(content)
# 70% of chunks get full entity extraction (configurable)
# Entities stored in Neo4j for Cypher traversal
```

### Infrastructure Requirements

**GraphRAG:**

```
PostgreSQL (required)
├── Documents & metadata
├── Event sourcing
└── pgvector embeddings

Neo4j/Kuzu/Memgraph (required)
├── Entity nodes
├── Relationship edges
└── Graph traversal
```

**Skeleton Construction:**

```
PostgreSQL (required)
├── Documents & metadata
├── pgvector embeddings
├── BM25 full-text (tsvector)
└── BRIN temporal indexes
```

**VectorCypher:**

```
PostgreSQL (required)
├── Documents & metadata
└── pgvector embeddings

Neo4j (required)
├── Entity nodes
├── Chunk nodes
├── MENTIONED_IN relationships
└── Graph traversal
```

**Alternative: SurrealDB (any engine)**

```
SurrealDB (single database)
├── Documents & metadata (relational)
├── Vector embeddings (native vector search)
├── Entity/relationship graph (native graph)
└── Event sourcing
```

SurrealDB can serve as all three backends in a single database, simplifying deployment. It's available as an alternative for any engine, though the PostgreSQL + Neo4j stack is more mature for production use.

### Embedded backend (`sqlite_lance`)

The embedded `sqlite_lance` backend (SQLite + LanceDB) is intended for evaluation, tests, and small single-process deployments. It does **not** support point-in-time / historical queries: VectorCypher's `_version_filter_entities` reads `version_valid_from` / `version_valid_to` columns that exist only in the Neo4j Entity-version graph. Calling `recall()` with a target date (either via `start_time` / `end_time` arguments or a query whose temporal detection produces an `EXPLICIT` category date) on `sqlite_lance` raises `NotImplementedError` immediately, before any storage I/O. Use the production stack (PostgreSQL+Neo4j) for historical/temporal queries (DYT-3550).

### Search Capabilities (VectorCypher)

**VectorCypher:**

| Mode | Description |
|------|-------------|
| `SIMPLE` | Vector-only search (fastest, no graph traversal) |
| `MODERATE` | Shallow graph expansion (depth=1) + vector fusion |
| `COMPLEX` | Deep graph traversal (depth=2-3) + weighted RRF fusion |

```python
# VectorCypher: Query routing determines the search path automatically
results = await lake.recall(
    "How are Alice and Bob connected through projects?",
    graph_depth=2,  # Or let the router decide
)
```

### Cost Analysis

For 10,000 documents:

| Operation | GraphRAG | Skeleton Construction | VectorCypher |
|-----------|----------|----------------------|--------------|
| Entity extraction calls | ~10,000 | ~1,000 | ~7,000 |
| Relationship extraction calls | ~10,000 | ~0 | ~7,000 |
| Embedding calls | ~10,000 | ~10,000 | ~10,000 |
| **Total LLM calls** | **~30,000** | **~11,000** | **~24,000** |

*Note: VectorCypher extraction counts assume the default 70% core ratio. Actual costs depend on document length, extraction complexity, and LLM pricing.*

## Use Case Guide

### Choose GraphRAG When...

- **Building a knowledge base**: Product documentation, research papers, FAQs
- **Entity relationships matter**: "Who reports to whom?", "What products does Company X make?"
- **Graph exploration needed**: Multi-hop traversal, relationship discovery
- **Accuracy over cost**: Willing to pay for comprehensive extraction
- **Long-term reference**: Content doesn't change frequently

**Example Use Cases:**
- Corporate knowledge management
- Research paper analysis
- Product catalog with relationships
- Organizational charts and hierarchies
- Documentation with cross-references

### Choose Skeleton Construction When...

- **Time is the primary dimension**: Chat logs, event streams, meeting notes (note: VectorCypher also handles temporal queries well via its TemporalDetector)
- **Cost optimization critical**: Large volumes, budget constraints
- **Freshness matters**: Recent events more important than old
- **Structured filtering needed**: By author, channel, tags, time ranges
- **Simple infrastructure**: Don't want to manage Neo4j
- **Real-time ingestion**: Streaming data, continuous updates

**Example Use Cases:**
- Slack/Discord message archives
- Meeting transcripts with timestamps
- Support ticket history
- Application logs and events
- News and social media monitoring
- Customer interaction history

### Choose VectorCypher When...

- **Multi-hop queries are common**: "How are X and Y connected through Z?"
- **Graph traversal is essential**: Organizational hierarchies, deal chains, team structures
- **Relationship discovery matters**: Finding implicit connections across data sources
- **Neo4j is available**: VectorCypher requires Neo4j (not optional)
- **Temporal reasoning needed**: Cascade temporal detection classifies queries into 7 categories with category-specific retrieval (recency boost, sorting, decay override)
- **Balanced cost/quality**: Willing to pay more than Skeleton but less than GraphRAG

**Example Use Cases:**
- CRM relationship mapping (deals → contacts → companies)
- Organizational knowledge graphs with multi-hop exploration
- Research collaboration networks
- Supply chain relationship tracking
- Any domain where "connections between things" is the primary query pattern

## Migration Considerations

### From GraphRAG to Skeleton Construction

1. **Graph features lost**: Entity relationships won't be pre-computed
2. **Query changes**: `SearchMode.GRAPH` not available; use `HYBRID`
3. **Entity methods unavailable**: `list_entities()`, `find_related_entities()`
4. **Temporal benefits**: Add `occurred_at` metadata for time-based queries

```python
# Before (GraphRAG)
results = await lake.recall(query, mode=SearchMode.GRAPH)
entities = await lake.find_related_entities(entity_id)

# After (Skeleton Construction)
results = await lake.recall(
    query,
    mode=SearchMode.HYBRID,
    temporal_filter={"occurred_after": "2024-01-01"}
)
# Entity relationships must be handled differently
```

### From Skeleton Construction to GraphRAG

1. **Temporal metadata preserved**: `occurred_at` becomes document metadata
2. **Re-extraction needed**: All documents must be re-processed for entities
3. **Infrastructure addition**: Neo4j/Kuzu/Memgraph required
4. **Higher initial cost**: Full entity extraction on migration

```python
# Before (Skeleton Construction)
results = await lake.recall(query, temporal_filter={...})

# After (GraphRAG)
results = await lake.recall(query, mode=SearchMode.HYBRID)
entities = await lake.list_entities(entity_type="PERSON")
```

## Hybrid Approach

For some use cases, you might use both engines:

1. **Skeleton Construction for ingestion**: Fast, cost-efficient initial storage
2. **Background GraphRAG enrichment**: Async entity extraction for important documents
3. **Query routing**: Time-sensitive queries → Skeleton Construction, entity queries → GraphRAG

```python
# Example: Dual-engine setup (conceptual)
async with MemoryLake(db_url, engine="skeleton") as skeleton_lake:
    async with MemoryLake(db_url, engine="graphrag") as graphrag_lake:
        # Fast ingestion via Skeleton Construction
        await skeleton_lake.remember(content, title="Event")

        # Background enrichment for important content
        if is_important(content):
            await graphrag_lake.remember(content, title="Event")
```

## Configuration Examples

### GraphRAG Setup

```yaml
# genesis.yaml
engine:
  name: graphrag

# Environment
KHORA_DATABASE_URL=postgresql://localhost/khora
KHORA_NEO4J_URL=bolt://localhost:7687
```

### Skeleton Construction Setup

```yaml
# genesis.yaml
engine:
  name: skeleton
  backend: pgvector  # or weaviate

temporal:
  default_lookback_days: 90
  hierarchy_enabled: true

query:
  hybrid_alpha: 0.7
  recency_decay_days: 30

# Environment
KHORA_DATABASE_URL=postgresql://localhost/khora
# No Neo4j required
```

## Performance Benchmarks

*Benchmarks on 10,000 documents, single-node deployment:*

| Metric | GraphRAG | Skeleton Construction | Notes |
|--------|----------|----------------------|-------|
| Ingestion time | ~45 min | ~8 min | Entity extraction overhead |
| Query latency (p50) | ~200ms | ~150ms | Graph traversal adds latency |
| Query latency (p99) | ~800ms | ~400ms | |
| Storage size | ~2.5 GB | ~1.5 GB | Graph storage overhead |
| Memory usage | ~4 GB | ~2 GB | Neo4j in-memory graph |

*Note: Actual performance varies by hardware, document size, and query patterns.*

## Related Documentation

- [Chronicle Engine](chronicle-engine.md) — Temporal-semantic memory with 4-channel retrieval
- [Skeleton Construction Engine](skeleton-engine.md) — Temporal-first, cost-optimized engine
- [VectorCypher Engine](vectorcypher-engine.md) — Hybrid vector + Cypher graph traversal
- [Temporal Model](temporal-model.md) — Bi-temporal design deep dive
- [Skeleton Indexing](skeleton-indexing.md) — Cost optimization via PageRank
- [Hybrid Search](hybrid-search.md) — Vector + BM25 fusion

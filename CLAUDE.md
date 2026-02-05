# Khora

Memory Lake combining knowledge graphs (Neo4j/Kuzu/Memgraph), vector database (pgvector), and PostgreSQL for unified knowledge storage and retrieval.

## Commands

```bash
make test              # Run tests (pytest, coverage ≥50%)
make format            # Format code (black, isort, ruff)
make lint              # Lint + typecheck
uv run khora serve --reload  # Dev server
uv run alembic upgrade head  # Run migrations
```

## Architecture

```
MemoryLake (facade) → Engine (graphrag | skeleton) → StorageCoordinator
                                                  ├── PostgreSQL (documents, tenancy)
                                                  ├── pgvector (embeddings)
                                                  └── Graph backend (entities, relationships)
```

**Pluggable Engines:**
- **GraphRAG** (`engine="graphrag"`) - Full knowledge graph extraction, requires Neo4j/Kuzu
- **Skeleton Construction** (`engine="skeleton"`) - Temporal-first, cost-optimized, Neo4j optional

**Key entry points:**
- `src/khora/memory_lake.py` - Public API: `remember()`, `recall()`, `forget()`, `remember_batch()`
- `src/khora/engines/graphrag/engine.py` - GraphRAG engine (default)
- `src/khora/engines/skeleton/engine.py` - Skeleton Construction temporal-first engine
- `src/khora/engines/skeleton/temporal_edges.py` - Bi-temporal edge storage
- `src/khora/engines/skeleton/time_hierarchy.py` - Hierarchical time graph (Year→Quarter→Month→Week→Day)
- `src/khora/engines/skeleton/skeleton.py` - PageRank-based skeleton indexing
- `src/khora/engines/skeleton/backends/pgvector.py` - PostgreSQL+pgvector backend
- `src/khora/engines/skeleton/backends/weaviate.py` - Weaviate backend
- `src/khora/query/engine.py` - HybridQueryEngine (search pipeline)
- `src/khora/pipelines/flows/ingest.py` - Document ingestion flow

**Multi-tenancy:** Organization → Workspace → MemoryNamespace

## Usage

```python
async with MemoryLake("postgresql://...") as lake:
    await lake.remember("content", title="Doc")
    results = await lake.recall("query")
```

## Key Patterns

- **Engines are pluggable** - See `khora.engines.protocol.MemoryEngineProtocol`
- **Graph backends are interchangeable** - All implement `GraphBackend` in `storage/backends/base.py`
- **Extraction skills are YAML-defined** - See `src/khora/extraction/skills/definitions/`
- **Config via env vars** - `KHORA_DATABASE_URL`, `KHORA_NEO4J_URL`, use `__` for nesting (e.g., `KHORA_QUERY__ENABLE_HYDE=true`)

## Engine Selection Guide

| Use Case | Engine | Reason |
|----------|--------|--------|
| Knowledge bases | `graphrag` | Rich entity/relationship extraction |
| Entity exploration | `graphrag` | Graph traversal support |
| Multi-hop queries | `vectorcypher` | Vector + Cypher graph traversal |
| Complex relationships | `vectorcypher` | RRF fusion, query routing |
| Chat/message history | `skeleton` | Skeleton Construction: Temporal-first, structured filters |
| Event streams/logs | `skeleton` | Skeleton Construction: Bi-temporal model |
| Cost-sensitive apps | `skeleton` | Skeleton Construction: 5-10x fewer LLM calls |
| Simple infrastructure | `skeleton` | Skeleton Construction: No Neo4j required |

**Skeleton Construction engine features:**
- Bi-temporal: `occurred_at` (event time) vs `ingested_at` (system time)
- Skeleton indexing: PageRank identifies ~10% core chunks for LLM extraction
- Time hierarchy: Year → Quarter → Month → Week → Day for range queries
- Hybrid search: Vector + BM25 with configurable `hybrid_alpha`
- Temporal filters: `author`, `channel`, `tags`, time ranges

## Testing

```bash
make test                           # Full suite
uv run pytest tests/unit/test_memory_lake.py -v  # Single file
uv run pytest -k "test_remember" -v              # By name
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`

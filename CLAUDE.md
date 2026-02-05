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
MemoryLake (facade) → Engine (graphrag default) → StorageCoordinator
                                                  ├── PostgreSQL (documents, tenancy)
                                                  ├── pgvector (embeddings)
                                                  └── Graph backend (entities, relationships)
```

**Key entry points:**
- `src/khora/memory_lake.py` - Public API: `remember()`, `recall()`, `forget()`, `remember_batch()`
- `src/khora/engines/graphrag/engine.py` - Default engine implementation
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

## Testing

```bash
make test                           # Full suite
uv run pytest tests/unit/test_memory_lake.py -v  # Single file
uv run pytest -k "test_remember" -v              # By name
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`

# Khora

Memory Lake library: knowledge graphs + vector search + PostgreSQL for unified knowledge storage. **Library, not an application.**

## Commands

```bash
make test              # pytest, coverage ≥30%
make format            # black, isort, ruff
make lint              # ruff + ty typecheck
make dev               # Start postgres + neo4j
uv run alembic upgrade head                       # Run migrations
uv run khora ontology construct --source <path>   # AI ontology generation
uv run khora ontology validate <file.yaml>        # Validate ontology YAML
uv run khora ontology preview <file.yaml>         # Rich preview
```

## Architecture

```
MemoryLake → Engine (graphrag | skeleton | vectorcypher) → StorageCoordinator
                                        ├── PostgreSQL (documents, tenancy)
                                        ├── pgvector (embeddings)
                                        └── Graph backend (Neo4j | SurrealDB | ...)

Traditional stack: PostgreSQL + pgvector + Neo4j  (three databases)
Unified stack:     SurrealDB                      (one database, all roles)
```

- **Engines:** implement `MemoryEngineProtocol` in `engines/protocol.py`
- **Graph backends:** Neo4j, SurrealDB, Memgraph, ArcadeDB — implement `GraphBackend` in `storage/backends/base.py`
- **SurrealDB:** unified backend (graph + vector + relational). Modes: `memory://`, `surrealkv://` (embedded), `ws://` (remote). Set `backend: surrealdb` in config
- **Extraction skills:** YAML-defined in `extraction/skills/builtin/`. Generate with `khora ontology construct`
- **Config:** env vars with `KHORA_` prefix, `__` nesting (e.g., `KHORA_QUERY__ENABLE_HYDE=true`)

## Public API

Exported from `khora/__init__.py`:

```python
from khora import (
    MemoryLake,           # Primary interface (async context manager via memory_lake())
    RememberResult,       # Result of remember()
    RecallResult,         # Result of recall()
    BatchResult,          # Result of remember_batch()
    Stats,                # Namespace statistics
    LLMUsage,             # Token/cost tracking (consumed by Poros/Peras — DYT-645)
    SearchMode,           # VECTOR | GRAPH | HYBRID | ALL
    KhoraConfig,          # Main Pydantic configuration
    DocumentSource,       # Lightweight doc metadata for attribution
    ExpertiseConfig,      # Domain expertise definition (ADR-022 stable API)
    EntityTypeConfig,     # Entity type definition
    RelationshipTypeConfig,  # Relationship type definition
    create_engine,        # Instantiate engine by name
    list_engines,         # ["graphrag", "skeleton", "vectorcypher"]
    register_engine,      # Register custom engine class
)
```

### MemoryLake Methods

| Method | Purpose |
|--------|---------|
| `remember(content, *, namespace, expertise, ...)` | Store content, extract entities |
| `remember_batch(documents, *, namespace, expertise, ...)` | Batch ingest with optimization |
| `recall(query, *, namespace, limit, mode, min_similarity, ...)` | Retrieve memories |
| `forget(document_id, *, namespace)` | Remove a memory |
| `create_namespace()` / `get_namespace()` / `get_namespace_by_stable_id()` | Namespace management |
| `get_entity()` / `list_entities()` / `search_entities()` / `find_related_entities()` | Entity operations |
| `get_document()` / `list_documents()` | Document retrieval |
| `stats(*, namespace)` | Namespace statistics |
| `health_check()` | Backend health status |
| `connect()` / `disconnect()` | Lifecycle (or use `async with memory_lake()`) |

### Result Types (frozen dataclasses)

**LLMUsage** — token tracking for cost attribution (Poros/Peras consume this):
- `operation`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_ms`, `batch_size`

**RememberResult** — single document ingest result:
- `document_id`, `chunks`, `entities`, `relationships`, `llm_usage: list[LLMUsage]`

**BatchResult** — batch ingest result:
- `total`, `processed`, `skipped`, `failed`, `chunks`, `entities`, `relationships`, `metadata`, `llm_usage: list[LLMUsage]`

**RecallResult** — query result:
- `query`, `namespace_id`, `chunks`, `entities`, `context_text` (pre-formatted for LLM), `relationships` (VectorCypher only), `llm_usage`

**Stats** — namespace counters

## Key Entry Points

- `memory_lake.py` — `remember()`, `recall()`, `forget()`, `remember_batch()`. Accepts `expertise: ExpertiseConfig`
- `extraction/skills/base.py` — `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` (ADR-022 stable)
- `storage/coordinator.py` — `transaction()` for atomic multi-backend ops
- `storage/backends/base.py` — `GraphBackend` protocol (implement for new backends)
- `storage/backends/surrealdb/` — Unified SurrealDB backend
- `db/models.py` — SQLAlchemy ORM (UUID columns use `as_uuid=True`)
- `_accel.py` — Rust/NumPy acceleration (MMR, cosine, pagerank, entity resolution, community detection, temporal)
- `cli/ontology/` — Ontology construction: `commands.py`, `flow.py`, `inference/`, `sources/`, `sampling/`
- `pipelines/flows/ingest.py` — Document ingestion pipeline (3-phase: stage → enrich → expand)
- `db/migrations/env.py` — Alembic with advisory locking
- `config/schema.py` — `KhoraConfig` Pydantic settings (storage, LLM, pipeline, query, tenancy)
- `telemetry/` — Optional PostgreSQL-backed telemetry collector + `@trace` decorator

## Entity Extraction Pipeline

3-phase pipeline in `pipelines/flows/ingest.py`:

```
Phase 1: Stage — checksum-based change detection, skip unchanged docs
Phase 2: Enrich (parallel per document):
  ├── Chunk (fixed | semantic | recursive | conversation)
  ├── Embed (LiteLLM, shared embedder) ─┐ concurrent
  ├── Extract entities (LLM) ───────────┘
  ├── Selective extraction — score chunks by importance, top-K to LLM
  ├── Entity deduplication (smart O(1) index or fuzzy resolution)
  ├── Co-occurrence edges + optional cross-chunk relationships
  └── Store (chunks → pgvector, entities → graph, embeddings → vector)
Phase 3: Semantic Expansion (optional):
  ├── Cross-tool entity unification
  └── Relationship inference (smart | batch | incremental | none)
```

**Chunkers** (`extraction/chunkers/`): `FixedChunker` (token-based), `SemanticChunker` (sentence boundaries), `RecursiveChunker` (hierarchical), `ConversationChunker` (speaker-aware)

**Builtin skills** (`extraction/skills/builtin/`): `general.yaml` (9 entity types, 21 relationship types), `slack.yaml` (Slack-optimized with channel/message correlation)

**Entity resolution** (`extraction/entity_resolution.py`): 5-strategy dedup (exact → alias → attribute → embedding → fuzzy). Per-type thresholds (PERSON 0.92, DATE 0.95, default 0.85)

**Semantic expansion** (`extraction/expansion/`): `SemanticExpander` orchestrates `CrossToolUnifier` + `RelationshipInferrer` + `RuleEngine`

## Configuration

`KhoraConfig` (Pydantic BaseSettings, env prefix `KHORA_`, nested delimiter `__`):

| Section | Key Settings |
|---------|-------------|
| **storage** | `backend` (`postgres`/`surrealdb`), graph config (Neo4j/Memgraph/ArcadeDB/SurrealDB), vector config (pgvector/ArcadeDB/SurrealDB), PostgreSQL pool tuning, HNSW parameters |
| **llm** | `model` (default `gpt-4o-mini`), `embedding_model` (`text-embedding-3-small`), `extraction_model`, `embedding_dimension` (1536), temperature, max_tokens, max_concurrent_llm_calls, LiteLLM router config |
| **pipeline** | `chunking_strategy`, `chunk_size` (512), `extract_entities`, `selective_extraction` (true), `extraction_importance_ratio` (0.7), `skip_embedding_entity_types` (DATE, URL, EMAIL) |
| **query** | `default_mode` (hybrid), fusion weights (vector 0.5, graph 0.3, keyword 0.2), reranking, HyDE, recency bias, entity linking, BM25, temporal resolver |
| **tenancy** | `default_mode` (shared/isolated), `enforce_namespace` |
| Top-level | `database_url`, `neo4j_url`, `debug`, `telemetry_database_url`, `telemetry_service_name` |

## Engine Selection

| Use Case | Engine | Requires |
|----------|--------|----------|
| Knowledge bases, entity exploration | `graphrag` | Neo4j or SurrealDB |
| Multi-hop queries, complex relationships | `vectorcypher` | Neo4j or SurrealDB |
| Chat history, cost-sensitive | `skeleton` | Graph backend optional |

## Acceleration (`_accel.py`)

3-tier: Rust (`khora-accel` wheel, Pyo3 + rayon) → NumPy/RapidFuzz → pure Python. Override: `KHORA_ACCEL_BACKEND` env var (`rust`/`numpy`/`python`).

Key functions: `cosine_similarity`, `batch_cosine_similarity`, `pairwise_cosine_above_threshold`, `levenshtein_similarity`, `batch_levenshtein`, `pagerank`, `reciprocal_rank_fusion`, `weighted_rrf`, `mmr_diversity_select`, `resolve_entities_batch`, `normalize_entity_name`, `detect_temporal_category`, `batch_temporal_filter`, `detect_communities`, `deduplicate_chunks`, `extract_keywords`

## Dependencies & Extras

Core: `sqlalchemy[asyncio]`, `asyncpg`, `pgvector`, `neo4j`, `litellm`, `tiktoken`, `sentence-transformers`, `pydantic-settings`, `click`, `rich`, `loguru`, `tenacity`, `dateparser`, `jinja2`, `pyyaml`

| Extra | Install | Purpose |
|-------|---------|---------|
| `surrealdb` | `pip install khora[surrealdb]` | SurrealDB unified backend |
| `embedded` | `pip install khora[embedded]` | SurrealDB embedded mode |
| `logfire` | `pip install khora[logfire]` | Logfire observability |
| `nlp` | `pip install khora[nlp]` | spaCy NLP |
| `accel` | `pip install khora[accel]` | RapidFuzz CPU acceleration |
| `rust` | `pip install khora[rust]` | Rust-accelerated ops (khora-accel) |
| `memgraph` | `pip install khora[memgraph]` | Memgraph backend |
| `arcadedb` | `pip install khora[arcadedb]` | ArcadeDB backend |
| `graph-all` | `pip install khora[graph-all]` | All graph backends |
| `all-backends` | `pip install khora[all-backends]` | All backends |
| `reranking` | `pip install khora[reranking]` | Neural reranking |
| `dev` | `pip install khora[dev]` | Testing & linting |

## Testing

```bash
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest -k "test_remember" -v         # By name
uv run pytest tests/unit/test_memory_lake.py  # Single file
```

Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`. Async: `asyncio_mode = "auto"`.

## Releasing

Versions from git tags — no manual bumps. `git tag vX.Y.Z && git push origin vX.Y.Z` triggers publish workflows. See [`docs/RELEASE.md`](docs/RELEASE.md).

## Gotchas

### Migrations & Schema
- **Never use `create_tables()`** — deprecated, bypasses Alembic. Use `run_migrations()` or `MemoryLake(run_migrations=True)`. Create new migrations with `uv run alembic revision --autogenerate -m "desc"`
- **Version table:** `khora_alembic_version` (not `alembic_version`) — avoids conflicts with downstream apps
- **Advisory lock:** `run_migrations()` uses `pg_advisory_xact_lock` (ID `6001515088189075507`), 60s timeout
- **Migrations bundled** in `src/khora/db/migrations/`, not `alembic/`. Root `alembic.ini` is dev-only
- **Skip-ahead:** When multiple services share a DB with different Khora versions, `run_migrations()` detects if the DB revision is unknown (ahead) and skips gracefully — returns `MigrationResult(success=True, skipped=True)`. Signaled via `_DatabaseAheadError` from `env.py` to `session.py`

### UUID & Type Handling
- **ORM:** all 52 UUID columns use `as_uuid=True` — native `uuid.UUID`, never `str()` wrap
- **Graph boundary:** Neo4j/Memgraph need `str(uuid)` at the driver boundary only
- **SurrealDB:** `RecordID` accepts UUID objects directly (no `str()` needed since SDK 2.0)

### Backend Specifics
- **Shared engine pools:** `StorageFactory` caches by URL. Shared-engine backends skip `dispose()`
- **Transactions:** `async with coordinator.transaction() as txn:` for atomic multi-backend ops
- **SurrealDB unified:** all four adapters share one `SurrealDBConnection`. Coordinator skips duplicate writes
- **SurrealDB schema:** declarative (`DEFINE IF NOT EXISTS`), auto-initializes on `connect()`. No Alembic
- **SurrealDB SDK:** pinned `>=2.0.0a1` for 3.x support. Install: `pip install khora[surrealdb]`
- **SurrealDB KNN broken:** `<|K|>` unreliable in embedded mode. Uses brute-force cosine + HNSW instead
- **SurrealDB entity gate:** `_SurrealDBEntityKeyGate` serializes concurrent upserts by (ns, name, type) key

### Extraction & Search
- **Pre-normalized embeddings** — L2-normalized at ingest. Uses `batch_dot_product` (3x faster than cosine)
- **Entity unique constraint** — `(namespace_id, name, entity_type)` UNIQUE in both PostgreSQL and SurrealDB
- **Namespace versioning** — dual IDs: `id` (row-level) vs `namespace_id` (stable). Public API uses `namespace_id`, resolves to `id` via indexed lookup
- **Selective extraction** — KET-RAG style: scores chunk importance, sends top 70% to LLM, rest get co-occurrence edges only
- **Entity resolution** — multi-strategy dedup with per-type thresholds (PERSON 0.92, DATE 0.95, default 0.85)
- **Semantic expansion** — optional cross-tool entity unification + relationship inference (4 modes: smart/batch/incremental/none)

### Optional Dependencies
- **spaCy:** `_HAS_SPACY` flag, falls back to regex sentence splitting
- **Logfire:** `_HAS_LOGFIRE` flag, `trace_span()` yields no-op when absent. Install: `pip install khora[logfire]`
- **`@trace` decorator:** `from khora.telemetry import trace`. Zero overhead when logfire absent
- **Telemetry collector:** `KHORA_TELEMETRY_DATABASE_URL` enables PostgreSQL-backed event recording. Without it, `NoOpCollector` is used (zero cost)

### Downstream
- `genesis` and `khora-benchmarks` depend on khora. `lake.storage` is a stable public API
- **LLMUsage contract:** `LLMUsage` fields are consumed by Poros/Peras for cost tracking (DYT-645) — changes require coordination
- **ExpertiseConfig contract:** ADR-022 stable API — `ExpertiseConfig`, `EntityTypeConfig`, `RelationshipTypeConfig` changes require coordination
- `scripts/` vendored from TTOJ — skip in audits

# Configuration

Khora is configured through environment variables prefixed `KHORA_` or a `KhoraConfig` instance constructed programmatically. Both paths are backed by the same pydantic-settings model in `src/khora/config/schema.py`.

## Two ways to configure

### Environment variables

All settings use the `KHORA_` prefix with single-underscore separators for nested fields. Examples:

```bash
KHORA_DATABASE_URL=postgresql://khora:khora@localhost:5434/khora
KHORA_NEO4J_URL=bolt://neo4j:pleaseletmein@localhost:7688
KHORA_LLM_MODEL=gpt-4o
KHORA_QUERY_ENABLE_HYDE=auto
KHORA_QUERY_DEFAULT_MODE=hybrid
KHORA_EXTRACTION_BATCH_SIZE=5
```

Legacy double-underscore nesting (`KHORA_STORAGE__POSTGRESQL_URL`) is still accepted for backwards compatibility.

### Programmatic

```python
from khora import KhoraConfig, Khora
from khora.config.schema import StorageSettings, LLMSettings

config = KhoraConfig(
    database_url="postgresql://khora@localhost/khora",
    neo4j_url="bolt://localhost:7687",
    llm=LLMSettings(model="gpt-4o", embedding_model="text-embedding-3-small"),
)

async with Khora(config) as kb:
    ...
```

Programmatic values take priority over environment variables.

## Install extras

| Extra | Purpose | Pulls in |
|---|---|---|
| *(default)* | Core: PostgreSQL + pgvector + Neo4j driver + litellm | — |
| `surrealdb` | **[experimental]** Unified SurrealDB backend (embedded or remote). SDK on alpha track; KNN unreliable in embedded mode | `surrealdb>=2.0.0a1` |
| `embedded` | Alias for `surrealdb` (zero-infrastructure path) — **experimental** | `surrealdb>=2.0.0a1` |
| `kuzu` | **[deprecated in 0.9.0, removal scheduled for 0.10]** Embedded graph backend | `kuzu>=0.11.3` |
| `memgraph` | Memgraph via Bolt | `neo4j>=6.1.0` |
| `neptune` | AWS Neptune via Bolt | `neo4j>=6.1.0` |
| `neptune-iam` | Neptune with IAM SigV4 | `neo4j>=6.1.0`, `boto3` |
| `age` | PostgreSQL AGE graph backend | `asyncpg` |
| `weaviate` | Weaviate vector store | `weaviate-client>=4.20.1` |
| `sqlite` | SQLite embedded relational + vector | `aiosqlite>=0.20.0` |
| `lancedb` | LanceDB embedded vector store | `lancedb>=0.17.0`, `pyarrow` |
| `sqlite-lance` | **[experimental]** Unified SQLite + LanceDB embedded backend. Recommended embedded stack; covers VectorCypher / Skeleton / Chronicle | `lancedb>=0.17.0`, `aiosqlite>=0.20.0`, `pyarrow` |
| `binary-readers` | PDF / docx / xlsx readers (used by khora-cli and downstream ingestors) | `pymupdf`, `openpyxl`, `python-docx` |
| `parquet` | Parquet readers | `pyarrow>=18.0.0` |
| `nlp` | spaCy-based sentence splitting | `spacy>=3.8` |
| `logfire` | Logfire integration + Neo4j pool OTel metrics | `logfire>=4.0` |
| `rust` | Rust acceleration (`khora-accel`) | `khora-accel>=0.1.0` |
| `all-backends` | Everything graph-and-vector (no logfire/nlp/rust) | — |

Combine extras as needed: `pip install 'khora[surrealdb,logfire]'`.

## Core settings

| Variable | Type | Default | Description |
|---|---|---|---|
| `KHORA_DATABASE_URL` | str | — | PostgreSQL URL (shortcut for `storage.postgresql_url`). |
| `KHORA_NEO4J_URL` | str | — | Neo4j URL (shortcut for `storage.graph.url`). |
| `KHORA_LLM_EXTRACTION_MODEL` | str | — | Override extraction model (shortcut for `llm.extraction_model`). |
| `KHORA_DEBUG` | bool | `false` | Enable debug-level logging. |
| `KHORA_ENVIRONMENT` | str | `development` | `development`, `staging`, or `production`. |
| `KHORA_AUTH_ENABLED` | bool | `true` | Disable for local experimentation. |
| `KHORA_APP_NAME` | str | `khora` | Used in logs and telemetry. |

## Storage

Prefix: `KHORA_STORAGE_`. See [architecture/storage-backends.md](architecture/storage-backends.md) for the full backend matrix.

| Variable | Default | Description |
|---|---|---|
| `KHORA_STORAGE_BACKEND` | `postgres` | `postgres` (PostgreSQL + pgvector + external graph DB), `surrealdb` (unified), or `sqlite_lance` (SQLite + LanceDB embedded). |
| `KHORA_STORAGE_POSTGRESQL_URL` | — | PostgreSQL connection URL. |
| `KHORA_STORAGE_POSTGRESQL_POOL_SIZE` | `50` | asyncpg pool size. |
| `KHORA_STORAGE_POSTGRESQL_MAX_OVERFLOW` | `30` | Max overflow connections. |
| `KHORA_STORAGE_POSTGRESQL_POOL_PRE_PING` | `false` | Validate connections before checkout (adds latency, prevents stale-connection errors). |
| `KHORA_STORAGE_HNSW_M` | `24` | HNSW index `M` (max connections per layer). |
| `KHORA_STORAGE_HNSW_EF_CONSTRUCTION` | `128` | Build-time HNSW search width. |
| `KHORA_STORAGE_HNSW_EF_SEARCH` | `100` | Query-time HNSW search width. |
| `KHORA_STORAGE_USE_HALFVEC` | `true` | Use `halfvec` (float16) for HNSW indexes. Requires pgvector >= 0.7.0; falls back gracefully. |

Graph and vector backends nest under `storage.graph` and `storage.vector`. The flat fields `KHORA_STORAGE_NEO4J_URL`, `KHORA_STORAGE_NEO4J_USER`, `KHORA_STORAGE_NEO4J_PASSWORD`, `KHORA_STORAGE_PGVECTOR_URL`, and `KHORA_STORAGE_EMBEDDING_DIMENSION` remain supported as a back-compat path and are migrated into the discriminated-union configs automatically.

### Neo4j pool metrics (logfire only)

With `logfire` installed, the Neo4j backend emits OTel metrics automatically. For high-frequency sub-minute sampling enable:

```bash
KHORA_STORAGE__GRAPH__POOL_SAMPLER_ENABLED=true
KHORA_STORAGE__GRAPH__POOL_SAMPLER_INTERVAL_MS=500    # clamped to [50, 60000]
```

### Chronicle: LanceDB embedded backend

The Chronicle engine can run on either PostgreSQL + pgvector (default) or
SQLite + LanceDB. The LanceDB path is composed from the existing
`sqlite_lance` storage backend — chunk metadata and FTS5 live in SQLite,
embeddings live in a sibling LanceDB directory. Pick it via the constructor:

```python
from khora import KhoraConfig
from khora.engines.chronicle import ChronicleEngine

config = KhoraConfig()  # no postgres URL needed for the embedded path
engine = ChronicleEngine(
    config,
    storage_backend="lancedb",
    lancedb_path="./data/chronicle.db",
)
await engine.connect()  # runs Alembic migrations against the SQLite file
```

Or set it globally via the storage backend selector — Chronicle will
inherit the choice when no `storage_backend` argument is passed:

```bash
KHORA_STORAGE_BACKEND=sqlite_lance
KHORA_STORAGE_SQLITE_LANCE__DB_PATH=./data/chronicle.db
KHORA_STORAGE_SQLITE_LANCE__EMBEDDING_DIMENSION=1536
```

Install with `pip install 'khora[sqlite-lance]'` (pulls in `aiosqlite` and
`lancedb`). The pgvector path is unchanged for existing deployments —
omit `storage_backend` to get the original behavior.

## Embedded backends (experimental)

The embedded paths (`sqlite_lance` and `surrealdb`) are marked **experimental in v0.9.0**. They are appropriate for demos, evaluation, tests, and small single-user CLIs. They are not the deployment story for v0.9.0; for production, use PostgreSQL + pgvector (+ Neo4j for VectorCypher).

### SQLite + LanceDB (recommended embedded stack)

Documented scale ceiling — performance and recall degrade noticeably above these thresholds:

- **~1M chunks** (LanceDB IVF-PQ training time + write serialisation start to dominate)
- **~100k entities** (recursive-CTE traversal cost on hub nodes)
- **~500k relationships**
- **Traversal depth ≤3** (the `instr(walk.visited, ...)` visited-set scan in `graph.py` is `O(depth × fan-out × visited-len)` and degrades sharply at depth ≥4 with high fan-out)

Known gaps and warts:

- **Partial atomicity in `coordinator.transaction()`** — only the SQL session is enrolled; LanceDB writes happen post-commit with compensating-delete-on-failure. A crash between SQLite commit and Lance write can leave orphaned vectors or missing embeddings; reconciliation runs on the next ingest.
- **Point-in-time queries are not supported** on the embedded stack (DYT-3550). The CTE port does not expose the equivalent of pgvector's PIT semantics.
- **FTS5 covers chunks only** — entity-anchored recall falls back to `LIKE` / JSON-equality. Recommend the PostgreSQL stack for entity-heavy corpora.
- **Install footprint** is ~130–180 MB unpacked (pyarrow + lancedb native + Arrow C++ runtime). "Embedded" means "no server", not "no native deps".
- **IVF-PQ retraining** is automatic when the corpus grows past `retrain_factor × (rows at last training)`. Tune via `KHORA_STORAGE_SQLITE_LANCE__RETRAIN_FACTOR` (see below).

Vector index tuning fields on the `sqlite_lance` storage config:

| Variable | Default | Description |
|---|---|---|
| `KHORA_STORAGE_SQLITE_LANCE__DB_PATH` | `./khora.db` | SQLite file path. |
| `KHORA_STORAGE_SQLITE_LANCE__LANCE_PATH` | sibling `.lance` dir | LanceDB directory. |
| `KHORA_STORAGE_SQLITE_LANCE__EMBEDDING_DIMENSION` | `1536` | Vector dimension. |
| `KHORA_STORAGE_SQLITE_LANCE__USE_HALFVEC` | `false` | Store as float16. |
| `KHORA_STORAGE_SQLITE_LANCE__LANCE_INDEX` | `auto` | `auto` / `ivf_pq` / `hnsw` / `brute`. |
| `KHORA_STORAGE_SQLITE_LANCE__IVF_PARTITIONS` | `null` (auto) | IVF partition count. |
| `KHORA_STORAGE_SQLITE_LANCE__HNSW_M` | `16` | HNSW `M`. |
| `KHORA_STORAGE_SQLITE_LANCE__RETRAIN_FACTOR` | `2.0` | Rebuild the LanceDB ANN index once the row count grows to `retrain_factor × (rows at last training)`. Default `2.0` retrains when the corpus has doubled. Set ≤ `1.0` to disable retraining. Added in v0.9.0 (DYT-3579). |

### SurrealDB (experimental, unified store)

The SurrealDB backend is feature-complete (relational + vector + graph + KV in a single store) but is **experimental in v0.9.0**:

- Python SDK is pinned to `>=2.0.0a1` — alpha track for SurrealDB 3.x compatibility.
- KNN expression `<|K|>` is unreliable in embedded mode; the backend falls back to brute-force cosine + HNSW.
- Concurrent upserts require the `_SurrealDBEntityKeyGate` to serialise on `(namespace_id, name, entity_type)` keys.
- BSL-1.1 license — review for downstream packaging concerns before adopting.

Connection schemes: `memory://` (in-process), `surrealkv://...` (embedded file), `ws://...` (remote). Note: `Khora("memory://")` does **not** route to SurrealDB today — the positional argument is treated as the PostgreSQL `database_url`. Set `KHORA_STORAGE_BACKEND=surrealdb` and the relevant `KHORA_STORAGE_SURREALDB_*` settings explicitly. Routing a `memory://` URI directly to the embedded stack is tracked for v0.10.

## LLM

Prefix: `KHORA_LLM_`. LiteLLM handles the provider dispatch.

| Variable | Default | Description |
|---|---|---|
| `KHORA_LLM_MODEL` | `gpt-4o-mini` | Primary model for generation. |
| `KHORA_LLM_API_KEY_ENV` | `OPENAI_API_KEY` | Environment variable holding the API key. |
| `KHORA_LLM_TEMPERATURE` | `0.7` | Sampling temperature. |
| `KHORA_LLM_MAX_TOKENS` | `12288` | Max output tokens per extraction call. |
| `KHORA_LLM_TIMEOUT` | `30` | Request timeout in seconds. |
| `KHORA_LLM_MAX_RETRIES` | `3` | Retry budget on failure. |
| `KHORA_LLM_MAX_CONCURRENT_LLM_CALLS` | `10` | Cap on concurrent in-flight LLM requests. |
| `KHORA_LLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `KHORA_LLM_EMBEDDING_DIMENSION` | `1536` | Must match your DB schema. |
| `KHORA_LLM_EXTRACTION_MODEL` | — | Override extraction model (falls back to `model`). Haiku / Gemini Flash work well here. |

## Pipeline (extraction)

Prefix: `KHORA_PIPELINES_`.

| Variable | Default | Description |
|---|---|---|
| `KHORA_PIPELINES_CHUNKING_STRATEGY` | `semantic` | `fixed`, `semantic`, or `recursive`. |
| `KHORA_PIPELINES_CHUNK_SIZE` | `512` | Target chunk size (tokens). |
| `KHORA_PIPELINES_CHUNK_OVERLAP` | `50` | Overlap between chunks. |
| `KHORA_PIPELINES_CONVERSATION_TIME_GAP_MINUTES` | `15` | Split conversations after this many quiet minutes. |
| `KHORA_PIPELINES_CONVERSATION_MAX_GROUP_SIZE` | `50` | Max messages per conversation chunk. |
| `KHORA_PIPELINES_CONVERSATION_MIN_GROUP_SIZE` | `2` | Merge groups below this size. |
| `KHORA_PIPELINES_EXTRACT_ENTITIES` | `true` | Run the entity extractor. |
| `KHORA_PIPELINES_ENTITY_TYPES` | `PERSON,ORGANIZATION,CONCEPT,LOCATION` | Entity type allowlist. |
| `KHORA_PIPELINES_SELECTIVE_EXTRACTION` | `true` | KET-RAG selective extraction (cost reduction). |
| `KHORA_PIPELINES_EXTRACTION_IMPORTANCE_RATIO` | `0.7` | Top fraction of chunks sent to LLM extraction. |
| `KHORA_PIPELINES_EXTRACTION_MIN_IMPORTANCE` | `0.2` | Minimum importance threshold; chunks above this are always extracted. |
| `KHORA_PIPELINES_SKIP_EMBEDDING_ENTITY_TYPES` | `DATE,URL,EMAIL` | Skip embeddings for these types when `mention_count` is low. |
| `KHORA_PIPELINES_SKIP_EMBEDDING_MENTION_THRESHOLD` | `1` | Skip embedding for rare-mention entities of the above types. |

## Query

Prefix: `KHORA_QUERY_`. See [query-engine/retrieval-tuning.md](query-engine/retrieval-tuning.md) for guidance.

| Variable | Default | Description |
|---|---|---|
| `KHORA_QUERY_DEFAULT_MODE` | `hybrid` | `vector`, `graph`, `hybrid`, or `all`. |
| `KHORA_QUERY_MIN_CHUNK_SIMILARITY` | `0.05` | Chunk similarity floor. |
| `KHORA_QUERY_MIN_ENTITY_SIMILARITY` | `0.05` | Entity similarity floor. |
| `KHORA_QUERY_VECTOR_WEIGHT` | `0.5` | Fusion weight. |
| `KHORA_QUERY_GRAPH_WEIGHT` | `0.3` | Fusion weight. |
| `KHORA_QUERY_KEYWORD_WEIGHT` | `0.2` | Fusion weight. |
| `KHORA_QUERY_APPLY_RECENCY_BIAS` | `false` | Bias scoring towards newer documents. |
| `KHORA_QUERY_RECENCY_WEIGHT` | `0.2` | How strong the recency bias is. |
| `KHORA_QUERY_ENABLE_HYDE` | `auto` | HyDE query expansion: `auto`, `always`, `never`. |
| `KHORA_QUERY_ENABLE_RERANKING` | `true` | Cross-encoder reranking of top candidates. |
| `KHORA_QUERY_TEMPORAL_SQL_PUSHDOWN` | `true` | Push relative-date filters into SQL WHERE clauses. |

## Tenancy

Prefix: `KHORA_TENANCY_`.

| Variable | Default | Description |
|---|---|---|
| `KHORA_TENANCY_DEFAULT_MODE` | `shared` | `shared` or `isolated`. |
| `KHORA_TENANCY_ENFORCE_NAMESPACE` | `true` | Fail closed if a call omits a namespace. |

## Telemetry

| Variable | Default | Description |
|---|---|---|
| `KHORA_TELEMETRY_DATABASE_URL` | — | PostgreSQL URL for the telemetry collector. If unset, the no-op collector is used (zero cost). |
| `KHORA_TELEMETRY_SERVICE_NAME` | `khora` | Service tag attached to events. |

The `@trace` decorator in `khora.telemetry` is a no-op when `logfire` is not installed.

## Logging

Khora uses loguru. Call `khora.logging_config.setup_logging()` once per process (or configure your own sinks with `enqueue=True`). See the Logging section of [CLAUDE.md](../CLAUDE.md) for the full rationale — short version: default loguru sinks are synchronous and will block an asyncio event loop on every `logger.*` call.

| Variable | Default | Description |
|---|---|---|
| `KHORA_NEO4J_LOG_LEVEL` | — | Neo4j driver log level (`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`, case-insensitive). Unset = no-op. See `examples/neo4j_debug_logging.py`. |

## Secrets

API keys (OpenAI, Anthropic, etc.) are read from the environment variable named by `KHORA_LLM_API_KEY_ENV` (default `OPENAI_API_KEY`). Khora never reads credentials from disk. Rotate at the environment level — no restart is required beyond whatever your process manager provides.

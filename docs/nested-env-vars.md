# Nested env vars

Reference for every Khora environment variable that lives on a **sub-object**
attached to a sub-settings class — graph backend, vector backend, the unified
SurrealDB store, the SQLite+LanceDB embedded stack, and the dream-phase
per-op toggles.

> **Spelling.** All env vars in this document use **single underscore** between
> every level — `KHORA_STORAGE_GRAPH_URL`, not `KHORA_STORAGE__GRAPH__URL`.
> The legacy double-underscore form (`KHORA_STORAGE__GRAPH__URL`) continues to
> work as a backwards-compatible alias on every nested-config field; it is no
> longer documented. New code and `.env` files should use the single-underscore
> form documented here.

## Sections

- [`KHORA_STORAGE_GRAPH_*` — graph backend](#khora_storage_graph_-graph-backend)
- [`KHORA_STORAGE_VECTOR_*` — vector backend](#khora_storage_vector_-vector-backend)
- [`KHORA_STORAGE_SURREALDB_*` — unified SurrealDB](#khora_storage_surrealdb_-unified-surrealdb)
- [`KHORA_STORAGE_SQLITE_LANCE_*` — SQLite + LanceDB unified](#khora_storage_sqlite_lance_-sqlite--lancedb-unified)
- [`KHORA_DREAM_OPS_*` — dream-phase per-op toggles](#khora_dream_ops_-dream-phase-per-op-toggles)

---

## `KHORA_STORAGE_GRAPH_*` — graph backend

Discriminated union over `Neo4jConfig | MemgraphConfig | NeptuneConfig | SurrealDBConfig | AGEConfig`, keyed by the `backend` field. The set of available fields depends on which backend you select.

### Always present (any graph backend)

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_GRAPH_BACKEND` | `neo4j` | Pick the graph adapter: `neo4j` / `memgraph` / `neptune` / `surrealdb` / `age`. |
| `KHORA_STORAGE_GRAPH_URL` | — | Bolt / connection URL. `SecretStr`. |

### Neo4j (`backend=neo4j`, the default)

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_GRAPH_USER` | `neo4j` | Username. |
| `KHORA_STORAGE_GRAPH_PASSWORD` | empty | Password. `SecretStr`. |
| `KHORA_STORAGE_GRAPH_DATABASE` | `neo4j` | Multi-database selector inside a Neo4j cluster. |
| `KHORA_STORAGE_GRAPH_MAX_CONNECTION_POOL_SIZE` | `100` | Lower on small drivers; raise for high concurrency. |
| `KHORA_STORAGE_GRAPH_CONNECTION_ACQUISITION_TIMEOUT` | `60.0` s | Lower for fast-fail under pool starvation. |
| `KHORA_STORAGE_GRAPH_RETRY_DELAY_JITTER_FACTOR` | `0.5` | Jitter (0.0–1.0) on transaction-retry backoff. Raise to spread retry storms. |
| `KHORA_STORAGE_GRAPH_MAX_CONNECTION_LIFETIME` | `900` s | Rotate connections before this. Set below your server-side TTL (Aura ~20 min) to avoid `BrokenPipe`. |
| `KHORA_STORAGE_GRAPH_LIVENESS_CHECK_TIMEOUT` | `30.0` s | Idle threshold before pre-checkout liveness check. `None` disables. |
| `KHORA_STORAGE_GRAPH_QUERY_TIMEOUT` | `5.0` s | Per-transaction read timeout (1–300 s, `None` disables). Raise for deep traversals; lower to fail fast. |
| `KHORA_STORAGE_GRAPH_ENTITY_WRITE_CONCURRENCY` | `12` | Concurrent entity-write transactions during ingest. Raise when Neo4j has headroom; lower on lock contention. |
| `KHORA_STORAGE_GRAPH_RELATIONSHIP_WRITE_CONCURRENCY` | `8` | Concurrent relationship-write transactions. |
| `KHORA_STORAGE_GRAPH_POOL_SAMPLER_ENABLED` | `false` | Opt-in high-frequency pool sampler. Requires an OTel backend installed. Enable to investigate pool exhaustion; zero-cost when off. |
| `KHORA_STORAGE_GRAPH_POOL_SAMPLER_INTERVAL_MS` | `500` | Sample cadence in ms (clamped 50–60000). Drop to 50–100 when chasing sub-second pool events. |
| `KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_DOCUMENT_IDS_MAX` | `100` | Cap on `Relationship.source_document_ids` retained after `MERGE`. When the cap is exceeded, the most-recent tail is kept and the dropped count is recorded on `khora.neo4j.relationship.source_id_truncated{field=source_document_ids}`. Raise for deep-provenance workloads. |
| `KHORA_STORAGE_GRAPH_RELATIONSHIP_SOURCE_CHUNK_IDS_MAX` | `250` | Same as above, for `source_chunk_ids`. |
| `KHORA_STORAGE_GRAPH_ENTITY_SOURCE_DOCUMENT_IDS_MAX` | `100` | Cap on `Entity.source_document_ids` retained after `MERGE`. Tail-keep semantics identical to the relationship cap; metric is `khora.neo4j.entity.source_id_truncated{field=source_document_ids}`. |
| `KHORA_STORAGE_GRAPH_ENTITY_SOURCE_CHUNK_IDS_MAX` | `250` | Same as above, for entity `source_chunk_ids`. |

### Memgraph (`backend=memgraph`)

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_GRAPH_BACKEND` | — | Set to `memgraph`. |
| `KHORA_STORAGE_GRAPH_URL` | — | Bolt URL. `SecretStr`. |
| `KHORA_STORAGE_GRAPH_USER` | `memgraph` | Username. |
| `KHORA_STORAGE_GRAPH_PASSWORD` | empty | Password. `SecretStr`. |

Memgraph speaks Bolt via the Neo4j driver, but its config model exposes only the four fields above — none of the pool / timeout / provenance-cap knobs from the Neo4j section apply.

### Neptune (`backend=neptune`)

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_GRAPH_USER` | empty | Basic-auth username. Empty when using IAM. |
| `KHORA_STORAGE_GRAPH_PASSWORD` | empty | Basic-auth password. Empty when using IAM. |
| `KHORA_STORAGE_GRAPH_IAM_AUTH` | `false` | Set `true` for AWS SigV4 instead of basic auth. Requires `khora[neptune-iam]`. |
| `KHORA_STORAGE_GRAPH_AWS_REGION` | `us-east-1` | Region for SigV4 signing. Must match the Neptune cluster's region. |
| `KHORA_STORAGE_GRAPH_MAX_CONNECTION_POOL_SIZE` | `100` | Bolt pool size (Neptune cluster max is 1000). |

### SurrealDB as graph (`backend=surrealdb`)

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_GRAPH_MODE` | `memory` | `memory` / `embedded` / `remote`. Pick `remote` for multi-process deployments. |
| `KHORA_STORAGE_GRAPH_PATH` | — | On-disk SurrealKV file (embedded mode only). |
| `KHORA_STORAGE_GRAPH_NAMESPACE` | `khora` | SurrealDB namespace. Vary to multiplex tenants in one instance. |
| `KHORA_STORAGE_GRAPH_DATABASE` | `default` | SurrealDB database. |
| `KHORA_STORAGE_GRAPH_USER` | `root` | Username. |
| `KHORA_STORAGE_GRAPH_PASSWORD` | `root` | Password. `SecretStr`. |
| `KHORA_STORAGE_GRAPH_EMBEDDING_DIMENSION` | `1536` | Must match LLM embedding dimension. |
| `KHORA_STORAGE_GRAPH_SYNC_DATA` | `true` | `SURREAL_SYNC_DATA` for crash-safe writes. Disable only for ephemeral test data. |

### AGE (`backend=age`)

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_GRAPH_GRAPH_NAME` | `khora_graph` | Name of the AGE graph object inside Postgres. The doubled `GRAPH` is not a typo — `GRAPH` comes from the sub-object name (`storage.graph`) and `GRAPH_NAME` from the field name (`AGEConfig.graph_name`). |
| `KHORA_STORAGE_GRAPH_POOL_SIZE` | `10` | asyncpg pool size dedicated to the graph layer. |
| `KHORA_STORAGE_GRAPH_MAX_OVERFLOW` | `20` | Max overflow connections beyond `pool_size`. |

---

## `KHORA_STORAGE_VECTOR_*` — vector backend

Discriminated union over `PgVectorConfig | SurrealDBVectorConfig | SQLiteVectorConfig`, keyed by `backend`. Default is `pgvector` via `default_factory=PgVectorConfig`.

| Variable | Default | Applies when | Why change it |
|---|---|---|---|
| `KHORA_STORAGE_VECTOR_BACKEND` | `pgvector` | always | `pgvector` / `surrealdb` / `sqlite`. |
| `KHORA_STORAGE_VECTOR_URL` | — | pgvector / sqlite | Connection URL. |
| `KHORA_STORAGE_VECTOR_EMBEDDING_DIMENSION` | `1536` | always | Must match the LLM embedding model. Changing requires a schema migration. |
| `KHORA_STORAGE_VECTOR_MODE` | `memory` | SurrealDB vector | `memory` / `embedded` / `remote`. |
| `KHORA_STORAGE_VECTOR_PATH` | — | SurrealDB vector (embedded) | On-disk SurrealKV file. |
| `KHORA_STORAGE_VECTOR_NAMESPACE` | `khora` | SurrealDB vector | SurrealDB namespace. |
| `KHORA_STORAGE_VECTOR_DATABASE` | `default` | SurrealDB vector | SurrealDB database. |
| `KHORA_STORAGE_VECTOR_USER` | `root` | SurrealDB vector | Username. |
| `KHORA_STORAGE_VECTOR_PASSWORD` | `root` | SurrealDB vector | Password. `SecretStr`. |

For `backend=sqlite`, only `BACKEND` / `URL` / `EMBEDDING_DIMENSION` are model-exposed.

---

## `KHORA_STORAGE_SURREALDB_*` — unified SurrealDB

Used when `KHORA_STORAGE_BACKEND=surrealdb` routes **both** graph and vector
roles to the same SurrealDB instance. Same fields as the graph-role config
above, attached at a different point on `StorageSettings`.

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_SURREALDB_MODE` | `memory` | `memory` / `embedded` / `remote`. Pick `remote` for production. |
| `KHORA_STORAGE_SURREALDB_URL` | — | `ws://...` WebSocket URL (remote mode). |
| `KHORA_STORAGE_SURREALDB_PATH` | — | On-disk SurrealKV file (embedded mode). |
| `KHORA_STORAGE_SURREALDB_NAMESPACE` | `khora` | SurrealDB namespace. |
| `KHORA_STORAGE_SURREALDB_DATABASE` | `default` | SurrealDB database. |
| `KHORA_STORAGE_SURREALDB_USER` | `root` | Username. |
| `KHORA_STORAGE_SURREALDB_PASSWORD` | `root` | Password. `SecretStr`. |
| `KHORA_STORAGE_SURREALDB_EMBEDDING_DIMENSION` | `1536` | Must match LLM embedding dimension. |
| `KHORA_STORAGE_SURREALDB_SYNC_DATA` | `true` | Crash-safe writes. Disable only for throwaway test data. |

---

## `KHORA_STORAGE_SQLITE_LANCE_*` — SQLite + LanceDB unified

Used when `KHORA_STORAGE_BACKEND=sqlite_lance`. Pairs an on-disk SQLite
database (graph + relational + event store) with a sibling LanceDB
directory (vector search). Zero infrastructure — both backends run
in-process.

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_STORAGE_SQLITE_LANCE_DB_PATH` | `./khora.db` | SQLite file path. Move to faster storage or a backup-friendly location. |
| `KHORA_STORAGE_SQLITE_LANCE_LANCE_PATH` | sibling `.lance` dir | Explicit LanceDB directory. Override to put vectors on different storage (e.g. SSD) than chunk metadata. |
| `KHORA_STORAGE_SQLITE_LANCE_EMBEDDING_DIMENSION` | `1536` | Must match LLM embedding model. |
| `KHORA_STORAGE_SQLITE_LANCE_USE_HALFVEC` | `false` | Float16 storage — halves index size with minor recall loss. Enable on memory-constrained boxes. |
| `KHORA_STORAGE_SQLITE_LANCE_LANCE_INDEX` | `auto` | `auto` / `ivf_pq` / `hnsw` / `brute`. Force `ivf_pq` above ~1M rows; `hnsw` for low-latency under ~1M; `brute` for tiny corpora. |
| `KHORA_STORAGE_SQLITE_LANCE_IVF_PARTITIONS` | `null` (auto) | Hand-tuned IVF partition count (`lance_index=ivf_pq` only). Override only if profiling shows recall miss. |
| `KHORA_STORAGE_SQLITE_LANCE_HNSW_M` | `16` | HNSW max connections per layer (`lance_index=hnsw` only). Raise for recall; linear memory cost. |
| `KHORA_STORAGE_SQLITE_LANCE_RETRAIN_FACTOR` | `2.0` | Trigger LanceDB ANN retrain when row count grows by this factor. Lower for fresher index; raise to defer re-training cost. `<= 1.0` disables. |

---

## `KHORA_DREAM_OPS_*` — dream-phase per-op toggles

`DreamConfig.ops: DreamOpsConfig` carries per-operation enable flags. Every
destructive op defaults to `false` — `KHORA_DREAM_ENABLED=true` alone runs
no destructive work; each op must be flipped explicitly.

See [dream-phase.md](dream-phase.md) for operational guidance, retention
floors, and the kill-switch (`KHORA_DREAM_DISABLE_APPLY`).

| Variable | Default | Why change it |
|---|---|---|
| `KHORA_DREAM_OPS_DEDUPE_ENTITIES` | `false` | Enable cross-batch entity dedupe (cosine-merge with verifier). Turn on after validating planner output in dry-run. |
| `KHORA_DREAM_OPS_PRUNE_EDGES` | `false` | Remove low-confidence / orphaned edges (default targets `ASSOCIATED_WITH` co-occurrence). Turn on when edge soup degrades retrieval. |
| `KHORA_DREAM_OPS_COMPACT_FACTS` | `false` | Hard-delete tombstoned `memory_facts` rows past the 7-day retention floor. **The only hard-delete op** — flip with care. |
| `KHORA_DREAM_OPS_CLUSTER_EVENTS` | `false` | Merge near-duplicate Chronicle events (cosine ≥ 0.95 within a 7-day window). |
| `KHORA_DREAM_OPS_RECOMPUTE_CENTROIDS` | `false` | Recompute entity / cluster centroid embeddings after dedupe. Pair with `DEDUPE_ENTITIES`. |

---

## Not on this list

These env vars reach top-level fields of each sub-settings class via the
sub-class's own `env_prefix` — there is no sub-object hop. See
[configuration.md](configuration.md) for full coverage.

- `KHORA_STORAGE_BACKEND`, `KHORA_STORAGE_POSTGRESQL_*`, `KHORA_STORAGE_HNSW_*`, `KHORA_STORAGE_USE_HALFVEC` — flat on `StorageSettings`.
- `KHORA_LLM_*` — every field on `LLMSettings` is top-level (no sub-objects).
- `KHORA_PIPELINES_*` — every field on `PipelineSettings` is top-level.
- `KHORA_QUERY_*` — every field on `QuerySettings` is top-level, including the Chronicle-specific `KHORA_QUERY_CHRONICLE_*` family.
- `KHORA_TENANCY_*`, `KHORA_TELEMETRY_*` — both flat.
- `KHORA_DREAM_*` (e.g. `KHORA_DREAM_ENABLED`, `KHORA_DREAM_DEFAULT_MODE`, `KHORA_DREAM_LLM_MAX_TOKENS_PER_RUN`) — flat on `DreamConfig`. Only the per-op toggles under the `ops:` sub-object are listed above.

## Related

- [configuration.md](configuration.md) — the full env-var surface (flat + nested) with section-by-section coverage.
- [architecture/multi-tenancy.md](architecture/multi-tenancy.md) — namespace isolation is enforced at the storage Protocol layer; `TenancyMode.ISOLATED` is reserved but not yet wired.
- [architecture/storage-backends.md](architecture/storage-backends.md) — per-backend deployment recipes.
- [dream-phase.md](dream-phase.md) — dream-phase operational guidance.

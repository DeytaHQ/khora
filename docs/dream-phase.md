# Dream phase

`khora.dream` is an **offline maintenance pass** for accumulated agentic memory. It runs against a single namespace on a schedule — cron, Temporal, k8s CronJob — auditing the graph and (in v0.15+) planning consolidation work. Audit ops are read-only and ship today. Planner ops emit a dry-run plan today and raise `NotImplementedError` if you try to apply it; apply mode lands in v0.16.

The naming follows the complementary learning systems framework from neuroscience: ingestion is the fast, episodic path (`Khora.remember`), dream phase is the slow, reorganizing path. Same store, different access regime, different objective function. See [Research & Prior Art](#research--prior-art) for the lineage.

> **v0.15.0 status — read this once before scheduling anything in prod.**
> Five Phase-1 audit ops are live and safe. Five Phase-2 planner ops are live in dry-run mode; their `apply` paths raise `NotImplementedError`. Nothing in v0.15 mutates your graph. The kill-criterion gate is whether operators actually call `kb.dream()` in production — `khora.dream.runs_total` is the telemetry to watch.

## When to use it

- Recall quality has drifted and you don't know which threshold to tune.
- You suspect duplicate entities from independent ingest batches (`"OpenAI"` and `"Open AI"`, `"Marie Curie"` from two different documents).
- `memory_facts` is growing and you can't tell how much of it is tombstoned.
- Recall latency is creeping up because `chunks.source_chunk_ids` arrays carry references to long-gone chunks.
- An `ExpertiseConfig` change landed weeks ago and you don't know whether the data has caught up.
- Operators want a periodic, human-readable "state of the graph" snapshot they can review before authorizing destructive ops.

If you want to **fix** any of these today, you can't — apply mode raises `NotImplementedError`. Dream phase v0.15 tells you the truth about your graph and produces a plan. The plan-then-apply split is deliberate (see [Research & Prior Art](#offline-rl-replay-buffers)).

## Quickstart

The master switch is `KhoraConfig.dream.enabled` (env var `KHORA_DREAM_ENABLED`). Default is `False` — dream phase is opt-in. Per-op flags are also off by default; turn on what you need.

```python
from khora import Khora, KhoraConfig, DreamConfig

kb = Khora(
    config=KhoraConfig(
        dream=DreamConfig(
            enabled=True,
            report_file_sink_enabled=True,
            report_event_sink_enabled=False,
            report_collector_sink_enabled=True,
        ),
    ),
)

result = await kb.dream(namespace_id, mode="dry-run")

# `result.ops` is the aggregate counter — one OpSummary per op kind.
for summary in result.ops:
    print(
        f"{summary.op_type}: planned={summary.planned} "
        f"skipped={summary.skipped} failed={summary.failed}"
    )

# Per-op detail (decision strings, outputs dicts, rationale) lives in
# the file sink's events.jsonl, or in `result.metadata["plan_payload"]`
# in dry-run mode.
```

Configuration knobs and env-var bindings live in [configuration.md](configuration.md); all settings reachable via `KHORA_DREAM_*`.

**Default file-sink location.** With `report_file_sink_enabled=True`, the default base directory is `<system temp dir>/khora-dream-reports`. On Linux that's `/tmp/...`, which is wiped on reboot — **set a persistent base directory** if you want audit history to survive restarts. The overridable hook lands in v0.15.x (#697 follow-up); until then operators on persistent workloads should run a sweep that copies reports out of the temp dir, or call `DreamFileSink` directly with `base_dir=` in a custom sink list.

## Operations

Every op returns a `DreamOp` with a `decision` string and a structured `outputs` dict. The orchestrator routes those through whichever sinks are enabled. The op never mutates state directly — even Phase 2 planner ops only describe what they would do.

### Audit operations (Phase 1)

Pure SELECT / pure observation. Zero LLM calls, zero mutations, zero risk to production graphs. `apply` mode is a pass-through (no destructive side effect to apply).

#### Chronicle: abstention-threshold drift

Reads the OpenTelemetry histogram of `top_score` / `combined_score` values that chronicle's `_compute_abstention_signals` emits on every recall (plus a bounded in-process ring buffer for the no-logfire path). Compares observed p50/p90/p99 against configured `abstention_min_top_score`, `abstention_min_chunks`, and `abstention_combined_threshold`.

`decision` values:
- `"recommend"` with `direction="lower"|"raise"|"calibrated"` and a rationale referencing the gap
- `"insufficient_data"` when fewer than `abstention_drift_min_samples` (default 1000) recalls have been observed

The op never auto-tunes thresholds. Threshold changes are operator policy.

#### Chronicle: `memory_facts` tombstone audit

Pure SELECT. Counts `active` (legacy `is_active=True`), `inactive` (legacy `is_active=False`), and `invalidated` (bi-temporal `invalidated_at IS NOT NULL`). Plus `tombstone_ratio`, oldest-tombstone age, p50/p90 ages, and top-K offenders by age. Recommends a `retention_days` threshold (default 365) for the eventual compaction op to consume.

#### VectorCypher: schema drift vs `ExpertiseConfig`

Multiset-diff between observed `entity_type` / `relationship_type` strings and what `ExpertiseConfig` declares.

| Bucket | Meaning |
|---|---|
| `new_entity_types` | Present in data, not declared in `ExpertiseConfig` |
| `unused_entity_types` | Declared in config, not used in data |
| `entity_frequency_delta` | Frequency changed by ≥50% since the previous dream run |
| `*_relationship_types` | Same three buckets for relationship types |

Never normalizes type names. `ExpertiseConfig` is declarative user intent; rewriting types in the data is a separate policy decision.

#### VectorCypher: PageRank-based orphan report

Builds the entity-relationship graph for the namespace, down-weights `ASSOCIATED_WITH` co-occurrence edges to `0.2` (so they don't dominate), runs the `_accel.pagerank` Rust kernel, then flags entities matching all of:
- PR score in the bottom `orphan_pr_percentile_threshold` percentile (default 5)
- `mention_count ≤ 1`
- No recent recall hits

Output is a list of `{entity_id, name, type, pr_score, mention_count}` with `archive_candidate=true`. The op never archives.

#### VectorCypher: `source_chunk_ids` array-length audit

Joins entities × chunks (Postgres `unnest`; SQLite Python-side) and reports total dead UUID references, array-length distribution (p50/p90/p99/max), and top-K offenders by array length. Surfaces the GC candidates without modifying any row. Feeds the Phase 2 GC planner.

### Planner operations (Phase 2)

Plan-only in v0.15. Each op emits one `DreamOp` per work item with `decision` describing what it would do; `apply` mode raises `NotImplementedError` (tracked under #669). The plan is checkpointed to `khora_dream_runs` so the apply phase, when it lands, can resume from the same plan.

#### VectorCypher: cross-batch entity dedupe

Cross-batch entity resolution against the full namespace. Buckets entities by `(name_lower, entity_type)`, computes pairwise cosine on pre-normalized embeddings using `batch_dot_product`, and emits a planned merge for any pair above the per-type threshold (default cosine ≥ 0.90, override per type via `dedupe_entities_per_type_thresholds`).

- Tighter than the online resolver's 0.85 default — this is the cross-batch pass with the benefit of all accumulated evidence.
- Skip-collisions (the same canonical entity would absorb two clusters) are reported under `outputs.skip_collision_count`; the op doesn't choose for you.
- See: Köpcke & Rahm 2010 on entity-matching frameworks.

#### VectorCypher: centroid recompute

For each cluster proposed by the dedupe op (or any external source), pick how the post-merge canonical embedding should be produced. Three outcomes:

| `decision` | When |
|---|---|
| `"centroid"` | All pairwise names within `centroid_lev_threshold` Levenshtein distance — variants of the same surface form (`"OpenAI"` / `"Open AI"`). Plan a weighted-mean fusion of the cluster's embeddings, L2-renormalize. |
| `"re_embed"` | Names lexically distant but semantically aligned (`"IBM"` / `"International Business Machines"`). Plan a re-embed of the canonical name. |
| `"skip_multimodal"` | Intra-cluster pairwise cosine drops below `centroid_min_intra_cluster_cosine`. The cluster spans more than one concept — the merge itself is the bug. Emit a finding, plan nothing. |
| `"skip_singleton"` | Fewer than 2 members after loading. |

Note: this op needs `rapidfuzz` at runtime (via the `[accel]` extra). The import is deferred so the module loads without it; the op fails with `ModuleNotFoundError` only at the point it actually runs.

#### VectorCypher: `source_chunk_ids` GC

Plans per-entity rewrites that drop dead chunk UUIDs from `Entity.source_chunk_ids`. Postgres path uses `unnest WITH ORDINALITY` for an in-DB join; SQLite path parses the JSON array Python-side. Threshold via `source_chunk_ids_gc_min_dead` (default 1 — every entity with ≥1 dead reference is planned).

#### Chronicle: `memory_facts` compaction

Plans hard deletes of tombstoned `memory_facts` rows older than `fact_compaction_retention_days` (default 365). The only Phase 2 op that hard-deletes — because the tombstone is itself the soft-delete marker; compaction is the second of the two phases. Output rows describe per-fact what would be deleted; apply mode raises `NotImplementedError`. See [Database compaction and tombstone GC](#database-compaction-and-tombstone-gc) for the prior art.

#### Chronicle: event clustering

Clusters near-duplicate `chronicle_events` within a `(namespace_id, subject)` bucket and a sliding `referenced_date` window. SVO-summary cosine ≥ `event_clustering_cosine_threshold` (default 0.95) within `event_clustering_window_days` (default 7) defines a cluster. Each cluster gets a canonical representative; the rest are tagged for soft-tombstoning when apply lands.

**Architectural promise:** `chronicle_events.chunk_id` is never proposed for mutation. That FK powers the temporal recall channel back-pointer; touching it breaks recall. The planner code is structured so its output payloads can't carry the bare key `chunk_id` (the field name is namespaced as `merged_source_chunk_ids_count` etc.), but this is **not a runtime assertion** — a future contributor who adds a literal `"chunk_id"` key won't trip a test. The test suite covers the negative-existence check (`test_invariant_no_chunk_id_mutation`); keep it green.

## Output channels (sinks)

Three sinks consuming the same `DreamOp` stream. Enable independently via `DreamConfig.report_*_sink_enabled`.

### File sink

Writes per-run artifacts under `{base_dir}/{namespace_id}/{date}/{run_id}.*`:

| File | Contents |
|---|---|
| `summary.md` | Human-readable executive summary + sampled high-impact ops |
| `events.jsonl` | One `DreamOp` per line, machine-readable, schema-versioned (`dream-report/1`) |
| `manifest.json` | Run metadata + checksum |
| `undo.json` | Empty in v0.15 (audit + planner ops have no undo state); populated when apply lands |

Schema version is asserted on read. `redact_text` (`"none" | "summary" | "all"`, default `"summary"`) governs raw-text exposure across all three sinks. Retention via `DreamConfig.retention_days` (default 30) and `retention_runs_per_namespace` (default 50) — rotation is a sweep, not real-time.

### Event sink

Bridges into the existing `HookDispatcher` via six `EventType.DREAM_*` values: `DREAM_RUN_STARTED`, `DREAM_PHASE_STARTED`, `DREAM_OP_DECIDED`, `DREAM_PHASE_COMPLETED`, `DREAM_RUN_COMPLETED`, `DREAM_RUN_FAILED`. Existing `SemanticFilter` filters work — including low-cost level-0 fields `dream_op_types` and `dream_decisions`. Callbacks subscribing to `DREAM_OP_DECIDED` receive a payload shape identical to one line of `events.jsonl`.

### Collector sink (OpenTelemetry)

Emits spans and metrics declared in [`telemetry-contract.json`](telemetry-contract.json). The drift gate at `tests/unit/telemetry/test_contract.py` enforces that any new span / metric introduced by the orchestrator or an op is registered.

**Public top-level spans** (operator-facing, stable):
- `khora.dream.run`, `khora.dream.phase`, `khora.dream.llm_call` (reserved for v0.16+), `khora.dream.undo` (reserved for v0.16+)

**Internal per-op spans** (names may evolve; don't pin dashboards):
- `khora.dream.chronicle.{abstention_drift,tombstone_audit,fact_compaction,event_clustering}`
- `khora.dream.vectorcypher.{schema_drift,orphan_report,source_chunk_ids_audit,source_chunk_ids_gc,centroid_recompute,dedupe_entities}`

**Public metrics** — aggregate-only, never labelled by `namespace_id` (cardinality rule):
- `khora.dream.runs_total {trigger, outcome}`
- `khora.dream.run.duration {trigger, outcome}` (histogram, seconds)
- `khora.dream.phase.duration {phase, outcome}` (histogram)
- `khora.dream.ops_total {phase, op_type, decision}`
- `khora.dream.llm.tokens {direction, model}` (reserved for v0.16+)
- `khora.dream.undo_invocations_total {op_type, outcome}` (reserved for v0.16+)
- `khora.dream.report.write_failures_total {reason}` (internal)

All free-text attributes (rationale strings, entity names) go through `khora.telemetry.bounded_text_hash` before becoming span attributes. Raw text is never exposed as a label — privacy and cardinality both.

## API reference

Every public symbol exported from `khora.dream.*` plus the top-level entry point. The dataclasses returned from `Khora.dream()` are **public** (subject to coordinated release with `khora-cli` / `khora-explorer`); the planner functions and orchestrator internals are **internal** (names may evolve through v0.16).

### Top-level entry points

Bound methods on `khora.Khora`:

```python
async def dream(
    namespace: str | UUID,
    *,
    mode: str = "dry-run",                # "dry-run" | "apply"
    scope: DreamScope | None = None,
    ops: Iterable[OpKind] | None = None,
    config: DreamConfig | None = None,
    on_progress: Callable[[DreamProgress], None] | None = None,
    resume_from: UUID | None = None,
) -> DreamResult

async def dream_status(run_id: UUID) -> dict[str, object]
async def dream_history(namespace: str | UUID, *, limit: int = 20) -> list[DreamRunInfo]
```

Functional equivalents live at `khora.dream.api.{dream, dream_status, dream_history, dream_cancel}`. `dream_cancel(run_id)` flips an in-process cancel flag, checked between ops. `Khora.dream()` raises `DreamDisabledError` when `DreamConfig.enabled` is False and `ValueError` for bad `mode` / non-UUID `namespace`.

**Embedded-backend caveat.** `dream_status(run_id)` returns `{}` and `dream_history(namespace, ...)` returns `[]` on every non-Postgres backend (sqlite_lance, SurrealDB embedded) — the `khora_dream_runs` checkpoint table is Postgres-only. On embedded, read the file-sink artifacts directly (`<base_dir>/<namespace_id>/<date>/<run_id>.{summary.md,events.jsonl,manifest.json}`) instead of querying the API.

**`mode="apply"` today.** Through `Khora.dream()`, apply is a pass-through for both Phase 1 audit ops (correct — audits have no destructive side effect) and Phase 2 planner ops (their actual mutation logic lands in v0.16 via #669). The per-op `NotImplementedError` guards in `plan_chronicle_fact_compaction(apply=True)` etc. fire only when the planner functions are called *directly*, not via `Khora.dream(mode="apply")`. Treat apply mode as "validated pass" until v0.16.

### Configuration

`khora.dream.config.DreamConfig` is a `pydantic_settings.BaseSettings` (env prefix `KHORA_DREAM_`, nested delimiter `__`). The full knob table:

| Field | Default | Notes |
|---|---|---|
| `enabled` | `False` | Master switch — `Khora.dream()` raises `DreamDisabledError` when False |
| `default_mode` | `"dry-run"` | Default when caller omits `mode=` |
| `ops.{dedupe_entities,prune_edges,compact_facts,cluster_events,recompute_centroids}` | `False` | Per-op enable flags; destructive ops default off |
| `llm_max_tokens_per_run` / `_per_namespace_per_day` | `200_000` / `1_000_000` | Run-scoped and rolling-day token budgets |
| `retention_days` / `retention_runs_per_namespace` | `30` / `50` | Report retention |
| `report_{file,event,collector}_sink_enabled` | `False` | Sink toggles |
| `redact_text` | `"summary"` | `"none" \| "summary" \| "all"` |
| `abstention_drift_min_samples` / `_sample_cap` | `1000` / `1024` | Floor before recommending; ring-buffer cap per namespace |
| `fact_compaction_retention_days` | `365` | Age threshold for tombstone hard-delete (apply v0.16) |
| `cooccurrence_edge_weight` | `0.2` | `ASSOCIATED_WITH` down-weight in orphan PageRank |
| `orphan_pr_percentile_threshold` | `5.0` | Bottom-percentile cut-off |
| `source_chunk_ids_gc_min_dead` | `1` | Min dead-UUID count to plan GC for an entity |
| `centroid_lev_threshold` | `2` | Max intra-cluster Levenshtein for centroid path |
| `centroid_min_intra_cluster_cosine` | `0.88` | Multimodal-cluster floor |
| `dedupe_entities_default_threshold` | `0.90` | Fallback cosine merge threshold |
| `dedupe_entities_per_type_thresholds` | `{}` | Per-`entity_type` overrides (e.g. `{"PERSON": 0.95}`) |
| `event_clustering_cosine_threshold` | `0.95` | Chronicle event near-dup threshold |
| `event_clustering_window_days` | `7` | Sliding `referenced_date` window half-width |

### Op kinds

`khora.dream.plan.OpKind` is a `StrEnum`. The *set* of values may grow during Phase 0; existing values are append-only.

| Member | Value | Phase | Apply mode |
|---|---|---|---|
| `VECTORCYPHER_SCHEMA_DRIFT_REPORT` | `vectorcypher_schema_drift_report` | 1 (audit) | pass-through |
| `VECTORCYPHER_ORPHAN_REPORT` | `vectorcypher_orphan_report` | 1 (audit) | pass-through |
| `VECTORCYPHER_SOURCE_CHUNK_IDS_AUDIT` | `vectorcypher_source_chunk_ids_audit` | 1 (audit) | pass-through |
| `CHRONICLE_ABSTENTION_DRIFT_REPORT` | `chronicle_abstention_drift_report` | 1 (audit) | pass-through |
| `CHRONICLE_TOMBSTONE_AUDIT` | `chronicle_tombstone_audit` | 1 (audit) | pass-through |
| `VECTORCYPHER_DEDUPE_ENTITIES` | `vectorcypher_dedupe_entities` | 2 (planner) | raises `NotImplementedError` until v0.16 (#669) |
| `VECTORCYPHER_CENTROID_RECOMPUTE` | `vectorcypher_centroid_recompute` | 2 (planner) | raises `NotImplementedError` until v0.16 |
| `VECTORCYPHER_SOURCE_CHUNK_IDS_GC` | `vectorcypher_source_chunk_ids_gc` | 2 (planner) | raises `NotImplementedError` until v0.16 |
| `CHRONICLE_FACT_COMPACTION` | `chronicle_fact_compaction` | 2 (planner) | raises `NotImplementedError` until v0.16 |
| `CHRONICLE_EVENT_CLUSTERING` | `chronicle_event_clustering` | 2 (planner) | `apply_event_clustering` raises `NotImplementedError` until v0.16 |

### Plan / scope / result dataclasses

All in `khora.dream.plan` / `khora.dream.result`. Frozen slotted dataclasses.

- `DreamScope(op_kinds, since, until, entity_ids, document_ids)` — **public**. `None` fields = no restriction.
- `DreamResult(run, diff, ops, llm_usage, metadata)` — **public**. `metadata` carries `plan_hash` + `plan_payload`.
- `DreamRunInfo(run_id, namespace_id, mode, started_at, finished_at, duration_ms, resume_of)` — **public**.
- `DreamMode = Literal["dry-run", "apply"]` — **public**.
- `DreamOp`, `DreamPlan`, `Checkpoint`, `DreamDiff`, `DreamProgress`, `OpSummary` — **internal**.

### Planner functions

Coroutines returning `DreamOp` (or `tuple[DreamOp, ...]`). **Internal** stability — call via the orchestrator unless writing tests.

```python
# Chronicle (khora.dream.engines.chronicle)
plan_chronicle_abstention_drift(namespace_id, *, engine, config, sample_rate=None)
plan_chronicle_tombstone_audit(namespace_id, *, session, config, recommended_retention_days=365)
plan_chronicle_fact_compaction(namespace_id, *, session, config, apply=False)  # apply=True → NotImplementedError
plan_chronicle_event_clustering(namespace_id, *, session, config)
apply_event_clustering(*args, **kwargs)                                         # raises NotImplementedError

# Vectorcypher (khora.dream.engines.vectorcypher)
plan_vectorcypher_schema_drift(namespace_id, *, coordinator, expertise, previous_run_id=None, ...)
plan_vectorcypher_orphan_report(namespace_id, *, coordinator, expertise=None, pr_percentile_threshold=5.0, ...)
plan_vectorcypher_source_chunk_ids_audit(namespace_id, *, coordinator, top_k_offenders=20)
plan_vectorcypher_source_chunk_ids_gc(namespace_id, *, coordinator, min_dead=1, mode="dry-run")
plan_vectorcypher_centroid_recompute(namespace_id, *, coordinator, merge_clusters, mode="dry-run", ...)
plan_vectorcypher_dedupe_entities(namespace_id, *, coordinator, default_threshold=0.90, mode="dry-run", ...)
```

For Phase 2 functions, `mode="apply"` raises `NotImplementedError`.

### Reporting sinks

Module `khora.dream.report`. All three implement `ReportSink` (`emit`, `flush`, `close`). The sinks themselves are **internal**; the `ReportSink` protocol is **public**.

```python
class DreamFileSink(base_dir: str | Path, *, redact_text: Literal["none","summary","all"] = "summary")
class DreamEventSink(dispatcher: HookDispatcher, *, delivery: Literal["sync","async"] = "async",
                     outbox_maxsize: int = 10_000, subscription_class: str = "dream")
class DreamCollectorSink()  # stateless; OTel spans + metrics
```

Also exported: `SCHEMA_VERSION = "dream-report/1"`, `load_manifest(path)`, `expire_dream_reports(...)`, `record_llm_tokens(*, direction, model, tokens)`, `record_undo_invocation(*, op_type, outcome)`, `DreamReportSchemaMismatchError`.

### Exceptions

All inherit from `khora.exceptions.KhoraError`. **Public** — pattern-match these from job runners.

- `DreamDisabledError` — `DreamConfig.enabled` is False
- `DreamForbiddenOpError` — plan contains a forbidden op (document delete, UNIQUE-collision write, read-only namespace)
- `DreamRunStuckError(run_id, heartbeat_age_seconds)` — prior run is `applying` with a stale heartbeat; resolve via `resume_from=run_id`
- `DreamLockUnavailable(namespace_id, timeout_seconds)` — advisory lock contention

### Protocol

```python
@runtime_checkable
class DreamCapable(Protocol):
    @property
    def dream_capabilities(self) -> frozenset[OpKind]: ...
    async def plan_dream(self, namespace_id, *, scope, config, expertise=None) -> DreamPlan: ...
    async def apply_dream(self, plan, *, checkpoint=None, on_progress=None) -> DreamResult: ...
```

Lives at `khora.dream.protocol.DreamCapable`. Engines opt into dream-phase by structurally implementing it; the orchestrator runtime-checks via the registered plugins in `engines/registry.py`.

## Storage substrate

### Migration 032 — `khora_dream_runs`

Postgres-only checkpoint table for crash-resume semantics.

| Column | Type | Purpose |
|---|---|---|
| `run_id` | UUID PK | |
| `namespace_id` | UUID NOT NULL | indexed; query handle |
| `trigger` | VARCHAR(32) | `"manual" \| "resume" \| ...` |
| `mode` | VARCHAR(16) | `"dry-run" \| "apply"` |
| `state` | VARCHAR(32) | `init \| planning \| applying \| completed \| cancelled \| crashed` |
| `plan_hash` | VARCHAR(64) | first 16 hex chars of canonical-JSON SHA1; detect plan drift on resume |
| `started_at`, `heartbeat_at`, `finished_at` | TIMESTAMPTZ | |
| `last_committed_op_seq` | INTEGER | resume cursor |
| `total_ops`, `total_decisions` | INTEGER | |
| `report_path`, `manifest_sha256`, `config_fingerprint` | varchar | |
| `error` | JSONB | populated on `state="crashed"` |

The embedded path (sqlite_lance) mirrors equivalent state via the file sink — migration 032 is a clean no-op on SQLite.

### Migration 033 — bi-temporal columns

Adds three NULLable columns to both `relationships` and `memory_facts`:

| Column | Semantics |
|---|---|
| `valid_to` | Real-world end of validity; NULL = "still valid" |
| `invalidated_at` | When the system marked this row superseded |
| `invalidated_by` | UUID of the dream op (or future apply-mode operation) that invalidated it |

Plus Postgres-only partial composite indexes `ix_relationships_live` and `ix_memory_facts_live` over `WHERE invalidated_at IS NULL`. Existing query paths filtering on the legacy `memory_facts.is_active` flag keep working unchanged; the two coexist until a future major version.

These columns are unused by Phase 1 audit and Phase 2 planner ops. They're already in place because apply mode (v0.16) needs them, and migrations land best ahead of the code that depends on them.

## Scheduling

A minimal cron recipe — same shape works under Temporal, k8s CronJob, GitHub Actions schedules, etc.

```python
# scripts/dream_nightly.py
import asyncio
from uuid import UUID
from khora import Khora, KhoraConfig, DreamConfig
from khora.dream.exceptions import DreamLockUnavailable, DreamRunStuckError

NAMESPACE_ID = UUID("...")  # stable namespace_id, not the row-level id

async def main() -> None:
    kb = Khora(config=KhoraConfig(dream=DreamConfig(
        enabled=True,
        report_file_sink_enabled=True,
        report_collector_sink_enabled=True,
    )))
    try:
        async with kb:
            result = await kb.dream(NAMESPACE_ID, mode="dry-run")
        # Wire result.ops + your alerting here. Failing counters are the
        # interesting signal; planned/skipped just describe the namespace.
        print(result.ops)
    except DreamLockUnavailable as e:
        # Another run is in flight against this namespace. Exit non-zero so
        # cron alerts on overlapping schedules but don't bash an in-flight run.
        raise SystemExit(75) from e
    except DreamRunStuckError as e:
        # Prior run crashed mid-apply. Surface to oncall; do NOT auto-resume.
        # Operator decides whether to `kb.dream(..., resume_from=e.run_id)`.
        raise SystemExit(2) from e

asyncio.run(main())
```

Backing out a rollout: unset `KHORA_DREAM_ENABLED` (or set `dream.enabled=False` in config). `Khora.dream()` then raises `DreamDisabledError` cleanly without touching the DB. The `khora_dream_runs` table and bi-temporal columns from migrations 032/033 remain in the schema (zero-cost when unused).

## Concurrency

A dream run holds a Postgres advisory lock — `pg_advisory_xact_lock`, ID derived from `namespace_id` via `blake2b` (domain-separated from the migration lock). A second concurrent run against the same namespace fast-fails with `DreamLockUnavailable`. Different namespaces dream in parallel without contention.

On embedded backends the lock degrades to an in-process `asyncio.Lock` keyed by `namespace_id`. Cross-process safety is **not** promised on sqlite_lance / surrealdb embedded — multi-process workers against an embedded DB must serialize dream calls themselves.

`resume_from=<run_id>` re-enters a crashed run from `khora_dream_runs.last_committed_op_seq + 1`. The plan is re-validated against the current world state; ops whose preconditions changed are marked `SKIPPED_STALE`. Cancel is between ops only — the current op completes (or rolls back via its own short-lived transaction) before the runner halts, setting `state="cancelled"` on `khora_dream_runs`.

## Research & Prior Art

The dream phase is not a novel invention. It is a deliberate composition of patterns the systems and ML communities have used for decades, applied to long-lived agentic memory stores. This section traces the intellectual lineage and is honest about which parts are load-bearing analogy versus direct re-implementation under a new name.

### Sleep and memory consolidation in neuroscience

The "dream" naming is borrowed from the complementary learning systems (CLS) framework: McClelland, McNaughton & O'Reilly, *"Why there are complementary learning systems in the hippocampus and neocortex"* (Psychological Review, 1995). The thesis — that fast, episodic encoding (hippocampus) and slow, structured consolidation (neocortex) require **separate substrates** to avoid catastrophic interference — maps cleanly onto online ingest (`Khora.remember`) versus offline replay (`Khora.dream`). Subsequent replay-during-sleep work (Wilson & McNaughton 1994; review in Klinzing, Niethard & Born, *Nature Neuroscience*, 2019) showed hippocampal sequence replay during slow-wave sleep driving cortical integration. The agent analog is not literal — there is no biological-fidelity claim — but the architectural shape (write-fast, reorganize-later, on a different schedule and with a different objective function) is the same.

### Offline RL replay buffers

Experience replay in DQN (Mnih et al., *"Human-level control through deep reinforcement learning"*, Nature 2015) and prioritized experience replay (Schaul et al., ICLR 2016, [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)) follow the same "ingest in one regime, consolidate in another" pattern. The replay buffer is to a Q-network what the namespace is to a Khora agent: an accumulator that the offline pass samples from to update the canonical representation. The audit-then-plan-then-apply split mirrors how RL frameworks separate trajectory collection from gradient updates.

### Database compaction and tombstone GC

Dream's `chronicle_fact_compaction` op is tombstone GC under a different name. LSM-tree compaction (O'Neil et al. 1996, *Acta Informatica*; see RocksDB and LevelDB design docs), Cassandra's `gc_grace_seconds` + SSTable compaction, HBase major compaction, and Postgres `VACUUM`/`autovacuum` are all instances of the same idea: tombstones accumulate at write time, a background pass reclaims space and rewrites the canonical store. Khora's `memory_facts` table already carries the tombstone columns; dream phase is where the GC eventually runs.

### Log compaction in distributed systems

Kafka log compaction (Kreps, Narkhede & Rao, *"Kafka: a distributed messaging system for log processing"*, NetDB 2011; see the Kafka design docs on compacted topics) collapses a per-key event stream to the latest value per key. Event-sourcing snapshots (Vernon, *Implementing Domain-Driven Design*, 2013) do the same for aggregate state. Dream's `chronicle_event_clustering` op is the same pattern applied to `chronicle_events` — collapse near-duplicate or causally-linked events into a single canonical representation while retaining the raw log for audit.

### Agentic memory frameworks

These systems define the *write* side of agent memory well; dream phase targets the *consolidation* side they each defer.

| System | Citation | What it does well | Gap dream phase addresses |
|---|---|---|---|
| MemGPT | Packer et al., [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) (2023, rev. 2024) | OS-style paged memory, recall vs. archival tiers | No structural audit of archival tier over time |
| GraphRAG | Edge et al., [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) (2024) | Community-summary index built at ingest | Re-indexing is full-rebuild; no incremental drift detection |
| Self-RAG | Asai et al., [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) (2023) | Retrieval-on-demand with reflection tokens | Online only; no offline corpus hygiene |
| Letta / Mem0 | Letta docs; Mem0 OSS | Structured user-facing memory blocks | No scheduled compaction or entity dedupe pass |

Dream phase is **complementary** to all four: it operates on the same substrate they write to, on a different cadence, with a different objective.

### Knowledge-graph maintenance

Entity resolution has a well-developed literature. Köpcke & Rahm, *"Frameworks for entity matching: a comparison"* (Data & Knowledge Engineering, 2010, [doi:10.1016/j.datak.2009.10.003](https://doi.org/10.1016/j.datak.2009.10.003)) and Christen, *Data Matching* (Springer, 2012) survey blocking, similarity functions (Levenshtein, Jaro-Winkler, embedding cosine), and threshold tuning. Khora's per-type thresholds (PERSON 0.92, DATE 0.95, default 0.85 online / 0.90 offline) sit squarely in this tradition. Centroid fusion for cluster representatives is the textbook follow-on step. Dream's `vectorcypher_dedupe_entities` is a scheduled re-run of the same algorithms the ingest pipeline runs once per document, this time across the entire namespace with the benefit of accumulated evidence.

### Tombstones, soft-delete, and bi-temporal modeling

Snodgrass, *Developing Time-Oriented Database Applications in SQL* (Morgan Kaufmann, 1999) is the canonical reference for bi-temporal schemas: `valid_time` (when the fact was true in the world) versus `transaction_time` (when the system knew it). Khora's `valid_to` / `invalidated_at` / `invalidated_by` columns implement exactly this split, which is what lets dream phase soft-delete without losing the ability to answer "what did the agent believe on date X" — a non-negotiable requirement for any system where the memory store feeds downstream decisions that may later be audited.

### OLTP vs. OLAP

The cleanest framing: dream phase is to memory what OLAP is to OLTP. Same store, different access regime, optimized for batch reorganization rather than per-request latency. Kimball's data-warehouse work and the Lambda Architecture (Marz & Warren, *Big Data*, 2015) make the same separation explicit at the systems level. Khora keeps it in-process — no separate warehouse — but the scheduling and access pattern are recognizable.

### Honest limits

Dream phase does not solve memory drift. It provides the substrate to **detect** drift (audit mode, live in v0.14) and **plan** corrective ops (planner mode, live in v0.15). The `apply` mode that would actually execute dedupe / compaction / clustering raises `NotImplementedError` as of v0.15. Treat the current release as instrumentation; the mutation ops land in v0.16 once the planning signals have been validated against real namespaces.

### LLM usage

**Dream phase makes zero LLM calls in v0.15.0.** Every op is pure SQL + pure-Python — entity dedupe uses pairwise embedding cosine (`_accel.batch_dot_product`), centroid recompute uses Levenshtein over name strings (`rapidfuzz`), event clustering uses SVO-summary cosine on pre-normalized embeddings, schema drift is multiset-diff against `ExpertiseConfig`, abstention drift reads OTel histograms. No prompts exist; no `litellm.acompletion` is called from any planner or audit op.

The plumbing for LLM-using ops is in place: `DreamConfig.llm_max_tokens_per_run` (default 200k) and `llm_max_tokens_per_namespace_per_day` (default 1M) are reserved budgets; `khora.dream.llm_call` span and `khora.dream.llm.tokens` metric are declared in the telemetry contract. These remain inert until Phase 5 (#670–#673) lands ops that genuinely need an LLM — community-summary generation (GraphRAG-style summaries over PageRank communities), contradiction detection across `memory_facts`, and operator-supplied schema-drift normalization. Those will be opt-in per op via `DreamConfig.ops.*` flags, with the budgets enforced before each call.

If you see token cost attributed to khora in v0.15, it is **not** from dream phase. The cost paths are recall-time HyDE expansion, ingest-time extraction, and the optional Level-2 hook evaluator — all documented separately under [observability.md](observability.md).

## Stability

| Symbol | Tag | Notes |
|---|---|---|
| `Khora.dream()`, `dream_status()`, `dream_history()` | **public** | Coordinated release with `khora-cli` / `khora-explorer` |
| `DreamConfig`, `DreamResult`, `DreamRunInfo`, `DreamMode`, `DreamScope`, `OpKind` | **public** | Re-exported from top-level `khora` |
| Dream-specific exceptions (`DreamDisabledError`, `DreamForbiddenOpError`, `DreamRunStuckError`, `DreamLockUnavailable`) | **public** | Pattern-match from job runners |
| `DreamOp`, `DreamPlan`, `DreamReportEvent`, `DreamProgress`, `DreamCapable` | **internal** | Importable but may evolve |
| `OpKind` enum *values* | **internal** | New values land per ticket; names may be renamed |
| Top-level OTel spans + aggregate metrics | **public** | Pin dashboards safely |
| Per-op OTel spans | **internal** | Names may evolve |
| Planner functions (`plan_*`) and the orchestrator | **internal** | Call via `Khora.dream()` unless writing tests |

## What's NOT in v0.15.0 — planned for v0.16+

Tracked under the umbrella [#649](https://github.com/DeytaHQ/khora/issues/649).

- **Phase 4 — apply mode** (#667, #668, #669). Flips the five Phase 2 planners from `NotImplementedError` to real mutation. The bi-temporal columns from migration 033 carry the audit trail.
- **Phase 5 — advanced operations** (#670, #671, #672, #673). Community detection + summaries (LLM-heavy, opt-in), edge pruning by weight × recency, contradiction detection, schema-drift normalization with an operator-supplied mapping.

The deliberate gate: do operators actually call `kb.dream()` against production namespaces? `khora.dream.runs_total` is the telemetry to watch. If yes, Phase 4 follows in v0.16. If no, the mutation track is deprecated and the planner ops stay as audit-grade signals.

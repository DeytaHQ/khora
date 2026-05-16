# Dream phase

`khora.dream` is an **offline maintenance pass** for accumulated memory state. It audits a namespace and emits structured reports describing drift, stale data, schema mismatches, and graph-health issues. In v0.14.0 it is **audit-only**: it never mutates the graph, never deletes a row, never calls an LLM. Operators run it on a schedule (cron, Temporal, k8s CronJob) and consume the output through three sinks.

This document is the operator-facing guide. The architectural decisions and the kill-criterion logic that informed the audit-only scope live in the umbrella tracking issue (`#649`) and the v0.14.0 CHANGELOG entry.

## When to use it

- You want to know whether your abstention thresholds are still calibrated against the observed score distribution after the namespace grew.
- You want to know how many `memory_facts` rows are tombstoned and whether they're old enough to compact.
- You want to know which entity / relationship types appear in your data but aren't declared in `ExpertiseConfig`.
- You want to surface long-tail orphan entities (low PageRank, low mention count) for archival review.
- You want to find entities whose `source_chunk_ids` array has accumulated references to chunks that no longer exist.

If you want to **fix** any of these, that's a separate concern: mutation ops are deferred to a follow-up release. v0.14.0 tells you the truth about your graph and stops there.

## Enable it

The master switch is `KhoraConfig.dream.enabled` (env var `KHORA_DREAM_ENABLED`). Default is `False`; the dream phase is opt-in.

```python
from khora import Khora, KhoraConfig, DreamConfig

kb = Khora(
    config=KhoraConfig(
        dream=DreamConfig(
            enabled=True,
            report_file_sink_enabled=True,   # write reports to disk
            report_event_sink_enabled=False,  # emit MemoryEvent via HookDispatcher
            report_collector_sink_enabled=True,  # OTel spans + metrics
        ),
    ),
)

result = await kb.dream(namespace_id, mode="dry-run")
```

All `DreamConfig` fields are listed in [`docs/configuration.md`](configuration.md) and reachable via the `KHORA_DREAM_*` env-var prefix (single-underscore flat form; `KHORA_DREAM_OPS__<NAME>` for nested op toggles).

## The five audit operations

Each op returns a `DreamOp` with a `decision` string and a structured `outputs` dict. The orchestrator routes those through whichever sinks are enabled. No op writes to any backend table.

### Chronicle: abstention-threshold drift

Reads the OpenTelemetry histogram of `top_score` and `combined_score` values that chronicle's `_compute_abstention_signals` emits on every recall (plus a bounded in-process ring buffer for the no-logfire path). Compares the observed p50/p90/p99 against the configured `abstention_min_top_score`, `abstention_min_chunks`, and `abstention_combined_threshold`.

Possible `decision` values:
- `"recommend"` with `direction="lower"|"raise"|"calibrated"` and a rationale referencing the gap
- `"insufficient_data"` when fewer than `abstention_drift_min_samples` (default 1000) recalls have been observed

The op never auto-tunes thresholds. Threshold changes are operator policy.

### Chronicle: `memory_facts` tombstone audit

Pure SELECT. Counts:
- `active` (legacy `is_active=True`)
- `inactive` (legacy `is_active=False`)
- `invalidated` (bi-temporal `invalidated_at IS NOT NULL`, from migration 033)

Plus `tombstone_ratio`, oldest-tombstone age, p50/p90 ages, and top-K offenders by age. Recommends a `retention_days` threshold (default 365) for the eventual compaction op to consume.

### VectorCypher: schema drift vs `ExpertiseConfig`

Multiset-diff between observed `entity_type` / `relationship_type` strings and what `ExpertiseConfig` declares. Four output buckets:

| Bucket | Meaning |
|---|---|
| `new_entity_types` | Present in data, not declared in `ExpertiseConfig` |
| `unused_entity_types` | Declared in config, not used in data |
| `entity_frequency_delta` | Frequency changed by ≥50% since the previous dream run |
| `*_relationship_types` | Same three buckets for relationship types |

Never normalizes type names. `ExpertiseConfig` is declarative user intent; rewriting types in the data is a separate policy decision.

### VectorCypher: PageRank-based orphan report

Builds the entity-relationship graph for the namespace, down-weights `ASSOCIATED_WITH` co-occurrence edges to `0.2` (so they don't dominate), runs the `_accel.pagerank` Rust kernel, then flags entities matching all of:
- PR score in the bottom `orphan_pr_percentile_threshold` percentile (default 5)
- `mention_count ≤ 1`
- No recent recall hits

Output is a list of `{entity_id, name, type, pr_score, mention_count}` with `archive_candidate=true`. The op never archives.

### VectorCypher: `source_chunk_ids` array-length audit

Joins entities × chunks (Postgres `unnest`; SQLite Python-side) and reports:
- Total dead UUID references (chunks that no longer exist)
- Array-length distribution (p50, p90, p99, max)
- Top-K offenders by array length

Surfaces the GC candidates without modifying any row.

## Output channels (sinks)

Three sinks, all consuming the same `DreamOp` stream. Enable them independently via `DreamConfig.report_*_sink_enabled`.

### File sink

Writes per-run artifacts under `{base_dir}/{namespace_id}/{date}/{run_id}.*`:

| File | Contents |
|---|---|
| `summary.md` | Human-readable executive summary + sampled high-impact ops |
| `events.jsonl` | One `DreamOp` per line, machine-readable, schema-versioned (`dream-report/1`) |
| `manifest.json` | Run metadata + checksum |
| `undo.json` | Empty in v0.14.0 (audit ops have no undo state); reserved for the apply-mode release |

Schema version is asserted on read. `redact_text` (`"none" | "summary" | "all"`, default `"summary"`) governs raw-text exposure across all three sinks.

Retention is `DreamConfig.retention_days` (default 30) and `retention_runs_per_namespace` (default 50). Rotation is a sweep, not real-time.

### Event sink

Bridges into the existing `HookDispatcher` via six new `EventType.DREAM_*` values:

- `DREAM_RUN_STARTED`
- `DREAM_PHASE_STARTED`
- `DREAM_OP_DECIDED`
- `DREAM_PHASE_COMPLETED`
- `DREAM_RUN_COMPLETED`
- `DREAM_RUN_FAILED`

Existing `SemanticFilter` filters work — including the new low-cost level-0 fields `dream_op_types` and `dream_decisions`. Operator-defined callbacks subscribing to `DREAM_OP_DECIDED` receive a payload-shape identical to a single line of `events.jsonl`.

### Collector sink (OpenTelemetry)

Emits spans and metrics declared in `docs/telemetry-contract.json`. The drift gate at `tests/unit/telemetry/test_contract.py` enforces that any new span / metric introduced by the orchestrator or an op is registered.

**Public top-level spans** (operator-facing, stable):
- `khora.dream.run`
- `khora.dream.phase`
- `khora.dream.llm_call` (unused in v0.14.0 — reserved for Phase 4)
- `khora.dream.undo` (unused in v0.14.0 — reserved for Phase 4)

**Internal inner spans** (may evolve; do not pin dashboards to these names):
- `khora.dream.op`, `khora.dream.entity_merge`, `khora.dream.edge_prune`, `khora.dream.community_summary`
- Per-op: `khora.dream.chronicle.abstention_drift`, `khora.dream.chronicle.tombstone_audit`, `khora.dream.vectorcypher.schema_drift`, `khora.dream.vectorcypher.orphan_report`, `khora.dream.vectorcypher.source_chunk_ids_audit`

**Public metrics** (aggregate-only, never labelled by `namespace_id` per the cardinality rule):
- `khora.dream.runs_total {trigger, outcome}`
- `khora.dream.run.duration {trigger, outcome}` (histogram, seconds)
- `khora.dream.phase.duration {phase, outcome}` (histogram)
- `khora.dream.ops_total {phase, op_type, decision}`
- `khora.dream.llm.tokens {direction, model}` (reserved for Phase 4)
- `khora.dream.undo_invocations_total {op_type, outcome}` (reserved for Phase 4)
- `khora.dream.report.write_failures_total {reason}` (internal)

All free-text attributes (rationale strings, entity names, recommended-threshold text) go through `khora.telemetry.bounded_text_hash` before becoming span attributes. Raw text is never exposed as a label.

## API surface

Three public entry points on `Khora`:

```python
async def dream(
    self,
    namespace_id: UUID,
    *,
    mode: Literal["dry-run", "apply"] = "dry-run",
    scope: DreamScope | None = None,
    config: DreamConfig | None = None,
    on_progress: Callable[[DreamProgress], Awaitable[None]] | None = None,
    resume_from: UUID | None = None,
) -> DreamResult: ...

async def dream_status(self, run_id: UUID) -> DreamRunInfo | None: ...

async def dream_history(self, namespace_id: UUID, *, limit: int = 20) -> list[DreamRunInfo]: ...
```

In v0.14.0, `mode="apply"` is accepted by the API surface but the orchestrator's apply path is a no-op pass-through for the five audit ops (they have no destructive side effect). The real apply-mode logic for mutation ops lands in the follow-up release.

`resume_from=<run_id>` re-enters a crashed run from `khora_dream_runs.last_committed_op_seq + 1`. The plan is re-validated against the current world state; ops whose preconditions changed are marked `SKIPPED_STALE`.

Cancel: between ops only. The current op completes (or rolls back via its own short-lived transaction) before the runner halts. Sets `state="cancelled"` on `khora_dream_runs`.

## Storage substrate

Two migrations land in v0.14.0:

### Migration 032 — `khora_dream_runs`

| Column | Type | Purpose |
|---|---|---|
| `run_id` | UUID PK | |
| `namespace_id` | UUID NOT NULL | indexed; query handle |
| `trigger` | VARCHAR(32) | `"manual" \| "resume" \| ...` |
| `mode` | VARCHAR(16) | `"dry-run" \| "apply"` |
| `state` | VARCHAR(32) | `"init" \| "planning" \| "applying" \| "completed" \| "cancelled" \| "crashed"` |
| `plan_hash` | VARCHAR(64) | canonical-JSON SHA1 of the plan; used to detect plan drift on resume |
| `started_at`, `heartbeat_at`, `finished_at` | TIMESTAMPTZ | |
| `last_committed_op_seq` | INTEGER | resume cursor |
| `total_ops`, `total_decisions` | INTEGER | |
| `report_path`, `manifest_sha256`, `config_fingerprint` | varchar | |
| `error` | JSONB | populated on `state="crashed"` |

Postgres-only. The embedded path (sqlite_lance) mirrors equivalent state via the file sink — running migrations against SQLite is a clean no-op for 032.

### Migration 033 — bi-temporal columns

Adds three NULLable columns to both `relationships` and `memory_facts`:

| Column | Semantics |
|---|---|
| `valid_to` | Real-world end of validity; NULL = "still valid" |
| `invalidated_at` | When the system marked this row superseded |
| `invalidated_by` | UUID of the dream op (or future apply-mode operation) that invalidated it |

Plus Postgres-only partial composite indexes `ix_relationships_live` and `ix_memory_facts_live` over `WHERE invalidated_at IS NULL`. Existing query paths that filter on the legacy `memory_facts.is_active` flag keep working unchanged; the two coexist until a future major version.

## Stability

| Symbol | Tag | Notes |
|---|---|---|
| `Khora.dream()`, `Khora.dream_status()`, `Khora.dream_history()` | **public** | Subject to coordinated release with `khora-cli` / `khora-explorer` |
| `DreamConfig`, `DreamResult`, `DreamMode`, `DreamScope`, `OpKind` | **public** | Re-exported from top-level `khora` |
| `DreamOp`, `DreamPlan`, `DreamReport`, `DreamProgress`, `DreamCapable` | **internal** | Importable but may evolve through Phase 1-3 without a major bump |
| Top-level OTel spans (`khora.dream.run`, `khora.dream.phase`) and aggregate metrics (`khora.dream.runs_total`, etc.) | **public** | Pin dashboards safely |
| Per-op spans (`khora.dream.<engine>.<op>`) | **internal** | Names may evolve |
| `OpKind` enum values | **internal** | New values land per ticket; names may be renamed |

## Concurrency

A dream run holds a Postgres advisory lock (`pg_advisory_xact_lock`, ID derived from `namespace_id` via `blake2b` for domain separation from the migration lock). A second concurrent run against the same namespace fast-fails with `DreamLockUnavailable`. Different namespaces dream in parallel without contention.

On embedded backends the lock degrades to an in-process `asyncio.Lock` keyed by `namespace_id`; cross-process safety is **not** promised on sqlite_lance / surrealdb embedded. Operators running multi-process workers against an embedded DB must serialize dream calls themselves.

## What's NOT in v0.14.0 (planned follow-ups)

These are tracked under the umbrella `#649`:

- **Phase 2 — mutation-planning ops, dry-run only.** Cross-batch entity resolution (`#658`), centroid recompute (`#660`), `source_chunk_ids` GC (`#662`), chronicle compaction (`#664`), chronicle event clustering (`#665`). Tickets are written; rollout is gated on operator uptake of v0.14.0 audit ops.
- **Phase 4 — apply mode.** Flips Phase 2 ops to actually mutate state. Bi-temporal soft-delete is already in the schema (migration 033) but unused until apply mode lands.
- **Phase 5 — advanced operations.** Community detection + summaries (LLM-heavy, opt-in), edge pruning by weight × recency, contradiction detection, schema-drift normalization (operator-supplied mapping).

The deliberate gate: do operators actually call `kb.dream()` against production namespaces? `khora.dream.runs_total` is the telemetry to watch. If yes, Phase 2 follows. If no, the mutation track is deprecated.

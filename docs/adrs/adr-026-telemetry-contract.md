# ADR-026: Telemetry Public Surface as a JSON Contract

- **Status:** Accepted
- **Date:** 2026-05-03
- **Deciders:** Khora architecture team
- **Related:** [ADR-022](adr-022-extraction-skills-public-api.md) (extraction skills), [ADR-024](adr-024-memory-lake-public-api.md) (memory-lake API), [ADR-025](adr-025-embedded-backend-realignment.md) (embedded realignment)
- **Supersedes:** none
- **PRs:** #504, #505, #506, #507, #508, #509

## Context

khora and its sibling project anima both ship a `telemetry/` package with the same 12-file layout. The packages were soft-forked rather than vendored, so the public surface — collector classes, event Pydantic models, span names, metric names, pipeline-stage names — has the same shape but is allowed to specialise (e.g. anima keys events on `agent_id`, khora keys on `namespace_id`). Until 2026-05 there was no machine-readable record of what that surface actually was; nothing failed CI when a developer added a new span or renamed a metric, and dashboards downstream of Logfire silently drifted.

Two changes raise the cost of that drift:

1. **OSS release plan.** khora ships as an independent OSS package (per the production-readiness scoping in ADR-025). Once external consumers wire dashboards or alerts onto khora's telemetry names, those names become a public API in everything but name. A rename or removal becomes a breaking change.
2. **Telemetry was being collected but not read.** The Devil's Advocate audit during the Phase-0 review of the v0.9 telemetry workstream specifically called out that the PostgreSQL-backed `TelemetryCollector` was wired but no operator dashboard or alert consumed the events. Locking the contract is a prerequisite for the dashboard / alert work that follows; it does not by itself fix the under-utilisation.

The strategic analysis that informed this ADR is collected in:

- `/tmp/khora-telemetry-architecture.md` — high-level architectural review of the collector seam, NoOp shape, and the LLM/Storage/Pipeline event shapes.
- `/tmp/khora-telemetry-sre.md` — SRE-perspective review of metric cardinality, billing exposure (Logfire / Prometheus charge per series), and alert ergonomics.
- `/tmp/khora-telemetry-rag.md` — RAG-quality view: which spans answer "is recall actually working", which metrics back abstention dashboards.
- `/tmp/khora-telemetry-critique.md` — devil's-advocate case against locking down names too early.
- `/tmp/khora-telemetry-phase0.md` — initial inventory and the cardinality audit that produced the ban on `namespace_id` as a metric label.

These are scratch-pad reports; they are not committed to the repo.

## Decision

### 1. Codify khora's telemetry public surface as a JSON contract

`docs/telemetry-contract.json` lists every:

- Public export from `khora.telemetry.__all__` (19 names).
- Pydantic event type (`LLMEvent`, `StorageEvent`, `PipelineEvent`) with its full field set (3 types).
- Collector method (`record_llm_call`, `record_storage_op`, `record_pipeline_stage`) (3 methods).
- `trace_span(...)` call site (58 spans — 22 tagged `public`, 36 tagged `internal`).
- `pipeline_stage(...)` / `record_pipeline_stage(stage=...)` pair (22 stages, all currently `internal`).
- `metric_counter` / `metric_histogram` / `metric_gauge_callback` registration (21 metrics — 16 `public`, 5 `internal`).

Each item carries a `stability` tag. Items tagged `public` cannot break without a major version bump and prior coordination with downstream consumers (genesis, khora-benchmarks, khora-explorer, khora-cli). Items tagged `internal` may be renamed freely as long as the contract is updated in the same PR. The sibling `docs/telemetry-contract.md` is the human-facing explainer.

### 2. Enforce the contract via a drift-detection test

`tests/unit/telemetry/test_contract.py` runs in CI as a 10-test gate. It asserts that:

1. Every name in `public_exports` is in `khora.telemetry.__all__`.
2. Every event type's Pydantic field set matches the contract exactly (renames, type changes, removals all fail).
3. `TelemetryCollector` and `NoOpCollector` both expose every `collector_method` as a callable.
4. Every `trace_span(...)` / `pipeline_stage(...)` / `metric_*(...)` call discovered via `ripgrep` walk of the codebase appears in the contract — adding a new instrumentation point without updating the JSON fails CI.
5. Every span name matches `^khora\.[a-z0-9_]+(\.[a-z0-9_]+)*$`.

### 3. Do NOT extract a shared `deyta-telemetry` package

We considered three shapes:

- **Shape A — shared `deyta-telemetry` package.** khora and anima depend on a common library that defines collectors, span helpers, and metric registration. Rejected: release-cycle coordination cost is too high. Cutting a khora release would block on a sibling library tag, and the public-surface specialisation (`namespace_id` vs `agent_id`, different span owners) is correctly different per project — collapsing it forces fake genericity.
- **Shape B — canonicalise + sync.** Each project keeps its own copy but runs a synchroniser that flags drift. Rejected: the divergence is intentional; a sync tool would generate noise on every legitimate per-project change and would still not lock any public surface against external consumers.
- **Shape C (chosen) — contract test per project.** Each project owns its `telemetry-contract.json` and its own drift gate. Names are public API for that project. Cross-project consistency becomes a code-review concern, not a tooling concern.

### 4. Cardinality rules — codified in the contract

The Phase-0 audit found that `namespace_id` had 438 distinct values over the production retention window in one deployment. Logfire and Prometheus bill per metric series, so attaching `namespace_id` as a metric label produces an unbounded cost curve as deployments grow.

- **`namespace_id` is a span attribute and a log field, never a metric label.**
- **Free-text span attributes** (raw user query, document content, chunk text) MUST go through `khora.telemetry.bounded_text_hash` (added in #504), which returns a SHA1[:8] hash. Raw text on a span is a cardinality bomb on top of a privacy hazard.
- **OTel semantic conventions** apply to new attributes: `gen_ai.*` for LLM, `db.*` for storage, `code.*` for stack info. This keeps khora vendor-neutral over the OTel exporter chain.

## Consequences

### Positive

- Names tagged `public` in the contract become part of the OSS API surface for khora — downstream dashboards and alerts can rely on them within a major version line.
- Internal items (e.g. inner-loop spans like `khora.vectorcypher.coherence_boost`) can be renamed freely with a contract-JSON update.
- The CI drift gate catches undeclared additions at PR time, before they leak into a release.
- The contract gives the operator-dashboard work that follows (reading the events that the collector has been writing) a stable name target.

### Negative / accepted limitations

- **The contract does not fix under-utilisation.** Telemetry has been collected to PostgreSQL since 0.4.0 but no operator dashboard or alert consumes the events. Locking the names is a prerequisite for that work; it is not the work itself. The Devil's Advocate audit was specifically right about this.
- **`khora.log.queue.depth`** (gauge, public) reports a proxy: the loguru-handler-error count, not the actual queue size. `loguru>=0.7.3` does not expose `qsize()` on its enqueue handler. The metric is in the contract because the *name* is what dashboards depend on; the implementation can switch when loguru exposes the real value.
- **`coordinator.transaction()` cross-store atomicity remains partial on embedded** (per ADR-025). This is independent of the telemetry contract — span instrumentation does not promise transactional semantics it does not have.
- **The sampler metrics** (`khora.neo4j.pool.sampled.*`) are tagged `internal`; they are an opt-in high-frequency burst-investigation tool, not a stable export.

### Public surface frozen by this ADR

- 16 `public` metrics: `khora.neo4j.pool.{acquire_duration,acquisition_time,timeout,connections.*,utilization}`, `khora.neo4j.session.duration`, `khora.chronicle.abstention_signal`, `khora.chronicle.abstention_combined_score`, `khora.memory.recall.duration`, `khora.memory.ingest.duration`, `khora.llm.tokens`, `khora.llm.cost_usd`, `khora.log.queue.depth`.
- 22 `public` spans: top-level memory-lake entry points (`khora.recall`, `khora.remember`, `khora.remember_batch`, `khora.forget`), the four engine entry points under `khora.vectorcypher.*`, skeleton entry points, embedder API calls, extraction entry points, and the four query-engine spans (`khora.query.{embedding,graph_search,hyde,rerank}`).
- 3 `public` event types with their full Pydantic field set.
- 19 `public` exports in `khora.telemetry.__all__`.

## References

- PR #504 — `bounded_text_hash` helper for free-text span attributes.
- PR #505 — `docs/telemetry-contract.json` + drift-detection test.
- PR #506 — fix for `storage_events.namespace_id` 100% NULL regression (silent break since Feb 2026).
- PR #507 — Chronicle abstention metrics (`khora.chronicle.abstention_signal`, `khora.chronicle.abstention_combined_score`).
- PR #508 — six LLM call sites instrumented (HyDE, listwise rerank, fact extraction, fact reconcile, event extraction; chat was already wired).
- PR #509 — five aggregate metrics (`khora.memory.recall.duration`, `khora.memory.ingest.duration`, `khora.llm.tokens`, `khora.llm.cost_usd`, `khora.log.queue.depth`).
- `docs/telemetry-contract.json` — the contract.
- `docs/telemetry-contract.md` — the human-facing explainer.
- `tests/unit/telemetry/test_contract.py` — the drift gate.

# Khora Telemetry Contract

`docs/telemetry-contract.json` is the machine-readable contract for khora's
telemetry public surface. It is enforced by
`tests/unit/telemetry/test_contract.py`, which runs in CI as a drift gate.

## What the contract guarantees

- **`public_exports`** — every name in `khora.telemetry.__all__`. Renames,
  removals, or signature changes require a major version bump.
- **`event_types`** — Pydantic models (`LLMEvent`, `StorageEvent`,
  `PipelineEvent`) with their full field set. These rows ship to Postgres and
  are read by downstream tooling (cost tracking, dashboards).
  Adding a field with a default is non-breaking; removing or renaming a
  field is breaking.
- **`collector_methods`** — the recording surface
  (`record_llm_call`, `record_storage_op`, `record_pipeline_stage`).
- **`spans`** — every `trace_span("…")` call site in the codebase. Names
  marked `stability: public` are part of the API; downstream dashboards may
  query by these names. Names marked `internal` may be renamed.
- **`pipeline_stages`** — every `(pipeline, stage)` pair passed to
  `pipeline_stage(...)` or `record_pipeline_stage(stage=…)`. Currently all
  marked `internal`.
- **`metrics`** — every metric name registered via
  `metric_counter` / `metric_histogram` / `metric_gauge_callback`. Public
  metrics back the Neo4j-pool dashboards documented in `CLAUDE.md`.
- **`span_name_regex`** — every span name must match this regex. Enforces
  lowercase, dot-separated, `khora.`-prefixed names.

## stability tags

- **public** — appears in dashboards, alerts, or downstream code that we
  don't control. Cannot break without a coordinated major version bump
  across downstream consumers (e.g. khora-cli, khora-explorer).
- **internal** — emitted today, but the names are not part of the public
  contract. Rename freely; just keep this file in sync with the codebase.

## How the drift gate works

`test_contract.py` does four things:

1. Asserts every name in `public_exports` is in `khora.telemetry.__all__`.
2. Asserts every event type's Pydantic field set matches the contract
   exactly.
3. Asserts `TelemetryCollector` and `NoOpCollector` both have every
   `collector_method` as a callable.
4. Walks the codebase via `ripgrep`, parses every `trace_span(...)` /
   `pipeline_stage(...)` / `metric_*(...)` call, and asserts the names
   appear in the contract. **A new span/stage/metric in code without a
   contract update fails CI.**

If you add or rename a span / stage / metric, update this JSON in the
same PR.

## Bumping `version`

The top-level `version` field is the contract format version, not the
khora package version. Bump it only if you change the JSON shape itself
(new top-level keys, new per-item fields). Item additions / removals do
not bump it.

## OSS implication

khora owns its own `telemetry-contract.json` and its own drift gate;
related OSS packages are expected to do the same rather than depend on
a shared telemetry library. Downstream consumers may rely on the names
tagged `public` in this file remaining stable within a major version
line.

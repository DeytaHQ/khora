# ADR-027: Rename `MemoryLake` → `Khora` (drop "Memory Lake" branding)

- **Status:** Accepted
- **Date:** 2026-05-11
- **Deciders:** Khora architecture team
- **Supersedes parts of:** ADR-024 (in-place revision; ADR-024 retitled "Khora Public API")

## Context

The product name "Khora" has been the official package name since the
project began. "Memory Lake" persisted as a parallel brand on the
top-level facade class (`MemoryLake`), in the module filename
(`src/khora/memory_lake.py`), and pervasively in documentation, error
messages, log lines, and a handful of LLM prompts. Two brands for one
product is a forcing function for confusion — particularly as khora
prepares for OSS release alongside `genesis` and `khora-benchmarks`,
both of which inherited the Memory Lake terminology from khora.

This ADR retires the "Memory Lake" brand in a single coordinated
0.10.0 release.

## Decision

1. **Class:** `MemoryLake` → `Khora`. Matches the package-name = primary-class
   convention used by `redis.Redis`, `docker.Docker`, `anthropic.Anthropic`,
   and `openai.OpenAI`. Read-time benefit: `from khora import Khora` is
   the obvious import after `import khora`. Stack-trace path is
   `khora.khora.Khora.recall` (the inner module is hidden by the
   `__init__.py` re-export at call sites).
2. **Module:** `src/khora/memory_lake.py` → `src/khora/khora.py`. The
   six public result types (`RememberResult`, `RecallResult`,
   `BatchResult`, `Stats`, `LLMUsage`, `DocumentResult`) move with it
   and remain importable from `khora` at the top level (preferred) or
   `khora.khora` (submodule).
3. **No deprecation shim.** The 0.10.0 release ships the rename as a
   hard break. There is no `MemoryLake = Khora` alias, and
   `from khora.memory_lake import …` raises `ImportError` on 0.10.0+.
   Downstream consumers (`genesis`, `khora-benchmarks`) update in
   lockstep PRs that land after khora 0.10.0 publishes. Both already
   pinned `khora<0.10` pre-flight (DYT-3969) to prevent Renovate from
   auto-pulling a broken release.
4. **Telemetry namespace kept.** `khora.memory.*` metric names
   (`khora.memory.recall.duration`, `khora.memory.ingest.duration`)
   remain — they describe the generic concept of memory storage, not
   the retired brand. Renaming them would break operator dashboards
   without informational gain. Only the internal `"owner": "memory_lake"`
   JSON tags on four facade spans (`khora.recall`, `khora.remember`,
   `khora.remember_batch`, `khora.forget`) change to `"owner": "khora"`.
5. **LLM prompts rephrased.** The two query-understanding prompts
   (`src/khora/query/understanding.py`) that previously framed the system
   as "a memory lake" now say "a knowledge base." Sent verbatim to the
   LLM; affects ranking only marginally but removes a brand reference
   from user-visible text.
6. **Untouched:** the SurrealDB `memory_namespace` table (generic name,
   not the brand), the `khora_alembic_version` table, the
   `pg_advisory_xact_lock` ID `6001515088189075507`, and the `KHORA_`
   env-var prefix.

## Why a hard break (not a deprecation shim)

A shim is the safer default for OSS libraries. We are choosing to skip
it here because:

- **Low blast radius.** Only two downstreams (`genesis`,
  `khora-benchmarks`) consume the public API; both are first-party and
  ship migration PRs in the same release window.
- **Pre-flight pins.** DYT-3969 pinned `khora<0.10` in both
  downstreams' `pyproject.toml` before the 0.10.0 release tag, so a
  Renovate / `uv lock --upgrade` cycle cannot silently break either
  consumer.
- **Cleanliness.** A shim that warns on every import for a single
  minor cycle, then gets ripped out, adds two PRs (introduce, remove)
  and a window of nominally-deprecated code in `src/khora/` for what
  amounts to one identifier rename.

If a third-party consumer surfaces after release, they will see a
clear `ImportError` at install time naming the missing symbol — easy
to diagnose, easy to fix.

ADR-024 has been updated in place to reflect the new symbol paths and
includes a one-line note pointing at this ADR. The historical title
"Memory Lake Public API" is preserved in the file as a `Note (revised…)`
header so the rename history is recoverable.

## Rollout

1. khora rename PR (DYT-3966) merges; tag `v0.10.0`.
2. Release pipeline publishes khora 0.10.0 + khora-accel 0.10.0 to
   CodeArtifact.
3. khora-benchmarks migration PR (DYT-3967) lands with
   `from khora import Khora` and `khora>=0.10,<0.11`.
4. genesis migration PR (DYT-3968) lands with the same import switch,
   `khora>=0.10,<0.11`, and TUI strings rebranded to plain "Search" /
   "Chat".

## Consequences

**Positive**
- Single brand. `khora` is the package, `Khora` is the class, `khora.recall`
  is the span; no parallel "Memory Lake" vocabulary to teach new users.
- LLM prompts no longer reference a name that's about to be retired
  from external docs.

**Negative**
- One-time coordination cost across three repositories (khora,
  khora-benchmarks, genesis). Already absorbed by the pre-flight pin
  PRs.
- Pickled `MemoryLake` instances with embedded `khora.memory_lake.MemoryLake`
  class paths fail to deserialize on 0.10.0+. None known in production;
  flagged here for completeness.

**Neutral**
- Telemetry namespace and DB schema are unchanged, so dashboards,
  alerts, and migrations carry forward without touchup.

## Related

- ADR-022 — Extraction skills public API (unaffected).
- ADR-024 — Khora public API (revised in-place by this ADR).
- ADR-026 — Telemetry contract (no metric-name change; only internal
  `owner` field on four spans).
- DYT-3965 — EPIC: Remove "Memory Lake" branding from Khora ecosystem.
- DYT-3966 — khora rename PR (this ADR's implementation ticket).
- DYT-3967 — khora-benchmarks migration.
- DYT-3968 — genesis migration.
- DYT-3969 — Pre-flight `khora<0.10` pins on downstreams.

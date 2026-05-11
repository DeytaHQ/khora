# ADR-025: Embedded Backend Realignment for v0.9.0

- **Status:** Accepted
- **Date:** 2026-05-02
- **Deciders:** Khora architecture team
- **Related:** [ADR-024](adr-024-memory-lake-public-api.md) (public API surface)
- **Supersedes:** none
- **Linear umbrella:** DYT-3545

## Context

Through v0.8.x the embedded story for khora drifted across multiple shapes — Kuzu (now archived upstream), SurrealDB (in-tree but with KNN warts), and an emerging SQLite + LanceDB backend (`sqlite_lance`) that landed engine-level integration tests during the DYT-3545 sprint. Around the same time the documentation began to imply that the embedded path was production-ready alongside the PostgreSQL + pgvector + Neo4j stack. A strategic review concluded that this framing was more aspirational than load-bearing:

- **No internal consumer pulls for `memory://` today.** A grep across `genesis`, `khora-benchmarks`, `khora-cli`, `khora-explorer` shows zero non-vendored hits — every downstream constructs a `Khora(KhoraConfig(...))` from a real config. The "embedded persona" is a hypothesis, not an installed user.
- **`memory://` does not parse as a backend selector.** `Khora("memory://")` treats the URL as the PostgreSQL `database_url`. The only `memory://` scheme defined in-tree is the SurrealDB in-process default (`storage/backends/surrealdb/connection.py:71`), which is not what the README's quickstart example was implying.
- **Two-engine seam (SQLite + LanceDB) has structural warts.** `coordinator.transaction()` only enrols the SQL session — graph writes go through raw aiosqlite, LanceDB writes happen post-commit with compensating-delete-on-failure. Partial atomicity is documented but not closed.
- **DYT-3550 (point-in-time queries on embedded) is not implementable** in the recursive-CTE port today. The engine × backend matrix has a structural hole.
- **"Embedded" is ~150 MB unpacked.** `pyarrow>=18` (90–120 MB) + `lancedb>=0.25` native (25 MB) + Arrow C++ runtime. Calling this "no native deps" overpromises — it is "no server", which is a different claim.

The strategic analysis that informed this ADR is collected in:

- `/tmp/khora-embedded-architect.md` — high-level architectural review of the four protocols and the storage abstraction shape.
- `/tmp/khora-embedded-sqlite-lance-analysis.md` — capability matrix, performance reality check, scale ceiling, sharp edges.
- `/tmp/khora-embedded-rag-needs.md` — per-engine recall recipe and the retrieval-correctness floor for VectorCypher / GraphRAG / Skeleton / Chronicle.
- `/tmp/khora-embedded-alternatives.md` — DB-expert survey of 13 candidate stacks (sqlite-vec, pgserver, DuckDB+VSS, Tantivy+Qdrant, Milvus Lite, etc.).
- `/tmp/khora-embedded-critique.md` — devil's-advocate case against shipping `sqlite_lance` "production-ready" in v0.9.0.

## Decision

### 1. Keep SQLite + LanceDB as the recommended embedded stack — but mark it experimental in v0.9.0

`sqlite_lance` is the only embedded backend where every storage protocol (relational, vector, graph, event-store) has a working implementation that has been exercised by integration tests against all four engines (DYT-3545 family). It is not yet stamped production-ready because:

- `coordinator.transaction()` does not deliver 3-way atomicity across SQL + graph + Lance (partial-atomicity-by-compensation, not ACID).
- Point-in-time queries (DYT-3550) are not supported on the CTE port.
- FTS5 covers chunks only; entity-anchored recall falls back to `LIKE` / JSON-equality.
- The `_lance_write_lock` (process-local, asyncio-only) serialises all add/delete during compaction.
- IVF-PQ recall degrades silently as the corpus grows past the initial training threshold — DYT-3579 (#485) added a `retrain_factor` config that retrains when the corpus has doubled (default `2.0`); this mitigates but does not eliminate the wart.

**Documented scale ceiling:** ~1M chunks, ~100k entities, ~500k relationships, traversal depth ≤3. Above these thresholds users should switch to the PostgreSQL + pgvector + Neo4j stack.

### 2. Production-readiness is per (engine × stack), not per engine

The v0.9.0 production-readiness matrix is codified in [`docs/engines/engine-comparison.md`](../engines/engine-comparison.md#production-readiness-by-stack-v090):

| Engine        | PG + pgvector + Neo4j  | PG + pgvector (no graph) | SQLite + LanceDB | SurrealDB     |
|---------------|------------------------|--------------------------|------------------|---------------|
| VectorCypher  | Production-ready       | n/a                      | Experimental     | Experimental  |
| Chronicle     | n/a                    | Production-ready         | Experimental     | Experimental  |
| GraphRAG      | Available              | n/a                      | Experimental     | Experimental  |
| Skeleton      | n/a                    | Available                | Experimental     | Experimental  |

Marketing language should never stamp an engine production-ready in isolation — every claim must name the stack.

### 3. SurrealDB stays in-tree, marked experimental

The SurrealDB backend is feature-complete (relational + vector + graph + KV in one store) but carries enough operational risk in v0.9.0 to warrant the experimental tag:

- Python SDK pinned to `>=2.0.0a1` — alpha track for SurrealDB 3.x compatibility.
- KNN expression `<|K|>` is unreliable in embedded mode; backend falls back to brute-force cosine + HNSW.
- Concurrent upserts require the `_SurrealDBEntityKeyGate` to serialise on `(namespace_id, name, entity_type)` keys.
- BSL-1.1 license — review for downstream packaging concerns before adopting.

### 4. Kuzu graph backend is deprecated in 0.9.0, scheduled for removal in 0.10

Kuzu was acquired by Apple in October 2025 and the upstream repository is archived. The deprecation warning has been live since v0.9.0; the module logs a `DeprecationWarning` on import and on `KuzuBackend()` construction. Removal is scheduled for v0.10.

### 5. Defer the lance-graph integration to v0.10

`lance-graph` is a second 0.x Rust crate. Pulling it into a "production-ready" path in v0.9.0 would commit khora to absorbing two independent native-crate breaking-change cycles (lance + lance-graph) on top of an already partial-atomicity story. The recursive-CTE port in `sqlite_lance/graph.py` is sufficient for the documented scale ceiling. Defer to v0.10 and re-justify with a real embedded user or a benchmark.

### 6. Default embedded URI routing is a v0.10 code change

`Khora("memory://")` currently treats the argument as the PostgreSQL `database_url`. The README quickstart has been corrected in v0.9.0 to recommend `Khora()` (no positional arg) for the production stack and `KHORA_STORAGE_BACKEND=sqlite_lance` for the embedded stack. Routing a `memory://` URI to the recommended embedded stack at the lake constructor is a behavioural change tracked separately for v0.10.

### 7. Defer sqlite-vec / pgserver to v0.10

DB Expert #2 (`/tmp/khora-embedded-alternatives.md`) identified two strong candidates for collapsing the dual-store and closing the partial-atomicity gap:

- **sqlite-vec** — single 1 MB SQLite extension. Vector ANN lives inside the SQLite transaction → true ACID across relational + vector + graph; install footprint drops from ~150 MB to ~5 MB. Blockers: aarch64 manylinux wheel gap, brute-force KNN at 100k+ vectors, v0.x ABI not frozen.
- **`pgserver` (embedded Postgres + pgvector + AGE)** — pip-installable PG cluster. Zero schema fork from the production stack; HNSW recall, real ACID, Cypher via AGE. Blockers: ~100 MB on disk, postmaster spawn cost (~1 s), single-maintainer bus-factor risk on `pgserver` itself.

Both are tracked for v0.10. They are deferred — not rejected — because they materially change the embedded shape, and v0.9.0's job is to be honest about what we have today, not to ship an aspirational replacement.

## Consequences

### Positive

- **Marketing matches reality.** The v0.9.0 surface promises only what is exercised by integration and e2e tests on the production stack. Embedded users get accurate expectations (scale ceiling, atomicity gap, retraining behaviour).
- **No second 0.x Rust crate enters a production-ready path.** lance-graph stays out of v0.9.0.
- **Kuzu's bus-factor risk is contained.** Deprecation in v0.9.0, removal in v0.10.
- **Downstream consumers unaffected.** ADR-024 surface is unchanged; this is a documentation realignment.

### Negative / costs

- **README quickstart no longer one-liner-zero-infra.** Embedded users must set `KHORA_STORAGE_BACKEND=sqlite_lance` and the corresponding `db_path`. The "zero-infrastructure" framing is genuinely harder to deliver honestly than the prior version implied.
- **Two experimental embedded paths to maintain.** `sqlite_lance` and `surrealdb` both carry experimental tags through v0.9.0. Until v0.10's choice is made, neither is stamped production-ready.
- **Documentation cost.** Future PRs adding engines or backends must update the production-readiness matrix in `engine-comparison.md`.

### Risks

- **A v0.10 candidate (sqlite-vec or pgserver) might fail platform support.** Mitigation: defer the choice; ship v0.9.0 honestly; gather one or two embedded user reports before committing.
- **lance-graph could become production-ready upstream before v0.10 ships.** Mitigation: re-evaluate lance-graph in the v0.10 design review; don't pre-commit.

## Alternatives considered

### A. Stamp `sqlite_lance` production-ready in v0.9.0

Rejected. The partial-atomicity gap, missing PIT queries, and stale-IVF-PQ recall behaviour are real. Stamping this combination production-ready commits us to a 991-line hand-rolled CTE Cypher port and a 0.x LanceDB dependency on the production support path. Ship it experimental, document the warts, defer the upgrade decision to v0.10.

### B. Drop the embedded story entirely; ship `khora[demo]` as cosine-in-Python

Rejected, but it was the strongest pushback in the strategic review (`/tmp/khora-embedded-critique.md`). Eliminating LanceDB and the CTE Cypher port entirely (using `_accel.batch_dot_product` for ≤5k chunks, no graph engine) would be honest, would shed ~150 MB of native deps, and would close DYT-3548 / DYT-3549 / DYT-3550 / DYT-3558 as a class. We rejected this because the DYT-3545 family already landed real engine × embedded integration tests and there is concrete value in exercising the full retrieval recipe (vector + BM25 + graph + temporal) against the embedded stack — even at small scale.

### C. Swap `memory://` to route to `sqlite_lance` in v0.9.0

Rejected for v0.9.0; deferred to v0.10. Quietly rerouting an established URI scheme inside a minor release would silently change behaviour for anyone passing `memory://` today (there are zero in-tree non-vendored hits, but the change is large enough to deserve its own PR). Ship the docs first; ship the routing change in v0.10.

### D. Adopt SurrealDB as the single embedded story

Rejected. The Python SDK is on the alpha track (`>=2.0.0a1`), KNN is unreliable in embedded mode, and BSL-1.1 license is a packaging concern downstream. Re-evaluate in v0.10 once the SDK lands stable.

### E. Adopt sqlite-vec or pgserver in v0.9.0

Rejected for v0.9.0. Both are strong contenders (see `/tmp/khora-embedded-alternatives.md` §A and §B) but each is a 1–2 week storage-layer rewrite. v0.9.0's job is to be honest about what we shipped and stop overpromising; v0.10 picks the next embedded shape with the benefit of one or two real-user signals.

## References

- `/tmp/khora-embedded-architect.md` — architectural review.
- `/tmp/khora-embedded-sqlite-lance-analysis.md` — capability + performance reality check.
- `/tmp/khora-embedded-alternatives.md` — DB Expert #2 survey of 13 candidate stacks.
- `/tmp/khora-embedded-rag-needs.md` — per-engine retrieval-correctness floor.
- `/tmp/khora-embedded-critique.md` — devil's-advocate case against shipping embedded as production-ready.
- DYT-3545 (Linear umbrella) and the DYT-354x / DYT-355x / DYT-358x children (engine wiring, retrieval correctness, IVF-PQ retraining).
- DYT-3550 — point-in-time queries on embedded (open).
- DYT-3579 (#485) — `retrain_factor` config for LanceDB IVF-PQ.
- ADR-024 — memory-lake public API (unchanged by this ADR).

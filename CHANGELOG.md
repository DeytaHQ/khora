# Changelog

All notable changes to Khora are documented here.

## [0.3.0] — Engineering Improvements

### Why: removing accidental complexity

Global state in database session management, UUID string wrapping across
52 ORM columns, redundant connection pools for backends sharing the same
database URL, and stale deprecated APIs that no longer matched the
codebase — none of these served users, and all of them created friction
for contributors. This release removes the accidental complexity so the
next round of features lands on cleaner ground.

### UUID migration

All 52 UUID columns in `db/models.py` now declare `as_uuid=True`,
mapping to native Python `uuid.UUID` objects. This is a Python-side-only
change — the PostgreSQL column type remains `UUID`. The practical effect
is that code no longer needs `str()` wrapping when building ORM models
or `UUID()` parsing when reading them. Graph backends (Neo4j, Kuzu,
Memgraph) still convert at the boundary because they don't support
native UUIDs.

### DatabaseManager

`db/session.py` previously used module-level globals for the async
engine and session factory. These are now encapsulated in a
`DatabaseManager` class that owns engine creation, session lifecycle,
and disposal. Backward-compatible module-level wrappers are preserved
so existing callers continue to work without changes.

### Shared connection pools

`StorageFactory` now caches async engines by normalized URL. When
PostgreSQL, pgvector, and the event store all point at the same
database (the common case), they share a single connection pool instead
of creating three independent ones. Backends using a shared engine skip
`dispose()` on disconnect to avoid pulling the pool out from under
siblings.

### TransactionContext

`StorageCoordinator.transaction()` returns an async context manager
that wraps multiple backend writes in a single database transaction.
`TransactionContext.savepoint()` creates nested savepoints for partial
rollback. Backend write methods accept an optional `session` parameter
to join the active transaction.

### Deprecated API cleanup

- `lake.storage` — promoted to stable public API (used by `genesis` and
  `khora-benchmarks`). The deprecation warning has been removed.
- `lake.query_engine` — removed. Use `lake.recall(raw=True)` for
  unprocessed search results.
- `remember_batch_legacy()` — removed. Use `remember_batch()`.

### Chat module tests

71 new tests across 4 files covering the chat module (`chat/engine.py`,
`chat/history.py`, `chat/persona.py`, `chat/prompt.py`). The module
itself is unchanged — these tests document and lock existing behavior.

### spaCy sentence splitting

The semantic chunker now uses spaCy's `sentencizer` component when
available, improving sentence boundary detection. Install with
`pip install khora[nlp]`. The sentencizer is a rule-based component
that ships with spaCy core — no model download needed. When spaCy is
not installed, the chunker falls back to its existing regex-based
splitter transparently.

### Docker removal

The `Dockerfile` and CI `docker-build` job have been removed. Khora is
a library, not a deployable application — the Dockerfile was never used
in production and added maintenance burden. Development databases
continue to use `compose.yaml` via `make dev`.

### Housekeeping

- Version bumped from 0.2.3 to 0.3.0 in `pyproject.toml`,
  `src/khora/__init__.py`, `rust/khora-accel/Cargo.toml`, and
  `rust/khora-accel/pyproject.toml`.

---

## [0.2.3] — Namespace Optimization Design

### Why: surfacing what's real vs. what's aspirational

A team of five specialist agents audited Khora's namespace isolation,
multi-tenancy enforcement, and temporal extraction paths. The audit
found that several documented features — `TenancyMode` routing, ACL
enforcement, bi-temporal edge storage, and the time hierarchy builder —
exist as code but are never exercised at runtime. Meanwhile, the
namespace-level row filtering that *is* active lacks an orphan-entity
cleanup path when documents are deleted. This release ships the
comprehensive design for fixing all of it, marks the stale
documentation, and inventories the dead code so the next releases can
act on it.

### Namespace optimization design

New `docs/design/namespace-optimization-plan.md` lays out a six-phase
implementation roadmap:

1. **Orphan fix** — delete graph entities left behind after `forget()`.
2. **Data-model hardening** — add `namespace_id` to Neo4j entity/chunk
   nodes and enforce it in Cypher queries.
3. **Isolated-mode core** — per-org connection routing driven by
   `TenancyMode.ISOLATED`.
4. **Shared-mode ACL** — wire `ACLEnforcer` into the API dependency
   chain for `TenancyMode.SHARED`.
5. **ACL enforcement** — row-level security policies and graph-side
   namespace filtering.
6. **Rust acceleration** — move hot-path namespace filtering into
   `khora-accel`.

### Dead-code inventory

- `TenancyMode` enum (`core/models/tenancy.py`) is defined but never
  checked at runtime — all orgs use implicit shared mode.
- `ACLEnforcer` and `ACLContext` (`acl/`) are importable but the API
  dependency in `api/deps.py` is disabled.
- `TemporalEdgeStorage` and `TimeHierarchyBuilder` (`engines/skeleton/`)
  exist as modules but are never called by any engine's ingest or recall
  paths. The `occurred_at` column on chunks works through the pgvector
  backend directly.

### Stale documentation fixes

Added status notices to five documentation files flagging features that
are designed but not yet wired:

- `docs/architecture/multi-tenancy.md` — TenancyMode and ACL sections.
- `docs/engines/temporal-model.md` — bi-temporal edge model.
- `docs/engines/skeleton-engine.md` — architecture diagram components.
- `README.md` — multi-tenancy feature bullet.
- `docs/architecture/overview.md` — ACL enforcer mention.

### Housekeeping

- Bumped version from 0.2.2 to 0.2.3.

---

## [0.2.2] — VectorCypher Optimization

### Why: making hybrid retrieval competitive on benchmarks

VectorCypher launched in 0.2.0 with sensible defaults, but benchmark runs
against GraphRAG-Bench showed that retrieval quality dropped on complex
multi-hop queries and that the configuration wasn't surfaced cleanly
through the public API. This release is the result of a benchmarking-
driven optimization cycle: tune retrieval, wire the knobs, add the
indexes to support it, and clean up the code that was left behind.

### Retrieval quality

**Per-complexity fusion weights.** The original retriever used a single
pair of vector/graph weights (0.6/0.4) for every query. Simple factual
queries don't benefit from graph expansion, while complex multi-hop
queries need more graph signal. The retriever now applies different
weights per complexity level: SIMPLE gets 0.8/0.2 (vector-heavy),
MODERATE keeps the 0.6/0.4 default, and COMPLEX flips to 0.4/0.6
(graph-heavy). These are configurable via `VectorCypherConfig`.

**Adaptive graph traversal depth.** Previously, graph depth was fixed
at 2 regardless of how many entry entities the vector search returned.
When many entities match (≥10), deep traversal explodes the candidate
set without adding signal — so the retriever now drops to depth 1.
Conversely, when very few entities match (≤2), it increases depth to
compensate. The thresholds are configurable via `RetrieverConfig`.

**Score normalization.** The fusion function (`weighted_rrf_normalized`)
now min-max normalizes vector and graph scores to [0, 1] before
computing RRF, producing more balanced fusion when score distributions
differ between the two sources.

**Entity resolution and graph density.** Improved entity similarity
thresholds (`min_entity_similarity=0.3`) and skeleton core ratio
(now 0.70 by default) increase the number of entities that get full
LLM extraction, producing a denser graph for traversal.

### Configuration wiring

**`VectorCypherConfig` dataclass.** All VectorCypher-specific knobs —
routing, skeleton indexing, graph traversal, fusion weights, temporal
settings, and search thresholds — live in a single dataclass that can
be passed through the `MemoryLake` constructor:

```python
from khora import MemoryLake
from khora.engines.vectorcypher import VectorCypherConfig

async with MemoryLake(
    db_url,
    engine="vectorcypher",
    engine_kwargs={"vectorcypher_config": VectorCypherConfig(
        skeleton_core_ratio=0.50,
        fusion_complex_vector_weight=0.3,
        fusion_complex_graph_weight=0.7,
    )},
) as lake:
    ...
```

**`engine_kwargs` passthrough.** The `MemoryLake` constructor now
accepts an `engine_kwargs` dict that is forwarded to the engine
constructor. This is the mechanism for passing `VectorCypherConfig`
(or any future engine-specific config) without changing the public API.

### Search indexes (migration 005)

Three new PostgreSQL indexes improve query-time performance:

- **GIN index** on `khora_chunks.tags` for array-containment queries
- **Composite index** on `(namespace_id, occurred_at)` for temporal
  filtering within a namespace
- **HNSW index** rebuilt with `ef_construction=128` (up from 64) for
  better vector recall at the same latency

### Housekeeping

- Skeleton engine code cleanup: removed 122 lines of dead formatting
  and redundant logic.
- Removed hardcoded fusion weights from the retriever in favor of
  config-driven values.

---

## [0.2.1] — Concurrency & Throughput

### Why: filling the gap Rust opened

Version 0.2.0 moved CPU-bound work (similarity scoring, PageRank, BM25
indexing) off the Python event loop and into native Rust threads. The
immediate effect was that CPU cycles were no longer the bottleneck during
large ingestion runs — network I/O to LLM and embedding providers was.
Concurrency limits that once protected against CPU saturation were now
artificially capping throughput: async tasks sat idle waiting for
semaphore permits while the CPU and network had headroom to spare.
Doubling the defaults across every concurrency-controlling parameter
lets Khora fill that idle time, keeping both the network pipe and the
Rust worker pool saturated.

### Concurrency changes by layer

**Configuration defaults.** The global LLM concurrency ceiling
(`max_concurrent_llm_calls`) moved from 10 to 20, and the embedding
concurrency limit from 25 to 50. These two knobs govern all downstream
semaphores, so raising them was the prerequisite for everything else.

**Extractors and embedders.** The LLM extractor's own semaphore doubled
from 5 to 10 concurrent calls. On the embedding side, the LiteLLM
embedder now batches 200 texts per request (up from 100) and runs 20
concurrent embedding calls (up from 10), reducing round-trip overhead
on high-throughput workloads.

**Ingestion pipeline.** The ingestion flow — Khora's primary data path —
doubled three independent limits: concurrent extractions (10 to 20),
embedding batch size (50 to 100), and concurrent document processing
(5 to 10). Together these allow the pipeline to keep more documents
in flight simultaneously.

**Engine-level parallelism.** Every engine's `max_concurrent` semaphore
doubled: GraphRAG from 5 to 10, Skeleton from 10 to 20, VectorCypher
from 10 to 20. The `remember_batch` entry points on MemoryLake and the
base engine protocol matched at 10 (up from 5). Entity expansion
semaphores in the expansion flow doubled from 20 to 40.

**Genesis (bulk loader).** Genesis configuration files for all three
engine profiles doubled their LLM/embedding concurrency (100 to 200),
document concurrency (50 to 100), and chunk concurrency (100 to 200).
The CLI default batch size moved from 10 to 20.

### Housekeeping

- Removed REPOMIX tooling: `REPOMIX.md`, `repomix.config.json`,
  `scripts/update_repomix.py`, and the `update-repomix` pre-commit hook
  (along with REPOMIX exclusions in other hooks).
- Deleted completed planning docs (`OPTIMIZATION_PLAN.md`,
  `RUST_ACCELERATION_PLAN.md`).
- Excluded `docs/REFERENCES.md` from version control (`.gitignore`).
- Bumped version from 0.2.0 to 0.2.1.

---

## [0.2.0] — Rust Acceleration Layer

### The problem

Profiling large ingestion runs showed that CPU-bound operations —
cosine similarity over dense embedding matrices, edit-distance
computations during entity resolution, PageRank convergence over chunk
graphs, and BM25 scoring — dominated wall-clock time once documents
were chunked and LLM calls returned. Python's GIL serialized these
hot loops, and even NumPy could not parallelize the non-BLAS workloads
(string comparisons, graph iteration, inverted-index lookups).

### The approach

Khora 0.2.0 introduces `khora-accel`, a Rust extension built with
PyO3 and maturin. The design philosophy is **zero mandatory
dependencies**: a three-tier fallback (`_accel.py`) checks for the
Rust extension first, then NumPy/RapidFuzz, then pure Python. Every
accelerated function is a drop-in replacement — the Python signature
and return type are identical across all three tiers. Set the
`KHORA_ACCEL_BACKEND` environment variable to `"rust"`, `"numpy"`, or
`"python"` to pin a specific tier; leave it unset for automatic
detection of the fastest available backend.

### Accelerated operations

**Vector similarity.** Cosine similarity (single-pair, one-to-many
batch, and all-pairs above threshold) is implemented with a fused
dot-product-and-norm single pass. Batch operations accept NumPy arrays
via zero-copy `PyReadonlyArray` bindings, copy once into owned Rust
vectors, then release the GIL and fan out across cores with rayon
parallel iterators. For a 10K-candidate batch, this eliminates both
the GIL bottleneck and the Python loop overhead.

**String similarity.** Levenshtein distance and sequence-match ratio use
the `strsim` crate, which implements single-row Wagner-Fischer DP
natively. Batch variants (`batch_levenshtein`, `batch_sequence_match`)
release the GIL and score candidates in parallel via rayon. This
matters for entity resolution, where every new entity must be compared
against hundreds or thousands of existing names.

**BM25 search.** `RustBM25Index` is a full inverted-index implementation
with tokenization, stopword filtering, and suffix-based stemming built
into the Rust layer. The inverted index narrows candidates before
scoring, and the entire scoring phase runs with the GIL released.
Unlike the pure-Python version (which re-tokenized queries on each
call), the Rust implementation tokenizes each query once and pre-computes
IDF scores across the candidate set.

**Graph algorithms.** PageRank and chunk-edge construction power Skeleton
Construction's core indexing step, which identifies the ~10% highest-
value chunks for targeted LLM extraction. Both functions release the
GIL and run pure Rust graph iteration — adjacency-list storage,
iterative convergence with early termination, and O(k^2) bidirectional
edge generation from keyword co-occurrence weighted by IDF.

**Entity resolution.** `resolve_entities_batch` implements the same
three-stage cascade as the Python original (exact name match, alias
match, fuzzy Levenshtein match) but pre-lowercases all existing names
and aliases once, then processes the full batch in parallel with rayon.
For workloads with hundreds of new entities against thousands of
existing ones, this turns an O(n*m) serial Python loop into a
rayon-parallelized Rust loop with no GIL contention.

**Text processing.** Keyword extraction (`extract_keywords`,
`extract_keywords_batch`) uses a compiled `LazyLock<Regex>` and a
`hashbrown::HashSet` stopword table. The batch variant parallelizes
across documents with rayon, which is particularly effective during
bulk ingestion when thousands of chunks need keyword tagging
simultaneously.

**Score fusion.** Reciprocal Rank Fusion (basic and weighted variants)
and min-max score normalization use `hashbrown::HashMap` for
accumulation and `OrderedFloat` for deterministic sorting. These are
lightweight operations, so the Rust version's advantage is mainly in
eliminating Python dict/sort overhead on large ranked lists.

### Integration

The `_accel.py` facade exposes 18 public functions consumed by:
- `engines/skeleton/skeleton.py` — PageRank, chunk edges, keywords, BM25
- `engines/vectorcypher/fusion.py` — RRF, weighted RRF, score normalization
- `query/engine.py` — cosine similarity, BM25 search
- `extraction/entity_resolution.py` — batch entity resolution
- `storage/` and `pipelines/` — embedding similarity, string matching

The active backend is logged at import time for observability.

### Other changes

- Improved upsert result mismatch diagnostics.
- Downgraded extraction log to debug level.

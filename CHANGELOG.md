# Changelog

All notable changes to Khora are documented here.

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

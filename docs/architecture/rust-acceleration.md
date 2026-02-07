# Rust Acceleration Layer

## Why Rust?

Profiling showed that CPU-bound operations dominated ingestion time in
large-scale workloads.  The hot spots were:

| Operation | Problem |
|-----------|---------|
| Entity resolution (Levenshtein) | O(n * m) string comparisons |
| Pairwise cosine similarity | O(n²) vector math |
| PageRank iteration | Tight numerical loop over adjacency lists |
| BM25 scoring | Per-token IDF/TF across thousands of documents |
| Keyword extraction | Regex + stopword filtering over bulk text |

Python's GIL prevents true parallelism for these workloads, and even
NumPy cannot help for string-heavy operations.

**PyO3** provides the bridge:

- **Zero-copy NumPy access** — `PyReadonlyArray1` / `PyReadonlyArray2`
  borrow the underlying numpy buffer without copying data across the
  FFI boundary.
- **GIL release** — `py.allow_threads(|| { ... })` frees the GIL so
  Python async tasks and other threads continue while Rust computes.
- **Rayon parallelism** — `.into_par_iter()` and `.par_iter()` provide
  work-stealing thread-pool parallelism across all available CPU cores.

## Architecture

### 3-Tier Fallback Design

Every accelerated operation has three implementation tiers.  The fastest
available tier is selected automatically at import time.

```
┌─────────────────────────────────────────────────────┐
│  Tier 0: Rust (khora-accel via PyO3)  — fastest     │
│  ● rayon parallelism, GIL release, zero-copy numpy  │
├─────────────────────────────────────────────────────┤
│  Tier 1: NumPy / RapidFuzz  — good, widely avail.   │
│  ● vectorized numpy ops, C-backed rapidfuzz         │
├─────────────────────────────────────────────────────┤
│  Tier 2: Pure Python  — always works, slowest       │
│  ● stdlib only: math, difflib, re                   │
└─────────────────────────────────────────────────────┘
```

All three tiers are centralised in a single file:

```
src/khora/_accel.py          # Python facade — no scattered imports
```

Callers import from `khora._accel` and get the fastest available backend
transparently:

```python
from khora._accel import cosine_similarity, levenshtein_similarity
```

### Runtime Backend Selection

The `KHORA_ACCEL_BACKEND` environment variable overrides auto-detection:

| Value     | Behaviour |
|-----------|-----------|
| *(unset)* | Auto-detect fastest available (default) |
| `"rust"`  | Use Rust if available, fall through otherwise |
| `"numpy"` | Skip Rust, use NumPy/RapidFuzz |
| `"python"`| Force pure Python (useful for debugging/testing) |

Backend availability is logged at import time via the `_HAS_RUST`,
`_HAS_NUMPY`, and `_HAS_RAPIDFUZZ` flags in `_accel.py`.

## Module Reference (8 Modules, 18 Exported Functions)

### `cosine.rs` — Vector Similarity (3 functions)

Provides single-pair, batch (1-to-N), and all-pairs cosine similarity.

| Function | Signature | Description |
|----------|-----------|-------------|
| `cosine_similarity` | `(vec1: Vec<f32>, vec2: Vec<f32>) -> f32` | Single-pair cosine similarity via fused dot+norm single pass. Accumulates in f64 for precision, casts result to f32. Returns 0.0 on dimension mismatch or zero norms. |
| `batch_cosine_similarity` | `(py, query: PyReadonlyArray1<f32>, candidates: PyReadonlyArray2<f32>, threshold: f32) -> Vec<(usize, f32)>` | 1-to-N cosine: one query vector against a matrix of candidates. Returns `(index, similarity)` pairs above threshold, sorted descending. |
| `pairwise_cosine_above_threshold` | `(py, embeddings: PyReadonlyArray2<f32>, threshold: f32) -> Vec<(usize, usize, f32)>` | All-pairs cosine similarity. Returns `(i, j, similarity)` triples where `i < j` and `similarity >= threshold`. |

**Rust techniques:**
- **NumPy zero-copy** — `PyReadonlyArray1` / `PyReadonlyArray2` borrow numpy buffers directly; owned copies are made only to release the GIL.
- **GIL release** — `py.allow_threads(|| { ... })` for batch and pairwise ops.
- **Rayon parallel** — `(0..n).into_par_iter()` distributes row-level work across the thread pool for batch operations; `flat_map` parallelises the outer loop for pairwise.
- **Pre-computed norms** — Query norm and per-row norms computed once, avoiding redundant sqrt calls.

**Python consumers:**
- `khora.extraction.expansion.entity_index` — `batch_cosine_similarity` for entity embedding similarity
- `khora.extraction.expansion.cross_tool_unifier` — `cosine_similarity` for entity deduplication
- `khora._accel.pairwise_cosine_above_threshold` — used by entity resolution pipelines

---

### `string_sim.rs` — String Similarity (4 functions)

Levenshtein and sequence-match similarity with batch variants.

| Function | Signature | Description |
|----------|-----------|-------------|
| `levenshtein_similarity` | `(s1: &str, s2: &str) -> f64` | Normalised Levenshtein similarity (1.0 = identical). Case-insensitive. Short-circuits on equal or empty strings. Uses `strsim::normalized_levenshtein`. |
| `sequence_match_ratio` | `(s1: &str, s2: &str) -> f64` | Sequence match ratio using Jaro-Winkler as an approximation of Python's `difflib.SequenceMatcher.ratio()`. Uses `strsim::jaro_winkler`. |
| `batch_levenshtein` | `(py, query: String, candidates: Vec<String>, threshold: f64) -> Vec<(usize, f64)>` | One query against N candidates using Levenshtein. Returns `(index, similarity)` pairs above threshold, sorted descending. |
| `batch_sequence_match` | `(py, query: String, candidates: Vec<String>, threshold: f64) -> Vec<(usize, f64)>` | One query against N candidates using Jaro-Winkler. Returns `(index, similarity)` pairs above threshold, sorted descending. |

**Rust techniques:**
- **strsim crate** — Provides optimised `normalized_levenshtein` and `jaro_winkler` implementations in pure Rust.
- **Rayon batch** — `candidates.par_iter().enumerate().filter_map(...)` parallelises scoring across all candidates.
- **GIL release** — `py.allow_threads(|| { ... })` for both batch functions.
- **Early exit** — Short-circuit returns for equal strings (→ 1.0) and empty strings (→ 0.0).

**Python consumers:**
- `khora.extraction.entity_resolution` — `levenshtein_similarity`, `sequence_match_ratio` for entity matching
- `khora.extraction.expansion.entity_index` — `levenshtein_similarity` for fuzzy name matching
- `khora.extraction.expansion.cross_tool_unifier` — `levenshtein_similarity` for cross-tool entity dedup
- `khora.query.linking` — `sequence_match_ratio` for query-entity linking

---

### `bm25.rs` — BM25 Full-Text Index (1 class, 5 methods)

A complete BM25 ranking index as a `#[pyclass]`, mirroring the Python
`BM25Index` in `khora.query.keyword`.

**Class:** `RustBM25Index`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(k1=1.5, b=0.75, use_stemming=true, remove_stopwords=true)` | Create a new index with configurable BM25 parameters and tokenisation options. |
| `add_document` | `(doc_id: String, text: &str)` | Add a single document to the index. Tokenises, builds term frequencies, updates inverted index and global stats. |
| `add_documents` | `(documents: Vec<(String, String)>)` | Batch add multiple documents. |
| `score` | `(query: &str, doc_id: &str) -> f32` | BM25 score for a single query-document pair. |
| `search` | `(py, query: &str, limit=10, min_score=0.0) -> Vec<(String, f32)>` | Search the index. Returns `(doc_id, score)` pairs sorted descending. Releases GIL during scoring phase. |

**Internal architecture:**
- **Inverted index** — `HashMap<u32, Vec<u32>>` maps token indices to posting lists of document indices.
- **Token interning** — Bidirectional `token_to_idx` / index lookup avoids repeated string comparisons during scoring.
- **Suffix stemming** — `basic_stem()` strips common English suffixes (`-ing`, `-ed`, `-tion`, `-ness`, `-ment`, `-able`, `-ible`, `-ful`, `-less`, `-ly`, `-er`, `-est`, `-es`, `-s`) when `use_stemming=true`, requiring the stem to be at least 3 characters.
- **Stopword removal** — 90+ English stopwords compiled into a `hashbrown::HashSet` via `LazyLock` for zero-allocation lookups.
- **IDF formula** — `ln((N - df + 0.5) / (df + 0.5) + 1.0)` (standard BM25 IDF).
- **GIL release** — The `search()` method releases the GIL during the candidate scoring loop via `py.allow_threads()`.
- **Candidate pruning** — Only documents containing at least one query term (via inverted index lookup) are scored, avoiding full-corpus scans.

**Rust techniques:**
- **hashbrown** — `HashMap` and `HashSet` from hashbrown for faster hashing than std.
- **LazyLock** — Static regex and stopword set initialised once, shared across all calls.
- **Token indexing** — Strings are interned to `u32` indices for cache-friendly scoring.

**Python consumers:**
- Exported as `RustBM25Index` via `_accel.py`. Not yet wired into the query module (the Python-side `BM25Index` in `khora.query.keyword` remains the active implementation). Available for opt-in use.

---

### `pagerank.rs` — Graph PageRank (2 functions)

Weighted PageRank for skeleton indexing, where ~10% of chunks are
identified as "core" for LLM extraction.

| Function | Signature | Description |
|----------|-----------|-------------|
| `pagerank` | `(py, n: usize, edges: Vec<(usize, usize, f64)>, damping: f64, max_iter: usize, tol: f64) -> Vec<f64>` | Compute PageRank on a weighted directed graph. Uniform init (`1/n`), weighted contributions (`score[src] * weight / out_degree[src]`), converges when total absolute diff < `tol`. |
| `build_chunk_edges` | `(py, n_chunks: usize, keyword_chunk_ids: Vec<Vec<usize>>, idf_scores: Vec<f64>) -> Vec<(usize, usize, f64)>` | Build chunk-to-chunk co-occurrence graph. For each keyword, creates bidirectional edges among all chunks sharing that keyword, weighted by IDF score. |

**Rust techniques:**
- **GIL release** — Both functions run their entire computation inside `py.allow_threads()`.
- **Adjacency list** — `Vec<Vec<(usize, f64)>>` for incoming edges, `Vec<f64>` for out-degree — cache-friendly iteration.
- **Convergence check** — Absolute diff sum checked each iteration for early termination.

**Python consumers:**
- `khora._accel.pagerank` — called by the skeleton engine's `_calculate_pagerank` (via the `_accel.py` facade)
- `khora._accel.build_chunk_edges` — called by the skeleton engine's `_build_chunk_edges`

---

### `rrf.rs` — Reciprocal Rank Fusion (3 functions)

RRF scoring and score normalisation for result fusion.

| Function | Signature | Description |
|----------|-----------|-------------|
| `reciprocal_rank_fusion` | `(ranked_lists: Vec<Vec<String>>, k: usize = 60) -> Vec<(String, f64)>` | Basic RRF over string ID lists. Score = `1 / (k + rank)` where rank is 1-indexed. Returns `(id, score)` sorted descending. |
| `weighted_rrf` | `(ranked_lists: Vec<(f64, Vec<String>)>, k: usize = 60) -> Vec<(String, f64)>` | Weighted RRF. Each list carries a weight. Score = `weight / (k + rank)`. Returns `(id, score)` sorted descending. |
| `normalize_scores` | `(scores: Vec<f64>) -> Vec<f64>` | Min-max normalise to `[0, 1]`. Returns all `1.0` when all scores are identical. |

**Rust techniques:**
- **hashbrown::HashMap** — Fast hash accumulation of scores across ranked lists.
- **OrderedFloat** — `ordered_float::OrderedFloat` wraps `f64` for total ordering, enabling safe `sort_by` without `unwrap_or` on `partial_cmp`.
- **No GIL release** — These are fast enough that GIL overhead would dominate; runs with GIL held.

**Python consumers:**
- `khora._accel.reciprocal_rank_fusion` — low-level string-ID RRF (the higher-level `khora.engines.vectorcypher.fusion` wraps this with `FusedResult` metadata tracking)
- `khora._accel.weighted_rrf` — used by VectorCypher retriever for weighted fusion
- `khora._accel.normalize_scores` — general-purpose score normalisation

---

### `entity_resolution.rs` — Entity Resolution (1 function)

Batch entity matching with a 3-stage cascade.

| Function | Signature | Description |
|----------|-----------|-------------|
| `resolve_entities_batch` | `(py, new_names: Vec<String>, existing_names: Vec<String>, existing_aliases: Vec<Vec<String>>, threshold: f64) -> Vec<Option<(usize, f64, String)>>` | For each new name, attempts matching in order: (1) exact case-insensitive name match, (2) alias match, (3) fuzzy Levenshtein above threshold. Returns parallel vec of `Some((index, score, match_type))` or `None`. |

**3-stage matching pipeline:**
1. **Exact match** — Case-insensitive comparison against existing entity names → score `1.0`, type `"exact"`
2. **Alias match** — Case-insensitive comparison against each entity's alias list → score `1.0`, type `"alias"`
3. **Fuzzy match** — `strsim::normalized_levenshtein` against all existing names, best score above threshold → type `"fuzzy"`

**Rust techniques:**
- **Pre-lowercasing** — All existing names and aliases are lowercased once before the hot loop, outside `allow_threads`.
- **Rayon parallel** — `new_names.par_iter().map(...)` parallelises resolution across all new names.
- **GIL release** — `py.allow_threads()` wraps the entire parallel resolution.
- **Early exit** — Each name short-circuits at the first matching stage.

**Python consumers:**
- `khora._accel.resolve_entities_batch` — used by entity resolution pipelines for bulk entity deduplication

---

### `keyword_extract.rs` — Keyword Extraction (2 functions)

Mirrors the `_extract_keywords` method in `SkeletonIndexer`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_keywords` | `(content: &str) -> Vec<String>` | Extract unique keywords from content. Tokenises with `\b[a-zA-Z]{3,}\b`, removes stopwords, deduplicates via `HashSet`. |
| `extract_keywords_batch` | `(py, contents: Vec<String>) -> Vec<Vec<String>>` | Batch extraction using rayon parallelism. Releases the GIL. |

**Rust techniques:**
- **LazyLock statics** — Compiled regex (`KEYWORD_RE`) and stopword set (`SKELETON_STOPWORDS`) are initialised once via `LazyLock` and shared across all invocations.
- **hashbrown::HashSet** — Fast deduplication of keywords with insertion-order preserved via separate `Vec`.
- **Rayon parallel** — `contents.par_iter().map(...)` in batch mode.
- **GIL release** — `py.allow_threads()` for batch extraction.

**Python consumers:**
- `khora._accel.extract_keywords` — used by skeleton indexing for per-chunk keyword extraction
- `khora._accel.extract_keywords_batch` — bulk extraction during batch ingestion

---

### `utils.rs` — Shared Utilities (1 function)

| Function | Signature | Description |
|----------|-----------|-------------|
| `min_max_normalize` | `(values: &[f64]) -> Vec<f64>` | Min-max normalise to `[0, 1]`. Returns empty vec for empty input, all `1.0` when values are identical. |

This is an internal Rust-only utility (not exported to Python via PyO3).
The Python-facing `normalize_scores` in `rrf.rs` provides the same
functionality as a `#[pyfunction]`.

## Installation & Building

### Requirements

- **Rust** >= 1.75 (edition 2021)
- **maturin** — PyO3 build tool
- **Python** >= 3.10 with NumPy

### Build from Source

```bash
# Development build (debug, fast compile)
cd rust/khora-accel && maturin develop

# Release build (optimised, ~5-10x faster than debug)
cd rust/khora-accel && maturin develop --release

# Build a wheel for distribution
cd rust/khora-accel && maturin build --release
```

### Install as Package

```bash
pip install khora-accel
```

### Verify Installation

```python
>>> import khora_accel
>>> khora_accel.cosine_similarity([1.0, 0.0], [0.0, 1.0])
0.0
>>> khora_accel.levenshtein_similarity("hello", "hallo")
0.8
```

## Performance Characteristics

### Speedup Ranges by Category

| Category | Operations | Estimated Speedup | Key Technique |
|----------|-----------|-------------------|---------------|
| Vector math | cosine, batch cosine, pairwise cosine | **5–10x** | NumPy zero-copy, rayon, fused dot+norm |
| String similarity | Levenshtein, Jaro-Winkler, batch variants | **10–40x** | strsim crate, rayon parallel batch |
| Entity resolution | 3-stage batch matching | **10–30x** | Pre-lowercasing, rayon, early exit |
| BM25 search | Index + score + search | **3–8x** | Inverted index, token interning, GIL release |
| PageRank | Weighted iterative PageRank | **5–15x** | GIL release, tight loop, no Python overhead |
| Keyword extraction | Regex tokenise + filter | **3–5x** | Compiled regex via LazyLock, rayon batch |
| RRF fusion | Reciprocal rank fusion, normalisation | **2–5x** | hashbrown, OrderedFloat sorting |

### When It Matters

- **Large-scale ingestion** (>1,000 documents) — Entity resolution and
  pairwise cosine dominate; Rust's rayon parallelism provides near-linear
  scaling across cores.
- **Skeleton indexing** — PageRank and keyword extraction run on every
  ingestion batch; Rust acceleration reduces per-batch overhead.
- **Real-time query** — BM25 search and RRF fusion benefit from lower
  per-query latency at scale (>10,000 indexed documents).
- **Small workloads** (<100 documents) — The Python/NumPy tiers are
  generally sufficient; Rust overhead is negligible but not necessary.

### Benchmark Infrastructure

Benchmarks use the [Criterion](https://github.com/bhavsec/criterion.rs)
micro-benchmarking framework:

```bash
cd rust/khora-accel && cargo bench
```

**Current benchmark harnesses:**

| Bench file | Status | What it measures |
|------------|--------|-----------------|
| `benches/cosine_bench.rs` | **Functional** | Single-pair cosine at dimensions 128, 384, 768, 1536 |
| `benches/bm25_bench.rs` | Placeholder (TODO) | BM25 index + search benchmarks |
| `benches/pagerank_bench.rs` | Placeholder (TODO) | PageRank iteration benchmarks |

## Dependencies

All dependencies are declared in `rust/khora-accel/Cargo.toml`:

| Crate | Version | Purpose |
|-------|---------|---------|
| **pyo3** | 0.23 | Python ↔ Rust FFI, `#[pyfunction]`/`#[pyclass]` macros, `extension-module` feature for building as a Python extension |
| **numpy** | 0.23 | Zero-copy access to NumPy arrays via `PyReadonlyArray1`/`PyReadonlyArray2` — avoids copying embedding matrices across FFI |
| **ndarray** | 0.16 | N-dimensional array type used internally with numpy crate for row/column access (`Array2`, `ArrayView`) |
| **rayon** | 1.10 | Work-stealing thread-pool parallelism: `par_iter()`, `into_par_iter()`, `flat_map()` for batch operations |
| **strsim** | 0.11 | String similarity algorithms: `normalized_levenshtein`, `jaro_winkler` — pure Rust, no C dependencies |
| **regex** | 1.10 | Compiled regular expressions for tokenisation in BM25 and keyword extraction |
| **hashbrown** | 0.15 | High-performance `HashMap`/`HashSet` (Swiss Table algorithm) — faster than std for the access patterns here |
| **ordered-float** | 4.0 | `OrderedFloat<f64>` wrapper providing total ordering for floats, used in RRF result sorting |

**Dev dependencies:**

| Crate | Version | Purpose |
|-------|---------|---------|
| **criterion** | 0.5 | Micro-benchmarking framework with HTML report generation |

**Feature flags:**

| Feature | Default | Purpose |
|---------|---------|---------|
| `parallel` | Yes | Enables rayon-based parallelism (can be disabled for single-threaded environments) |

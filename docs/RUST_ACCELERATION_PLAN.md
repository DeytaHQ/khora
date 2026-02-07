# Rust Acceleration Plan for Khora

## Executive Summary

Khora's hot paths — entity resolution, BM25 search, PageRank, cosine similarity, and RRF fusion — are currently implemented in Python with optional numpy/rapidfuzz acceleration. Profiling identifies **4 P0-critical** and **3 P1-high** priority components where Rust rewrites via PyO3/maturin would deliver 5-40x speedups by eliminating Python object overhead in tight loops, enabling SIMD vectorization, and leveraging rayon parallelism.

The recommended approach ships Rust as a **separate companion package** (`khora-accel`) in the same monorepo, preserving the existing `uv_build` system and providing a 3-tier fallback chain: **Rust → numpy/rapidfuzz → pure Python**. Only `_accel.py` changes; all other modules remain untouched.

---

## 1. Components to Rewrite in Rust

### 1.1 Priority Matrix

| Priority | Component | Complexity | Bottleneck | Est. Speedup | File |
|----------|-----------|------------|------------|-------------|------|
| **P0** | PageRank (`_calculate_pagerank`) | O(I×(V+E)) | CPU | 10-30x | `skeleton.py` |
| **P0** | Chunk edge building (`_build_chunk_edges`) | O(K×C²) | CPU/mem | 5-15x | `skeleton.py` |
| **P0** | BM25 index & search | O(P×Q) | CPU | 5-15x | `keyword.py` |
| **P0** | MMR diversity selection | O(k²×n×d) | CPU/mem | 5-15x | `engine.py` |
| **P1** | Batch cosine similarity | O(n×d) | CPU | 2-5x | `_accel.py` |
| **P1** | Entity resolution (fuzzy loop) | O(n×k) | CPU | 3-8x | `entity_resolution.py` |
| **P1** | RRF fusion (all variants) | O(L×n) | CPU | 2-4x | `fusion.py` |
| **P2** | Entity index blocking | O(T+k×log(k)) | CPU | 2-4x | `entity_index.py` |
| **P2** | Keyword extraction | O(w) | CPU | 3-5x | `skeleton.py` |
| **P3** | Single cosine similarity | O(d) | CPU | 1.5-2x | `_accel.py` |
| **P3** | Levenshtein/sequence match | O(m×n) | CPU | 1.0-1.2x | `_accel.py` |

### 1.2 Justification for P0 Targets

**PageRank** (`skeleton.py:577-601`): Pure Python dict-based iterative computation over potentially millions of edges. 100 iterations × (V + E) per iteration with Python dict lookups per vertex. Dense array iteration in Rust is 10-30x faster than Python dict iteration. This is the canonical "rewrite in Rust" target.

**Chunk Edge Building** (`skeleton.py:540-548`): O(K×C²) where a common keyword in 500 chunks creates 125K edges for ONE keyword. The inner loop creates millions of Python tuples. Rust eliminates per-element allocation overhead with compact edge arrays.

**BM25 Index & Search** (`keyword.py:162-323`): Pure Python with no C extensions. The `score()` method re-tokenizes the query for every candidate document. Pre-computing query tokens once and batch-scoring in Rust with inverted index lookups delivers 5-15x speedup.

**MMR Diversity Selection** (`engine.py:2167-2202`): Creates `np.array()` inside a triple-nested loop — approximately 5,000 numpy array allocations per query (each allocating 768-1536 floats, ~30-50MB temporary allocations). Precomputing the embedding matrix once and computing all similarities in native code eliminates this entirely.

### 1.3 Not Recommended for Rust

- **Reranking** (`reranking.py`): Model inference (CrossEncoder/LLM) dominates; score normalization is <1% of cost
- **Search metrics** (`metrics.py`): Trivial counters and timers, O(1) operations
- **Async I/O paths**: Storage access, embedding API calls — I/O-bound, not CPU-bound

---

## 2. Data Size Sensitivity

| Scenario | Chunks | Entities | Keywords | PageRank Edges | BM25 Docs |
|----------|--------|----------|----------|----------------|-----------|
| Small (dev) | 100 | 50 | 500 | 5K | 100 |
| Medium (prod) | 5,000 | 2,000 | 10K | 500K | 5,000 |
| Large (enterprise) | 50,000 | 20,000 | 100K | 5M+ | 50,000 |

At **medium scale**, P0 targets become noticeable (>100ms for PageRank, >50ms for BM25). At **large scale**, they become critical (>5s for PageRank, >1s for BM25, >500ms for MMR).

---

## 3. PyO3/Maturin Integration Design

### 3.1 Separate Package Strategy

Ship Rust as a separate companion package in the same monorepo:

```
pip install khora           # Pure Python, always works
pip install khora[rust]     # Adds khora-accel Rust extension
pip install khora-accel     # Standalone Rust wheel
```

**Rationale**: The current build backend (`uv_build`) does not support native extensions. Replacing it with maturin would require all developers and CI to have the Rust toolchain installed. A separate package preserves the existing workflow entirely.

### 3.1.1 Downstream Consumer Deployment (Git Dependencies)

Downstream projects like **genesis** pull khora directly from GitHub:

```toml
# genesis/pyproject.toml (current)
dependencies = [
    "khora[accel] @ git+https://github.com/DeytaHQ/khora.git@main",
]
reinstall-package = ["khora"]
```

Since `khora-accel` lives in the same monorepo, pip/uv's `#subdirectory=` syntax allows pulling both packages from the same git commit, avoiding version skew:

```toml
# genesis/pyproject.toml (with Rust acceleration)
dependencies = [
    "khora[accel] @ git+https://github.com/DeytaHQ/khora.git@main",
    "khora-accel @ git+https://github.com/DeytaHQ/khora.git@main#subdirectory=rust/khora-accel",
]
reinstall-package = ["khora", "khora-accel"]
```

**How it works:**
- `khora` installs as before — pure Python via `uv_build`, no Rust toolchain needed
- `khora-accel` is built from `rust/khora-accel/` in the same repo at the same commit — maturin compiles the Rust extension during `uv sync` / `pip install`
- Both packages always come from the **same git commit**, eliminating version skew
- If Rust toolchain is not available, simply omit the `khora-accel` line — `_accel.py` falls back to numpy/rapidfuzz/pure Python

**Requirements for the build environment** (genesis CI / Docker):
- Rust toolchain (rustup) must be installed in the builder stage
- maturin is pulled automatically as a build dependency (declared in `rust/khora-accel/pyproject.toml`)
- No changes needed to khora's own CI or build system

**Dockerfile example for genesis:**
```dockerfile
# Builder stage
FROM python:3.13-slim AS builder
RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync  # Builds khora-accel from source via maturin

# Runtime stage — no Rust toolchain needed
FROM python:3.13-slim
COPY --from=builder /app/.venv /app/.venv
```

**Opting out of Rust acceleration:**
Remove the `khora-accel` dependency line from genesis. No other changes needed — `_accel.py` auto-detects the absence and falls back.

### 3.2 Project Layout

```
khora/                          # Git repository root
├── pyproject.toml              # Python package (uv_build, unchanged except new [rust] extra)
├── src/khora/
│   ├── _accel.py               # Dispatch: Rust → numpy/rapidfuzz → pure Python
│   └── ...                     # Everything else unchanged
├── rust/
│   ├── Cargo.toml              # Workspace manifest
│   └── khora-accel/
│       ├── Cargo.toml          # PyO3 + maturin crate config
│       ├── pyproject.toml      # Maturin build system for this crate
│       ├── benches/
│       │   ├── cosine_bench.rs
│       │   ├── bm25_bench.rs
│       │   ├── pagerank_bench.rs
│       │   └── entity_resolution_bench.rs
│       └── src/
│           ├── lib.rs          # PyO3 module definition
│           ├── cosine.rs       # Cosine similarity (single + batch, SIMD)
│           ├── bm25.rs         # BM25 index + scoring + tokenization
│           ├── pagerank.rs     # PageRank on sparse graphs
│           ├── rrf.rs          # Reciprocal Rank Fusion variants
│           ├── string_sim.rs   # Levenshtein + sequence matching
│           ├── entity_resolution.rs  # Batch entity resolution
│           ├── keyword_extract.rs    # Keyword extraction with stopwords
│           └── utils.rs        # Shared utilities
├── benchmarks/
│   └── bench_accel.py          # Comparative Python benchmarks
└── .github/workflows/
    ├── ci.yml                  # Existing CI (add test matrix for Rust)
    └── rust-wheels.yml         # New: build + publish Rust wheels
```

### 3.3 Cargo.toml

```toml
[package]
name = "khora-accel"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[lib]
name = "khora_accel"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.10"
strsim = "0.11"
regex = "1.10"
hashbrown = "0.15"
ordered-float = "4.0"

[features]
default = ["parallel"]
parallel = []  # Enable rayon parallelism

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "cosine_bench"
harness = false

[[bench]]
name = "bm25_bench"
harness = false

[[bench]]
name = "pagerank_bench"
harness = false
```

### 3.4 PyO3 Module Definition

```rust
// lib.rs
use pyo3::prelude::*;

mod cosine;
mod bm25;
mod pagerank;
mod rrf;
mod string_sim;
mod entity_resolution;
mod keyword_extract;

#[pymodule]
fn khora_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Cosine similarity
    m.add_function(wrap_pyfunction!(cosine::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(cosine::batch_cosine_similarity, m)?)?;

    // String similarity
    m.add_function(wrap_pyfunction!(string_sim::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::sequence_match_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::batch_sequence_match, m)?)?;

    // BM25
    m.add_class::<bm25::BM25Index>()?;

    // PageRank
    m.add_function(wrap_pyfunction!(pagerank::pagerank, m)?)?;

    // RRF Fusion
    m.add_function(wrap_pyfunction!(rrf::weighted_rrf, m)?)?;
    m.add_function(wrap_pyfunction!(rrf::reciprocal_rank_fusion, m)?)?;

    // Entity resolution
    m.add_function(wrap_pyfunction!(entity_resolution::resolve_entities_batch, m)?)?;

    // Keyword extraction
    m.add_function(wrap_pyfunction!(keyword_extract::extract_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(keyword_extract::extract_keywords_batch, m)?)?;

    Ok(())
}
```

---

## 4. Build System Changes

### 4.1 Main `pyproject.toml` (Minimal Change)

Only one addition needed:

```toml
[project.optional-dependencies]
# ... existing extras unchanged ...
rust = ["khora-accel>=0.1.0"]
```

The build backend stays as `uv_build`. No other changes.

### 4.2 Rust Crate `pyproject.toml` (`rust/khora-accel/pyproject.toml`)

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "khora-accel"
version = "0.1.0"
requires-python = ">=3.13"
description = "Rust-accelerated operations for Khora memory lake"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### 4.3 Makefile Additions

```makefile
# Rust acceleration
rust-build:
	cd rust/khora-accel && maturin develop --release

rust-test:
	cd rust/khora-accel && cargo test

rust-bench:
	cd rust/khora-accel && cargo bench
```

---

## 5. Fallback Strategy

### 5.1 Three-Tier Dispatch in `_accel.py`

```python
import os

# ---------------------------------------------------------------------------
# Runtime backend override via KHORA_ACCEL_BACKEND env var
# Values: "rust" | "numpy" | "python" | unset (auto-detect fastest)
# ---------------------------------------------------------------------------
_FORCE_BACKEND = os.environ.get("KHORA_ACCEL_BACKEND")

# Tier 0: Rust native acceleration (fastest)
try:
    from khora_accel import (
        cosine_similarity as _rust_cosine,
        batch_cosine_similarity as _rust_batch_cosine,
        levenshtein_similarity as _rust_levenshtein,
        sequence_match_ratio as _rust_sequence_match,
        batch_levenshtein as _rust_batch_levenshtein,
        pagerank as _rust_pagerank,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

# Tier 1: NumPy / RapidFuzz (existing, medium)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from rapidfuzz.distance import Levenshtein as _rf_lev
    from rapidfuzz.fuzz import ratio as _rf_ratio
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# Tier 2: Pure Python (existing, slowest)

# Apply runtime override — disable higher tiers to force a specific backend
if _FORCE_BACKEND == "python":
    _HAS_RUST = False
    _HAS_NUMPY = False
    _HAS_RAPIDFUZZ = False
elif _FORCE_BACKEND == "numpy":
    _HAS_RUST = False
# "rust" or unset: use auto-detected fastest path

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if _HAS_RUST:
        return _rust_cosine(vec1, vec2)
    if _HAS_NUMPY:
        # ... existing numpy path ...
    # ... existing pure-Python fallback ...
```

### 5.2 Runtime Backend Control

The `KHORA_ACCEL_BACKEND` environment variable provides runtime control over the acceleration tier:

| Value | Behavior | Use Case |
|-------|----------|----------|
| unset | Auto-detect fastest (Rust → numpy → Python) | Production (default) |
| `rust` | Same as unset — use Rust if available | Explicit opt-in |
| `numpy` | Skip Rust, use numpy/rapidfuzz | Debugging Rust issues, A/B perf testing |
| `python` | Force pure Python — disable all native acceleration | Debugging, correctness testing, CI parity checks |

**Examples:**
```bash
# Force pure Python for debugging
KHORA_ACCEL_BACKEND=python uv run khora serve

# Skip Rust, use numpy (e.g., to isolate a Rust-specific bug)
KHORA_ACCEL_BACKEND=numpy uv run pytest tests/

# Production — just let it auto-detect
uv run khora serve
```

This follows the existing Khora config pattern (env vars with `KHORA_` prefix). The override is applied at import time via module-level flag manipulation — zero overhead at call time.

### 5.3 Key Design Principles

1. **`_accel.py` is the ONLY file that changes** — the 4 consumer modules (`entity_index.py`, `cross_tool_unifier.py`, `entity_resolution.py`, `query/linking.py`) continue importing from `khora._accel` unchanged
2. **Module-level detection**: Feature flags (`_HAS_RUST`, `_HAS_NUMPY`, `_HAS_RAPIDFUZZ`) computed once at import time
3. **Identical signatures**: All tiers share the exact same function signatures and return types
4. **No Python objects cross the boundary**: Only `list[float]`, `str`, `float`, `int`, and `list[tuple[...]]` cross the FFI boundary
5. **UUIDs stay in Python**: Convert to integer indices before calling Rust; map back after

---

## 6. API Boundary Design

### 6.1 Type Mapping

| Python Type | Rust Type | Strategy |
|-------------|-----------|----------|
| `list[float]` | `Vec<f32>` | PyO3 extract (copies) |
| `numpy.ndarray` (1D) | `PyReadonlyArray1<f32>` | Zero-copy via numpy crate |
| `numpy.ndarray` (2D) | `PyReadonlyArray2<f32>` | Zero-copy via numpy crate |
| `str` | `&str` | Zero-copy borrow |
| `list[str]` | `Vec<String>` | PyO3 extract (copies) |
| `float` | `f64` | PyO3 automatic |
| `int` | `usize` | PyO3 automatic |
| `list[tuple[int, float]]` | `Vec<(usize, f64)>` | PyO3 automatic |
| `UUID` | NOT recommended | Convert to `str` or index in Python first |

### 6.2 Tier 1: Direct Drop-in (Extend `_accel.py` API)

These match existing function signatures exactly:

| Function | Rust Input | Rust Output |
|----------|-----------|------------|
| `cosine_similarity(vec1, vec2)` | `Vec<f32>, Vec<f32>` | `f32` |
| `batch_cosine_similarity(query, candidates, threshold)` | `PyReadonlyArray1<f32>, PyReadonlyArray2<f32>, f32` | `Vec<(usize, f32)>` |
| `levenshtein_similarity(s1, s2)` | `&str, &str` | `f64` |
| `sequence_match_ratio(s1, s2)` | `&str, &str` | `f64` |

### 6.3 Tier 2: New Batch APIs (Amortize FFI Overhead)

| New Function | Purpose | Replaces |
|-------------|---------|----------|
| `batch_levenshtein(query, candidates, threshold)` | Entity resolution fuzzy matching | N individual FFI crossings |
| `batch_sequence_match(query, candidates, threshold)` | Entity linking fuzzy matching | N individual FFI crossings |
| `pairwise_cosine_above_threshold(embeddings, threshold)` | Cross-tool unifier matching | O(n²) Python loop |
| `pairwise_levenshtein_above_threshold(names, threshold)` | Cross-tool unifier fuzzy matching | O(n²) Python loop |

### 6.4 Tier 3: Algorithm Acceleration (New Implementations)

| Function | Replaces | Design |
|----------|---------|--------|
| `pagerank(n, edges, damping, max_iter, tol)` | Dict-based iterative PageRank | Dense Vec<f64> scores, CSR-like adjacency |
| `BM25Index` (PyClass) | Python BM25Index dataclass | Rust-native inverted index, pre-computed IDF |
| `weighted_rrf(ranked_lists, k)` | Dict-based RRF accumulation | HashMap<String, f64> with pre-allocation |
| `resolve_entities_batch(new_names, existing_names, aliases, threshold)` | Python loops over entities | Rayon-parallel multi-strategy matching |

### 6.5 GIL Management

| Function | Release GIL? | Reason |
|----------|-------------|--------|
| `batch_cosine_similarity` | Yes | Heavy computation, rayon parallelism |
| `pagerank` | Yes | Iterative, long-running |
| `batch_levenshtein` | Yes | Many string comparisons |
| `resolve_entities_batch` | Yes | Combines multiple operations |
| `BM25Index.search` | Yes | Index traversal + scoring |
| `cosine_similarity` (single) | No | Overhead of releasing > computation |
| `levenshtein_similarity` (single) | No | Short strings, fast |
| `extract_keywords` (single) | No | Single document, fast |

---

## 7. Python-Specific Complications

### 7.1 Async Functions (LOW risk)

All `_accel.py` functions are synchronous. The async context exists at the caller level (`EntityResolver.resolve()` is async for storage access). Rust functions stay synchronous — clean separation where Python async handles I/O and Rust handles CPU computation.

### 7.2 Lambda/Callable in RRF (MEDIUM risk)

`reciprocal_rank_fusion()` in `query/fusion.py` accepts `id_extractor: callable`. **Solution**: Pre-extract IDs in Python, pass index arrays to Rust, map back after. Alternatively, keep the generic RRF in Python since the arithmetic is simple — accelerate only the VectorCypher-specific RRF which has typed inputs.

### 7.3 Python Objects (MEDIUM risk)

`Entity`, `Chunk`, `FusedResult` should NOT cross into Rust. **Pattern**: Extract primitive fields → process in Rust → return indices/scores → reconstruct Python objects.

### 7.4 No Generators/Context Managers (NONE)

No generators or context managers in any hot-path function. All are simple functions with list/scalar returns.

---

## 8. Benchmarking Approach

### 8.1 Metrics

| Metric | Tool | Significance |
|--------|------|-------------|
| Latency (p50, p95, p99) | criterion.rs / pytest-benchmark | User-facing response time |
| Throughput (ops/sec) | criterion.rs | Batch processing capacity |
| Memory usage | tracemalloc / `/proc/self/status` | Resource constraints |
| Allocation count | dhat (Rust), tracemalloc (Python) | GC pressure indicator |

### 8.2 Rust-Side (criterion.rs)

Parametric benchmarks across realistic input sizes:

- **Cosine**: D ∈ {128, 384, 768, 1536, 3072}
- **Batch cosine**: (N, D) ∈ {(100, 1536), (500, 1536), (1000, 1536), (5000, 1536)}
- **BM25 search**: N_docs ∈ {100, 1000, 10000, 100000}
- **PageRank**: N_nodes ∈ {100, 1000, 10000}, iterations until convergence
- **Entity resolution**: N_new × N_existing ∈ {(10, 100), (100, 1000), (100, 10000)}

### 8.3 Python-Side (pytest-benchmark)

Three-way comparison for each function: Pure Python vs numpy/rapidfuzz vs Rust. Same parametric input sizes as criterion benchmarks. Stored as CI artifacts for regression detection.

### 8.4 Expected Speedups (Conservative)

| Operation | Current | Expected (Rust) | Speedup |
|-----------|---------|-----------------|---------|
| `cosine_similarity` (single, D=1536) | ~5μs (NumPy) | ~0.5μs | **10x** |
| `batch_cosine_similarity` (N=1000, D=1536) | ~2ms (NumPy) | ~0.2ms | **10x** |
| `batch_levenshtein` (N=1000) | ~2ms (rapidfuzz loop) | ~0.1ms | **20x** |
| `BM25.search` (1000 docs) | ~5ms (Python) | ~0.2ms | **25x** |
| `PageRank` (1000 nodes, 100 iter) | ~50ms (Python) | ~2ms | **25x** |
| `RRF fusion` (3×200 items) | ~0.5ms (Python) | ~0.05ms | **10x** |
| `resolve_entities_batch` (100×1000) | ~200ms (Python) | ~5ms | **40x** |

---

## 9. Implementation Phases

### Phase 1: Foundation (Core Numerics)

**Goal**: Ship the first `khora-accel` wheel with the highest-impact, simplest functions.

1. Create `rust/khora-accel/` directory structure with Cargo.toml and maturin pyproject.toml
2. Implement `cosine_similarity` and `batch_cosine_similarity` with PyO3 + numpy zero-copy
3. Implement `levenshtein_similarity`, `sequence_match_ratio`, and new `batch_levenshtein`
4. Implement `keyword_extract` (regex + stopword filtering)
5. Update `_accel.py` with Rust-first dispatch (3-tier fallback)
6. Add `rust = ["khora-accel>=0.1.0"]` extra to main `pyproject.toml`
7. Write comprehensive tests (Rust vs Python parity)
8. Create criterion.rs and pytest-benchmark suites

**Deliverable**: `khora-accel` 0.1.0 — drop-in acceleration for `_accel.py` functions.

### Phase 2: Algorithm Acceleration

**Goal**: Accelerate the P0 algorithm targets.

9. Implement Rust `BM25Index` as `#[pyclass]` with add_document, search, batch_add
10. Implement Rust `pagerank()` with dense Vec<f64> scores and CSR-like adjacency
11. Implement Rust `_build_chunk_edges()` equivalent for skeleton engine
12. Implement `resolve_entities_batch()` with rayon parallelism
13. Wire BM25 and PageRank into Python callers via new `_accel.py` functions
14. Extend benchmark suite for new components

**Deliverable**: `khora-accel` 0.2.0 — BM25, PageRank, entity resolution in Rust.

### Phase 3: Fusion & Integration

**Goal**: Complete the acceleration layer and polish.

15. Implement Rust `weighted_rrf` and `reciprocal_rank_fusion`
16. Implement `pairwise_cosine_above_threshold` and `pairwise_levenshtein_above_threshold` for O(n²) loops
17. Wire fusion functions into VectorCypher and query engine
18. Implement MMR diversity selection as a single Rust call (precomputed similarity matrix)
19. Full integration testing with all engines

**Deliverable**: `khora-accel` 0.3.0 — comprehensive acceleration for all hot paths.

### Phase 4: CI/CD & Release

**Goal**: Production-ready wheel distribution.

20. Add `.github/workflows/rust-wheels.yml` using `PyO3/maturin-action@v1`
21. Build wheels for: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
22. Extend test CI matrix: with and without Rust extension
23. Publish `khora-accel` to PyPI
24. Update Makefile with `rust-build`, `rust-test`, `rust-bench` targets
25. Update documentation and CLAUDE.md

**Deliverable**: Published wheels, CI pipeline, documentation.

---

## 10. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Build complexity for contributors** | Medium | Python-only devs never touch `rust/`. Fallback ensures everything works without Rust. Clear CONTRIBUTING.md section. |
| **Version skew between packages** | Low | Git consumers use `#subdirectory=` to build both from same commit. PyPI consumers use pinned versions (`khora-accel>=0.1,<0.2`). Shared test suite validates both paths. |
| **Binary compatibility** | Low-Medium | Maturin uses manylinux containers. macOS universal2 wheels. `_accel.py` fallback if wheel fails to import. |
| **PyO3 / Python ABI compat** | Low | Build per Python minor version. Current target is `>=3.13` only. PyO3 abi3 stable ABI for forward compat. |
| **Float precision differences** | Low | Property-based tests comparing Python and Rust within tolerance. Document any known behavioral differences. |
| **Increased CI time** | Low | Rust builds triggered only on `rust/` changes. Cargo caching via sccache. Separate workflow. |
| **Memory safety** | Negligible | No `unsafe` code needed. PyO3 handles memory management. `PyReadonlyArray` for safe numpy access. |

---

## 11. Key Design Decisions Summary

1. **Separate package** (`khora-accel`), not replacing `uv_build` with maturin
2. **`_accel.py` is the only file that changes** — single dispatch point for all tiers
3. **3-tier fallback**: Rust → numpy/rapidfuzz → pure Python
4. **Runtime control** via `KHORA_ACCEL_BACKEND` env var (force `python`, `numpy`, or `rust`)
5. **Git subdirectory install** for downstream consumers — `#subdirectory=rust/khora-accel` ensures same-commit builds, no version skew
6. **No Python objects cross FFI boundary** — only primitives and arrays
7. **Batch APIs** for amortizing FFI overhead in entity resolution loops
8. **Rayon parallelism** with GIL release for heavy computations
9. **Numpy zero-copy** via PyO3 numpy crate for embedding operations
10. **Feature flags** in Cargo.toml for optional components
11. **Same monorepo** — `rust/` directory alongside `src/`, shared test suite

---

## Appendix A: Current `_accel.py` Consumer Map

```
cosine_similarity
  ├── extraction/expansion/cross_tool_unifier.py :: _find_embedding_matches()
  └── tests/unit/test_entity_index.py

batch_cosine_similarity
  └── extraction/expansion/entity_index.py :: find_embedding_candidates()

levenshtein_similarity
  ├── extraction/entity_resolution.py :: _compute_*_attribute_similarity()
  ├── extraction/expansion/entity_index.py :: find_fuzzy_candidates()
  └── extraction/expansion/cross_tool_unifier.py :: _find_fuzzy_matches()

sequence_match_ratio
  ├── extraction/entity_resolution.py :: EntityResolver.resolve() fuzzy match
  └── query/linking.py :: EntityLinker._fuzzy_name_match()
```

## Appendix B: Rust Crate Ecosystem

| Crate | Purpose | Version |
|-------|---------|---------|
| `pyo3` | Python FFI bindings | 0.22+ |
| `numpy` (pyo3) | Zero-copy numpy array access | 0.22+ |
| `ndarray` | N-dimensional arrays | 0.16+ |
| `rayon` | Data parallelism | 1.10+ |
| `strsim` | String similarity algorithms | 0.11+ |
| `regex` | Fast regex for tokenization | 1.10+ |
| `hashbrown` | Fast HashMap | 0.15+ |
| `ordered-float` | Sortable floats | 4.0+ |
| `criterion` | Benchmarking (dev) | 0.5+ |

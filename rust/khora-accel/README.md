# khora-accel

Rust-accelerated operations for [Khora](https://github.com/DeytaHQ/khora), a knowledge graph + vector + SQL storage library.

This crate provides PyO3 bindings for performance-critical operations:

- MMR (Maximal Marginal Relevance) re-ranking
- Cosine similarity (batched, normalized)
- PageRank
- Entity resolution (fuzzy matching with per-type thresholds)
- Community detection
- Temporal scoring

## Installation

```bash
pip install khora-accel
```

`khora-accel` ships as a source distribution. A Rust toolchain (`rustup` with a stable `cargo` on PATH) is required at install time. Maturin compiles the extension automatically as part of the standard PEP 517 build flow.

## Usage

Used internally by `khora` via `khora._accel`. Direct consumption from user code is not a stable API.

## License

Apache-2.0

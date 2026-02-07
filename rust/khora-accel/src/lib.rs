//! Khora acceleration library — Rust implementations of CPU-intensive operations.
//!
//! Provides PyO3 bindings for cosine similarity, BM25 search, PageRank,
//! RRF fusion, string similarity, entity resolution, and keyword extraction.

use pyo3::prelude::*;

mod bm25;
mod cosine;
mod entity_resolution;
mod keyword_extract;
mod pagerank;
mod rrf;
mod string_sim;
mod utils;

/// khora_accel — Rust-accelerated operations for Khora
#[pymodule]
fn khora_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Cosine similarity
    m.add_function(wrap_pyfunction!(cosine::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(cosine::batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(
        cosine::pairwise_cosine_above_threshold,
        m
    )?)?;

    // String similarity
    m.add_function(wrap_pyfunction!(string_sim::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::sequence_match_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::batch_sequence_match, m)?)?;

    // BM25
    m.add_class::<bm25::RustBM25Index>()?;

    // PageRank
    m.add_function(wrap_pyfunction!(pagerank::pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank::build_chunk_edges, m)?)?;

    // RRF Fusion
    m.add_function(wrap_pyfunction!(rrf::reciprocal_rank_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(rrf::weighted_rrf, m)?)?;
    m.add_function(wrap_pyfunction!(rrf::normalize_scores, m)?)?;

    // Entity resolution
    m.add_function(wrap_pyfunction!(
        entity_resolution::resolve_entities_batch,
        m
    )?)?;

    // Keyword extraction
    m.add_function(wrap_pyfunction!(keyword_extract::extract_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(
        keyword_extract::extract_keywords_batch,
        m
    )?)?;

    Ok(())
}

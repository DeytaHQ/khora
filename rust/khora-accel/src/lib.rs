//! Khora acceleration library — Rust implementations of CPU-intensive operations.
//!
//! Provides PyO3 bindings for cosine similarity, BM25 search, PageRank,
//! RRF fusion, string similarity, entity resolution, and keyword extraction.

use pyo3::prelude::*;

/// Configure the global rayon thread pool.
///
/// Must be called before any parallel work is spawned.
/// `num_threads = 0` uses mode-based defaults:
///   - "query" (default): `num_cpus / 2` — lower latency for concurrent queries
///   - "ingest": `num_cpus * 3 / 4` — higher throughput for batch ingestion
/// Returns `Ok(())` on success. Logs a warning if the pool was already initialised
/// (rayon only allows one global pool per process).
#[pyfunction]
#[pyo3(signature = (num_threads=0, mode="query"))]
fn configure_thread_pool(num_threads: usize, mode: &str) -> PyResult<()> {
    let cpus = num_cpus::get();
    let threads = if num_threads > 0 {
        num_threads
    } else {
        match mode {
            "ingest" => std::cmp::max(1, cpus * 3 / 4),
            _ => std::cmp::max(1, cpus / 2), // "query" default
        }
    };

    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        Ok(()) => {
            eprintln!("[khora-accel] rayon global thread pool configured with {threads} threads");
            Ok(())
        }
        Err(e) => {
            eprintln!(
                "[khora-accel] warning: rayon global pool already initialised, ignoring configure_thread_pool: {e}"
            );
            Ok(())
        }
    }
}

mod bm25;
mod community;
mod cosine;
mod dedup;
mod entity_resolution;
mod keyword_extract;
mod mmr;
mod pagerank;
mod rrf;
mod string_sim;
mod temporal;
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

    // Embedding normalization and dot product
    m.add_function(wrap_pyfunction!(
        cosine::normalize_embeddings_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(cosine::batch_dot_product, m)?)?;

    // String similarity
    m.add_function(wrap_pyfunction!(string_sim::levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::sequence_match_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::batch_sequence_match, m)?)?;
    m.add_function(wrap_pyfunction!(string_sim::normalize_entity_name, m)?)?;
    m.add_function(wrap_pyfunction!(
        string_sim::normalize_entity_names_batch,
        m
    )?)?;

    // BM25
    m.add_class::<bm25::RustBM25Index>()?;

    // PageRank
    m.add_function(wrap_pyfunction!(pagerank::pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank::build_chunk_edges, m)?)?;

    // RRF Fusion
    m.add_function(wrap_pyfunction!(rrf::reciprocal_rank_fusion, m)?)?;
    m.add_function(wrap_pyfunction!(rrf::weighted_rrf, m)?)?;
    m.add_function(wrap_pyfunction!(rrf::normalize_scores, m)?)?;
    m.add_function(wrap_pyfunction!(rrf::weighted_rrf_normalized, m)?)?;
    m.add_function(wrap_pyfunction!(
        rrf::weighted_rrf_normalized_with_provenance,
        m
    )?)?;

    // Entity resolution
    m.add_function(wrap_pyfunction!(
        entity_resolution::resolve_entities_batch,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        entity_resolution::resolve_entities_enhanced,
        m
    )?)?;

    // Keyword extraction
    m.add_function(wrap_pyfunction!(keyword_extract::extract_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(
        keyword_extract::extract_keywords_batch,
        m
    )?)?;

    // Temporal filtering
    m.add_function(wrap_pyfunction!(temporal::batch_temporal_filter, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::batch_recency_scores, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::detect_temporal_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(temporal::detect_temporal_category, m)?)?;

    // MMR diversity selection
    m.add_function(wrap_pyfunction!(mmr::mmr_diversity_select, m)?)?;

    // Community detection
    m.add_function(wrap_pyfunction!(community::detect_communities, m)?)?;

    // Chunk deduplication
    m.add_function(wrap_pyfunction!(dedup::deduplicate_chunks, m)?)?;

    // Thread pool configuration
    m.add_function(wrap_pyfunction!(configure_thread_pool, m)?)?;

    Ok(())
}

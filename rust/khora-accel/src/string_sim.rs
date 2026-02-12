//! String similarity operations with PyO3 bindings.
//!
//! Provides Levenshtein similarity, sequence match ratio, and batch variants
//! with GIL release and rayon parallelism.

use pyo3::prelude::*;
use rayon::prelude::*;
use strsim::{jaro_winkler, normalized_levenshtein};

/// Levenshtein similarity between two strings (case-insensitive).
///
/// Returns a value in [0.0, 1.0] where 1.0 means identical strings.
/// Short-circuits on equal strings (→ 1.0) or empty strings (→ 0.0).
#[pyfunction]
pub fn levenshtein_similarity(s1: &str, s2: &str) -> f64 {
    let a = s1.to_lowercase();
    let b = s2.to_lowercase();

    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    normalized_levenshtein(&a, &b)
}

/// Sequence match ratio between two strings.
///
/// Uses Jaro-Winkler similarity as an approximation of Python's
/// `difflib.SequenceMatcher.ratio()`.
#[pyfunction]
pub fn sequence_match_ratio(s1: &str, s2: &str) -> f64 {
    let a = s1.to_lowercase();
    let b = s2.to_lowercase();

    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    jaro_winkler(&a, &b)
}

/// Batch Levenshtein similarity: one query against N candidates.
///
/// Uses rayon for parallelism and releases the GIL during computation.
/// Returns `(index, similarity)` pairs above `threshold`, sorted descending.
#[pyfunction]
pub fn batch_levenshtein(
    py: Python<'_>,
    query: String,
    candidates: Vec<String>,
    threshold: f64,
) -> Vec<(usize, f64)> {
    py.allow_threads(|| {
        let q = query.to_lowercase();

        let mut results: Vec<(usize, f64)> = candidates
            .par_iter()
            .enumerate()
            .filter_map(|(i, candidate)| {
                let c = candidate.to_lowercase();

                let sim = if q == c {
                    1.0
                } else if q.is_empty() || c.is_empty() {
                    0.0
                } else {
                    normalized_levenshtein(&q, &c)
                };

                if sim >= threshold {
                    Some((i, sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    })
}

/// Normalize an entity name for deduplication.
///
/// - Lowercase
/// - Strip honorifics (Mr., Mrs., Ms., Dr., Prof., etc.)
/// - Collapse multiple whitespace to single space
/// - Strip leading/trailing whitespace and punctuation
#[pyfunction]
pub fn normalize_entity_name(name: &str) -> String {
    let mut result = name.to_lowercase();

    // Strip common honorifics
    let honorifics = [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sir ", "lord ", "lady ",
    ];
    for h in &honorifics {
        if result.starts_with(h) {
            result = result[h.len()..].to_string();
        }
    }

    // Collapse whitespace
    let parts: Vec<&str> = result.split_whitespace().collect();
    result = parts.join(" ");

    // Strip leading/trailing punctuation
    result = result
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_string();

    result
}

/// Batch normalize entity names (rayon parallel).
#[pyfunction]
pub fn normalize_entity_names_batch(py: Python<'_>, names: Vec<String>) -> Vec<String> {
    py.allow_threads(|| {
        names
            .par_iter()
            .map(|n| {
                let mut result = n.to_lowercase();
                let honorifics = [
                    "mr.", "mrs.", "ms.", "dr.", "prof.", "sir ", "lord ", "lady ",
                ];
                for h in &honorifics {
                    if result.starts_with(h) {
                        result = result[h.len()..].to_string();
                    }
                }
                let parts: Vec<&str> = result.split_whitespace().collect();
                result = parts.join(" ");
                result
                    .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
                    .to_string()
            })
            .collect()
    })
}

/// Batch sequence match ratio: one query against N candidates.
///
/// Uses rayon for parallelism and releases the GIL during computation.
/// Returns `(index, similarity)` pairs above `threshold`, sorted descending.
#[pyfunction]
pub fn batch_sequence_match(
    py: Python<'_>,
    query: String,
    candidates: Vec<String>,
    threshold: f64,
) -> Vec<(usize, f64)> {
    py.allow_threads(|| {
        let q = query.to_lowercase();

        let mut results: Vec<(usize, f64)> = candidates
            .par_iter()
            .enumerate()
            .filter_map(|(i, candidate)| {
                let c = candidate.to_lowercase();

                let sim = if q == c {
                    1.0
                } else if q.is_empty() || c.is_empty() {
                    0.0
                } else {
                    jaro_winkler(&q, &c)
                };

                if sim >= threshold {
                    Some((i, sim))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    })
}

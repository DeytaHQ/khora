//! String similarity operations with PyO3 bindings.
//!
//! Provides Levenshtein similarity, sequence match ratio, and batch variants
//! with GIL release and rayon parallelism.

use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::prelude::*;
use strsim::{jaro_winkler, normalized_levenshtein};

/// Normalize a single entity name: lowercase, strip honorifics, collapse
/// whitespace, trim punctuation.
fn normalize_single(name: &str) -> String {
    let mut result = name.to_lowercase();

    // Strip common honorifics (first match only)
    let honorifics = [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sir ", "lord ", "lady ",
    ];
    for h in &honorifics {
        if result.starts_with(h) {
            result = result[h.len()..].to_string();
            break;
        }
    }

    // Collapse whitespace
    let parts: Vec<&str> = result.split_whitespace().collect();
    result = parts.join(" ");

    // Strip leading/trailing punctuation
    result
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_string()
}

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
#[pyo3(signature = (query, candidates, threshold))]
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

        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
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
#[pyo3(signature = (name))]
pub fn normalize_entity_name(name: &str) -> String {
    normalize_single(name)
}

/// Batch normalize entity names (rayon parallel).
#[pyfunction]
#[pyo3(signature = (names))]
pub fn normalize_entity_names_batch(py: Python<'_>, names: Vec<String>) -> Vec<String> {
    py.allow_threads(|| names.par_iter().map(|n| normalize_single(n)).collect())
}

/// Batch sequence match ratio: one query against N candidates.
///
/// Uses rayon for parallelism and releases the GIL during computation.
/// Returns `(index, similarity)` pairs above `threshold`, sorted descending.
#[pyfunction]
#[pyo3(signature = (query, candidates, threshold))]
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

        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        results
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_honorific() {
        assert_eq!(normalize_single("Dr. Smith"), "smith");
        assert_eq!(normalize_single("Mr. Jones"), "jones");
        assert_eq!(normalize_single("Prof. Einstein"), "einstein");
    }

    #[test]
    fn test_normalize_punctuation() {
        assert_eq!(normalize_single("...hello..."), "hello");
        assert_eq!(normalize_single("  spaces  between  "), "spaces between");
    }

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_single("  multiple   spaces  "), "multiple spaces");
    }

    #[test]
    fn test_normalize_only_first_honorific() {
        // "Mr. Dr. Smith" — should strip "Mr." but NOT also strip "Dr."
        // since we break after the first match
        assert_eq!(normalize_single("Mr. Dr. Smith"), "dr. smith");
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_similarity("hello", "hello"), 1.0);
    }

    #[test]
    fn test_levenshtein_case_insensitive() {
        assert_eq!(levenshtein_similarity("Hello", "hello"), 1.0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(levenshtein_similarity("hello", ""), 0.0);
        assert_eq!(levenshtein_similarity("", "hello"), 0.0);
    }
}

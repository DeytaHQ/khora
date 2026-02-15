//! Batch entity resolution with PyO3 bindings.
//!
//! Matches new entity names against existing entities using exact match,
//! alias match, and fuzzy matching. Uses rayon for parallelism.
//!
//! Two variants:
//! - `resolve_entities_batch` — basic (exact + alias + Levenshtein)
//! - `resolve_entities_enhanced` — Jaro-Winkler + token overlap + per-type thresholds

use hashbrown::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use strsim::{jaro_winkler, normalized_levenshtein};

/// Resolve a batch of new entity names against existing entities.
///
/// For each new name, attempts matching in order:
/// 1. **Exact match** — case-insensitive match against existing names
/// 2. **Alias match** — case-insensitive match against each entity's aliases
/// 3. **Fuzzy match** — normalized Levenshtein above `threshold`
///
/// Returns a `Vec` parallel to `new_names`. Each element is either:
/// - `Some((existing_index, score, match_type))` where match_type is
///   `"exact"`, `"alias"`, or `"fuzzy"`
/// - `None` if no match was found
#[pyfunction]
#[pyo3(signature = (new_names, existing_names, existing_aliases, threshold))]
pub fn resolve_entities_batch(
    py: Python<'_>,
    new_names: Vec<String>,
    existing_names: Vec<String>,
    existing_aliases: Vec<Vec<String>>,
    threshold: f64,
) -> Vec<Option<(usize, f64, String)>> {
    // Pre-lowercase existing names and aliases outside the hot loop
    let existing_lower: Vec<String> = existing_names.iter().map(|n| n.to_lowercase()).collect();
    let aliases_lower: Vec<Vec<String>> = existing_aliases
        .iter()
        .map(|aliases| aliases.iter().map(|a| a.to_lowercase()).collect())
        .collect();

    py.allow_threads(|| {
        new_names
            .par_iter()
            .map(|new_name| {
                let query = new_name.to_lowercase();

                // Step 1: Exact name match
                for (idx, existing) in existing_lower.iter().enumerate() {
                    if query == *existing {
                        return Some((idx, 1.0, "exact".to_string()));
                    }
                }

                // Step 2: Alias match
                for (idx, aliases) in aliases_lower.iter().enumerate() {
                    for alias in aliases {
                        if query == *alias {
                            return Some((idx, 1.0, "alias".to_string()));
                        }
                    }
                }

                // Step 3: Fuzzy match — find best above threshold
                let mut best_idx = None;
                let mut best_score = threshold;

                for (idx, existing) in existing_lower.iter().enumerate() {
                    if query.is_empty() || existing.is_empty() {
                        continue;
                    }
                    let sim = normalized_levenshtein(&query, existing);
                    if sim > best_score {
                        best_score = sim;
                        best_idx = Some(idx);
                    }
                }

                best_idx.map(|idx| (idx, best_score, "fuzzy".to_string()))
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Token overlap helper
// ---------------------------------------------------------------------------

/// Compute token overlap ratio between two lowercased strings.
///
/// Tokenises on whitespace, computes |intersection| / max(|A|, |B|).
/// Returns 0.0 if either side has no tokens.
#[inline]
fn token_overlap(a: &str, b: &str) -> f64 {
    let tokens_a: Vec<&str> = a.split_whitespace().collect();
    let tokens_b: Vec<&str> = b.split_whitespace().collect();

    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let set_a: hashbrown::HashSet<&str> = tokens_a.iter().copied().collect();
    let set_b: hashbrown::HashSet<&str> = tokens_b.iter().copied().collect();
    let shared = set_a.intersection(&set_b).count();
    let max_len = set_a.len().max(set_b.len());

    shared as f64 / max_len as f64
}

// ---------------------------------------------------------------------------
// Enhanced entity resolution
// ---------------------------------------------------------------------------

/// Enhanced entity resolution using Jaro-Winkler + token overlap + per-type thresholds.
///
/// For each new entity (name + type), attempts matching against existing entities:
/// 1. **Exact match** — case-insensitive name equality (score 1.0)
/// 2. **Alias match** — case-insensitive alias equality (score 1.0)
/// 3. **Enhanced fuzzy** — combined Jaro-Winkler (0.6) + token overlap (0.4),
///    checked against the per-type threshold for the new entity's type.
///
/// Args:
///   - `new_names` / `new_types`: parallel arrays for new entities
///   - `existing_names` / `existing_aliases` / `existing_types`: existing entities
///   - `type_thresholds_keys` / `type_thresholds_vals`: per-type thresholds
///   - `default_threshold`: fallback when type not in map
///
/// Returns a `Vec` parallel to `new_names`. Each element is either
/// `Some((existing_index, score, match_type))` or `None`.
#[pyfunction]
#[pyo3(signature = (new_names, new_types, existing_names, existing_aliases, existing_types, type_thresholds_keys, type_thresholds_vals, default_threshold))]
#[allow(clippy::too_many_arguments)]
pub fn resolve_entities_enhanced(
    py: Python<'_>,
    new_names: Vec<String>,
    new_types: Vec<String>,
    existing_names: Vec<String>,
    existing_aliases: Vec<Vec<String>>,
    existing_types: Vec<String>,
    type_thresholds_keys: Vec<String>,
    type_thresholds_vals: Vec<f64>,
    default_threshold: f64,
) -> Vec<Option<(usize, f64, String)>> {
    // Build per-type threshold map
    let type_thresholds: HashMap<String, f64> = type_thresholds_keys
        .into_iter()
        .zip(type_thresholds_vals)
        .collect();

    // Pre-lowercase existing names, aliases, types
    let existing_lower: Vec<String> = existing_names.iter().map(|n| n.to_lowercase()).collect();
    let aliases_lower: Vec<Vec<String>> = existing_aliases
        .iter()
        .map(|aliases| aliases.iter().map(|a| a.to_lowercase()).collect())
        .collect();
    let existing_types_upper: Vec<String> =
        existing_types.iter().map(|t| t.to_uppercase()).collect();

    py.allow_threads(|| {
        new_names
            .par_iter()
            .zip(new_types.par_iter())
            .map(|(new_name, new_type)| {
                let query = new_name.to_lowercase();
                let query_type = new_type.to_uppercase();

                // Look up per-type threshold
                let threshold = type_thresholds
                    .get(&query_type)
                    .copied()
                    .unwrap_or(default_threshold);

                // Step 1: Exact name match (same type only)
                for (idx, existing) in existing_lower.iter().enumerate() {
                    if query == *existing && existing_types_upper[idx] == query_type {
                        return Some((idx, 1.0, "exact".to_string()));
                    }
                }

                // Step 2: Alias match (same type only)
                for (idx, aliases) in aliases_lower.iter().enumerate() {
                    if existing_types_upper[idx] != query_type {
                        continue;
                    }
                    for alias in aliases {
                        if query == *alias {
                            return Some((idx, 1.0, "alias".to_string()));
                        }
                    }
                }

                // Step 3: Enhanced fuzzy — Jaro-Winkler (0.6) + token overlap (0.4)
                let mut best_idx: Option<usize> = None;
                let mut best_score: f64 = threshold;

                for (idx, existing) in existing_lower.iter().enumerate() {
                    if existing_types_upper[idx] != query_type {
                        continue;
                    }
                    if query.is_empty() || existing.is_empty() {
                        continue;
                    }

                    let jw = jaro_winkler(&query, existing);
                    let tok = token_overlap(&query, existing);
                    let combined = 0.6 * jw + 0.4 * tok;

                    if combined > best_score {
                        best_score = combined;
                        best_idx = Some(idx);
                    }
                }

                best_idx.map(|idx| (idx, best_score, "fuzzy".to_string()))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_overlap_identical() {
        assert!((token_overlap("machine learning", "machine learning") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_overlap_partial() {
        // "machine learning" vs "machine learning systems" → 2/3
        let score = token_overlap("machine learning", "machine learning systems");
        assert!((score - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_token_overlap_empty() {
        assert_eq!(token_overlap("", "hello"), 0.0);
        assert_eq!(token_overlap("hello", ""), 0.0);
    }

    #[test]
    fn test_token_overlap_disjoint() {
        assert_eq!(token_overlap("alpha beta", "gamma delta"), 0.0);
    }
}

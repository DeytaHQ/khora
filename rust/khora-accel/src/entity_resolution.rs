//! Batch entity resolution with PyO3 bindings.
//!
//! Matches new entity names against existing entities using exact match,
//! alias match, and fuzzy (Levenshtein) matching. Uses rayon for parallelism.

use pyo3::prelude::*;
use rayon::prelude::*;
use strsim::normalized_levenshtein;

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

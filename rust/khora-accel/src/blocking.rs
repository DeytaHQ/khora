//! Pairwise similarity with name-token-prefix blocking.
//!
//! Cross-batch entity resolution wants to find every pair of entities
//! whose embedding cosine is above a threshold AND whose names share at
//! least one tokenisation signal. Naive `pairwise_cosine_above_threshold`
//! is O(N^2) and saturates at ~100k entities; token-prefix blocking cuts
//! the candidate set to roughly the rows that share a 3-char prefix on
//! any token, which on a realistic name distribution is ~100x smaller.
//!
//! See `khora._accel.block_and_score_pairs` for the Python wrapper.

use hashbrown::{HashMap, HashSet};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

const TOKEN_PREFIX_LEN: usize = 3;

/// Tokenize a single name into lowercase tokens of length >= 2.
///
/// Splits on every character that is not alphanumeric. Drops 1-char
/// tokens (they're too common to be useful blocking keys).
fn tokenize(name: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in name.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                current.push(low);
            }
        } else if !current.is_empty() {
            if current.chars().count() >= 2 {
                tokens.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if !current.is_empty() && current.chars().count() >= 2 {
        tokens.push(current);
    }
    tokens
}

/// First `TOKEN_PREFIX_LEN` characters of a token (or the whole token if
/// shorter). Operates on characters, not bytes, so non-ASCII names work.
fn prefix_key(token: &str) -> String {
    token.chars().take(TOKEN_PREFIX_LEN).collect()
}

/// Pairwise cosine similarity with optional name-token-prefix blocking.
///
/// `embeddings`: `(N, D)` pre-normalised f32 matrix (so dot == cosine).
/// `names`: length-N list of entity names, one per row.
/// `threshold`: only pairs with similarity >= threshold are returned.
/// `name_token_blocking`: when true, two rows are only scored if their
///     names share at least one token prefix key (lowercase, alphanumeric
///     tokens of length >= 2, first 3 chars). When false, the kernel is
///     equivalent to `pairwise_cosine_above_threshold` — every i < j is
///     scored.
///
/// Returns `Vec<(i, j, similarity)>` with i < j, in row-major i order
/// (parallel collect preserves the outer-iter order via rayon's
/// flat_map-then-collect contract).
#[pyfunction]
#[pyo3(signature = (embeddings, names, threshold, name_token_blocking=true))]
pub fn block_and_score_pairs(
    py: Python<'_>,
    embeddings: PyReadonlyArray2<'_, f32>,
    names: Vec<String>,
    threshold: f32,
    name_token_blocking: bool,
) -> PyResult<Vec<(usize, usize, f32)>> {
    let e_array = embeddings.as_array();
    let n = e_array.nrows();
    if names.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "names length ({}) does not match embeddings rows ({})",
            names.len(),
            n
        )));
    }
    let e_owned = e_array.to_owned();

    let result = py.detach(|| {
        if n < 2 {
            return Vec::new();
        }

        if !name_token_blocking {
            // Behavioural parity with pairwise dot product over pre-normalised
            // vectors. Pre-normalised => dot product == cosine.
            return (0..n)
                .into_par_iter()
                .flat_map(|i| {
                    let row_i = e_owned.row(i);
                    let mut local = Vec::new();
                    for j in (i + 1)..n {
                        let row_j = e_owned.row(j);
                        let dot: f64 = row_i
                            .iter()
                            .zip(row_j.iter())
                            .map(|(&a, &b)| (a as f64) * (b as f64))
                            .sum();
                        let sim = dot as f32;
                        if sim >= threshold {
                            local.push((i, j, sim));
                        }
                    }
                    local
                })
                .collect();
        }

        // Token-prefix blocking path.
        // 1. Per-row blocking keys (HashSet so we can intersect cheaply).
        let row_keys: Vec<HashSet<String>> = names
            .par_iter()
            .map(|name| {
                let mut keys = HashSet::new();
                for tok in tokenize(name) {
                    keys.insert(prefix_key(&tok));
                }
                keys
            })
            .collect();

        // 2. Inverted index: prefix key -> rows containing it.
        let mut inverted: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, keys) in row_keys.iter().enumerate() {
            for k in keys {
                inverted.entry(k.clone()).or_default().push(i);
            }
        }

        // 3. For each row i, gather candidate j > i from the inverted
        //    index, dedupe, then score.
        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let mut local = Vec::new();
                let keys = &row_keys[i];
                if keys.is_empty() {
                    return local;
                }
                let mut candidates: HashSet<usize> = HashSet::new();
                for k in keys {
                    if let Some(rows) = inverted.get(k) {
                        for &j in rows {
                            if j > i {
                                candidates.insert(j);
                            }
                        }
                    }
                }
                let row_i = e_owned.row(i);
                let mut candidates: Vec<usize> = candidates.into_iter().collect();
                candidates.sort_unstable();
                for j in candidates {
                    let row_j = e_owned.row(j);
                    let dot: f64 = row_i
                        .iter()
                        .zip(row_j.iter())
                        .map(|(&a, &b)| (a as f64) * (b as f64))
                        .sum();
                    let sim = dot as f32;
                    if sim >= threshold {
                        local.push((i, j, sim));
                    }
                }
                local
            })
            .collect()
    });

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let toks = tokenize("Acme Corp, Inc.");
        assert_eq!(toks, vec!["acme", "corp", "inc"]);
    }

    #[test]
    fn test_tokenize_drops_short_tokens() {
        // 1-char tokens are dropped, 2-char survives.
        let toks = tokenize("A B Co");
        assert_eq!(toks, vec!["co"]);
    }

    #[test]
    fn test_tokenize_unicode() {
        // Non-ASCII letters are alphanumeric and lowercased.
        let toks = tokenize("Café München");
        assert_eq!(toks, vec!["café", "münchen"]);
    }

    #[test]
    fn test_prefix_key_short_token() {
        assert_eq!(prefix_key("co"), "co");
        assert_eq!(prefix_key("acme"), "acm");
    }

    #[test]
    fn test_prefix_key_unicode() {
        // First 3 chars by character, not byte.
        let key = prefix_key("café");
        assert_eq!(key.chars().count(), 3);
        assert_eq!(key, "caf");
    }
}

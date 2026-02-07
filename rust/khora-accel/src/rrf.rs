//! Reciprocal Rank Fusion (RRF) with PyO3 bindings.
//!
//! Mirrors `khora.engines.vectorcypher.fusion` — basic RRF, weighted RRF,
//! and min-max score normalisation.

use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

/// Basic Reciprocal Rank Fusion.
///
/// `ranked_lists` is a list of ranked ID lists. For each list the score
/// contribution is `1 / (k + rank)` where rank is **1-indexed**.
///
/// Returns `Vec<(id, score)>` sorted descending by score.
#[pyfunction]
#[pyo3(signature = (ranked_lists, k = 60))]
pub fn reciprocal_rank_fusion(
    ranked_lists: Vec<Vec<String>>,
    k: usize,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for list in &ranked_lists {
        for (rank_0, item_id) in list.iter().enumerate() {
            let rank = rank_0 + 1; // 1-indexed
            *scores.entry(item_id.clone()).or_insert(0.0) += 1.0 / (k as f64 + rank as f64);
        }
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
    results
}

/// Weighted Reciprocal Rank Fusion.
///
/// Each ranked list carries a weight. Score contribution per item:
/// `weight / (k + rank + 1)` where rank is **0-indexed** (so +1 converts to 1-indexed).
///
/// Returns `Vec<(id, score)>` sorted descending by score.
#[pyfunction]
#[pyo3(signature = (ranked_lists, k = 60))]
pub fn weighted_rrf(
    ranked_lists: Vec<(f64, Vec<String>)>,
    k: usize,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (weight, list) in &ranked_lists {
        for (rank_0, item_id) in list.iter().enumerate() {
            *scores.entry(item_id.clone()).or_insert(0.0) +=
                weight / (k as f64 + rank_0 as f64 + 1.0);
        }
    }

    let mut results: Vec<(String, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
    results
}

/// Min-max normalise a list of scores to `[0, 1]`.
///
/// If all scores are identical, returns a vector of `1.0`.
#[pyfunction]
pub fn normalize_scores(scores: Vec<f64>) -> Vec<f64> {
    if scores.is_empty() {
        return scores;
    }

    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max - min).abs() < f64::EPSILON {
        return vec![1.0; scores.len()];
    }

    let range = max - min;
    scores.iter().map(|&s| (s - min) / range).collect()
}

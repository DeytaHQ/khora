//! Reciprocal Rank Fusion (RRF) with PyO3 bindings.
//!
//! Mirrors `khora.engines.vectorcypher.fusion` — basic RRF, weighted RRF,
//! and min-max score normalisation.

use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::utils::min_max_normalize;

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

/// Weighted RRF with score normalization before fusion.
///
/// 1. Min-max normalizes vector and graph scores independently
/// 2. Computes weighted RRF: weight / (k + rank)
/// 3. Adds small normalized score contribution for tiebreaking (0.01 * weight * norm_score)
/// 4. Returns sorted (id, combined_score) pairs
///
/// Releases the GIL for computation.
#[pyfunction]
#[pyo3(signature = (vector_results, graph_results, k = 60, vector_weight = 0.6, graph_weight = 0.4))]
pub fn weighted_rrf_normalized(
    py: Python<'_>,
    vector_results: Vec<(String, f64)>,
    graph_results: Vec<(String, f64)>,
    k: usize,
    vector_weight: f64,
    graph_weight: f64,
) -> Vec<(String, f64)> {
    py.allow_threads(|| {
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut score_contributions: HashMap<String, f64> = HashMap::new();

        // Normalize and process vector results
        if !vector_results.is_empty() {
            let raw_scores: Vec<f64> = vector_results.iter().map(|(_, s)| *s).collect();
            let normalized = min_max_normalize(&raw_scores);

            for (rank_0, ((item_id, _), norm_score)) in
                vector_results.iter().zip(normalized.iter()).enumerate()
            {
                let rank = rank_0 + 1;
                *scores.entry(item_id.clone()).or_insert(0.0) +=
                    vector_weight / (k as f64 + rank as f64);
                *score_contributions.entry(item_id.clone()).or_insert(0.0) +=
                    vector_weight * norm_score * 0.01;
            }
        }

        // Normalize and process graph results
        if !graph_results.is_empty() {
            let raw_scores: Vec<f64> = graph_results.iter().map(|(_, s)| *s).collect();
            let normalized = min_max_normalize(&raw_scores);

            for (rank_0, ((item_id, _), norm_score)) in
                graph_results.iter().zip(normalized.iter()).enumerate()
            {
                let rank = rank_0 + 1;
                *scores.entry(item_id.clone()).or_insert(0.0) +=
                    graph_weight / (k as f64 + rank as f64);
                *score_contributions.entry(item_id.clone()).or_insert(0.0) +=
                    graph_weight * norm_score * 0.01;
            }
        }

        // Combine
        let mut results: Vec<(String, f64)> = scores
            .into_iter()
            .map(|(id, rrf)| {
                let contrib = score_contributions.get(&id).copied().unwrap_or(0.0);
                (id, rrf + contrib)
            })
            .collect();

        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        results
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_rrf() {
        let lists = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["b".to_string(), "a".to_string(), "d".to_string()],
        ];
        let results = reciprocal_rank_fusion(lists, 60);
        assert_eq!(results.len(), 4); // a, b, c, d

        // "a" and "b" appear in both lists at symmetric positions → tied
        let a_score = results.iter().find(|(id, _)| id == "a").unwrap().1;
        let b_score = results.iter().find(|(id, _)| id == "b").unwrap().1;
        assert!((a_score - b_score).abs() < 1e-10);

        // "c" and "d" appear once each at rank 3 → tied
        let c_score = results.iter().find(|(id, _)| id == "c").unwrap().1;
        let d_score = results.iter().find(|(id, _)| id == "d").unwrap().1;
        assert!((c_score - d_score).abs() < 1e-10);

        // Items in two lists should score higher than items in one
        assert!(a_score > c_score);
    }

    #[test]
    fn test_empty_input() {
        let results = reciprocal_rank_fusion(vec![], 60);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_list() {
        let lists = vec![vec!["x".to_string(), "y".to_string()]];
        let results = reciprocal_rank_fusion(lists, 60);
        assert_eq!(results.len(), 2);
        // First item (rank 1) should have higher score than second (rank 2)
        assert!(results[0].1 > results[1].1);
        assert_eq!(results[0].0, "x");
    }

    #[test]
    fn test_normalize_scores() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_scores(scores);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_scores_identical() {
        let scores = vec![3.0, 3.0, 3.0];
        let normalized = normalize_scores(scores);
        assert!(normalized.iter().all(|&s| (s - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_normalize_scores_empty() {
        let normalized = normalize_scores(vec![]);
        assert!(normalized.is_empty());
    }
}

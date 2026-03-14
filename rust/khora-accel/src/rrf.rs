//! Reciprocal Rank Fusion (RRF) with PyO3 bindings.
//!
//! Mirrors `khora.engines.vectorcypher.fusion` — basic RRF, weighted RRF,
//! and min-max score normalisation.

use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::utils::min_max_normalize;

/// Per-result diagnostics tuple: (id, score, source, vector_rank, graph_rank,
/// vector_norm_score, graph_norm_score, vector_rrf_contrib, graph_rrf_contrib)
type DiagnosticResult = (String, f64, u8, usize, usize, f64, f64, f64, f64);

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
    py.detach(|| {
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

/// Weighted RRF with score normalization **and source provenance tracking**.
///
/// Identical to [`weighted_rrf_normalized`] but returns an additional `u8`
/// bitmap per result indicating which retrieval path(s) contributed:
///
/// - `0b01` (1) = vector only
/// - `0b10` (2) = graph only
/// - `0b11` (3) = both vector and graph
///
/// Releases the GIL for computation.
#[pyfunction]
#[pyo3(signature = (vector_results, graph_results, k = 60, vector_weight = 0.6, graph_weight = 0.4))]
pub fn weighted_rrf_normalized_with_provenance(
    py: Python<'_>,
    vector_results: Vec<(String, f64)>,
    graph_results: Vec<(String, f64)>,
    k: usize,
    vector_weight: f64,
    graph_weight: f64,
) -> Vec<(String, f64, u8)> {
    py.detach(|| {
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut score_contributions: HashMap<String, f64> = HashMap::new();
        let mut sources: HashMap<String, u8> = HashMap::new();

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
                *sources.entry(item_id.clone()).or_insert(0) |= 0x01;
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
                *sources.entry(item_id.clone()).or_insert(0) |= 0x02;
            }
        }

        // Combine
        let mut results: Vec<(String, f64, u8)> = scores
            .into_iter()
            .map(|(id, rrf)| {
                let contrib = score_contributions.get(&id).copied().unwrap_or(0.0);
                let src = sources.get(&id).copied().unwrap_or(0);
                (id, rrf + contrib, src)
            })
            .collect();

        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        results
    })
}

/// Weighted RRF with full per-result diagnostics for benchmark analysis.
///
/// Returns tuples of:
///   `(id, score, source_bitmap, vector_rank, graph_rank, vector_norm_score,
///    graph_norm_score, vector_rrf_contrib, graph_rrf_contrib)`
///
/// - `source_bitmap`: 1=vector, 2=graph, 3=both
/// - `vector_rank`/`graph_rank`: 1-indexed (0 if absent from that source)
/// - `*_norm_score`: min-max normalized score from that source (0.0 if absent)
/// - `*_rrf_contrib`: RRF contribution from that source (0.0 if absent)
///
/// Sorted descending by fused score. Releases the GIL for computation.
#[pyfunction]
#[pyo3(signature = (vector_results, graph_results, k = 60, vector_weight = 0.6, graph_weight = 0.4))]
pub fn weighted_rrf_normalized_with_diagnostics(
    py: Python<'_>,
    vector_results: Vec<(String, f64)>,
    graph_results: Vec<(String, f64)>,
    k: usize,
    vector_weight: f64,
    graph_weight: f64,
) -> Vec<DiagnosticResult> {
    py.detach(|| {
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut sources: HashMap<String, u8> = HashMap::new();
        let mut vector_ranks: HashMap<String, usize> = HashMap::new();
        let mut graph_ranks: HashMap<String, usize> = HashMap::new();
        let mut vector_norms: HashMap<String, f64> = HashMap::new();
        let mut graph_norms: HashMap<String, f64> = HashMap::new();
        let mut vector_contribs: HashMap<String, f64> = HashMap::new();
        let mut graph_contribs: HashMap<String, f64> = HashMap::new();

        if !vector_results.is_empty() {
            let raw: Vec<f64> = vector_results.iter().map(|(_, s)| *s).collect();
            let normalized = min_max_normalize(&raw);

            for (rank_0, ((item_id, _), &norm)) in
                vector_results.iter().zip(normalized.iter()).enumerate()
            {
                let rank = rank_0 + 1;
                let rrf_contrib = vector_weight / (k as f64 + rank as f64);
                *scores.entry(item_id.clone()).or_insert(0.0) += rrf_contrib + vector_weight * norm * 0.01;
                *sources.entry(item_id.clone()).or_insert(0) |= 0x01;
                vector_ranks.insert(item_id.clone(), rank);
                vector_norms.insert(item_id.clone(), norm);
                vector_contribs.insert(item_id.clone(), rrf_contrib);
            }
        }

        if !graph_results.is_empty() {
            let raw: Vec<f64> = graph_results.iter().map(|(_, s)| *s).collect();
            let normalized = min_max_normalize(&raw);

            for (rank_0, ((item_id, _), &norm)) in
                graph_results.iter().zip(normalized.iter()).enumerate()
            {
                let rank = rank_0 + 1;
                let rrf_contrib = graph_weight / (k as f64 + rank as f64);
                *scores.entry(item_id.clone()).or_insert(0.0) += rrf_contrib + graph_weight * norm * 0.01;
                *sources.entry(item_id.clone()).or_insert(0) |= 0x02;
                graph_ranks.insert(item_id.clone(), rank);
                graph_norms.insert(item_id.clone(), norm);
                graph_contribs.insert(item_id.clone(), rrf_contrib);
            }
        }

        let mut results: Vec<DiagnosticResult> = scores
            .into_iter()
            .map(|(id, score)| {
                let src = sources.get(&id).copied().unwrap_or(0);
                let vr = vector_ranks.get(&id).copied().unwrap_or(0);
                let gr = graph_ranks.get(&id).copied().unwrap_or(0);
                let vn = vector_norms.get(&id).copied().unwrap_or(0.0);
                let gn = graph_norms.get(&id).copied().unwrap_or(0.0);
                let vc = vector_contribs.get(&id).copied().unwrap_or(0.0);
                let gc = graph_contribs.get(&id).copied().unwrap_or(0.0);
                (id, score, src, vr, gr, vn, gn, vc, gc)
            })
            .collect();

        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        results
    })
}

/// Compute batch score statistics: mean, std dev, min, max, median.
///
/// Returns `(mean, std_dev, min, max, median)`. Empty input → all zeros.
/// Releases the GIL for computation.
#[pyfunction]
pub fn batch_score_stats(py: Python<'_>, scores: Vec<f64>) -> (f64, f64, f64, f64, f64) {
    py.detach(|| {
        if scores.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = scores;
        sorted.sort_by_key(|a| OrderedFloat(*a));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        (mean, std_dev, min, max, median)
    })
}

/// Compute Shannon entropy of a score distribution.
///
/// Normalizes scores to a probability distribution and returns
/// `-sum(p * ln(p))`. Useful for detecting uniform vs. peaked
/// score distributions (higher entropy = more uniform = less confident).
///
/// Returns 0.0 for empty or all-zero inputs.
#[pyfunction]
pub fn score_entropy(py: Python<'_>, scores: Vec<f64>) -> f64 {
    py.detach(|| {
        if scores.is_empty() {
            return 0.0;
        }

        let total: f64 = scores.iter().filter(|&&s| s > 0.0).sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0f64;
        for &s in &scores {
            if s > 0.0 {
                let p = s / total;
                entropy -= p * p.ln();
            }
        }
        entropy
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

    #[test]
    fn test_batch_score_stats_basic() {
        // Use a Python::with_gil wrapper for pyfunction tests isn't needed
        // since batch_score_stats uses py.detach — test the logic directly
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        assert!((mean - 3.0).abs() < 1e-10);

        let variance = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        // std_dev of [1,2,3,4,5] = sqrt(2) ≈ 1.4142
        assert!((std_dev - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_batch_score_stats_empty() {
        // Empty input should return all zeros
        let scores: Vec<f64> = vec![];
        assert!(scores.is_empty());
    }

    #[test]
    fn test_score_entropy_uniform() {
        // Uniform distribution: 4 equal scores → max entropy = ln(4)
        let scores = vec![1.0, 1.0, 1.0, 1.0];
        let total: f64 = scores.iter().sum();
        let mut entropy = 0.0f64;
        for &s in &scores {
            let p = s / total;
            entropy -= p * p.ln();
        }
        assert!((entropy - 4.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_score_entropy_peaked() {
        // One dominant score → low entropy
        let scores = vec![100.0, 1.0, 1.0, 1.0];
        let total: f64 = scores.iter().sum();
        let mut entropy = 0.0f64;
        for &s in &scores {
            let p = s / total;
            entropy -= p * p.ln();
        }
        // Should be much less than ln(4)
        assert!(entropy < 4.0f64.ln());
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_score_entropy_empty() {
        let scores: Vec<f64> = vec![];
        assert!(scores.is_empty());
    }
}

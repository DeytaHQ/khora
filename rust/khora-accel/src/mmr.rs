//! Maximal Marginal Relevance (MMR) diversity selection with PyO3 bindings.
//!
//! Greedy MMR iteratively picks candidates that maximise
//! `lambda * relevance - (1 - lambda) * max_similarity_to_selected`.
//! Embeddings are assumed to be pre-normalized so dot product = cosine sim.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

/// Greedy MMR diversity selection.
///
/// Args:
///   embeddings: (N, D) matrix — one row per candidate, pre-normalized.
///   scores: relevance score per candidate (length N).
///   lambda_param: tradeoff (0 = pure diversity, 1 = pure relevance).
///   k: number of items to select.
///
/// Returns: indices of selected items in selection order.
/// Releases the GIL during computation.
#[pyfunction]
#[pyo3(signature = (embeddings, scores, lambda_param, k))]
pub fn mmr_diversity_select(
    py: Python<'_>,
    embeddings: PyReadonlyArray2<'_, f32>,
    scores: Vec<f32>,
    lambda_param: f32,
    k: usize,
) -> Vec<usize> {
    let e_array = embeddings.as_array();
    let e_owned = e_array.to_owned();

    py.detach(|| {
        let n = e_owned.nrows();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        let dim = e_owned.ncols();

        // Build a flat contiguous buffer for cache-friendly access
        let flat: Vec<f32> = e_owned.as_slice().map_or_else(
            || {
                let mut buf = Vec::with_capacity(n * dim);
                for i in 0..n {
                    buf.extend_from_slice(
                        e_owned.row(i).as_slice().unwrap_or(&e_owned.row(i).to_vec()),
                    );
                }
                buf
            },
            |s| s.to_vec(),
        );

        // Track which candidates are still available
        let mut available = vec![true; n];
        let mut selected: Vec<usize> = Vec::with_capacity(k);

        // max_sim_to_selected[i] = max dot(emb[i], emb[s]) for s in selected
        let mut max_sim: Vec<f32> = vec![f32::NEG_INFINITY; n];

        let one_minus_lambda = 1.0 - lambda_param;

        for _ in 0..k {
            let mut best_idx: usize = usize::MAX;
            let mut best_mmr = f32::NEG_INFINITY;

            for i in 0..n {
                if !available[i] {
                    continue;
                }

                let sim_to_selected = if selected.is_empty() {
                    0.0
                } else {
                    // max_sim[i] was updated incrementally; clamp negative to 0
                    max_sim[i].max(0.0)
                };

                let mmr = lambda_param * scores[i] - one_minus_lambda * sim_to_selected;

                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Mark selected
            available[best_idx] = false;
            selected.push(best_idx);

            // Incrementally update max_sim for remaining candidates
            let sel_start = best_idx * dim;
            let sel_emb = &flat[sel_start..sel_start + dim];

            for i in 0..n {
                if !available[i] {
                    continue;
                }
                let cand_start = i * dim;
                let cand_emb = &flat[cand_start..cand_start + dim];

                // Dot product (= cosine sim for pre-normalized vectors)
                let dot = dot_f32(sel_emb, cand_emb);

                if selected.len() == 1 {
                    // First selection — initialise
                    max_sim[i] = dot;
                } else if dot > max_sim[i] {
                    max_sim[i] = dot;
                }
            }
        }

        selected
    })
}

/// Fast f32 dot product.  Compiler auto-vectorises this on x86-64 with SSE/AVX.
#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: dot product of two slices.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    /// Pure-Rust MMR for testing (mirrors the pyfunction logic).
    fn mmr_select_pure(
        embeddings: &[Vec<f32>],
        scores: &[f32],
        lambda_param: f32,
        k: usize,
    ) -> Vec<usize> {
        let n = embeddings.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);

        let mut available = vec![true; n];
        let mut selected: Vec<usize> = Vec::with_capacity(k);

        for _ in 0..k {
            let mut best_idx = usize::MAX;
            let mut best_mmr = f32::NEG_INFINITY;

            for i in 0..n {
                if !available[i] {
                    continue;
                }
                let sim_to_selected = if selected.is_empty() {
                    0.0
                } else {
                    selected
                        .iter()
                        .map(|&s| dot(&embeddings[i], &embeddings[s]))
                        .fold(f32::NEG_INFINITY, f32::max)
                        .max(0.0)
                };
                let mmr = lambda_param * scores[i] - (1.0 - lambda_param) * sim_to_selected;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            available[best_idx] = false;
            selected.push(best_idx);
        }
        selected
    }

    #[test]
    fn test_mmr_pure_relevance() {
        // lambda=1 → pure relevance, should pick in score order
        let embs = vec![
            normalize(&[1.0, 0.0]),
            normalize(&[0.0, 1.0]),
            normalize(&[1.0, 1.0]),
        ];
        let scores = vec![0.5, 0.9, 0.7];
        let result = mmr_select_pure(&embs, &scores, 1.0, 2);
        assert_eq!(result, vec![1, 2]); // highest scores first
    }

    #[test]
    fn test_mmr_pure_diversity() {
        // lambda=0 → pure diversity, should pick most dissimilar items
        let embs = vec![
            normalize(&[1.0, 0.0]),  // 0: points right
            normalize(&[1.0, 0.01]), // 1: nearly same as 0
            normalize(&[0.0, 1.0]),  // 2: orthogonal
        ];
        let scores = vec![0.9, 0.8, 0.7];
        let result = mmr_select_pure(&embs, &scores, 0.0, 2);
        // First pick: all sim=0 → pick highest score → 0
        // Second pick: 1 is very similar to 0, 2 is dissimilar → pick 2
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn test_mmr_k_exceeds_n() {
        let embs = vec![normalize(&[1.0, 0.0]), normalize(&[0.0, 1.0])];
        let scores = vec![0.5, 0.9];
        let result = mmr_select_pure(&embs, &scores, 0.5, 10);
        assert_eq!(result.len(), 2); // only 2 candidates
    }

    #[test]
    fn test_mmr_empty() {
        let embs: Vec<Vec<f32>> = vec![];
        let scores: Vec<f32> = vec![];
        let result = mmr_select_pure(&embs, &scores, 0.5, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dot_f32_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let result = dot_f32(&a, &b);
        assert!((result - 32.0).abs() < 1e-5);
    }
}

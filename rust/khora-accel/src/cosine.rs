//! Cosine similarity operations with PyO3 bindings.
//!
//! Provides single-pair, batch (1-to-N), and pairwise (all-pairs) cosine
//! similarity with numpy zero-copy access and GIL release for batch ops.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Cosine similarity between two vectors using a fused dot+norm single pass.
///
/// Returns 0.0 on dimension mismatch or zero norms.
#[pyfunction]
pub fn cosine_similarity(vec1: Vec<f32>, vec2: Vec<f32>) -> f32 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let (mut dot, mut norm1, mut norm2) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in vec1.iter().zip(vec2.iter()) {
        let a = *a as f64;
        let b = *b as f64;
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    (dot / (norm1.sqrt() * norm2.sqrt())) as f32
}

/// Batch cosine similarity: one query against N candidate vectors.
///
/// Accepts numpy arrays via zero-copy. Releases GIL during computation.
/// Returns `(index, similarity)` pairs above `threshold`, sorted descending.
#[pyfunction]
#[pyo3(signature = (query, candidates, threshold))]
pub fn batch_cosine_similarity(
    py: Python<'_>,
    query: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, f32>,
    threshold: f32,
) -> Vec<(usize, f32)> {
    let q_array = query.as_array();
    let c_array = candidates.as_array();

    // Copy to owned arrays so we can release the GIL
    let q_owned = q_array.to_owned();
    let c_owned = c_array.to_owned();

    py.allow_threads(|| {
        let q = q_owned.as_slice().unwrap();
        let n_candidates = c_owned.nrows();

        // Pre-compute query norm
        let q_norm: f64 = q.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if q_norm == 0.0 {
            return Vec::new();
        }
        let q_norm = q_norm.sqrt();

        let mut results: Vec<(usize, f32)> = (0..n_candidates)
            .into_par_iter()
            .filter_map(|i| {
                let row = c_owned.row(i);
                let (mut dot, mut c_norm) = (0.0f64, 0.0f64);
                for (a, b) in q.iter().zip(row.iter()) {
                    let a = *a as f64;
                    let b = *b as f64;
                    dot += a * b;
                    c_norm += b * b;
                }
                if c_norm == 0.0 {
                    return None;
                }
                let sim = (dot / (q_norm * c_norm.sqrt())) as f32;
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

/// All-pairs cosine similarity above a threshold.
///
/// Uses rayon for parallelism. Returns `(i, j, similarity)` triples where
/// `i < j` and `similarity >= threshold`.
#[pyfunction]
#[pyo3(signature = (embeddings, threshold))]
pub fn pairwise_cosine_above_threshold(
    py: Python<'_>,
    embeddings: PyReadonlyArray2<'_, f32>,
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let e_array = embeddings.as_array();
    let e_owned = e_array.to_owned();

    py.allow_threads(|| {
        let n = e_owned.nrows();
        if n < 2 {
            return Vec::new();
        }

        // Pre-compute norms
        let norms: Vec<f64> = (0..n)
            .map(|i| {
                let row = e_owned.row(i);
                let sq_sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
                sq_sum.sqrt()
            })
            .collect();

        // Parallel over rows, collect pairs where i < j
        (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_results = Vec::new();
                if norms[i] == 0.0 {
                    return local_results;
                }
                let row_i = e_owned.row(i);
                for j in (i + 1)..n {
                    if norms[j] == 0.0 {
                        continue;
                    }
                    let row_j = e_owned.row(j);
                    let dot: f64 = row_i
                        .iter()
                        .zip(row_j.iter())
                        .map(|(&a, &b)| (a as f64) * (b as f64))
                        .sum();
                    let sim = (dot / (norms[i] * norms[j])) as f32;
                    if sim >= threshold {
                        local_results.push((i, j, sim));
                    }
                }
                local_results
            })
            .collect()
    })
}

/// L2-normalize a batch of embedding vectors.
///
/// Each vector is divided by its L2 norm. Zero vectors are returned as-is.
/// Uses rayon for parallelism on batches > 64 vectors.
/// Releases the GIL during computation.
#[pyfunction]
pub fn normalize_embeddings_batch(
    py: Python<'_>,
    vectors: Vec<Vec<f32>>,
) -> Vec<Vec<f32>> {
    py.allow_threads(|| {
        let normalize_one = |vec: &Vec<f32>| -> Vec<f32> {
            let sq_sum: f64 = vec.iter().map(|&v| (v as f64) * (v as f64)).sum();
            if sq_sum == 0.0 {
                return vec.clone();
            }
            let norm = sq_sum.sqrt();
            vec.iter().map(|&v| (v as f64 / norm) as f32).collect()
        };

        if vectors.len() < 64 {
            vectors.iter().map(normalize_one).collect()
        } else {
            vectors.par_iter().map(normalize_one).collect()
        }
    })
}

/// Batch dot product: one query against N candidate vectors (pre-normalized).
///
/// For pre-normalized vectors, dot product equals cosine similarity.
/// Skips norm computation for faster scoring.
///
/// Accepts numpy arrays via zero-copy. Releases GIL during computation.
/// Returns `(index, similarity)` pairs above `threshold`, sorted descending.
#[pyfunction]
#[pyo3(signature = (query, candidates, threshold))]
pub fn batch_dot_product(
    py: Python<'_>,
    query: PyReadonlyArray1<'_, f32>,
    candidates: PyReadonlyArray2<'_, f32>,
    threshold: f32,
) -> Vec<(usize, f32)> {
    let q_array = query.as_array();
    let c_array = candidates.as_array();

    let q_owned = q_array.to_owned();
    let c_owned = c_array.to_owned();

    py.allow_threads(|| {
        let q = q_owned.as_slice().unwrap();
        let n_candidates = c_owned.nrows();

        let mut results: Vec<(usize, f32)> = (0..n_candidates)
            .into_par_iter()
            .filter_map(|i| {
                let row = c_owned.row(i);
                let dot: f64 = q
                    .iter()
                    .zip(row.iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum();
                let sim = dot as f32;
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
    fn test_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(v.clone(), v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        assert!(cosine_similarity(v1, v2).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector() {
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(v1, v2), 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(v1, v2), 0.0);
    }

    #[test]
    fn test_opposite_vectors() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        let sim = cosine_similarity(v1, v2);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_unit_vector() {
        // Already unit-length vector should stay the same
        let vectors = vec![vec![1.0, 0.0, 0.0]];
        let result = normalize_one(&vectors[0]);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!(result[1].abs() < 1e-6);
        assert!(result[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize_arbitrary_vector() {
        let vectors = vec![vec![3.0, 4.0]];
        let result = normalize_one(&vectors[0]);
        // norm = 5.0, so normalized = [0.6, 0.8]
        assert!((result[0] - 0.6).abs() < 1e-5);
        assert!((result[1] - 0.8).abs() < 1e-5);
        // Check that L2 norm is 1.0
        let norm: f64 = result.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let vectors = vec![vec![0.0, 0.0, 0.0]];
        let result = normalize_one(&vectors[0]);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dot_product_normalized_equals_cosine() {
        // For normalized vectors, dot product should equal cosine similarity
        let v1 = vec![3.0f32, 4.0];
        let v2 = vec![1.0f32, 2.0];
        let cosine = cosine_similarity(v1.clone(), v2.clone());

        // Normalize
        let n1 = normalize_one(&v1);
        let n2 = normalize_one(&v2);

        // Dot product of normalized vectors
        let dot: f32 = n1.iter().zip(n2.iter()).map(|(a, b)| a * b).sum();
        assert!((dot - cosine).abs() < 1e-5);
    }

    fn normalize_one(vec: &Vec<f32>) -> Vec<f32> {
        let sq_sum: f64 = vec.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if sq_sum == 0.0 {
            return vec.clone();
        }
        let norm = sq_sum.sqrt();
        vec.iter().map(|&v| (v as f64 / norm) as f32).collect()
    }
}

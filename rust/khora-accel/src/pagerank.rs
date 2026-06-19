//! PageRank on sparse weighted graphs with PyO3 bindings.
//!
//! Replicates the algorithm from `skeleton.py` (lines 551-601) for
//! skeleton indexing, where PageRank identifies ~10% core chunks for
//! LLM extraction.

use pyo3::prelude::*;

/// Compute PageRank scores on a weighted directed graph.
///
/// Matches the algorithm in `skeleton.py:_calculate_pagerank` and also
/// supports Personalized PageRank (Issue #597) via the optional
/// `personalization` argument. When `personalization` is `None` or
/// L1-normalizes to the uniform distribution, this reduces to standard
/// PageRank — preserving the behaviour every existing caller relies on.
///
/// PPR formula: `r = (1 - d) * p + d * M^T r`, where `p` is the
/// personalization vector (L1-normalized). Existing PageRank: `p = [1/n, ...]`.
///
/// - Uniform initialization: scores start at `p` when provided, else `1/n`
/// - Weighted contributions: `score[src] * weight / out_degree[src]`
/// - Damping: `new_score = (1-d) * p[node] + d * contrib`
/// - Converges when total absolute diff < `tol`
///
/// # Arguments
/// * `n` — number of nodes (IDs are `0..n`)
/// * `edges` — `(src, dst, weight)` triples (directed, pre-indexed)
/// * `damping` — damping factor (typically 0.85)
/// * `max_iter` — maximum iterations (typically 100)
/// * `tol` — convergence threshold (typically 1e-6)
/// * `personalization` — optional L1-normalizable seed distribution of length `n`.
///   Negatives are clipped to 0; if the resulting sum is 0, falls back to uniform.
///
/// # Returns
/// `Vec<f64>` of length `n` with PageRank scores indexed by node ID.
#[pyfunction]
#[pyo3(signature = (n, edges, damping, max_iter, tol, personalization=None))]
pub fn pagerank(
    py: Python<'_>,
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    damping: f64,
    max_iter: usize,
    tol: f64,
    personalization: Option<Vec<f64>>,
) -> Vec<f64> {
    py.detach(|| pagerank_inner(n, &edges, damping, max_iter, tol, personalization))
}

/// Pure-Rust implementation (no Python dependency), used by both PyO3 binding and tests.
pub fn pagerank_inner(
    n: usize,
    edges: &[(usize, usize, f64)],
    damping: f64,
    max_iter: usize,
    tol: f64,
    personalization: Option<Vec<f64>>,
) -> Vec<f64> {
    {
        if n == 0 {
            return Vec::new();
        }

        // Resolve the teleport distribution `p`. Validation rules:
        //   - length mismatch → fall back to uniform (defensive; never crashes a query)
        //   - negative entries → clipped to 0
        //   - all-zero after clipping → fall back to uniform
        let uniform = 1.0 / n as f64;
        let p: Vec<f64> = match personalization {
            Some(v) if v.len() == n => {
                let clipped: Vec<f64> = v.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect();
                let sum: f64 = clipped.iter().sum();
                if sum > 0.0 {
                    clipped.iter().map(|x| x / sum).collect()
                } else {
                    vec![uniform; n]
                }
            }
            _ => vec![uniform; n],
        };

        // Build incoming adjacency list: for each dst, store (src, weight)
        let mut incoming: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        // Accumulate out-degree (sum of outgoing weights) per node
        let mut out_degree: Vec<f64> = vec![0.0; n];

        for &(src, dst, weight) in edges {
            if src < n && dst < n {
                incoming[dst].push((src, weight));
                out_degree[src] += weight;
            }
        }

        // Initialize from the personalization distribution so the first
        // iteration already biases toward the seed neighbourhood.
        let mut scores: Vec<f64> = p.clone();

        for _iter in 0..max_iter {
            let mut new_scores: Vec<f64> = vec![0.0; n];
            let mut diff = 0.0f64;

            for node in 0..n {
                let mut contrib = 0.0f64;
                for &(src, weight) in &incoming[node] {
                    if out_degree[src] > 0.0 {
                        contrib += scores[src] * weight / out_degree[src];
                    }
                }
                let new_score = (1.0 - damping) * p[node] + damping * contrib;
                diff += (new_score - scores[node]).abs();
                new_scores[node] = new_score;
            }

            scores = new_scores;
            if diff < tol {
                break;
            }
        }

        scores
    }
}

/// Build chunk-to-chunk edges from keyword memberships (co-occurrence graph).
///
/// Replicates `skeleton.py:_build_chunk_edges` (lines 531-548).
/// For each keyword, creates bidirectional edges among all chunks sharing
/// that keyword, weighted by the keyword's IDF score.
///
/// # Arguments
/// * `n_chunks` — total number of chunks (used for validation)
/// * `keyword_chunk_ids` — for each keyword, the list of chunk indices that contain it
/// * `idf_scores` — IDF score for each keyword (same length as `keyword_chunk_ids`)
///
/// # Returns
/// Flat edge list of `(src, dst, weight)` triples (bidirectional).
#[pyfunction]
pub fn build_chunk_edges(
    py: Python<'_>,
    n_chunks: usize,
    keyword_chunk_ids: Vec<Vec<usize>>,
    idf_scores: Vec<f64>,
) -> Vec<(usize, usize, f64)> {
    py.detach(|| {
        let mut edges = Vec::new();

        for (keyword_idx, chunk_ids) in keyword_chunk_ids.iter().enumerate() {
            let weight = idf_scores.get(keyword_idx).copied().unwrap_or(0.0);
            let len = chunk_ids.len();
            for i in 0..len {
                let cid1 = chunk_ids[i];
                if cid1 >= n_chunks {
                    continue;
                }
                for j in (i + 1)..len {
                    let cid2 = chunk_ids[j];
                    if cid2 >= n_chunks {
                        continue;
                    }
                    // Bidirectional edges, matching Python implementation
                    edges.push((cid1, cid2, weight));
                    edges.push((cid2, cid1, weight));
                }
            }
        }

        edges
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph_convergence() {
        // Simple 3-node cycle: 0 → 1, 1 → 2, 2 → 0
        let edges = vec![
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
        ];
        let scores = pagerank_inner(3, &edges, 0.85, 100, 1e-6, None);
        assert_eq!(scores.len(), 3);
        // All nodes should have equal scores in a symmetric cycle
        assert!((scores[0] - scores[1]).abs() < 1e-4);
        assert!((scores[1] - scores[2]).abs() < 1e-4);
    }

    #[test]
    fn test_isolated_nodes() {
        // 3 nodes, no edges → all get base score (1-d)/n
        let scores = pagerank_inner(3, &[], 0.85, 100, 1e-6, None);
        assert_eq!(scores.len(), 3);
        let expected = 0.15 / 3.0;
        for s in &scores {
            assert!((*s - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_empty_graph() {
        let scores = pagerank_inner(0, &[], 0.85, 100, 1e-6, None);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_star_graph() {
        // All nodes point to node 0
        let edges = vec![
            (1, 0, 1.0),
            (2, 0, 1.0),
            (3, 0, 1.0),
        ];
        let scores = pagerank_inner(4, &edges, 0.85, 100, 1e-6, None);
        assert_eq!(scores.len(), 4);
        // Node 0 should have the highest score
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
        assert!(scores[0] > scores[3]);
    }

    #[test]
    fn test_personalized_seeded_chain() {
        // PPR seeded on node 0 of a 3-node chain (0 → 1 → 2): seed dominates,
        // then 1, then 2 — the depth-decay property the retriever needs.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let personalization = Some(vec![1.0, 0.0, 0.0]);
        let scores = pagerank_inner(3, &edges, 0.85, 200, 1e-8, personalization);
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_uniform_personalization_matches_default() {
        // Explicit uniform p must produce the same scores as None.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let default_scores = pagerank_inner(3, &edges, 0.85, 200, 1e-9, None);
        let uniform_scores = pagerank_inner(3, &edges, 0.85, 200, 1e-9, Some(vec![1.0 / 3.0; 3]));
        for i in 0..3 {
            assert!((default_scores[i] - uniform_scores[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_personalization_length_mismatch_falls_back_to_uniform() {
        // Wrong-length p → uniform; must not panic.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let scores = pagerank_inner(3, &edges, 0.85, 100, 1e-6, Some(vec![1.0, 0.0]));
        // Symmetric cycle + uniform → equal scores
        assert!((scores[0] - scores[1]).abs() < 1e-4);
        assert!((scores[1] - scores[2]).abs() < 1e-4);
    }
}

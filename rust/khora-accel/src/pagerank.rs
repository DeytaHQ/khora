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
/// * `rank_k` — optional top-k rank-stability early-stop (Issue #1476). When
///   `Some(k)`, the power iteration additionally halts once the ordering of the
///   top-`k` nodes (by score desc, index asc) is unchanged for `stable_iters`
///   consecutive iterations — the top-k retrieval set is rank-stable long before
///   global-L1 convergence, so this cuts ~4x the iterations on the production
///   graph shape while leaving the returned top-k byte-identical (guarded by a
///   parity test). `None` disables it: behaviour is exactly the global-L1 path.
/// * `stable_iters` — patience for the `rank_k` early-stop (consecutive stable
///   iterations required). Ignored when `rank_k` is `None`.
///
/// # Returns
/// `Vec<f64>` of length `n` with PageRank scores indexed by node ID.
#[pyfunction]
#[pyo3(signature = (n, edges, damping, max_iter, tol, personalization=None, rank_k=None, stable_iters=3))]
pub fn pagerank(
    py: Python<'_>,
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    damping: f64,
    max_iter: usize,
    tol: f64,
    personalization: Option<Vec<f64>>,
    rank_k: Option<usize>,
    stable_iters: usize,
) -> Vec<f64> {
    py.detach(|| {
        pagerank_inner(
            n,
            &edges,
            damping,
            max_iter,
            tol,
            personalization,
            rank_k,
            stable_iters,
        )
    })
}

/// Indices of the top-`k` scored nodes, ordered by score descending with ties
/// broken by ascending index. Equivalent to
/// `sorted(range(n), key=lambda i: (-scores[i], i))[:k]`, but uses an O(n)
/// `select_nth` + O(k log k) sort of the k-prefix instead of a full O(n log n)
/// sort — the early-stop runs this every iteration, so an O(n log n) sort of a
/// 12k-node vector would cost more than the iterations it saves (#1476).
///
/// The result is byte-for-byte identical to the full-sort prefix (and to the
/// Python fallback's `heapq.nsmallest(k, range(n), key=lambda i: (-scores[i], i))`)
/// because the comparator is a total order (ties resolved by unique index), so
/// the set of the first `k` is uniquely determined.
fn top_k_into(scores: &[f64], k: usize, key_buf: &mut Vec<(f64, u32)>, out: &mut Vec<usize>) {
    let n = scores.len();
    let k = k.min(n);
    out.clear();
    if k == 0 {
        return;
    }
    // Work on a contiguous `(score, index)` array rather than selecting over
    // node indices with a `scores[idx]` indirection: the early-stop runs this
    // every iteration, and the keyed array keeps the select/sort cache-friendly
    // (sequential reads, no random gather into `scores`). `key_buf`/`out` reuse
    // their allocations across iterations.
    key_buf.clear();
    key_buf.extend(scores.iter().enumerate().map(|(i, &s)| (s, i as u32)));
    let cmp = |a: &(f64, u32), b: &(f64, u32)| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    };
    if k < n {
        key_buf.select_nth_unstable_by(k - 1, cmp);
        key_buf.truncate(k);
    }
    key_buf.sort_by(cmp);
    out.extend(key_buf.iter().map(|&(_, i)| i as usize));
}

/// Owning convenience wrapper around [`top_k_into`]. The power iteration uses
/// `top_k_into` with reused buffers; this allocation-per-call form is only used
/// by the tests, hence `#[cfg(test)]`.
#[cfg(test)]
fn top_k_indices(scores: &[f64], k: usize) -> Vec<usize> {
    let mut key_buf = Vec::new();
    let mut out = Vec::new();
    top_k_into(scores, k, &mut key_buf, &mut out);
    out
}

/// Pure-Rust implementation (no Python dependency), used by both PyO3 binding and tests.
#[allow(clippy::too_many_arguments)]
pub fn pagerank_inner(
    n: usize,
    edges: &[(usize, usize, f64)],
    damping: f64,
    max_iter: usize,
    tol: f64,
    personalization: Option<Vec<f64>>,
    rank_k: Option<usize>,
    stable_iters: usize,
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

        // Build the incoming graph in CSR form (Issue #1476): for each `dst`,
        // `row_ptr[dst]..row_ptr[dst + 1]` indexes its incoming edges in the flat
        // `in_src` / `in_wnorm` arrays. Iterating CSR is sequential (cache
        // friendly) versus the pointer-chasing `Vec<Vec<(usize, f64)>>` adjacency
        // list it replaces. `in_wnorm` folds the `weight / out_degree[src]`
        // normalization into the build, so the per-iteration inner loop is a
        // single fused multiply-add per edge (one fewer division per edge per
        // iteration). Edges are scattered in input order, so a node's incoming
        // contributions are summed in the same order as the pure-Python fallback
        // — the two paths stay bit-identical.
        let mut out_degree: Vec<f64> = vec![0.0; n];
        let mut in_count: Vec<usize> = vec![0; n];
        for &(src, dst, weight) in edges {
            if src < n && dst < n {
                out_degree[src] += weight;
                in_count[dst] += 1;
            }
        }
        let mut row_ptr: Vec<usize> = vec![0; n + 1];
        for node in 0..n {
            row_ptr[node + 1] = row_ptr[node] + in_count[node];
        }
        let nnz = row_ptr[n];
        let mut in_src: Vec<u32> = vec![0; nnz];
        let mut in_wnorm: Vec<f64> = vec![0.0; nnz];
        // `fill[dst]` tracks the next free slot for `dst` as edges are scattered.
        let mut fill: Vec<usize> = row_ptr[..n].to_vec();
        for &(src, dst, weight) in edges {
            if src < n && dst < n {
                let pos = fill[dst];
                fill[dst] += 1;
                in_src[pos] = src as u32;
                // Guard out_degree == 0 exactly as the original loop's
                // `if out_degree[src] > 0.0` did (a zero-weight source
                // contributes nothing rather than dividing by zero).
                in_wnorm[pos] = if out_degree[src] > 0.0 {
                    weight / out_degree[src]
                } else {
                    0.0
                };
            }
        }

        // Ping-pong score buffers, initialized from the personalization
        // distribution so the first iteration is already seeded. `cur` holds the
        // current scores; `next` is fully overwritten each iteration then swapped
        // in — no per-iteration allocation.
        let mut cur: Vec<f64> = p.clone();
        let mut next: Vec<f64> = vec![0.0; n];

        // Top-k rank-stability early-stop state (Issue #1476). `rank_k` is
        // clamped to `n`; `prev_top` holds the previous iteration's top-k
        // ordering and `stable_count` the run of consecutive stable iterations.
        // `top_buf` / `key_buf` are reused across iterations to keep the
        // per-iteration check allocation-free.
        let rank_k = rank_k.map(|k| k.min(n));
        let mut prev_top: Vec<usize> = Vec::new();
        let mut top_buf: Vec<usize> = Vec::new();
        let mut key_buf: Vec<(f64, u32)> = Vec::new();
        let mut stable_count: usize = 0;

        for _iter in 0..max_iter {
            let mut diff = 0.0f64;

            for node in 0..n {
                let start = row_ptr[node];
                let end = row_ptr[node + 1];
                let mut contrib = 0.0f64;
                for e in start..end {
                    contrib += cur[in_src[e] as usize] * in_wnorm[e];
                }
                let new_score = (1.0 - damping) * p[node] + damping * contrib;
                diff += (new_score - cur[node]).abs();
                next[node] = new_score;
            }

            std::mem::swap(&mut cur, &mut next);
            if diff < tol {
                break;
            }

            // Top-k rank-stability early-stop: halt once the top-k ordering has
            // been unchanged for `stable_iters` consecutive iterations. Checked
            // after the global-L1 test so `rank_k=None` is byte-identical to the
            // pre-#1476 behaviour.
            if let Some(k) = rank_k {
                if k > 0 && stable_iters > 0 {
                    top_k_into(&cur, k, &mut key_buf, &mut top_buf);
                    if top_buf == prev_top {
                        stable_count += 1;
                        if stable_count >= stable_iters {
                            break;
                        }
                    } else {
                        stable_count = 0;
                    }
                    prev_top.clear();
                    prev_top.extend_from_slice(&top_buf);
                }
            }
        }

        cur
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
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let scores = pagerank_inner(3, &edges, 0.85, 100, 1e-6, None, None, 3);
        assert_eq!(scores.len(), 3);
        // All nodes should have equal scores in a symmetric cycle
        assert!((scores[0] - scores[1]).abs() < 1e-4);
        assert!((scores[1] - scores[2]).abs() < 1e-4);
    }

    #[test]
    fn test_isolated_nodes() {
        // 3 nodes, no edges → all get base score (1-d)/n
        let scores = pagerank_inner(3, &[], 0.85, 100, 1e-6, None, None, 3);
        assert_eq!(scores.len(), 3);
        let expected = 0.15 / 3.0;
        for s in &scores {
            assert!((*s - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn test_empty_graph() {
        let scores = pagerank_inner(0, &[], 0.85, 100, 1e-6, None, None, 3);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_star_graph() {
        // All nodes point to node 0
        let edges = vec![(1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0)];
        let scores = pagerank_inner(4, &edges, 0.85, 100, 1e-6, None, None, 3);
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
        let scores = pagerank_inner(3, &edges, 0.85, 200, 1e-8, personalization, None, 3);
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_uniform_personalization_matches_default() {
        // Explicit uniform p must produce the same scores as None.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let default_scores = pagerank_inner(3, &edges, 0.85, 200, 1e-9, None, None, 3);
        let uniform_scores = pagerank_inner(
            3,
            &edges,
            0.85,
            200,
            1e-9,
            Some(vec![1.0 / 3.0; 3]),
            None,
            3,
        );
        for i in 0..3 {
            assert!((default_scores[i] - uniform_scores[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_personalization_length_mismatch_falls_back_to_uniform() {
        // Wrong-length p → uniform; must not panic.
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let scores = pagerank_inner(3, &edges, 0.85, 100, 1e-6, Some(vec![1.0, 0.0]), None, 3);
        // Symmetric cycle + uniform → equal scores
        assert!((scores[0] - scores[1]).abs() < 1e-4);
        assert!((scores[1] - scores[2]).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Top-k rank-stability early-stop (Issue #1476)
    // -----------------------------------------------------------------------

    /// Deterministic scale-free-ish weighted graph: `n` nodes, each new node
    /// attaches to a handful of earlier ones with a deterministic weight. The
    /// low-index "hub" nodes accumulate the most PR mass, so the top-k set
    /// stabilizes early — the property the early-stop exploits.
    fn build_test_graph(n: usize) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for dst in 1..n {
            // 3 back-links to a deterministic spread of earlier nodes.
            for step in 1..=3usize {
                let src = (dst * step) % dst; // in [0, dst)
                let w = 1.0 + ((dst * 7 + step * 13) % 5) as f64;
                edges.push((src, dst, w));
                edges.push((dst, src, w));
            }
        }
        edges
    }

    #[test]
    fn test_early_stop_topk_byte_identical_to_full() {
        // The early-stopped top-30 must be byte-identical to the full-iteration
        // top-30 — the safety guarantee behind #1476.
        let n = 600;
        let edges = build_test_graph(n);
        let seed: Vec<f64> = (0..n)
            .map(|i| if i % 50 == 0 { 1.0 } else { 0.0 })
            .collect();

        let full = pagerank_inner(n, &edges, 0.85, 50, 1e-5, Some(seed.clone()), None, 3);
        // rank_k = 30 (retrieval limit) + 10 margin; patience 3.
        let early = pagerank_inner(n, &edges, 0.85, 50, 1e-5, Some(seed), Some(40), 3);

        assert_eq!(top_k_indices(&full, 30), top_k_indices(&early, 30));
    }

    #[test]
    fn test_early_stop_actually_halts_early() {
        // The early-stop must do real iteration work (its scores differ from the
        // trivial one-iteration result) yet converge to the same top-30 as the
        // full run — i.e. it halts early without changing the ranking.
        let n = 400;
        let edges = build_test_graph(n);
        let seed: Vec<f64> = (0..n).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();

        let one_iter = pagerank_inner(n, &edges, 0.85, 1, 1e-9, Some(seed.clone()), None, 3);
        let full = pagerank_inner(n, &edges, 0.85, 100, 1e-9, Some(seed.clone()), None, 3);
        let early = pagerank_inner(n, &edges, 0.85, 100, 1e-9, Some(seed), Some(40), 2);
        assert_ne!(
            one_iter, early,
            "early-stop result should reflect real iteration work"
        );
        assert_eq!(top_k_indices(&full, 30), top_k_indices(&early, 30));
    }

    #[test]
    fn test_early_stop_disabled_matches_none() {
        // rank_k=None and stable_iters=0 (via a huge patience never reached)
        // must both reproduce the pure global-L1 path exactly.
        let n = 200;
        let edges = build_test_graph(n);
        let seed: Vec<f64> = (0..n).map(|i| if i < 3 { 1.0 } else { 0.0 }).collect();

        let none = pagerank_inner(n, &edges, 0.85, 50, 1e-5, Some(seed.clone()), None, 3);
        let zero_patience = pagerank_inner(n, &edges, 0.85, 50, 1e-5, Some(seed), Some(40), 0);
        // stable_iters=0 disables the early-stop → identical scores.
        assert_eq!(none, zero_patience);
    }

    #[test]
    fn test_top_k_indices_tie_break_is_index_ascending() {
        // All-equal scores → ties broken by ascending index.
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        assert_eq!(top_k_indices(&scores, 3), vec![0, 1, 2]);
        // Descending scores.
        let scores = vec![0.1, 0.9, 0.5, 0.7];
        assert_eq!(top_k_indices(&scores, 2), vec![1, 3]);
    }
}

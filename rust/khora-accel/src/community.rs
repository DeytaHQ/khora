//! Community detection via Louvain-style modularity optimization.
//!
//! Implements a simplified Louvain algorithm where each node is greedily
//! moved to the neighboring community offering the maximum modularity gain.

use hashbrown::HashMap;
use pyo3::prelude::*;

/// Detect communities using Louvain-style modularity optimization.
///
/// Each node starts in its own community. Iteratively, each node is moved
/// to the neighbor community that maximises the modularity gain:
///
///   ΔQ = k_i_in / m - γ · (Σ_tot · k_i) / (2m²)
///
/// where `k_i_in` is the sum of edge weights from node i to community C
/// (after temporarily removing i), `Σ_tot` is the total weighted degree of
/// community C (after removal), `k_i` is node i's weighted degree, and `m`
/// is half the total directed-edge weight (i.e. total undirected weight).
///
/// Iteration stops when no node changes community or `max_iter` passes
/// have been completed.
///
/// # Arguments
/// * `n` — number of nodes (IDs are `0..n`)
/// * `edges` — `(src, dst, weight)` triples (undirected — provide both directions)
/// * `resolution` — modularity resolution parameter γ (higher → smaller communities)
/// * `max_iter` — maximum optimization passes
///
/// # Returns
/// `Vec<i32>` of length `n`: community ID per node (0-indexed, -1 for isolated nodes).
#[pyfunction]
pub fn detect_communities(
    py: Python<'_>,
    n: usize,
    edges: Vec<(usize, usize, f64)>,
    resolution: f64,
    max_iter: usize,
) -> Vec<i32> {
    py.allow_threads(|| {
        if n == 0 {
            return Vec::new();
        }

        // Build adjacency list and compute node strengths (weighted degrees)
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut strengths: Vec<f64> = vec![0.0; n];
        let mut total_weight = 0.0f64;

        for &(src, dst, weight) in &edges {
            if src < n && dst < n && src != dst {
                adj[src].push((dst, weight));
                strengths[src] += weight;
                total_weight += weight;
            }
        }

        // Edges are provided in both directions; m counts each undirected edge once
        let m = total_weight / 2.0;

        if m == 0.0 {
            // No edges — all nodes are isolated
            return vec![-1i32; n];
        }

        // Initialise: each node is its own community
        let mut community: Vec<usize> = (0..n).collect();
        // sigma_tot[c] = sum of weighted degrees of all nodes in community c
        let mut sigma_tot: Vec<f64> = strengths.clone();

        for _iter in 0..max_iter {
            let mut moved = false;

            for i in 0..n {
                if strengths[i] == 0.0 {
                    // Isolated node — skip
                    continue;
                }

                let ki = strengths[i];
                let ci = community[i];

                // Temporarily remove node i from its current community
                sigma_tot[ci] -= ki;

                // Sum edge weights from i to each neighboring community
                let mut k_in: HashMap<usize, f64> = HashMap::new();
                for &(nb, w) in &adj[i] {
                    let cnb = community[nb];
                    *k_in.entry(cnb).or_insert(0.0) += w;
                }

                // Compute gain of staying in ci (Σ_tot already excludes i)
                let k_i_in_ci = k_in.get(&ci).copied().unwrap_or(0.0);
                let gain_ci = k_i_in_ci / m
                    - resolution * sigma_tot[ci] * ki / (2.0 * m * m);

                // Search neighbor communities for a better placement
                let mut best_c = ci;
                let mut best_gain = gain_ci;

                for (&c, &k_i_in_c) in &k_in {
                    if c == ci {
                        continue;
                    }
                    let gain = k_i_in_c / m
                        - resolution * sigma_tot[c] * ki / (2.0 * m * m);
                    if gain > best_gain {
                        best_gain = gain;
                        best_c = c;
                    }
                }

                // Place node i in the winning community
                community[i] = best_c;
                sigma_tot[best_c] += ki;

                if best_c != ci {
                    moved = true;
                }
            }

            if !moved {
                break;
            }
        }

        // Remap internal community IDs to compact 0-indexed IDs
        let mut id_map: HashMap<usize, i32> = HashMap::new();
        let mut next_id = 0i32;
        let mut result: Vec<i32> = vec![-1i32; n];

        for i in 0..n {
            if strengths[i] > 0.0 {
                let c = community[i];
                let mapped = *id_map.entry(c).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                result[i] = mapped;
            }
        }

        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let result = detect_communities(py, 0, vec![], 1.0, 10);
            assert!(result.is_empty());
        });
    }

    #[test]
    fn test_single_component() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Triangle: 0-1-2, all strongly connected
            let edges = vec![
                (0, 1, 1.0),
                (1, 0, 1.0),
                (1, 2, 1.0),
                (2, 1, 1.0),
                (0, 2, 1.0),
                (2, 0, 1.0),
            ];
            let result = detect_communities(py, 3, edges, 1.0, 10);
            assert_eq!(result.len(), 3);
            // All nodes should end up in the same community
            assert_eq!(result[0], result[1]);
            assert_eq!(result[1], result[2]);
            assert!(result[0] >= 0);
        });
    }

    #[test]
    fn test_two_clear_clusters() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Two dense triangles {0,1,2} and {3,4,5} with a weak bridge
            let edges = vec![
                // Cluster A
                (0, 1, 10.0),
                (1, 0, 10.0),
                (1, 2, 10.0),
                (2, 1, 10.0),
                (0, 2, 10.0),
                (2, 0, 10.0),
                // Cluster B
                (3, 4, 10.0),
                (4, 3, 10.0),
                (4, 5, 10.0),
                (5, 4, 10.0),
                (3, 5, 10.0),
                (5, 3, 10.0),
                // Weak inter-cluster bridge
                (2, 3, 0.1),
                (3, 2, 0.1),
            ];
            let result = detect_communities(py, 6, edges, 1.0, 10);
            assert_eq!(result.len(), 6);
            // Each cluster forms one community
            assert_eq!(result[0], result[1]);
            assert_eq!(result[1], result[2]);
            assert_eq!(result[3], result[4]);
            assert_eq!(result[4], result[5]);
            // The two clusters must be distinct communities
            assert_ne!(result[0], result[3]);
        });
    }

    #[test]
    fn test_disconnected_components() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Two disconnected pairs {0,1} and {2,3}, plus isolated node 4
            let edges = vec![
                (0, 1, 1.0),
                (1, 0, 1.0),
                (2, 3, 1.0),
                (3, 2, 1.0),
            ];
            let result = detect_communities(py, 5, edges, 1.0, 10);
            assert_eq!(result.len(), 5);
            // Each pair shares a community
            assert_eq!(result[0], result[1]);
            assert_eq!(result[2], result[3]);
            // The two pairs are in different communities
            assert_ne!(result[0], result[2]);
            // Isolated node has no community
            assert_eq!(result[4], -1);
        });
    }
}

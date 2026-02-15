use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_pagerank(c: &mut Criterion) {
    let n = 100;
    let damping = 0.85;
    let max_iter = 100;
    let tol = 1e-6;

    // Build a graph: each node i links to (i+1)%n and (i+2)%n with weight 1.0
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n, 1.0));
        edges.push((i, (i + 2) % n, 1.0));
    }

    // Pre-build adjacency lists (mirrors pagerank.rs logic)
    let mut incoming: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut out_degree: Vec<f64> = vec![0.0; n];
    for &(src, dst, weight) in &edges {
        incoming[dst].push((src, weight));
        out_degree[src] += weight;
    }

    c.bench_function("pagerank_100nodes", |bench| {
        bench.iter(|| {
            let base = (1.0 - damping) / n as f64;
            let mut scores: Vec<f64> = vec![1.0 / n as f64; n];

            for _ in 0..max_iter {
                let mut new_scores: Vec<f64> = vec![0.0; n];
                let mut diff = 0.0f64;

                for node in 0..n {
                    let mut contrib = 0.0f64;
                    for &(src, weight) in &incoming[node] {
                        if out_degree[src] > 0.0 {
                            contrib += scores[src] * weight / out_degree[src];
                        }
                    }
                    let new_score = base + damping * contrib;
                    diff += (new_score - scores[node]).abs();
                    new_scores[node] = new_score;
                }

                scores = new_scores;
                if diff < tol {
                    break;
                }
            }
            black_box(&scores);
        })
    });
}

criterion_group!(benches, bench_pagerank);
criterion_main!(benches);

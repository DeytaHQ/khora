use criterion::{criterion_group, criterion_main, Criterion};

fn bench_pagerank_placeholder(_c: &mut Criterion) {
    // TODO: Add PageRank benchmarks once the crate compiles
}

criterion_group!(benches, bench_pagerank_placeholder);
criterion_main!(benches);

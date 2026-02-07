use criterion::{criterion_group, criterion_main, Criterion};

fn bench_bm25_placeholder(_c: &mut Criterion) {
    // TODO: Add BM25 benchmarks once the crate compiles
}

criterion_group!(benches, bench_bm25_placeholder);
criterion_main!(benches);

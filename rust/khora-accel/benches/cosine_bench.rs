use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 384, 768, 1536] {
        let v1: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        group.bench_with_input(BenchmarkId::new("single", dim), &dim, |b, _| {
            b.iter(|| {
                let (mut dot, mut n1, mut n2) = (0.0f32, 0.0f32, 0.0f32);
                for (a, b_val) in v1.iter().zip(v2.iter()) {
                    dot += a * b_val;
                    n1 += a * a;
                    n2 += b_val * b_val;
                }
                black_box(dot / (n1.sqrt() * n2.sqrt()))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cosine_similarity);
criterion_main!(benches);

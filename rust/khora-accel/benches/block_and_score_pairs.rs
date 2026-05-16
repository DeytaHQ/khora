//! Benchmark `block_and_score_pairs` vs the naive `pairwise_cosine_above_threshold`.
//!
//! The kernel is intended for dream-phase cross-batch entity resolution
//! at namespace scale (N >= 10k). At those sizes the unblocked path is
//! ~30s; we want the blocked path under 1s. The bench at N=10k is a
//! tractable proxy that exercises the same hot path.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hashbrown::{HashMap, HashSet};

fn make_corpus(n: usize, d: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<String>) {
    // Cheap deterministic pseudo-random embeddings + names. We pre-normalise
    // so cosine == dot.
    let mut s = seed;
    let next = |s: &mut u64| -> f32 {
        // xorshift64*
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        let v = (*s as f64 / u64::MAX as f64) * 2.0 - 1.0;
        v as f32
    };
    let mut embeddings = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row: Vec<f32> = (0..d).map(|_| next(&mut s)).collect();
        let norm: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v = (*v as f64 / norm) as f32;
            }
        }
        embeddings.push(row);
    }
    // Realistic name distribution: ~50 distinct first-tokens, names with 2-3 tokens.
    let first_tokens = [
        "acme", "zenith", "atlas", "delta", "globex", "initech", "umbrella", "wayne", "stark",
        "wonka", "tyrell", "soylent", "weyland", "yutani", "massive", "encom", "cyberdyne",
        "oscorp", "lexcorp", "lumon", "pied", "piper", "hooli", "raviga", "endframe", "aviato",
        "biotechnica", "arasaka", "nakatomi", "veridian", "primatech", "rekall", "spacely",
        "vandelay", "krusty", "duff", "globo", "planet", "express", "monarch", "abstergo",
        "hyperion", "merrick", "buynlarge", "engulf", "devour", "morbo", "spheris", "nimbus",
        "solaris",
    ];
    let suffix_tokens = ["corp", "inc", "holdings", "group", "labs", "industries", "co"];
    let mut names = Vec::with_capacity(n);
    for i in 0..n {
        let f = first_tokens[i % first_tokens.len()];
        let sfx = suffix_tokens[(i / first_tokens.len()) % suffix_tokens.len()];
        names.push(format!("{f} {sfx} {i}"));
    }
    (embeddings, names)
}

const TOKEN_PREFIX_LEN: usize = 3;

fn tokenize(name: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in name.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                current.push(low);
            }
        } else if !current.is_empty() {
            if current.chars().count() >= 2 {
                tokens.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if !current.is_empty() && current.chars().count() >= 2 {
        tokens.push(current);
    }
    tokens
}

fn prefix_key(token: &str) -> String {
    token.chars().take(TOKEN_PREFIX_LEN).collect()
}

/// Reference implementation of the blocked kernel. Mirrors blocking.rs but
/// avoids the PyO3 wrapper so we can bench from native Rust.
fn blocked_score(
    embeddings: &[Vec<f32>],
    names: &[String],
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let n = embeddings.len();
    let row_keys: Vec<HashSet<String>> = names
        .iter()
        .map(|name| {
            let mut keys = HashSet::new();
            for tok in tokenize(name) {
                keys.insert(prefix_key(&tok));
            }
            keys
        })
        .collect();
    let mut inverted: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, keys) in row_keys.iter().enumerate() {
        for k in keys {
            inverted.entry(k.clone()).or_default().push(i);
        }
    }
    let mut out = Vec::new();
    for i in 0..n {
        let mut candidates: HashSet<usize> = HashSet::new();
        for k in &row_keys[i] {
            if let Some(rows) = inverted.get(k) {
                for &j in rows {
                    if j > i {
                        candidates.insert(j);
                    }
                }
            }
        }
        for j in candidates {
            let row_i = &embeddings[i];
            let row_j = &embeddings[j];
            let dot: f64 = row_i
                .iter()
                .zip(row_j.iter())
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum();
            let sim = dot as f32;
            if sim >= threshold {
                out.push((i, j, sim));
            }
        }
    }
    out
}

/// Reference naive implementation (every i<j scored).
fn naive_score(
    embeddings: &[Vec<f32>],
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let n = embeddings.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f64 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum();
            let sim = dot as f32;
            if sim >= threshold {
                out.push((i, j, sim));
            }
        }
    }
    out
}

fn bench_block_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_and_score_pairs");
    group.sample_size(10);

    for n in [1000usize, 5000, 10_000] {
        let (embeddings, names) = make_corpus(n, 128, 0xC0FFEE);
        group.bench_with_input(BenchmarkId::new("blocked", n), &n, |b, _| {
            b.iter(|| blocked_score(&embeddings, &names, 0.85))
        });
        if n <= 5000 {
            // Naive at N=10k takes minutes single-threaded — skip it.
            group.bench_with_input(BenchmarkId::new("naive", n), &n, |b, _| {
                b.iter(|| naive_score(&embeddings, 0.85))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_block_vs_naive);
criterion_main!(benches);

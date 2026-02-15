use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

fn bench_bm25_search(c: &mut Criterion) {
    // Build a simple BM25-like index with ~100 documents
    let documents: Vec<Vec<String>> = (0..100)
        .map(|i| {
            (0..50)
                .map(|j| format!("word{}", (i * 7 + j * 3) % 200))
                .collect()
        })
        .collect();

    // Build inverted index: term → Vec<(doc_idx, tf)>
    let mut inverted: HashMap<String, Vec<(usize, u32)>> = HashMap::new();
    let mut doc_lengths: Vec<u32> = Vec::new();
    let total_docs = documents.len();

    for (idx, tokens) in documents.iter().enumerate() {
        let mut freq: HashMap<&str, u32> = HashMap::new();
        for t in tokens {
            *freq.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, count) in &freq {
            inverted
                .entry(term.to_string())
                .or_default()
                .push((idx, *count));
        }
        doc_lengths.push(tokens.len() as u32);
    }

    let avg_dl: f32 = doc_lengths.iter().sum::<u32>() as f32 / total_docs as f32;
    let k1: f32 = 1.5;
    let b: f32 = 0.75;
    let n = total_docs as f32;

    let query_terms = vec![
        "word10".to_string(),
        "word42".to_string(),
        "word99".to_string(),
    ];

    c.bench_function("bm25_search_100docs", |bench| {
        bench.iter(|| {
            let mut doc_scores: HashMap<usize, f32> = HashMap::new();

            for term in &query_terms {
                if let Some(postings) = inverted.get(term) {
                    let df = postings.len() as f32;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    for &(doc_idx, tf) in postings {
                        let dl = doc_lengths[doc_idx] as f32;
                        let tf = tf as f32;
                        let num = tf * (k1 + 1.0);
                        let den = tf + k1 * (1.0 - b + b * dl / avg_dl);
                        *doc_scores.entry(doc_idx).or_insert(0.0) += idf * num / den;
                    }
                }
            }

            let mut scores: Vec<(usize, f32)> = doc_scores.drain().collect();
            scores.sort_by(|a, b_| b_.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(10);
            black_box(&scores);
        })
    });
}

criterion_group!(benches, bench_bm25_search);
criterion_main!(benches);

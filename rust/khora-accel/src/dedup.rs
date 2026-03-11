//! MinHash-based near-duplicate text chunk detection with PyO3 bindings.
//!
//! Detects near-duplicate text chunks BEFORE they are sent to LLM extraction,
//! reducing unnecessary LLM calls. Uses MinHash signatures with LSH banding
//! for efficient pairwise similarity estimation.

use hashbrown::HashMap;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Core hashing helpers
// ---------------------------------------------------------------------------

/// Hash a string slice with a given seed using SipHash (via DefaultHasher).
#[inline]
fn hash_with_seed(s: &str, seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    s.hash(&mut hasher);
    hasher.finish()
}

/// Extract word-level n-grams from text.
///
/// Tokenises on whitespace, lowercases, then yields sliding windows of `n` words
/// joined by a single space.
fn word_ngrams(text: &str, n: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        // If fewer words than n-gram size, return the whole text as one token
        let joined = words.join(" ").to_lowercase();
        if joined.is_empty() {
            return Vec::new();
        }
        return vec![joined];
    }
    words
        .windows(n)
        .map(|w| {
            w.iter()
                .map(|s| s.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// MinHash computation
// ---------------------------------------------------------------------------

/// Compute a MinHash signature for the given text.
///
/// Tokenises into word 3-grams, then for each of `num_perm` permutations
/// (seeded hash functions), takes the minimum hash value across all n-grams.
pub fn compute_minhash(text: &str, num_perm: usize) -> Vec<u64> {
    let ngrams = word_ngrams(text, 3);
    if ngrams.is_empty() {
        return vec![u64::MAX; num_perm];
    }

    let mut signature = Vec::with_capacity(num_perm);
    for seed in 0..num_perm as u64 {
        let min_hash = ngrams
            .iter()
            .map(|ng| hash_with_seed(ng, seed))
            .min()
            .unwrap_or(u64::MAX);
        signature.push(min_hash);
    }
    signature
}

/// Estimate Jaccard similarity from two MinHash signatures.
#[inline]
fn minhash_similarity(sig_a: &[u64], sig_b: &[u64]) -> f64 {
    debug_assert_eq!(sig_a.len(), sig_b.len());
    let matching = sig_a.iter().zip(sig_b.iter()).filter(|(a, b)| a == b).count();
    matching as f64 / sig_a.len() as f64
}

// ---------------------------------------------------------------------------
// LSH banding
// ---------------------------------------------------------------------------

/// Compute optimal number of bands for a given threshold and num_perm.
///
/// For LSH with `b` bands of `r` rows each (b * r = num_perm),
/// the probability of two items with Jaccard similarity `t` being
/// placed in the same bucket is approximately 1 - (1 - t^r)^b.
/// We pick `r` such that the approximate threshold (1/b)^(1/r) ≈ target.
fn optimal_bands(num_perm: usize, threshold: f64) -> usize {
    let mut best_bands = 1;
    let mut best_diff = f64::MAX;
    for bands in 1..=num_perm {
        let rows = num_perm / bands;
        if rows == 0 || bands * rows != num_perm {
            continue;
        }
        // Approximate threshold for this band configuration
        let approx_thresh = (1.0 / bands as f64).powf(1.0 / rows as f64);
        let diff = (approx_thresh - threshold).abs();
        if diff < best_diff {
            best_diff = diff;
            best_bands = bands;
        }
    }
    best_bands
}

/// Hash a band (sub-slice of signature) into a bucket key.
fn hash_band(band: &[u64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    band.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Deduplication entry point
// ---------------------------------------------------------------------------

/// Deduplicate text chunks using MinHash + LSH.
///
/// Returns a list of `(chunk_index, duplicate_of_index)` pairs.
/// `None` for `duplicate_of_index` means the chunk is unique (canonical).
///
/// Args:
///   chunks: text chunks to deduplicate.
///   threshold: Jaccard similarity threshold (default 0.85).
///   num_perm: number of MinHash permutations (default 64).
///
/// The algorithm:
/// 1. Compute MinHash signature for each chunk.
/// 2. Use LSH banding to find candidate pairs efficiently.
/// 3. For each candidate pair, verify with exact MinHash similarity.
/// 4. Mark later chunks as duplicates of earlier ones.
#[pyfunction]
#[pyo3(signature = (chunks, threshold=0.85, num_perm=64))]
pub fn deduplicate_chunks(
    py: Python<'_>,
    chunks: Vec<String>,
    threshold: f64,
    num_perm: usize,
) -> Vec<(usize, Option<usize>)> {
    py.detach(|| deduplicate_chunks_inner(&chunks, threshold, num_perm))
}

/// Pure-Rust implementation (no Python dependency), used by both PyO3 binding and tests.
pub fn deduplicate_chunks_inner(
    chunks: &[String],
    threshold: f64,
    num_perm: usize,
) -> Vec<(usize, Option<usize>)> {
    let n = chunks.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Compute signatures
    let signatures: Vec<Vec<u64>> = chunks.iter().map(|c| compute_minhash(c, num_perm)).collect();

    // 2. LSH banding for candidate generation
    let bands = optimal_bands(num_perm, threshold);
    let rows = num_perm / bands;

    // bucket_key → list of chunk indices that hashed into this bucket
    let mut buckets: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    for (idx, sig) in signatures.iter().enumerate() {
        for band_idx in 0..bands {
            let start = band_idx * rows;
            let end = start + rows;
            let band_hash = hash_band(&sig[start..end]);
            buckets
                .entry((band_idx, band_hash))
                .or_default()
                .push(idx);
        }
    }

    // 3. Collect candidate pairs from LSH buckets
    // Use a set to avoid duplicate pair checks
    let mut candidate_pairs: hashbrown::HashSet<(usize, usize)> = hashbrown::HashSet::new();
    for members in buckets.values() {
        if members.len() < 2 {
            continue;
        }
        for (i, &a) in members.iter().enumerate() {
            for &b in &members[i + 1..] {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                candidate_pairs.insert((lo, hi));
            }
        }
    }

    // 4. Verify candidates and build duplicate mapping
    // duplicate_of[i] = Some(j) means chunk i is a duplicate of chunk j (j < i)
    let mut duplicate_of: Vec<Option<usize>> = vec![None; n];

    // Sort candidates so we process in deterministic order
    let mut pairs: Vec<(usize, usize)> = candidate_pairs.into_iter().collect();
    pairs.sort_unstable();

    for (a, b) in pairs {
        // b > a always. If b is already marked as duplicate, skip.
        if duplicate_of[b].is_some() {
            continue;
        }
        // Find the canonical representative for a (follow chain)
        let canonical_a = {
            let mut c = a;
            while let Some(parent) = duplicate_of[c] {
                c = parent;
            }
            c
        };

        let sim = minhash_similarity(&signatures[canonical_a], &signatures[b]);
        if sim >= threshold {
            duplicate_of[b] = Some(canonical_a);
        }
    }

    // 5. Build result
    (0..n).map(|i| (i, duplicate_of[i])).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts_are_duplicates() {
        let text = "The quick brown fox jumps over the lazy dog and runs away fast".to_string();
        let chunks = vec![text.clone(), text.clone(), "something completely different here today now".to_string()];
        let result = deduplicate_chunks_inner(&chunks, 0.85, 128);

        assert_eq!(result.len(), 3);
        // First chunk is always canonical
        assert_eq!(result[0], (0, None));
        // Second chunk is duplicate of first
        assert_eq!(result[1], (1, Some(0)));
        // Third chunk is unique
        assert_eq!(result[2], (2, None));
    }

    #[test]
    fn test_completely_different_texts() {
        let chunks = vec![
            "The quick brown fox jumps over the lazy dog near the river bank".to_string(),
            "Quantum computing leverages superposition and entanglement for parallel processing tasks".to_string(),
            "Traditional Japanese gardens emphasize harmony between natural elements and architecture beautifully".to_string(),
        ];
        let result = deduplicate_chunks_inner(&chunks, 0.85, 128);

        // All should be unique
        for (i, (idx, dup)) in result.iter().enumerate() {
            assert_eq!(*idx, i);
            assert_eq!(*dup, None, "chunk {i} should be unique");
        }
    }

    #[test]
    fn test_similar_texts_small_edit() {
        let base = "The quick brown fox jumps over the lazy dog near the river bank in the morning light";
        let edited = "The quick brown fox leaps over the lazy dog near the river bank in the morning light";
        let chunks = vec![base.to_string(), edited.to_string()];
        let result = deduplicate_chunks_inner(&chunks, 0.5, 128);

        // With a low threshold, small edit should still be detected as duplicate
        assert_eq!(result[0], (0, None));
        assert_eq!(result[1].0, 1);
        assert!(
            result[1].1.is_some(),
            "similar text with small edit should be detected as duplicate at threshold 0.5"
        );
    }

    #[test]
    fn test_empty_input() {
        let chunks: Vec<String> = vec![];
        let result = deduplicate_chunks_inner(&chunks, 0.85, 128);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_chunk() {
        let chunks = vec!["hello world this is a test of the system".to_string()];
        let result = deduplicate_chunks_inner(&chunks, 0.85, 128);
        assert_eq!(result, vec![(0, None)]);
    }

    #[test]
    fn test_minhash_identical_signature() {
        let text = "the quick brown fox jumps over the lazy dog";
        let sig1 = compute_minhash(text, 128);
        let sig2 = compute_minhash(text, 128);
        let sim = minhash_similarity(&sig1, &sig2);
        assert!((sim - 1.0).abs() < f64::EPSILON, "identical text should have similarity 1.0");
    }

    #[test]
    fn test_minhash_different_signature() {
        let sig1 = compute_minhash("the quick brown fox jumps over the lazy dog near the old farm", 128);
        let sig2 = compute_minhash(
            "quantum computing uses superposition entanglement for parallel processing of complex problems",
            128,
        );
        let sim = minhash_similarity(&sig1, &sig2);
        assert!(sim < 0.3, "completely different texts should have low similarity, got {sim}");
    }

    #[test]
    fn test_word_ngrams_basic() {
        let ngrams = word_ngrams("hello world foo bar", 3);
        assert_eq!(ngrams.len(), 2);
        assert_eq!(ngrams[0], "hello world foo");
        assert_eq!(ngrams[1], "world foo bar");
    }

    #[test]
    fn test_word_ngrams_short_text() {
        let ngrams = word_ngrams("hello world", 3);
        assert_eq!(ngrams.len(), 1);
        assert_eq!(ngrams[0], "hello world");
    }

    #[test]
    fn test_word_ngrams_empty() {
        let ngrams = word_ngrams("", 3);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_optimal_bands_powers_of_two() {
        // 128 permutations, threshold 0.5 — should pick a valid band count
        let bands = optimal_bands(128, 0.5);
        assert!(128 % bands == 0, "bands must divide num_perm evenly");
        assert!(bands >= 1);
    }
}

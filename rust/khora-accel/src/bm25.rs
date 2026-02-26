//! BM25 index with PyO3 bindings.
//!
//! Provides a full BM25 ranking index that mirrors the Python `BM25Index`
//! in `khora.query.keyword`, with faster tokenization, inverted-index
//! lookups, and GIL-released batch scoring.

use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Static resources
// ---------------------------------------------------------------------------

static TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-zA-Z0-9]+\b").expect("token regex"));

static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
        "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "this",
        "but", "they", "have", "had", "what", "when", "where", "who", "which", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "can", "should",
        "now", "i", "you", "we", "our", "your", "my", "me", "him", "her", "them", "their", "been",
        "being", "do", "does", "did", "doing", "would", "could", "if", "then", "else", "or",
        "because", "until", "while", "am",
    ]
    .into_iter()
    .collect()
});

static SUFFIXES: &[&str] = &[
    "ing", "ed", "tion", "ness", "ment", "able", "ible", "ful", "less", "ly", "er", "est", "es",
    "s",
];

// ---------------------------------------------------------------------------
// Tokenisation helpers (pure Rust, no Python interaction)
// ---------------------------------------------------------------------------

#[inline]
fn basic_stem(word: &str) -> &str {
    for suffix in SUFFIXES {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            return &word[..word.len() - suffix.len()];
        }
    }
    word
}

/// Tokenise text: lowercase → regex split → stopword removal → stem → length filter.
///
/// Returns a `Vec` of *owned* `String`s because the stemmed forms may be
/// sub-slices of the lowered text that outlive the call.
fn tokenize(text: &str, use_stemming: bool, remove_stopwords: bool) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();

    for m in TOKEN_RE.find_iter(&lower) {
        let word = m.as_str();

        if remove_stopwords && STOPWORDS.contains(word) {
            continue;
        }

        let stemmed = if use_stemming {
            basic_stem(word)
        } else {
            word
        };

        if stemmed.len() > 2 {
            tokens.push(stemmed.to_owned());
        }
    }
    tokens
}

/// Same as `tokenize` but returns indices (u32) into the token→index map,
/// growing the map as needed. Used internally for fast scoring.
fn tokenize_indexed(
    text: &str,
    use_stemming: bool,
    remove_stopwords: bool,
    token_to_idx: &mut HashMap<String, u32>,
) -> Vec<u32> {
    let lower = text.to_lowercase();
    let mut indices: Vec<u32> = Vec::new();

    for m in TOKEN_RE.find_iter(&lower) {
        let word = m.as_str();

        if remove_stopwords && STOPWORDS.contains(word) {
            continue;
        }

        let stemmed = if use_stemming {
            basic_stem(word)
        } else {
            word
        };

        if stemmed.len() <= 2 {
            continue;
        }

        let idx = if let Some(&idx) = token_to_idx.get(stemmed) {
            idx
        } else {
            let idx = token_to_idx.len() as u32;
            token_to_idx.insert(stemmed.to_owned(), idx);
            idx
        };

        indices.push(idx);
    }
    indices
}

// ---------------------------------------------------------------------------
// BM25 index
// ---------------------------------------------------------------------------

/// High-performance BM25 index.
///
/// All index state lives in Rust; the Python side only touches thin PyO3
/// wrappers so the GIL can be released during scoring and search.
#[pyclass]
pub struct RustBM25Index {
    k1: f32,
    b: f32,
    use_stemming: bool,
    remove_stopwords: bool,

    // token string ↔ u32 mapping
    token_to_idx: HashMap<String, u32>,

    // Per-document data (keyed by internal u32 doc index)
    doc_id_to_idx: HashMap<String, u32>,
    doc_ids: Vec<String>,                    // idx → doc_id
    doc_lengths: Vec<u32>,                   // idx → token count
    doc_freqs: Vec<HashMap<u32, u32>>,       // idx → { token_idx → count }

    // Global stats
    term_doc_freqs: HashMap<u32, u32>,       // token_idx → # docs containing it
    inverted_index: HashMap<u32, Vec<u32>>,  // token_idx → [doc_idx …]
    total_length: u64,
    total_docs: u32,
    avg_doc_length: f32,
}

#[pymethods]
impl RustBM25Index {
    #[new]
    #[pyo3(signature = (k1 = 1.5, b = 0.75, use_stemming = true, remove_stopwords = true))]
    fn new(k1: f32, b: f32, use_stemming: bool, remove_stopwords: bool) -> Self {
        Self {
            k1,
            b,
            use_stemming,
            remove_stopwords,
            token_to_idx: HashMap::new(),
            doc_id_to_idx: HashMap::new(),
            doc_ids: Vec::new(),
            doc_lengths: Vec::new(),
            doc_freqs: Vec::new(),
            term_doc_freqs: HashMap::new(),
            inverted_index: HashMap::new(),
            total_length: 0,
            total_docs: 0,
            avg_doc_length: 0.0,
        }
    }

    /// Add a single document to the index.
    fn add_document(&mut self, doc_id: String, text: &str) {
        let tokens = tokenize_indexed(text, self.use_stemming, self.remove_stopwords, &mut self.token_to_idx);
        self.add_document_tokens(doc_id, tokens);
        self.update_avg_length();
    }

    /// Add multiple documents at once.
    fn add_documents(&mut self, documents: Vec<(String, String)>) {
        for (doc_id, text) in documents {
            let tokens = tokenize_indexed(&text, self.use_stemming, self.remove_stopwords, &mut self.token_to_idx);
            self.add_document_tokens(doc_id, tokens);
        }
        self.update_avg_length();
    }

    /// BM25 score for a single query-document pair.
    fn score(&self, query: &str, doc_id: &str) -> f32 {
        let doc_idx = match self.doc_id_to_idx.get(doc_id) {
            Some(&idx) => idx as usize,
            None => return 0.0,
        };

        let query_tokens = tokenize(query, self.use_stemming, self.remove_stopwords);
        let doc_freq = &self.doc_freqs[doc_idx];
        let doc_len = self.doc_lengths[doc_idx] as f32;
        let avg_dl = self.avg_doc_length.max(1.0);

        let mut score = 0.0f32;
        for token_str in &query_tokens {
            let tok_idx = match self.token_to_idx.get(token_str.as_str()) {
                Some(&idx) => idx,
                None => continue,
            };
            let tf = match doc_freq.get(&tok_idx) {
                Some(&c) => c as f32,
                None => continue,
            };

            let idf = self.idf(tok_idx);
            let num = tf * (self.k1 + 1.0);
            let den = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avg_dl);
            score += idf * num / den;
        }
        score
    }

    /// Search the index. Returns `Vec<(doc_id, score)>` sorted descending.
    ///
    /// Releases the GIL during the scoring phase.
    #[pyo3(signature = (query, limit = 10, min_score = 0.0))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
        min_score: f32,
    ) -> Vec<(String, f32)> {
        // Tokenize once (fixes the Python bug that re-tokenizes per candidate)
        let query_tokens = tokenize(query, self.use_stemming, self.remove_stopwords);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Resolve query tokens to indices
        let query_tok_indices: Vec<u32> = query_tokens
            .iter()
            .filter_map(|t| self.token_to_idx.get(t.as_str()).copied())
            .collect();

        if query_tok_indices.is_empty() {
            return Vec::new();
        }

        // Gather candidate doc indices via inverted index
        let mut candidate_set: HashSet<u32> = HashSet::new();
        for &tok_idx in &query_tok_indices {
            if let Some(postings) = self.inverted_index.get(&tok_idx) {
                for &doc_idx in postings {
                    candidate_set.insert(doc_idx);
                }
            }
        }

        let avg_dl = self.avg_doc_length.max(1.0);
        let k1 = self.k1;
        let b = self.b;

        // Pre-compute IDFs for query terms
        let idfs: Vec<f32> = query_tok_indices.iter().map(|&t| self.idf(t)).collect();

        // Snapshot data needed for scoring (all borrowed from &self, no Python objects)
        let doc_freqs = &self.doc_freqs;
        let doc_lengths = &self.doc_lengths;

        // Release GIL for scoring — parallelised with rayon
        let mut results: Vec<(u32, f32)> = py.detach(|| {
            let candidates: Vec<u32> = candidate_set.into_iter().collect();

            let mut results: Vec<(u32, f32)> = candidates
                .into_par_iter()
                .filter_map(|doc_idx| {
                    let df = &doc_freqs[doc_idx as usize];
                    let dl = doc_lengths[doc_idx as usize] as f32;

                    let mut s = 0.0f32;
                    for (i, &tok_idx) in query_tok_indices.iter().enumerate() {
                        let tf = match df.get(&tok_idx) {
                            Some(&c) => c as f32,
                            None => continue,
                        };
                        let num = tf * (k1 + 1.0);
                        let den = tf + k1 * (1.0 - b + b * dl / avg_dl);
                        s += idfs[i] * num / den;
                    }

                    if s >= min_score { Some((doc_idx, s)) } else { None }
                })
                .collect();

            results.sort_by(|a, b_| b_.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(limit);
            results
        });

        // Map back to doc_id strings (bounds-safe)
        results
            .drain(..)
            .filter_map(|(idx, score)| {
                self.doc_ids.get(idx as usize).map(|id| (id.clone(), score))
            })
            .collect()
    }
}

// Private helpers (not exposed to Python)
impl RustBM25Index {
    fn update_avg_length(&mut self) {
        if self.total_docs > 0 {
            self.avg_doc_length = self.total_length as f32 / self.total_docs as f32;
        }
    }

    #[inline]
    fn idf(&self, tok_idx: u32) -> f32 {
        let n = self.total_docs as f32;
        let df = self.term_doc_freqs.get(&tok_idx).copied().unwrap_or(0) as f32;
        if df == 0.0 {
            return 0.0;
        }
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Insert pre-tokenized document into the index (shared by add_document / add_documents).
    fn add_document_tokens(&mut self, doc_id: String, tokens: Vec<u32>) {
        let doc_idx = self.doc_ids.len() as u32;

        self.doc_id_to_idx.insert(doc_id.clone(), doc_idx);
        self.doc_ids.push(doc_id);

        let token_len = tokens.len() as u32;
        self.doc_lengths.push(token_len);

        // Build per-doc term frequency map
        let mut freq: HashMap<u32, u32> = HashMap::new();
        for &tok in &tokens {
            *freq.entry(tok).or_insert(0) += 1;
        }

        // Update global term-doc frequencies and inverted index (unique terms only)
        for &tok in freq.keys() {
            *self.term_doc_freqs.entry(tok).or_insert(0) += 1;
            self.inverted_index.entry(tok).or_default().push(doc_idx);
        }

        self.doc_freqs.push(freq);
        self.total_docs += 1;
        self.total_length += token_len as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_score() {
        let mut index = RustBM25Index::new(1.5, 0.75, true, true);
        index.add_document("doc1".to_string(), "the quick brown fox jumps over the lazy dog");
        index.add_document("doc2".to_string(), "a brown dog runs through the field");

        let score = index.score("brown fox", "doc1");
        assert!(score > 0.0);
    }

    #[test]
    fn test_empty_index_score() {
        let index = RustBM25Index::new(1.5, 0.75, true, true);
        let score = index.score("anything", "nonexistent");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_multiple_documents() {
        let mut index = RustBM25Index::new(1.5, 0.75, true, true);
        index.add_documents(vec![
            ("doc1".to_string(), "machine learning algorithms".to_string()),
            ("doc2".to_string(), "deep learning neural networks".to_string()),
            ("doc3".to_string(), "cooking recipes for pasta".to_string()),
        ]);

        // "learning" stems to "learn" — appears in doc1 and doc2, not doc3
        let score1 = index.score("learning", "doc1");
        let score2 = index.score("learning", "doc2");
        let score3 = index.score("learning", "doc3");
        assert!(score1 > 0.0);
        assert!(score2 > 0.0);
        assert_eq!(score3, 0.0);
    }

    #[test]
    fn test_basic_stem() {
        assert_eq!(basic_stem("running"), "runn");
        assert_eq!(basic_stem("played"), "play");
        assert_eq!(basic_stem("go"), "go"); // too short to stem
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("The quick brown fox", true, true);
        // "the" is a stopword, should be removed
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()) || tokens.contains(&"quic".to_string()));
    }
}

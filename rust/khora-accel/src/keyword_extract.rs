//! Keyword extraction with PyO3 bindings.
//!
//! Mirrors the `_extract_keywords` method in
//! `khora.engines.skeleton.skeleton.SkeletonIndexer`: regex tokenise,
//! filter stopwords, deduplicate via `HashSet`.

use hashbrown::HashSet;
use pyo3::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Static resources
// ---------------------------------------------------------------------------

/// Regex: 3+ ASCII-letter words (matches `r"\b[a-zA-Z]{3,}\b"` in Python).
static KEYWORD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[a-zA-Z]{3,}\b").expect("keyword regex"));

/// Stopwords matching the skeleton.py `_extract_keywords` set exactly.
static SKELETON_STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "must", "that", "this",
        "these", "those", "it", "its", "he", "she", "they", "them", "his", "her", "their", "we",
        "our", "you", "your", "i", "me", "my", "what", "which", "who", "whom", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "not", "only", "own", "same", "so", "than", "too", "very", "just", "can",
    ]
    .into_iter()
    .collect()
});

// ---------------------------------------------------------------------------
// Core extraction (pure Rust)
// ---------------------------------------------------------------------------

fn extract_keywords_inner(content: &str) -> Vec<String> {
    let lower = content.to_lowercase();
    let mut seen = HashSet::new();
    let mut keywords = Vec::new();

    for m in KEYWORD_RE.find_iter(&lower) {
        let word = m.as_str();
        if !SKELETON_STOPWORDS.contains(word) && seen.insert(word.to_owned()) {
            keywords.push(word.to_owned());
        }
    }
    keywords
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

/// Extract unique keywords from a single piece of content.
///
/// Tokenises with `\b[a-zA-Z]{3,}\b`, removes stopwords, deduplicates.
#[pyfunction]
pub fn extract_keywords(content: &str) -> Vec<String> {
    extract_keywords_inner(content)
}

/// Batch keyword extraction using rayon parallelism.
///
/// Releases the GIL so Python threads are not blocked.
#[pyfunction]
pub fn extract_keywords_batch(py: Python<'_>, contents: Vec<String>) -> Vec<Vec<String>> {
    py.detach(|| {
        contents
            .par_iter()
            .map(|c| extract_keywords_inner(c))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_extraction() {
        let keywords = extract_keywords_inner("Machine learning algorithms process data efficiently");
        assert!(keywords.contains(&"machine".to_string()));
        assert!(keywords.contains(&"learning".to_string()));
        assert!(keywords.contains(&"algorithms".to_string()));
        assert!(keywords.contains(&"process".to_string()));
        assert!(keywords.contains(&"data".to_string()));
    }

    #[test]
    fn test_stopword_removal() {
        let keywords = extract_keywords_inner("the quick and the slow");
        // "the" and "and" are stopwords
        assert!(!keywords.contains(&"the".to_string()));
        assert!(!keywords.contains(&"and".to_string()));
        assert!(keywords.contains(&"quick".to_string()));
        assert!(keywords.contains(&"slow".to_string()));
    }

    #[test]
    fn test_deduplication() {
        let keywords = extract_keywords_inner("data data data processing data");
        let data_count = keywords.iter().filter(|k| *k == "data").count();
        assert_eq!(data_count, 1);
    }

    #[test]
    fn test_empty_input() {
        let keywords = extract_keywords_inner("");
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_short_words_excluded() {
        // Words shorter than 3 chars should be excluded by the regex
        let keywords = extract_keywords_inner("go to be at");
        assert!(keywords.is_empty());
    }
}

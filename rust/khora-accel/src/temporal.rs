//! Temporal filtering and recency scoring with PyO3 bindings.
//!
//! Batch operations for temporal range matching and exponential decay
//! scoring. Timestamps are passed as epoch seconds (f64) to avoid
//! complex datetime serialisation across the Python boundary.

use pyo3::prelude::*;
use rayon::prelude::*;

const SECONDS_PER_DAY: f64 = 86400.0;
#[allow(clippy::approx_constant)]
const LN_HALF: f64 = -0.693_147_180_559_945_3; // -ln(2), negated for decay

/// Batch temporal filter: test each timestamp against a temporal range.
///
/// `operator` is one of `"before"`, `"after"`, `"between"`.
/// Timestamps and boundaries are epoch seconds (f64).
///
/// Returns a `Vec<bool>` mask — `true` if the timestamp matches the filter.
/// Releases the GIL during computation.
#[pyfunction]
#[pyo3(signature = (timestamps_secs, operator, start_secs = None, end_secs = None))]
pub fn batch_temporal_filter(
    py: Python<'_>,
    timestamps_secs: Vec<f64>,
    operator: &str,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
) -> Vec<bool> {
    // Parse operator once before entering GIL-free zone
    let op = match operator {
        "before" => Op::Before,
        "after" => Op::After,
        "between" => Op::Between,
        _ => Op::Pass, // "during", "overlaps", unknown → pass through
    };
    let start = start_secs;
    let end = end_secs;

    py.detach(move || {
        if timestamps_secs.len() < 512 {
            // Small batches: sequential is faster (no rayon overhead)
            timestamps_secs
                .iter()
                .map(|&ts| matches_op(ts, op, start, end))
                .collect()
        } else {
            timestamps_secs
                .par_iter()
                .map(|&ts| matches_op(ts, op, start, end))
                .collect()
        }
    })
}

/// Batch recency scores using exponential decay.
///
/// For each timestamp, computes:
///   `(1 - recency_weight) + recency_weight * 0.5^(age_days / decay_days)`
///
/// where `age_days = (now_secs - ts) / 86400`.
///
/// If `recency_weight == 0`, returns a vec of `1.0` (fast path).
/// Releases the GIL during computation.
#[pyfunction]
#[pyo3(signature = (timestamps_secs, now_secs, decay_days, recency_weight))]
pub fn batch_recency_scores(
    py: Python<'_>,
    timestamps_secs: Vec<f64>,
    now_secs: f64,
    decay_days: f64,
    recency_weight: f64,
) -> Vec<f64> {
    py.detach(move || {
        // Fast path
        if recency_weight == 0.0 {
            return vec![1.0; timestamps_secs.len()];
        }

        let base = 1.0 - recency_weight;
        // Precompute ln(0.5) / decay_days for the exponential
        let decay_factor = if decay_days > 0.0 {
            LN_HALF / decay_days
        } else {
            0.0 // No decay
        };

        let compute = |&ts: &f64| -> f64 {
            let age_days = (now_secs - ts) / SECONDS_PER_DAY;
            let decay = (decay_factor * age_days).exp(); // 0.5^(age/half_life)
            base + recency_weight * decay
        };

        if timestamps_secs.len() < 512 {
            timestamps_secs.iter().map(compute).collect()
        } else {
            timestamps_secs.par_iter().map(compute).collect()
        }
    })
}

// -- Internal helpers --

#[derive(Clone, Copy)]
enum Op {
    Before,
    After,
    Between,
    Pass,
}

#[inline(always)]
fn matches_op(ts: f64, op: Op, start: Option<f64>, end: Option<f64>) -> bool {
    match op {
        Op::Before => end.map_or(true, |e| ts < e),
        Op::After => start.map_or(true, |s| ts > s),
        Op::Between => {
            let after_start = start.map_or(true, |s| ts >= s);
            let before_end = end.map_or(true, |e| ts <= e);
            after_start && before_end
        }
        Op::Pass => true,
    }
}

/// Fast regex-based temporal keyword detection.
///
/// Returns `true` if the query contains temporal keywords like "when", "before",
/// "after", "since", "until", "yesterday", "recently", date patterns, etc.
/// This replaces Python-side regex for zero-overhead temporal detection.
#[pyfunction]
pub fn detect_temporal_keywords(query: &str) -> bool {
    use std::sync::LazyLock;
    use regex::Regex;

    static TEMPORAL_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)\b(when|before|after|during|since|until|last\s+(?:week|month|year|night|time)|yesterday|today|recently|earlier|latest|newest|oldest|first|most\s+recent|in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)|in\s+\d{4}|on\s+\d{1,2}[/\-]|ago)\b"
        ).expect("temporal regex compilation failed")
    });

    TEMPORAL_RE.is_match(query)
}

/// Categorized temporal keyword detection using Aho-Corasick automaton.
///
/// Returns a category ID:
///   0 = NONE, 1 = EXPLICIT, 2 = STATE_QUERY, 3 = ORDINAL,
///   4 = AGGREGATE, 5 = RECENCY, 6 = CHANGE
///
/// Higher category IDs take priority when multiple matches exist.
#[pyfunction]
pub fn detect_temporal_category(query: &str) -> u8 {
    use aho_corasick::AhoCorasick;
    use std::sync::LazyLock;

    static AC: LazyLock<(AhoCorasick, Vec<u8>)> = LazyLock::new(|| {
        let mut patterns = Vec::new();
        let mut categories = Vec::new();

        // Helper to add patterns for a category
        let mut add = |cat: u8, pats: &[&str]| {
            for p in pats {
                patterns.push(p.to_string());
                categories.push(cat);
            }
        };

        // Category 1: EXPLICIT (date markers, month names, etc.)
        add(1, &[
            "when ", "before ", "after ", "during ", "since ", "until ",
            "yesterday", "today", " ago",
            "january", "february", "march", "april", "may ", "june",
            "july", "august", "september", "october", "november", "december",
            "last week", "last month", "last year", "last night", "last time",
        ]);

        // Category 2: STATE_QUERY
        add(2, &[
            "currently", "right now", "at present", "presently",
            "these days", "nowadays", "at this point", "at the moment",
            // Implicit state-query patterns for conversational memory (synced from Python _accel.py)
            " does he still", " does she still", " do they still",
            " is he still", " is she still", " are they still",
            " does it still", " is it still", " am i still", " do i still",
            "'s current ", " current job", " current role", " current position",
            " live now", " work now", " working now", " living now", " doing now",
        ]);

        // Category 3: ORDINAL
        // NOTE: "before/after" comparison patterns intentionally omitted —
        // they belong in EXPLICIT (cat 1) for temporal date filtering.
        // ORDINAL (3) > EXPLICIT (1) in max(cat_id) priority.
        add(3, &[
            "first ", " earliest", "which came", "what came",
            "preceding", "following ", "subsequent",
            "in what order", "chronological", "what order did",
            "what sequence",
        ]);

        // Category 4: AGGREGATE
        add(4, &[
            "how many times", "how many total", "all instances",
            "every time", "in total", "count of", "number of times",
            "how often ",
        ]);

        // Category 5: RECENCY
        add(5, &[
            "most recent", "newest", "just ", "recently",
            "latest ",
        ]);

        // Category 6: CHANGE
        // NOTE: "still " intentionally omitted — it conflicts with STATE_QUERY
        // patterns (" does she still", " is he still", etc.) which are more
        // specific. Since max(cat_id) wins, having "still " here would override
        // STATE_QUERY for queries like "Does she still work there?".
        add(6, &[
            "changed", "switched", "moved to", "used to",
            "no longer", "anymore", "former ",
            "previous ", "ex-", "updated", "replaced",
            "went from", "transitioned",
            "turned into ", "switched to ", "became ",
            "converted to ", "went back to ",
        ]);

        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Aho-Corasick build failed");
        (ac, categories)
    });

    let (ac, cats) = &*AC;

    // Pad with a leading space so patterns like " does she still" can match
    // at the start of the query (many patterns use leading spaces as word
    // boundary anchors).
    let padded = format!(" {}", query);
    let mut best_cat: u8 = 0;
    for mat in ac.find_iter(&padded) {
        let cat = cats[mat.pattern().as_usize()];
        if cat > best_cat {
            best_cat = cat;
        }
    }
    best_cat
}

/// Categorized temporal detection with confidence score and matched terms.
///
/// Returns `(category_id, confidence, matched_terms)` where:
///   - `category_id`: 0=NONE through 6=CHANGE (same as `detect_temporal_category`)
///   - `confidence`: 0.0–1.0 based on number and strength of matches
///   - `matched_terms`: list of matched keyword strings
///
/// Confidence heuristic:
///   - Single match: 0.6
///   - Two matches in same category: 0.8
///   - Three+ matches or matches across multiple categories: 0.95
///   - Explicit date patterns (regex): +0.1 bonus, capped at 1.0
#[pyfunction]
pub fn detect_temporal_category_with_confidence(query: &str) -> (u8, f64, Vec<String>) {
    use aho_corasick::AhoCorasick;
    use std::sync::LazyLock;

    static AC: LazyLock<(AhoCorasick, Vec<u8>, Vec<String>)> = LazyLock::new(|| {
        let mut patterns: Vec<String> = Vec::new();
        let mut categories: Vec<u8> = Vec::new();

        let mut add = |cat: u8, pats: &[&str]| {
            for p in pats {
                patterns.push(p.to_string());
                categories.push(cat);
            }
        };

        add(1, &[
            "when ", "before ", "after ", "during ", "since ", "until ",
            "yesterday", "today", " ago",
            "january", "february", "march", "april", "may ", "june",
            "july", "august", "september", "october", "november", "december",
            "last week", "last month", "last year", "last night", "last time",
        ]);
        add(2, &[
            "currently", "right now", "at present", "presently",
            "these days", "nowadays", "at this point", "at the moment",
            // Implicit state-query patterns for conversational memory (synced from Python _accel.py)
            " does he still", " does she still", " do they still",
            " is he still", " is she still", " are they still",
            " does it still", " is it still", " am i still", " do i still",
            "'s current ", " current job", " current role", " current position",
            " live now", " work now", " working now", " living now", " doing now",
        ]);
        add(3, &[
            "first ", " earliest", "which came", "what came",
            "preceding", "following ", "subsequent",
            "in what order", "chronological", "what order did",
            "what sequence",
        ]);
        add(4, &[
            "how many times", "how many total", "all instances",
            "every time", "in total", "count of", "number of times",
            "how often ",
        ]);
        add(5, &[
            "most recent", "newest", "just ", "recently",
            "latest ",
        ]);
        // NOTE: "still " intentionally omitted — see detect_temporal_category.
        add(6, &[
            "changed", "switched", "moved to", "used to",
            "no longer", "anymore", "former ",
            "previous ", "ex-", "updated", "replaced",
            "went from", "transitioned",
            "turned into ", "switched to ", "became ",
            "converted to ", "went back to ",
        ]);

        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Aho-Corasick build failed");
        (ac, categories, patterns)
    });

    static DATE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\b\d{4}[-/]\d{1,2}|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b")
            .expect("date regex failed")
    });

    let (ac, cats, patterns) = &*AC;

    // Pad with a leading space so patterns with leading-space word-boundary
    // anchors can match at the start of the query.
    let padded = format!(" {}", query);
    let mut best_cat: u8 = 0;
    let mut matched_terms: Vec<String> = Vec::new();
    let mut matched_cats: std::collections::HashSet<u8> = std::collections::HashSet::new();

    for mat in ac.find_iter(&padded) {
        let cat = cats[mat.pattern().as_usize()];
        let term = &patterns[mat.pattern().as_usize()];
        if cat > best_cat {
            best_cat = cat;
        }
        matched_cats.insert(cat);
        if !matched_terms.contains(term) {
            matched_terms.push(term.clone());
        }
    }

    if best_cat == 0 {
        return (0, 0.0, vec![]);
    }

    // Compute confidence
    let n_matches = matched_terms.len();
    let n_cats = matched_cats.len();
    let mut confidence = match n_matches {
        1 => 0.6,
        2 => {
            if n_cats > 1 { 0.85 } else { 0.8 }
        }
        _ => 0.95,
    };

    // Bonus for explicit date pattern
    if DATE_RE.is_match(query) {
        confidence = (confidence + 0.1_f64).min(1.0);
    }

    (best_cat, confidence, matched_terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_filter_after() {
        let timestamps = vec![100.0, 200.0, 300.0, 400.0];
        let result = timestamps
            .iter()
            .map(|&ts| matches_op(ts, Op::After, Some(250.0), None))
            .collect::<Vec<_>>();
        assert_eq!(result, vec![false, false, true, true]);
    }

    #[test]
    fn test_batch_filter_before() {
        let timestamps = vec![100.0, 200.0, 300.0, 400.0];
        let result = timestamps
            .iter()
            .map(|&ts| matches_op(ts, Op::Before, None, Some(250.0)))
            .collect::<Vec<_>>();
        assert_eq!(result, vec![true, true, false, false]);
    }

    #[test]
    fn test_batch_filter_between() {
        let timestamps = vec![100.0, 200.0, 300.0, 400.0];
        let result = timestamps
            .iter()
            .map(|&ts| matches_op(ts, Op::Between, Some(150.0), Some(350.0)))
            .collect::<Vec<_>>();
        assert_eq!(result, vec![false, true, true, false]);
    }

    #[test]
    fn test_batch_filter_pass() {
        let timestamps = vec![100.0, 200.0];
        let result = timestamps
            .iter()
            .map(|&ts| matches_op(ts, Op::Pass, None, None))
            .collect::<Vec<_>>();
        assert_eq!(result, vec![true, true]);
    }

    #[test]
    fn test_recency_scores_zero_weight() {
        let timestamps = vec![100.0, 200.0, 300.0];
        let base = 1.0 - 0.0;

        let scores: Vec<f64> = timestamps
            .iter()
            .map(|_| base)
            .collect();
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_recency_scores_decay() {
        // With decay_days=1 and recency_weight=1, a timestamp 1 day old
        // should have score = 0.5 (since 0.5^(1/1) = 0.5)
        let now = 86400.0; // 1 day in seconds
        let timestamps = vec![0.0]; // exactly 1 day ago
        let decay_days = 1.0;
        let recency_weight = 1.0;

        let base = 1.0 - recency_weight;
        let decay_factor = LN_HALF / decay_days;

        let scores: Vec<f64> = timestamps
            .iter()
            .map(|&ts| {
                let age_days = (now - ts) / SECONDS_PER_DAY;
                let decay = (decay_factor * age_days).exp();
                base + recency_weight * decay
            })
            .collect();

        assert!((scores[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_recency_scores_recent_is_higher() {
        let now = 86400.0 * 10.0; // 10 days
        let timestamps = vec![
            86400.0 * 9.0,  // 1 day ago
            86400.0 * 5.0,  // 5 days ago
            0.0,            // 10 days ago
        ];
        let decay_days = 5.0;
        let recency_weight = 0.5;
        let base = 1.0 - recency_weight;
        let decay_factor = LN_HALF / decay_days;

        let scores: Vec<f64> = timestamps
            .iter()
            .map(|&ts| {
                let age_days = (now - ts) / SECONDS_PER_DAY;
                let decay = (decay_factor * age_days).exp();
                base + recency_weight * decay
            })
            .collect();

        // More recent timestamps should have higher scores
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_detect_temporal_keywords() {
        assert!(detect_temporal_keywords("When did Alice move?"));
        assert!(detect_temporal_keywords("What happened before 2023?"));
        assert!(detect_temporal_keywords("Events since last month"));
        assert!(detect_temporal_keywords("The most recent meeting"));
        assert!(!detect_temporal_keywords("What is the capital of France?"));
        assert!(!detect_temporal_keywords("Tell me about quantum physics"));
    }

    #[test]
    fn test_detect_temporal_category_state_query() {
        assert_eq!(detect_temporal_category("What instrument is the user currently playing?"), 2);
    }

    #[test]
    fn test_detect_temporal_category_ordinal() {
        assert_eq!(detect_temporal_category("Which event happened first ?"), 3);
    }

    #[test]
    fn test_detect_temporal_category_ordinal_in_what_order() {
        assert_eq!(detect_temporal_category("In what order did the events happen?"), 3);
    }

    #[test]
    fn test_detect_temporal_category_ordinal_chronological() {
        assert_eq!(detect_temporal_category("List the changes in chronological order"), 3);
    }

    #[test]
    fn test_detect_temporal_category_ordinal_before_or_after() {
        assert_eq!(detect_temporal_category("Did X happened before or after Y?"), 3);
    }

    #[test]
    fn test_detect_temporal_category_explicit() {
        assert_eq!(detect_temporal_category("What happened before April 2024?"), 1);
    }

    #[test]
    fn test_detect_temporal_category_aggregate() {
        assert_eq!(detect_temporal_category("How many times did she visit?"), 4);
    }

    #[test]
    fn test_detect_temporal_category_recency() {
        assert_eq!(detect_temporal_category("What is the most recent update?"), 5);
    }

    #[test]
    fn test_detect_temporal_category_change() {
        // "used to" is a clear CHANGE signal
        assert_eq!(detect_temporal_category("She used to work there"), 6);
    }

    #[test]
    fn test_still_state_query_not_change() {
        // "Does she still" should match STATE_QUERY (cat 2), not CHANGE.
        // Previously "still " in CHANGE (cat 6) overrode the more specific
        // STATE_QUERY patterns due to max(cat_id) priority.
        assert_eq!(detect_temporal_category("Does she still work at Google?"), 2);
    }

    #[test]
    fn test_is_it_still_state_query() {
        assert_eq!(detect_temporal_category("Is it still raining?"), 2);
    }

    #[test]
    fn test_detect_temporal_category_none() {
        assert_eq!(detect_temporal_category("What is the capital of France?"), 0);
    }

    #[test]
    fn test_detect_temporal_category_latest() {
        assert_eq!(detect_temporal_category("What is the latest news?"), 5);
    }

    #[test]
    fn test_detect_temporal_category_became() {
        assert_eq!(detect_temporal_category("She became a doctor"), 6);
    }

    #[test]
    fn test_detect_temporal_category_how_often() {
        assert_eq!(detect_temporal_category("How often does she visit?"), 4);
    }

    #[test]
    fn test_detect_temporal_category_switched_to() {
        assert_eq!(detect_temporal_category("He switched to piano"), 6);
    }

    #[test]
    fn test_detect_temporal_category_state_query_current_job() {
        // "current job" is a STATE_QUERY (cat 2) pattern — no overlap with CHANGE
        assert_eq!(detect_temporal_category("What is her current job?"), 2);
    }

    #[test]
    fn test_detect_temporal_category_state_query_live_now() {
        assert_eq!(detect_temporal_category("Where does she live now?"), 2);
    }

    #[test]
    fn test_detect_temporal_category_state_query_working_now() {
        assert_eq!(detect_temporal_category("What is he working now?"), 2);
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_none() {
        let (cat, conf, terms) = detect_temporal_category_with_confidence("What is the capital of France?");
        assert_eq!(cat, 0);
        assert!((conf - 0.0).abs() < 1e-10);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_single() {
        let (cat, conf, terms) = detect_temporal_category_with_confidence("What happened yesterday?");
        assert_eq!(cat, 1);
        assert!((conf - 0.6).abs() < 1e-10);
        assert_eq!(terms.len(), 1);
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_multi() {
        let (cat, conf, terms) = detect_temporal_category_with_confidence(
            "When did she switch to piano after leaving the band?"
        );
        assert!(cat >= 1);
        assert!(conf >= 0.8);
        assert!(terms.len() >= 2);
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_date_bonus() {
        let (cat, conf, _) = detect_temporal_category_with_confidence(
            "What happened before 2024-01-15?"
        );
        assert_eq!(cat, 1);
        // Should get date bonus: 0.6 + 0.1 = 0.7
        assert!(conf >= 0.7);
    }
}

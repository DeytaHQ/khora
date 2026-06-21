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
/// where `age_days = max(0, (now_secs - ts) / 86400)` (future timestamps are
/// clamped to age 0, matching the NumPy/Python fallback).
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
            // Clamp future timestamps (clock skew, deliberate forward-dating)
            // to age=0 so a forward-dated chunk gets full freshness rather than
            // decay > 1.0 from exp(positive). Mirrors the `max(0.0, ...)` clamp
            // in the NumPy/Python fallback (_accel.py) and chronicle's
            // `_apply_temporal_decay`, keeping Rust and fallback in parity.
            let age_days = ((now_secs - ts) / SECONDS_PER_DAY).max(0.0);
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
    use regex::Regex;
    use std::sync::LazyLock;

    static TEMPORAL_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)\b(when|before|after|during|since|until|last\s+(?:week|month|year|night|time)|yesterday|today|recently|earlier|latest|newest|oldest|first|most\s+recent|in\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)|in\s+\d{4}|on\s+\d{1,2}[/\-]|ago)\b"
        ).expect("temporal regex compilation failed")
    });

    TEMPORAL_RE.is_match(query)
}

/// Whether `c` counts as a word character for boundary checks.
///
/// Mirrors Python's default `\b` semantics (Unicode-aware): a word char is an
/// alphanumeric or underscore. Used to reject substring matches like
/// "changed" inside "unchanged" or "march" inside "marched" (#981).
#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Verify that an Aho-Corasick match span sits on word boundaries, mirroring
/// the Python fallback's word-boundary regex (`\b` only on the side where the
/// pattern begins/ends with a word character).
///
/// `text` is the (space-padded) query. `start`/`end` are the byte offsets of
/// the match. `pattern` is the matched dictionary term (may carry leading or
/// trailing padding spaces, in which case no boundary is required on that side).
fn match_on_word_boundary(text: &str, start: usize, end: usize, pattern: &str) -> bool {
    let first = pattern.chars().next();
    let last = pattern.chars().last();

    // Left boundary: only required when the pattern starts with a word char.
    if first.is_some_and(is_word_char) {
        let prev = text[..start].chars().next_back();
        if prev.is_some_and(is_word_char) {
            return false;
        }
    }
    // Right boundary: only required when the pattern ends with a word char.
    if last.is_some_and(is_word_char) {
        let next = text[end..].chars().next();
        if next.is_some_and(is_word_char) {
            return false;
        }
    }
    true
}

/// Temporal prepositions that, when they precede the month name "may", make it
/// read as the month rather than the modal verb / proper name (#981).
const MAY_TEMPORAL_PREPS: &[&str] = &[
    "in", "on", "by", "since", "until", "before", "after", "during", "early", "late",
];

/// Disambiguate a whole-word "may" match: the month name is ambiguous with the
/// modal verb ("you may") and a proper name ("May Corp"), so it only classifies
/// as a date when it sits in a temporal context (#981) — either preceded by a
/// temporal preposition ("in May", "since May") or adjacent to a number
/// ("May 2024", "May 5", "5 May"). Bare "May" / "May Department Stores" is NOT
/// temporal.
///
/// `text` is the space-padded query; `start`/`end` are the byte offsets of the
/// "may" span within it.
fn may_is_temporal(text: &str, start: usize, end: usize) -> bool {
    // Number adjacent on the right: "May 2024", "May 5".
    let after = text[end..].trim_start();
    if after.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }
    // Tokens before "may" (the text up to the match, reversed word-by-word).
    let before = text[..start].trim_end();
    let prev_word = before
        .rsplit(|c: char| !is_word_char(c))
        .find(|w| !w.is_empty());
    if let Some(word) = prev_word {
        // Number adjacent on the left: "5 May" (prev_word is always non-empty).
        if word.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
        // Temporal preposition immediately before: "in May", "since May".
        let lower = word.to_ascii_lowercase();
        if MAY_TEMPORAL_PREPS.contains(&lower.as_str()) {
            return true;
        }
    }
    false
}

/// Categorized temporal keyword detection using Aho-Corasick automaton.
///
/// Returns a category ID:
///   0 = NONE, 1 = EXPLICIT, 2 = STATE_QUERY, 3 = ORDINAL,
///   4 = AGGREGATE, 5 = RECENCY, 6 = CHANGE
///
/// Higher category IDs take priority when multiple matches exist.
///
/// Matches are word-boundary-aware (#981): a keyword only fires as a whole
/// word/phrase, not as a substring of a larger word ("changed" inside
/// "unchanged", "march" inside "marched").
#[pyfunction]
pub fn detect_temporal_category(query: &str) -> u8 {
    use aho_corasick::AhoCorasick;
    use std::sync::LazyLock;

    static AC: LazyLock<(AhoCorasick, Vec<u8>, Vec<String>)> = LazyLock::new(|| {
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
        add(
            1,
            &[
                "when ",
                "before ",
                "after ",
                "during ",
                "since ",
                "until ",
                "yesterday",
                "today",
                " ago",
                "january",
                "february",
                "march",
                "april",
                // "may" is whole-word ambiguous (modal verb / proper name);
                // gated to temporal context by may_is_temporal (#981).
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
                "last week",
                "last month",
                "last year",
                "last night",
                "last time",
            ],
        );

        // Category 2: STATE_QUERY
        add(
            2,
            &[
                "currently",
                "right now",
                "at present",
                "presently",
                "these days",
                "nowadays",
                "at this point",
                "at the moment",
                // Implicit state-query patterns for conversational memory (synced from Python _accel.py)
                " does he still",
                " does she still",
                " do they still",
                " is he still",
                " is she still",
                " are they still",
                " does it still",
                " is it still",
                " am i still",
                " do i still",
                "'s current ",
                " current job",
                " current role",
                " current position",
                " live now",
                " work now",
                " working now",
                " living now",
                " doing now",
                // Enterprise domain compound current-state query patterns.
                // Generic " current " was too broad — triggered on recency lookups like
                // "current quota" that work better without temporal intelligence.
                " current status",
                " current stage",
                " current state",
                " current health",
                " current deal",
                " current project",
                " current plan",
                " current team",
                // Bare " active " was whole-word ambiguous ("the active
                // variable" — math, not temporal); narrowed to state-query
                // noun phrases (#981).
                "active deal",
                "active deals",
                "active project",
                "active projects",
                "active since",
                "currently active",
                "who is the ",  // implicit state: "Who is the account manager?"
                "who are the ", // "Who are the team members involved?"
                "up-to-date",
                "up to date",      // "more up-to-date"
                "authoritative",   // "authoritative source"
                "most reliable",   // "most reliable source"
                "official record", // "official record of"
            ],
        );

        // Category 3: ORDINAL
        // "before or after" is a compound phrase that unambiguously signals
        // ordering intent (not date filtering like bare "before "/"after ").
        // ORDINAL (3) > EXPLICIT (1) so it wins when both match.
        add(
            3,
            &[
                "first ",
                " earliest",
                "which came",
                "what came",
                "preceding",
                "following ",
                "subsequent",
                "in what order",
                "chronological",
                "what order did",
                "what sequence",
                "before or after",
                "happened first",
                "closed first",
                "created first",
                "came first",
                "started first",
                "completed first",
            ],
        );

        // Category 4: AGGREGATE
        add(
            4,
            &[
                "how many times",
                "how many total",
                "all instances",
                "every time",
                "in total",
                "count of",
                "number of times",
                "how often ",
            ],
        );

        // Category 5: RECENCY
        // Bare "just " was whole-word ambiguous ("just confirm" — adverb
        // meaning "only/simply", not temporal); narrowed to "just"+recency
        // phrases (#981).
        add(
            5,
            &[
                "most recent",
                "newest",
                "recently",
                "latest ",
                "just now",
                "just released",
                "just announced",
                "just shipped",
                "just launched",
                "just published",
                "just landed",
                "just happened",
                "just finished",
                "just completed",
                "just started",
                "just arrived",
            ],
        );

        // Category 6: CHANGE
        // NOTE: "still " intentionally omitted — it conflicts with STATE_QUERY
        // patterns (" does she still", " is he still", etc.) which are more
        // specific. Since max(cat_id) wins, having "still " here would override
        // STATE_QUERY for queries like "Does she still work there?".
        add(
            6,
            &[
                "changed",
                "switched",
                "moved to",
                "used to",
                "no longer",
                "anymore",
                "former ",
                "previous ",
                "ex-",
                "updated",
                "replaced",
                "went from",
                "transitioned",
                "turned into ",
                "switched to ",
                "became ",
                "converted to ",
                "went back to ",
            ],
        );

        let ac = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&patterns)
            .expect("Aho-Corasick build failed");
        (ac, categories, patterns)
    });

    let (ac, cats, patterns) = &*AC;

    // Pad with a leading space so patterns like " does she still" can match
    // at the start of the query (many patterns use leading spaces as word
    // boundary anchors).
    let padded = format!(" {}", query);
    let mut best_cat: u8 = 0;
    for mat in ac.find_overlapping_iter(&padded) {
        let pattern = &patterns[mat.pattern().as_usize()];
        if !match_on_word_boundary(&padded, mat.start(), mat.end(), pattern) {
            continue;
        }
        // "may" only counts as the month in a temporal context (#981).
        if pattern == "may" && !may_is_temporal(&padded, mat.start(), mat.end()) {
            continue;
        }
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

        add(
            1,
            &[
                "when ",
                "before ",
                "after ",
                "during ",
                "since ",
                "until ",
                "yesterday",
                "today",
                " ago",
                "january",
                "february",
                "march",
                "april",
                // "may" is whole-word ambiguous; gated by may_is_temporal (#981).
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
                "last week",
                "last month",
                "last year",
                "last night",
                "last time",
            ],
        );
        add(
            2,
            &[
                "currently",
                "right now",
                "at present",
                "presently",
                "these days",
                "nowadays",
                "at this point",
                "at the moment",
                " does he still",
                " does she still",
                " do they still",
                " is he still",
                " is she still",
                " are they still",
                " does it still",
                " is it still",
                " am i still",
                " do i still",
                "'s current ",
                " current job",
                " current role",
                " current position",
                " live now",
                " work now",
                " working now",
                " living now",
                " doing now",
                // Enterprise domain patterns
                " current status",
                " current stage",
                " current state",
                " current health",
                " current deal",
                " current project",
                " current plan",
                " current team",
                // Narrowed from bare " active " (#981).
                "active deal",
                "active deals",
                "active project",
                "active projects",
                "active since",
                "currently active",
                "who is the ",
                "who are the ",
                "up-to-date",
                "up to date",
                "authoritative",
                "most reliable",
                "official record",
            ],
        );
        add(
            3,
            &[
                "first ",
                " earliest",
                "which came",
                "what came",
                "preceding",
                "following ",
                "subsequent",
                "in what order",
                "chronological",
                "what order did",
                "what sequence",
                "before or after",
                "happened first",
                "closed first",
                "created first",
                "came first",
                "started first",
                "completed first",
            ],
        );
        add(
            4,
            &[
                "how many times",
                "how many total",
                "all instances",
                "every time",
                "in total",
                "count of",
                "number of times",
                "how often ",
            ],
        );
        // Bare "just " narrowed to recency phrases (#981).
        add(
            5,
            &[
                "most recent",
                "newest",
                "recently",
                "latest ",
                "just now",
                "just released",
                "just announced",
                "just shipped",
                "just launched",
                "just published",
                "just landed",
                "just happened",
                "just finished",
                "just completed",
                "just started",
                "just arrived",
            ],
        );
        // NOTE: "still " intentionally omitted — see detect_temporal_category.
        add(
            6,
            &[
                "changed",
                "switched",
                "moved to",
                "used to",
                "no longer",
                "anymore",
                "former ",
                "previous ",
                "ex-",
                "updated",
                "replaced",
                "went from",
                "transitioned",
                "turned into ",
                "switched to ",
                "became ",
                "converted to ",
                "went back to ",
            ],
        );

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

    for mat in ac.find_overlapping_iter(&padded) {
        let term = &patterns[mat.pattern().as_usize()];
        if !match_on_word_boundary(&padded, mat.start(), mat.end(), term) {
            continue;
        }
        // "may" only counts as the month in a temporal context (#981).
        if term == "may" && !may_is_temporal(&padded, mat.start(), mat.end()) {
            continue;
        }
        let cat = cats[mat.pattern().as_usize()];
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
            if n_cats > 1 {
                0.85
            } else {
                0.8
            }
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

        let scores: Vec<f64> = timestamps.iter().map(|_| base).collect();
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
            86400.0 * 9.0, // 1 day ago
            86400.0 * 5.0, // 5 days ago
            0.0,           // 10 days ago
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

    /// Plain-Rust mirror of `batch_recency_scores`' compute, callable without a
    /// `Python<'_>` token so the future-timestamp clamp can be unit-tested.
    fn batch_recency_scores_inner(
        timestamps_secs: &[f64],
        now_secs: f64,
        decay_days: f64,
        recency_weight: f64,
    ) -> Vec<f64> {
        if recency_weight == 0.0 {
            return vec![1.0; timestamps_secs.len()];
        }
        let base = 1.0 - recency_weight;
        let decay_factor = if decay_days > 0.0 {
            LN_HALF / decay_days
        } else {
            0.0
        };
        timestamps_secs
            .iter()
            .map(|&ts| {
                let age_days = ((now_secs - ts) / SECONDS_PER_DAY).max(0.0);
                base + recency_weight * (decay_factor * age_days).exp()
            })
            .collect()
    }

    #[test]
    fn test_recency_scores_future_timestamp_clamped() {
        // A forward-dated timestamp (ts > now) must be clamped to age 0 so its
        // score equals base + recency_weight (1.0 here), not an exp(positive)
        // blowup. Mirrors the `max(0.0, ...)` clamp in the Python fallback (#1130).
        let now = 86400.0 * 365.0;
        let future = now + 86400.0 * 365.0; // one year ahead
        let result = batch_recency_scores_inner(&[now, future], now, 30.0, 0.5);
        // Present-time ts: age 0 -> decay 1.0 -> base + weight == 1.0.
        assert!((result[0] - 1.0).abs() < 1e-10);
        // Future ts: clamped to age 0 -> identical to the present score, bounded.
        assert!((result[1] - 1.0).abs() < 1e-10);
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
        assert_eq!(
            detect_temporal_category("What instrument is the user currently playing?"),
            2
        );
    }

    #[test]
    fn test_detect_temporal_category_ordinal() {
        assert_eq!(detect_temporal_category("Which event happened first ?"), 3);
    }

    #[test]
    fn test_detect_temporal_category_ordinal_in_what_order() {
        assert_eq!(
            detect_temporal_category("In what order did the events happen?"),
            3
        );
    }

    #[test]
    fn test_detect_temporal_category_ordinal_chronological() {
        assert_eq!(
            detect_temporal_category("List the changes in chronological order"),
            3
        );
    }

    #[test]
    fn test_detect_temporal_category_ordinal_before_or_after() {
        assert_eq!(
            detect_temporal_category("Did X happened before or after Y?"),
            3
        );
    }

    #[test]
    fn test_detect_temporal_category_ordinal_closed_first() {
        assert_eq!(
            detect_temporal_category("Which deal closed first: Acme or Pinnacle?"),
            3
        );
    }

    #[test]
    fn test_detect_temporal_category_ordinal_created_first() {
        assert_eq!(
            detect_temporal_category("Which support ticket was created first?"),
            3
        );
    }

    #[test]
    fn test_detect_temporal_category_ordinal_started_before_or_after() {
        assert_eq!(
            detect_temporal_category(
                "Did the sabbatical start before or after the price increase?"
            ),
            3
        );
    }

    #[test]
    fn test_detect_temporal_category_explicit() {
        assert_eq!(
            detect_temporal_category("What happened before April 2024?"),
            1
        );
    }

    #[test]
    fn test_detect_temporal_category_aggregate() {
        assert_eq!(detect_temporal_category("How many times did she visit?"), 4);
    }

    #[test]
    fn test_detect_temporal_category_recency() {
        assert_eq!(
            detect_temporal_category("What is the most recent update?"),
            5
        );
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
        assert_eq!(
            detect_temporal_category("Does she still work at Google?"),
            2
        );
    }

    #[test]
    fn test_is_it_still_state_query() {
        assert_eq!(detect_temporal_category("Is it still raining?"), 2);
    }

    #[test]
    fn test_detect_temporal_category_none() {
        assert_eq!(
            detect_temporal_category("What is the capital of France?"),
            0
        );
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
        let (cat, conf, terms) =
            detect_temporal_category_with_confidence("What is the capital of France?");
        assert_eq!(cat, 0);
        assert!((conf - 0.0).abs() < 1e-10);
        assert!(terms.is_empty());
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_single() {
        let (cat, conf, terms) =
            detect_temporal_category_with_confidence("What happened yesterday?");
        assert_eq!(cat, 1);
        assert!((conf - 0.6).abs() < 1e-10);
        assert_eq!(terms.len(), 1);
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_multi() {
        let (cat, conf, terms) = detect_temporal_category_with_confidence(
            "When did she switch to piano after leaving the band?",
        );
        assert!(cat >= 1);
        assert!(conf >= 0.8);
        assert!(terms.len() >= 2);
    }

    #[test]
    fn test_detect_temporal_category_with_confidence_date_bonus() {
        let (cat, conf, _) =
            detect_temporal_category_with_confidence("What happened before 2024-01-15?");
        assert_eq!(cat, 1);
        // Should get date bonus: 0.6 + 0.1 = 0.7
        assert!(conf >= 0.7);
    }

    // Enterprise domain STATE_QUERY patterns
    #[test]
    fn test_state_current_status() {
        assert_eq!(
            detect_temporal_category("What is the current status of the Acme deal?"),
            2
        );
    }

    #[test]
    fn test_state_active_deals() {
        assert_eq!(
            detect_temporal_category("What are all the active deals in the pipeline?"),
            2
        );
    }

    #[test]
    fn test_state_who_is() {
        assert_eq!(
            detect_temporal_category("Who is the account manager for GreenWave?"),
            2
        );
    }

    #[test]
    fn test_state_authoritative() {
        assert_eq!(
            detect_temporal_category("Which system is the authoritative source for deal terms?"),
            2
        );
    }

    #[test]
    fn test_state_up_to_date() {
        assert_eq!(
            detect_temporal_category("Does the wiki or CRM have the more up-to-date status?"),
            2
        );
    }

    #[test]
    fn test_change_still_wins_over_current() {
        // "changed" is CHANGE (cat 6), " current status" is STATE_QUERY (cat 2)
        // CHANGE should win via max(cat_id)
        assert_eq!(
            detect_temporal_category("How has the current deal stage changed?"),
            6
        );
    }

    // Possessive "X's current ..." reads as a state query ("what is the
    // current state of X's quota"), so STATE_QUERY (2) is the intended
    // category here (#1285, decision (b) — the old assert-0 expectation was
    // stale, not the implementation). The possessive "'s current " pattern is
    // deliberately kept.
    #[test]
    fn test_current_quota_is_state_query() {
        assert_eq!(
            detect_temporal_category("What is Sarah's current quota attainment?"),
            2
        );
    }

    #[test]
    fn test_current_pipeline_not_state_query() {
        // No possessive and no compound " current <noun>" pattern matches, so
        // this stays a plain recency lookup (category 0).
        assert_eq!(
            detect_temporal_category("What is the current pipeline value?"),
            0
        );
    }

    // #981: word-boundary matching — keywords must not match as substrings of
    // larger words.
    #[test]
    fn test_change_not_in_unchanged() {
        // "changed" inside "unchanged" must NOT trip CHANGE.
        assert_eq!(
            detect_temporal_category("The contract terms remain unchanged."),
            0
        );
    }

    #[test]
    fn test_change_not_in_exchanged() {
        // "changed" inside "exchanged" must NOT trip CHANGE.
        assert_eq!(
            detect_temporal_category("The exchanged emails were archived."),
            0
        );
    }

    #[test]
    fn test_march_not_in_marched() {
        // "march" inside "marched" must NOT trip EXPLICIT.
        assert_eq!(
            detect_temporal_category("The team marched on with their plan."),
            0
        );
    }

    #[test]
    fn test_word_boundary_keeps_real_matches() {
        // The boundary fix must not regress genuine whole-word matches.
        assert_eq!(detect_temporal_category("What happened in March 2024?"), 1);
        assert_eq!(
            detect_temporal_category("The contract terms changed last week."),
            6
        );
        assert_eq!(
            detect_temporal_category("Does the wiki have the more up-to-date status?"),
            2
        );
    }

    // #981: Tier-1 disambiguation of whole-word ambiguous English keywords.
    // "may" (month vs. modal verb / proper name), "just" (recency vs. adverb),
    // "active" (state-query vs. generic adjective) only classify as temporal in
    // a temporal context.

    #[test]
    fn test_may_company_name_not_explicit() {
        // "May" as a company name must NOT trip EXPLICIT.
        assert_eq!(
            detect_temporal_category("Describe the May Department Stores company."),
            0
        );
        assert_eq!(detect_temporal_category("Tell me about May Corp."), 0);
    }

    #[test]
    fn test_may_modal_verb_not_explicit() {
        // "may" as a modal verb must NOT trip EXPLICIT.
        assert_eq!(detect_temporal_category("You may proceed."), 0);
    }

    #[test]
    fn test_may_month_with_preposition_is_explicit() {
        // "in May" / "since May" read as the month -> EXPLICIT.
        assert_eq!(detect_temporal_category("What shipped in May?"), 1);
        assert_eq!(detect_temporal_category("Everything since May."), 1);
    }

    #[test]
    fn test_may_month_with_year_is_explicit() {
        // "May 2024" / "5 May" read as the month -> EXPLICIT.
        assert_eq!(detect_temporal_category("What did Acme ship May 2024?"), 1);
        assert_eq!(
            detect_temporal_category("The release on 5 May went out."),
            1
        );
    }

    #[test]
    fn test_just_adverb_not_recency() {
        // "just" as the adverb "only/simply" must NOT trip RECENCY.
        assert_eq!(
            detect_temporal_category("Just confirm the data structure."),
            0
        );
    }

    #[test]
    fn test_just_recency_phrase_is_recency() {
        // "just now" / "just released" / "just shipped" -> RECENCY.
        // (NB: "just updated" trips CHANGE via "updated" — cat 6 wins by
        // max-priority; both are temporal, so that is acceptable behavior.)
        assert_eq!(detect_temporal_category("What happened just now?"), 5);
        assert_eq!(detect_temporal_category("What was just released?"), 5);
        assert_eq!(detect_temporal_category("The build just shipped."), 5);
    }

    #[test]
    fn test_active_generic_adjective_not_state_query() {
        // "active" as a generic adjective (math) must NOT trip STATE_QUERY.
        assert_eq!(
            detect_temporal_category("Apply the formula to the active variable."),
            0
        );
    }

    #[test]
    fn test_active_deals_is_state_query() {
        // "active deals" / "active projects" remain STATE_QUERY.
        assert_eq!(
            detect_temporal_category("What are all the active deals in the pipeline?"),
            2
        );
        assert_eq!(
            detect_temporal_category("List the active projects for Q3."),
            2
        );
    }

    #[test]
    fn test_currently_active_is_state_query() {
        // "is it currently active?" stays STATE_QUERY (via "currently").
        assert_eq!(detect_temporal_category("Is it currently active?"), 2);
    }

    #[test]
    fn test_disambiguation_in_confidence_path() {
        // The confidence path applies the same disambiguation.
        let (cat, _, _) =
            detect_temporal_category_with_confidence("Describe the May Department Stores company.");
        assert_eq!(cat, 0);
        let (cat, _, _) = detect_temporal_category_with_confidence("Just confirm the structure.");
        assert_eq!(cat, 0);
        let (cat, _, _) =
            detect_temporal_category_with_confidence("Apply the formula to the active variable.");
        assert_eq!(cat, 0);
        // Genuine temporal forms still classify.
        let (cat, _, _) = detect_temporal_category_with_confidence("What shipped in May 2024?");
        assert_eq!(cat, 1);
    }
}

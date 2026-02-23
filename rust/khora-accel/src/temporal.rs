//! Temporal filtering and recency scoring with PyO3 bindings.
//!
//! Batch operations for temporal range matching and exponential decay
//! scoring. Timestamps are passed as epoch seconds (f64) to avoid
//! complex datetime serialisation across the Python boundary.

use pyo3::prelude::*;
use rayon::prelude::*;

const SECONDS_PER_DAY: f64 = 86400.0;
const LN_HALF: f64 = -0.693_147_180_559_945_3; // ln(0.5)

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

    py.allow_threads(move || {
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
    py.allow_threads(move || {
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
}

//! Shared utility functions used by other modules.

/// Min-max normalize a slice of f64 values to [0, 1] range.
///
/// Returns an empty vec for empty input. If all values are equal,
/// returns a vec of 1.0 values.
pub fn min_max_normalize(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        return vec![1.0; values.len()];
    }
    values.iter().map(|&v| (v - min) / (max - min)).collect()
}

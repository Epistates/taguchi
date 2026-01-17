//! Signal-to-Noise ratio calculations for DOE analysis.
//!
//! Taguchi's S/N ratios measure both the mean and variation of responses,
//! with the goal of maximizing the S/N ratio.

use ndarray::Array2;

use super::types::{OptimizationType, SNRatioEffect};

/// Maximum S/N ratio (dB) to avoid infinite values.
const MAX_SN: f64 = 100.0;
/// Minimum S/N ratio (dB) to avoid negative infinite values.
const MIN_SN: f64 = -100.0;

/// Calculate S/N ratio for a single run.
///
/// # Arguments
/// * `values` - Replicate measurements for the run
/// * `optimization_type` - The optimization goal
/// * `target_value` - Target for nominal-is-best (uses mean if None)
///
/// # Returns
/// * S/N ratio in dB, clamped to [MIN_SN, MAX_SN]
///
/// # Formulas
/// - Larger-is-better: η = -10 * log₁₀(mean(1/y²))
/// - Smaller-is-better: η = -10 * log₁₀(mean(y²))
/// - Nominal-is-best: η = 10 * log₁₀(ȳ²/s²)
pub fn calculate_sn_ratio(
    values: &[f64],
    optimization_type: &OptimizationType,
    target_value: Option<f64>,
) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;

    let result = match optimization_type {
        OptimizationType::LargerIsBetter => {
            // η = -10 * log₁₀(Σ(1/y²)/n)
            // Filter out zeros to avoid division by zero
            let valid_values: Vec<f64> = values
                .iter()
                .filter(|&&v| v != 0.0)
                .copied()
                .collect();

            if valid_values.is_empty() {
                // All zeros - worst case for larger-is-better
                return MIN_SN;
            }

            let sum_inv_sq: f64 = valid_values.iter().map(|v| 1.0 / (v * v)).sum();
            let n_valid = valid_values.len() as f64;
            -10.0 * (sum_inv_sq / n_valid).log10()
        }
        OptimizationType::SmallerIsBetter => {
            // η = -10 * log₁₀(Σy²/n)
            let sum_sq: f64 = values.iter().map(|v| v * v).sum();

            if sum_sq == 0.0 {
                // All zeros - best case for smaller-is-better
                return MAX_SN;
            }

            -10.0 * (sum_sq / n).log10()
        }
        OptimizationType::NominalIsBest => {
            // η = 10 * log₁₀(ȳ²/s²)
            let mean = values.iter().sum::<f64>() / n;
            let target = target_value.unwrap_or(mean);

            // Calculate variance around target
            let variance: f64 = values
                .iter()
                .map(|v| (v - target).powi(2))
                .sum::<f64>() / n;

            if variance == 0.0 {
                // Perfect - all values equal target
                return MAX_SN;
            }

            if mean == 0.0 {
                // Mean is zero but variance isn't - poor performance
                return MIN_SN;
            }

            10.0 * (mean * mean / variance).log10()
        }
    };

    // Clamp to finite range and handle NaN
    if result.is_nan() {
        0.0
    } else {
        result.clamp(MIN_SN, MAX_SN)
    }
}

/// Calculate S/N ratio effects for all factors.
///
/// For each factor, calculates the mean S/N ratio at each level
/// and identifies the optimal level (highest S/N).
///
/// # Arguments
/// * `array_data` - Orthogonal array matrix (runs × factors)
/// * `response_data` - Response data (runs × replicates)
/// * `optimization_type` - The optimization goal
/// * `target_value` - Target for nominal-is-best (uses mean if None)
///
/// # Returns
/// * Vector of SNRatioEffect for each factor
pub fn calculate_sn_ratios(
    array_data: &Array2<u32>,
    response_data: &[Vec<f64>],
    optimization_type: &OptimizationType,
    target_value: Option<f64>,
) -> Vec<SNRatioEffect> {
    let num_factors = array_data.ncols();

    // Calculate S/N ratio for each run
    let run_sn_ratios: Vec<f64> = response_data
        .iter()
        .map(|reps| calculate_sn_ratio(reps, optimization_type, target_value))
        .collect();

    let mut effects: Vec<SNRatioEffect> = Vec::with_capacity(num_factors);

    for factor_idx in 0..num_factors {
        let column = array_data.column(factor_idx);
        let num_levels = column.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);

        if num_levels == 0 {
            continue;
        }

        // Sum S/N ratios for each level
        let mut level_sn_sums: Vec<f64> = vec![0.0; num_levels];
        let mut level_counts: Vec<usize> = vec![0; num_levels];

        for (run_idx, &level) in column.iter().enumerate() {
            let level_idx = level as usize;
            if level_idx < num_levels && run_idx < run_sn_ratios.len() {
                level_sn_sums[level_idx] += run_sn_ratios[run_idx];
                level_counts[level_idx] += 1;
            }
        }

        // Calculate mean S/N for each level
        let level_sn_ratios: Vec<f64> = level_sn_sums
            .iter()
            .zip(level_counts.iter())
            .map(|(&sum, &count)| {
                if count > 0 {
                    sum / count as f64
                } else {
                    0.0
                }
            })
            .collect();

        // Find optimal level (highest S/N ratio)
        let optimal_level = level_sn_ratios
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        effects.push(SNRatioEffect {
            factor_index: factor_idx,
            level_sn_ratios,
            optimal_level,
        });
    }

    effects
}

/// Calculate the grand mean of all S/N ratios.
///
/// This is needed for the additive model prediction.
///
/// # Arguments
/// * `response_data` - Response data (runs × replicates)
/// * `optimization_type` - The optimization goal
/// * `target_value` - Target for nominal-is-best
///
/// # Returns
/// * Grand mean S/N ratio
pub fn calculate_sn_grand_mean(
    response_data: &[Vec<f64>],
    optimization_type: &OptimizationType,
    target_value: Option<f64>,
) -> f64 {
    if response_data.is_empty() {
        return 0.0;
    }

    let run_sn_ratios: Vec<f64> = response_data
        .iter()
        .map(|reps| calculate_sn_ratio(reps, optimization_type, target_value))
        .collect();

    run_sn_ratios.iter().sum::<f64>() / run_sn_ratios.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sn_larger_is_better() {
        let values = vec![10.0, 20.0, 30.0];
        let sn = calculate_sn_ratio(&values, &OptimizationType::LargerIsBetter, None);

        // mean(1/y²) = (1/100 + 1/400 + 1/900) / 3 = 0.00481...
        // η = -10 * log₁₀(0.00481) ≈ 23.18
        assert!(sn > 20.0 && sn < 25.0);
    }

    #[test]
    fn test_sn_smaller_is_better() {
        let values = vec![1.0, 2.0, 3.0];
        let sn = calculate_sn_ratio(&values, &OptimizationType::SmallerIsBetter, None);

        // mean(y²) = (1 + 4 + 9) / 3 = 4.667
        // η = -10 * log₁₀(4.667) ≈ -6.69
        assert!(sn > -10.0 && sn < 0.0);
    }

    #[test]
    fn test_sn_nominal_is_best() {
        let values = vec![9.0, 10.0, 11.0];
        let sn = calculate_sn_ratio(&values, &OptimizationType::NominalIsBest, Some(10.0));

        // mean = 10, variance = 0.667
        // η = 10 * log₁₀(100 / 0.667) ≈ 21.76
        assert!(sn > 20.0 && sn < 25.0);
    }

    #[test]
    fn test_sn_edge_cases() {
        // All zeros for larger-is-better
        let sn = calculate_sn_ratio(&[0.0, 0.0], &OptimizationType::LargerIsBetter, None);
        assert_eq!(sn, MIN_SN);

        // All zeros for smaller-is-better
        let sn = calculate_sn_ratio(&[0.0, 0.0], &OptimizationType::SmallerIsBetter, None);
        assert_eq!(sn, MAX_SN);

        // Perfect for nominal-is-best
        let sn = calculate_sn_ratio(&[10.0, 10.0], &OptimizationType::NominalIsBest, Some(10.0));
        assert_eq!(sn, MAX_SN);
    }

    #[test]
    fn test_sn_ratios_l4() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        let response_data = vec![
            vec![10.0],
            vec![20.0],
            vec![30.0],
            vec![40.0],
        ];

        let effects = calculate_sn_ratios(
            &array_data,
            &response_data,
            &OptimizationType::LargerIsBetter,
            None,
        );

        assert_eq!(effects.len(), 2);

        // Each factor should have 2 levels
        assert_eq!(effects[0].level_sn_ratios.len(), 2);
        assert_eq!(effects[1].level_sn_ratios.len(), 2);

        // For larger-is-better, level 1 should be optimal for both factors
        // (since higher values give higher S/N)
        assert_eq!(effects[0].optimal_level, 1);
        assert_eq!(effects[1].optimal_level, 1);
    }

    #[test]
    fn test_sn_grand_mean() {
        let response_data = vec![
            vec![10.0],
            vec![20.0],
            vec![30.0],
            vec![40.0],
        ];

        let grand_mean = calculate_sn_grand_mean(
            &response_data,
            &OptimizationType::LargerIsBetter,
            None,
        );

        // Calculate individual S/N ratios and average
        let sn1 = calculate_sn_ratio(&[10.0], &OptimizationType::LargerIsBetter, None);
        let sn2 = calculate_sn_ratio(&[20.0], &OptimizationType::LargerIsBetter, None);
        let sn3 = calculate_sn_ratio(&[30.0], &OptimizationType::LargerIsBetter, None);
        let sn4 = calculate_sn_ratio(&[40.0], &OptimizationType::LargerIsBetter, None);

        let expected = (sn1 + sn2 + sn3 + sn4) / 4.0;
        assert!((grand_mean - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sn_with_replicates() {
        let response_data = vec![
            vec![10.0, 11.0, 12.0], // Run 1 with 3 replicates
            vec![20.0, 21.0, 22.0], // Run 2 with 3 replicates
        ];

        let sn1 = calculate_sn_ratio(&response_data[0], &OptimizationType::LargerIsBetter, None);
        let sn2 = calculate_sn_ratio(&response_data[1], &OptimizationType::LargerIsBetter, None);

        // S/N for run 2 should be higher (larger values)
        assert!(sn2 > sn1);
    }
}

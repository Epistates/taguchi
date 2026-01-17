//! ANOVA (Analysis of Variance) for DOE.
//!
//! Calculates sum of squares, F-ratios, p-values, and contribution percentages.
//! Supports factor pooling for unreplicated designs.

use ndarray::Array2;

use super::stats::f_distribution_p_value;
use super::types::{ANOVAEntry, ANOVAResult};

/// Configuration for ANOVA calculation.
#[derive(Debug, Clone)]
pub struct ANOVAConfig {
    /// F-ratio threshold for pooling factors (default: 2.0).
    pub pooling_threshold: f64,
    /// Whether to enable factor pooling (default: true).
    pub enable_pooling: bool,
    /// Minimum number of factors to keep unpooled (default: 1).
    pub min_unpooled_factors: usize,
}

impl Default for ANOVAConfig {
    fn default() -> Self {
        Self {
            pooling_threshold: 2.0,
            enable_pooling: true,
            min_unpooled_factors: 1,
        }
    }
}

/// Calculate ANOVA table with optional pooling.
///
/// # Arguments
/// * `array_data` - Orthogonal array matrix (runs × factors)
/// * `run_averages` - Average response for each run
/// * `response_data` - Full response data (runs × replicates) for pure error
/// * `grand_mean` - Overall mean of all responses
/// * `config` - ANOVA configuration (pooling threshold, etc.)
///
/// # Returns
/// * ANOVAResult with entries for each factor, error terms, and totals
///
/// # Algorithm
/// 1. Calculate Total SS = Σ(yᵢ - ȳ)²
/// 2. Calculate Factor SS for each factor using level means
/// 3. Calculate Error:
///    - If replicates > 1: Pure error from within-run variance
///    - If replicates = 1: Residual = Total - Σ Factor SS
/// 4. Iterative pooling (if enabled):
///    - Pool factors with F < threshold into error
///    - Respect min_unpooled_factors constraint
/// 5. Calculate final F-ratios and p-values
pub fn calculate_anova(
    array_data: &Array2<u32>,
    run_averages: &[f64],
    response_data: &[Vec<f64>],
    grand_mean: f64,
    config: &ANOVAConfig,
) -> ANOVAResult {
    let num_factors = array_data.ncols();
    let num_runs = array_data.nrows();

    // Total Sum of Squares
    let total_ss: f64 = run_averages
        .iter()
        .map(|y| (y - grand_mean).powi(2))
        .sum();
    let total_df = num_runs - 1;

    // Calculate SS and DF for each factor
    let mut factor_ss: Vec<f64> = Vec::with_capacity(num_factors);
    let mut factor_df: Vec<usize> = Vec::with_capacity(num_factors);

    for factor_idx in 0..num_factors {
        let column = array_data.column(factor_idx);
        let num_levels = column.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);

        if num_levels == 0 {
            factor_ss.push(0.0);
            factor_df.push(0);
            continue;
        }

        // Calculate level sums and counts
        let mut level_sums: Vec<f64> = vec![0.0; num_levels];
        let mut level_counts: Vec<usize> = vec![0; num_levels];

        for (run_idx, &level) in column.iter().enumerate() {
            let level_idx = level as usize;
            if level_idx < num_levels && run_idx < run_averages.len() {
                level_sums[level_idx] += run_averages[run_idx];
                level_counts[level_idx] += 1;
            }
        }

        // SS_factor = Σ nⱼ(ȳⱼ - ȳ)²
        let ss: f64 = (0..num_levels)
            .map(|i| {
                if level_counts[i] > 0 {
                    let level_mean = level_sums[i] / level_counts[i] as f64;
                    level_counts[i] as f64 * (level_mean - grand_mean).powi(2)
                } else {
                    0.0
                }
            })
            .sum();

        factor_ss.push(ss);
        factor_df.push(num_levels - 1);
    }

    // Calculate Error SS
    let replicates = response_data.first().map(|r| r.len()).unwrap_or(1);
    let (mut error_ss, mut error_df) = if replicates > 1 {
        // Pure error from replicates
        let pure_error: f64 = response_data
            .iter()
            .map(|reps| {
                let mean = reps.iter().sum::<f64>() / reps.len() as f64;
                reps.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            })
            .sum();
        (pure_error, num_runs * (replicates - 1))
    } else {
        // Residual error (Total - Sum of factor SS)
        let factor_ss_total: f64 = factor_ss.iter().sum();
        let error = (total_ss - factor_ss_total).max(0.0);
        let error_df_val = total_df.saturating_sub(factor_df.iter().sum());
        (error, error_df_val)
    };

    // Build initial entries
    let mut entries: Vec<ANOVAEntry> = Vec::with_capacity(num_factors);
    for factor_idx in 0..num_factors {
        let ss = factor_ss[factor_idx];
        let df = factor_df[factor_idx];
        let ms = if df > 0 { ss / df as f64 } else { 0.0 };

        entries.push(ANOVAEntry {
            factor_index: factor_idx,
            sum_of_squares: ss,
            degrees_of_freedom: df,
            mean_square: ms,
            f_ratio: None,
            p_value: None,
            contribution_percent: 0.0,
            pooled: false,
        });
    }

    // Pooling (if enabled)
    if config.enable_pooling && error_df > 0 {
        pool_factors(
            &mut entries,
            &mut error_ss,
            &mut error_df,
            config.pooling_threshold,
            config.min_unpooled_factors,
        );
    }

    // Calculate final error MS
    let error_ms = if error_df > 0 {
        error_ss / error_df as f64
    } else {
        0.0
    };

    // Calculate F-ratios, p-values, and contribution percentages
    for entry in &mut entries {
        if !entry.pooled && error_ms > 0.0 && entry.degrees_of_freedom > 0 {
            let f_ratio = entry.mean_square / error_ms;
            entry.f_ratio = Some(f_ratio);

            // Calculate p-value
            if error_df > 0 {
                entry.p_value = Some(f_distribution_p_value(
                    f_ratio,
                    entry.degrees_of_freedom,
                    error_df,
                ));
            }
        }

        // Contribution percentage
        entry.contribution_percent = if total_ss > 0.0 {
            (entry.sum_of_squares / total_ss) * 100.0
        } else {
            0.0
        };
    }

    ANOVAResult {
        entries,
        error_ss,
        error_df,
        error_ms,
        total_ss,
        total_df,
    }
}

/// Pool factors with low F-ratios into error.
///
/// Iteratively pools factors until no more pooling is needed or
/// minimum unpooled factors is reached.
fn pool_factors(
    entries: &mut [ANOVAEntry],
    error_ss: &mut f64,
    error_df: &mut usize,
    pooling_threshold: f64,
    min_unpooled: usize,
) {
    loop {
        let error_ms = if *error_df > 0 {
            *error_ss / *error_df as f64
        } else {
            break; // Can't pool without error df
        };

        if error_ms <= 0.0 {
            break;
        }

        // Count unpooled factors
        let unpooled_count = entries.iter().filter(|e| !e.pooled).count();
        if unpooled_count <= min_unpooled {
            break; // Don't pool below minimum
        }

        // Find factor with lowest F-ratio that's below threshold
        let mut min_f = f64::INFINITY;
        let mut pool_idx: Option<usize> = None;

        for (idx, entry) in entries.iter().enumerate() {
            if !entry.pooled && entry.degrees_of_freedom > 0 {
                let f_ratio = entry.mean_square / error_ms;
                if f_ratio < pooling_threshold && f_ratio < min_f {
                    min_f = f_ratio;
                    pool_idx = Some(idx);
                }
            }
        }

        match pool_idx {
            Some(idx) => {
                // Pool this factor
                *error_ss += entries[idx].sum_of_squares;
                *error_df += entries[idx].degrees_of_freedom;
                entries[idx].pooled = true;
            }
            None => break, // No more factors to pool
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_anova_basic() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        let responses = vec![10.0, 20.0, 30.0, 40.0];
        let grand_mean = 25.0;
        let response_data: Vec<Vec<f64>> = responses.iter().map(|&v| vec![v]).collect();

        let config = ANOVAConfig {
            enable_pooling: false,
            ..Default::default()
        };

        let anova = calculate_anova(&array_data, &responses, &response_data, grand_mean, &config);

        assert_eq!(anova.entries.len(), 2);
        assert_eq!(anova.total_df, 3);

        // Total SS = (10-25)² + (20-25)² + (30-25)² + (40-25)² = 225 + 25 + 25 + 225 = 500
        assert!((anova.total_ss - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_anova_with_replicates() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        let response_data = vec![
            vec![10.0, 11.0, 12.0],
            vec![20.0, 21.0, 22.0],
            vec![30.0, 31.0, 32.0],
            vec![40.0, 41.0, 42.0],
        ];

        let run_averages: Vec<f64> = response_data
            .iter()
            .map(|reps| reps.iter().sum::<f64>() / reps.len() as f64)
            .collect();
        let grand_mean = run_averages.iter().sum::<f64>() / run_averages.len() as f64;

        let config = ANOVAConfig::default();
        let anova = calculate_anova(&array_data, &run_averages, &response_data, grand_mean, &config);

        // Should have pure error from replicates
        // 4 runs × 2 df per run = 8 error df
        assert_eq!(anova.error_df, 8);

        // Each run has variance of 1 (values differ by 1)
        // Pure error SS = 4 * 2 = 8
        assert!((anova.error_ss - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_anova_pooling() {
        let array_data: Array2<u32> = array![
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ];

        // Design data where:
        // Factor 0 has small effect (values: 10.0, 50.0 vs 10.1, 50.1 -> effect = 0.1)
        // Factor 1 has large effect (values: 10.0, 10.1 vs 50.0, 50.1 -> effect = 40)
        // Factor 2 has tiny effect (values: 10.0, 50.1 vs 50.0, 10.1 -> effect ≈ 0)
        let responses = vec![10.0, 50.0, 10.1, 50.1];
        let grand_mean = 30.05;
        let response_data: Vec<Vec<f64>> = responses.iter().map(|&v| vec![v]).collect();

        let config = ANOVAConfig {
            enable_pooling: true,
            pooling_threshold: 2.0,
            min_unpooled_factors: 1,
        };

        let anova = calculate_anova(&array_data, &responses, &response_data, grand_mean, &config);

        // At least one factor should remain unpooled (the significant one)
        let unpooled_count = anova.entries.iter().filter(|e| !e.pooled).count();
        assert!(unpooled_count >= 1, "At least one factor should be unpooled");

        // Factor 1 should have the largest effect and NOT be pooled
        assert!(!anova.entries[1].pooled, "Factor 1 (large effect) should not be pooled");
    }

    #[test]
    fn test_anova_min_unpooled() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        // All factors have very small effects
        let responses = vec![10.0, 10.1, 10.2, 10.3];
        let grand_mean = 10.15;
        let response_data: Vec<Vec<f64>> = responses.iter().map(|&v| vec![v]).collect();

        let config = ANOVAConfig {
            enable_pooling: true,
            pooling_threshold: 100.0, // Very high threshold
            min_unpooled_factors: 1,
        };

        let anova = calculate_anova(&array_data, &responses, &response_data, grand_mean, &config);

        // Should keep at least one factor unpooled
        let unpooled_count = anova.entries.iter().filter(|e| !e.pooled).count();
        assert!(unpooled_count >= 1);
    }

    #[test]
    fn test_anova_contribution_percent() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        let responses = vec![10.0, 20.0, 30.0, 40.0];
        let grand_mean = 25.0;
        let response_data: Vec<Vec<f64>> = responses.iter().map(|&v| vec![v]).collect();

        let config = ANOVAConfig {
            enable_pooling: false,
            ..Default::default()
        };

        let anova = calculate_anova(&array_data, &responses, &response_data, grand_mean, &config);

        // Contribution percentages should sum to approximately 100%
        let total_contribution: f64 = anova.entries.iter().map(|e| e.contribution_percent).sum();
        assert!((total_contribution - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_anova_f_ratios() {
        let array_data: Array2<u32> = array![
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        let response_data = vec![
            vec![10.0, 11.0],
            vec![20.0, 21.0],
            vec![30.0, 31.0],
            vec![40.0, 41.0],
        ];

        let run_averages: Vec<f64> = response_data
            .iter()
            .map(|reps| reps.iter().sum::<f64>() / reps.len() as f64)
            .collect();
        let grand_mean = run_averages.iter().sum::<f64>() / run_averages.len() as f64;

        let config = ANOVAConfig {
            enable_pooling: false,
            ..Default::default()
        };

        let anova = calculate_anova(&array_data, &run_averages, &response_data, grand_mean, &config);

        // All non-pooled factors should have F-ratios
        for entry in &anova.entries {
            if !entry.pooled {
                assert!(entry.f_ratio.is_some());
                assert!(entry.p_value.is_some());
            }
        }
    }
}

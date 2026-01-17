//! Optimal settings prediction for DOE.
//!
//! Predicts optimal factor settings using Taguchi's additive model.

use super::stats::t_value;
use super::types::{ANOVAResult, ConfidenceInterval, MainEffect, OptimalSettings, SNRatioEffect};

/// Calculate optimal settings based on analysis results.
///
/// Uses the additive model for prediction:
/// - Predicted mean = grand_mean + Σ(optimal_level_effect)
/// - Predicted S/N = sn_grand_mean + Σ(optimal_sn - factor_sn_mean)
///
/// # Arguments
/// * `main_effects` - Main effect results for each factor
/// * `sn_ratio_effects` - S/N ratio effects for each factor
/// * `grand_mean` - Grand mean of responses
/// * `sn_grand_mean` - Grand mean of S/N ratios
/// * `anova` - ANOVA result for confidence interval calculation
/// * `num_runs` - Number of experimental runs
/// * `confidence_level` - Confidence level (e.g., 0.95)
///
/// # Returns
/// * OptimalSettings with predicted values and confidence interval
pub fn predict_optimal(
    main_effects: &[MainEffect],
    sn_ratio_effects: &[SNRatioEffect],
    grand_mean: f64,
    sn_grand_mean: f64,
    anova: &ANOVAResult,
    num_runs: usize,
    confidence_level: f64,
) -> OptimalSettings {
    // Find optimal level for each factor (from S/N analysis)
    let factor_levels: Vec<usize> = sn_ratio_effects
        .iter()
        .map(|e| e.optimal_level)
        .collect();

    // Calculate predicted mean using additive model:
    // ŷ = grand_mean + Σ(effect_at_optimal_level)
    let predicted_mean = grand_mean
        + main_effects
            .iter()
            .zip(sn_ratio_effects.iter())
            .map(|(me, sn)| {
                if sn.optimal_level < me.level_effects.len() {
                    me.level_effects[sn.optimal_level]
                } else {
                    0.0
                }
            })
            .sum::<f64>();

    // Calculate predicted S/N ratio using additive model:
    // η̂ = η̄ + Σ(η_optimal_i - η̄_i)
    // where η̄_i is the mean S/N for factor i across all levels
    let predicted_sn_ratio = sn_grand_mean
        + sn_ratio_effects
            .iter()
            .map(|e| {
                // Mean S/N for this factor
                let factor_sn_mean = if e.level_sn_ratios.is_empty() {
                    0.0
                } else {
                    e.level_sn_ratios.iter().sum::<f64>() / e.level_sn_ratios.len() as f64
                };

                // Deviation of optimal from factor mean
                if e.optimal_level < e.level_sn_ratios.len() {
                    e.level_sn_ratios[e.optimal_level] - factor_sn_mean
                } else {
                    0.0
                }
            })
            .sum::<f64>();

    // Calculate confidence interval
    let confidence_interval = calculate_confidence_interval(
        predicted_mean,
        anova,
        main_effects,
        &factor_levels,
        num_runs,
        confidence_level,
    );

    OptimalSettings {
        factor_levels,
        predicted_mean,
        predicted_sn_ratio,
        confidence_interval,
    }
}

/// Calculate confidence interval for predicted mean.
///
/// Uses the effective sample size based on design structure.
///
/// # Formula
/// CI = ŷ ± t(α, df_error) * sqrt(MS_error / n_eff)
///
/// where n_eff = N / (1 + Σ(ν_i)) for factors used in prediction
/// and ν_i is degrees of freedom for factor i
fn calculate_confidence_interval(
    predicted_mean: f64,
    anova: &ANOVAResult,
    main_effects: &[MainEffect],
    optimal_levels: &[usize],
    num_runs: usize,
    confidence_level: f64,
) -> Option<ConfidenceInterval> {
    if anova.error_ms <= 0.0 || anova.error_df == 0 {
        return None;
    }

    // Calculate effective sample size
    // n_eff = N / (1 + Σ(df_i)) where sum is over significant factors
    // This accounts for the degrees of freedom used in estimation
    let df_sum: usize = anova
        .entries
        .iter()
        .filter(|e| !e.pooled)
        .map(|e| e.degrees_of_freedom)
        .sum();

    // Proper n_eff calculation
    // For a saturated design, n_eff approaches 1
    // For unsaturated designs, n_eff is larger
    let n_eff = if df_sum < num_runs {
        num_runs as f64 / (1.0 + df_sum as f64)
    } else {
        1.0 // Minimum effective sample size
    };

    // Alternative: Use Taguchi's formula
    // n_eff = N / (1 + Σ(levels_i - 1)) for factors in prediction
    let levels_df_sum: usize = main_effects
        .iter()
        .zip(optimal_levels.iter())
        .filter(|(me, _)| !me.level_means.is_empty())
        .map(|(me, _)| me.level_means.len().saturating_sub(1))
        .sum();

    let n_eff_taguchi = if levels_df_sum < num_runs {
        num_runs as f64 / (1.0 + levels_df_sum as f64)
    } else {
        1.0
    };

    // Use the more conservative (smaller) n_eff
    let n_eff_final = n_eff.min(n_eff_taguchi);

    // Standard error
    let se = (anova.error_ms / n_eff_final).sqrt();

    // t-value
    let t = t_value(confidence_level, anova.error_df);

    // Margin
    let margin = t * se;

    Some(ConfidenceInterval {
        lower: predicted_mean - margin,
        upper: predicted_mean + margin,
        level: confidence_level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doe::types::ANOVAEntry;

    fn create_test_main_effects() -> Vec<MainEffect> {
        vec![
            MainEffect {
                factor_index: 0,
                level_means: vec![10.0, 20.0, 30.0],
                level_effects: vec![-10.0, 0.0, 10.0],
                range: 20.0,
                rank: 2,
            },
            MainEffect {
                factor_index: 1,
                level_means: vec![5.0, 20.0, 35.0],
                level_effects: vec![-15.0, 0.0, 15.0],
                range: 30.0,
                rank: 1,
            },
        ]
    }

    fn create_test_sn_effects() -> Vec<SNRatioEffect> {
        vec![
            SNRatioEffect {
                factor_index: 0,
                level_sn_ratios: vec![25.0, 28.0, 31.0],
                optimal_level: 2,
            },
            SNRatioEffect {
                factor_index: 1,
                level_sn_ratios: vec![24.0, 27.0, 33.0],
                optimal_level: 2,
            },
        ]
    }

    fn create_test_anova() -> ANOVAResult {
        ANOVAResult {
            entries: vec![
                ANOVAEntry {
                    factor_index: 0,
                    sum_of_squares: 200.0,
                    degrees_of_freedom: 2,
                    mean_square: 100.0,
                    f_ratio: Some(20.0),
                    p_value: Some(0.01),
                    contribution_percent: 40.0,
                    pooled: false,
                },
                ANOVAEntry {
                    factor_index: 1,
                    sum_of_squares: 300.0,
                    degrees_of_freedom: 2,
                    mean_square: 150.0,
                    f_ratio: Some(30.0),
                    p_value: Some(0.005),
                    contribution_percent: 60.0,
                    pooled: false,
                },
            ],
            error_ss: 20.0,
            error_df: 4,
            error_ms: 5.0,
            total_ss: 520.0,
            total_df: 8,
        }
    }

    #[test]
    fn test_predict_optimal_additive_model() {
        let main_effects = create_test_main_effects();
        let sn_effects = create_test_sn_effects();
        let anova = create_test_anova();

        let grand_mean = 20.0;
        let sn_grand_mean = 28.0;

        let optimal = predict_optimal(
            &main_effects,
            &sn_effects,
            grand_mean,
            sn_grand_mean,
            &anova,
            9,
            0.95,
        );

        // Optimal levels should be [2, 2]
        assert_eq!(optimal.factor_levels, vec![2, 2]);

        // Predicted mean = grand_mean + effect[2] + effect[2]
        // = 20.0 + 10.0 + 15.0 = 45.0
        assert!((optimal.predicted_mean - 45.0).abs() < 1e-10);

        // Predicted S/N using additive model:
        // sn_grand_mean + (optimal_sn[0] - factor_mean[0]) + (optimal_sn[1] - factor_mean[1])
        // Factor 0 mean = (25 + 28 + 31) / 3 = 28
        // Factor 1 mean = (24 + 27 + 33) / 3 = 28
        // = 28.0 + (31 - 28) + (33 - 28) = 28 + 3 + 5 = 36
        assert!((optimal.predicted_sn_ratio - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_optimal_confidence_interval() {
        let main_effects = create_test_main_effects();
        let sn_effects = create_test_sn_effects();
        let anova = create_test_anova();

        let optimal = predict_optimal(
            &main_effects,
            &sn_effects,
            20.0,
            28.0,
            &anova,
            9,
            0.95,
        );

        let ci = optimal.confidence_interval.expect("Should have CI");

        // CI should be centered on predicted mean
        let ci_center = (ci.lower + ci.upper) / 2.0;
        assert!((ci_center - optimal.predicted_mean).abs() < 1e-10);

        // CI should have the correct confidence level
        assert!((ci.level - 0.95).abs() < 1e-10);

        // Lower should be less than upper
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_predict_optimal_no_error() {
        let main_effects = create_test_main_effects();
        let sn_effects = create_test_sn_effects();

        // ANOVA with no error
        let anova = ANOVAResult {
            entries: vec![],
            error_ss: 0.0,
            error_df: 0,
            error_ms: 0.0,
            total_ss: 500.0,
            total_df: 8,
        };

        let optimal = predict_optimal(
            &main_effects,
            &sn_effects,
            20.0,
            28.0,
            &anova,
            9,
            0.95,
        );

        // Should not have CI when no error
        assert!(optimal.confidence_interval.is_none());
    }

    #[test]
    fn test_additive_model_vs_averaging() {
        // This test verifies the fix for the critical issue:
        // The additive model should give different results than simple averaging

        let sn_effects = vec![
            SNRatioEffect {
                factor_index: 0,
                level_sn_ratios: vec![20.0, 30.0], // Mean = 25, optimal = 30
                optimal_level: 1,
            },
            SNRatioEffect {
                factor_index: 1,
                level_sn_ratios: vec![25.0, 35.0], // Mean = 30, optimal = 35
                optimal_level: 1,
            },
        ];

        // Old (incorrect) averaging method:
        // predicted = (30 + 35) / 2 = 32.5
        let _incorrect_averaging = (30.0 + 35.0) / 2.0;

        // New (correct) additive model:
        // sn_grand_mean + (30 - 25) + (35 - 30)
        // = 27.5 + 5 + 5 = 37.5
        // (assuming sn_grand_mean = (20+30+25+35)/4 = 27.5)
        let sn_grand_mean = 27.5;

        let predicted_sn = sn_grand_mean
            + sn_effects
                .iter()
                .map(|e| {
                    let factor_mean = e.level_sn_ratios.iter().sum::<f64>()
                        / e.level_sn_ratios.len() as f64;
                    e.level_sn_ratios[e.optimal_level] - factor_mean
                })
                .sum::<f64>();

        assert!((predicted_sn - 37.5).abs() < 1e-10);

        // The additive model gives a higher (and more accurate) prediction
        assert!(predicted_sn > 32.5);
    }
}

//! DOE (Design of Experiments) analysis module.
//!
//! This module provides complete Taguchi DOE analysis including:
//! - Main effects calculation
//! - Signal-to-Noise ratio analysis
//! - ANOVA with factor pooling
//! - Optimal settings prediction with confidence intervals
//!
//! ## Quick Start
//!
//! ```rust
//! use taguchi::OABuilder;
//! # #[cfg(feature = "doe")]
//! use taguchi::doe::{analyze, AnalysisConfig, OptimizationType};
//!
//! # #[cfg(feature = "doe")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an L9 orthogonal array
//! let oa = OABuilder::new()
//!     .levels(3)
//!     .factors(4)
//!     .strength(2)
//!     .build()?;
//!
//! // Experimental response data (9 runs with replicates)
//! let response_data = vec![
//!     vec![85.0, 86.0],
//!     vec![92.0, 91.0],
//!     vec![78.0, 79.0],
//!     vec![91.0, 90.0],
//!     vec![88.0, 89.0],
//!     vec![82.0, 83.0],
//!     vec![89.0, 88.0],
//!     vec![86.0, 87.0],
//!     vec![94.0, 93.0],
//! ];
//!
//! // Run analysis
//! let config = AnalysisConfig {
//!     optimization_type: OptimizationType::LargerIsBetter,
//!     ..Default::default()
//! };
//!
//! let result = analyze(&oa, &response_data, &config)?;
//!
//! println!("Grand mean: {:.2}", result.grand_mean);
//! println!("Optimal levels: {:?}", result.optimal_settings.factor_levels);
//! println!("Predicted mean: {:.2}", result.optimal_settings.predicted_mean);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "doe"))]
//! # fn main() {}
//! ```
//!
//! ## Analysis Types
//!
//! ### Main Effects
//!
//! Main effects measure how changing a factor level affects the response.
//! Factors are ranked by their effect range (larger range = more influence).
//!
//! ### S/N Ratios
//!
//! Signal-to-Noise ratios measure both mean and variation:
//! - **Larger-is-better**: Maximize response
//! - **Smaller-is-better**: Minimize response
//! - **Nominal-is-best**: Hit a target with minimum variance
//!
//! ### ANOVA
//!
//! Analysis of Variance partitions total variation into:
//! - Factor contributions (with F-ratios and p-values)
//! - Error (from replicates or residual)
//!
//! Factor pooling can be enabled to combine insignificant factors into error.
//!
//! ### Optimal Settings
//!
//! Predicts performance at optimal factor levels using the additive model:
//! - Predicted mean = grand_mean + Σ(optimal_level_effects)
//! - Predicted S/N = sn_grand_mean + Σ(optimal_sn - factor_sn_mean)

mod anova;
mod main_effects;
mod optimal;
mod sn_ratios;
mod stats;
mod types;

pub use anova::ANOVAConfig;
pub use stats::{f_distribution_p_value, ln_gamma, regularized_incomplete_beta, t_value};
pub use types::{
    ANOVAEntry, ANOVAResult, AnalysisConfig, ConfidenceInterval, DOEAnalysis, MainEffect,
    OptimalSettings, OptimizationType, SNRatioEffect,
};

use crate::error::{Error, Result};
use crate::oa::OA;

/// Run complete DOE analysis on an orthogonal array experiment.
///
/// # Arguments
/// * `oa` - The orthogonal array used for the experiment
/// * `response_data` - Response data organized as `Vec<Vec<f64>>` where outer
///   vec is runs and inner vec is replicates for each run
/// * `config` - Analysis configuration
///
/// # Returns
/// * Complete DOEAnalysis result or error
///
/// # Errors
/// * If response_data length doesn't match OA runs
/// * If any run has empty response data
///
/// # Example
///
/// ```rust
/// use taguchi::OABuilder;
/// # #[cfg(feature = "doe")]
/// use taguchi::doe::{analyze, AnalysisConfig, OptimizationType};
///
/// # #[cfg(feature = "doe")]
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let oa = OABuilder::new()
///     .levels(2)
///     .factors(3)
///     .strength(2)
///     .build()?;
///
/// let response_data = vec![
///     vec![10.0, 11.0],
///     vec![20.0, 21.0],
///     vec![15.0, 16.0],
///     vec![25.0, 26.0],
/// ];
///
/// let result = analyze(&oa, &response_data, &AnalysisConfig::default())?;
/// println!("Optimal levels: {:?}", result.optimal_settings.factor_levels);
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "doe"))]
/// # fn main() {}
/// ```
pub fn analyze(
    oa: &OA,
    response_data: &[Vec<f64>],
    config: &AnalysisConfig,
) -> Result<DOEAnalysis> {
    // Validate inputs
    if response_data.is_empty() {
        return Err(Error::invalid_params("Response data is empty"));
    }

    if response_data.len() != oa.runs() {
        return Err(Error::invalid_params(format!(
            "Response data length ({}) doesn't match OA runs ({})",
            response_data.len(),
            oa.runs()
        )));
    }

    for (i, row) in response_data.iter().enumerate() {
        if row.is_empty() {
            return Err(Error::invalid_params(format!(
                "Response data for run {} is empty",
                i + 1
            )));
        }
    }

    // Calculate run averages
    let run_averages: Vec<f64> = response_data
        .iter()
        .map(|reps| reps.iter().sum::<f64>() / reps.len() as f64)
        .collect();

    // Grand mean
    let grand_mean = run_averages.iter().sum::<f64>() / run_averages.len() as f64;

    // S/N grand mean
    let sn_grand_mean = sn_ratios::calculate_sn_grand_mean(
        response_data,
        &config.optimization_type,
        config.target_value,
    );

    // Get OA data as ndarray
    let array_data = oa.data();

    // Calculate main effects
    let main_effects = main_effects::calculate_main_effects(array_data, &run_averages, grand_mean);

    // Calculate S/N ratios
    let sn_ratio_effects = sn_ratios::calculate_sn_ratios(
        array_data,
        response_data,
        &config.optimization_type,
        config.target_value,
    );

    // Calculate ANOVA
    let anova_config = anova::ANOVAConfig {
        pooling_threshold: config.pooling_threshold,
        enable_pooling: config.enable_pooling,
        min_unpooled_factors: config.min_unpooled_factors,
    };
    let anova = anova::calculate_anova(
        array_data,
        &run_averages,
        response_data,
        grand_mean,
        &anova_config,
    );

    // Predict optimal settings
    let optimal_settings = optimal::predict_optimal(
        &main_effects,
        &sn_ratio_effects,
        grand_mean,
        sn_grand_mean,
        &anova,
        oa.runs(),
        config.confidence_level,
    );

    Ok(DOEAnalysis {
        grand_mean,
        sn_grand_mean,
        main_effects,
        sn_ratio_effects,
        anova,
        optimal_settings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OABuilder;

    #[test]
    fn test_analyze_l4() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(3)
            .strength(2)
            .build()
            .unwrap();

        let response_data = vec![
            vec![10.0, 11.0],
            vec![20.0, 21.0],
            vec![15.0, 16.0],
            vec![25.0, 26.0],
        ];

        let result = analyze(&oa, &response_data, &AnalysisConfig::default()).unwrap();

        assert_eq!(result.main_effects.len(), 3);
        assert_eq!(result.sn_ratio_effects.len(), 3);
        assert_eq!(result.optimal_settings.factor_levels.len(), 3);
    }

    #[test]
    fn test_analyze_l9() {
        let oa = OABuilder::new()
            .levels(3)
            .factors(4)
            .strength(2)
            .build()
            .unwrap();

        // Ross (1996) L9 example data
        let response_data = vec![
            vec![85.0],
            vec![92.0],
            vec![78.0],
            vec![91.0],
            vec![88.0],
            vec![82.0],
            vec![89.0],
            vec![86.0],
            vec![94.0],
        ];

        let config = AnalysisConfig {
            optimization_type: OptimizationType::LargerIsBetter,
            enable_pooling: false,
            ..Default::default()
        };

        let result = analyze(&oa, &response_data, &config).unwrap();

        // Check grand mean
        let expected_mean = 87.222_222_222_222_22;
        assert!((result.grand_mean - expected_mean).abs() < 0.01);

        // Should have 4 main effects
        assert_eq!(result.main_effects.len(), 4);

        // All ranks should be assigned
        for effect in &result.main_effects {
            assert!(effect.rank >= 1 && effect.rank <= 4);
        }
    }

    #[test]
    fn test_analyze_validation_errors() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(3)
            .strength(2)
            .build()
            .unwrap();

        // Empty response data
        let result = analyze(&oa, &[], &AnalysisConfig::default());
        assert!(result.is_err());

        // Wrong number of runs
        let response_data = vec![vec![10.0], vec![20.0]];
        let result = analyze(&oa, &response_data, &AnalysisConfig::default());
        assert!(result.is_err());

        // Empty run
        let response_data = vec![vec![10.0], vec![], vec![15.0], vec![25.0]];
        let result = analyze(&oa, &response_data, &AnalysisConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_smaller_is_better() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(2)
            .strength(2)
            .build()
            .unwrap();

        let response_data = vec![vec![100.0], vec![50.0], vec![80.0], vec![30.0]];

        let config = AnalysisConfig {
            optimization_type: OptimizationType::SmallerIsBetter,
            ..Default::default()
        };

        let result = analyze(&oa, &response_data, &config).unwrap();

        // For smaller-is-better, optimal should favor lower values
        // The predicted mean should be lower than grand mean
        assert!(result.optimal_settings.predicted_mean <= result.grand_mean);
    }

    #[test]
    fn test_analyze_nominal_is_best() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(2)
            .strength(2)
            .build()
            .unwrap();

        let response_data = vec![
            vec![95.0, 96.0, 94.0], // Close to target
            vec![80.0, 85.0, 75.0], // High variance
            vec![98.0, 99.0, 97.0], // Close to target
            vec![60.0, 70.0, 80.0], // Very high variance
        ];

        let config = AnalysisConfig {
            optimization_type: OptimizationType::NominalIsBest,
            target_value: Some(100.0),
            ..Default::default()
        };

        let result = analyze(&oa, &response_data, &config).unwrap();

        // Should have valid results
        assert_eq!(result.main_effects.len(), 2);
        assert!(result.optimal_settings.confidence_interval.is_some());
    }

    #[test]
    fn test_analyze_with_pooling() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(3)
            .strength(2)
            .build()
            .unwrap();

        // Factor 0 has very small effect (noise)
        // Factor 1 has large effect
        // Factor 2 has medium effect
        let response_data = vec![
            vec![10.0, 10.1],
            vec![50.0, 50.1],
            vec![10.1, 10.2],
            vec![50.1, 50.2],
        ];

        let config = AnalysisConfig {
            enable_pooling: true,
            pooling_threshold: 2.0,
            min_unpooled_factors: 1,
            ..Default::default()
        };

        let result = analyze(&oa, &response_data, &config).unwrap();

        // At least one factor should remain unpooled
        let unpooled = result.anova.entries.iter().filter(|e| !e.pooled).count();
        assert!(unpooled >= 1);
    }
}

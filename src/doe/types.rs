//! DOE analysis types.
//!
//! Core types for Taguchi Design of Experiments analysis.

/// Optimization goal for Taguchi analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimizationType {
    /// Maximize the response value.
    /// S/N = -10 * log10(mean(1/y^2))
    LargerIsBetter,
    /// Minimize the response value.
    /// S/N = -10 * log10(mean(y^2))
    SmallerIsBetter,
    /// Hit a target value with minimum variance.
    /// S/N = 10 * log10(mean^2/variance)
    NominalIsBest,
}

impl Default for OptimizationType {
    fn default() -> Self {
        Self::LargerIsBetter
    }
}

/// Main effect analysis result for a single factor.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MainEffect {
    /// Factor index (0-based column in OA).
    pub factor_index: usize,
    /// Mean response at each level.
    pub level_means: Vec<f64>,
    /// Effect at each level (level_mean - grand_mean).
    pub level_effects: Vec<f64>,
    /// Range of level means (max - min).
    pub range: f64,
    /// Rank by importance (1 = most important).
    pub rank: usize,
}

/// Signal-to-Noise ratio analysis result for a single factor.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SNRatioEffect {
    /// Factor index (0-based column in OA).
    pub factor_index: usize,
    /// Mean S/N ratio at each level.
    pub level_sn_ratios: Vec<f64>,
    /// Optimal level (index with highest S/N ratio).
    pub optimal_level: usize,
}

/// ANOVA table entry for a single factor.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ANOVAEntry {
    /// Factor index (0-based column in OA).
    pub factor_index: usize,
    /// Sum of squares for this factor.
    pub sum_of_squares: f64,
    /// Degrees of freedom.
    pub degrees_of_freedom: usize,
    /// Mean square (SS / df).
    pub mean_square: f64,
    /// F-ratio (MS_factor / MS_error), None if pooled.
    pub f_ratio: Option<f64>,
    /// P-value from F-distribution, None if pooled.
    pub p_value: Option<f64>,
    /// Percent contribution to total variance.
    pub contribution_percent: f64,
    /// Whether this factor was pooled into error.
    pub pooled: bool,
}

/// Complete ANOVA result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ANOVAResult {
    /// ANOVA entries for each factor.
    pub entries: Vec<ANOVAEntry>,
    /// Error sum of squares.
    pub error_ss: f64,
    /// Error degrees of freedom.
    pub error_df: usize,
    /// Error mean square.
    pub error_ms: f64,
    /// Total sum of squares.
    pub total_ss: f64,
    /// Total degrees of freedom.
    pub total_df: usize,
}

/// Confidence interval.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceInterval {
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%).
    pub level: f64,
}

/// Optimal settings prediction result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimalSettings {
    /// Optimal level for each factor (index by factor_index).
    pub factor_levels: Vec<usize>,
    /// Predicted mean response at optimal settings.
    pub predicted_mean: f64,
    /// Predicted S/N ratio at optimal settings (using additive model).
    pub predicted_sn_ratio: f64,
    /// Confidence interval for predicted mean.
    pub confidence_interval: Option<ConfidenceInterval>,
}

/// Complete DOE analysis result.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DOEAnalysis {
    /// Grand mean of all responses.
    pub grand_mean: f64,
    /// Grand mean of all S/N ratios.
    pub sn_grand_mean: f64,
    /// Main effects for each factor.
    pub main_effects: Vec<MainEffect>,
    /// S/N ratio effects for each factor.
    pub sn_ratio_effects: Vec<SNRatioEffect>,
    /// ANOVA result.
    pub anova: ANOVAResult,
    /// Optimal settings prediction.
    pub optimal_settings: OptimalSettings,
}

/// Configuration for DOE analysis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnalysisConfig {
    /// Optimization type (larger/smaller/nominal is better).
    pub optimization_type: OptimizationType,
    /// Target value for nominal-is-best (uses mean if None).
    pub target_value: Option<f64>,
    /// F-ratio threshold for pooling factors into error (default: 2.0).
    pub pooling_threshold: f64,
    /// Whether to enable factor pooling (default: true).
    pub enable_pooling: bool,
    /// Minimum number of factors to keep unpooled (default: 1).
    pub min_unpooled_factors: usize,
    /// Confidence level for intervals (default: 0.95).
    pub confidence_level: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            optimization_type: OptimizationType::LargerIsBetter,
            target_value: None,
            pooling_threshold: 2.0,
            enable_pooling: true,
            min_unpooled_factors: 1,
            confidence_level: 0.95,
        }
    }
}

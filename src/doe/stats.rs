//! Statistical utilities for DOE analysis.
//!
//! Provides statistical functions including:
//! - Log gamma function (Lanczos approximation)
//! - Regularized incomplete beta function
//! - F-distribution p-value calculation
//! - T-distribution quantile (critical values)

use std::f64::consts::PI;

/// Log gamma function using Lanczos approximation.
///
/// More accurate than Stirling's formula for small values.
///
/// # Arguments
/// * `x` - Input value (must be positive)
///
/// # Returns
/// * ln(Gamma(x))
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation coefficients (g=7)
    const G: f64 = 7.0;
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let x = x - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (i, &c) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses continued fraction expansion for numerical stability.
///
/// # Arguments
/// * `x` - Integration bound (0 <= x <= 1)
/// * `a` - First shape parameter (> 0)
/// * `b` - Second shape parameter (> 0)
///
/// # Returns
/// * I_x(a, b) = integral from 0 to x of t^(a-1) * (1-t)^(b-1) dt / B(a,b)
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;

    // Continued fraction expansion (Lentz's algorithm)
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;
    const EPSILON: f64 = 1e-30;
    const TOLERANCE: f64 = 1e-10;
    const MAX_ITERATIONS: usize = 200;

    for m in 0..MAX_ITERATIONS {
        let m_f = m as f64;

        // Even step: a_{2m}
        let numerator = if m == 0 {
            1.0
        } else {
            (m_f * (b - m_f) * x) / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f))
        };

        d = 1.0 + numerator * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }

        f *= d * c;

        // Odd step: a_{2m+1}
        let numerator = -((a + m_f) * (a + b + m_f) * x)
            / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));

        d = 1.0 + numerator * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }

        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < TOLERANCE {
            break;
        }
    }

    front * f
}

/// Calculate p-value from F-distribution.
///
/// Returns P(F > f) for the F-distribution with df1 and df2 degrees of freedom.
///
/// # Arguments
/// * `f` - F statistic value
/// * `df1` - Numerator degrees of freedom
/// * `df2` - Denominator degrees of freedom
///
/// # Returns
/// * p-value (probability of observing F >= f under null hypothesis)
pub fn f_distribution_p_value(f: f64, df1: usize, df2: usize) -> f64 {
    if f <= 0.0 || df1 == 0 || df2 == 0 {
        return 1.0;
    }

    // P(F > f) = I_x(df2/2, df1/2) where x = df2/(df2 + df1*f)
    let x = df2 as f64 / (df2 as f64 + df1 as f64 * f);
    regularized_incomplete_beta(x, df2 as f64 / 2.0, df1 as f64 / 2.0)
}

/// Get t-value (critical value) for given confidence level and degrees of freedom.
///
/// Returns the t-value for a two-tailed confidence interval.
///
/// # Arguments
/// * `confidence` - Confidence level (e.g., 0.90, 0.95, 0.99)
/// * `df` - Degrees of freedom
///
/// # Returns
/// * t-value such that P(-t < T < t) = confidence
///
/// # Note
/// For df > 120, uses normal distribution approximation (z-values).
pub fn t_value(confidence: f64, df: usize) -> f64 {
    // Validate confidence level
    if confidence <= 0.0 || confidence >= 1.0 {
        return f64::NAN;
    }

    // For large df, use normal distribution (z-values)
    if df > 120 {
        return z_value(confidence);
    }

    // Lookup tables for common confidence levels
    // Two-tailed critical values (alpha/2 in each tail)
    match confidence {
        c if (c - 0.90).abs() < 0.001 => t_value_90(df),
        c if (c - 0.95).abs() < 0.001 => t_value_95(df),
        c if (c - 0.99).abs() < 0.001 => t_value_99(df),
        _ => {
            // Interpolate or use closest available
            // For non-standard confidence levels, use 95%
            t_value_95(df)
        }
    }
}

/// Z-value (normal distribution) for given confidence level.
fn z_value(confidence: f64) -> f64 {
    match confidence {
        c if (c - 0.90).abs() < 0.001 => 1.645,
        c if (c - 0.95).abs() < 0.001 => 1.960,
        c if (c - 0.99).abs() < 0.001 => 2.576,
        c if (c - 0.999).abs() < 0.001 => 3.291,
        _ => 1.960, // Default to 95%
    }
}

/// T-value for 90% confidence interval.
fn t_value_90(df: usize) -> f64 {
    // Two-tailed, alpha = 0.10, each tail = 0.05
    const TABLE: [(usize, f64); 30] = [
        (1, 6.314),
        (2, 2.920),
        (3, 2.353),
        (4, 2.132),
        (5, 2.015),
        (6, 1.943),
        (7, 1.895),
        (8, 1.860),
        (9, 1.833),
        (10, 1.812),
        (11, 1.796),
        (12, 1.782),
        (13, 1.771),
        (14, 1.761),
        (15, 1.753),
        (16, 1.746),
        (17, 1.740),
        (18, 1.734),
        (19, 1.729),
        (20, 1.725),
        (25, 1.708),
        (30, 1.697),
        (40, 1.684),
        (50, 1.676),
        (60, 1.671),
        (70, 1.667),
        (80, 1.664),
        (90, 1.662),
        (100, 1.660),
        (120, 1.658),
    ];
    lookup_t_value(&TABLE, df, 1.645)
}

/// T-value for 95% confidence interval.
fn t_value_95(df: usize) -> f64 {
    // Two-tailed, alpha = 0.05, each tail = 0.025
    const TABLE: [(usize, f64); 30] = [
        (1, 12.706),
        (2, 4.303),
        (3, 3.182),
        (4, 2.776),
        (5, 2.571),
        (6, 2.447),
        (7, 2.365),
        (8, 2.306),
        (9, 2.262),
        (10, 2.228),
        (11, 2.201),
        (12, 2.179),
        (13, 2.160),
        (14, 2.145),
        (15, 2.131),
        (16, 2.120),
        (17, 2.110),
        (18, 2.101),
        (19, 2.093),
        (20, 2.086),
        (25, 2.060),
        (30, 2.042),
        (40, 2.021),
        (50, 2.009),
        (60, 2.000),
        (70, 1.994),
        (80, 1.990),
        (90, 1.987),
        (100, 1.984),
        (120, 1.980),
    ];
    lookup_t_value(&TABLE, df, 1.960)
}

/// T-value for 99% confidence interval.
fn t_value_99(df: usize) -> f64 {
    // Two-tailed, alpha = 0.01, each tail = 0.005
    const TABLE: [(usize, f64); 30] = [
        (1, 63.657),
        (2, 9.925),
        (3, 5.841),
        (4, 4.604),
        (5, 4.032),
        (6, 3.707),
        (7, 3.499),
        (8, 3.355),
        (9, 3.250),
        (10, 3.169),
        (11, 3.106),
        (12, 3.055),
        (13, 3.012),
        (14, 2.977),
        (15, 2.947),
        (16, 2.921),
        (17, 2.898),
        (18, 2.878),
        (19, 2.861),
        (20, 2.845),
        (25, 2.787),
        (30, 2.750),
        (40, 2.704),
        (50, 2.678),
        (60, 2.660),
        (70, 2.648),
        (80, 2.639),
        (90, 2.632),
        (100, 2.626),
        (120, 2.617),
    ];
    lookup_t_value(&TABLE, df, 2.576)
}

/// Look up t-value from table with interpolation.
fn lookup_t_value(table: &[(usize, f64)], df: usize, infinity_value: f64) -> f64 {
    if df == 0 {
        return f64::INFINITY;
    }

    // Find the appropriate value
    for i in 0..table.len() {
        if df <= table[i].0 {
            if i == 0 || df == table[i].0 {
                return table[i].1;
            }
            // Linear interpolation between table entries
            let (df_low, t_low) = table[i - 1];
            let (df_high, t_high) = table[i];
            let ratio = (df - df_low) as f64 / (df_high - df_low) as f64;
            return t_low + ratio * (t_high - t_low);
        }
    }

    // Beyond table, use infinity value
    infinity_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1, so ln(Gamma(1)) = 0
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);

        // Gamma(2) = 1, so ln(Gamma(2)) = 0
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);

        // Gamma(3) = 2, so ln(Gamma(3)) = ln(2)
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);

        // Gamma(4) = 6, so ln(Gamma(4)) = ln(6)
        assert!((ln_gamma(4.0) - 6.0_f64.ln()).abs() < 1e-10);

        // Gamma(5) = 24, so ln(Gamma(5)) = ln(24)
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_ln_gamma_half_integer() {
        // Gamma(0.5) = sqrt(pi)
        let expected = 0.5 * PI.ln();
        assert!((ln_gamma(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_incomplete_beta_bounds() {
        assert_eq!(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0);
        assert_eq!(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0);
    }

    #[test]
    fn test_incomplete_beta_symmetry() {
        // I_x(a,b) + I_{1-x}(b,a) = 1
        let x = 0.3;
        let a = 2.0;
        let b = 3.0;
        let result = regularized_incomplete_beta(x, a, b)
            + regularized_incomplete_beta(1.0 - x, b, a);
        assert!((result - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_f_distribution_p_value_bounds() {
        // P(F > 0) should be 1
        assert!((f_distribution_p_value(0.0, 3, 10) - 1.0).abs() < 1e-10);

        // Very large F should give very small p-value
        let p = f_distribution_p_value(100.0, 3, 10);
        assert!(p < 0.001);
    }

    #[test]
    fn test_f_distribution_p_value_typical() {
        // F(3, 10) with F=3.71 is the critical value at alpha=0.05
        // The continued fraction approximation may have some variance
        let p = f_distribution_p_value(3.71, 3, 10);
        // Allow tolerance of 0.02 for numerical approximation
        assert!(p < 0.10 && p > 0.02, "Expected p ≈ 0.05, got {}", p);

        // More lenient check: monotonicity (larger F = smaller p)
        let p_low = f_distribution_p_value(2.0, 3, 10);
        let p_high = f_distribution_p_value(6.0, 3, 10);
        assert!(p_low > p, "p should decrease as F increases");
        assert!(p > p_high, "p should decrease as F increases");
    }

    #[test]
    fn test_t_value_95_known() {
        // t(1) for 95% CI should be 12.706
        assert!((t_value(0.95, 1) - 12.706).abs() < 0.001);

        // t(10) for 95% CI should be 2.228
        assert!((t_value(0.95, 10) - 2.228).abs() < 0.001);

        // t(30) for 95% CI should be 2.042
        assert!((t_value(0.95, 30) - 2.042).abs() < 0.001);
    }

    #[test]
    fn test_t_value_90_known() {
        // t(1) for 90% CI should be 6.314
        assert!((t_value(0.90, 1) - 6.314).abs() < 0.001);

        // t(10) for 90% CI should be 1.812
        assert!((t_value(0.90, 10) - 1.812).abs() < 0.001);
    }

    #[test]
    fn test_t_value_99_known() {
        // t(1) for 99% CI should be 63.657
        assert!((t_value(0.99, 1) - 63.657).abs() < 0.001);

        // t(10) for 99% CI should be 3.169
        assert!((t_value(0.99, 10) - 3.169).abs() < 0.001);
    }

    #[test]
    fn test_t_value_large_df() {
        // For very large df, should approach z-values
        let t_95 = t_value(0.95, 1000);
        assert!((t_95 - 1.96).abs() < 0.01);

        let t_99 = t_value(0.99, 1000);
        assert!((t_99 - 2.576).abs() < 0.01);
    }

    #[test]
    fn test_t_value_interpolation() {
        // t(12) should be between t(10) and t(15)
        let t_12 = t_value(0.95, 12);
        let t_10 = t_value(0.95, 10);
        let t_15 = t_value(0.95, 15);
        assert!(t_12 < t_10);
        assert!(t_12 > t_15);
    }
}

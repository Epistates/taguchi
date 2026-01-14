//! Orthogonal array verification algorithms.
//!
//! This module provides functions to verify that an array has the claimed
//! orthogonal array properties, particularly the strength property.

use std::collections::HashMap;

use crate::error::Result;
use crate::utils::combinations;

use super::OA;

/// Result of verifying an orthogonal array.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the array passes verification.
    pub is_valid: bool,
    /// The claimed strength.
    pub claimed_strength: u32,
    /// The actual verified strength (highest t for which the array is balanced).
    pub actual_strength: u32,
    /// Details about any issues found.
    pub issues: Vec<VerificationIssue>,
}

/// A specific issue found during verification.
#[derive(Debug, Clone)]
pub enum VerificationIssue {
    /// A value is out of the valid range.
    ValueOutOfRange {
        row: usize,
        col: usize,
        value: u32,
        max: u32,
    },
    /// A subarray is not balanced (some t-tuples appear more/less often than expected).
    ImbalancedSubarray {
        columns: Vec<usize>,
        expected_count: usize,
        tuple_counts: HashMap<Vec<u32>, usize>,
    },
}

/// Verify that an orthogonal array has the claimed strength.
///
/// This function checks that for every selection of `strength` columns,
/// all possible tuples of levels appear the same number of times.
///
/// # Algorithm
///
/// For strength t:
/// 1. For each combination of t columns
/// 2. Count occurrences of each t-tuple
/// 3. Verify all counts equal λ = N / s^t
///
/// Time complexity: O(N * C(k, t) * t) where C(k, t) is binomial coefficient.
///
/// # Errors
///
/// Returns an error if verification encounters an unexpected condition.
pub fn verify_strength(oa: &OA, strength: u32) -> Result<VerificationResult> {
    let mut issues = Vec::new();
    let levels = oa.levels_vec();
    let runs = oa.runs();
    let factors = oa.factors();

    // First check all values are in range
    for row in 0..runs {
        for col in 0..factors {
            let val = oa.get(row, col);
            if val >= levels[col] {
                issues.push(VerificationIssue::ValueOutOfRange {
                    row,
                    col,
                    value: val,
                    max: levels[col] - 1,
                });
            }
        }
    }

    if !issues.is_empty() {
        return Ok(VerificationResult {
            is_valid: false,
            claimed_strength: strength,
            actual_strength: 0,
            issues,
        });
    }

    // Verify balance for each strength level from 1 to claimed strength
    let mut verified_strength = 0;

    for t in 1..=strength {
        if t as usize > factors {
            break;
        }

        let mut balanced_at_t = true;

        // Check all C(k, t) combinations of t columns
        for col_combo in combinations(factors, t as usize) {
            // Calculate s_to_t for this combination
            let s_to_t: usize = col_combo.iter().map(|&c| levels[c] as usize).product();

            if runs % s_to_t != 0 {
                // Can't have strength t for this combo if N is not divisible by product of levels
                balanced_at_t = false;
                issues.push(VerificationIssue::ImbalancedSubarray {
                    columns: col_combo,
                    expected_count: 0, // Placeholder
                    tuple_counts: HashMap::new(),
                });
                continue;
            }

            let expected_count = runs / s_to_t;
            let mut tuple_counts: HashMap<Vec<u32>, usize> = HashMap::new();

            // Count occurrences of each t-tuple
            for row in 0..runs {
                let tuple: Vec<u32> = col_combo.iter().map(|&c| oa.get(row, c)).collect();
                *tuple_counts.entry(tuple).or_insert(0) += 1;
            }

            // Verify all possible tuples appear exactly expected_count times
            let num_tuples = tuple_counts.len();
            let all_equal = tuple_counts.values().all(|&c| c == expected_count);

            if num_tuples != s_to_t || !all_equal {
                balanced_at_t = false;
                issues.push(VerificationIssue::ImbalancedSubarray {
                    columns: col_combo,
                    expected_count,
                    tuple_counts,
                });
            }
        }

        if balanced_at_t {
            verified_strength = t;
        } else {
            break;
        }
    }

    let is_valid = issues.is_empty() && verified_strength >= strength;

    Ok(VerificationResult {
        is_valid,
        claimed_strength: strength,
        actual_strength: verified_strength,
        issues,
    })
}

/// Compute the actual strength of an orthogonal array.
///
/// This finds the highest value of t for which the array is balanced.
pub fn compute_strength(oa: &OA, max_check: u32) -> Result<u32> {
    let max_t = max_check.min(oa.factors() as u32);

    for t in (1..=max_t).rev() {
        let result = verify_strength(oa, t)?;
        if result.is_valid {
            return Ok(t);
        }
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::oa::OAParams;

    fn make_l4() -> OA {
        // L4(2^3) - standard Taguchi array
        let params = OAParams::new(4, 3, 2, 2).unwrap();
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                0, 0, 0,
                0, 1, 1,
                1, 0, 1,
                1, 1, 0,
            ],
        )
        .unwrap();
        OA::new(data, params)
    }

    #[test]
    fn test_verify_l4() {
        let oa = make_l4();
        let result = verify_strength(&oa, 2).unwrap();

        assert!(result.is_valid);
        assert_eq!(result.actual_strength, 2);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_verify_invalid_values() {
        let params = OAParams::new(4, 3, 2, 2).unwrap();
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                0, 0, 0,
                0, 1, 2,  // 2 is out of range for levels=2
                1, 0, 1,
                1, 1, 0,
            ],
        )
        .unwrap();
        let oa = OA::new(data, params);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(!result.is_valid);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_verify_imbalanced() {
        // Create an array that is NOT balanced
        let params = OAParams::new(4, 3, 2, 2).unwrap();
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                0, 0, 0,
                0, 0, 0,  // Duplicate row
                1, 0, 1,
                1, 1, 0,
            ],
        )
        .unwrap();
        let oa = OA::new(data, params);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(!result.is_valid);
        assert_eq!(result.actual_strength, 0); // Fails even strength 1
    }

    #[test]
    fn test_compute_strength() {
        let oa = make_l4();
        let strength = compute_strength(&oa, 10).unwrap();
        assert_eq!(strength, 2);
    }
}

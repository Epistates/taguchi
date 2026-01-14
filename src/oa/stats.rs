//! Statistical analysis for orthogonal arrays.
//!
//! This module provides utilities for analyzing the statistical properties
//! of orthogonal arrays, such as balance, correlation, and space-filling metrics.

use ndarray::Array2;
use std::collections::HashMap;

use super::OA;

/// A report on the balance of an orthogonal array.
#[derive(Debug, Clone)]
pub struct BalanceReport {
    /// Whether each factor is perfectly balanced (each level appears equally).
    pub factor_balance: Vec<bool>,
    /// The counts of each level for each factor.
    pub level_counts: Vec<HashMap<u32, usize>>,
    /// Expected count for each level if balanced.
    pub expected_count: usize,
}

impl OA {
    /// Perform a balance check on all factors.
    ///
    /// Checks if each level appears exactly N/s times in each column.
    #[must_use]
    pub fn balance_report(&self) -> BalanceReport {
        let n = self.runs();
        let k = self.factors();

        let mut factor_balance = Vec::with_capacity(k);
        let mut level_counts = Vec::with_capacity(k);

        for col in 0..k {
            let s = self.levels_for(col);
            let expected = n / s as usize;

            let mut counts = HashMap::new();
            for row in 0..n {
                *counts.entry(self.get(row, col)).or_insert(0) += 1;
            }

            let mut balanced = counts.len() == s as usize;
            if balanced {
                for &count in counts.values() {
                    if count != expected {
                        balanced = false;
                        break;
                    }
                }
            }

            factor_balance.push(balanced);
            level_counts.push(counts);
        }

        let expected_count = if self.params().is_symmetric() {
            n / self.levels() as usize
        } else {
            0
        };

        BalanceReport {
            factor_balance,
            level_counts,
            expected_count,
        }
    }

    /// Compute the correlation matrix between factors.
    ///
    /// For an orthogonal array of strength ≥ 2, the off-diagonal correlations
    /// should be zero if the levels are treated as numeric values.
    #[must_use]
    pub fn correlation_matrix(&self) -> Array2<f64> {
        let k = self.factors();
        let n = self.runs() as f64;
        let mut corr = Array2::zeros((k, k));

        // Convert data to floats and center it
        let mut centered_data = Array2::zeros((self.runs(), k));
        for col in 0..k {
            let column = self.column(col);
            let sum: u32 = column.iter().sum();
            let mean = f64::from(sum) / n;
            for row in 0..self.runs() {
                centered_data[[row, col]] = f64::from(column[row]) - mean;
            }
        }

        // Compute correlations
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    corr[[i, j]] = 1.0;
                    continue;
                }

                let mut dot = 0.0;
                let mut norm_i = 0.0;
                let mut norm_j = 0.0;

                for r in 0..self.runs() {
                    let vi = centered_data[[r, i]];
                    let vj = centered_data[[r, j]];
                    dot += vi * vj;
                    norm_i += vi * vi;
                    norm_j += vj * vj;
                }

                if norm_i > 0.0 && norm_j > 0.0 {
                    corr[[i, j]] = dot / (norm_i.sqrt() * norm_j.sqrt());
                } else {
                    corr[[i, j]] = 0.0;
                }
            }
        }

        corr
    }

    /// Compute the Generalized Word Length Pattern (GWLP) for resolution analysis.
    ///
    /// This is a SOTA metric for evaluating the quality of non-regular designs.
    /// For strength t, B_1 = B_2 = ... = B_t = 0.
    /// Note: This implementation currently only supports symmetric OAs for GWLP.
    #[must_use]
    pub fn gwlp(&self) -> Vec<f64> {
        assert!(
            self.params().is_symmetric(),
            "GWLP currently only supports symmetric OAs"
        );

        let n = self.runs();
        let k = self.factors();
        let s = self.levels() as f64;

        let mut b = vec![0.0; k + 1];
        b[0] = 1.0;

        // Distance enumerator using MacWilliams-like identity for OAs
        // For each pair of rows, compute the Hamming distance
        let mut distances = vec![0usize; k + 1];
        for i in 0..n {
            for j in 0..n {
                let mut dist = 0;
                for col in 0..k {
                    if self.get(i, col) != self.get(j, col) {
                        dist += 1;
                    }
                }
                distances[dist] += 1;
            }
        }

        // B_w = (1/N) * sum_{i,j} P_w(dist(row_i, row_j), k, s)
        // where P_w is the Krawtchouk polynomial
        for w in 1..=k {
            let mut sum = 0.0;
            for d in 0..=k {
                if distances[d] > 0 {
                    sum += (distances[d] as f64) * krawtchouk(w, d, k, s);
                }
            }
            b[w] = sum / (n as f64);
        }

        b
    }
}

/// Compute the Krawtchouk polynomial P_w(d; k, s).
fn krawtchouk(w: usize, d: usize, k: usize, s: f64) -> f64 {
    let mut sum = 0.0;
    for j in 0..=w {
        let term = (-1.0f64).powi(j as i32)
            * (s - 1.0).powi((w - j) as i32)
            * binomial_f64(d, j)
            * binomial_f64(k - d, w - j);
        sum += term;
    }
    sum
}

fn binomial_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let mut res = 1.0;
    let k = k.min(n - k);
    for i in 0..k {
        res = res * (n - i) as f64 / (i + 1) as f64;
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::OABuilder;

    #[test]
    fn test_balance_report() {
        let oa = OABuilder::new().levels(3).factors(4).build().unwrap();
        let report = oa.balance_report();

        for &balanced in &report.factor_balance {
            assert!(balanced);
        }
        assert_eq!(report.expected_count, 3); // 9 / 3
    }

    #[test]
    fn test_correlation_matrix() {
        let oa = OABuilder::new().levels(2).factors(3).build().unwrap();
        let corr = oa.correlation_matrix();

        // Strength 2 OA should have zero off-diagonal correlation
        for i in 0..oa.factors() {
            for j in 0..oa.factors() {
                if i != j {
                    assert!(
                        corr[[i, j]].abs() < 1e-10,
                        "Corr between {} and {} is {}",
                        i,
                        j,
                        corr[[i, j]]
                    );
                } else {
                    assert!((corr[[i, j]] - 1.0).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_gwlp_l9() {
        let oa = OABuilder::new().levels(3).factors(4).build().unwrap();
        let b = oa.gwlp();

        // For strength 2, B1 = B2 = 0
        assert!(b[1].abs() < 1e-10);
        assert!(b[2].abs() < 1e-10);
        // B3, B4 can be non-zero
    }
}

//! Rao-Hamming construction for orthogonal arrays.
//!
//! The Rao-Hamming construction produces OA(q^m, k, q, 2) where:
//! - q is a prime power (number of levels)
//! - m is an integer ≥ 2 (determines runs N = q^m)
//! - k ≤ (q^m - 1) / (q - 1) (number of factors)
//! - strength is always 2
//!
//! This construction is based on linear error-correcting codes. It provides the
//! maximum possible number of factors for a given number of runs and levels for strength 2.
//!
//! ## Algorithm
//!
//! 1. Generate all non-zero vectors of length m over GF(q).
//! 2. Select a subset of vectors such that no two vectors are linearly dependent.
//!    This is equivalent to picking vectors where the first non-zero entry is 1.
//! 3. There are (q^m - 1) / (q - 1) such vectors. These form the columns of a matrix G.
//! 4. The rows of the OA are all possible linear combinations of the rows of G.
//!
//! ## Example
//!
//! ```
//! use taguchi::construct::{Constructor, RaoHamming};
//!
//! // Create OA(25, 6, 5, 2) - same as Bose(5)
//! let rh = RaoHamming::new(5, 2).unwrap();
//! let oa = rh.construct(6).unwrap();
//!
//! assert_eq!(oa.runs(), 25);
//! assert_eq!(oa.factors(), 6);
//! assert_eq!(oa.levels(), 5);
//! assert_eq!(oa.strength(), 2);
//! ```

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OAParams, OA};
use crate::utils::is_prime_power;

/// Rao-Hamming construction for strength-2 orthogonal arrays.
///
/// Produces OA(q^m, k, q, 2) where q is a prime power and k ≤ (q^m-1)/(q-1).
#[derive(Debug, Clone)]
pub struct RaoHamming {
    /// The number of levels (must be a prime power).
    q: u32,
    /// The exponent m where runs N = q^m.
    m: u32,
    /// The Galois field for arithmetic.
    field: DynamicGf,
}

impl RaoHamming {
    /// Create a new Rao-Hamming constructor.
    ///
    /// # Arguments
    ///
    /// * `q` - Number of levels (must be a prime power)
    /// * `m` - Exponent (must be ≥ 2)
    ///
    /// # Errors
    ///
    /// Returns an error if `q` is not a prime power or `m < 2`.
    pub fn new(q: u32, m: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "RaoHamming",
            });
        }

        if m < 2 {
            return Err(Error::invalid_params("RaoHamming requires m >= 2"));
        }

        let field = DynamicGf::new(q)?;

        Ok(Self { q, m, field })
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.q
    }

    /// Get the number of runs.
    #[must_use]
    pub fn runs(&self) -> usize {
        self.q.pow(self.m) as usize
    }

    /// Get the maximum number of factors.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        let q = self.q as usize;
        let m = self.m as u32;
        (q.pow(m) - 1) / (q - 1)
    }

    /// Generate the generator columns for the construction.
    ///
    /// These are the k vectors where the first non-zero entry is 1.
    fn generate_columns(&self, factors: usize) -> Vec<Vec<u32>> {
        let q = self.q;
        let m = self.m as usize;
        let mut columns = Vec::with_capacity(factors);

        // We iterate through all q^m vectors and pick those whose first non-zero is 1
        for i in 1..q.pow(self.m) {
            let mut vec = Vec::with_capacity(m);
            let mut temp = i;
            for _ in 0..m {
                vec.push(temp % q);
                temp /= q;
            }
            // Reverse to have standard ordering (optional but nice)
            vec.reverse();

            // Check first non-zero entry
            for &val in &vec {
                if val != 0 {
                    if val == 1 {
                        columns.push(vec);
                    }
                    break;
                }
            }

            if columns.len() == factors {
                break;
            }
        }

        columns
    }
}

impl Constructor for RaoHamming {
    fn name(&self) -> &'static str {
        "RaoHamming"
    }

    fn family(&self) -> &'static str {
        "OA(q^m, k, q, 2), k ≤ (q^m-1)/(q-1)"
    }

    fn levels(&self) -> u32 {
        self.q
    }

    fn strength(&self) -> u32 {
        2
    }

    fn runs(&self) -> usize {
        self.q.pow(self.m) as usize
    }

    fn max_factors(&self) -> usize {
        self.max_factors()
    }

    fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "RaoHamming",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let m = self.m as usize;
        let runs = self.runs();
        let mut data = Array2::zeros((runs, factors));

        // Generate the generator matrix columns
        let cols = self.generate_columns(factors);

        // Each row of the OA is a linear combination of the generator matrix rows.
        // Or equivalently, row r of OA has entry (r_vec · col_j) at column j,
        // where r_vec is the q-ary representation of r.
        for row_idx in 0..runs {
            // q-ary representation of row_idx
            let mut r_vec = Vec::with_capacity(m);
            let mut temp = row_idx as u32;
            for _ in 0..m {
                r_vec.push(temp % q);
                temp /= q;
            }
            // Note: temp /= q fills from least significant, which is fine

            for (col_idx, col_vec) in cols.iter().enumerate() {
                // Dot product in GF(q)
                let mut sum = self.field.zero();
                for i in 0..m {
                    let a = self.field.element(r_vec[i]);
                    let b = self.field.element(col_vec[m - 1 - i]); // Match significance
                    sum = sum.add(a.mul(b));
                }
                data[[row_idx, col_idx]] = sum.to_u32();
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_rao_hamming_creation() {
        let rh = RaoHamming::new(3, 2).unwrap();
        assert_eq!(rh.levels(), 3);
        assert_eq!(rh.runs(), 9);
        assert_eq!(rh.max_factors(), 4); // (3^2-1)/(3-1) = 8/2 = 4
    }

    #[test]
    fn test_rao_hamming_l9() {
        let rh = RaoHamming::new(3, 2).unwrap();
        let oa = rh.construct(4).unwrap();

        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 4);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "RaoHamming L9 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_rao_hamming_l27() {
        let rh = RaoHamming::new(3, 3).unwrap();
        let oa = rh.construct(13).unwrap(); // (27-1)/2 = 13

        assert_eq!(oa.runs(), 27);
        assert_eq!(oa.factors(), 13);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "RaoHamming L27 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_rao_hamming_l8() {
        let rh = RaoHamming::new(2, 3).unwrap();
        let oa = rh.construct(7).unwrap(); // (8-1)/1 = 7

        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 7);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "RaoHamming L8 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_rao_hamming_too_many_factors() {
        let rh = RaoHamming::new(3, 2).unwrap();
        assert!(rh.construct(5).is_err());
    }
}

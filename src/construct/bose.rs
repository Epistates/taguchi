//! Bose construction for orthogonal arrays.
//!
//! The Bose construction produces OA(q², k, q, 2) where:
//! - q is a prime power (the number of levels)
//! - k ≤ q + 1 (the number of factors)
//! - strength is always 2
//!
//! This is the simplest and most commonly used construction for strength-2 arrays.
//!
//! ## Algorithm
//!
//! For each row (i, j) where i, j ∈ GF(q):
//! - Column 0: j
//! - Column c (for c = 1, 2, ..., q): i + c*j (computed in GF(q))
//!
//! This produces q² rows with up to q+1 columns.
//!
//! ## Example
//!
//! ```
//! use taguchi::construct::{Constructor, Bose};
//!
//! // Create L9 array (9 runs, 4 factors, 3 levels)
//! let bose = Bose::new(3);
//! let oa = bose.construct(4).unwrap();
//!
//! assert_eq!(oa.runs(), 9);
//! assert_eq!(oa.factors(), 4);
//! assert_eq!(oa.levels(), 3);
//! assert_eq!(oa.strength(), 2);
//! ```

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OA, OAParams};
use crate::utils::is_prime_power;

/// Bose construction for strength-2 orthogonal arrays.
///
/// Produces OA(q², k, q, 2) where q is a prime power and k ≤ q+1.
#[derive(Debug, Clone)]
pub struct Bose {
    /// The number of levels (must be a prime power).
    q: u32,
    /// The Galois field for arithmetic.
    field: DynamicGf,
}

impl Bose {
    /// Create a new Bose constructor for the given number of levels.
    ///
    /// # Panics
    ///
    /// Panics if `q` is not a prime power.
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::Bose;
    ///
    /// let bose3 = Bose::new(3);  // For 3-level arrays
    /// let bose7 = Bose::new(7);  // For 7-level arrays
    /// let bose9 = Bose::new(9);  // For 9-level arrays (GF(3²))
    /// ```
    #[must_use]
    pub fn new(q: u32) -> Self {
        Self::try_new(q).expect("q must be a prime power")
    }

    /// Create a new Bose constructor, returning an error if q is invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if `q` is not a prime power.
    pub fn try_new(q: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "Bose",
            });
        }

        let field = DynamicGf::new(q)?;

        Ok(Self { q, field })
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.q
    }

    /// Get the number of runs this constructor produces.
    #[must_use]
    pub fn runs(&self) -> usize {
        (self.q * self.q) as usize
    }

    /// Get the maximum number of factors this constructor can produce.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        (self.q + 1) as usize
    }

    /// Construct the orthogonal array.
    ///
    /// # Errors
    ///
    /// Returns an error if `factors` exceeds `q + 1`.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "Bose",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let runs = self.runs();
        let mut data = Array2::zeros((runs, factors));

        // Fill the array using Bose construction
        // Row (i, j) where i, j ∈ {0, 1, ..., q-1}
        // Column 0: j
        // Column c (c ≥ 1): i + c*j (in GF(q))
        for i in 0..q {
            for j in 0..q {
                let row = (i * q + j) as usize;

                // Column 0: j
                data[[row, 0]] = j;

                // Columns 1 to factors-1: i + c*j
                // We want to compute: val[c] = j*c + i for c in 1..factors
                // This is a linear transform y = a*x + b where:
                // a = j
                // b = i
                // x = c (the column index)
                
                // Prepare points (column indices)
                // In a real optimized scenario, 'points' would be constant and reused
                // But for now, we just construct it.
                // Actually, we can just iterate since we have direct access now.
                
                let tables = self.field.tables();
                for c in 1..factors {
                    // val = i + c*j
                    // We must map column index c to field element.
                    // c ranges from 1 to q (inclusive for max factors).
                    // In GF(q), we take c % q.
                    // e.g. for q=3: c=1->1, c=2->2, c=3->0
                    let c_val = (c as u32) % self.q;
                    let term = tables.mul(c_val, j);
                    data[[row, c]] = tables.add(i, term);
                }
            }
        }

        // Strength is min(2, factors) since we can't claim higher strength than columns
        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for Bose {
    fn name(&self) -> &'static str {
        "Bose"
    }

    fn family(&self) -> &'static str {
        "OA(q², k, q, 2), k ≤ q+1"
    }

    fn levels(&self) -> u32 {
        self.q
    }

    fn strength(&self) -> u32 {
        2
    }

    fn runs(&self) -> usize {
        (self.q * self.q) as usize
    }

    fn max_factors(&self) -> usize {
        (self.q + 1) as usize
    }

    fn construct(&self, factors: usize) -> Result<OA> {
        Bose::construct(self, factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_bose_creation() {
        let bose = Bose::new(3);
        assert_eq!(bose.levels(), 3);
        assert_eq!(bose.runs(), 9);
        assert_eq!(bose.max_factors(), 4);
    }

    #[test]
    fn test_bose_invalid() {
        assert!(Bose::try_new(6).is_err());
        assert!(Bose::try_new(10).is_err());
    }

    #[test]
    fn test_bose_construct_l9() {
        let bose = Bose::new(3);
        let oa = bose.construct(4).unwrap();

        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels(), 3);
        assert_eq!(oa.strength(), 2);

        // Verify it's a valid OA
        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L9 should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_bose_construct_l4() {
        let bose = Bose::new(2);
        let oa = bose.construct(3).unwrap();

        assert_eq!(oa.runs(), 4);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.levels(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L4 should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_bose_construct_l25() {
        let bose = Bose::new(5);
        let oa = bose.construct(6).unwrap();

        assert_eq!(oa.runs(), 25);
        assert_eq!(oa.factors(), 6);
        assert_eq!(oa.levels(), 5);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L25 should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_bose_construct_l49() {
        let bose = Bose::new(7);
        let oa = bose.construct(8).unwrap();

        assert_eq!(oa.runs(), 49);
        assert_eq!(oa.factors(), 8);
        assert_eq!(oa.levels(), 7);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L49 should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_bose_prime_power() {
        // Test with GF(4) = GF(2²)
        let bose = Bose::new(4);
        let oa = bose.construct(5).unwrap();

        assert_eq!(oa.runs(), 16);
        assert_eq!(oa.factors(), 5);
        assert_eq!(oa.levels(), 4);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L16(4-level) should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_bose_too_many_factors() {
        let bose = Bose::new(3);
        assert!(bose.construct(5).is_err());  // max is 4
    }

    #[test]
    fn test_bose_zero_factors() {
        let bose = Bose::new(3);
        assert!(bose.construct(0).is_err());
    }

    #[test]
    fn test_bose_single_factor() {
        let bose = Bose::new(3);
        let oa = bose.construct(1).unwrap();

        assert_eq!(oa.factors(), 1);
        // With 1 factor, every row just has the j value
        // Should contain 0,0,0,1,1,1,2,2,2 (each value 3 times)
        let mut counts = [0usize; 3];
        for row in 0..9 {
            counts[oa.get(row, 0) as usize] += 1;
        }
        assert_eq!(counts, [3, 3, 3]);
    }
}

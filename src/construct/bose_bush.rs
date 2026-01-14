//! Bose-Bush construction for orthogonal arrays.
//!
//! The Bose-Bush construction produces OA(2q², k, q, 2) where:
//! - q is a power of 2 (currently only q = 2 is supported)
//! - k ≤ 2q + 1 (the number of factors)
//! - strength is always 2
//!
//! This construction doubles the number of columns compared to Bose for the same
//! number of runs, at the cost of requiring q to be a power of 2.
//!
//! ## Algorithm
//!
//! The construction uses the structure of GF(2q) and GF(q) together.
//! It creates 2q² rows by combining elements from GF(q) in a specific pattern.
//!
//! For elements i, j ∈ GF(q):
//! - First q² rows (block 0): standard Bose-like pattern
//! - Second q² rows (block 1): modified pattern using field structure
//!
//! ## Example
//!
//! ```
//! use taguchi::construct::{Constructor, BoseBush};
//!
//! // Create OA(8, 5, 2, 2)
//! let bb = BoseBush::new(2).unwrap();
//! let oa = bb.construct(5).unwrap();
//!
//! assert_eq!(oa.runs(), 8);     // 2 * 2²
//! assert_eq!(oa.factors(), 5);  // up to 2*2 + 1 = 5
//! assert_eq!(oa.levels(), 2);
//! assert_eq!(oa.strength(), 2);
//! ```

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OAParams, OA};
use crate::utils::factor_prime_power;

/// Bose-Bush construction for strength-2 orthogonal arrays.
///
/// Produces OA(2q², k, q, 2) where q is a power of 2 and k ≤ 2q+1.
/// This provides more columns than Bose for the same number of levels.
#[derive(Debug, Clone)]
pub struct BoseBush {
    /// The number of levels (must be a power of 2).
    q: u32,
    /// The exponent m where q = 2^m.
    _m: u32,
    /// The Galois field GF(q) for arithmetic.
    field_q: DynamicGf,
    /// The Galois field GF(2q) for extended arithmetic.
    _field_2q: DynamicGf,
}

impl BoseBush {
    /// Create a new Bose-Bush constructor for the given number of levels.
    ///
    /// # Arguments
    ///
    /// * `q` - Number of levels (currently only q=2 is supported)
    ///
    /// # Errors
    ///
    /// Returns an error if `q` is not 2. Support for larger powers of 2
    /// (4, 8, 16, ...) is planned for a future release.
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::BoseBush;
    ///
    /// let bb2 = BoseBush::new(2).unwrap();   // OA(8, 5, 2, 2)
    ///
    /// // Larger powers of 2 not yet supported
    /// assert!(BoseBush::new(4).is_err());
    /// assert!(BoseBush::new(3).is_err());    // 3 is not a power of 2
    /// ```
    pub fn new(q: u32) -> Result<Self> {
        // Check that q is a power of 2
        let factorization = factor_prime_power(q).ok_or(Error::RequiresPowerOfTwo {
            levels: q,
            algorithm: "BoseBush",
        })?;

        if factorization.prime != 2 {
            return Err(Error::RequiresPowerOfTwo {
                levels: q,
                algorithm: "BoseBush",
            });
        }

        let m = factorization.exponent;

        // Currently only q=2 is supported. The general Bose-Bush construction
        // for q=2^m (m>1) requires sophisticated GF(2q) to GF(q) mappings
        // that are not yet implemented.
        if q > 2 {
            return Err(Error::invalid_params(
                "BoseBush currently only supports q=2. Support for q=4,8,16,... is planned.",
            ));
        }

        // We need GF(q) and GF(2q)
        let field_q = DynamicGf::new(q)?;
        let _field_2q = DynamicGf::new(2 * q)?;

        Ok(Self {
            q,
            _m: m,
            field_q,
            _field_2q,
        })
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.q
    }

    /// Get the number of runs this constructor produces.
    #[must_use]
    pub fn runs(&self) -> usize {
        (2 * self.q * self.q) as usize
    }

    /// Get the maximum number of factors this constructor can produce.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        (2 * self.q + 1) as usize
    }

    /// Construct the orthogonal array.
    ///
    /// # Errors
    ///
    /// Returns an error if `factors` exceeds `2q + 1`.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "BoseBush",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let runs = self.runs();
        let mut data = Array2::zeros((runs, factors));

        // Bose-Bush construction OA(2q², 2q+1, q, 2) for q = 2^m
        //
        // Rows are indexed by (b, i, j) where b ∈ {0,1} and i, j ∈ GF(q).
        // Row number = b*q² + i*q + j
        //
        // The construction extends Bose's OA(q², q+1, q, 2) to double the rows
        // and nearly double the columns by using two "blocks" that differ in
        // the extended columns.
        //
        // Column layout:
        // - Column 0: j
        // - Columns 1 to q: i + k*j for k = 1, ..., q (Bose-like)
        // - Columns q+1 to 2q: Extended columns that differ between blocks
        //
        // For extended columns, we need the pair (column c, column q+c) to be
        // orthogonal. This is achieved by:
        // - Block 0: Use a formula based on (c*j, c*i)
        // - Block 1: Use a formula based on (c*i, c*j) with block offset

        for b in 0..2u32 {
            for i in 0..q {
                for j in 0..q {
                    let row = (b * q * q + i * q + j) as usize;

                    let elem_i = self.field_q.element(i);
                    let elem_j = self.field_q.element(j);

                    for c in 0..factors {
                        let col = c as u32;

                        if col == 0 {
                            // Column 0: j (same for both blocks)
                            data[[row, c]] = j;
                        } else if col <= q {
                            // Columns 1 to q: i + k*j where k = col
                            // Same formula for both blocks
                            let k = col % q; // k = col, but col = q means k = 0
                            let k_elem = self.field_q.element(k);
                            let val = elem_i.add(k_elem.mul(elem_j.clone())).to_u32();
                            data[[row, c]] = val;
                        } else {
                            // Extended columns q+1 to 2q
                            // k = col - q (ranges from 1 to q)
                            let k = col - q;

                            // The Bose-Bush construction uses the key formula:
                            // Column q+k = (Column k) + b
                            //
                            // Where b is the block indicator (0 or 1).
                            //
                            // This ensures orthogonality between columns k and q+k:
                            // - Block 0: both columns have same value (diagonal pairs)
                            // - Block 1: column q+k = column k + 1 (off-diagonal pairs)
                            // Together: all pairs equally represented
                            //
                            // Note: Column k uses formula i + k*j, so
                            // Column q+k = (i + k*j) + b

                            // Compute base value: i + k*j (same as column k)
                            let k_elem = self.field_q.element(k % q);
                            let base_val = elem_i.add(k_elem.mul(elem_j.clone()));

                            // Add b (the block indicator) to get column q+k
                            let b_elem = self.field_q.element(b);
                            let val = base_val.add(b_elem).to_u32();
                            data[[row, c]] = val;
                        }
                    }
                }
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for BoseBush {
    fn name(&self) -> &'static str {
        "BoseBush"
    }

    fn family(&self) -> &'static str {
        "OA(2q², k, q, 2), q=2^m, k ≤ 2q+1"
    }

    fn levels(&self) -> u32 {
        self.q
    }

    fn strength(&self) -> u32 {
        2
    }

    fn runs(&self) -> usize {
        (2 * self.q * self.q) as usize
    }

    fn max_factors(&self) -> usize {
        (2 * self.q + 1) as usize
    }

    fn construct(&self, factors: usize) -> Result<OA> {
        BoseBush::construct(self, factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_bose_bush_creation() {
        let bb = BoseBush::new(2).unwrap();
        assert_eq!(bb.levels(), 2);
        assert_eq!(bb.runs(), 8);
        assert_eq!(bb.max_factors(), 5);
    }

    #[test]
    fn test_bose_bush_invalid() {
        // Not a power of 2
        assert!(BoseBush::new(3).is_err());
        assert!(BoseBush::new(5).is_err());
        assert!(BoseBush::new(6).is_err());
        assert!(BoseBush::new(7).is_err());
        assert!(BoseBush::new(9).is_err());
    }

    #[test]
    fn test_bose_bush_unsupported_q() {
        // Powers of 2 > 2 are not yet supported
        assert!(BoseBush::new(4).is_err());
        assert!(BoseBush::new(8).is_err());
        assert!(BoseBush::new(16).is_err());
    }

    #[test]
    fn test_bose_bush_q2_valid() {
        // Only q=2 is currently supported
        assert!(BoseBush::new(2).is_ok());
    }

    #[test]
    fn test_bose_bush_q2() {
        // OA(8, 5, 2, 2)
        let bb = BoseBush::new(2).unwrap();
        let oa = bb.construct(5).unwrap();

        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 5);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "BoseBush(2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bose_bush_fewer_factors() {
        let bb = BoseBush::new(2).unwrap();
        let oa = bb.construct(3).unwrap();

        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.levels(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "BoseBush(2) with 3 factors should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bose_bush_too_many_factors() {
        let bb = BoseBush::new(2).unwrap();
        assert!(bb.construct(6).is_err()); // max is 5
    }

    #[test]
    fn test_bose_bush_zero_factors() {
        let bb = BoseBush::new(2).unwrap();
        assert!(bb.construct(0).is_err());
    }

    #[test]
    fn test_bose_bush_single_factor() {
        let bb = BoseBush::new(2).unwrap();
        let oa = bb.construct(1).unwrap();

        assert_eq!(oa.factors(), 1);
        assert_eq!(oa.runs(), 8);

        // Each level should appear 4 times (8/2)
        let mut counts = [0usize; 2];
        for row in 0..8 {
            counts[oa.get(row, 0) as usize] += 1;
        }
        assert_eq!(counts, [4, 4]);
    }
}

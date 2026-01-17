//! Bush construction for orthogonal arrays.
//!
//! The Bush construction produces OA(q^t, k, q, t) where:
//! - q is a prime power (the number of levels)
//! - t is the strength (2 ≤ t ≤ q)
//! - k ≤ q + 1 (the number of factors)
//!
//! This construction generalizes the Bose construction to higher strengths.
//!
//! ## Algorithm
//!
//! Each row corresponds to a polynomial of degree < t over GF(q).
//! There are q^t such polynomials.
//!
//! For a polynomial p(x) = a_0 + a_1*x + ... + a_{t-1}*x^{t-1}:
//! - Row index encodes coefficients (a_0, a_1, ..., a_{t-1})
//! - Column c contains p(c) evaluated in GF(q) for c = 0, 1, ..., q
//!
//! This produces q^t rows with up to q+1 columns.
//!
//! ## Example
//!
//! ```
//! use taguchi::construct::{Constructor, Bush};
//!
//! // Create OA(27, 4, 3, 3) - strength 3 array
//! let bush = Bush::new(3, 3).unwrap();
//! let oa = bush.construct(4).unwrap();
//!
//! assert_eq!(oa.runs(), 27);    // 3^3
//! assert_eq!(oa.factors(), 4);
//! assert_eq!(oa.levels(), 3);
//! assert_eq!(oa.strength(), 3);
//! ```
//!
//! ## Relation to Bose
//!
//! When t = 2, the Bush construction is equivalent to the Bose construction.

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OA, OAParams};
use crate::utils::is_prime_power;

/// Bush construction for strength-t orthogonal arrays.
///
/// Produces OA(q^t, k, q, t) where q is a prime power, t is the strength,
/// and k ≤ q+1.
#[derive(Debug, Clone)]
pub struct Bush {
    /// The number of levels (must be a prime power).
    q: u32,
    /// The strength of the array.
    strength: u32,
    /// The Galois field for arithmetic.
    field: DynamicGf,
}

impl Bush {
    /// Create a new Bush constructor for the given levels and strength.
    ///
    /// # Arguments
    ///
    /// * `q` - Number of levels (must be a prime power)
    /// * `strength` - Desired strength (must be 2 ≤ t ≤ q)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `q` is not a prime power
    /// - `strength` is less than 2 or greater than q
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::Bush;
    ///
    /// let bush3_2 = Bush::new(3, 2).unwrap();  // Same as Bose(3)
    /// let bush3_3 = Bush::new(3, 3).unwrap();  // Strength 3
    /// let bush5_3 = Bush::new(5, 3).unwrap();  // 5 levels, strength 3
    /// ```
    pub fn new(q: u32, strength: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "Bush",
            });
        }

        if strength < 2 {
            return Err(Error::InvalidStrength {
                strength,
                min: 2,
                max: q,
                algorithm: "Bush",
            });
        }

        if strength > q {
            return Err(Error::InvalidStrength {
                strength,
                min: 2,
                max: q,
                algorithm: "Bush",
            });
        }

        let field = DynamicGf::new(q)?;

        Ok(Self { q, strength, field })
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.q
    }

    /// Get the strength.
    #[must_use]
    pub fn strength(&self) -> u32 {
        self.strength
    }

    /// Get the number of runs this constructor produces.
    #[must_use]
    pub fn runs(&self) -> usize {
        self.q.pow(self.strength) as usize
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
                algorithm: "Bush",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let t = self.strength;
        let runs = self.runs();
        let mut data = Array2::zeros((runs, factors));

        // Each row corresponds to a polynomial p(x) = a_0 + a_1*x + ... + a_{t-1}*x^{t-1}
        // Row index r encodes the coefficients:
        //   a_0 = r % q
        //   a_1 = (r / q) % q
        //   a_2 = (r / q^2) % q
        //   etc.
        //
        // For columns 0 to q-1: column c contains p(c) = a_0 + a_1*c + ... + a_{t-1}*c^{t-1}
        // For column q (if included): contains the leading coefficient a_{t-1}
        // This ensures the array has the full strength t property.

        // Pre-allocate buffer for results
        let mut results = vec![0u32; factors];
        // Column indices as points to evaluate
        let points: Vec<u32> = (0..factors as u32).collect();

        for row in 0..runs {
            // Extract polynomial coefficients from row index
            // ... (keep existing coeff extraction logic or optimize it too)
            // Actually, let's just optimize the evaluation part first

            let mut coeffs = Vec::with_capacity(t as usize);
            let mut temp_row = row as u32;
            for _ in 0..t {
                coeffs.push(temp_row % q);
                temp_row /= q;
            }

            // Columns 0 to q-1: evaluate polynomial at x = col
            // For Bush, typically factors <= q+1.
            // If factors <= q, we just eval at 0..factors

            // We can use bulk_eval_poly for the first q columns (or all if factors <= q)
            let eval_count = factors.min(q as usize);

            self.field
                .bulk_eval_poly(&coeffs, &points[0..eval_count], &mut results[0..eval_count]);

            // Copy results to data
            for col in 0..eval_count {
                data[[row, col]] = results[col];
            }

            // Handle the point at infinity if factors > q
            // Column q: the leading coefficient a_{t-1}
            if factors > q as usize {
                data[[row, q as usize]] = coeffs[t as usize - 1];
            }
        }

        // Actual strength is min(t, factors) since we can't claim more strength than columns
        let actual_strength = t.min(factors as u32);
        let params = OAParams::new(runs, factors, q, actual_strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for Bush {
    fn name(&self) -> &'static str {
        "Bush"
    }

    fn family(&self) -> &'static str {
        "OA(q^t, k, q, t), k ≤ q+1"
    }

    fn levels(&self) -> u32 {
        self.q
    }

    fn strength(&self) -> u32 {
        self.strength
    }

    fn runs(&self) -> usize {
        self.q.pow(self.strength) as usize
    }

    fn max_factors(&self) -> usize {
        (self.q + 1) as usize
    }

    fn construct(&self, factors: usize) -> Result<OA> {
        Bush::construct(self, factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_bush_creation() {
        let bush = Bush::new(3, 3).unwrap();
        assert_eq!(bush.levels(), 3);
        assert_eq!(bush.strength(), 3);
        assert_eq!(bush.runs(), 27);
        assert_eq!(bush.max_factors(), 4);
    }

    #[test]
    fn test_bush_invalid_levels() {
        assert!(Bush::new(6, 2).is_err()); // 6 not prime power
        assert!(Bush::new(10, 2).is_err()); // 10 not prime power
    }

    #[test]
    fn test_bush_invalid_strength() {
        assert!(Bush::new(3, 1).is_err()); // strength < 2
        assert!(Bush::new(3, 4).is_err()); // strength > q
        assert!(Bush::new(5, 6).is_err()); // strength > q
    }

    #[test]
    fn test_bush_strength_2_equals_bose() {
        // Bush with t=2 should produce the same structure as Bose
        let bush = Bush::new(3, 2).unwrap();
        let oa = bush.construct(4).unwrap();

        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels(), 3);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Bush t=2 should be valid strength-2: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bush_strength_3() {
        let bush = Bush::new(3, 3).unwrap();
        let oa = bush.construct(4).unwrap();

        assert_eq!(oa.runs(), 27); // 3^3
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels(), 3);
        assert_eq!(oa.strength(), 3);

        // Verify strength 3 (should also pass strength 2)
        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Bush t=3 should be valid strength-2: {:?}",
            result.issues
        );

        let result = verify_strength(&oa, 3).unwrap();
        assert!(
            result.is_valid,
            "Bush t=3 should be valid strength-3: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bush_5_3() {
        let bush = Bush::new(5, 3).unwrap();
        let oa = bush.construct(6).unwrap();

        assert_eq!(oa.runs(), 125); // 5^3
        assert_eq!(oa.factors(), 6);
        assert_eq!(oa.levels(), 5);
        assert_eq!(oa.strength(), 3);

        let result = verify_strength(&oa, 3).unwrap();
        assert!(
            result.is_valid,
            "Bush(5,3) should be valid strength-3: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bush_7_2() {
        let bush = Bush::new(7, 2).unwrap();
        let oa = bush.construct(8).unwrap();

        assert_eq!(oa.runs(), 49);
        assert_eq!(oa.factors(), 8);
        assert_eq!(oa.levels(), 7);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Bush(7,2) should be valid strength-2: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bush_prime_power_4() {
        // Test with GF(4) = GF(2^2)
        let bush = Bush::new(4, 2).unwrap();
        let oa = bush.construct(5).unwrap();

        assert_eq!(oa.runs(), 16);
        assert_eq!(oa.factors(), 5);
        assert_eq!(oa.levels(), 4);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Bush(4,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bush_prime_power_8() {
        // Test with GF(8) = GF(2^3)
        let bush = Bush::new(8, 2).unwrap();
        let oa = bush.construct(9).unwrap();

        assert_eq!(oa.runs(), 64);
        assert_eq!(oa.factors(), 9);
        assert_eq!(oa.levels(), 8);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Bush(8,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_bush_too_many_factors() {
        let bush = Bush::new(3, 3).unwrap();
        assert!(bush.construct(5).is_err()); // max is 4
    }

    #[test]
    fn test_bush_zero_factors() {
        let bush = Bush::new(3, 3).unwrap();
        assert!(bush.construct(0).is_err());
    }

    #[test]
    fn test_bush_single_factor() {
        let bush = Bush::new(3, 3).unwrap();
        let oa = bush.construct(1).unwrap();

        assert_eq!(oa.factors(), 1);
        assert_eq!(oa.strength(), 1); // Can't have strength > factors

        // With 1 factor, should contain each value q^(t-1) times
        // 27 rows, 3 levels -> each level appears 9 times
        let mut counts = [0usize; 3];
        for row in 0..27 {
            counts[oa.get(row, 0) as usize] += 1;
        }
        assert_eq!(counts, [9, 9, 9]);
    }
}

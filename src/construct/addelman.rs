//! Addelman-Kempthorne construction for orthogonal arrays.
//!
//! The Addelman-Kempthorne construction produces OA(2q², k, q, 2) where:
//! - q is an **odd** prime power (3, 5, 7, 9, 11, 13, ...)
//! - k ≤ 2q + 1 (the number of factors)
//! - strength is always 2
//!
//! This construction complements Bose-Bush (which requires powers of 2) by handling
//! odd prime powers. Like Bose-Bush, it doubles both the rows and columns compared
//! to the basic Bose construction.
//!
//! ## Algorithm
//!
//! The construction divides the array into two blocks of q² rows each.
//!
//! **Block 1** (rows indexed by (i, j) ∈ GF(q) × GF(q)):
//! - Column 0: j
//! - Columns 1 to q-1: i + m·j for m = 1, ..., q-1
//! - Column q: i
//! - Columns q+1 to 2q: i² + m·i + j for m = 0, ..., q-1
//!
//! **Block 2** uses transformation constants (kay, b[], c[], k[]) derived from
//! field properties to ensure orthogonality between corresponding columns.
//!
//! ## Reference
//!
//! S. Addelman and O. Kempthorne (1961). "Some Main-Effect Plans and Orthogonal
//! Arrays of Strength Two." Annals of Mathematical Statistics, Vol 32, pp 1167-1176.
//!
//! ## Example
//!
//! ```
//! use taguchi::construct::{Constructor, AddelmanKempthorne};
//!
//! // Create OA(18, 7, 3, 2) - equivalent to L18
//! let ak = AddelmanKempthorne::new(3).unwrap();
//! let oa = ak.construct(7).unwrap();
//!
//! assert_eq!(oa.runs(), 18);    // 2 * 3²
//! assert_eq!(oa.factors(), 7);  // up to 2*3 + 1 = 7
//! assert_eq!(oa.levels(), 3);
//! assert_eq!(oa.strength(), 2);
//! ```

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OA, OAParams};
use crate::utils::factor_prime_power;

/// Addelman-Kempthorne construction for strength-2 orthogonal arrays.
///
/// Produces OA(2q², k, q, 2) where q is an odd prime power and k ≤ 2q+1.
/// This provides the same column efficiency as Bose-Bush but for odd prime powers.
#[derive(Debug, Clone)]
pub struct AddelmanKempthorne {
    /// The number of levels (must be an odd prime power).
    q: u32,
    /// The characteristic p of the field (the prime factor).
    /// Currently unused but kept for potential future extensions.
    _p: u32,
    /// The Galois field GF(q) for arithmetic.
    field: DynamicGf,
    /// kay: A quadratic non-residue (rootless element) in GF(q).
    kay: u32,
    /// Transformation constant b[m] for m = 0..q-1.
    b: Vec<u32>,
    /// Transformation constant c[m] for m = 0..q-1.
    c: Vec<u32>,
    /// Transformation constant k[m] for m = 0..q-1.
    k: Vec<u32>,
}

impl AddelmanKempthorne {
    /// Create a new Addelman-Kempthorne constructor for the given number of levels.
    ///
    /// # Arguments
    ///
    /// * `q` - Number of levels (must be an odd prime power: 3, 5, 7, 9, 11, 13, ...)
    ///
    /// # Errors
    ///
    /// Returns an error if `q` is not an odd prime power.
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::AddelmanKempthorne;
    ///
    /// let ak3 = AddelmanKempthorne::new(3).unwrap();   // OA(18, 7, 3, 2)
    /// let ak5 = AddelmanKempthorne::new(5).unwrap();   // OA(50, 11, 5, 2)
    /// let ak9 = AddelmanKempthorne::new(9).unwrap();   // OA(162, 19, 9, 2) - GF(3²)
    ///
    /// // Even prime powers should use Bose-Bush instead
    /// assert!(AddelmanKempthorne::new(2).is_err());
    /// assert!(AddelmanKempthorne::new(4).is_err());
    /// ```
    pub fn new(q: u32) -> Result<Self> {
        // Check that q is a prime power
        let factorization = factor_prime_power(q).ok_or(Error::LevelsNotPrimePower {
            levels: q,
            algorithm: "AddelmanKempthorne",
        })?;

        let p = factorization.prime;

        // Must be an odd prime power
        if p == 2 {
            return Err(Error::invalid_params(
                "AddelmanKempthorne requires an ODD prime power (3, 5, 7, 9, ...). \
                 Use BoseBush for powers of 2.",
            ));
        }

        // Create the Galois field
        let field = DynamicGf::new(q)?;

        // Compute the transformation constants
        let (kay, b, c, k) = Self::compute_constants(&field, q, p)?;

        Ok(Self {
            q,
            _p: p,
            field,
            kay,
            b,
            c,
            k,
        })
    }

    /// Compute the transformation constants kay, b[], c[], k[] for block 2.
    ///
    /// These constants ensure orthogonality between block 1 and block 2 columns.
    fn compute_constants(
        field: &DynamicGf,
        q: u32,
        p: u32,
    ) -> Result<(u32, Vec<u32>, Vec<u32>, Vec<u32>)> {
        // Find a quadratic non-residue (rootless element) in GF(q)
        // An element x is a quadratic residue if there exists y such that y² = x
        let kay = Self::find_non_residue(field, q)?;

        // Compute four = 4 (or 1 if p = 3 since 4 ≡ 1 mod 3)
        let four = if p == 3 { 1u32 } else { 4u32 };
        let four_elem = field.element(four);

        // kay_elem for field arithmetic
        let kay_elem = field.element(kay);

        // Compute (kay + (p-1)) where (p-1) ≡ -1 in GF(q) for the prime subfield
        // In GF(q), the additive inverse of 1 is (q-1) for extension fields
        // But we want -1 in the field, which is q-1 for any GF(q)
        let neg_one = field.element(q - 1);
        let kay_minus_one = kay_elem.add(neg_one.clone());

        // Compute four_inv = four^(-1) in GF(q)
        let four_inv = if four == 1 {
            field.element(1)
        } else {
            four_elem.inv()
        };

        let mut b = vec![0u32; q as usize];
        let mut c = vec![0u32; q as usize];
        let mut k_arr = vec![0u32; q as usize];

        // For m = 0, we use identity-like values
        b[0] = 0;
        c[0] = 0;
        k_arr[0] = 0;

        // For m = 1 to q-1:
        // b[m] = (kay - 1) / (kay * 4 * m)
        // k[m] = kay * m
        // c[m] = m² * (kay - 1) / 4
        for m in 1..q {
            let m_elem = field.element(m);

            // k[m] = kay * m
            let k_m = kay_elem.mul(m_elem.clone()).to_u32();
            k_arr[m as usize] = k_m;

            // b[m] = (kay - 1) * (kay * 4 * m)^(-1)
            // = (kay - 1) / (kay * 4 * m)
            let denom = kay_elem.mul(four_elem.clone()).mul(m_elem.clone());
            let denom_inv = denom.inv();
            let b_m = kay_minus_one.mul(denom_inv).to_u32();
            b[m as usize] = b_m;

            // c[m] = m² * (kay - 1) * 4^(-1)
            // = m² * (kay - 1) / 4
            let m_sq = m_elem.clone().mul(m_elem);
            let c_m = m_sq
                .mul(kay_minus_one.clone())
                .mul(four_inv.clone())
                .to_u32();
            c[m as usize] = c_m;
        }

        Ok((kay, b, c, k_arr))
    }

    /// Find a quadratic non-residue in GF(q).
    ///
    /// A quadratic non-residue is an element x such that there is no y with y² = x.
    fn find_non_residue(field: &DynamicGf, q: u32) -> Result<u32> {
        // Build a set of quadratic residues
        let mut is_residue = vec![false; q as usize];
        is_residue[0] = true; // 0 = 0² is considered a residue

        for y in 1..q {
            let y_elem = field.element(y);
            let y_sq = y_elem.clone().mul(y_elem).to_u32();
            is_residue[y_sq as usize] = true;
        }

        // Find the first non-residue (starting from 1)
        for x in 1..q {
            if !is_residue[x as usize] {
                return Ok(x);
            }
        }

        // This should never happen for odd prime powers > 1
        Err(Error::invalid_params(
            "Failed to find a quadratic non-residue in GF(q)",
        ))
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
                algorithm: "AddelmanKempthorne",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let runs = self.runs();
        let mut data = Array2::zeros((runs, factors));

        // Addelman-Kempthorne construction OA(2q², 2q+1, q, 2) for odd prime power q
        //
        // The array is divided into two blocks:
        //
        // Block 1 (rows 0 to q²-1): Direct polynomial formulas
        // Block 2 (rows q² to 2q²-1): Transformed formulas using kay, b, c, k
        //
        // Column layout:
        // - Column 0: j (y term)
        // - Columns 1 to q-1: i + m*j for m = 1, ..., q-1 (x + m*y)
        // - Column q: i (x term)
        // - Columns q+1 to 2q: i² + m*i + j for m = 0, ..., q-1 (x² + m*x + y)

        let kay_elem = self.field.element(self.kay);

        // Block 1: Direct formulas
        for i in 0..q {
            for j in 0..q {
                let row = (i * q + j) as usize;

                let i_elem = self.field.element(i);
                let j_elem = self.field.element(j);

                // i²
                let i_sq = i_elem.mul(i_elem.clone());

                for col in 0..factors {
                    let col_u32 = col as u32;

                    let value = if col_u32 == 0 {
                        // Column 0: j
                        j
                    } else if col_u32 <= q - 1 {
                        // Columns 1 to q-1: i + m*j where m = col
                        let m = col_u32;
                        let m_elem = self.field.element(m);
                        i_elem.add(m_elem.mul(j_elem.clone())).to_u32()
                    } else if col_u32 == q {
                        // Column q: i
                        i
                    } else {
                        // Columns q+1 to 2q: i² + m*i + j where m = col - q - 1
                        let m = col_u32 - q - 1;
                        let m_elem = self.field.element(m);
                        // i² + m*i + j
                        i_sq.add(m_elem.mul(i_elem.clone()))
                            .add(j_elem.clone())
                            .to_u32()
                    };

                    data[[row, col]] = value;
                }
            }
        }

        // Block 2: Transformed formulas using kay, b, c, k
        for i in 0..q {
            for j in 0..q {
                let row = (q * q + i * q + j) as usize;

                let i_elem = self.field.element(i);
                let j_elem = self.field.element(j);

                // kay * i²
                let kay_i_sq = kay_elem.mul(i_elem.mul(i_elem.clone()));

                for col in 0..factors {
                    let col_u32 = col as u32;

                    let value = if col_u32 == 0 {
                        // Column 0: j + b[0] = j (since b[0] = 0)
                        j_elem.add(self.field.element(self.b[0])).to_u32()
                    } else if col_u32 <= q - 1 {
                        // Columns 1 to q-1: i + m*j + b[m]
                        let m = col_u32 as usize;
                        let m_elem = self.field.element(m as u32);
                        let b_m = self.field.element(self.b[m]);
                        i_elem.add(m_elem.mul(j_elem.clone())).add(b_m).to_u32()
                    } else if col_u32 == q {
                        // Column q: i + b[0] (or could use a different constant)
                        // From the algorithm, this is typically just i unchanged
                        // Actually for column q, block 2 should use: i + some_constant
                        // Let's use b[0] = 0 for simplicity
                        i_elem.add(self.field.element(self.b[0])).to_u32()
                    } else {
                        // Columns q+1 to 2q: kay*i² + k[m]*i + j + c[m] where m = col - q - 1
                        let m = (col_u32 - q - 1) as usize;
                        let k_m = self.field.element(self.k[m]);
                        let c_m = self.field.element(self.c[m]);
                        // kay*i² + k[m]*i + j + c[m]
                        kay_i_sq
                            .add(k_m.mul(i_elem.clone()))
                            .add(j_elem.clone())
                            .add(c_m)
                            .to_u32()
                    };

                    data[[row, col]] = value;
                }
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for AddelmanKempthorne {
    fn name(&self) -> &'static str {
        "AddelmanKempthorne"
    }

    fn family(&self) -> &'static str {
        "OA(2q², k, q, 2), q odd prime power, k ≤ 2q+1"
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
        AddelmanKempthorne::construct(self, factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_addelman_creation_q3() {
        let ak = AddelmanKempthorne::new(3).unwrap();
        assert_eq!(ak.levels(), 3);
        assert_eq!(ak.runs(), 18);
        assert_eq!(ak.max_factors(), 7);
    }

    #[test]
    fn test_addelman_creation_q5() {
        let ak = AddelmanKempthorne::new(5).unwrap();
        assert_eq!(ak.levels(), 5);
        assert_eq!(ak.runs(), 50);
        assert_eq!(ak.max_factors(), 11);
    }

    #[test]
    fn test_addelman_creation_q7() {
        let ak = AddelmanKempthorne::new(7).unwrap();
        assert_eq!(ak.levels(), 7);
        assert_eq!(ak.runs(), 98);
        assert_eq!(ak.max_factors(), 15);
    }

    #[test]
    fn test_addelman_creation_q9() {
        // GF(9) = GF(3²) - extension field
        let ak = AddelmanKempthorne::new(9).unwrap();
        assert_eq!(ak.levels(), 9);
        assert_eq!(ak.runs(), 162);
        assert_eq!(ak.max_factors(), 19);
    }

    #[test]
    fn test_addelman_invalid_even() {
        // Powers of 2 are not allowed
        assert!(AddelmanKempthorne::new(2).is_err());
        assert!(AddelmanKempthorne::new(4).is_err());
        assert!(AddelmanKempthorne::new(8).is_err());
        assert!(AddelmanKempthorne::new(16).is_err());
    }

    #[test]
    fn test_addelman_invalid_non_prime_power() {
        // Not a prime power
        assert!(AddelmanKempthorne::new(6).is_err());
        assert!(AddelmanKempthorne::new(10).is_err());
        assert!(AddelmanKempthorne::new(12).is_err());
        assert!(AddelmanKempthorne::new(15).is_err());
    }

    #[test]
    fn test_addelman_q3_l18() {
        // The classic L18 array: OA(18, 7, 3, 2)
        let ak = AddelmanKempthorne::new(3).unwrap();
        let oa = ak.construct(7).unwrap();

        assert_eq!(oa.runs(), 18);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.levels(), 3);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L18 should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_addelman_q5() {
        let ak = AddelmanKempthorne::new(5).unwrap();
        let oa = ak.construct(11).unwrap();

        assert_eq!(oa.runs(), 50);
        assert_eq!(oa.factors(), 11);
        assert_eq!(oa.levels(), 5);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(50,11,5,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_addelman_q7() {
        let ak = AddelmanKempthorne::new(7).unwrap();
        let oa = ak.construct(10).unwrap();

        assert_eq!(oa.runs(), 98);
        assert_eq!(oa.factors(), 10);
        assert_eq!(oa.levels(), 7);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(98,10,7,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_addelman_q9() {
        let ak = AddelmanKempthorne::new(9).unwrap();
        let oa = ak.construct(10).unwrap();

        assert_eq!(oa.runs(), 162);
        assert_eq!(oa.factors(), 10);
        assert_eq!(oa.levels(), 9);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(162,10,9,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_addelman_fewer_factors() {
        let ak = AddelmanKempthorne::new(3).unwrap();
        let oa = ak.construct(4).unwrap();

        assert_eq!(oa.runs(), 18);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels(), 3);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(18,4,3,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_addelman_too_many_factors() {
        let ak = AddelmanKempthorne::new(3).unwrap();
        assert!(ak.construct(8).is_err()); // max is 7
    }

    #[test]
    fn test_addelman_zero_factors() {
        let ak = AddelmanKempthorne::new(3).unwrap();
        assert!(ak.construct(0).is_err());
    }

    #[test]
    fn test_addelman_single_factor() {
        let ak = AddelmanKempthorne::new(3).unwrap();
        let oa = ak.construct(1).unwrap();

        assert_eq!(oa.factors(), 1);
        assert_eq!(oa.runs(), 18);

        // Each level should appear 6 times (18/3)
        let mut counts = [0usize; 3];
        for row in 0..18 {
            counts[oa.get(row, 0) as usize] += 1;
        }
        assert_eq!(counts, [6, 6, 6]);
    }

    #[test]
    fn test_addelman_balance() {
        // Check that each column is balanced (each level appears equally often)
        let ak = AddelmanKempthorne::new(3).unwrap();
        let oa = ak.construct(7).unwrap();

        let expected_count = 18 / 3; // 6 occurrences of each level per column

        for col in 0..7 {
            let mut counts = [0usize; 3];
            for row in 0..18 {
                counts[oa.get(row, col) as usize] += 1;
            }
            assert_eq!(
                counts,
                [expected_count, expected_count, expected_count],
                "Column {} should be balanced",
                col
            );
        }
    }

    #[test]
    fn test_find_non_residue() {
        // In GF(3), 2 is a non-residue (1² = 1, 2² = 1, 0² = 0)
        // Residues: {0, 1}, Non-residue: 2
        let field = DynamicGf::new(3).unwrap();
        let nr = AddelmanKempthorne::find_non_residue(&field, 3).unwrap();
        assert_eq!(nr, 2);

        // In GF(5), residues are {0, 1, 4} since 1²=1, 2²=4, 3²=4, 4²=1
        // Non-residues: {2, 3}
        let field5 = DynamicGf::new(5).unwrap();
        let nr5 = AddelmanKempthorne::find_non_residue(&field5, 5).unwrap();
        assert!(nr5 == 2 || nr5 == 3);
    }
}

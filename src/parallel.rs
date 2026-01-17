//! Parallel construction support for orthogonal arrays.
//!
//! This module provides parallel versions of the construction algorithms using Rayon.
//! Enable with the `parallel` feature flag.
//!
//! # Usage
//!
//! ```ignore
//! use taguchi::parallel::ParBose;
//! use taguchi::construct::Constructor;
//!
//! let bose = ParBose::new(7);
//! let oa = bose.construct(8).unwrap();
//! assert_eq!(oa.runs(), 49);
//! ```
//!
//! # Performance
//!
//! Parallel construction is most beneficial for:
//! - Large arrays (q ≥ 16 or many factors)
//! - Higher-strength Bush constructions
//! - Systems with multiple cores
//!
//! For small arrays, the sequential versions may be faster due to parallelization overhead.

use ndarray::Array2;
use rayon::prelude::*;

use crate::construct::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OA, OAParams};
use crate::utils::{factor_prime_power, is_prime_power};

/// Parallel Bose construction for strength-2 orthogonal arrays.
///
/// Produces OA(q², k, q, 2) where q is a prime power and k ≤ q+1.
/// Uses parallel row generation for improved performance on multi-core systems.
#[derive(Debug, Clone)]
pub struct ParBose {
    q: u32,
    field: DynamicGf,
}

impl ParBose {
    /// Create a new parallel Bose constructor.
    ///
    /// # Panics
    ///
    /// Panics if `q` is not a prime power.
    #[must_use]
    pub fn new(q: u32) -> Self {
        Self::try_new(q).expect("q must be a prime power")
    }

    /// Create a new parallel Bose constructor, returning an error if q is invalid.
    pub fn try_new(q: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "ParBose",
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

    /// Get the number of runs.
    #[must_use]
    pub fn runs(&self) -> usize {
        (self.q * self.q) as usize
    }

    /// Get the maximum number of factors.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        (self.q + 1) as usize
    }

    /// Construct the orthogonal array using parallel row generation.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "ParBose",
            });
        }
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let runs = self.runs();

        // Generate rows in parallel
        let rows: Vec<Vec<u32>> = (0..runs)
            .into_par_iter()
            .map(|row_idx| {
                let i = (row_idx as u32) / q;
                let j = (row_idx as u32) % q;

                let elem_i = self.field.element(i);
                let elem_j = self.field.element(j);

                let mut row = Vec::with_capacity(factors);

                // Column 0: j
                row.push(j);

                // Columns 1 to factors-1: i + c*j
                for c in 1..factors {
                    let c_elem = self.field.element(c as u32);
                    let value = elem_i.add(c_elem.mul(elem_j.clone()));
                    row.push(value.to_u32());
                }

                row
            })
            .collect();

        // Convert to ndarray
        let mut data = Array2::zeros((runs, factors));
        for (row_idx, row) in rows.into_iter().enumerate() {
            for (col_idx, val) in row.into_iter().enumerate() {
                data[[row_idx, col_idx]] = val;
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for ParBose {
    fn name(&self) -> &'static str {
        "ParBose"
    }

    fn family(&self) -> &'static str {
        "OA(q², k, q, 2), k ≤ q+1 [parallel]"
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
        ParBose::construct(self, factors)
    }
}

/// Parallel Bush construction for higher-strength orthogonal arrays.
///
/// Produces OA(q^t, k, q, t) where q is a prime power and k ≤ t+1.
#[derive(Debug, Clone)]
pub struct ParBush {
    q: u32,
    t: u32,
    field: DynamicGf,
}

impl ParBush {
    /// Create a new parallel Bush constructor.
    pub fn new(q: u32, t: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "ParBush",
            });
        }
        if t < 2 {
            return Err(Error::invalid_params("strength must be at least 2"));
        }
        let field = DynamicGf::new(q)?;
        Ok(Self { q, t, field })
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.q
    }

    /// Get the strength.
    #[must_use]
    pub fn strength(&self) -> u32 {
        self.t
    }

    /// Get the number of runs.
    #[must_use]
    pub fn runs(&self) -> usize {
        self.q.pow(self.t) as usize
    }

    /// Get the maximum number of factors.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        (self.t + 1) as usize
    }

    /// Construct the orthogonal array using parallel row generation.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "ParBush",
            });
        }
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let t = self.t;
        let runs = self.runs();

        // Generate rows in parallel
        let rows: Vec<Vec<u32>> = (0..runs)
            .into_par_iter()
            .map(|row_idx| {
                // Extract polynomial coefficients from row index
                let mut coeffs = Vec::with_capacity(t as usize);
                let mut temp = row_idx as u32;
                for _ in 0..t {
                    coeffs.push(temp % q);
                    temp /= q;
                }

                let mut row = Vec::with_capacity(factors);

                // Columns 0 to t-1: Evaluate polynomial at x = 0, 1, ..., t-1
                for col in 0..factors.min(t as usize) {
                    let x = col as u32;
                    let mut val = self.field.element(0);

                    // Horner's method: p(x) = a_0 + x*(a_1 + x*(a_2 + ...))
                    for i in (0..t as usize).rev() {
                        let coeff = self.field.element(coeffs[i]);
                        let x_elem = self.field.element(x);
                        val = coeff.add(x_elem.mul(val));
                    }
                    row.push(val.to_u32());
                }

                // Column t: Leading coefficient (point at infinity)
                if factors > t as usize {
                    row.push(coeffs[(t - 1) as usize]);
                }

                row
            })
            .collect();

        // Convert to ndarray
        let mut data = Array2::zeros((runs, factors));
        for (row_idx, row) in rows.into_iter().enumerate() {
            for (col_idx, val) in row.into_iter().enumerate() {
                data[[row_idx, col_idx]] = val;
            }
        }

        let strength = t.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for ParBush {
    fn name(&self) -> &'static str {
        "ParBush"
    }

    fn family(&self) -> &'static str {
        "OA(q^t, k, q, t), k ≤ t+1 [parallel]"
    }

    fn levels(&self) -> u32 {
        self.q
    }

    fn strength(&self) -> u32 {
        self.t
    }

    fn runs(&self) -> usize {
        self.q.pow(self.t) as usize
    }

    fn max_factors(&self) -> usize {
        (self.t + 1) as usize
    }

    fn construct(&self, factors: usize) -> Result<OA> {
        ParBush::construct(self, factors)
    }
}

/// Parallel Addelman-Kempthorne construction for odd prime powers.
///
/// Produces OA(2q², k, q, 2) where q is an odd prime power and k ≤ 2q+1.
#[derive(Debug, Clone)]
pub struct ParAddelmanKempthorne {
    q: u32,
    field: DynamicGf,
    kay: u32,
    b: Vec<u32>,
    c: Vec<u32>,
    k: Vec<u32>,
}

impl ParAddelmanKempthorne {
    /// Create a new parallel Addelman-Kempthorne constructor.
    pub fn new(q: u32) -> Result<Self> {
        let factorization = factor_prime_power(q).ok_or(Error::LevelsNotPrimePower {
            levels: q,
            algorithm: "ParAddelmanKempthorne",
        })?;

        if factorization.prime == 2 {
            return Err(Error::invalid_params(
                "ParAddelmanKempthorne requires an ODD prime power. Use ParBoseBush for powers of 2.",
            ));
        }

        let field = DynamicGf::new(q)?;
        let p = factorization.prime;

        // Compute transformation constants
        let kay = Self::find_non_residue(&field, q)?;

        let four = if p == 3 { 1u32 } else { 4u32 };
        let four_elem = field.element(four);
        let kay_elem = field.element(kay);
        let neg_one = field.element(q - 1);
        let kay_minus_one = kay_elem.add(neg_one);

        let four_inv = if four == 1 {
            field.element(1)
        } else {
            four_elem.inv()
        };

        let mut b = vec![0u32; q as usize];
        let mut c = vec![0u32; q as usize];
        let mut k_arr = vec![0u32; q as usize];

        for m in 1..q {
            let m_elem = field.element(m);
            k_arr[m as usize] = kay_elem.mul(m_elem.clone()).to_u32();

            let denom = kay_elem.mul(four_elem.clone()).mul(m_elem.clone());
            let denom_inv = denom.inv();
            b[m as usize] = kay_minus_one.mul(denom_inv).to_u32();

            let m_sq = m_elem.clone().mul(m_elem);
            c[m as usize] = m_sq
                .mul(kay_minus_one.clone())
                .mul(four_inv.clone())
                .to_u32();
        }

        Ok(Self {
            q,
            field,
            kay,
            b,
            c,
            k: k_arr,
        })
    }

    fn find_non_residue(field: &DynamicGf, q: u32) -> Result<u32> {
        let mut is_residue = vec![false; q as usize];
        is_residue[0] = true;

        for y in 1..q {
            let y_elem = field.element(y);
            let y_sq = y_elem.clone().mul(y_elem).to_u32();
            is_residue[y_sq as usize] = true;
        }

        for x in 1..q {
            if !is_residue[x as usize] {
                return Ok(x);
            }
        }

        Err(Error::invalid_params(
            "Failed to find quadratic non-residue",
        ))
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.q
    }

    /// Get the number of runs.
    #[must_use]
    pub fn runs(&self) -> usize {
        (2 * self.q * self.q) as usize
    }

    /// Get the maximum number of factors.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        (2 * self.q + 1) as usize
    }

    /// Construct the orthogonal array using parallel row generation.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "ParAddelmanKempthorne",
            });
        }
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let q = self.q;
        let runs = self.runs();
        let kay_elem = self.field.element(self.kay);

        // Generate rows in parallel
        let rows: Vec<Vec<u32>> = (0..runs)
            .into_par_iter()
            .map(|row_idx| {
                let block = row_idx / (q * q) as usize;
                let within_block = row_idx % (q * q) as usize;
                let i = (within_block as u32) / q;
                let j = (within_block as u32) % q;

                let i_elem = self.field.element(i);
                let j_elem = self.field.element(j);
                let i_sq = i_elem.mul(i_elem.clone());

                let mut row = Vec::with_capacity(factors);

                if block == 0 {
                    // Block 1: Direct formulas
                    for col in 0..factors {
                        let col_u32 = col as u32;
                        let value = if col_u32 == 0 {
                            j
                        } else if col_u32 <= q - 1 {
                            let m = col_u32;
                            let m_elem = self.field.element(m);
                            i_elem.add(m_elem.mul(j_elem.clone())).to_u32()
                        } else if col_u32 == q {
                            i
                        } else {
                            let m = col_u32 - q - 1;
                            let m_elem = self.field.element(m);
                            i_sq.add(m_elem.mul(i_elem.clone()))
                                .add(j_elem.clone())
                                .to_u32()
                        };
                        row.push(value);
                    }
                } else {
                    // Block 2: Transformed formulas
                    let kay_i_sq = kay_elem.mul(i_elem.mul(i_elem.clone()));

                    for col in 0..factors {
                        let col_u32 = col as u32;
                        let value = if col_u32 == 0 {
                            j_elem.add(self.field.element(self.b[0])).to_u32()
                        } else if col_u32 <= q - 1 {
                            let m = col_u32 as usize;
                            let m_elem = self.field.element(m as u32);
                            let b_m = self.field.element(self.b[m]);
                            i_elem.add(m_elem.mul(j_elem.clone())).add(b_m).to_u32()
                        } else if col_u32 == q {
                            i_elem.add(self.field.element(self.b[0])).to_u32()
                        } else {
                            let m = (col_u32 - q - 1) as usize;
                            let k_m = self.field.element(self.k[m]);
                            let c_m = self.field.element(self.c[m]);
                            kay_i_sq
                                .add(k_m.mul(i_elem.clone()))
                                .add(j_elem.clone())
                                .add(c_m)
                                .to_u32()
                        };
                        row.push(value);
                    }
                }

                row
            })
            .collect();

        // Convert to ndarray
        let mut data = Array2::zeros((runs, factors));
        for (row_idx, row) in rows.into_iter().enumerate() {
            for (col_idx, val) in row.into_iter().enumerate() {
                data[[row_idx, col_idx]] = val;
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, q, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for ParAddelmanKempthorne {
    fn name(&self) -> &'static str {
        "ParAddelmanKempthorne"
    }

    fn family(&self) -> &'static str {
        "OA(2q², k, q, 2), q odd prime power, k ≤ 2q+1 [parallel]"
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
        ParAddelmanKempthorne::construct(self, factors)
    }
}

/// Parallel Hadamard-Sylvester construction for binary arrays.
///
/// Produces OA(n, k, 2, 2) where n is a power of 2 and k ≤ n-1.
#[derive(Debug, Clone)]
pub struct ParHadamardSylvester {
    n: usize,
    /// Exponent m where n = 2^m (kept for potential future use)
    _m: u32,
}

impl ParHadamardSylvester {
    /// Create a new parallel Hadamard-Sylvester constructor.
    pub fn new(n: usize) -> Result<Self> {
        if n < 4 || !n.is_power_of_two() {
            return Err(Error::invalid_params(
                "n must be a power of 2 >= 4 for Hadamard-Sylvester",
            ));
        }
        let m = n.trailing_zeros();
        Ok(Self { n, _m: m })
    }

    /// Create a constructor for the given number of factors.
    pub fn for_factors(factors: usize) -> Result<Self> {
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }
        let n = (factors + 1).next_power_of_two().max(4);
        Self::new(n)
    }

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        2
    }

    /// Get the number of runs.
    #[must_use]
    pub fn runs(&self) -> usize {
        self.n
    }

    /// Get the maximum number of factors.
    #[must_use]
    pub fn max_factors(&self) -> usize {
        self.n - 1
    }

    /// Construct the orthogonal array using parallel row generation.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "ParHadamardSylvester",
            });
        }
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        let n = self.n;

        // Generate rows in parallel using Sylvester construction
        // H[i,j] = (-1)^(popcount(i & j))
        let rows: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = Vec::with_capacity(factors);
                // Skip column 0 (all same sign), start from column 1
                for j in 1..=factors {
                    // Sylvester Hadamard: entry = (-1)^popcount(i & j)
                    let bits = (i & j).count_ones();
                    // +1 -> 0, -1 -> 1
                    row.push(bits % 2);
                }
                row
            })
            .collect();

        // Convert to ndarray
        let mut data = Array2::zeros((n, factors));
        for (row_idx, row) in rows.into_iter().enumerate() {
            for (col_idx, val) in row.into_iter().enumerate() {
                data[[row_idx, col_idx]] = val;
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(n, factors, 2, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for ParHadamardSylvester {
    fn name(&self) -> &'static str {
        "ParHadamardSylvester"
    }

    fn family(&self) -> &'static str {
        "OA(2^m, k, 2, 2), k ≤ 2^m - 1 [parallel]"
    }

    fn levels(&self) -> u32 {
        2
    }

    fn strength(&self) -> u32 {
        2
    }

    fn runs(&self) -> usize {
        self.n
    }

    fn max_factors(&self) -> usize {
        self.n - 1
    }

    fn construct(&self, factors: usize) -> Result<OA> {
        ParHadamardSylvester::construct(self, factors)
    }
}

/// Convenience function to construct an OA in parallel.
///
/// Automatically selects the best parallel constructor based on parameters.
pub fn par_build_oa(levels: u32, factors: usize, strength: u32) -> Result<OA> {
    let prime_power = factor_prime_power(levels);

    // For binary factors with strength 2, use Hadamard-Sylvester
    if levels == 2 && strength == 2 {
        let n = (factors + 1).next_power_of_two().max(4);
        if n - 1 >= factors {
            let h = ParHadamardSylvester::new(n)?;
            return h.construct(factors);
        }
    }

    // For strength 2 with prime power levels
    if strength == 2 {
        if let Some(ref pf) = prime_power {
            let q = levels;

            // Try Bose first (fewer runs)
            if factors <= (q + 1) as usize {
                let bose = ParBose::try_new(q)?;
                return bose.construct(factors);
            }

            // Try Addelman-Kempthorne for odd prime powers
            if pf.prime != 2 && factors <= (2 * q + 1) as usize {
                let ak = ParAddelmanKempthorne::new(q)?;
                return ak.construct(factors);
            }
        }
    }

    // For higher strength, try Bush
    if let Some(ref _pf) = prime_power {
        let q = levels;
        let t = strength;
        if factors <= (t + 1) as usize {
            let bush = ParBush::new(q, t)?;
            return bush.construct(factors);
        }
    }

    Err(Error::invalid_params(format!(
        "No parallel construction available for OA(?, {}, {}, {})",
        factors, levels, strength
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_par_bose_q3() {
        let bose = ParBose::new(3);
        let oa = bose.construct(4).unwrap();

        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.symmetric_levels(), 3);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "ParBose L9 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_par_bose_q7() {
        let bose = ParBose::new(7);
        let oa = bose.construct(8).unwrap();

        assert_eq!(oa.runs(), 49);
        assert_eq!(oa.factors(), 8);
        assert_eq!(oa.symmetric_levels(), 7);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "ParBose L49 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_par_bush_strength_3() {
        let bush = ParBush::new(3, 3).unwrap();
        let oa = bush.construct(4).unwrap();

        assert_eq!(oa.runs(), 27);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.strength(), 3);
        assert_eq!(oa.symmetric_levels(), 3);

        let result = verify_strength(&oa, 3).unwrap();
        assert!(
            result.is_valid,
            "ParBush should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_par_addelman_q3() {
        let ak = ParAddelmanKempthorne::new(3).unwrap();
        let oa = ak.construct(7).unwrap();

        assert_eq!(oa.runs(), 18);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.symmetric_levels(), 3);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "ParAddelmanKempthorne L18 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_par_hadamard_sylvester() {
        let h = ParHadamardSylvester::new(8).unwrap();
        let oa = h.construct(7).unwrap();

        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.symmetric_levels(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "ParHadamardSylvester should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_par_hadamard_sylvester_16() {
        let h = ParHadamardSylvester::new(16).unwrap();
        let oa = h.construct(15).unwrap();

        assert_eq!(oa.runs(), 16);
        assert_eq!(oa.factors(), 15);
        assert_eq!(oa.symmetric_levels(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "ParHadamardSylvester 16 should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_par_build_oa_binary() {
        let oa = par_build_oa(2, 7, 2).unwrap();
        assert_eq!(oa.symmetric_levels(), 2);
        assert_eq!(oa.factors(), 7);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_par_build_oa_ternary() {
        let oa = par_build_oa(3, 4, 2).unwrap();
        assert_eq!(oa.symmetric_levels(), 3);
        assert_eq!(oa.factors(), 4);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_par_equivalence_bose() {
        // Verify parallel version produces equivalent results to sequential
        use crate::construct::Bose;

        let seq = Bose::new(5);
        let par = ParBose::new(5);

        let oa_seq = seq.construct(6).unwrap();
        let oa_par = par.construct(6).unwrap();

        // Both should have same dimensions
        assert_eq!(oa_seq.runs(), oa_par.runs());
        assert_eq!(oa_seq.factors(), oa_par.factors());

        // Both should be valid
        let result_seq = verify_strength(&oa_seq, 2).unwrap();
        let result_par = verify_strength(&oa_par, 2).unwrap();
        assert!(result_seq.is_valid);
        assert!(result_par.is_valid);
    }

    #[test]
    fn test_par_equivalence_addelman() {
        use crate::construct::AddelmanKempthorne;

        let seq = AddelmanKempthorne::new(5).unwrap();
        let par = ParAddelmanKempthorne::new(5).unwrap();

        let oa_seq = seq.construct(11).unwrap();
        let oa_par = par.construct(11).unwrap();

        assert_eq!(oa_seq.runs(), oa_par.runs());
        assert_eq!(oa_seq.factors(), oa_par.factors());

        let result_seq = verify_strength(&oa_seq, 2).unwrap();
        let result_par = verify_strength(&oa_par, 2).unwrap();
        assert!(result_seq.is_valid);
        assert!(result_par.is_valid);
    }
}

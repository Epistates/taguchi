//! Hadamard-based constructions for orthogonal arrays.
//!
//! Hadamard matrices give efficient OA(n, n-1, 2, 2) for various orders n.
//! These are particularly useful for screening experiments with many binary factors.
//!
//! ## Sylvester Construction
//!
//! For any power of 2, n = 2^k (k ≥ 2), there exists a Hadamard matrix of order n.
//! This gives OA(n, n-1, 2, 2).
//!
//! ## Example
//!
//! ```
//! use taguchi::construct::{Constructor, HadamardSylvester};
//!
//! // Create OA(8, 7, 2, 2) - 8 runs, 7 binary factors
//! let h = HadamardSylvester::new(8).unwrap();
//! let oa = h.construct(7).unwrap();
//!
//! assert_eq!(oa.runs(), 8);
//! assert_eq!(oa.factors(), 7);
//! assert_eq!(oa.levels(), 2);
//! assert_eq!(oa.strength(), 2);
//! ```

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::oa::{OA, OAParams};

/// Sylvester-Hadamard construction for strength-2 binary orthogonal arrays.
///
/// Produces OA(n, k, 2, 2) where n = 2^m for some m ≥ 2, and k ≤ n-1.
///
/// The Sylvester construction builds Hadamard matrices recursively:
/// - H₁ = \[1\]
/// - H_{2n} = \[\[H_n, H_n\], \[H_n, -H_n\]\]
///
/// This is the most efficient construction for binary OAs when the number
/// of runs is a power of 2.
///
/// # Properties
///
/// - Runs: n (power of 2)
/// - Factors: up to n-1
/// - Levels: 2
/// - Strength: 2
///
/// # Example
///
/// ```
/// use taguchi::construct::{Constructor, HadamardSylvester};
///
/// // OA(16, 15, 2, 2) - can handle up to 15 binary factors
/// let h = HadamardSylvester::new(16).unwrap();
/// let oa = h.construct(10).unwrap();  // Use only 10 factors
///
/// assert_eq!(oa.runs(), 16);
/// assert_eq!(oa.factors(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct HadamardSylvester {
    /// Order of the Hadamard matrix (must be a power of 2, at least 4).
    n: usize,
    /// Exponent m where n = 2^m.
    _m: u32,
}

impl HadamardSylvester {
    /// Create a new Sylvester-Hadamard constructor.
    ///
    /// # Arguments
    ///
    /// * `n` - Order of the Hadamard matrix (must be a power of 2, at least 4)
    ///
    /// # Errors
    ///
    /// Returns an error if `n` is not a power of 2 or is less than 4.
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::HadamardSylvester;
    ///
    /// let h4 = HadamardSylvester::new(4).unwrap();   // OA(4, 3, 2, 2)
    /// let h8 = HadamardSylvester::new(8).unwrap();   // OA(8, 7, 2, 2)
    /// let h16 = HadamardSylvester::new(16).unwrap(); // OA(16, 15, 2, 2)
    ///
    /// assert!(HadamardSylvester::new(3).is_err());   // Not a power of 2
    /// assert!(HadamardSylvester::new(2).is_err());   // Too small
    /// ```
    pub fn new(n: usize) -> Result<Self> {
        // Check n is a power of 2
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(Error::invalid_params(format!(
                "HadamardSylvester requires n to be a power of 2, got {n}"
            )));
        }

        // n must be at least 4 (n=2 gives only 1 factor which is trivial)
        if n < 4 {
            return Err(Error::invalid_params(
                "HadamardSylvester requires n >= 4 for a valid OA",
            ));
        }

        let m = n.trailing_zeros();

        Ok(Self { n, _m: m })
    }

    /// Create a Hadamard constructor for a given number of factors.
    ///
    /// Finds the smallest power of 2 that can accommodate the requested
    /// number of factors.
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::HadamardSylvester;
    ///
    /// let h = HadamardSylvester::for_factors(5).unwrap();
    /// assert_eq!(h.runs(), 8);  // Smallest 2^k where k-1 >= 5 is 8
    /// ```
    pub fn for_factors(factors: usize) -> Result<Self> {
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        // Find smallest n = 2^k where n-1 >= factors
        let mut n = 4;
        while n - 1 < factors {
            n *= 2;
            if n > 1 << 24 {
                return Err(Error::invalid_params(format!(
                    "Cannot create Hadamard OA for {factors} factors (would require too many runs)"
                )));
            }
        }

        Self::new(n)
    }

    /// Build the Sylvester-Hadamard matrix of order n.
    ///
    /// The matrix has entries +1 and -1, with first row and column all +1.
    fn build_hadamard_matrix(&self) -> Array2<i8> {
        let n = self.n;
        let mut h = Array2::from_elem((n, n), 1i8);

        // Build iteratively by doubling the size each step.
        // Start with H_1 = [1] implicitly (the top-left corner).
        //
        // Sylvester recursion: H_{2n} = [[H_n, H_n], [H_n, -H_n]]
        let mut size = 1;
        while size < n {
            // At this point, h[0..size, 0..size] contains H_size.
            // Fill in the three other quadrants.

            // Top-right: copy of H_size
            for i in 0..size {
                for j in 0..size {
                    h[[i, j + size]] = h[[i, j]];
                }
            }

            // Bottom-left: copy of H_size
            for i in 0..size {
                for j in 0..size {
                    h[[i + size, j]] = h[[i, j]];
                }
            }

            // Bottom-right: negation of H_size
            for i in 0..size {
                for j in 0..size {
                    h[[i + size, j + size]] = -h[[i, j]];
                }
            }

            size *= 2;
        }

        h
    }

    /// Get the order of the Hadamard matrix.
    #[must_use]
    pub fn order(&self) -> usize {
        self.n
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

    /// Construct the orthogonal array.
    ///
    /// # Errors
    ///
    /// Returns an error if `factors` exceeds `n - 1`.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "HadamardSylvester",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        // Build the Hadamard matrix
        let h = self.build_hadamard_matrix();

        // Convert to OA:
        // - All n rows
        // - Skip column 0 (all ones), take columns 1..=factors
        // - Map: 1 → 0, -1 → 1
        let n = self.n;
        let mut data = Array2::zeros((n, factors));

        for i in 0..n {
            for j in 0..factors {
                // Column j+1 in Hadamard matrix -> column j in OA
                data[[i, j]] = if h[[i, j + 1]] == 1 { 0 } else { 1 };
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(n, factors, 2, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for HadamardSylvester {
    fn name(&self) -> &'static str {
        "HadamardSylvester"
    }

    fn family(&self) -> &'static str {
        "OA(2^m, 2^m-1, 2, 2), m ≥ 2"
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
        HadamardSylvester::construct(self, factors)
    }
}

/// Paley-Hadamard construction for strength-2 binary orthogonal arrays.
///
/// Produces OA(p+1, k, 2, 2) where p ≡ 3 (mod 4) is prime, and k ≤ p.
///
/// The Paley construction uses quadratic residues to build Hadamard matrices
/// of order p+1 for primes p ≡ 3 (mod 4). This complements the Sylvester
/// construction by providing Hadamard matrices of orders that are not powers of 2.
///
/// # Properties
///
/// - Runs: p+1 (one more than a prime p ≡ 3 (mod 4))
/// - Factors: up to p
/// - Levels: 2
/// - Strength: 2
///
/// # Example
///
/// ```
/// use taguchi::construct::{Constructor, HadamardPaley};
///
/// // OA(12, 11, 2, 2) using p=11
/// let h = HadamardPaley::new(11).unwrap();
/// let oa = h.construct(11).unwrap();
///
/// assert_eq!(oa.runs(), 12);
/// assert_eq!(oa.factors(), 11);
/// assert_eq!(oa.levels(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct HadamardPaley {
    /// The prime p where p ≡ 3 (mod 4).
    p: u32,
    /// Order of the Hadamard matrix (p+1).
    n: usize,
}

impl HadamardPaley {
    /// Create a new Paley-Hadamard constructor.
    ///
    /// # Arguments
    ///
    /// * `p` - A prime congruent to 3 (mod 4)
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is not a prime or p ≢ 3 (mod 4).
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::HadamardPaley;
    ///
    /// // Valid primes p ≡ 3 (mod 4)
    /// assert!(HadamardPaley::new(3).is_ok());   // OA(4, 3, 2, 2)
    /// assert!(HadamardPaley::new(7).is_ok());   // OA(8, 7, 2, 2)
    /// assert!(HadamardPaley::new(11).is_ok());  // OA(12, 11, 2, 2)
    /// assert!(HadamardPaley::new(19).is_ok());  // OA(20, 19, 2, 2)
    ///
    /// // Invalid: not prime or p ≢ 3 (mod 4)
    /// assert!(HadamardPaley::new(5).is_err());  // 5 ≡ 1 (mod 4)
    /// assert!(HadamardPaley::new(9).is_err());  // 9 = 3² not prime
    /// assert!(HadamardPaley::new(13).is_err()); // 13 ≡ 1 (mod 4)
    /// ```
    pub fn new(p: u32) -> Result<Self> {
        // Check that p is prime
        if !crate::utils::is_prime(p) {
            return Err(Error::invalid_params(format!(
                "HadamardPaley requires a prime, but {p} is not prime"
            )));
        }

        // Check p ≡ 3 (mod 4)
        if p % 4 != 3 {
            return Err(Error::invalid_params(format!(
                "HadamardPaley requires p ≡ 3 (mod 4), but {p} ≡ {} (mod 4)",
                p % 4
            )));
        }

        let n = (p + 1) as usize;

        Ok(Self { p, n })
    }

    /// Find the smallest valid prime p ≡ 3 (mod 4) that can accommodate
    /// the given number of factors.
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::construct::HadamardPaley;
    ///
    /// let h = HadamardPaley::for_factors(10).unwrap();
    /// assert_eq!(h.prime(), 11);  // Smallest p ≡ 3 (mod 4) where p >= 10
    /// assert_eq!(h.runs(), 12);
    /// ```
    pub fn for_factors(factors: usize) -> Result<Self> {
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        // Find smallest prime p ≡ 3 (mod 4) where p >= factors
        // Start from factors, find next candidate
        let mut candidate = factors as u32;

        // Adjust to be ≡ 3 (mod 4)
        let rem = candidate % 4;
        if rem != 3 {
            candidate += (3 + 4 - rem) % 4;
            if candidate < factors as u32 {
                candidate += 4;
            }
        }

        // Search for prime
        while candidate < (1 << 24) {
            if crate::utils::is_prime(candidate) && candidate % 4 == 3 {
                return Self::new(candidate);
            }
            candidate += 4; // Stay ≡ 3 (mod 4)
        }

        Err(Error::invalid_params(format!(
            "Cannot find suitable prime for {factors} factors"
        )))
    }

    /// Compute the Legendre symbol (a/p).
    ///
    /// Returns 1 if a is a quadratic residue mod p, -1 otherwise, 0 if a ≡ 0.
    fn legendre(&self, a: u32) -> i8 {
        let a = a % self.p;
        if a == 0 {
            return 0;
        }

        // Use Euler's criterion: a^((p-1)/2) ≡ (a/p) (mod p)
        let exp = (self.p - 1) / 2;
        let result = crate::utils::mod_pow(u64::from(a), u64::from(exp), u64::from(self.p));

        if result == 1 {
            1
        } else {
            -1 // result == p-1 ≡ -1 (mod p)
        }
    }

    /// Build the Paley-Hadamard matrix of order p+1.
    ///
    /// Uses the Paley Type I construction for p ≡ 3 (mod 4), then normalizes
    /// so that both the first row and first column are all +1s.
    ///
    /// The construction is:
    /// 1. Build the raw Paley matrix with first row all +1, first column mixed
    /// 2. Normalize by negating rows where first element is -1
    fn build_hadamard_matrix(&self) -> Array2<i8> {
        let n = self.n;
        let p = self.p;
        let mut h = Array2::from_elem((n, n), 1i8);

        // First row: all 1s (already set)

        // First column: [1, -1, -1, ..., -1] in raw form
        for i in 1..n {
            h[[i, 0]] = -1;
        }

        // Interior (1..n) × (1..n): Q + I
        // Q[i,j] = χ(j-i) where indices are 0..p-1 (shifted by 1 from matrix indices)
        for i in 1..n {
            for j in 1..n {
                // Map matrix indices to field elements
                let fi = (i - 1) as u32; // 0..p-1
                let fj = (j - 1) as u32; // 0..p-1

                if fi == fj {
                    // Diagonal: I contribution
                    h[[i, j]] = 1;
                } else {
                    // Off-diagonal: χ(j-i)
                    let diff = (fj + p - fi) % p;
                    h[[i, j]] = self.legendre(diff);
                }
            }
        }

        // Normalize: negate rows where first element is -1
        // This ensures first column is all +1s while preserving orthogonality
        for i in 0..n {
            if h[[i, 0]] == -1 {
                for j in 0..n {
                    h[[i, j]] = -h[[i, j]];
                }
            }
        }

        // Also normalize columns if needed (though rows should suffice for Paley)
        for j in 0..n {
            if h[[0, j]] == -1 {
                for i in 0..n {
                    h[[i, j]] = -h[[i, j]];
                }
            }
        }

        h
    }

    /// Get the prime used in this construction.
    #[must_use]
    pub fn prime(&self) -> u32 {
        self.p
    }

    /// Get the order of the Hadamard matrix.
    #[must_use]
    pub fn order(&self) -> usize {
        self.n
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

    /// Construct the orthogonal array.
    ///
    /// # Errors
    ///
    /// Returns an error if `factors` exceeds `p`.
    pub fn construct(&self, factors: usize) -> Result<OA> {
        let max = self.max_factors();
        if factors > max {
            return Err(Error::TooManyFactors {
                factors,
                max,
                algorithm: "HadamardPaley",
            });
        }

        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        // Build the Hadamard matrix
        let h = self.build_hadamard_matrix();

        // Convert to OA (same as Sylvester)
        let n = self.n;
        let mut data = Array2::zeros((n, factors));

        for i in 0..n {
            for j in 0..factors {
                // Column j+1 in Hadamard matrix -> column j in OA
                data[[i, j]] = if h[[i, j + 1]] == 1 { 0 } else { 1 };
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(n, factors, 2, strength)?;
        Ok(OA::new(data, params))
    }
}

impl Constructor for HadamardPaley {
    fn name(&self) -> &'static str {
        "HadamardPaley"
    }

    fn family(&self) -> &'static str {
        "OA(p+1, p, 2, 2), p ≡ 3 (mod 4) prime"
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
        HadamardPaley::construct(self, factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_hadamard_sylvester_creation() {
        assert!(HadamardSylvester::new(4).is_ok());
        assert!(HadamardSylvester::new(8).is_ok());
        assert!(HadamardSylvester::new(16).is_ok());
        assert!(HadamardSylvester::new(32).is_ok());
        assert!(HadamardSylvester::new(64).is_ok());
    }

    #[test]
    fn test_hadamard_sylvester_invalid() {
        assert!(HadamardSylvester::new(0).is_err());
        assert!(HadamardSylvester::new(1).is_err());
        assert!(HadamardSylvester::new(2).is_err()); // Too small
        assert!(HadamardSylvester::new(3).is_err()); // Not power of 2
        assert!(HadamardSylvester::new(5).is_err());
        assert!(HadamardSylvester::new(6).is_err());
        assert!(HadamardSylvester::new(7).is_err());
        assert!(HadamardSylvester::new(12).is_err());
    }

    #[test]
    fn test_hadamard_for_factors() {
        let h = HadamardSylvester::for_factors(3).unwrap();
        assert_eq!(h.order(), 4); // 4-1 = 3 >= 3

        let h = HadamardSylvester::for_factors(5).unwrap();
        assert_eq!(h.order(), 8); // 8-1 = 7 >= 5

        let h = HadamardSylvester::for_factors(7).unwrap();
        assert_eq!(h.order(), 8); // 8-1 = 7 >= 7

        let h = HadamardSylvester::for_factors(8).unwrap();
        assert_eq!(h.order(), 16); // 16-1 = 15 >= 8
    }

    #[test]
    fn test_hadamard_sylvester_oa4() {
        let h = HadamardSylvester::new(4).unwrap();
        let oa = h.construct(3).unwrap();

        assert_eq!(oa.runs(), 4);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(4,3,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_sylvester_oa8() {
        let h = HadamardSylvester::new(8).unwrap();
        let oa = h.construct(7).unwrap();

        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(8,7,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_sylvester_oa16() {
        let h = HadamardSylvester::new(16).unwrap();
        let oa = h.construct(15).unwrap();

        assert_eq!(oa.runs(), 16);
        assert_eq!(oa.factors(), 15);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(16,15,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_sylvester_oa32() {
        let h = HadamardSylvester::new(32).unwrap();
        let oa = h.construct(31).unwrap();

        assert_eq!(oa.runs(), 32);
        assert_eq!(oa.factors(), 31);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "OA(32,31,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_sylvester_fewer_factors() {
        let h = HadamardSylvester::new(8).unwrap();

        for k in 1..=7 {
            let oa = h.construct(k).unwrap();
            assert_eq!(oa.factors(), k);

            if k >= 2 {
                let result = verify_strength(&oa, 2).unwrap();
                assert!(
                    result.is_valid,
                    "OA(8,{},2,2) should be valid: {:?}",
                    k, result.issues
                );
            }
        }
    }

    #[test]
    fn test_hadamard_sylvester_too_many_factors() {
        let h = HadamardSylvester::new(8).unwrap();
        assert!(h.construct(8).is_err());
        assert!(h.construct(10).is_err());
    }

    #[test]
    fn test_hadamard_sylvester_single_factor() {
        let h = HadamardSylvester::new(4).unwrap();
        let oa = h.construct(1).unwrap();

        assert_eq!(oa.factors(), 1);
        assert_eq!(oa.runs(), 4);

        // Each level should appear 2 times (4/2)
        let mut counts = [0usize; 2];
        for row in 0..4 {
            counts[oa.get(row, 0) as usize] += 1;
        }
        assert_eq!(counts, [2, 2]);
    }

    #[test]
    fn test_hadamard_matrix_structure() {
        let h = HadamardSylvester::new(4).unwrap();
        let mat = h.build_hadamard_matrix();

        // First row should be all 1s
        for j in 0..4 {
            assert_eq!(mat[[0, j]], 1, "First row should be all 1s");
        }

        // First column should be all 1s
        for i in 0..4 {
            assert_eq!(mat[[i, 0]], 1, "First column should be all 1s");
        }

        // All entries should be ±1
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    mat[[i, j]] == 1 || mat[[i, j]] == -1,
                    "All entries should be ±1"
                );
            }
        }
    }

    #[test]
    fn test_hadamard_matrix_orthogonality() {
        // H * H^T = n * I for a Hadamard matrix
        let h = HadamardSylvester::new(8).unwrap();
        let mat = h.build_hadamard_matrix();
        let n = 8i32;

        for i in 0..8 {
            for j in 0..8 {
                let mut dot_product: i32 = 0;
                for k in 0..8 {
                    dot_product += mat[[i, k]] as i32 * mat[[j, k]] as i32;
                }

                if i == j {
                    assert_eq!(dot_product, n, "Diagonal should be n");
                } else {
                    assert_eq!(dot_product, 0, "Off-diagonal should be 0");
                }
            }
        }
    }

    #[test]
    fn test_hadamard_balance() {
        // In each non-first column, there should be equal +1 and -1
        let h = HadamardSylvester::new(8).unwrap();
        let mat = h.build_hadamard_matrix();

        for j in 1..8 {
            let mut plus_count = 0;
            let mut minus_count = 0;

            for i in 0..8 {
                if mat[[i, j]] == 1 {
                    plus_count += 1;
                } else {
                    minus_count += 1;
                }
            }

            assert_eq!(plus_count, 4, "Column {} should have 4 ones", j);
            assert_eq!(minus_count, 4, "Column {} should have 4 minus-ones", j);
        }
    }

    // ========== Paley Tests ==========

    #[test]
    fn test_hadamard_paley_creation() {
        // Valid primes p ≡ 3 (mod 4)
        assert!(HadamardPaley::new(3).is_ok());
        assert!(HadamardPaley::new(7).is_ok());
        assert!(HadamardPaley::new(11).is_ok());
        assert!(HadamardPaley::new(19).is_ok());
        assert!(HadamardPaley::new(23).is_ok());
        assert!(HadamardPaley::new(31).is_ok());
        assert!(HadamardPaley::new(43).is_ok());
    }

    #[test]
    fn test_hadamard_paley_invalid() {
        // Not prime
        assert!(HadamardPaley::new(9).is_err()); // 9 = 3²
        assert!(HadamardPaley::new(15).is_err()); // 15 = 3×5
        assert!(HadamardPaley::new(21).is_err()); // 21 = 3×7

        // Prime but p ≡ 1 (mod 4)
        assert!(HadamardPaley::new(5).is_err()); // 5 ≡ 1 (mod 4)
        assert!(HadamardPaley::new(13).is_err()); // 13 ≡ 1 (mod 4)
        assert!(HadamardPaley::new(17).is_err()); // 17 ≡ 1 (mod 4)
        assert!(HadamardPaley::new(29).is_err()); // 29 ≡ 1 (mod 4)

        // p ≡ 2 (mod 4) - only p=2, which is too small
        assert!(HadamardPaley::new(2).is_err());
    }

    #[test]
    fn test_hadamard_paley_for_factors() {
        let h = HadamardPaley::for_factors(3).unwrap();
        assert_eq!(h.prime(), 3); // Smallest p ≡ 3 (mod 4) where p >= 3

        let h = HadamardPaley::for_factors(5).unwrap();
        assert_eq!(h.prime(), 7); // 5, 6 don't work, 7 is next

        let h = HadamardPaley::for_factors(10).unwrap();
        assert_eq!(h.prime(), 11); // 11 is next valid

        let h = HadamardPaley::for_factors(12).unwrap();
        assert_eq!(h.prime(), 19); // 12-18 don't work, 19 is next
    }

    #[test]
    fn test_hadamard_paley_oa4() {
        // p=3: OA(4, 3, 2, 2)
        let h = HadamardPaley::new(3).unwrap();
        let oa = h.construct(3).unwrap();

        assert_eq!(oa.runs(), 4);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Paley OA(4,3,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_paley_oa8() {
        // p=7: OA(8, 7, 2, 2)
        let h = HadamardPaley::new(7).unwrap();
        let oa = h.construct(7).unwrap();

        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 7);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Paley OA(8,7,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_paley_oa12() {
        // p=11: OA(12, 11, 2, 2)
        let h = HadamardPaley::new(11).unwrap();
        let oa = h.construct(11).unwrap();

        assert_eq!(oa.runs(), 12);
        assert_eq!(oa.factors(), 11);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Paley OA(12,11,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_paley_oa20() {
        // p=19: OA(20, 19, 2, 2)
        let h = HadamardPaley::new(19).unwrap();
        let oa = h.construct(19).unwrap();

        assert_eq!(oa.runs(), 20);
        assert_eq!(oa.factors(), 19);
        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.strength(), 2);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(
            result.is_valid,
            "Paley OA(20,19,2,2) should be valid: {:?}",
            result.issues
        );
    }

    #[test]
    fn test_hadamard_paley_fewer_factors() {
        let h = HadamardPaley::new(11).unwrap();

        for k in 1..=11 {
            let oa = h.construct(k).unwrap();
            assert_eq!(oa.factors(), k);

            if k >= 2 {
                let result = verify_strength(&oa, 2).unwrap();
                assert!(
                    result.is_valid,
                    "Paley OA(12,{},2,2) should be valid: {:?}",
                    k, result.issues
                );
            }
        }
    }

    #[test]
    fn test_hadamard_paley_too_many_factors() {
        let h = HadamardPaley::new(7).unwrap();
        assert!(h.construct(8).is_err());
        assert!(h.construct(10).is_err());
    }

    #[test]
    fn test_hadamard_paley_matrix_orthogonality() {
        // H * H^T = n * I for a Hadamard matrix
        let h = HadamardPaley::new(7).unwrap();
        let mat = h.build_hadamard_matrix();
        let n = 8i32;

        for i in 0..8 {
            for j in 0..8 {
                let mut dot_product: i32 = 0;
                for k in 0..8 {
                    dot_product += mat[[i, k]] as i32 * mat[[j, k]] as i32;
                }

                if i == j {
                    assert_eq!(dot_product, n, "Diagonal should be n");
                } else {
                    assert_eq!(dot_product, 0, "Off-diagonal should be 0");
                }
            }
        }
    }

    #[test]
    fn test_hadamard_paley_balance() {
        // In each non-first column, there should be equal +1 and -1
        let h = HadamardPaley::new(7).unwrap();
        let mat = h.build_hadamard_matrix();

        for j in 1..8 {
            let mut plus_count = 0;
            let mut minus_count = 0;

            for i in 0..8 {
                if mat[[i, j]] == 1 {
                    plus_count += 1;
                } else {
                    minus_count += 1;
                }
            }

            assert_eq!(plus_count, 4, "Column {} should have 4 ones", j);
            assert_eq!(minus_count, 4, "Column {} should have 4 minus-ones", j);
        }
    }

    #[test]
    fn test_legendre_symbol() {
        let h = HadamardPaley::new(7).unwrap();

        // Quadratic residues mod 7: 1, 2, 4 (since 1²=1, 2²=4, 3²=2 mod 7)
        assert_eq!(h.legendre(0), 0);
        assert_eq!(h.legendre(1), 1); // QR
        assert_eq!(h.legendre(2), 1); // QR
        assert_eq!(h.legendre(3), -1); // NQR
        assert_eq!(h.legendre(4), 1); // QR
        assert_eq!(h.legendre(5), -1); // NQR
        assert_eq!(h.legendre(6), -1); // NQR
    }
}

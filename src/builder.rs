//! Builder pattern for constructing orthogonal arrays.
//!
//! The builder provides a convenient API for creating orthogonal arrays
//! without needing to know which construction algorithm to use.
//!
//! # Example
//!
//! ```
//! use taguchi::OABuilder;
//!
//! // Automatically selects the best construction
//! let oa = OABuilder::new()
//!     .levels(3)
//!     .factors(4)
//!     .strength(2)
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(oa.runs(), 9);    // Uses Bose: 3²
//! assert_eq!(oa.factors(), 4);
//! assert_eq!(oa.levels(), 3);
//! ```
//!
//! # Construction Selection
//!
//! The builder automatically selects the most efficient construction based on:
//!
//! - **Binary factors (levels=2)**: Hadamard-Sylvester for many factors
//! - **Prime power levels, strength 2**: Bose construction
//! - **Higher strength**: Bush construction
//! - **Power-of-2 levels, many factors**: Bose-Bush (when available)

use crate::construct::{
    AddelmanKempthorne, Bose, BoseBush, Bush, Constructor, HadamardSylvester, RaoHamming,
};
use crate::error::{Error, Result};
use crate::oa::OA;
use crate::utils::factor_prime_power;

/// Builder for constructing orthogonal arrays.
///
/// This builder automatically selects the most appropriate construction
/// algorithm based on the specified parameters.
///
/// # Example
///
/// ```
/// use taguchi::OABuilder;
///
/// // Simple construction
/// let oa = OABuilder::new()
///     .levels(5)
///     .factors(4)
///     .build()
///     .unwrap();
///
/// // With explicit strength
/// let oa = OABuilder::new()
///     .levels(3)
///     .factors(4)
///     .strength(3)
///     .build()
///     .unwrap();
///
/// assert_eq!(oa.strength(), 3);
/// assert_eq!(oa.runs(), 27);  // Bush: 3³
/// ```
#[derive(Debug, Clone, Default)]
pub struct OABuilder {
    levels: Option<Vec<u32>>,
    factors: Option<usize>,
    strength: Option<u32>,
    min_runs: Option<usize>,
}

impl OABuilder {
    /// Create a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of levels for all factors (symmetric OA).
    #[must_use]
    pub fn levels(mut self, levels: u32) -> Self {
        self.levels = Some(vec![levels]);
        self
    }

    /// Set the number of levels for each factor (mixed-level OA).
    #[must_use]
    pub fn mixed_levels(mut self, levels: Vec<u32>) -> Self {
        self.levels = Some(levels);
        self.factors = Some(self.levels.as_ref().unwrap().len());
        self
    }

    /// Set the number of factors.
    ///
    /// For symmetric OAs, this must be set. For mixed-level OAs, this is
    /// automatically set by `mixed_levels`.
    #[must_use]
    pub fn factors(mut self, factors: usize) -> Self {
        self.factors = Some(factors);
        if let Some(ref mut lv) = self.levels {
            if lv.len() == 1 && factors > 1 {
                let s = lv[0];
                *lv = vec![s; factors];
            }
        }
        self
    }

    /// Set the strength of the array.
    ///
    /// Strength t means any t columns contain all possible t-tuples equally often.
    /// Default is 2 if not specified.
    #[must_use]
    pub fn strength(mut self, strength: u32) -> Self {
        self.strength = Some(strength);
        self
    }

    /// Set a minimum number of runs.
    ///
    /// The builder will select a construction with at least this many runs.
    /// Useful when you need a minimum sample size.
    #[must_use]
    pub fn min_runs(mut self, min_runs: usize) -> Self {
        self.min_runs = Some(min_runs);
        self
    }

    /// Build the orthogonal array.
    ///
    /// Automatically selects the best construction algorithm based on
    /// the specified parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Required parameters (levels, factors) are not specified
    /// - No suitable construction exists for the given parameters
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::OABuilder;
    ///
    /// let oa = OABuilder::new()
    ///     .levels(7)
    ///     .factors(5)
    ///     .strength(2)
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(oa.runs(), 49);  // Bose: 7²
    /// ```
    pub fn build(self) -> Result<OA> {
        let levels_vec = self
            .levels
            .clone()
            .ok_or_else(|| Error::invalid_params("levels must be specified"))?;

        let factors = self
            .factors
            .ok_or_else(|| Error::invalid_params("factors must be specified"))?;

        let strength = self.strength.unwrap_or(2);
        let min_runs = self.min_runs.unwrap_or(0);

        // Validate basic constraints
        if levels_vec.is_empty() {
            return Err(Error::invalid_params("levels must not be empty"));
        }
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }
        if strength == 0 {
            return Err(Error::invalid_params("strength must be at least 1"));
        }
        if strength as usize > factors {
            return Err(Error::invalid_params(
                "strength cannot exceed number of factors",
            ));
        }

        // Handle symmetric case
        let is_symmetric = levels_vec.len() == 1 || levels_vec.iter().all(|&s| s == levels_vec[0]);
        if is_symmetric {
            let levels = levels_vec[0];
            return self.auto_select(levels, factors, strength, min_runs);
        }

        // Handle mixed-level case by finding a suitable base symmetric OA
        // For now, we look for a prime power q that is a multiple of all s_i
        let max_s = *levels_vec.iter().max().unwrap();
        
        // Try prime powers q starting from max_s up to a reasonable limit
        for q in max_s..=256 {
            if crate::utils::is_prime_power(q) && levels_vec.iter().all(|&s| q % s == 0) {
                // Try to build symmetric OA(N, q^factors, strength)
                if let Ok(base_oa) = self.auto_select(q, factors, strength, min_runs) {
                    // Collapse columns to desired levels
                    let mut mixed_oa = base_oa;
                    for (i, &s) in levels_vec.iter().enumerate() {
                        if s < q {
                            mixed_oa = mixed_oa.collapse_levels(i, s)?;
                        }
                    }
                    return Ok(mixed_oa);
                }
            }
        }

        // No suitable construction found
        Err(Error::invalid_params(format!(
            "No construction available for mixed-level OA with levels {:?}. \
             Try different parameters or a smaller number of factors.",
            levels_vec
        )))
    }

    /// Automatically select the best construction for the given parameters.
    fn auto_select(
        &self,
        levels: u32,
        factors: usize,
        strength: u32,
        min_runs: usize,
    ) -> Result<OA> {
        // Check if levels is a prime power
        let prime_power = factor_prime_power(levels);

        // For binary factors, Hadamard-Sylvester is often best
        if levels == 2 && strength == 2 {
            // Find smallest power of 2 >= max(factors+1, min_runs)
            let mut n = 4;
            while n - 1 < factors || n < min_runs {
                n *= 2;
                if n > 1 << 20 {
                    break; // Avoid extremely large arrays
                }
            }

            if n - 1 >= factors && n >= min_runs {
                if let Ok(h) = HadamardSylvester::new(n) {
                    if let Ok(oa) = h.construct(factors) {
                        return Ok(oa);
                    }
                }
            }
        }

        // For strength 2 with prime power levels, try Bose-Bush then Bose
        if strength == 2 {
            // Try Bose-Bush for q=2 (gives more factors than Bose)
            if levels == 2 {
                let bb_runs = 8; // 2 * 2²
                let bb_max_factors = 5; // 2*2 + 1

                if factors <= bb_max_factors && bb_runs >= min_runs {
                    if let Ok(bb) = BoseBush::new(2) {
                        if let Ok(oa) = bb.construct(factors) {
                            return Ok(oa);
                        }
                    }
                }
            }

            // Try Bose for any prime power
            if let Some(ref _pf) = prime_power {
                let q = levels;
                let bose_max_factors = (q + 1) as usize;
                let bose_runs = (q * q) as usize;

                if factors <= bose_max_factors && bose_runs >= min_runs {
                    let bose = Bose::new(q);
                    if let Ok(oa) = bose.construct(factors) {
                        return Ok(oa);
                    }
                }
            }

            // Try Addelman-Kempthorne for odd prime powers (gives 2q+1 factors)
            if let Some(ref pf) = prime_power {
                if pf.prime != 2 {
                    // Odd prime power - can use Addelman-Kempthorne
                    let q = levels;
                    let ak_max_factors = (2 * q + 1) as usize;
                    let ak_runs = (2 * q * q) as usize;

                    if factors <= ak_max_factors && ak_runs >= min_runs {
                        if let Ok(ak) = AddelmanKempthorne::new(q) {
                            if let Ok(oa) = ak.construct(factors) {
                                return Ok(oa);
                            }
                        }
                    }
                }
            }

            // Try Rao-Hamming for any prime power (supports larger N = q^m)
            if let Some(ref _pf) = prime_power {
                let q = levels;
                // Try m = 2, 3, 4, ... until we find enough factors or runs
                for m in 2..=10 {
                    let rh_runs = (q as usize).pow(m);
                    let rh_max_factors = (rh_runs - 1) / (q as usize - 1);

                    if factors <= rh_max_factors && rh_runs >= min_runs {
                        if let Ok(rh) = RaoHamming::new(q, m) {
                            if let Ok(oa) = rh.construct(factors) {
                                return Ok(oa);
                            }
                        }
                    }

                    if rh_runs > 1024 && rh_runs > min_runs {
                        break;
                    }
                }
            }
        }

        // For higher strength, try Bush
        if let Some(ref _pf) = prime_power {
            let q = levels;
            let t = strength;
            let bush_max_factors = (t + 1) as usize;
            let bush_runs = q.pow(t) as usize;

            if factors <= bush_max_factors && bush_runs >= min_runs {
                if let Ok(bush) = Bush::new(q, t) {
                    if let Ok(oa) = bush.construct(factors) {
                        return Ok(oa);
                    }
                }
            }
        }

        // No suitable construction found
        Err(Error::invalid_params(format!(
            "No construction available for OA(?, {}, {}, {}). \
             Try different parameters or a smaller number of factors.",
            factors, levels, strength
        )))
    }
}

/// Convenience function to build an orthogonal array.
///
/// This is a shorthand for using the builder.
///
/// # Example
///
/// ```
/// use taguchi::build_oa;
///
/// let oa = build_oa(3, 4, 2).unwrap();
/// assert_eq!(oa.levels(), 3);
/// assert_eq!(oa.factors(), 4);
/// assert_eq!(oa.runs(), 9);  // Bose: 3²
/// ```
pub fn build_oa(levels: u32, factors: usize, strength: u32) -> Result<OA> {
    OABuilder::new()
        .levels(levels)
        .factors(factors)
        .strength(strength)
        .build()
}

/// Get information about available constructions for given parameters.
///
/// Returns a list of construction options with their runs and max factors.
///
/// # Example
///
/// ```
/// use taguchi::available_constructions;
///
/// let options = available_constructions(3, 2);
/// for (name, runs, max_factors) in options {
///     println!("{}: {} runs, up to {} factors", name, runs, max_factors);
/// }
/// ```
pub fn available_constructions(levels: u32, strength: u32) -> Vec<(&'static str, usize, usize)> {
    let mut options = Vec::new();

    let prime_power = factor_prime_power(levels);

    // Hadamard-Sylvester (binary only, powers of 2)
    if levels == 2 && strength == 2 {
        for m in 2..=10 {
            let n = 1 << m;
            options.push(("HadamardSylvester", n, n - 1));
        }
    }

    // Hadamard-Paley (binary only, p+1 where p ≡ 3 (mod 4))
    if levels == 2 && strength == 2 {
        // Common Paley primes: 3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83
        for &p in &[3u32, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83] {
            options.push(("HadamardPaley", (p + 1) as usize, p as usize));
        }
    }

    // Bose-Bush (q=2 only currently)
    if levels == 2 && strength == 2 {
        options.push(("BoseBush", 8, 5));
    }

    // Bose
    if strength == 2 {
        if let Some(ref _pf) = prime_power {
            let q = levels;
            options.push(("Bose", (q * q) as usize, (q + 1) as usize));
        }
    }

    // Addelman-Kempthorne (odd prime powers only)
    if strength == 2 {
        if let Some(ref pf) = prime_power {
            if pf.prime != 2 {
                let q = levels;
                options.push(("AddelmanKempthorne", (2 * q * q) as usize, (2 * q + 1) as usize));
            }
        }
    }

    // Rao-Hamming
    if strength == 2 {
        if let Some(ref _pf) = prime_power {
            let q = levels;
            for m in 2..=5 {
                let n = (q as usize).pow(m);
                let k = (n - 1) / (q as usize - 1);
                options.push(("RaoHamming", n, k));
            }
        }
    }

    // Bush
    if let Some(ref _pf) = prime_power {
        let q = levels;
        let t = strength;
        options.push(("Bush", q.pow(t) as usize, (t + 1) as usize));
    }

    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_builder_basic() {
        let oa = OABuilder::new()
            .levels(3)
            .factors(4)
            .strength(2)
            .build()
            .unwrap();

        assert_eq!(oa.levels(), 3);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.runs(), 9); // Bose: 3²

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_builder_binary_many_factors() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(7)
            .strength(2)
            .build()
            .unwrap();

        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.factors(), 7);
        // Should use Hadamard-Sylvester: OA(8, 7, 2, 2)
        assert_eq!(oa.runs(), 8);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_builder_binary_few_factors() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(3)
            .strength(2)
            .build()
            .unwrap();

        assert_eq!(oa.levels(), 2);
        assert_eq!(oa.factors(), 3);
        // Could use Hadamard(4), Bose(4), or BoseBush(8)
        // Hadamard(4) gives fewest runs

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_builder_strength_default() {
        let oa = OABuilder::new().levels(3).factors(3).build().unwrap();

        // Default strength is 2
        assert_eq!(oa.strength(), 2);
    }

    #[test]
    fn test_builder_min_runs() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(3)
            .strength(2)
            .min_runs(16)
            .build()
            .unwrap();

        assert!(oa.runs() >= 16);
    }

    #[test]
    fn test_builder_missing_levels() {
        let result = OABuilder::new().factors(4).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_factors() {
        let result = OABuilder::new().levels(3).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_strength() {
        let result = OABuilder::new()
            .levels(3)
            .factors(2)
            .strength(3) // strength > factors
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_build_oa_convenience() {
        let oa = build_oa(5, 4, 2).unwrap();

        assert_eq!(oa.levels(), 5);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.runs(), 25); // Bose: 5²

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_builder_high_strength() {
        let oa = OABuilder::new()
            .levels(3)
            .factors(4)
            .strength(3)
            .build()
            .unwrap();

        assert_eq!(oa.levels(), 3);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.strength(), 3);
        assert_eq!(oa.runs(), 27); // Bush: 3³

        let result = verify_strength(&oa, 3).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_builder_various_prime_powers() {
        for q in [2, 3, 4, 5, 7, 8, 9] {
            let oa = OABuilder::new()
                .levels(q)
                .factors(3)
                .strength(2)
                .build()
                .unwrap();

            assert_eq!(oa.levels(), q);

            let result = verify_strength(&oa, 2).unwrap();
            assert!(result.is_valid, "OA with {} levels should be valid", q);
        }
    }

    #[test]
    fn test_builder_large_binary() {
        let oa = OABuilder::new()
            .levels(2)
            .factors(15)
            .strength(2)
            .build()
            .unwrap();

        assert_eq!(oa.symmetric_levels(), 2);
        assert_eq!(oa.factors(), 15);
        assert_eq!(oa.runs(), 16); // Hadamard-Sylvester

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_builder_mixed_levels() {
        // Construct OA(16, 2^3 4^1, 2)
        // Base will be OA(16, 4^4, 2)
        let oa = OABuilder::new()
            .mixed_levels(vec![2, 2, 2, 4])
            .strength(2)
            .build()
            .unwrap();

        assert_eq!(oa.runs(), 16);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels_vec(), &[2, 2, 2, 4]);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "Mixed OA should be valid: {:?}", result.issues);
    }

    #[test]
    fn test_available_constructions() {
        let options = available_constructions(3, 2);

        // Should include Bose, Bush, and RaoHamming
        let names: Vec<_> = options.iter().map(|(name, _, _)| *name).collect();
        assert!(names.contains(&"Bose"));
        assert!(names.contains(&"Bush"));
        assert!(names.contains(&"RaoHamming"));
    }

    #[test]
    fn test_available_constructions_binary() {
        let options = available_constructions(2, 2);

        let names: Vec<_> = options.iter().map(|(name, _, _)| *name).collect();
        assert!(names.contains(&"HadamardSylvester"));
        assert!(names.contains(&"BoseBush"));
        assert!(names.contains(&"Bose"));
        assert!(names.contains(&"RaoHamming"));
    }
}

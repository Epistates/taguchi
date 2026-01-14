//! Orthogonal array core types and operations.
//!
//! This module provides the fundamental data structures for representing
//! and manipulating orthogonal arrays.
//!
//! ## Overview
//!
//! - [`OA`]: The main orthogonal array type
//! - [`OAParams`]: Parameters describing an orthogonal array
//!
//! ## Notation
//!
//! An orthogonal array OA(N, k, s, t) has:
//! - N rows (runs/experiments)
//! - k columns (factors/parameters)
//! - s levels (symbols 0, 1, ..., s-1)
//! - strength t (balance property)
//!
//! The defining property is that every N×t subarray contains each possible
//! t-tuple exactly λ = N/s^t times.

mod stats;
mod verify;

pub use stats::BalanceReport;
pub use verify::{compute_strength, verify_strength, VerificationResult};

use ndarray::Array2;
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Parameters describing an orthogonal array.
///
/// This struct encapsulates the mathematical parameters of an OA and
/// provides validation to ensure consistency.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OAParams {
    /// Number of runs (rows).
    pub runs: usize,
    /// Number of factors (columns).
    pub factors: usize,
    /// Number of levels for each factor.
    pub levels: Vec<u32>,
    /// Strength (orthogonality degree).
    pub strength: u32,
}

impl OAParams {
    /// Create new symmetric OA parameters with automatic validation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - N is not divisible by s^t
    /// - strength exceeds factors
    /// - levels is 0 or 1
    pub fn new(runs: usize, factors: usize, levels: u32, strength: u32) -> Result<Self> {
        if levels < 2 {
            return Err(Error::invalid_params("levels must be at least 2"));
        }

        if strength as usize > factors {
            return Err(Error::invalid_params(format!(
                "strength {} cannot exceed factors {}",
                strength, factors
            )));
        }

        let s_to_t = (levels as usize)
            .checked_pow(strength)
            .ok_or_else(|| Error::invalid_params(format!("{}^{} overflows", levels, strength)))?;

        if runs % s_to_t != 0 {
            return Err(Error::invalid_params(format!(
                "runs {} must be divisible by levels^strength = {}^{} = {}",
                runs, levels, strength, s_to_t
            )));
        }

        Ok(Self {
            runs,
            factors,
            levels: vec![levels; factors],
            strength,
        })
    }

    /// Create new mixed-level OA parameters.
    pub fn new_mixed(runs: usize, levels: Vec<u32>, strength: u32) -> Result<Self> {
        let factors = levels.len();
        if factors == 0 {
            return Err(Error::invalid_params("factors must be at least 1"));
        }

        for (i, &s) in levels.iter().enumerate() {
            if s < 2 {
                return Err(Error::invalid_params(format!(
                    "levels for factor {} must be at least 2, got {}",
                    i, s
                )));
            }
        }

        if strength as usize > factors {
            return Err(Error::invalid_params(format!(
                "strength {} cannot exceed factors {}",
                strength, factors
            )));
        }

        Ok(Self {
            runs,
            factors,
            levels,
            strength,
        })
    }

    /// Check if the array is symmetric (all factors have same number of levels).
    #[must_use]
    pub fn is_symmetric(&self) -> bool {
        if self.levels.is_empty() {
            return true;
        }
        let s0 = self.levels[0];
        self.levels.iter().all(|&s| s == s0)
    }

    /// Get the common number of levels if symmetric.
    ///
    /// # Panics
    ///
    /// Panics if the array is not symmetric or has no factors.
    #[must_use]
    pub fn symmetric_levels(&self) -> u32 {
        assert!(self.is_symmetric(), "OA is not symmetric");
        self.levels[0]
    }

    /// Get the index (lambda) for symmetric OAs.
    ///
    /// For mixed-level OAs, this is not a single value.
    #[must_use]
    pub fn index(&self) -> usize {
        if !self.is_symmetric() || self.levels.is_empty() {
            return 0; // Not well-defined for mixed-level
        }
        let s = self.levels[0] as usize;
        let s_to_t = s.pow(self.strength);
        self.runs / s_to_t
    }

    /// Validate that these parameters are internally consistent.
    pub fn validate(&self) -> Result<()> {
        if self.factors != self.levels.len() {
            return Err(Error::invalid_params("factors must match levels length"));
        }

        if self.strength as usize > self.factors {
            return Err(Error::invalid_params("strength cannot exceed factors"));
        }

        // For mixed level, divisibility check is more complex.
        // For any t columns, runs must be divisible by product of their levels.
        // We only check a few small combinations if k is large, or all if k is small.
        // For now, let's at least check that runs is divisible by each individual level.
        for (i, &s) in self.levels.iter().enumerate() {
            if s < 2 {
                return Err(Error::invalid_params(format!(
                    "level for factor {} is < 2",
                    i
                )));
            }
            if self.runs % (s as usize) != 0 {
                return Err(Error::invalid_params(format!(
                    "runs {} must be divisible by level {} of factor {}",
                    self.runs, s, i
                )));
            }
        }

        Ok(())
    }
}

impl fmt::Display for OAParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_symmetric() {
            write!(
                f,
                "OA({}, {}, {}, {})",
                self.runs, self.factors, self.levels[0], self.strength
            )
        } else {
            let mut levels_map: HashMap<u32, usize> = HashMap::new();
            for &s in &self.levels {
                *levels_map.entry(s).or_insert(0) += 1;
            }
            let mut levels_sorted: Vec<_> = levels_map.into_iter().collect();
            levels_sorted.sort_by_key(|&(s, _)| s);

            let levels_str: Vec<String> = levels_sorted
                .into_iter()
                .map(|(s, k)| {
                    if k == 1 {
                        s.to_string()
                    } else {
                        format!("{}^{}", s, k)
                    }
                })
                .collect();

            write!(
                f,
                "OA({}, {}, {}, {})",
                self.runs,
                levels_str.join(" "),
                self.strength,
                self.runs // Placeholder for index? No, just omit it for mixed
            )
        }
    }
}

/// An orthogonal array.
///
/// This is the main data structure representing an orthogonal array.
/// The array data is stored as a 2D matrix of integers where each
/// column j has elements in the range [0, levels[j]).
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OA {
    /// The array data, shape (runs, factors).
    data: Array2<u32>,
    /// Array parameters.
    params: OAParams,
}

impl OA {
    /// Create a new orthogonal array from data and parameters.
    ///
    /// # Panics
    ///
    /// Panics if the data dimensions don't match the parameters.
    #[must_use]
    pub fn new(data: Array2<u32>, params: OAParams) -> Self {
        assert_eq!(
            data.nrows(),
            params.runs,
            "data rows {} must match params.runs {}",
            data.nrows(),
            params.runs
        );
        assert_eq!(
            data.ncols(),
            params.factors,
            "data cols {} must match params.factors {}",
            data.ncols(),
            params.factors
        );

        Self { data, params }
    }

    /// Create a new orthogonal array, validating data dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the data dimensions don't match the parameters.
    pub fn try_new(data: Array2<u32>, params: OAParams) -> Result<Self> {
        if data.nrows() != params.runs {
            return Err(Error::DimensionMismatch {
                expected: format!("{} rows", params.runs),
                actual: format!("{} rows", data.nrows()),
            });
        }
        if data.ncols() != params.factors {
            return Err(Error::DimensionMismatch {
                expected: format!("{} columns", params.factors),
                actual: format!("{} columns", data.ncols()),
            });
        }

        Ok(Self { data, params })
    }

    /// Get the number of runs (rows).
    #[must_use]
    pub fn runs(&self) -> usize {
        self.params.runs
    }

    /// Get the number of factors (columns).
    #[must_use]
    pub fn factors(&self) -> usize {
        self.params.factors
    }

    /// Get the common number of levels for a symmetric OA.
    ///
    /// # Panics
    ///
    /// Panics if the OA is mixed-level and has no factors.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.params.levels[0]
    }

    /// Get the levels for all factors.
    #[must_use]
    pub fn levels_vec(&self) -> &[u32] {
        &self.params.levels
    }

    /// Get the number of levels for a specific factor.
    #[must_use]
    pub fn levels_for(&self, factor: usize) -> u32 {
        self.params.levels[factor]
    }

    /// Get the common number of levels for a symmetric OA.
    ///
    /// # Panics
    ///
    /// Panics if the OA is mixed-level.
    #[must_use]
    pub fn symmetric_levels(&self) -> u32 {
        self.params.symmetric_levels()
    }

    /// Get the strength.
    #[must_use]
    pub fn strength(&self) -> u32 {
        self.params.strength
    }

    /// Get the index (lambda) if symmetric.
    #[must_use]
    pub fn index(&self) -> usize {
        self.params.index()
    }

    /// Get the parameters.
    #[must_use]
    pub fn params(&self) -> &OAParams {
        &self.params
    }

    /// Get a reference to the underlying data.
    #[must_use]
    pub fn data(&self) -> &Array2<u32> {
        &self.data
    }

    /// Get a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut Array2<u32> {
        &mut self.data
    }

    /// Consume the OA and return the underlying data.
    #[must_use]
    pub fn into_data(self) -> Array2<u32> {
        self.data
    }

    /// Get the value at a specific position.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> u32 {
        self.data[[row, col]]
    }

    /// Get a row of the array as a slice.
    #[must_use]
    pub fn row(&self, idx: usize) -> ndarray::ArrayView1<'_, u32> {
        self.data.row(idx)
    }

    /// Get a column of the array as a slice.
    #[must_use]
    pub fn column(&self, idx: usize) -> ndarray::ArrayView1<'_, u32> {
        self.data.column(idx)
    }

    /// Iterate over rows.
    pub fn rows(&self) -> impl Iterator<Item = ndarray::ArrayView1<'_, u32>> {
        self.data.rows().into_iter()
    }

    /// Select a subset of columns, returning a new OA.
    ///
    /// # Panics
    ///
    /// Panics if any column index is out of bounds.
    #[must_use]
    pub fn select_columns(&self, cols: &[usize]) -> Self {
        let new_factors = cols.len();
        let mut new_data = Array2::zeros((self.runs(), new_factors));
        let mut new_levels = Vec::with_capacity(new_factors);

        for (new_col, &old_col) in cols.iter().enumerate() {
            new_levels.push(self.params.levels[old_col]);
            for row in 0..self.runs() {
                new_data[[row, new_col]] = self.data[[row, old_col]];
            }
        }

        let new_params = OAParams {
            runs: self.runs(),
            factors: new_factors,
            levels: new_levels,
            strength: self.strength().min(new_factors as u32),
        };

        Self::new(new_data, new_params)
    }

    /// Verify that this array has the claimed strength.
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails or finds issues.
    pub fn verify(&self) -> Result<VerificationResult> {
        verify_strength(self, self.strength())
    }

    /// Check if all values are in the valid range [0, levels).
    #[must_use]
    pub fn values_in_range(&self) -> bool {
        for col in 0..self.factors() {
            let s = self.params.levels[col];
            for row in 0..self.runs() {
                if self.data[[row, col]] >= s {
                    return false;
                }
            }
        }
        true
    }

    /// Collapse the levels of a specific factor.
    ///
    /// This reduces the number of levels for a factor by mapping the original
    /// levels `v` to `v % new_levels`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `new_levels` is less than 2
    /// - `new_levels` does not divide the original number of levels
    ///
    /// # Example
    ///
    /// ```
    /// # use taguchi::OABuilder;
    /// let oa = OABuilder::new().levels(4).factors(3).build().unwrap();
    /// let mixed = oa.collapse_levels(0, 2).unwrap();
    /// assert_eq!(mixed.levels_for(0), 2);
    /// assert_eq!(mixed.levels_for(1), 4);
    /// ```
    pub fn collapse_levels(&self, factor: usize, new_levels: u32) -> Result<Self> {
        if factor >= self.factors() {
            return Err(Error::IndexOutOfBounds {
                index: factor,
                size: self.factors(),
            });
        }

        let old_levels = self.params.levels[factor];
        if new_levels < 2 {
            return Err(Error::invalid_params("new levels must be at least 2"));
        }

        if old_levels % new_levels != 0 {
            return Err(Error::invalid_params(format!(
                "new levels {} must divide old levels {}",
                new_levels, old_levels
            )));
        }

        let mut new_data = self.data.clone();
        for row in 0..self.runs() {
            new_data[[row, factor]] %= new_levels;
        }

        let mut new_levels_vec = self.params.levels.clone();
        new_levels_vec[factor] = new_levels;

        let new_params = OAParams::new_mixed(self.runs(), new_levels_vec, self.strength())?;

        Ok(Self::new(new_data, new_params))
    }
}

impl fmt::Debug for OA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} with data {:?}", self.params, self.data)
    }
}

impl fmt::Display for OA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.params)?;
        for row in self.data.rows() {
            let row_str: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            writeln!(f, "  {}", row_str.join(" "))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_new() {
        let params = OAParams::new(9, 4, 3, 2).unwrap();
        assert_eq!(params.runs, 9);
        assert_eq!(params.factors, 4);
        assert_eq!(params.levels[0], 3);
        assert_eq!(params.strength, 2);
        assert_eq!(params.index(), 1);
    }

    #[test]
    fn test_params_invalid() {
        // levels < 2
        assert!(OAParams::new(9, 4, 1, 2).is_err());

        // strength > factors
        assert!(OAParams::new(9, 2, 3, 3).is_err());

        // runs not divisible by s^t
        assert!(OAParams::new(10, 4, 3, 2).is_err());
    }

    #[test]
    fn test_oa_creation() {
        let params = OAParams::new(4, 3, 2, 2).unwrap();
        let data =
            Array2::from_shape_vec((4, 3), vec![0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]).unwrap();

        let oa = OA::new(data, params);
        assert_eq!(oa.runs(), 4);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.symmetric_levels(), 2);
    }

    #[test]
    fn test_select_columns() {
        let params = OAParams::new(4, 4, 2, 2).unwrap();
        let data =
            Array2::from_shape_vec((4, 4), vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0])
                .unwrap();

        let oa = OA::new(data, params);
        let sub = oa.select_columns(&[0, 2]);

        assert_eq!(sub.factors(), 2);
        assert_eq!(sub.get(0, 0), 0);
        assert_eq!(sub.get(0, 1), 0);
        assert_eq!(sub.get(2, 0), 1);
        assert_eq!(sub.get(2, 1), 1);
    }

    #[test]
    fn test_display() {
        let params = OAParams::new(4, 3, 2, 2).unwrap();
        assert_eq!(format!("{}", params), "OA(4, 3, 2, 2)");
    }
}

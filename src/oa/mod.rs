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

mod verify;

pub use verify::{verify_strength, VerificationResult};

use ndarray::Array2;
use std::fmt;

use crate::error::{Error, Result};

/// Parameters describing an orthogonal array.
///
/// This struct encapsulates the mathematical parameters of an OA and
/// provides validation to ensure consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OAParams {
    /// Number of runs (rows).
    pub runs: usize,
    /// Number of factors (columns).
    pub factors: usize,
    /// Number of levels (symbols 0..levels-1).
    pub levels: u32,
    /// Strength (orthogonality degree).
    pub strength: u32,
    /// Index (lambda) - number of times each t-tuple appears.
    pub index: usize,
}

impl OAParams {
    /// Create new OA parameters with automatic index calculation.
    ///
    /// The index λ is computed as N / s^t.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - N is not divisible by s^t
    /// - strength exceeds factors
    /// - levels is 0 or 1
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::oa::OAParams;
    ///
    /// let params = OAParams::new(9, 4, 3, 2).unwrap();
    /// assert_eq!(params.runs, 9);
    /// assert_eq!(params.factors, 4);
    /// assert_eq!(params.levels, 3);
    /// assert_eq!(params.strength, 2);
    /// assert_eq!(params.index, 1);  // 9 / 3^2 = 1
    /// ```
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

        let s_to_t = (levels as usize).checked_pow(strength).ok_or_else(|| {
            Error::invalid_params(format!("{}^{} overflows", levels, strength))
        })?;

        if runs % s_to_t != 0 {
            return Err(Error::invalid_params(format!(
                "runs {} must be divisible by levels^strength = {}^{} = {}",
                runs, levels, strength, s_to_t
            )));
        }

        let index = runs / s_to_t;

        Ok(Self {
            runs,
            factors,
            levels,
            strength,
            index,
        })
    }

    /// Create parameters with explicit index.
    ///
    /// This allows creating parameters without validation, useful for
    /// testing or when the parameters are known to be valid.
    #[must_use]
    pub fn with_index(runs: usize, factors: usize, levels: u32, strength: u32, index: usize) -> Self {
        Self {
            runs,
            factors,
            levels,
            strength,
            index,
        }
    }

    /// Validate that these parameters are internally consistent.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameters are inconsistent.
    pub fn validate(&self) -> Result<()> {
        if self.levels < 2 {
            return Err(Error::invalid_params("levels must be at least 2"));
        }

        if self.strength as usize > self.factors {
            return Err(Error::invalid_params(format!(
                "strength {} cannot exceed factors {}",
                self.strength, self.factors
            )));
        }

        let s_to_t = (self.levels as usize).checked_pow(self.strength).ok_or_else(|| {
            Error::invalid_params(format!("{}^{} overflows", self.levels, self.strength))
        })?;

        let expected_index = self.runs / s_to_t;
        if self.index != expected_index {
            return Err(Error::invalid_params(format!(
                "index {} does not match expected {} (runs/levels^strength)",
                self.index, expected_index
            )));
        }

        if self.runs != self.index * s_to_t {
            return Err(Error::invalid_params(format!(
                "runs {} must equal index * levels^strength = {} * {} = {}",
                self.runs,
                self.index,
                s_to_t,
                self.index * s_to_t
            )));
        }

        Ok(())
    }
}

impl fmt::Display for OAParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OA({}, {}, {}, {})",
            self.runs, self.factors, self.levels, self.strength
        )
    }
}

/// An orthogonal array.
///
/// This is the main data structure representing an orthogonal array.
/// The array data is stored as a 2D matrix of integers where each
/// element is in the range [0, levels).
///
/// # Example
///
/// ```
/// use taguchi::oa::{OA, OAParams};
/// use ndarray::Array2;
///
/// let params = OAParams::new(9, 4, 3, 2).unwrap();
/// let data = Array2::zeros((9, 4));
/// let oa = OA::new(data, params);
///
/// assert_eq!(oa.runs(), 9);
/// assert_eq!(oa.factors(), 4);
/// ```
#[derive(Clone)]
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

    /// Get the number of levels.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.params.levels
    }

    /// Get the strength.
    #[must_use]
    pub fn strength(&self) -> u32 {
        self.params.strength
    }

    /// Get the index (lambda).
    #[must_use]
    pub fn index(&self) -> usize {
        self.params.index
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
    pub fn row(&self, idx: usize) -> ndarray::ArrayView1<u32> {
        self.data.row(idx)
    }

    /// Get a column of the array as a slice.
    #[must_use]
    pub fn column(&self, idx: usize) -> ndarray::ArrayView1<u32> {
        self.data.column(idx)
    }

    /// Iterate over rows.
    pub fn rows(&self) -> impl Iterator<Item = ndarray::ArrayView1<u32>> {
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

        for (new_col, &old_col) in cols.iter().enumerate() {
            for row in 0..self.runs() {
                new_data[[row, new_col]] = self.data[[row, old_col]];
            }
        }

        let new_params = OAParams::with_index(
            self.runs(),
            new_factors,
            self.levels(),
            self.strength().min(new_factors as u32),
            self.index(),
        );

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
        self.data.iter().all(|&v| v < self.params.levels)
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
        assert_eq!(params.levels, 3);
        assert_eq!(params.strength, 2);
        assert_eq!(params.index, 1);
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
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        )
        .unwrap();

        let oa = OA::new(data, params);
        assert_eq!(oa.runs(), 4);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.levels(), 2);
    }

    #[test]
    fn test_select_columns() {
        let params = OAParams::new(4, 4, 2, 2).unwrap();
        let data = Array2::from_shape_vec(
            (4, 4),
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        )
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

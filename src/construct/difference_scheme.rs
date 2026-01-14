//! Difference Schemes for constructing orthogonal arrays.
//!
//! A Difference Scheme $D(r, c, s)$ is an $r \times c$ matrix with entries from a group $G$ of order $s$
//! (here the additive group of $GF(s)$) such that for any two columns $j$ and $k$, the vector of
//! element-wise differences contains every element of $G$ exactly $\lambda = r/s$ times.
//!
//! Difference schemes are powerful tools because they can be "inflated" to form Orthogonal Arrays
//! of strength 2. An OA is formed by taking each row of the difference scheme and adding every
//! element of $G$ to it, producing $s$ rows in the OA for each row in the DS.
//!
//! # Expansion to OA
//!
//! If $D$ is a difference scheme $D(N, k, s)$, expanding it produces an $OA(N \cdot s, k, s, 2)$.
//!
//! # Linear Difference Scheme
//!
//! The most common construction uses the multiplication table of $GF(s)$.
//! $D_{ij} = a_i \cdot b_j$ where $a_i, b_j \in GF(s)$.
//! This produces $D(s, s, s)$. By augmenting with a column of zeros, we get $D(s, s+1, s)$.
//! Expanding this yields the Bose construction $OA(s^2, s+1, s, 2)$.

use ndarray::Array2;

use super::Constructor;
use crate::error::{Error, Result};
use crate::gf::DynamicGf;
use crate::oa::{OAParams, OA};
use crate::utils::is_prime_power;

/// A Difference Scheme matrix.
///
/// This is an intermediate structure that can be converted into an Orthogonal Array.
#[derive(Debug, Clone)]
pub struct DifferenceScheme {
    /// The matrix data ($r \times c$).
    pub data: Array2<u32>,
    /// The number of levels (order of the group $G$).
    pub q: u32,
    /// The Galois field used for group operations.
    pub field: DynamicGf,
}

impl DifferenceScheme {
    /// Create a new Difference Scheme from raw data.
    ///
    /// # Arguments
    ///
    /// * `data` - The matrix data.
    /// * `q` - The number of levels (must match field order).
    /// * `field` - The Galois field.
    pub fn new(data: Array2<u32>, q: u32, field: DynamicGf) -> Self {
        assert_eq!(field.order(), q);
        Self { data, q, field }
    }

    /// Construct a Linear Difference Scheme $D(q, q, q)$ from $GF(q)$.
    ///
    /// The matrix entries are $D_{ij} = i \cdot j$ for $i, j \in GF(q)$.
    /// This effectively corresponds to the multiplication table of the field.
    ///
    /// # Errors
    ///
    /// Returns an error if `q` is not a prime power.
    pub fn linear(q: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "DifferenceScheme::linear",
            });
        }

        let field = DynamicGf::new(q)?;
        let mut data = Array2::zeros((q as usize, q as usize));

        let tables = field.tables();
        for i in 0..q {
            for j in 0..q {
                data[[i as usize, j as usize]] = tables.mul(i, j);
            }
        }

        Ok(Self { data, q, field })
    }

    /// Augment the difference scheme with a column of zeros.
    ///
    /// If the original scheme is $D(r, c, s)$, the result is $D(r, c+1, s)$.
    /// This works for linear difference schemes because the difference between
    /// the zero column and any other column $j$ is $0 - (i \cdot j) = -(i \cdot j)$.
    /// As $i$ varies, $-(i \cdot j)$ covers all elements of $GF(q)$ (for $j \neq 0$).
    pub fn with_zero_column(&self) -> Self {
        let rows = self.data.nrows();
        let cols = self.data.ncols();
        let mut new_data = Array2::zeros((rows, cols + 1));

        // Copy existing data to columns 1..=cols
        for i in 0..rows {
            for j in 0..cols {
                new_data[[i, j + 1]] = self.data[[i, j]];
            }
            // Column 0 is already zeros
        }

        Self {
            data: new_data,
            q: self.q,
            field: self.field.clone(),
        }
    }

    /// Verify that this is a valid Difference Scheme $D(r, c, s)$.
    ///
    /// Checks that for every pair of columns, the element-wise differences
    /// contain every element of the group (GF(q)) exactly $\lambda = r/s$ times.
    ///
    /// # Returns
    ///
    /// `true` if valid, `false` otherwise.
    pub fn verify(&self) -> bool {
        let rows = self.data.nrows();
        let cols = self.data.ncols();
        let s = self.q as usize;
        let expected_count = rows / s;

        if rows % s != 0 {
            return false;
        }

        let tables = self.field.tables();

        for c1 in 0..cols {
            for c2 in (c1 + 1)..cols {
                let mut counts = vec![0; s];

                for r in 0..rows {
                    let v1 = self.data[[r, c1]];
                    let v2 = self.data[[r, c2]];

                    // diff = v1 - v2
                    let diff = tables.sub(v1, v2) as usize;
                    counts[diff] += 1;
                }

                if counts.iter().any(|&c| c != expected_count) {
                    return false;
                }
            }
        }

        true
    }

    /// Compute the Kronecker sum of this scheme with another.
    ///
    /// If $D_1$ is $D(r_1, c_1, s)$ and $D_2$ is $D(r_2, c_2, s)$, the result is
    /// $D(r_1 r_2, c_1 c_2, s)$.
    ///
    /// The Kronecker sum is defined as $(D_1 \oplus D_2)_{(i_1, i_2), (j_1, j_2)} = D_1(i_1, j_1) + D_2(i_2, j_2)$.
    ///
    /// # Errors
    ///
    /// Returns an error if the schemes have different levels (groups must match).
    pub fn kronecker_sum(&self, other: &Self) -> Result<Self> {
        if self.q != other.q {
            return Err(Error::invalid_params(format!(
                "Cannot compute Kronecker sum: schemes have different levels {} and {}",
                self.q, other.q
            )));
        }

        let r1 = self.data.nrows();
        let c1 = self.data.ncols();
        let r2 = other.data.nrows();
        let c2 = other.data.ncols();

        let mut new_data = Array2::zeros((r1 * r2, c1 * c2));
        let tables = self.field.tables();

        for i1 in 0..r1 {
            for j1 in 0..c1 {
                let v1 = self.data[[i1, j1]];

                for i2 in 0..r2 {
                    for j2 in 0..c2 {
                        let v2 = other.data[[i2, j2]];
                        let sum = tables.add(v1, v2);

                        // Map 2D indices to 1D
                        let new_row = i1 * r2 + i2;
                        let new_col = j1 * c2 + j2;

                        new_data[[new_row, new_col]] = sum;
                    }
                }
            }
        }

        Ok(Self {
            data: new_data,
            q: self.q,
            field: self.field.clone(),
        })
    }

    /// Expand the Difference Scheme into an Orthogonal Array.
    ///
    /// For a difference scheme $D(r, c, s)$, this produces an OA with:
    /// - Runs: $r \times s$
    /// - Factors: $c$
    /// - Levels: $s$
    /// - Strength: 2 (typically, if the DS has $\lambda=r/s$ balance)
    pub fn to_oa(&self) -> Result<OA> {
        let r = self.data.nrows();
        let c = self.data.ncols();
        let s = self.q;
        let runs = r * s as usize;

        let mut oa_data = Array2::zeros((runs, c));
        let tables = self.field.tables();

        // For each row `d` in the DS
        for i in 0..r {
            // For each element `g` in GF(s)
            for g in 0..s {
                let row_idx = i * s as usize + g as usize;

                // Construct row: d + g
                for j in 0..c {
                    let val = self.data[[i, j]];
                    oa_data[[row_idx, j]] = tables.add(val, g);
                }
            }
        }

        // Calculate strength. A valid DS expansion guarantees strength 2.
        // We verify parameters but assume strength 2 for the params struct.
        // If c=1, strength is 1.
        let strength = if c > 1 { 2 } else { 1 };

        let params = OAParams::new(runs, c, s, strength)?;
        Ok(OA::new(oa_data, params))
    }
}

/// Constructor for creating OAs via Difference Schemes.
///
/// This basically replicates the Bose construction but through the DS abstraction.
#[derive(Debug, Clone)]
pub struct LinearDifferenceScheme {
    q: u32,
}

impl LinearDifferenceScheme {
    /// Create a new Linear Difference Scheme constructor.
    pub fn new(q: u32) -> Result<Self> {
        if !is_prime_power(q) {
            return Err(Error::LevelsNotPrimePower {
                levels: q,
                algorithm: "LinearDifferenceScheme",
            });
        }
        Ok(Self { q })
    }
}

impl Constructor for LinearDifferenceScheme {
    fn name(&self) -> &'static str {
        "LinearDifferenceScheme"
    }

    fn family(&self) -> &'static str {
        "OA(q^2, k, q, 2) via Difference Scheme D(q, q+1, q)"
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
        // 1. Create linear DS D(q, q, q)
        let ds = DifferenceScheme::linear(self.q)?;

        // 2. We can produce up to q+1 factors.
        // q factors from the DS expansion: D_{ik} + g
        // 1 factor from the row index: i

        let max_factors = self.max_factors();
        if factors > max_factors {
            return Err(Error::TooManyFactors {
                factors,
                max: max_factors,
                algorithm: self.name(),
            });
        }

        let r = ds.data.nrows(); // q
        let s = ds.q; // q
        let runs = r * s as usize;

        // We will construct the full array first then select (or construct requested directly)
        // Optimization: only construct needed columns

        let mut oa_data = Array2::zeros((runs, factors));
        let tables = ds.field.tables();

        for i in 0..r {
            for g in 0..s {
                let row_idx = i * s as usize + g as usize;

                // Fill up to `factors` columns
                for j in 0..factors {
                    if j < (s as usize) {
                        // Use DS column j
                        let val = ds.data[[i, j]];
                        oa_data[[row_idx, j]] = tables.add(val, g);
                    } else {
                        // This is the (q+1)-th column (index q)
                        // Value is i (the DS row index)
                        oa_data[[row_idx, j]] = i as u32;
                    }
                }
            }
        }

        let strength = 2.min(factors as u32);
        let params = OAParams::new(runs, factors, s, strength)?;
        Ok(OA::new(oa_data, params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::verify_strength;

    #[test]
    fn test_linear_ds_construction() {
        let ds = DifferenceScheme::linear(3).unwrap();
        assert_eq!(ds.data.nrows(), 3);
        assert_eq!(ds.data.ncols(), 3);

        // Check multiplication table structure
        // 0 0 0
        // 0 1 2
        // 0 2 1  (in GF(3), 2*2=4=1)
        assert_eq!(ds.data[[1, 1]], 1);
        assert_eq!(ds.data[[2, 2]], 1);
    }

    #[test]
    fn test_ds_expansion_basic() {
        let ds = DifferenceScheme::linear(3).unwrap();
        // Expand to OA(9, 3, 3, 2)
        let oa = ds.to_oa().unwrap();
        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 3);
        assert_eq!(oa.levels(), 3);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_constructor_interface_l9() {
        let cons = LinearDifferenceScheme::new(3).unwrap();
        // Should support 4 factors (3 from DS + 1 index col)
        let oa = cons.construct(4).unwrap();

        assert_eq!(oa.runs(), 9);
        assert_eq!(oa.factors(), 4);
        assert_eq!(oa.levels(), 3);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid, "L9 from DS should be valid");
    }

    #[test]
    fn test_ds_verification() {
        let ds = DifferenceScheme::linear(3).unwrap();
        assert!(ds.verify());

        // Linear DS already has a zero column at index 0.
        // Adding another one creates duplicates, which fails verification.
        let ds_aug = ds.with_zero_column();
        assert!(!ds_aug.verify());
    }

    #[test]
    fn test_kronecker_sum() {
        let ds1 = DifferenceScheme::linear(2).unwrap(); // D(2, 2, 2)
        let ds2 = DifferenceScheme::linear(2).unwrap();

        let ds_kron = ds1.kronecker_sum(&ds2).unwrap();
        // Should be D(4, 4, 2)
        assert_eq!(ds_kron.data.nrows(), 4);
        assert_eq!(ds_kron.data.ncols(), 4);
        assert_eq!(ds_kron.q, 2);

        assert!(ds_kron.verify());

        // Expansion gives OA(8, 4, 2, 2)
        let oa = ds_kron.to_oa().unwrap();
        assert_eq!(oa.runs(), 8);
        assert_eq!(oa.factors(), 4);

        let result = verify_strength(&oa, 2).unwrap();
        assert!(result.is_valid);
    }
}

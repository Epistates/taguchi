//! Galois field (finite field) arithmetic.
//!
//! This module provides implementations of Galois fields GF(q) where q is a prime power.
//! Galois fields are fundamental to orthogonal array constructions as they provide the
//! algebraic structure for systematic level assignment.
//!
//! ## Overview
//!
//! - [`GaloisField`]: Core trait defining field operations
//! - [`GfPrime`]: Compile-time prime field GF(p) with const generics
//! - [`DynamicGf`]: Runtime-configured field with precomputed tables
//! - [`GfElement`]: Element in a dynamic Galois field
//!
//! ## Example
//!
//! ```
//! use taguchi::gf::{DynamicGf, GaloisField};
//!
//! // Create GF(7) - a prime field
//! let gf7 = DynamicGf::new(7).unwrap();
//!
//! // Create elements
//! let a = gf7.element(3);
//! let b = gf7.element(5);
//!
//! // Perform arithmetic
//! let sum = a.add(b.clone());  // 3 + 5 = 8 ≡ 1 (mod 7)
//! let prod = a.mul(b);         // 3 * 5 = 15 ≡ 1 (mod 7)
//! ```

mod prime;
mod tables;
mod element;
mod poly;

pub use prime::GfPrime;
pub use tables::GfTables;
pub use element::{DynamicGf, GfElement};
pub use poly::{IrreduciblePoly, IRREDUCIBLE_POLYS};

use std::fmt::Debug;
use std::hash::Hash;

/// Core trait for Galois field arithmetic.
///
/// This trait defines the fundamental operations for elements of a finite field.
/// All operations are closed within the field - the result is always another
/// element of the same field.
///
/// # Field Axioms
///
/// Implementations must satisfy these axioms:
/// - **Closure**: add, mul produce field elements
/// - **Associativity**: (a + b) + c = a + (b + c), (a * b) * c = a * (b * c)
/// - **Commutativity**: a + b = b + a, a * b = b * a
/// - **Identity**: a + 0 = a, a * 1 = a
/// - **Inverse**: a + (-a) = 0, a * a^(-1) = 1 (for a ≠ 0)
/// - **Distributivity**: a * (b + c) = a * b + a * c
pub trait GaloisField: Copy + Clone + Eq + PartialEq + Hash + Debug + Default + Send + Sync {
    /// The order of the field (number of elements).
    fn order(&self) -> u32;

    /// The characteristic of the field (the prime p where q = p^n).
    fn characteristic(&self) -> u32;

    /// The degree of the field extension (n where q = p^n).
    fn degree(&self) -> u32;

    /// The additive identity (zero element).
    fn zero(&self) -> Self;

    /// The multiplicative identity (one element).
    fn one(&self) -> Self;

    /// Create an element from its integer representation.
    ///
    /// The value is taken modulo the field order.
    fn from_u32(&self, val: u32) -> Self;

    /// Convert the element to its integer representation.
    fn to_u32(&self) -> u32;

    /// Check if this element is zero.
    fn is_zero(&self) -> bool;

    /// Check if this element is one.
    fn is_one(&self) -> bool;

    /// Additive inverse (-a).
    fn neg(&self) -> Self;

    /// Multiplicative inverse (a^(-1)).
    ///
    /// # Panics
    ///
    /// Panics if called on zero.
    fn inv(&self) -> Self;

    /// Checked multiplicative inverse.
    ///
    /// Returns `None` if called on zero.
    fn checked_inv(&self) -> Option<Self>;

    /// Field addition.
    fn add(&self, rhs: Self) -> Self;

    /// Field subtraction.
    fn sub(&self, rhs: Self) -> Self;

    /// Field multiplication.
    fn mul(&self, rhs: Self) -> Self;

    /// Field division.
    ///
    /// # Panics
    ///
    /// Panics if rhs is zero.
    fn div(&self, rhs: Self) -> Self;

    /// Checked field division.
    ///
    /// Returns `None` if rhs is zero.
    fn checked_div(&self, rhs: Self) -> Option<Self>;

    /// Exponentiation by squaring.
    ///
    /// Computes self^exp efficiently in O(log exp) operations.
    fn pow(&self, exp: u32) -> Self;
}

/// Trait for fields that support quadratic character (Legendre symbol).
///
/// This is used in Paley constructions for Hadamard matrices.
pub trait QuadraticCharacter: GaloisField {
    /// Compute the quadratic character (Legendre symbol) of this element.
    ///
    /// Returns:
    /// - `0` if self is zero
    /// - `1` if self is a non-zero quadratic residue
    /// - `-1` if self is a quadratic non-residue
    fn chi(&self) -> i8;

    /// Check if this element is a quadratic residue.
    ///
    /// Zero is not considered a quadratic residue.
    fn is_quadratic_residue(&self) -> bool {
        self.chi() == 1
    }

    /// Check if this element is a quadratic non-residue.
    fn is_quadratic_nonresidue(&self) -> bool {
        self.chi() == -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf_prime_basic() {
        let gf7 = DynamicGf::new(7).unwrap();

        // Test zero and one
        let zero = gf7.zero();
        let one = gf7.one();
        assert!(zero.is_zero());
        assert!(one.is_one());
        assert_eq!(zero.to_u32(), 0);
        assert_eq!(one.to_u32(), 1);

        // Test addition
        let a = gf7.element(3);
        let b = gf7.element(5);
        assert_eq!(a.add(b.clone()).to_u32(), 1); // 3 + 5 = 8 ≡ 1 (mod 7)

        // Test multiplication
        assert_eq!(a.mul(b.clone()).to_u32(), 1); // 3 * 5 = 15 ≡ 1 (mod 7)

        // Test inverse
        assert_eq!(a.mul(a.inv()).to_u32(), 1);
        assert_eq!(b.mul(b.inv()).to_u32(), 1);
    }

    #[test]
    fn test_field_axioms() {
        let gf5 = DynamicGf::new(5).unwrap();

        // Test all pairs for commutativity
        for i in 0..5 {
            for j in 0..5 {
                let a = gf5.element(i);
                let b = gf5.element(j);

                // Commutativity
                assert_eq!(a.add(b.clone()).to_u32(), b.add(a.clone()).to_u32());
                assert_eq!(a.mul(b.clone()).to_u32(), b.mul(a.clone()).to_u32());

                // Identity
                assert_eq!(a.add(gf5.zero()).to_u32(), a.to_u32());
                assert_eq!(a.mul(gf5.one()).to_u32(), a.to_u32());

                // Additive inverse
                assert_eq!(a.add(a.neg()).to_u32(), 0);

                // Multiplicative inverse (for non-zero)
                if !a.is_zero() {
                    assert_eq!(a.mul(a.inv()).to_u32(), 1);
                }
            }
        }
    }
}

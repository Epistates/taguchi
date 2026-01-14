//! Dynamic Galois field and element types.
//!
//! This module provides a runtime-configured Galois field implementation
//! where the field order is determined at runtime rather than compile time.
//! This is necessary for orthogonal array construction where the number of
//! levels can be any prime power.

use std::fmt;
use std::sync::Arc;

use super::GfTables;
use crate::error::Result;

/// A dynamically-configured Galois field.
///
/// This struct holds the precomputed arithmetic tables for a Galois field
/// of order q = p^n, where p is prime and n >= 1.
///
/// The field is reference-counted internally, so cloning is cheap.
///
/// # Example
///
/// ```
/// use taguchi::gf::DynamicGf;
///
/// let gf7 = DynamicGf::new(7).unwrap();
/// let a = gf7.element(3);
/// let b = gf7.element(5);
///
/// let sum = a.add(b);
/// assert_eq!(sum.to_u32(), 1); // 3 + 5 = 8 ≡ 1 (mod 7)
/// ```
#[derive(Clone)]
pub struct DynamicGf {
    tables: Arc<GfTables>,
}

impl DynamicGf {
    /// Create a new Galois field of the given order.
    ///
    /// The order must be a prime power (p^n for some prime p and n >= 1).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The order is not a prime power
    /// - No irreducible polynomial is available for extension fields
    ///
    /// # Example
    ///
    /// ```
    /// use taguchi::gf::DynamicGf;
    ///
    /// let gf7 = DynamicGf::new(7).unwrap();  // Prime field
    /// let gf9 = DynamicGf::new(9).unwrap();  // Extension field GF(3^2)
    ///
    /// assert!(DynamicGf::new(6).is_err());   // 6 is not a prime power
    /// ```
    pub fn new(order: u32) -> Result<Self> {
        let tables = GfTables::new_extension(order)?;
        Ok(Self {
            tables: Arc::new(tables),
        })
    }

    /// Get the field order (number of elements).
    #[must_use]
    pub fn order(&self) -> u32 {
        self.tables.order()
    }

    /// Get the field characteristic (the prime p where q = p^n).
    #[must_use]
    pub fn characteristic(&self) -> u32 {
        self.tables.characteristic()
    }

    /// Get the extension degree (n where q = p^n).
    #[must_use]
    pub fn degree(&self) -> u32 {
        self.tables.degree()
    }

    /// Create a field element from an integer value.
    ///
    /// The value should be in the range [0, order). Values outside this
    /// range will be reduced modulo the order.
    #[must_use]
    pub fn element(&self, value: u32) -> GfElement {
        GfElement {
            value: value % self.tables.order(),
            field: self.clone(),
        }
    }

    /// Get the zero element (additive identity).
    #[must_use]
    pub fn zero(&self) -> GfElement {
        self.element(0)
    }

    /// Get the one element (multiplicative identity).
    #[must_use]
    pub fn one(&self) -> GfElement {
        self.element(1)
    }

    /// Iterate over all elements of the field.
    pub fn elements(&self) -> impl Iterator<Item = GfElement> + '_ {
        (0..self.order()).map(move |v| self.element(v))
    }

    /// Iterate over all non-zero elements of the field.
    pub fn units(&self) -> impl Iterator<Item = GfElement> + '_ {
        (1..self.order()).map(move |v| self.element(v))
    }

    /// Evaluate a polynomial at multiple points efficiently.
    ///
    /// This uses Horner's method optimized for batch processing.
    /// - `coeffs`: Polynomial coefficients [a_0, a_1, ..., a_n] for a_0 + a_1*x + ...
    /// - `points`: Points x to evaluate at.
    /// - `results`: Mutable slice to store results.
    ///
    /// # Panics
    ///
    /// Panics if `points.len() != results.len()`.
    pub fn bulk_eval_poly(&self, coeffs: &[u32], points: &[u32], results: &mut [u32]) {
        assert_eq!(points.len(), results.len());

        if coeffs.is_empty() {
            results.fill(0);
            return;
        }

        // Initialize with the highest degree coefficient
        let last_coeff = coeffs[coeffs.len() - 1];
        results.fill(last_coeff);

        // Iterate backwards through coefficients (Horner's method)
        for &coeff in coeffs.iter().rev().skip(1) {
            // result = result * x + coeff
            for i in 0..points.len() {
                let x = points[i];
                let current_val = results[i];
                // Manual mul + add to avoid GfElement overhead
                let mul_res = self.tables.mul(current_val, x);
                results[i] = self.tables.add(mul_res, coeff);
            }
        }
    }

    /// Perform bulk linear transformation: y = a*x + b
    ///
    /// Computes `results[i] = a * points[i] + b` for all i.
    pub fn bulk_linear_transform(&self, a: u32, b: u32, points: &[u32], results: &mut [u32]) {
        assert_eq!(points.len(), results.len());

        for i in 0..points.len() {
            let x = points[i];
            let ax = self.tables.mul(a, x);
            results[i] = self.tables.add(ax, b);
        }
    }

    /// Access the underlying tables for direct operations.
    #[must_use]
    pub fn tables(&self) -> &GfTables {
        &self.tables
    }
}

impl fmt::Debug for DynamicGf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.degree() == 1 {
            write!(f, "GF({})", self.order())
        } else {
            write!(f, "GF({}^{})", self.characteristic(), self.degree())
        }
    }
}

impl fmt::Display for DynamicGf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.degree() == 1 {
            write!(f, "GF({})", self.order())
        } else {
            write!(f, "GF({}^{})", self.characteristic(), self.degree())
        }
    }
}

/// An element of a dynamic Galois field.
///
/// This type holds both the element value and a reference to the field.
/// Arithmetic operations are performed using the field's precomputed tables.
#[derive(Clone)]
pub struct GfElement {
    value: u32,
    field: DynamicGf,
}

impl GfElement {
    /// Get the integer representation of this element.
    #[must_use]
    pub fn to_u32(&self) -> u32 {
        self.value
    }

    /// Get the integer representation of this element.
    #[must_use]
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Get the field this element belongs to.
    #[must_use]
    pub fn field(&self) -> &DynamicGf {
        &self.field
    }

    /// Check if this element is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Check if this element is one.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Additive inverse (-a).
    #[must_use]
    pub fn neg(&self) -> Self {
        Self {
            value: self.field.tables.neg(self.value),
            field: self.field.clone(),
        }
    }

    /// Multiplicative inverse (a^(-1)).
    ///
    /// # Panics
    ///
    /// Panics if called on zero.
    #[must_use]
    pub fn inv(&self) -> Self {
        assert!(!self.is_zero(), "cannot compute inverse of zero");
        Self {
            value: self.field.tables.inv(self.value),
            field: self.field.clone(),
        }
    }

    /// Checked multiplicative inverse.
    ///
    /// Returns `None` if called on zero.
    #[must_use]
    pub fn checked_inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(Self {
                value: self.field.tables.inv(self.value),
                field: self.field.clone(),
            })
        }
    }

    /// Field addition.
    #[must_use]
    pub fn add(&self, rhs: Self) -> Self {
        Self {
            value: self.field.tables.add(self.value, rhs.value),
            field: self.field.clone(),
        }
    }

    /// Field subtraction.
    #[must_use]
    pub fn sub(&self, rhs: Self) -> Self {
        Self {
            value: self.field.tables.sub(self.value, rhs.value),
            field: self.field.clone(),
        }
    }

    /// Field multiplication.
    #[must_use]
    pub fn mul(&self, rhs: Self) -> Self {
        Self {
            value: self.field.tables.mul(self.value, rhs.value),
            field: self.field.clone(),
        }
    }

    /// Field division.
    ///
    /// # Panics
    ///
    /// Panics if rhs is zero.
    #[must_use]
    pub fn div(&self, rhs: Self) -> Self {
        assert!(!rhs.is_zero(), "division by zero");
        Self {
            value: self.field.tables.div(self.value, rhs.value),
            field: self.field.clone(),
        }
    }

    /// Checked field division.
    ///
    /// Returns `None` if rhs is zero.
    #[must_use]
    pub fn checked_div(&self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() {
            None
        } else {
            Some(Self {
                value: self.field.tables.div(self.value, rhs.value),
                field: self.field.clone(),
            })
        }
    }

    /// Exponentiation by squaring.
    #[must_use]
    pub fn pow(&self, exp: u32) -> Self {
        Self {
            value: self.field.tables.pow(self.value, exp),
            field: self.field.clone(),
        }
    }
}

impl PartialEq for GfElement {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.field.order() == other.field.order()
    }
}

impl Eq for GfElement {}

impl std::hash::Hash for GfElement {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.field.order().hash(state);
    }
}

impl fmt::Debug for GfElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.field, self.value)
    }
}

impl fmt::Display for GfElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Default for GfElement {
    fn default() -> Self {
        // Default to GF(2) with value 0
        let field = DynamicGf::new(2).expect("GF(2) should always be constructible");
        field.zero()
    }
}

// Implement standard operators
impl std::ops::Add for GfElement {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        GfElement::add(&self, rhs)
    }
}

impl std::ops::Sub for GfElement {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        GfElement::sub(&self, rhs)
    }
}

impl std::ops::Mul for GfElement {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        GfElement::mul(&self, rhs)
    }
}

impl std::ops::Div for GfElement {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        GfElement::div(&self, rhs)
    }
}

impl std::ops::Neg for GfElement {
    type Output = Self;

    fn neg(self) -> Self::Output {
        GfElement::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf7_creation() {
        let gf7 = DynamicGf::new(7).unwrap();
        assert_eq!(gf7.order(), 7);
        assert_eq!(gf7.characteristic(), 7);
        assert_eq!(gf7.degree(), 1);
    }

    #[test]
    fn test_gf9_creation() {
        let gf9 = DynamicGf::new(9).unwrap();
        assert_eq!(gf9.order(), 9);
        assert_eq!(gf9.characteristic(), 3);
        assert_eq!(gf9.degree(), 2);
    }

    #[test]
    fn test_invalid_order() {
        assert!(DynamicGf::new(6).is_err());
        assert!(DynamicGf::new(10).is_err());
        assert!(DynamicGf::new(1).is_err());
        assert!(DynamicGf::new(0).is_err());
    }

    #[test]
    fn test_element_arithmetic() {
        let gf7 = DynamicGf::new(7).unwrap();
        let a = gf7.element(3);
        let b = gf7.element(5);

        assert_eq!(a.add(b.clone()).to_u32(), 1);
        assert_eq!(a.sub(b.clone()).to_u32(), 5); // 3 - 5 = -2 ≡ 5 (mod 7)
        assert_eq!(a.mul(b.clone()).to_u32(), 1);
        assert_eq!(a.div(b).to_u32(), 2); // 3 / 5 = 3 * 3 = 9 ≡ 2 (mod 7)
    }

    #[test]
    fn test_element_operators() {
        let gf5 = DynamicGf::new(5).unwrap();
        let a = gf5.element(3);
        let b = gf5.element(2);

        assert_eq!((a.clone() + b.clone()).to_u32(), 0);
        assert_eq!((a.clone() - b.clone()).to_u32(), 1);
        assert_eq!((a.clone() * b.clone()).to_u32(), 1);
        assert_eq!((a / b).to_u32(), 4); // 3 / 2 = 3 * 3 = 9 ≡ 4 (mod 5)
    }

    #[test]
    fn test_field_iteration() {
        let gf5 = DynamicGf::new(5).unwrap();

        let elements: Vec<u32> = gf5.elements().map(|e| e.to_u32()).collect();
        assert_eq!(elements, vec![0, 1, 2, 3, 4]);

        let units: Vec<u32> = gf5.units().map(|e| e.to_u32()).collect();
        assert_eq!(units, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_power() {
        let gf7 = DynamicGf::new(7).unwrap();
        let a = gf7.element(3);

        assert_eq!(a.pow(0).to_u32(), 1);
        assert_eq!(a.pow(1).to_u32(), 3);
        assert_eq!(a.pow(2).to_u32(), 2); // 9 mod 7
        assert_eq!(a.pow(6).to_u32(), 1); // Fermat's little theorem
    }

    #[test]
    fn test_display() {
        let gf7 = DynamicGf::new(7).unwrap();
        assert_eq!(format!("{}", gf7), "GF(7)");

        let gf9 = DynamicGf::new(9).unwrap();
        assert_eq!(format!("{}", gf9), "GF(3^2)");

        let elem = gf7.element(5);
        assert_eq!(format!("{}", elem), "5");
    }
}

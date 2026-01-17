//! Prime field implementation GF(p) using const generics.
//!
//! This module provides a compile-time prime field implementation where the
//! prime modulus is a const generic parameter. This allows for zero-cost
//! abstractions when the field size is known at compile time.

use std::fmt;

/// An element of the prime field GF(P).
///
/// This is a compile-time implementation where P must be a prime number.
/// The type parameter P is the field order.
///
/// # Example
///
/// ```
/// use taguchi::gf::GfPrime;
///
/// type GF7 = GfPrime<7>;
///
/// let a = GF7::from_value(3);
/// let b = GF7::from_value(5);
/// let sum = a + b; // 3 + 5 = 8 ≡ 1 (mod 7)
/// assert_eq!(sum.value(), 1);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GfPrime<const P: u32> {
    value: u32,
}

impl<const P: u32> GfPrime<P> {
    /// Create a new field element from a value.
    ///
    /// The value is reduced modulo P.
    #[must_use]
    pub const fn from_value(value: u32) -> Self {
        Self { value: value % P }
    }

    /// Get the value of this element.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    /// Create the zero element.
    #[must_use]
    pub const fn zero_element() -> Self {
        Self { value: 0 }
    }

    /// Create the one element.
    #[must_use]
    pub const fn one_element() -> Self {
        Self { value: 1 }
    }

    /// Add two elements.
    #[must_use]
    pub const fn add_elements(self, rhs: Self) -> Self {
        Self {
            value: (self.value + rhs.value) % P,
        }
    }

    /// Subtract two elements.
    #[must_use]
    pub const fn sub_elements(self, rhs: Self) -> Self {
        Self {
            value: (self.value + P - rhs.value) % P,
        }
    }

    /// Multiply two elements.
    #[must_use]
    pub const fn mul_elements(self, rhs: Self) -> Self {
        Self {
            value: ((self.value as u64 * rhs.value as u64) % P as u64) as u32,
        }
    }

    /// Negate an element.
    #[must_use]
    pub const fn neg_element(self) -> Self {
        if self.value == 0 {
            self
        } else {
            Self {
                value: P - self.value,
            }
        }
    }

    /// Compute the multiplicative inverse using Fermat's little theorem.
    ///
    /// For prime p: a^(-1) = a^(p-2) (mod p)
    #[must_use]
    pub const fn inv_element(self) -> Self {
        self.pow_element(P - 2)
    }

    /// Compute self^exp using binary exponentiation.
    #[must_use]
    pub const fn pow_element(self, mut exp: u32) -> Self {
        let mut result = Self::one_element();
        let mut base = self;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_elements(base);
            }
            exp >>= 1;
            base = base.mul_elements(base);
        }

        result
    }
}

impl<const P: u32> Default for GfPrime<P> {
    fn default() -> Self {
        Self::zero_element()
    }
}

impl<const P: u32> fmt::Debug for GfPrime<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GF({})[{}]", P, self.value)
    }
}

impl<const P: u32> fmt::Display for GfPrime<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<const P: u32> From<u32> for GfPrime<P> {
    fn from(value: u32) -> Self {
        Self::from_value(value)
    }
}

impl<const P: u32> From<GfPrime<P>> for u32 {
    fn from(elem: GfPrime<P>) -> Self {
        elem.value
    }
}

// Implement standard operators
impl<const P: u32> std::ops::Add for GfPrime<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_elements(rhs)
    }
}

impl<const P: u32> std::ops::Sub for GfPrime<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_elements(rhs)
    }
}

impl<const P: u32> std::ops::Mul for GfPrime<P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_elements(rhs)
    }
}

impl<const P: u32> std::ops::Neg for GfPrime<P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.neg_element()
    }
}

impl<const P: u32> std::ops::Div for GfPrime<P> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.mul_elements(rhs.inv_element())
    }
}

impl<const P: u32> std::ops::AddAssign for GfPrime<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> std::ops::SubAssign for GfPrime<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> std::ops::MulAssign for GfPrime<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> std::ops::DivAssign for GfPrime<P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Type alias for GF(2)
pub type GF2 = GfPrime<2>;
/// Type alias for GF(3)
pub type GF3 = GfPrime<3>;
/// Type alias for GF(5)
pub type GF5 = GfPrime<5>;
/// Type alias for GF(7)
pub type GF7 = GfPrime<7>;
/// Type alias for GF(11)
pub type GF11 = GfPrime<11>;
/// Type alias for GF(13)
pub type GF13 = GfPrime<13>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf7_arithmetic() {
        let a = GF7::from_value(3);
        let b = GF7::from_value(5);

        // Addition: 3 + 5 = 8 ≡ 1 (mod 7)
        assert_eq!((a + b).value(), 1);

        // Subtraction: 3 - 5 = -2 ≡ 5 (mod 7)
        assert_eq!((a - b).value(), 5);

        // Multiplication: 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!((a * b).value(), 1);

        // Division: 3 / 5 = 3 * 5^(-1) = 3 * 3 = 9 ≡ 2 (mod 7)
        // Because 5 * 3 = 15 ≡ 1 (mod 7), so 5^(-1) = 3
        assert_eq!((a / b).value(), 2);

        // Negation: -3 ≡ 4 (mod 7)
        assert_eq!((-a).value(), 4);

        // Inverse: 3^(-1) = 5 (because 3 * 5 ≡ 1 mod 7)
        assert_eq!(a.inv_element().value(), 5);
    }

    #[test]
    fn test_gf7_identity() {
        let zero = GF7::zero_element();
        let one = GF7::one_element();
        let a = GF7::from_value(4);

        assert_eq!((a + zero).value(), a.value());
        assert_eq!((a * one).value(), a.value());
        assert_eq!((a * zero).value(), 0);
    }

    #[test]
    fn test_gf7_inverse_all() {
        // Verify a * a^(-1) = 1 for all non-zero elements
        for i in 1..7u32 {
            let a = GF7::from_value(i);
            let inv = a.inv_element();
            assert_eq!((a * inv).value(), 1, "Failed for {}", i);
        }
    }

    #[test]
    fn test_gf_power() {
        let a = GF7::from_value(3);

        // 3^0 = 1
        assert_eq!(a.pow_element(0).value(), 1);
        // 3^1 = 3
        assert_eq!(a.pow_element(1).value(), 3);
        // 3^2 = 9 ≡ 2 (mod 7)
        assert_eq!(a.pow_element(2).value(), 2);
        // 3^3 = 27 ≡ 6 (mod 7)
        assert_eq!(a.pow_element(3).value(), 6);
        // 3^6 = 1 (Fermat's little theorem: a^(p-1) = 1)
        assert_eq!(a.pow_element(6).value(), 1);
    }

    #[test]
    fn test_gf_operators() {
        let mut a = GF5::from_value(3);
        let b = GF5::from_value(2);

        a += b;
        assert_eq!(a.value(), 0); // 3 + 2 = 5 ≡ 0 (mod 5)

        a = GF5::from_value(3);
        a -= b;
        assert_eq!(a.value(), 1); // 3 - 2 = 1

        a = GF5::from_value(3);
        a *= b;
        assert_eq!(a.value(), 1); // 3 * 2 = 6 ≡ 1 (mod 5)
    }

    #[test]
    fn test_gf2() {
        let zero = GF2::zero_element();
        let one = GF2::one_element();

        // GF(2) addition is XOR
        assert_eq!((zero + zero).value(), 0);
        assert_eq!((zero + one).value(), 1);
        assert_eq!((one + zero).value(), 1);
        assert_eq!((one + one).value(), 0);

        // GF(2) multiplication is AND
        assert_eq!((zero * zero).value(), 0);
        assert_eq!((zero * one).value(), 0);
        assert_eq!((one * zero).value(), 0);
        assert_eq!((one * one).value(), 1);
    }
}

//! Precomputed arithmetic tables for Galois fields.
//!
//! This module provides lookup tables for fast field arithmetic.
//! For small fields (order < ~1000), table-based arithmetic is significantly
//! faster than computing operations on the fly.

use crate::error::{Error, Result};
use crate::utils::{factor_prime_power, is_prime};

/// Precomputed arithmetic tables for a Galois field.
///
/// The tables allow O(1) field operations at the cost of O(q²) memory
/// for addition/multiplication tables, or O(q) for log/exp tables.
#[derive(Debug, Clone)]
pub struct GfTables {
    /// The order of the field.
    order: u32,
    /// The prime characteristic.
    characteristic: u32,
    /// The extension degree.
    degree: u32,
    /// Multiplication table: mul[a * order + b] = a * b
    mul: Vec<u32>,
    /// Addition table: add[a * order + b] = a + b
    add: Vec<u32>,
    /// Multiplicative inverse table: inv[a] = a^(-1) (inv[0] is undefined)
    inv: Vec<u32>,
    /// Additive inverse (negation) table: neg[a] = -a
    neg: Vec<u32>,
}

impl GfTables {
    /// Create arithmetic tables for a prime field GF(p).
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is not prime.
    pub fn new_prime(p: u32) -> Result<Self> {
        if !is_prime(p) {
            return Err(Error::NotPrimePower(p));
        }

        let order = p;
        let size = (order * order) as usize;

        let mut add = vec![0u32; size];
        let mut mul = vec![0u32; size];
        let mut inv = vec![0u32; order as usize];
        let mut neg = vec![0u32; order as usize];

        // Build addition table
        for a in 0..order {
            for b in 0..order {
                add[(a * order + b) as usize] = (a + b) % order;
            }
            // Negation
            neg[a as usize] = if a == 0 { 0 } else { order - a };
        }

        // Build multiplication table
        for a in 0..order {
            for b in 0..order {
                mul[(a * order + b) as usize] = ((a as u64 * b as u64) % order as u64) as u32;
            }
        }

        // Build inverse table using extended Euclidean algorithm or Fermat's little theorem
        inv[0] = 0; // Undefined, but we set to 0 for safety
        for a in 1..order {
            // a^(-1) = a^(p-2) mod p (Fermat's little theorem)
            inv[a as usize] = mod_pow(a, order - 2, order);
        }

        Ok(Self {
            order,
            characteristic: p,
            degree: 1,
            mul,
            add,
            inv,
            neg,
        })
    }

    /// Create arithmetic tables for a prime power field GF(p^n).
    ///
    /// This requires an irreducible polynomial of degree n over GF(p).
    ///
    /// # Errors
    ///
    /// Returns an error if `q` is not a prime power or if no irreducible
    /// polynomial is available for this field.
    pub fn new_extension(q: u32) -> Result<Self> {
        let factorization = factor_prime_power(q).ok_or(Error::NotPrimePower(q))?;

        if factorization.exponent == 1 {
            // This is actually a prime field
            return Self::new_prime(q);
        }

        let p = factorization.prime;
        let n = factorization.exponent;

        // Get irreducible polynomial for GF(p^n)
        let irr_poly =
            super::poly::get_irreducible_poly(p, n).ok_or(Error::NoIrreduciblePolynomial(q))?;

        Self::build_extension_tables(p, n, &irr_poly)
    }

    /// Build tables for an extension field given an irreducible polynomial.
    fn build_extension_tables(p: u32, n: u32, irr_poly: &[u32]) -> Result<Self> {
        let order = p.pow(n);
        let size = (order * order) as usize;

        let mut add = vec![0u32; size];
        let mut mul = vec![0u32; size];
        let mut inv = vec![0u32; order as usize];
        let mut neg = vec![0u32; order as usize];

        // Elements of GF(p^n) are represented as polynomials of degree < n
        // over GF(p). We encode them as integers:
        // a_0 + a_1*p + a_2*p^2 + ... + a_{n-1}*p^{n-1}
        // where each a_i is in {0, 1, ..., p-1}

        // Build addition table (polynomial addition mod p)
        for a in 0..order {
            for b in 0..order {
                add[(a * order + b) as usize] = poly_add(a, b, p, n);
            }
            neg[a as usize] = poly_neg(a, p, n);
        }

        // Build multiplication table (polynomial multiplication mod irr_poly)
        for a in 0..order {
            for b in 0..order {
                mul[(a * order + b) as usize] = poly_mul(a, b, p, n, irr_poly);
            }
        }

        // Build inverse table
        inv[0] = 0;
        for a in 1..order {
            inv[a as usize] = poly_inv(a, p, n, &mul, order);
        }

        Ok(Self {
            order,
            characteristic: p,
            degree: n,
            mul,
            add,
            inv,
            neg,
        })
    }

    /// Get the field order.
    #[must_use]
    pub fn order(&self) -> u32 {
        self.order
    }

    /// Get the field characteristic.
    #[must_use]
    pub fn characteristic(&self) -> u32 {
        self.characteristic
    }

    /// Get the extension degree.
    #[must_use]
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// Add two field elements.
    #[must_use]
    pub fn add(&self, a: u32, b: u32) -> u32 {
        if self.characteristic == 2 {
            a ^ b
        } else {
            self.add[(a * self.order + b) as usize]
        }
    }

    /// Subtract two field elements.
    #[must_use]
    pub fn sub(&self, a: u32, b: u32) -> u32 {
        if self.characteristic == 2 {
            a ^ b
        } else {
            self.add[(a * self.order + self.neg[b as usize]) as usize]
        }
    }

    /// Multiply two field elements.
    #[must_use]
    pub fn mul(&self, a: u32, b: u32) -> u32 {
        self.mul[(a * self.order + b) as usize]
    }

    /// Divide two field elements.
    ///
    /// # Panics
    ///
    /// Panics if b is zero.
    #[must_use]
    pub fn div(&self, a: u32, b: u32) -> u32 {
        assert!(b != 0, "division by zero");
        self.mul[(a * self.order + self.inv[b as usize]) as usize]
    }

    /// Get the additive inverse (negation) of an element.
    #[must_use]
    pub fn neg(&self, a: u32) -> u32 {
        self.neg[a as usize]
    }

    /// Get the multiplicative inverse of an element.
    ///
    /// # Panics
    ///
    /// Panics if a is zero.
    #[must_use]
    pub fn inv(&self, a: u32) -> u32 {
        assert!(a != 0, "inverse of zero");
        self.inv[a as usize]
    }

    /// Check if an element has a multiplicative inverse.
    #[must_use]
    pub fn has_inv(&self, a: u32) -> bool {
        a != 0
    }

    /// Compute a^exp using repeated squaring with table lookups.
    #[must_use]
    pub fn pow(&self, mut base: u32, mut exp: u32) -> u32 {
        let mut result = 1u32;
        while exp > 0 {
            if exp & 1 == 1 {
                result = self.mul(result, base);
            }
            exp >>= 1;
            base = self.mul(base, base);
        }
        result
    }
}

/// Modular exponentiation: base^exp mod modulus
fn mod_pow(mut base: u32, mut exp: u32, modulus: u32) -> u32 {
    let mut result = 1u32;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u64 * base as u64) % modulus as u64) as u32;
        }
        exp >>= 1;
        base = ((base as u64 * base as u64) % modulus as u64) as u32;
    }
    result
}

/// Add two polynomials represented as integers (coefficient-wise mod p).
fn poly_add(a: u32, b: u32, p: u32, n: u32) -> u32 {
    let mut result = 0u32;
    let mut pow_p = 1u32;
    let mut a = a;
    let mut b = b;

    for _ in 0..n {
        let coef_a = a % p;
        let coef_b = b % p;
        let sum = (coef_a + coef_b) % p;
        result += sum * pow_p;

        a /= p;
        b /= p;
        pow_p *= p;
    }

    result
}

/// Negate a polynomial (negate each coefficient mod p).
fn poly_neg(a: u32, p: u32, n: u32) -> u32 {
    let mut result = 0u32;
    let mut pow_p = 1u32;
    let mut a = a;

    for _ in 0..n {
        let coef = a % p;
        let neg_coef = if coef == 0 { 0 } else { p - coef };
        result += neg_coef * pow_p;

        a /= p;
        pow_p *= p;
    }

    result
}

/// Multiply two polynomials and reduce modulo the irreducible polynomial.
fn poly_mul(a: u32, b: u32, p: u32, n: u32, irr_poly: &[u32]) -> u32 {
    // Extract coefficients of a and b
    let mut a_coeffs = vec![0u32; n as usize];
    let mut b_coeffs = vec![0u32; n as usize];
    let mut temp_a = a;
    let mut temp_b = b;

    for i in 0..n as usize {
        a_coeffs[i] = temp_a % p;
        b_coeffs[i] = temp_b % p;
        temp_a /= p;
        temp_b /= p;
    }

    // Multiply polynomials (result has degree up to 2n-2)
    let mut product = vec![0u32; (2 * n - 1) as usize];
    for i in 0..n as usize {
        for j in 0..n as usize {
            product[i + j] = (product[i + j] + a_coeffs[i] * b_coeffs[j]) % p;
        }
    }

    // Reduce modulo the irreducible polynomial
    // irr_poly represents x^n + c_{n-1}*x^{n-1} + ... + c_0
    // So x^n = -c_{n-1}*x^{n-1} - ... - c_0 (mod irr_poly)
    for i in ((n as usize)..product.len()).rev() {
        if product[i] != 0 {
            let coef = product[i];
            product[i] = 0;
            // Subtract coef * irr_poly shifted by (i - n) positions
            for j in 0..n as usize {
                let sub = (coef * irr_poly[j]) % p;
                product[i - n as usize + j] = (product[i - n as usize + j] + p - sub) % p;
            }
        }
    }

    // Convert back to integer representation
    let mut result = 0u32;
    let mut pow_p = 1u32;
    for i in 0..n as usize {
        result += product[i] * pow_p;
        pow_p *= p;
    }

    result
}

/// Find the multiplicative inverse using brute force search.
/// This is simple but works for any field.
fn poly_inv(a: u32, _p: u32, _n: u32, mul_table: &[u32], order: u32) -> u32 {
    if a == 0 {
        return 0;
    }

    // Find x such that a * x = 1
    for x in 1..order {
        if mul_table[(a * order + x) as usize] == 1 {
            return x;
        }
    }

    // This should never happen for a valid field element
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_field_tables() {
        let gf7 = GfTables::new_prime(7).unwrap();

        // Test addition
        assert_eq!(gf7.add(3, 5), 1); // 3 + 5 = 8 ≡ 1 (mod 7)
        assert_eq!(gf7.add(6, 1), 0); // 6 + 1 = 7 ≡ 0 (mod 7)

        // Test multiplication
        assert_eq!(gf7.mul(3, 5), 1); // 3 * 5 = 15 ≡ 1 (mod 7)
        assert_eq!(gf7.mul(2, 4), 1); // 2 * 4 = 8 ≡ 1 (mod 7)

        // Test inverse
        for a in 1..7u32 {
            let inv_a = gf7.inv(a);
            assert_eq!(gf7.mul(a, inv_a), 1, "a={}, inv={}", a, inv_a);
        }

        // Test negation
        for a in 0..7u32 {
            let neg_a = gf7.neg(a);
            assert_eq!(gf7.add(a, neg_a), 0, "a={}, neg={}", a, neg_a);
        }
    }

    #[test]
    fn test_gf4_extension() {
        // GF(4) = GF(2^2) with irreducible polynomial x^2 + x + 1
        let gf4 = GfTables::new_extension(4).unwrap();

        assert_eq!(gf4.order(), 4);
        assert_eq!(gf4.characteristic(), 2);
        assert_eq!(gf4.degree(), 2);

        // In GF(2^n), addition is XOR
        assert_eq!(gf4.add(0, 0), 0);
        assert_eq!(gf4.add(1, 1), 0);
        assert_eq!(gf4.add(2, 2), 0);
        assert_eq!(gf4.add(3, 3), 0);

        // Test that all non-zero elements have inverses
        for a in 1..4u32 {
            let inv_a = gf4.inv(a);
            assert_eq!(gf4.mul(a, inv_a), 1, "a={}, inv={}", a, inv_a);
        }
    }

    #[test]
    fn test_gf9_extension() {
        // GF(9) = GF(3^2)
        let gf9 = GfTables::new_extension(9).unwrap();

        assert_eq!(gf9.order(), 9);
        assert_eq!(gf9.characteristic(), 3);
        assert_eq!(gf9.degree(), 2);

        // Test field axioms
        for a in 0..9u32 {
            // Additive identity
            assert_eq!(gf9.add(a, 0), a);
            // Additive inverse
            assert_eq!(gf9.add(a, gf9.neg(a)), 0);
            // Multiplicative identity
            assert_eq!(gf9.mul(a, 1), a);
            // Multiplicative inverse (for non-zero)
            if a != 0 {
                assert_eq!(gf9.mul(a, gf9.inv(a)), 1);
            }
        }
    }

    #[test]
    fn test_not_prime_power() {
        assert!(GfTables::new_prime(6).is_err());
        assert!(GfTables::new_extension(6).is_err());
        assert!(GfTables::new_extension(10).is_err());
    }
}

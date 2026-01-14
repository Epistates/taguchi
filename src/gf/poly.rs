//! Irreducible polynomials for extension field construction.
//!
//! This module provides a database of irreducible polynomials over GF(p)
//! for constructing extension fields GF(p^n).
//!
//! An irreducible polynomial of degree n over GF(p) is required to construct
//! GF(p^n). The polynomial is represented as a vector of coefficients
//! [c_0, c_1, ..., c_{n-1}] where the polynomial is:
//! x^n + c_{n-1}*x^{n-1} + ... + c_1*x + c_0
//!
//! Note: The leading coefficient (for x^n) is always 1 and is implicit.

/// An irreducible polynomial over GF(p).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrreduciblePoly {
    /// The prime characteristic.
    pub p: u32,
    /// The degree of the polynomial.
    pub n: u32,
    /// Coefficients [c_0, c_1, ..., c_{n-1}] for x^n + c_{n-1}*x^{n-1} + ... + c_0
    pub coeffs: Vec<u32>,
}

impl IrreduciblePoly {
    /// Create a new irreducible polynomial.
    #[must_use]
    pub fn new(p: u32, coeffs: Vec<u32>) -> Self {
        Self {
            p,
            n: coeffs.len() as u32,
            coeffs,
        }
    }

    /// Get the field order this polynomial defines: p^n.
    #[must_use]
    pub fn field_order(&self) -> u32 {
        self.p.pow(self.n)
    }
}

/// Lookup table for irreducible polynomials.
///
/// These are carefully chosen primitive polynomials (when possible) for
/// common field orders. The coefficients represent the polynomial
/// x^n + c_{n-1}*x^{n-1} + ... + c_1*x + c_0 where the vector is [c_0, c_1, ..., c_{n-1}].
pub static IRREDUCIBLE_POLYS: &[(u32, u32, &[u32])] = &[
    // GF(2^n) - Binary extension fields
    // x^2 + x + 1
    (2, 2, &[1, 1]),
    // x^3 + x + 1
    (2, 3, &[1, 1, 0]),
    // x^4 + x + 1
    (2, 4, &[1, 1, 0, 0]),
    // x^5 + x^2 + 1
    (2, 5, &[1, 0, 1, 0, 0]),
    // x^6 + x + 1
    (2, 6, &[1, 1, 0, 0, 0, 0]),
    // x^7 + x^3 + 1
    (2, 7, &[1, 0, 0, 1, 0, 0, 0]),
    // x^8 + x^4 + x^3 + x + 1 (AES polynomial)
    (2, 8, &[1, 1, 0, 1, 1, 0, 0, 0]),
    // GF(3^n) - Ternary extension fields
    // x^2 + 1
    (3, 2, &[1, 0]),
    // x^3 + 2x + 1
    (3, 3, &[1, 2, 0]),
    // x^4 + 2x^3 + 2
    (3, 4, &[2, 0, 0, 2]),
    // GF(5^n) - Extension fields over GF(5)
    // x^2 + 2
    (5, 2, &[2, 0]),
    // x^3 + x + 2
    (5, 3, &[2, 1, 0]),
    // GF(7^n) - Extension fields over GF(7)
    // x^2 + 1
    (7, 2, &[1, 0]),
    // GF(11^n)
    // x^2 + 1
    (11, 2, &[1, 0]),
    // GF(13^n)
    // x^2 + 2
    (13, 2, &[2, 0]),
];

/// Get an irreducible polynomial for GF(p^n).
///
/// Returns `None` if no polynomial is available for the given parameters.
#[must_use]
pub fn get_irreducible_poly(p: u32, n: u32) -> Option<Vec<u32>> {
    for &(poly_p, poly_n, coeffs) in IRREDUCIBLE_POLYS {
        if poly_p == p && poly_n == n {
            return Some(coeffs.to_vec());
        }
    }
    None
}

/// Check if an irreducible polynomial is available for GF(p^n).
#[must_use]
pub fn has_irreducible_poly(p: u32, n: u32) -> bool {
    get_irreducible_poly(p, n).is_some()
}

/// Get all available field orders.
#[must_use]
pub fn available_field_orders() -> Vec<u32> {
    IRREDUCIBLE_POLYS
        .iter()
        .map(|&(p, n, _)| p.pow(n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_irreducible_poly() {
        // GF(4) = GF(2^2)
        let poly = get_irreducible_poly(2, 2).unwrap();
        assert_eq!(poly, vec![1, 1]); // x^2 + x + 1

        // GF(8) = GF(2^3)
        let poly = get_irreducible_poly(2, 3).unwrap();
        assert_eq!(poly, vec![1, 1, 0]); // x^3 + x + 1

        // GF(9) = GF(3^2)
        let poly = get_irreducible_poly(3, 2).unwrap();
        assert_eq!(poly, vec![1, 0]); // x^2 + 1

        // Non-existent
        assert!(get_irreducible_poly(17, 5).is_none());
    }

    #[test]
    fn test_has_irreducible_poly() {
        assert!(has_irreducible_poly(2, 2));
        assert!(has_irreducible_poly(2, 8));
        assert!(has_irreducible_poly(3, 2));
        assert!(!has_irreducible_poly(17, 5));
    }

    #[test]
    fn test_available_orders() {
        let orders = available_field_orders();
        assert!(orders.contains(&4)); // 2^2
        assert!(orders.contains(&8)); // 2^3
        assert!(orders.contains(&9)); // 3^2
        assert!(orders.contains(&256)); // 2^8
    }

    #[test]
    fn test_irreducible_poly_struct() {
        let poly = IrreduciblePoly::new(2, vec![1, 1]);
        assert_eq!(poly.p, 2);
        assert_eq!(poly.n, 2);
        assert_eq!(poly.field_order(), 4);
    }
}

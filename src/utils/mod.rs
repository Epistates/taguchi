//! Utility functions for primality testing, combinatorics, and other helpers.
//!
//! This module provides fundamental mathematical utilities used throughout
//! the library, particularly for validating prime power orders in Galois fields.

mod primality;

pub use primality::{
    factor_prime_power, is_prime, is_prime_power, smallest_prime_factor, PrimePowerFactorization,
};

/// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
///
/// Returns `None` if the result would overflow `u64`.
///
/// # Examples
///
/// ```
/// use taguchi::utils::binomial;
///
/// assert_eq!(binomial(5, 2), Some(10));
/// assert_eq!(binomial(10, 5), Some(252));
/// assert_eq!(binomial(5, 0), Some(1));
/// assert_eq!(binomial(5, 5), Some(1));
/// assert_eq!(binomial(3, 5), Some(0)); // k > n
/// ```
#[must_use]
pub fn binomial(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }

    // Use symmetry: C(n, k) = C(n, n-k)
    let k = k.min(n - k);

    if k == 0 {
        return Some(1);
    }

    let mut result: u64 = 1;
    for i in 0..k {
        // result = result * (n - i) / (i + 1)
        // To avoid overflow, we divide as we go
        result = result.checked_mul(n - i)?;
        result /= i + 1;
    }

    Some(result)
}

/// Compute the power of a base modulo a modulus using binary exponentiation.
///
/// Computes `base^exp mod modulus` efficiently in O(log exp) time.
///
/// # Panics
///
/// Panics if `modulus` is 0.
///
/// # Examples
///
/// ```
/// use taguchi::utils::mod_pow;
///
/// assert_eq!(mod_pow(2, 10, 1000), 24);  // 2^10 = 1024, 1024 mod 1000 = 24
/// assert_eq!(mod_pow(3, 5, 7), 5);       // 3^5 = 243, 243 mod 7 = 5
/// ```
#[must_use]
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    assert!(modulus > 0, "modulus must be positive");

    if modulus == 1 {
        return 0;
    }

    let mut result = 1u64;
    base %= modulus;

    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp >>= 1;
        base = base.wrapping_mul(base) % modulus;
    }

    result
}

/// Generate all k-combinations of indices 0..n.
///
/// Returns an iterator over all ways to choose k items from n items.
///
/// # Examples
///
/// ```
/// use taguchi::utils::combinations;
///
/// let combos: Vec<Vec<usize>> = combinations(4, 2).collect();
/// assert_eq!(combos.len(), 6); // C(4,2) = 6
/// assert_eq!(combos[0], vec![0, 1]);
/// assert_eq!(combos[5], vec![2, 3]);
/// ```
pub fn combinations(n: usize, k: usize) -> impl Iterator<Item = Vec<usize>> {
    CombinationIterator::new(n, k)
}

/// Iterator over k-combinations of 0..n.
struct CombinationIterator {
    n: usize,
    k: usize,
    indices: Vec<usize>,
    finished: bool,
}

impl CombinationIterator {
    fn new(n: usize, k: usize) -> Self {
        if k > n || k == 0 {
            return Self {
                n,
                k,
                indices: Vec::new(),
                finished: k > n,
            };
        }

        Self {
            n,
            k,
            indices: (0..k).collect(),
            finished: false,
        }
    }
}

impl Iterator for CombinationIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if self.k == 0 {
            self.finished = true;
            return Some(Vec::new());
        }

        let result = self.indices.clone();

        // Find rightmost index that can be incremented
        let mut i = self.k;
        while i > 0 {
            i -= 1;
            if self.indices[i] < self.n - self.k + i {
                // Increment this index and reset all following indices
                self.indices[i] += 1;
                for j in (i + 1)..self.k {
                    self.indices[j] = self.indices[j - 1] + 1;
                }
                return Some(result);
            }
        }

        // No more combinations
        self.finished = true;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // This is an approximation; exact count would require more computation
            let count =
                binomial(self.n as u64, self.k as u64).unwrap_or(usize::MAX as u64) as usize;
            (0, Some(count))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(0, 0), Some(1));
        assert_eq!(binomial(5, 0), Some(1));
        assert_eq!(binomial(5, 5), Some(1));
        assert_eq!(binomial(5, 2), Some(10));
        assert_eq!(binomial(10, 3), Some(120));
        assert_eq!(binomial(20, 10), Some(184_756));
        assert_eq!(binomial(3, 5), Some(0)); // k > n
    }

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(2, 0, 7), 1);
        assert_eq!(mod_pow(0, 5, 7), 0);
        assert_eq!(mod_pow(3, 4, 5), 1); // 81 mod 5 = 1
        assert_eq!(mod_pow(7, 3, 11), 2); // 343 mod 11 = 2
    }

    #[test]
    fn test_combinations() {
        let c: Vec<_> = combinations(4, 2).collect();
        assert_eq!(c.len(), 6);
        assert_eq!(c[0], vec![0, 1]);
        assert_eq!(c[1], vec![0, 2]);
        assert_eq!(c[2], vec![0, 3]);
        assert_eq!(c[3], vec![1, 2]);
        assert_eq!(c[4], vec![1, 3]);
        assert_eq!(c[5], vec![2, 3]);

        let c: Vec<_> = combinations(5, 3).collect();
        assert_eq!(c.len(), 10);

        let c: Vec<_> = combinations(3, 0).collect();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0], Vec::<usize>::new());

        let c: Vec<_> = combinations(3, 4).collect();
        assert_eq!(c.len(), 0);
    }
}

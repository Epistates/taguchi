//! Primality testing and prime power factorization.
//!
//! This module provides efficient algorithms for:
//! - Testing whether a number is prime (Miller-Rabin)
//! - Testing whether a number is a prime power
//! - Factoring prime powers into (prime, exponent) pairs

use super::mod_pow;

/// Result of factoring a prime power.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimePowerFactorization {
    /// The prime base.
    pub prime: u32,
    /// The exponent (power).
    pub exponent: u32,
}

impl PrimePowerFactorization {
    /// Compute the value p^k.
    #[must_use]
    pub fn value(&self) -> u64 {
        (self.prime as u64).pow(self.exponent)
    }
}

/// Test if a number is prime using the Miller-Rabin primality test.
///
/// For n < 2^32, this is deterministic (no false positives) using a fixed
/// set of witnesses that covers all 32-bit integers.
///
/// # Examples
///
/// ```
/// use taguchi::utils::is_prime;
///
/// assert!(is_prime(2));
/// assert!(is_prime(3));
/// assert!(!is_prime(4));
/// assert!(is_prime(7));
/// assert!(!is_prime(9));
/// assert!(is_prime(97));
/// assert!(!is_prime(100));
/// ```
#[must_use]
pub fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    if n < 9 {
        return true;
    }
    if n % 3 == 0 {
        return false;
    }

    // Miller-Rabin with witnesses that work for all n < 2^32
    // These witnesses are sufficient for deterministic primality testing
    // for all 32-bit integers.
    let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    // Write n-1 as 2^r * d where d is odd
    let n_minus_1 = (n - 1) as u64;
    let r = n_minus_1.trailing_zeros();
    let d = n_minus_1 >> r;

    'witness: for &a in witnesses {
        if a >= n as u64 {
            continue;
        }

        let mut x = mod_pow(a, d, n as u64);

        if x == 1 || x == n_minus_1 {
            continue 'witness;
        }

        for _ in 0..(r - 1) {
            x = x.wrapping_mul(x) % (n as u64);
            if x == n_minus_1 {
                continue 'witness;
            }
        }

        return false;
    }

    true
}

/// Test if a number is a prime power (p^k for some prime p and k >= 1).
///
/// # Examples
///
/// ```
/// use taguchi::utils::is_prime_power;
///
/// assert!(is_prime_power(2));   // 2^1
/// assert!(is_prime_power(4));   // 2^2
/// assert!(is_prime_power(8));   // 2^3
/// assert!(is_prime_power(9));   // 3^2
/// assert!(is_prime_power(27));  // 3^3
/// assert!(is_prime_power(25));  // 5^2
/// assert!(!is_prime_power(6));  // 2 * 3
/// assert!(!is_prime_power(10)); // 2 * 5
/// assert!(!is_prime_power(1));  // Not a prime power
/// assert!(!is_prime_power(0));  // Not a prime power
/// ```
#[must_use]
pub fn is_prime_power(n: u32) -> bool {
    factor_prime_power(n).is_some()
}

/// Factor a number as a prime power if possible.
///
/// Returns `Some((p, k))` if `n = p^k` for some prime p and k >= 1,
/// otherwise returns `None`.
///
/// # Examples
///
/// ```
/// use taguchi::utils::{factor_prime_power, PrimePowerFactorization};
///
/// assert_eq!(factor_prime_power(8), Some(PrimePowerFactorization { prime: 2, exponent: 3 }));
/// assert_eq!(factor_prime_power(9), Some(PrimePowerFactorization { prime: 3, exponent: 2 }));
/// assert_eq!(factor_prime_power(7), Some(PrimePowerFactorization { prime: 7, exponent: 1 }));
/// assert_eq!(factor_prime_power(6), None);  // 2 * 3
/// assert_eq!(factor_prime_power(1), None);
/// assert_eq!(factor_prime_power(0), None);
/// ```
#[must_use]
pub fn factor_prime_power(n: u32) -> Option<PrimePowerFactorization> {
    if n < 2 {
        return None;
    }

    // First, check if n is prime
    if is_prime(n) {
        return Some(PrimePowerFactorization {
            prime: n,
            exponent: 1,
        });
    }

    // Try to find a prime base p such that n = p^k
    // We only need to check primes up to n^(1/2)

    // Check powers of 2 first (most common case)
    if n.is_power_of_two() {
        return Some(PrimePowerFactorization {
            prime: 2,
            exponent: n.trailing_zeros(),
        });
    }

    // For other primes, we check if n = p^k by testing roots
    // If n = p^k, then p = n^(1/k) for some k >= 2

    // Maximum possible exponent: log_2(n)
    let max_exp = 32 - n.leading_zeros();

    for k in 2..=max_exp {
        // Compute the k-th root of n
        if let Some(root) = integer_kth_root(n as u64, k) {
            let root = root as u32;
            if root > 1 && is_prime(root) {
                // Verify that root^k == n
                if root.checked_pow(k).map_or(false, |v| v == n) {
                    return Some(PrimePowerFactorization {
                        prime: root,
                        exponent: k,
                    });
                }
            }
        }
    }

    None
}

/// Compute the integer k-th root of n (floor(n^(1/k))).
fn integer_kth_root(n: u64, k: u32) -> Option<u64> {
    if k == 0 {
        return None;
    }
    if n == 0 {
        return Some(0);
    }
    if k == 1 {
        return Some(n);
    }
    if n == 1 {
        return Some(1);
    }

    // Use Newton's method to find floor(n^(1/k))
    // x_{n+1} = ((k-1) * x_n + n / x_n^(k-1)) / k

    // Initial guess: 2^(ceil(log2(n) / k))
    let bits = 64 - n.leading_zeros();
    let mut x = 1u64 << ((bits + k - 1) / k);

    loop {
        // Compute x^(k-1) carefully to avoid overflow
        let x_pow_k_minus_1 = match x.checked_pow(k - 1) {
            Some(v) => v,
            None => {
                // x is too large, reduce it
                x /= 2;
                continue;
            }
        };

        if x_pow_k_minus_1 == 0 {
            return Some(x);
        }

        let n_div_x_pow = n / x_pow_k_minus_1;
        let new_x = ((k as u64 - 1) * x + n_div_x_pow) / (k as u64);

        if new_x >= x {
            // Verify the result
            if let Some(x_pow_k) = x.checked_pow(k) {
                if x_pow_k == n {
                    return Some(x);
                }
            }
            return None;
        }

        x = new_x;
    }
}

/// Get the smallest prime factor of n.
///
/// Returns `None` if n < 2.
#[must_use]
pub fn smallest_prime_factor(n: u32) -> Option<u32> {
    if n < 2 {
        return None;
    }
    if n % 2 == 0 {
        return Some(2);
    }

    let sqrt_n = (n as f64).sqrt() as u32 + 1;
    let mut i = 3;
    while i <= sqrt_n {
        if n % i == 0 {
            return Some(i);
        }
        i += 2;
    }

    Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        // Small primes
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(is_prime(17));
        assert!(is_prime(19));
        assert!(is_prime(23));

        // Non-primes
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(6));
        assert!(!is_prime(8));
        assert!(!is_prime(9));
        assert!(!is_prime(10));
        assert!(!is_prime(100));

        // Larger primes
        assert!(is_prime(97));
        assert!(is_prime(101));
        assert!(is_prime(1009));
        assert!(is_prime(10007));
        assert!(is_prime(100003));

        // Carmichael numbers (must be correctly identified as composite)
        assert!(!is_prime(561)); // 3 * 11 * 17
        assert!(!is_prime(1105)); // 5 * 13 * 17
        assert!(!is_prime(1729)); // 7 * 13 * 19 (Hardy-Ramanujan number)
    }

    #[test]
    fn test_is_prime_power() {
        // Prime powers of 2
        assert!(is_prime_power(2));
        assert!(is_prime_power(4));
        assert!(is_prime_power(8));
        assert!(is_prime_power(16));
        assert!(is_prime_power(32));
        assert!(is_prime_power(64));

        // Prime powers of 3
        assert!(is_prime_power(3));
        assert!(is_prime_power(9));
        assert!(is_prime_power(27));
        assert!(is_prime_power(81));

        // Prime powers of 5
        assert!(is_prime_power(5));
        assert!(is_prime_power(25));
        assert!(is_prime_power(125));

        // Other primes (p^1)
        assert!(is_prime_power(7));
        assert!(is_prime_power(11));
        assert!(is_prime_power(13));

        // Not prime powers
        assert!(!is_prime_power(0));
        assert!(!is_prime_power(1));
        assert!(!is_prime_power(6)); // 2 * 3
        assert!(!is_prime_power(10)); // 2 * 5
        assert!(!is_prime_power(12)); // 2^2 * 3
        assert!(!is_prime_power(15)); // 3 * 5
        assert!(!is_prime_power(18)); // 2 * 3^2
        assert!(!is_prime_power(20)); // 2^2 * 5
    }

    #[test]
    fn test_factor_prime_power() {
        assert_eq!(
            factor_prime_power(8),
            Some(PrimePowerFactorization {
                prime: 2,
                exponent: 3
            })
        );
        assert_eq!(
            factor_prime_power(9),
            Some(PrimePowerFactorization {
                prime: 3,
                exponent: 2
            })
        );
        assert_eq!(
            factor_prime_power(16),
            Some(PrimePowerFactorization {
                prime: 2,
                exponent: 4
            })
        );
        assert_eq!(
            factor_prime_power(27),
            Some(PrimePowerFactorization {
                prime: 3,
                exponent: 3
            })
        );
        assert_eq!(
            factor_prime_power(7),
            Some(PrimePowerFactorization {
                prime: 7,
                exponent: 1
            })
        );
        assert_eq!(
            factor_prime_power(125),
            Some(PrimePowerFactorization {
                prime: 5,
                exponent: 3
            })
        );

        // Not prime powers
        assert_eq!(factor_prime_power(0), None);
        assert_eq!(factor_prime_power(1), None);
        assert_eq!(factor_prime_power(6), None);
        assert_eq!(factor_prime_power(12), None);
    }

    #[test]
    fn test_smallest_prime_factor() {
        assert_eq!(smallest_prime_factor(0), None);
        assert_eq!(smallest_prime_factor(1), None);
        assert_eq!(smallest_prime_factor(2), Some(2));
        assert_eq!(smallest_prime_factor(3), Some(3));
        assert_eq!(smallest_prime_factor(4), Some(2));
        assert_eq!(smallest_prime_factor(6), Some(2));
        assert_eq!(smallest_prime_factor(9), Some(3));
        assert_eq!(smallest_prime_factor(15), Some(3));
        assert_eq!(smallest_prime_factor(17), Some(17));
        assert_eq!(smallest_prime_factor(35), Some(5));
    }

    #[test]
    fn test_prime_power_factorization_value() {
        let f = PrimePowerFactorization {
            prime: 2,
            exponent: 10,
        };
        assert_eq!(f.value(), 1024);

        let f = PrimePowerFactorization {
            prime: 3,
            exponent: 5,
        };
        assert_eq!(f.value(), 243);
    }
}

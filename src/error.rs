//! Error types for the taguchi library.
//!
//! This module provides comprehensive error handling using the `thiserror` crate,
//! with specific error variants for Galois field operations, OA construction,
//! parameter validation, and verification.

use thiserror::Error;

/// The main error type for the taguchi library.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    // ============ Galois Field Errors ============
    /// The specified order is not a prime power.
    #[error("order {0} is not a prime power (must be p^k for prime p and k >= 1)")]
    NotPrimePower(u32),

    /// Attempted division by zero in a Galois field.
    #[error("division by zero in GF({order})")]
    DivisionByZero {
        /// The order of the field where division by zero occurred.
        order: u32,
    },

    /// Element value is out of range for the specified field.
    #[error("element {value} is out of range for GF({order}), must be in 0..{order}")]
    ElementOutOfRange {
        /// The invalid element value.
        value: u32,
        /// The order of the field.
        order: u32,
    },

    /// No irreducible polynomial is known for the specified field order.
    #[error("no irreducible polynomial available for GF({0})")]
    NoIrreduciblePolynomial(u32),

    // ============ Parameter Validation Errors ============
    /// Invalid OA parameters.
    #[error("invalid OA parameters: {message}")]
    InvalidParams {
        /// Description of what is invalid.
        message: String,
    },

    /// The number of factors exceeds the maximum allowed by the construction.
    #[error("factors {factors} exceeds maximum {max} for {algorithm} construction")]
    TooManyFactors {
        /// Requested number of factors.
        factors: usize,
        /// Maximum allowed factors.
        max: usize,
        /// Name of the construction algorithm.
        algorithm: &'static str,
    },

    /// The number of levels is not a prime power as required.
    #[error("levels {levels} is not a prime power as required by {algorithm}")]
    LevelsNotPrimePower {
        /// The invalid levels value.
        levels: u32,
        /// Name of the construction algorithm.
        algorithm: &'static str,
    },

    /// The construction requires levels to be a power of 2.
    #[error("{algorithm} requires levels to be a power of 2, got {levels}")]
    RequiresPowerOfTwo {
        /// The invalid levels value.
        levels: u32,
        /// Name of the construction algorithm.
        algorithm: &'static str,
    },

    /// The strength is invalid for the construction.
    #[error("strength {strength} is invalid for {algorithm} (valid range: {min}..={max})")]
    InvalidStrength {
        /// The requested strength.
        strength: u32,
        /// Minimum valid strength.
        min: u32,
        /// Maximum valid strength.
        max: u32,
        /// Name of the construction algorithm.
        algorithm: &'static str,
    },

    // ============ Construction Errors ============
    /// Construction failed with a specific reason.
    #[error("construction failed: {message}")]
    ConstructionFailed {
        /// Description of why construction failed.
        message: String,
    },

    /// No suitable algorithm found for the requested parameters.
    #[error("no suitable algorithm found for OA({runs}, {factors}, {levels}, {strength})")]
    NoSuitableAlgorithm {
        /// Requested number of runs.
        runs: usize,
        /// Requested number of factors.
        factors: usize,
        /// Requested number of levels.
        levels: u32,
        /// Requested strength.
        strength: u32,
    },

    // ============ Verification Errors ============
    /// Verification of OA properties failed.
    #[error("verification failed: {message}")]
    VerificationFailed {
        /// Description of what verification failed.
        message: String,
    },

    /// The array does not have the claimed strength.
    #[error("strength mismatch: claimed {claimed}, actual {actual}")]
    StrengthMismatch {
        /// The claimed strength.
        claimed: u32,
        /// The actual verified strength.
        actual: u32,
    },

    // ============ Dimension Errors ============
    /// Array dimensions are inconsistent.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension description.
        expected: String,
        /// Actual dimension description.
        actual: String,
    },

    /// Index is out of bounds.
    #[error("index {index} is out of bounds for size {size}")]
    IndexOutOfBounds {
        /// The invalid index.
        index: usize,
        /// The maximum valid size.
        size: usize,
    },
}

/// A specialized `Result` type for taguchi operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// Create a new `InvalidParams` error.
    #[must_use]
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::InvalidParams {
            message: message.into(),
        }
    }

    /// Create a new `ConstructionFailed` error.
    #[must_use]
    pub fn construction_failed(message: impl Into<String>) -> Self {
        Self::ConstructionFailed {
            message: message.into(),
        }
    }

    /// Create a new `VerificationFailed` error.
    #[must_use]
    pub fn verification_failed(message: impl Into<String>) -> Self {
        Self::VerificationFailed {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::NotPrimePower(6);
        assert!(err.to_string().contains("6"));
        assert!(err.to_string().contains("prime power"));

        let err = Error::DivisionByZero { order: 7 };
        assert!(err.to_string().contains("division by zero"));
        assert!(err.to_string().contains("GF(7)"));

        let err = Error::TooManyFactors {
            factors: 10,
            max: 8,
            algorithm: "Bose",
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("8"));
        assert!(err.to_string().contains("Bose"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = Error::NotPrimePower(6);
        let err2 = Error::NotPrimePower(6);
        let err3 = Error::NotPrimePower(10);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}

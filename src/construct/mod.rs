//! Orthogonal array construction algorithms.
//!
//! This module provides various algorithms for constructing orthogonal arrays.
//! Each construction method has specific requirements and produces arrays with
//! particular properties.
//!
//! ## Available Constructions
//!
//! | Construction | Parameters | Requirements |
//! |-------------|------------|--------------|
//! | [`Bose`] | OA(q², k, q, 2) | Prime power q, k ≤ q+1 |
//! | [`Bush`] | OA(q^t, k, q, t) | Prime power q, k ≤ t+1 |
//! | [`BoseBush`] | OA(2q², k, q, 2) | q = 2^m, k ≤ 2q+1 |
//! | [`HadamardSylvester`] | OA(2^m, k, 2, 2) | k ≤ 2^m - 1 |
//! | [`HadamardPaley`] | OA(p+1, k, 2, 2) | p ≡ 3 (mod 4) prime, k ≤ p |
//! | [`AddelmanKempthorne`] | OA(2q², k, q, 2) | Odd prime power q, k ≤ 2q+1 |
//!
//! ## Usage
//!
//! All constructors implement the [`Constructor`] trait:
//!
//! ```
//! use taguchi::construct::{Constructor, Bose};
//!
//! let bose = Bose::new(3);
//! let oa = bose.construct(4).expect("construction failed");
//!
//! assert_eq!(oa.runs(), 9);
//! assert_eq!(oa.factors(), 4);
//! assert_eq!(oa.levels(), 3);
//! ```
//!
//! ## Choosing a Construction
//!
//! - For **binary factors**: Use [`HadamardSylvester`] for efficient 2-level designs
//! - For **strength 2**: Use [`Bose`] (most common) or [`BoseBush`] for more factors
//! - For **higher strength**: Use [`Bush`] which supports arbitrary strength t
//! - For **odd prime power levels with many factors**: Use [`AddelmanKempthorne`]

mod addelman;
mod bose;
mod bose_bush;
mod bush;
mod hadamard;

pub use addelman::AddelmanKempthorne;
pub use bose::Bose;
pub use bose_bush::BoseBush;
pub use bush::Bush;
pub use hadamard::{HadamardPaley, HadamardSylvester};

use crate::error::Result;
use crate::oa::OA;

/// Trait for orthogonal array construction algorithms.
///
/// All construction algorithms implement this trait, providing a uniform
/// interface for generating orthogonal arrays.
pub trait Constructor: Send + Sync {
    /// Get the name of this construction method.
    fn name(&self) -> &'static str;

    /// Get a description of the family of OAs this constructor produces.
    fn family(&self) -> &'static str;

    /// Get the number of levels for arrays produced by this constructor.
    fn levels(&self) -> u32;

    /// Get the strength of arrays produced by this constructor.
    fn strength(&self) -> u32;

    /// Get the number of runs for arrays produced by this constructor.
    fn runs(&self) -> usize;

    /// Get the maximum number of factors this constructor can produce.
    fn max_factors(&self) -> usize;

    /// Construct an orthogonal array with the specified number of factors.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - factors exceeds max_factors()
    /// - construction fails for any other reason
    fn construct(&self, factors: usize) -> Result<OA>;
}

/// Trait for parallel construction (feature-gated).
#[cfg(feature = "parallel")]
pub trait ParConstructor: Constructor {
    /// Construct using parallel algorithms.
    fn construct_par(&self, factors: usize) -> Result<OA>;
}

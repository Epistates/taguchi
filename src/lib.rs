//! # Taguchi
//!
//! A state-of-the-art orthogonal array library for experimental design, Monte Carlo sampling,
//! and combinatorial testing.
//!
//! ## Overview
//!
//! Orthogonal arrays (OAs) are mathematical structures used in:
//! - **Design of Experiments (DOE)**: Taguchi methods for quality engineering
//! - **Monte Carlo Integration**: Quasi-random sampling with better uniformity
//! - **Software Testing**: Combinatorial test case generation (OATS)
//!
//! This library provides:
//! - Multiple construction algorithms (Bose, Bush, Bose-Bush, Addelman-Kempthorne, Hadamard)
//! - Full prime power support via custom Galois field arithmetic
//! - Verification and validation of array properties
//! - Modern Rust API with builder patterns and comprehensive error handling
//!
//! ## Quick Start
//!
//! The easiest way to create an orthogonal array is with the builder:
//!
//! ```rust
//! use taguchi::OABuilder;
//!
//! // Automatically selects the best construction
//! let oa = OABuilder::new()
//!     .levels(3)
//!     .factors(4)
//!     .strength(2)
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(oa.runs(), 9);    // Bose: 3²
//! assert_eq!(oa.factors(), 4);
//! assert_eq!(oa.levels(), 3);
//! ```
//!
//! Or use a specific construction directly:
//!
//! ```rust
//! use taguchi::construct::{Constructor, Bose};
//!
//! let oa = Bose::new(3)
//!     .construct(4)
//!     .expect("Failed to construct OA");
//!
//! assert_eq!(oa.runs(), 9);
//! assert_eq!(oa.factors(), 4);
//! assert_eq!(oa.levels(), 3);
//! ```
//!
//! ## Notation
//!
//! An orthogonal array is denoted as OA(N, k, s, t) where:
//! - **N**: Number of runs (rows)
//! - **k**: Number of factors (columns)
//! - **s**: Number of levels (symbols 0, 1, ..., s-1)
//! - **t**: Strength (every t-column subarray contains all s^t tuples equally)
//!
//! ## Features
//!
//! - `serde`: Enable serialization/deserialization of OA structures
//! - `parallel`: Enable parallel construction using rayon
//! - `stats`: Enable statistical analysis utilities
//! - `python`: Enable Python bindings via PyO3

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod builder;
pub mod catalogue;
pub mod construct;
#[cfg(feature = "doe")]
pub mod doe;
pub mod error;
pub mod gf;
pub mod oa;
#[cfg(feature = "python")]
pub mod python;
pub mod utils;

#[cfg(feature = "parallel")]
pub mod parallel;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::builder::{build_oa, OABuilder};
    pub use crate::construct::{
        AddelmanKempthorne, Bose, BoseBush, Bush, Constructor, HadamardPaley, HadamardSylvester,
        RaoHamming,
    };
    pub use crate::error::{Error, Result};
    pub use crate::gf::{
        available_field_orders, get_irreducible_poly, has_irreducible_poly, DynamicGf, GaloisField,
        GfElement, GF11, GF13, GF2, GF3, GF5, GF7,
    };
    pub use crate::oa::{compute_strength, verify_strength, BalanceReport, OAParams, OA};
    pub use crate::utils::{factor_prime_power, is_prime, is_prime_power, smallest_prime_factor};

    #[cfg(feature = "parallel")]
    pub use crate::parallel::{
        par_build_oa, ParAddelmanKempthorne, ParBose, ParBush, ParHadamardSylvester,
    };

    #[cfg(feature = "doe")]
    pub use crate::doe::{
        analyze, ANOVAConfig, ANOVAEntry, ANOVAResult, AnalysisConfig, ConfidenceInterval,
        DOEAnalysis, MainEffect, OptimalSettings, OptimizationType, SNRatioEffect,
    };
}

// Re-export commonly used items at crate root
pub use builder::{available_constructions, build_oa, OABuilder};
pub use catalogue::get_by_name as get_standard_oa;
pub use error::{Error, Result};
pub use oa::{compute_strength, verify_strength};
pub use utils::{is_prime, is_prime_power};

#[cfg(feature = "parallel")]
pub use parallel::{par_build_oa, ParAddelmanKempthorne, ParBose, ParBush, ParHadamardSylvester};

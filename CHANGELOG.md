# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-17

### Added

#### DOE Analysis Module (`doe` feature)
Complete Taguchi DOE analysis functionality for experimental data:

- **Main Effects Analysis**: Calculate level means, effects, ranges, and factor rankings
- **Signal-to-Noise Ratios**: Support for all three optimization types:
  - Larger-is-better: `S/N = -10 * log10(mean(1/y^2))`
  - Smaller-is-better: `S/N = -10 * log10(mean(y^2))`
  - Nominal-is-best: `S/N = 10 * log10(mean^2/variance)`
- **ANOVA with Factor Pooling**:
  - Sum of squares, degrees of freedom, mean squares
  - F-ratios and p-values using regularized incomplete beta function
  - Configurable pooling threshold with minimum unpooled factors constraint
  - Contribution percentages
- **Optimal Settings Prediction**:
  - Uses correct additive model for predictions
  - Confidence intervals with proper effective sample size calculation
  - Support for 90%, 95%, and 99% confidence levels
- **Statistical Utilities**:
  - `ln_gamma()` - Lanczos approximation
  - `regularized_incomplete_beta()` - Continued fraction expansion
  - `f_distribution_p_value()` - F-distribution CDF
  - `t_value()` - Complete lookup tables for common confidence levels

#### Parallel Construction (`parallel` feature)
High-performance parallel row generation using rayon:

- `ParBose` - Parallel Bose construction
- `ParBush` - Parallel Bush construction
- `ParAddelmanKempthorne` - Parallel Addelman-Kempthorne construction
- `ParHadamardSylvester` - Parallel Hadamard-Sylvester construction
- `par_build_oa()` - Convenience function with automatic algorithm selection

#### Addelman-Kempthorne Construction
New construction algorithm for odd prime power levels:

- Produces OA(2q^2, 2q+1, q, 2) for odd prime power q
- Complements Bose-Bush (which handles powers of 2)
- Uses quadratic non-residues for transformation constants
- Integrated into builder auto-selection

### Changed

- **Rust edition 2024**: Updated to latest Rust edition (requires Rust 1.85+)
- Updated `full` feature to include `doe`
- Improved builder auto-selection to consider Addelman-Kempthorne for odd prime powers
- Enhanced prelude exports with parallel and DOE types

### Fixed

- N/A (first feature release)

### Documentation

- Added comprehensive module documentation for all new features
- Updated roadmap with completed milestones
- Added DOE migration documentation

---

## [0.1.0] - 2025-01-15

### Added

#### Core Infrastructure
- Project structure with feature flags (`serde`, `parallel`, `stats`, `python`)
- Comprehensive error types using `thiserror`
- Miller-Rabin primality testing
- Prime power factorization utilities

#### Galois Field Arithmetic
- `GaloisField` trait with complete field operations
- Const-generic prime fields (`GfPrime<const P>`)
- Precomputed lookup tables for O(1) arithmetic
- Extension field support: GF(4), GF(8), GF(9), GF(16), GF(25), GF(27)
- Irreducible polynomial database
- Runtime-configured fields (`DynamicGf`, `GfElement`)
- Operator overloading for ergonomic arithmetic

#### OA Core
- `OAParams` struct with validation
- `OA` struct with ndarray storage
- Strength verification algorithm
- Row/column iteration and access
- Balance checking and correlation analysis
- Generalized Word Length Pattern (GWLP)

#### Construction Algorithms
- **Bose**: OA(q^2, q+1, q, 2) for any prime power q
- **Bush**: OA(q^t, t+1, q, t) for any prime power q, strength t
- **Bose-Bush**: OA(2q^2, 2q+1, q, 2) for q=2
- **Hadamard-Sylvester**: OA(2^m, 2^m-1, 2, 2) for m >= 2
- **Hadamard-Paley**: OA(p+1, p, 2, 2) for p = 3 (mod 4) prime
- **Rao-Hamming**: OAs from linear error-correcting codes

#### Builder & Catalogue
- `OABuilder` for fluent construction API
- `build_oa()` convenience function
- `available_constructions()` for parameter exploration
- Automatic selection of optimal construction algorithm
- Standard Taguchi array catalogue (L4, L8, L9, L12, L16, L18, L25, L27)

#### Mixed-Level Support
- Level collapsing for mixed-level designs
- Automatic base array selection

[0.2.0]: https://github.com/nicholaspaterno/taguchi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nicholaspaterno/taguchi/releases/tag/v0.1.0

# Taguchi: State-of-the-Art Orthogonal Array Library in Rust

## Overview

**Taguchi** is a robust, world-class Rust library for constructing and analyzing orthogonal arrays (OAs). Orthogonal arrays are combinatorial structures fundamental to Design of Experiments (DOE), Monte Carlo simulation, software testing, and quasi-random number generation.

An orthogonal array OA(N, k, s, t) consists of:
- **N** rows (runs/experiments)
- **k** columns (factors/parameters)
- **s** levels (values each factor can take)
- **t** strength (any t columns contain all s^t combinations equally often)

## Project Goals

1. **Custom Galois Field Implementation**: No external math dependencies; full control over GF(q) arithmetic for both prime fields and extension fields
2. **Comprehensive Construction Coverage**: Implement all major OA construction algorithms (Bose, Bush, Bose-Bush, Addelman-Kempthorne, Hadamard, Rao-Hamming)
3. **Modern Rust Idioms**: Type-safe, zero-cost abstractions, const generics, proper error handling
4. **Performance**: Precomputed lookup tables, optional parallelization via rayon
5. **General Purpose**: Support DOE, Monte Carlo, software testing, and research applications

## Architecture

```
taguchi/
├── src/
│   ├── lib.rs              # Crate root, prelude
│   ├── error.rs            # Error types (thiserror)
│   ├── utils/
│   │   ├── mod.rs
│   │   └── primality.rs    # Miller-Rabin, prime power factorization
│   ├── gf/
│   │   ├── mod.rs          # GaloisField trait
│   │   ├── prime.rs        # GfPrime<const P> for prime fields
│   │   ├── tables.rs       # Precomputed arithmetic tables
│   │   ├── poly.rs         # Irreducible polynomials
│   │   └── element.rs      # DynamicGf, GfElement (runtime-configured)
│   ├── oa/
│   │   ├── mod.rs          # OA struct, OAParams
│   │   └── verify.rs       # Strength verification
│   └── construct/
│       ├── mod.rs          # Constructor trait
│       ├── bose.rs         # Bose: OA(q², q+1, q, 2)
│       ├── bush.rs         # Bush: OA(q^t, t+1, q, t)
│       ├── bose_bush.rs    # Bose-Bush: OA(2q², 2q+1, q, 2)
│       ├── addelman.rs     # Addelman-Kempthorne (planned)
│       ├── hadamard.rs     # Hadamard constructions (planned)
│       └── rao_hamming.rs  # Rao-Hamming (planned)
├── Cargo.toml
├── PLAN.md                 # This file
└── docs/
    ├── architecture.md     # Detailed architecture
    ├── constructions.md    # Construction algorithms
    └── roadmap.md          # Development roadmap
```

## Implementation Status

### Phase 1: Foundation ✅
- [x] Project setup with Cargo.toml
- [x] Error handling with thiserror
- [x] Utility functions (primality, prime power factorization)

### Phase 2: Galois Fields ✅
- [x] GaloisField trait with full arithmetic
- [x] Prime fields via const generics (GfPrime<P>)
- [x] Precomputed lookup tables for fast arithmetic
- [x] Extension fields (GF(4), GF(8), GF(9), etc.)
- [x] Irreducible polynomial database
- [x] DynamicGf for runtime-configured fields

### Phase 3: OA Core ✅
- [x] OAParams validation
- [x] OA struct with ndarray storage
- [x] Strength verification algorithm
- [x] Row/column iteration

### Phase 4: Constructors (In Progress)
- [x] Bose construction: OA(q², q+1, q, 2)
- [x] Bush construction: OA(q^t, t+1, q, t)
- [x] Bose-Bush construction: OA(2q², 2q+1, q, 2) for q=2
- [x] Hadamard-Sylvester: OA(2^m, 2^m-1, 2, 2)
- [x] Hadamard-Paley: OA(p+1, p, 2, 2) for p ≡ 3 (mod 4)
- [ ] Addelman-Kempthorne: Mixed-level arrays
- [ ] Rao-Hamming: Linear codes

### Phase 5: Advanced Features
- [x] Builder pattern for easy OA construction
- [x] Auto-select best construction for given parameters
- [ ] Parallel construction with rayon
- [ ] Serialization with serde
- [ ] Statistical analysis utilities

## Key Design Decisions

### 1. Custom Galois Field Implementation
We implement GF(q) arithmetic from scratch rather than using external crates because:
- Full control over performance optimizations
- No dependency conflicts
- Tailored for OA construction needs
- Support for both compile-time (const generics) and runtime field configuration

### 2. Precomputed Lookup Tables
For fields up to order 256, we precompute addition, multiplication, and inverse tables. This trades memory for speed—critical for large OA constructions.

### 3. ndarray for Storage
We use `ndarray` for 2D array storage because:
- Efficient memory layout (row-major or column-major)
- Rich slicing and iteration API
- Foundation for future statistical analysis

### 4. Constructor Trait
All construction algorithms implement the `Constructor` trait:
```rust
pub trait Constructor {
    fn name(&self) -> &'static str;
    fn family(&self) -> &'static str;
    fn levels(&self) -> u32;
    fn strength(&self) -> u32;
    fn runs(&self) -> usize;
    fn max_factors(&self) -> usize;
    fn construct(&self, factors: usize) -> Result<OA>;
}
```

## Dependencies

```toml
[dependencies]
ndarray = "0.16"
thiserror = "2.0"

[features]
serde = ["dep:serde", "ndarray/serde"]
parallel = ["dep:rayon"]
stats = []
```

## Testing Strategy

- Unit tests for each module
- Property-based tests with proptest
- Integration tests for full construction pipelines
- Benchmarks with criterion

## References

1. Hedayat, Sloane, Stufken. "Orthogonal Arrays: Theory and Applications" (1999)
2. Bose, R.C. "On the construction of balanced incomplete block designs" (1939)
3. Bush, K.A. "Orthogonal arrays of index unity" (1952)
4. Addelman, S. & Kempthorne, O. "Orthogonal main-effect plans" (1961)
5. Owen, A.B. "Monte Carlo theory, methods and examples" (2013)

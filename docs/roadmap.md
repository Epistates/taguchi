# Development Roadmap

## Current Status

**Version**: 0.1.0-dev (Pre-release)

**Test Status**: 139 unit tests + 36 doc tests passing (with parallel feature)

### Completed Features

#### Core Infrastructure
- [x] Project structure and Cargo.toml with feature flags
- [x] Comprehensive error types with thiserror
- [x] Miller-Rabin primality testing
- [x] Prime power factorization

#### Galois Field Arithmetic
- [x] `GaloisField` trait with complete arithmetic operations
- [x] Const-generic prime fields (`GfPrime<const P>`)
- [x] Precomputed lookup tables for O(1) arithmetic
- [x] Extension field support (GF(4), GF(8), GF(9), GF(16), GF(25), GF(27))
- [x] Irreducible polynomial database
- [x] Runtime-configured fields (`DynamicGf`, `GfElement`)
- [x] Operator overloading for ergonomic arithmetic

#### OA Core
- [x] `OAParams` with validation
- [x] `OA` struct with ndarray storage
- [x] Strength verification algorithm
- [x] Row/column iteration and access

#### Construction Algorithms
- [x] **Bose**: OA(q², q+1, q, 2) for any prime power q
- [x] **Bush**: OA(q^t, t+1, q, t) for any prime power q, strength t
- [x] **Bose-Bush**: OA(2q², 2q+1, q, 2) for q=2
- [x] **Hadamard-Sylvester**: OA(2^m, 2^m-1, 2, 2) for m ≥ 2
- [x] **Hadamard-Paley**: OA(p+1, p, 2, 2) for p ≡ 3 (mod 4) prime
- [x] **Addelman-Kempthorne**: OA(2q², 2q+1, q, 2) for odd prime power q

#### Builder & Auto-Selection
- [x] `OABuilder` for fluent construction API
- [x] `build_oa()` convenience function
- [x] `available_constructions()` for parameter exploration
- [x] Automatic selection of optimal construction algorithm

#### Parallel Construction (parallel feature)
- [x] `ParBose`: Parallel row generation for Bose construction
- [x] `ParBush`: Parallel row generation for Bush construction
- [x] `ParAddelmanKempthorne`: Parallel row generation for Addelman-Kempthorne
- [x] `ParHadamardSylvester`: Parallel Hadamard construction
- [x] `par_build_oa()` convenience function with auto-selection

---

## Short-Term Goals (Next Milestone)

### Construction Algorithms
- [ ] **Bose-Bush**: Extend to q=4, 8, 16 (higher powers of 2)
- [ ] **Rao-Hamming**: OAs from linear error-correcting codes
- [ ] **Mixed-Level OAs**: Different levels per factor (extension of Addelman-Kempthorne)

### Builder Pattern ✅ COMPLETE

```rust
// Implemented API
let oa = OABuilder::new()
    .levels(3)
    .factors(4)
    .strength(2)
    .build()?;

// Auto-selects best construction (Bose in this case)
```

### Auto-Selection Logic ✅ COMPLETE

The builder now automatically selects the optimal construction:

1. For binary factors (levels=2): Try Hadamard-Sylvester first
2. For prime power levels, strength 2: Try Bose
3. For power-of-2 levels with many factors: Try Bose-Bush
4. For higher strength: Try Bush

---

## Medium-Term Goals

### Additional Constructions
- [ ] **Rao-Hamming**: OAs from linear codes
- [ ] **Latin Square**: From mutually orthogonal Latin squares (MOLS)
- [ ] **Difference Schemes**: OAs from difference sets

### Performance Optimization
- [x] **Parallel Construction**: Use rayon for row generation ✅ COMPLETE
```rust
use taguchi::parallel::{ParBose, par_build_oa};

// Parallel construction
let oa = par_build_oa(5, 6, 2).unwrap();

// Or use specific parallel constructor
let bose = ParBose::new(7);
let oa = bose.construct(8).unwrap();
```

- [ ] **SIMD Arithmetic**: Vectorized GF operations for large constructions
- [ ] **Memory Optimization**: Streaming construction for very large OAs

### Serialization (serde feature)
```rust
#[derive(Serialize, Deserialize)]
pub struct OA { ... }

// JSON, TOML, bincode support
let json = serde_json::to_string(&oa)?;
```

### Statistical Analysis (stats feature)
```rust
impl OA {
    pub fn correlation_matrix(&self) -> Array2<f64>;
    pub fn balance_check(&self) -> BalanceReport;
    pub fn space_filling_metric(&self) -> f64;
}
```

---

## Long-Term Vision

### Advanced Features
- [ ] **Nested OAs**: Hierarchical experimental designs
- [ ] **Resolution Analysis**: Higher-order interaction aliasing
- [ ] **Optimal Design Selection**: Choose best OA for given criteria
- [ ] **Response Surface Integration**: Connect with RSM libraries

### Interoperability
- [ ] **Python Bindings**: Via PyO3
```python
import taguchi
oa = taguchi.construct(levels=3, factors=4, strength=2)
```

- [ ] **R Integration**: Via extendr
- [ ] **CSV/Excel Export**: Common data formats
- [ ] **Design-Expert Compatibility**: Import/export formats

### Specialized Applications

#### Monte Carlo Quasi-Random
- [ ] Stratified sampling using OAs
- [ ] Owen scrambling for randomization
- [ ] Integration error bounds

#### Software Testing
- [ ] Covering array extensions
- [ ] Constraint handling
- [ ] Test case generation with domain knowledge

#### Machine Learning
- [ ] Hyperparameter space exploration
- [ ] Neural architecture search designs
- [ ] Bayesian optimization integration

---

## API Stability

### Stable (1.0 target)
- `OA`, `OAParams` structs
- `Constructor` trait
- `GaloisField` trait
- Core error types

### Experimental
- `DynamicGf` implementation details
- Builder pattern API
- Statistical methods

### Internal
- Lookup table implementation
- Polynomial arithmetic internals
- Verification algorithm details

---

## Testing Strategy

### Current
- Unit tests per module
- Doc tests for examples
- 74+ passing tests

### Planned
- [ ] Property-based testing with proptest
```rust
proptest! {
    #[test]
    fn bose_always_valid(q in prop::sample::select(&[2, 3, 4, 5, 7, 8, 9])) {
        let bose = Bose::new(q)?;
        let oa = bose.construct(q as usize)?;
        let result = verify_strength(&oa, 2)?;
        prop_assert!(result.is_valid);
    }
}
```

- [ ] Benchmark suite with criterion
```rust
fn bench_bose_construction(c: &mut Criterion) {
    c.bench_function("bose_q9", |b| {
        b.iter(|| Bose::new(9).unwrap().construct(10))
    });
}
```

- [ ] Fuzzing for parser/validation code
- [ ] Integration tests with real-world datasets

---

## Documentation

### Current
- [x] Module-level documentation
- [x] Function/struct documentation
- [x] Doc tests with examples

### Planned
- [ ] Book-style user guide
- [ ] Tutorial: First OA design
- [ ] Tutorial: Custom GF arithmetic
- [ ] API reference improvements
- [ ] Performance guide

---

## Release Checklist

### 0.1.0 (Alpha)
- [x] Core types implemented
- [x] 3+ construction algorithms
- [ ] All doc tests passing
- [ ] README with examples
- [ ] CHANGELOG started

### 0.2.0 (Beta)
- [ ] Builder pattern
- [ ] Auto-selection
- [ ] Parallel feature
- [ ] 5+ construction algorithms
- [ ] Property tests

### 1.0.0 (Stable)
- [ ] Complete API documentation
- [ ] Comprehensive test coverage (>90%)
- [ ] Benchmarks published
- [ ] serde feature complete
- [ ] No breaking changes planned

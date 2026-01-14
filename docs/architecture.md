# Taguchi Architecture

## Module Structure

### `error.rs` - Error Handling

Uses `thiserror` for ergonomic error definitions:

```rust
#[derive(Debug, Error)]
pub enum Error {
    #[error("Value {value} is not a prime power")]
    NotPrimePower { value: u32 },

    #[error("Division by zero in GF({order})")]
    DivisionByZero { order: u32 },

    #[error("{algorithm} supports at most {max} factors, got {factors}")]
    TooManyFactors { factors: usize, max: usize, algorithm: &'static str },

    // ... more variants
}
```

### `utils/primality.rs` - Number Theory Utilities

**Miller-Rabin Primality Test**
- Deterministic for n < 3,317,044,064,679,887,385,961,981
- Uses witness set: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

**Prime Power Factorization**
- Returns `PrimePowerFactorization { prime, exponent }` for q = p^m
- Used to construct appropriate Galois field

### `gf/` - Galois Field Arithmetic

#### Design Philosophy

We provide two complementary approaches:

1. **Compile-time fields** (`GfPrime<const P>`): Zero-cost abstractions when field is known at compile time
2. **Runtime fields** (`DynamicGf`, `GfElement`): Flexibility when field is determined at runtime

#### `GaloisField` Trait

```rust
pub trait GaloisField: Copy + Clone + Eq + Hash + Debug + Default + Send + Sync {
    fn order(&self) -> u32;
    fn zero(&self) -> Self;
    fn one(&self) -> Self;
    fn add(&self, rhs: Self) -> Self;
    fn sub(&self, rhs: Self) -> Self;
    fn neg(&self) -> Self;
    fn mul(&self, rhs: Self) -> Self;
    fn inv(&self) -> Self;
    fn div(&self, rhs: Self) -> Self;
    fn pow(&self, exp: u32) -> Self;
    fn is_zero(&self) -> bool;
    fn to_u32(&self) -> u32;
}
```

#### Prime Fields (`prime.rs`)

For prime p, GF(p) = {0, 1, ..., p-1} with:
- Addition: (a + b) mod p
- Multiplication: (a × b) mod p
- Inverse: Extended Euclidean algorithm

```rust
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct GfPrime<const P: u32> {
    value: u32,
}

impl<const P: u32> GfPrime<P> {
    pub const fn new(value: u32) -> Self {
        Self { value: value % P }
    }
}
```

#### Extension Fields (`tables.rs`, `poly.rs`)

For q = p^m (m > 1), GF(q) is constructed as polynomials over GF(p) modulo an irreducible polynomial.

**Polynomial Representation**: Element a₀ + a₁x + ... + aₘ₋₁x^(m-1) stored as integer Σ aᵢpⁱ

**Arithmetic via Lookup Tables** (`GfTables`):
- `add_table`: q × q → q
- `mul_table`: q × q → q
- `inv_table`: q → q (0 maps to 0)

**Irreducible Polynomials** (`poly.rs`):
```rust
pub fn get_irreducible_poly(p: u32, m: u32) -> Option<Polynomial> {
    // Returns minimal degree-m irreducible polynomial over GF(p)
    // Database covers common cases:
    // - GF(4) = GF(2²): x² + x + 1
    // - GF(8) = GF(2³): x³ + x + 1
    // - GF(9) = GF(3²): x² + 1
    // - GF(16) = GF(2⁴): x⁴ + x + 1
    // - etc.
}
```

#### Dynamic Fields (`element.rs`)

For runtime-configured fields:

```rust
pub struct DynamicGf {
    tables: Arc<GfTables>,
}

pub struct GfElement {
    value: u32,
    field: DynamicGf,
}
```

`GfElement` implements all arithmetic operations with operator overloading.

### `oa/` - Orthogonal Array Core

#### `OAParams` - Validated Parameters

```rust
pub struct OAParams {
    runs: usize,      // N
    factors: usize,   // k
    levels: u32,      // s
    strength: u32,    // t
}

impl OAParams {
    pub fn new(runs: usize, factors: usize, levels: u32, strength: u32) -> Result<Self> {
        // Validates:
        // - runs > 0
        // - factors > 0
        // - levels >= 2
        // - strength >= 1 && strength <= factors
        // - runs divisible by levels^strength
    }
}
```

#### `OA` - Orthogonal Array

```rust
pub struct OA {
    data: Array2<u32>,
    params: OAParams,
}
```

Provides:
- `get(row, col)` - Element access
- `row(idx)` - Row view
- `column(idx)` - Column view
- `rows()` - Row iterator
- `subarray(columns)` - Extract sub-array
- `into_data()` - Consume and return raw data

#### Strength Verification (`verify.rs`)

```rust
pub fn verify_strength(oa: &OA, t: u32) -> Result<VerificationResult> {
    // For each C(k, t) subset of t columns:
    //   Count occurrences of each s^t tuple
    //   Verify all tuples appear equally (λ = N / s^t times)
}
```

### `construct/` - Construction Algorithms

#### `Constructor` Trait

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

#### Bose Construction (`bose.rs`)

**Parameters**: OA(q², q+1, q, 2) for prime power q

**Algorithm**:
- Rows indexed by (i, j) ∈ GF(q) × GF(q), row = i×q + j
- Column 0: j
- Column c (1 ≤ c ≤ q): i + c×j in GF(q)

**Why It Works**: For any two columns c₁, c₂ (0 ≤ c₁ < c₂ ≤ q):
- If c₁ = 0: pairs are (j, i + c₂×j) — bijection for each fixed i
- Otherwise: pairs (i + c₁×j, i + c₂×j) — difference is (c₂-c₁)×j, bijection since c₂-c₁ ≠ 0

#### Bush Construction (`bush.rs`)

**Parameters**: OA(q^t, t+1, q, t) for prime power q, strength t

**Algorithm**:
- Rows indexed by coefficients (a₀, ..., a_{t-1}) ∈ GF(q)^t
- Each row defines polynomial p(x) = a₀ + a₁x + ... + a_{t-1}x^{t-1}
- Columns 0 to t-1: Evaluate p(c) at field elements c = 0, 1, ..., t-1
- Column t: Leading coefficient a_{t-1} ("point at infinity")

**Why It Works**: Any t+1 columns form a system with full rank (Vandermonde-like structure).

#### Bose-Bush Construction (`bose_bush.rs`)

**Parameters**: OA(2q², 2q+1, q, 2) for q = 2^m (currently only q=2)

**Algorithm**:
- Extends Bose by doubling rows via two blocks b ∈ {0, 1}
- Row indexed by (b, i, j), row = b×q² + i×q + j
- Column 0: j
- Columns 1 to q: i + k×j (Bose-like)
- Columns q+1 to 2q: (i + k×j) + b (block offset creates orthogonality)

**Why It Works**: For columns c and q+c:
- Block 0: identical values → diagonal pairs (0,0), (1,1)
- Block 1: offset by 1 → off-diagonal pairs (0,1), (1,0)
- Combined: all pairs equally represented

## Memory Layout

OA data stored in row-major order via `ndarray::Array2<u32>`:
- Contiguous rows for efficient row iteration
- Column views for factor analysis
- Cache-friendly for typical DOE workflows

## Thread Safety

All core types are `Send + Sync`:
- `GfTables` wrapped in `Arc` for shared ownership
- `OA` owns its data exclusively
- Constructors are stateless (can be shared across threads)

## Performance Considerations

1. **Lookup Tables**: GF arithmetic is O(1) via precomputed tables
2. **Lazy Verification**: Strength verified only when requested
3. **Minimal Allocations**: Single allocation for OA data
4. **Future**: Parallel construction via rayon feature flag

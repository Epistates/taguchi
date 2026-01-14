# Orthogonal Array Construction Algorithms

This document details all construction algorithms implemented or planned for the Taguchi library.

## Overview

| Construction | OA Parameters | Requirements | Status |
|-------------|---------------|--------------|--------|
| Bose | OA(q², q+1, q, 2) | q prime power | ✅ Complete |
| Bush | OA(q^t, t+1, q, t) | q prime power, t ≥ 2 | ✅ Complete |
| Bose-Bush | OA(2q², 2q+1, q, 2) | q = 2^m | ✅ q=2 only |
| Hadamard-Sylvester | OA(2^m, 2^m-1, 2, 2) | m ≥ 2 | ✅ Complete |
| Hadamard-Paley | OA(p+1, p, 2, 2) | p ≡ 3 (mod 4) prime | ✅ Complete |
| Addelman-Kempthorne | OA(2q², 2q+1, q, 2) | q odd prime power | ✅ Complete |
| Rao-Hamming | From linear codes | Code parameters | 🔲 Planned |

---

## Bose Construction

### Mathematical Foundation

**Reference**: Bose, R.C. (1939). "On the construction of balanced incomplete block designs"

**Theorem**: For any prime power q, there exists OA(q², q+1, q, 2).

### Algorithm

Given GF(q) with elements {0, 1, α, α², ..., α^(q-2)} where α is a primitive element:

1. **Rows**: Index by pairs (i, j) ∈ GF(q) × GF(q)
   - Row number = i × q + j
   - Total rows = q²

2. **Columns**:
   - Column 0: value = j
   - Column c (1 ≤ c ≤ q): value = i + c·j in GF(q)

### Example: OA(9, 4, 3, 2)

For q = 3, GF(3) = {0, 1, 2}:

```
Row  (i,j)  Col0(j)  Col1(i+j)  Col2(i+2j)  Col3(i)
 0   (0,0)    0         0          0          0
 1   (0,1)    1         1          2          0
 2   (0,2)    2         2          1          0
 3   (1,0)    0         1          1          1
 4   (1,1)    1         2          0          1
 5   (1,2)    2         0          2          1
 6   (2,0)    0         2          2          2
 7   (2,1)    1         0          1          2
 8   (2,2)    2         1          0          2
```

### Proof of Strength 2

For any pair of columns c₁ < c₂:

**Case 1**: c₁ = 0
- Pair is (j, i + c₂·j)
- For each value v₁ of j, as i ranges over GF(q), i + c₂·j takes all values
- All q² pairs appear exactly once

**Case 2**: c₁, c₂ ≥ 1
- Pair is (i + c₁·j, i + c₂·j)
- Difference: (c₂ - c₁)·j ranges over all of GF(q) as j varies
- For each fixed difference d, there's exactly one j giving that difference
- All pairs appear exactly once

---

## Bush Construction

### Mathematical Foundation

**Reference**: Bush, K.A. (1952). "Orthogonal arrays of index unity"

**Theorem**: For any prime power q and strength t ≥ 2, there exists OA(q^t, t+1, q, t).

### Algorithm

Given GF(q):

1. **Rows**: Index by coefficient vectors (a₀, a₁, ..., a_{t-1}) ∈ GF(q)^t
   - Each row represents polynomial p(x) = Σ aᵢxⁱ
   - Total rows = q^t

2. **Columns**:
   - Columns 0 to t-1: Evaluate p(c) at c = 0, 1, ..., t-1
   - Column t: Leading coefficient a_{t-1} (the "point at infinity")

### Example: OA(8, 4, 2, 3)

For q = 2, t = 3, GF(2) = {0, 1}:

Polynomials: p(x) = a₀ + a₁x + a₂x²

```
Row (a₀,a₁,a₂)  p(0)  p(1)  p(α)  a₂
 0  (0,0,0)       0     0     0    0
 1  (0,0,1)       0     1     α    1
 2  (0,1,0)       0     1     α    0
 3  (0,1,1)       0     0     0    1
 4  (1,0,0)       1     1     1    0
 5  (1,0,1)       1     0    1+α   1
 6  (1,1,0)       1     0    1+α   0
 7  (1,1,1)       1     1     1    1
```

(Note: For GF(2), α = 1, so p(α) = p(1))

### Proof of Strength t

For any t columns c₁, ..., c_t:
- If all columns are evaluation columns: Vandermonde matrix is invertible
- If column t (leading coeff) included: System still has unique solution

The key insight is that specifying t values (either t evaluations or t-1 evaluations plus leading coefficient) uniquely determines a polynomial of degree < t.

### Why Column t is Special

Column t contains the leading coefficient a_{t-1}, not p(t). This represents the "point at infinity" in projective geometry. Without this special handling, the construction would fail because:

1. Any polynomial of degree < t is determined by t evaluation points
2. But we need t+1 columns for an OA of strength t
3. The leading coefficient provides the (t+1)th dimension while maintaining orthogonality

---

## Bose-Bush Construction

### Mathematical Foundation

**Reference**: Bush, K.A. (1952). Extended construction using characteristic 2

**Theorem**: For q = 2^m, there exists OA(2q², 2q+1, q, 2).

### Algorithm (for q = 2)

Uses two "blocks" indexed by b ∈ {0, 1}:

1. **Rows**: Index by (b, i, j) ∈ {0,1} × GF(q) × GF(q)
   - Row = b·q² + i·q + j
   - Total rows = 2q²

2. **Columns**:
   - Column 0: value = j
   - Columns 1 to q: value = i + k·j (Bose pattern)
   - Columns q+1 to 2q: value = (i + k·j) + b (shifted by block index)

### Example: OA(8, 5, 2, 2) for q = 2

```
Row (b,i,j)  Col0  Col1  Col2  Col3  Col4
 0  (0,0,0)   0     0     0     0     0
 1  (0,0,1)   1     1     0     1     0
 2  (0,1,0)   0     1     1     1     1
 3  (0,1,1)   1     0     1     0     1
 4  (1,0,0)   0     0     0     1     1
 5  (1,0,1)   1     1     0     0     1
 6  (1,1,0)   0     1     1     0     0
 7  (1,1,1)   1     0     1     1     0
```

### Proof of Strength 2

The key is ensuring orthogonality between columns c and q+c:

- In block 0: Col[q+c] = Col[c] + 0 = Col[c], giving pairs (v, v)
- In block 1: Col[q+c] = Col[c] + 1 = Col[c] ⊕ 1, giving pairs (v, v⊕1)

Combined: pairs (0,0), (1,1) from block 0; pairs (0,1), (1,0) from block 1.
All 4 pairs appear exactly 2 times in 8 rows.

---

## Hadamard-Sylvester Construction

### Mathematical Foundation

**Reference**: Sylvester, J.J. (1867). "Thoughts on orthogonal matrices"

**Theorem**: For any n = 2^k (k ≥ 2), there exists a Hadamard matrix H_n of order n,
giving OA(n, n-1, 2, 2).

### Algorithm

The Sylvester construction builds Hadamard matrices recursively:

```
H₁ = [1]
H_{2n} = [[H_n,  H_n ],
          [H_n, -H_n]]
```

To convert Hadamard matrix to OA:
1. First row and column are all +1 by construction
2. Delete the first column (all +1s)
3. Map remaining entries: +1 → 0, -1 → 1

### Example: OA(8, 7, 2, 2)

H₈ (normalized):
```
[+  +  +  +  +  +  +  +]
[+  -  +  -  +  -  +  -]
[+  +  -  -  +  +  -  -]
[+  -  -  +  +  -  -  +]
[+  +  +  +  -  -  -  -]
[+  -  +  -  -  +  -  +]
[+  +  -  -  -  -  +  +]
[+  -  -  +  -  +  +  -]
```

After removing column 0 and converting (+→0, -→1):
```
Row  C0 C1 C2 C3 C4 C5 C6
 0    0  0  0  0  0  0  0
 1    1  0  1  0  1  0  1
 2    0  1  1  0  0  1  1
 3    1  1  0  0  1  1  0
 4    0  0  0  1  1  1  1
 5    1  0  1  1  0  1  0
 6    0  1  1  1  1  0  0
 7    1  1  0  1  0  0  1
```

Each column has exactly 4 zeros and 4 ones. ✓

### Properties

- **Balance**: Each column has n/2 zeros and n/2 ones
- **Orthogonality**: Any two columns together show all 4 binary pairs equally
- **Efficiency**: Maximum n-1 factors for n runs

---

## Hadamard-Paley Construction

### Mathematical Foundation

**Reference**: Paley, R.E.A.C. (1933). "On orthogonal matrices"

**Theorem**: For prime p ≡ 3 (mod 4), there exists a Hadamard matrix of order p+1,
giving OA(p+1, p, 2, 2).

### Algorithm

The Paley Type I construction uses quadratic residues:

1. **Legendre Symbol**: For a ∈ Z_p:
   - χ(a) = 0 if a = 0
   - χ(a) = +1 if a is a quadratic residue (∃b: b² ≡ a mod p)
   - χ(a) = -1 if a is a non-residue

2. **Jacobsthal Matrix**: p×p matrix Q where Q[i,j] = χ(j-i)

3. **Raw Hadamard Matrix**:
   ```
   H = [[1,  1,  1, ..., 1 ],
        [-1,                ],
        [-1,    Q + I_p     ],
        [⋮                  ],
        [-1                 ]]
   ```

4. **Normalization**: Negate rows with -1 in first column to get standard form

### Example: OA(8, 7, 2, 2) using p=7

Quadratic residues mod 7: {1, 2, 4} (since 1²=1, 2²=4, 3²=2 mod 7)
Non-residues mod 7: {3, 5, 6}

The normalized Hadamard matrix gives a valid OA after removing the first column.

### Available Paley Primes

| p | Runs (p+1) | Max Factors (p) |
|---|-----------|-----------------|
| 3 | 4 | 3 |
| 7 | 8 | 7 |
| 11 | 12 | 11 |
| 19 | 20 | 19 |
| 23 | 24 | 23 |
| 31 | 32 | 31 |
| 43 | 44 | 43 |
| 47 | 48 | 47 |

### Comparison with Sylvester

- **Sylvester**: Orders 4, 8, 16, 32, 64, ... (powers of 2)
- **Paley**: Orders 4, 8, 12, 20, 24, 32, 44, 48, ... (p+1 for p ≡ 3 mod 4)
- **Together**: Cover orders 4, 8, 12, 16, 20, 24, 32, ... (most multiples of 4)

---

## Addelman-Kempthorne Construction

### Mathematical Foundation

**Reference**: Addelman, S. & Kempthorne, O. (1961). "Some Main-Effect Plans and Orthogonal
Arrays of Strength Two." Annals of Mathematical Statistics, Vol 32, pp 1167-1176.

**Theorem**: For any odd prime power q (3, 5, 7, 9, 11, ...), there exists OA(2q², 2q+1, q, 2).

This construction complements Bose-Bush (which handles powers of 2) by providing the same
2q+1 column efficiency for odd prime powers.

### Algorithm

The array is divided into two blocks of q² rows each.

**Block 1** (rows indexed by (i, j) ∈ GF(q) × GF(q)):
- Column 0: j
- Columns 1 to q-1: i + m·j for m = 1, ..., q-1
- Column q: i
- Columns q+1 to 2q: i² + m·i + j for m = 0, ..., q-1

**Block 2** uses transformation constants (kay, b[], c[], k[]) derived from quadratic
non-residues in GF(q) to ensure orthogonality:
- kay: A quadratic non-residue in GF(q)
- b[m] = (kay - 1) / (kay · 4 · m)
- k[m] = kay · m
- c[m] = m² · (kay - 1) / 4

The second block applies these transformations to create complementary patterns that
ensure all column pairs exhibit perfect orthogonality.

### Example: OA(18, 7, 3, 2) - The L18 Array

For q = 3, this produces 18 runs with up to 7 factors at 3 levels each.
This is the famous L18 Taguchi array, one of the most widely used designs in quality engineering.

```
Block 1 (rows 0-8):  Standard polynomial formulas
Block 2 (rows 9-17): Transformed with kay=2 (the non-residue in GF(3))
```

### Available Arrays

| q | Runs (2q²) | Max Factors (2q+1) | Notes |
|---|-----------|-------------------|-------|
| 3 | 18 | 7 | Classic L18 array |
| 5 | 50 | 11 | |
| 7 | 98 | 15 | |
| 9 | 162 | 19 | Uses GF(3²) |
| 11 | 242 | 23 | |
| 13 | 338 | 27 | |

### Comparison with Bose-Bush

- **Addelman-Kempthorne**: For odd prime powers (3, 5, 7, 9, ...)
- **Bose-Bush**: For powers of 2 (2, 4, 8, 16, ...)
- **Together**: Cover all prime powers for OA(2q², 2q+1, q, 2)

---

## Rao-Hamming Construction (Planned)

### Mathematical Foundation

Uses linear codes over finite fields to construct OAs.

### Algorithm Overview

Given a linear [n, k, d] code over GF(q):
1. Generator matrix G gives OA structure
2. Dual code parity check matrix H defines columns
3. Minimum distance d relates to strength t

### Relationship to Error-Correcting Codes

- OA columns ↔ codewords
- Strength t ↔ minimum distance d ≥ t+1
- Hamming codes give optimal certain parameters

---

## Construction Selection Guide

### Given Parameters (N, k, s, t), Choose:

| Strength t | Levels s | Best Construction |
|------------|----------|-------------------|
| 2 | Any prime power q | Bose: N = q², k ≤ q+1 |
| 2 | Power of 2 | Bose-Bush: N = 2q², k ≤ 2q+1 |
| t ≥ 2 | Any prime power q | Bush: N = q^t, k ≤ t+1 |
| 2 | 2 | Hadamard: N = 4n, k ≤ 4n-1 |
| Any | Mixed | Addelman-Kempthorne |

### Trade-offs

1. **Bose vs Bose-Bush**: Bose-Bush gives more columns (2q+1 vs q+1) but requires power-of-2 levels and doubles runs.

2. **Bush vs Bose**: Bush allows higher strength but is less column-efficient for strength 2.

3. **Hadamard**: Best for binary factors with many runs; very column-efficient.

---

## Implementation Notes

### Galois Field Requirements

All constructions except Hadamard require GF(q) arithmetic:
- Addition, multiplication in GF(q)
- For extension fields (q = p^m), need irreducible polynomial

### Verification

After construction, always verify using `verify_strength()`:
```rust
let oa = constructor.construct(k)?;
let result = verify_strength(&oa, t)?;
assert!(result.is_valid);
```

### Error Handling

Constructors fail fast on invalid parameters:
- `TooManyFactors`: k exceeds maximum for construction
- `InvalidStrength`: t not achievable
- `RequiresPowerOfTwo`: Bose-Bush requires q = 2^m

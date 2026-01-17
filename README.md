# Taguchi: State-of-the-Art Orthogonal Array Library in Rust

[![Crates.io](https://img.shields.io/crates/v/taguchi.svg)](https://crates.io/crates/taguchi)
[![Documentation](https://docs.rs/taguchi/badge.svg)](https://docs.rs/taguchi)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**Taguchi** is a robust, world-class Rust library for constructing and analyzing orthogonal arrays (OAs). Orthogonal arrays are fundamental to Design of Experiments (DOE), Monte Carlo simulation, combinatorial software testing (OATS), and quasi-random sampling.

## Key Features

- **Robust Construction Algorithms**: Includes Bose, Bush, Bose-Bush, Addelman-Kempthorne, Hadamard (Sylvester & Paley), and Rao-Hamming.
- **Mixed-Level Support**: SOTA support for arrays with different levels per factor via level collapsing.
- **Custom Galois Field Arithmetic**: Full control over $GF(q)$ arithmetic for both prime and extension fields with zero dependencies.
- **Statistical Analysis**: Built-in utilities for balance checking, correlation analysis, and Generalized Word Length Pattern (GWLP).
- **Parallel Construction**: High-performance row generation using `rayon` for large-scale experimental designs.
- **DOE Analysis**: Complete Taguchi analysis with main effects, S/N ratios, ANOVA, and optimal settings prediction.
- **Modern API**: Fluent builder pattern with automatic optimal construction selection.

## Quick Start

```rust
use taguchi::OABuilder;

fn main() {
    // Automatically selects the best construction (Bose in this case)
    let oa = OABuilder::new()
        .levels(3)
        .factors(4)
        .strength(2)
        .build()
        .unwrap();

    println!("Runs: {}", oa.runs());       // 9
    println!("Factors: {}", oa.factors()); // 4
    
    // Perform statistical analysis
    let report = oa.balance_report();
    assert!(report.factor_balance.iter().all(|&b| b));
}
```

### Mixed-Level Design

```rust
use taguchi::OABuilder;

// Construct mixed-level OA(16, 2^3 4^1, 2)
let oa = OABuilder::new()
    .mixed_levels(vec![2, 2, 2, 4])
    .strength(2)
    .build()
    .unwrap();

assert_eq!(oa.runs(), 16);
```

### Standard Taguchi Arrays (Catalogue)

If you are familiar with standard Taguchi array names (e.g., L8, L9, L18), you can use the catalogue:

```rust
use taguchi::catalogue::get_by_name;

let l9 = get_by_name("L9").unwrap();
assert_eq!(l9.runs(), 9);
assert_eq!(l9.factors(), 4);
```

## DOE Analysis

Analyze experimental results with the `doe` feature:

```toml
[dependencies]
taguchi = { version = "0.2", features = ["doe"] }
```

```rust
use taguchi::OABuilder;
use taguchi::doe::{analyze, AnalysisConfig, OptimizationType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create L9 orthogonal array
    let oa = OABuilder::new()
        .levels(3)
        .factors(4)
        .strength(2)
        .build()?;

    // Experimental response data (9 runs)
    let response_data = vec![
        vec![85.0], vec![92.0], vec![78.0],
        vec![91.0], vec![88.0], vec![82.0],
        vec![89.0], vec![86.0], vec![94.0],
    ];

    // Run Taguchi analysis
    let config = AnalysisConfig {
        optimization_type: OptimizationType::LargerIsBetter,
        confidence_level: 0.95,
        ..Default::default()
    };

    let result = analyze(&oa, &response_data, &config)?;

    println!("Grand mean: {:.2}", result.grand_mean);
    println!("Optimal levels: {:?}", result.optimal_settings.factor_levels);
    println!("Predicted mean: {:.2}", result.optimal_settings.predicted_mean);

    // ANOVA results
    for entry in &result.anova.entries {
        if !entry.pooled {
            println!("Factor {}: SS={:.2}, F={:.2}, p={:.4}",
                entry.factor_index,
                entry.sum_of_squares,
                entry.f_ratio.unwrap_or(0.0),
                entry.p_value.unwrap_or(1.0));
        }
    }
    Ok(())
}
```

## Performance

Taguchi uses precomputed arithmetic tables for small fields and optimized `ndarray` storage.
Recent optimizations include batch polynomial evaluation and direct table access, yielding **~10x speedups** for common constructions.

For massive arrays, enable the `parallel` feature:

```toml
[dependencies]
taguchi = { version = "0.2", features = ["parallel"] }
```

## Mathematical Background

An orthogonal array $OA(N, k, s, t)$ is an $N \times k$ matrix with entries from a set of $s$ symbols such that in any $N \times t$ subarray, every possible $t$-tuple appears exactly $\lambda = N/s^t$ times.

Taguchi supports:
- **Strength 2**: Main effects are clear of each other.
- **Higher Strength**: Interaction analysis support via Bush construction.
- **Linear Codes**: SOTA Rao-Hamming construction for maximum factor density.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

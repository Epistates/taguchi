//! Basic usage example for the taguchi library.
//!
//! This example demonstrates how to construct and verify orthogonal arrays
//! using the Bose construction.

use taguchi::construct::Bose;
use taguchi::verify_strength;

fn main() {
    println!("Taguchi Library - Basic Usage Example\n");

    // Construct L9 array (9 runs, 4 factors, 3 levels, strength 2)
    println!("Constructing L9 array using Bose construction...");
    let bose3 = Bose::new(3);
    let l9 = bose3.construct(4).expect("Failed to construct L9");

    println!("L9 Array:");
    println!("  Runs: {}", l9.runs());
    println!("  Factors: {}", l9.factors());
    println!("  Levels: {}", l9.levels());
    println!("  Strength: {}", l9.strength());
    println!();

    // Display the array
    println!("Array contents:");
    println!("{}", l9);

    // Verify the array
    println!("Verifying strength-2 property...");
    let result = verify_strength(&l9, 2).expect("Verification failed");
    if result.is_valid {
        println!("✓ Array is a valid strength-2 orthogonal array");
    } else {
        println!("✗ Array failed verification");
        for issue in &result.issues {
            println!("  Issue: {:?}", issue);
        }
    }

    println!();

    // Construct L25 array
    println!("Constructing L25 array using Bose construction...");
    let bose5 = Bose::new(5);
    let l25 = bose5.construct(6).expect("Failed to construct L25");

    println!("L25 Array:");
    println!("  Runs: {}", l25.runs());
    println!("  Factors: {}", l25.factors());
    println!("  Levels: {}", l25.levels());
    println!();

    let result = verify_strength(&l25, 2).expect("Verification failed");
    if result.is_valid {
        println!("✓ L25 is a valid strength-2 orthogonal array");
    }

    println!();

    // Demonstrate prime power support
    println!("Constructing array with GF(4) = GF(2²)...");
    let bose4 = Bose::new(4);
    let l16_4 = bose4.construct(5).expect("Failed to construct array");

    println!("GF(4) Array:");
    println!("  Runs: {}", l16_4.runs());
    println!("  Factors: {}", l16_4.factors());
    println!("  Levels: {} (using GF(2²))", l16_4.levels());

    let result = verify_strength(&l16_4, 2).expect("Verification failed");
    if result.is_valid {
        println!("✓ Valid strength-2 orthogonal array with prime power levels");
    }
}

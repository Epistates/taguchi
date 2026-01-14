//! Benchmarks for Galois field arithmetic.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use taguchi::gf::DynamicGf;

fn bench_gf_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("GF Multiplication");

    for order in [7u32, 11, 13, 17, 19, 23] {
        let gf = DynamicGf::new(order).unwrap();

        group.bench_with_input(BenchmarkId::new("order", order), &gf, |b, gf| {
            let a = gf.element(3);
            let b_elem = gf.element(5);
            b.iter(|| {
                let mut result = a.clone();
                for _ in 0..100 {
                    result = result.mul(b_elem.clone());
                }
                result
            });
        });
    }

    group.finish();
}

fn bench_gf_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("GF Creation");

    for order in [7u32, 11, 13, 17, 19, 23] {
        group.bench_with_input(BenchmarkId::new("order", order), &order, |b, &order| {
            b.iter(|| DynamicGf::new(order).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gf_multiplication, bench_gf_creation);
criterion_main!(benches);

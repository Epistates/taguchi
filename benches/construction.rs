//! Benchmarks for orthogonal array construction.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use taguchi::construct::{Bose, Constructor};

fn bench_bose_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bose Construction");

    for q in [3u32, 5, 7, 11, 13] {
        let bose = Bose::new(q);
        let factors = bose.max_factors();

        group.bench_with_input(
            BenchmarkId::new("q", q),
            &(bose, factors),
            |b, (bose, factors)| {
                b.iter(|| bose.construct(*factors).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_bose_construction);
criterion_main!(benches);

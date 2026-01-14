use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use taguchi::construct::{Bose, Bush, Constructor, RaoHamming};
use taguchi::OABuilder;

fn bench_bose(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bose");

    for q in [3, 5, 7, 11] {
        group.bench_with_input(BenchmarkId::from_parameter(q), &q, |b, &q| {
            let bose = Bose::new(q as u32);
            let factors = (q + 1) as usize;
            b.iter(|| bose.construct(factors).unwrap());
        });
    }
    group.finish();
}

fn bench_bush(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bush_Strength3");

    for q in [3, 5, 7] {
        group.bench_with_input(BenchmarkId::from_parameter(q), &q, |b, &q| {
            let bush = Bush::new(q as u32, 3).unwrap();
            let factors = (q + 1) as usize;
            b.iter(|| bush.construct(factors).unwrap());
        });
    }
    group.finish();
}

fn bench_rao_hamming(c: &mut Criterion) {
    let mut group = c.benchmark_group("RaoHamming");

    // q=2, m=3,4,5...
    for m in [3, 4, 5] {
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, &m| {
            let rh = RaoHamming::new(2, m).unwrap();
            let factors = rh.max_factors();
            b.iter(|| rh.construct(factors).unwrap());
        });
    }
    group.finish();
}

fn bench_builder_auto(c: &mut Criterion) {
    let mut group = c.benchmark_group("Builder_Auto");

    // Compare auto-selection overhead vs direct construction
    // L9: 3 levels, 4 factors, strength 2 (Should use Bose)
    group.bench_function("L9", |b| {
        b.iter(|| {
            OABuilder::new()
                .levels(3)
                .factors(4)
                .strength(2)
                .build()
                .unwrap()
        });
    });

    // L16: 2 levels, 15 factors, strength 2 (Should use Hadamard)
    group.bench_function("L16_Hadamard", |b| {
        b.iter(|| {
            OABuilder::new()
                .levels(2)
                .factors(15)
                .strength(2)
                .build()
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bose,
    bench_bush,
    bench_rao_hamming,
    bench_builder_auto
);
criterion_main!(benches);

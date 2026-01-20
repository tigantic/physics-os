//! Benchmark for FluidElite ZK constraint system
//!
//! Measures actual constraint counts and proof times.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use fluidelite_zk::circuit::config::CircuitConfig;
use fluidelite_zk::field::Q16;
use fluidelite_zk::mpo::MPO;
use fluidelite_zk::mps::MPS;
use fluidelite_zk::ops::{apply_mpo, add_mps, fluidelite_step};

fn bench_mpo_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpo_apply");

    for chi in [4, 8, 16, 32, 64].iter() {
        let mps = MPS::new(16, *chi, 2);
        let mpo = MPO::identity(16, 2);

        group.bench_with_input(
            BenchmarkId::from_parameter(chi),
            chi,
            |b, _| {
                b.iter(|| {
                    let result = apply_mpo(black_box(&mps), black_box(&mpo));
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn bench_mps_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_add");

    for chi in [4, 8, 16, 32, 64].iter() {
        let a = MPS::new(16, *chi, 2);
        let b = MPS::new(16, *chi, 2);

        group.bench_with_input(
            BenchmarkId::from_parameter(chi),
            chi,
            |b, _| {
                b.iter(|| {
                    let result = add_mps(black_box(&a), black_box(&b));
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn bench_fluidelite_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("fluidelite_step");

    for chi in [4, 8, 16, 32, 64].iter() {
        let context = MPS::new(16, *chi, 2);
        let w_hidden = MPO::identity(16, 2);
        let w_input = MPO::identity(16, 2);

        group.bench_with_input(
            BenchmarkId::from_parameter(chi),
            chi,
            |b, chi| {
                b.iter(|| {
                    let result = fluidelite_step(
                        black_box(&context),
                        black_box(42),
                        black_box(&w_hidden),
                        black_box(&w_input),
                        black_box(**chi),
                    );
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn bench_constraint_estimate(c: &mut Criterion) {
    c.bench_function("constraint_estimate", |b| {
        let config = CircuitConfig::production();
        b.iter(|| {
            let count = config.estimate_constraints();
            black_box(count)
        })
    });
}

criterion_group!(
    benches,
    bench_mpo_apply,
    bench_mps_add,
    bench_fluidelite_step,
    bench_constraint_estimate,
);
criterion_main!(benches);

//! Proof generation benchmarks
//!
//! Run with:
//! ```bash
//! cargo bench --features halo2 --bench proof_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "halo2")]
use fluidelite_zk::{
    circuit::config::CircuitConfig,
    field::Q16,
    mpo::MPO,
    mps::MPS,
    prover::FluidEliteProver,
};

#[cfg(feature = "halo2")]
fn bench_proof_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("proof_generation");
    group.sample_size(10); // Proof generation is slow

    // Test different circuit sizes
    for k in [8, 10, 12] {
        let config = CircuitConfig {
            k,
            num_sites: 8,
            chi_max: 16,
            phys_dim: 2,
            vocab_size: 64,
        };

        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
        let prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

        let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

        group.throughput(Throughput::Elements(1)); // 1 proof per iteration

        group.bench_with_input(BenchmarkId::new("prove", format!("k={}", k)), &k, |b, _| {
            b.iter(|| {
                let token = 42u64;
                let proof = prover.prove(black_box(&context), black_box(token));
                black_box(proof)
            })
        });
    }

    group.finish();
}

#[cfg(feature = "halo2")]
fn bench_context_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("context_sizes");
    group.sample_size(10);

    // Test different context sizes (chi_max)
    for chi_max in [8, 16, 32] {
        let config = CircuitConfig {
            k: 10,
            num_sites: 8,
            chi_max,
            phys_dim: 2,
            vocab_size: 64,
        };

        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout_weights = vec![Q16::from_f64(0.1); chi_max * config.vocab_size];
        let prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

        let context = MPS::new(config.num_sites, chi_max, config.phys_dim);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("chi={}", chi_max)),
            &chi_max,
            |b, _| {
                b.iter(|| {
                    let proof = prover.prove(black_box(&context), black_box(42u64));
                    black_box(proof)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "halo2")]
fn bench_site_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("site_counts");
    group.sample_size(10);

    // Test different site counts (L)
    for num_sites in [4, 8, 12] {
        let config = CircuitConfig {
            k: 10,
            num_sites,
            chi_max: 16,
            phys_dim: 2,
            vocab_size: 64,
        };

        let w_hidden = MPO::identity(num_sites, config.phys_dim);
        let w_input = MPO::identity(num_sites, config.phys_dim);
        let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
        let prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

        let context = MPS::new(num_sites, config.chi_max, config.phys_dim);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("L={}", num_sites)),
            &num_sites,
            |b, _| {
                b.iter(|| {
                    let proof = prover.prove(black_box(&context), black_box(42u64));
                    black_box(proof)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "halo2")]
fn bench_mock_prover(c: &mut Criterion) {
    use fluidelite_zk::circuit::halo2_impl::FluidEliteCircuit;
    use halo2_axiom::dev::MockProver;

    let mut group = c.benchmark_group("mock_prover");
    group.sample_size(20);

    let context = MPS::new(4, 2, 2);
    let w_hidden = MPO::identity(4, 2);
    let w_input = MPO::identity(4, 2);
    let readout_weights = vec![Q16::from_f64(0.1); 2 * 16];

    group.bench_function("mock_verify", |b| {
        b.iter(|| {
            let circuit =
                FluidEliteCircuit::new(42, context.clone(), w_hidden.clone(), w_input.clone(), readout_weights.clone());
            let k = 10;
            let prover = MockProver::run(k, &circuit, vec![circuit.public_inputs()]).unwrap();
            let result = prover.verify();
            black_box(result)
        })
    });

    group.finish();
}

#[cfg(feature = "halo2")]
criterion_group!(
    benches,
    bench_proof_generation,
    bench_context_sizes,
    bench_site_counts,
    bench_mock_prover
);

#[cfg(feature = "halo2")]
criterion_main!(benches);

#[cfg(not(feature = "halo2"))]
fn main() {
    println!("Benchmarks require 'halo2' feature. Run with:");
    println!("  cargo bench --features halo2 --bench proof_bench");
}

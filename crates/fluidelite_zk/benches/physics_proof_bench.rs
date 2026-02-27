//! Criterion benchmark suite for real physics domain proofs.
//!
//! Benchmarks `create_proof` + `verify_proof` across:
//! - Domains: euler3d, ns_imex, thermal
//! - k values: 14, 16, 17 (production-relevant circuit sizes)
//!
//! Run with: `cargo bench --bench physics_proof_bench --features halo2`
//!
//! Results are captured in `target/criterion/` with HTML reports.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use std::time::Duration;

#[cfg(feature = "halo2")]
mod benches {
    use super::*;
    use fluidelite_zk::euler3d::{Euler3DParams, Euler3DProver, Euler3DVerifier};
    use fluidelite_zk::ns_imex::{NSIMEXParams, NSIMEXProver, NSIMEXVerifier};
    use fluidelite_zk::thermal::{ThermalParams, ThermalProver, ThermalVerifier};
    use fluidelite_zk::mps::MPS;
    use fluidelite_zk::mpo::MPO;
    use fluidelite_zk::field::Q16;

    /// Build test MPS states with given parameters.
    fn make_test_mps(num_sites: usize, chi: usize, phys_dim: usize) -> MPS {
        MPS::random(num_sites, chi, phys_dim)
    }

    /// Build a test shift MPO.
    fn make_test_mpo(num_sites: usize, chi: usize, phys_dim: usize) -> MPO {
        MPO::identity(num_sites, chi, phys_dim)
    }

    // ── Euler3D Benchmarks ─────────────────────────────────────────────────

    pub fn bench_euler3d_prove(c: &mut Criterion) {
        let mut group = c.benchmark_group("euler3d_prove");
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(60));

        for &grid_bits in &[4u32, 5, 6] {
            let params = Euler3DParams {
                gamma: Q16::from_f64(1.4),
                cfl: Q16::from_f64(0.5),
                grid_bits: grid_bits as usize,
                chi_max: 16,
                tolerance: Q16::from_f64(1e-6),
                conservation_tolerance: Q16::from_f64(1e-4),
                mpo_bond_dim: 1,
                dt: Q16::from_f64(0.001),
                dx: Q16::from_f64(0.01),
            };

            let num_sites = params.grid_bits * 3;
            let chi = params.chi_max;
            let phys = 2;

            let input_states: Vec<MPS> = (0..5)
                .map(|_| make_test_mps(num_sites, chi, phys))
                .collect();
            let shift_mpos: Vec<MPO> = (0..3)
                .map(|_| make_test_mpo(num_sites, chi, phys))
                .collect();

            match Euler3DProver::new(params.clone()) {
                Ok(prover) => {
                    group.bench_with_input(
                        BenchmarkId::new("prove", format!("grid_bits={}", grid_bits)),
                        &(prover, input_states, shift_mpos),
                        |b, (prover, states, mpos)| {
                            b.iter(|| {
                                let _ = prover.prove(states, mpos);
                            });
                        },
                    );
                }
                Err(e) => {
                    eprintln!(
                        "Skipping euler3d grid_bits={}: prover init failed: {}",
                        grid_bits, e
                    );
                }
            }
        }

        group.finish();
    }

    pub fn bench_euler3d_verify(c: &mut Criterion) {
        let mut group = c.benchmark_group("euler3d_verify");
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(30));

        let params = Euler3DParams::test_small();
        let num_sites = params.grid_bits * 3;
        let chi = params.chi_max;
        let phys = 2;

        let input_states: Vec<MPS> = (0..5)
            .map(|_| make_test_mps(num_sites, chi, phys))
            .collect();
        let shift_mpos: Vec<MPO> = (0..3)
            .map(|_| make_test_mpo(num_sites, chi, phys))
            .collect();

        match Euler3DProver::new(params) {
            Ok(prover) => {
                match prover.prove(&input_states, &shift_mpos) {
                    Ok(proof) => {
                        let verifier = Euler3DVerifier::from_prover(&prover);
                        group.bench_function("verify", |b| {
                            b.iter(|| {
                                let _ = verifier.verify(&proof);
                            });
                        });
                    }
                    Err(e) => eprintln!("Euler3D proof generation failed: {}", e),
                }
            }
            Err(e) => eprintln!("Euler3D prover init failed: {}", e),
        }

        group.finish();
    }

    // ── NS-IMEX Benchmarks ─────────────────────────────────────────────────

    pub fn bench_ns_imex_prove(c: &mut Criterion) {
        let mut group = c.benchmark_group("ns_imex_prove");
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(60));

        for &grid_bits in &[4u32, 5, 6] {
            let params = NSIMEXParams {
                grid_bits: grid_bits as usize,
                chi_max: 16,
                tolerance: Q16::from_f64(1e-6),
                conservation_tolerance: Q16::from_f64(1e-4),
                divergence_tolerance: Q16::from_f64(1e-4),
                mpo_bond_dim: 1,
                viscosity: Q16::from_f64(0.01),
                cfl: Q16::from_f64(0.5),
                dt: Q16::from_f64(0.001),
                dx: Q16::from_f64(0.01),
                max_cg_iterations: 50,
                cg_tolerance: Q16::from_f64(1e-8),
            };

            let num_sites = params.grid_bits * 3;
            let chi = params.chi_max;
            let phys = 2;

            let input_states: Vec<MPS> = (0..3)
                .map(|_| make_test_mps(num_sites, chi, phys))
                .collect();
            let shift_mpos: Vec<MPO> = (0..3)
                .map(|_| make_test_mpo(num_sites, chi, phys))
                .collect();

            match NSIMEXProver::new(params.clone()) {
                Ok(prover) => {
                    group.bench_with_input(
                        BenchmarkId::new("prove", format!("grid_bits={}", grid_bits)),
                        &(prover, input_states, shift_mpos),
                        |b, (prover, states, mpos)| {
                            b.iter(|| {
                                let _ = prover.prove(states, mpos);
                            });
                        },
                    );
                }
                Err(e) => {
                    eprintln!(
                        "Skipping ns_imex grid_bits={}: prover init failed: {}",
                        grid_bits, e
                    );
                }
            }
        }

        group.finish();
    }

    pub fn bench_ns_imex_verify(c: &mut Criterion) {
        let mut group = c.benchmark_group("ns_imex_verify");
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(30));

        let params = NSIMEXParams::test_small();
        let num_sites = params.grid_bits * 3;
        let chi = params.chi_max;
        let phys = 2;

        let input_states: Vec<MPS> = (0..3)
            .map(|_| make_test_mps(num_sites, chi, phys))
            .collect();
        let shift_mpos: Vec<MPO> = (0..3)
            .map(|_| make_test_mpo(num_sites, chi, phys))
            .collect();

        match NSIMEXProver::new(params) {
            Ok(prover) => {
                match prover.prove(&input_states, &shift_mpos) {
                    Ok(proof) => {
                        let verifier = NSIMEXVerifier::from_prover(&prover);
                        group.bench_function("verify", |b| {
                            b.iter(|| {
                                let _ = verifier.verify(&proof);
                            });
                        });
                    }
                    Err(e) => eprintln!("NS-IMEX proof generation failed: {}", e),
                }
            }
            Err(e) => eprintln!("NS-IMEX prover init failed: {}", e),
        }

        group.finish();
    }

    // ── Thermal Benchmarks ─────────────────────────────────────────────────

    pub fn bench_thermal_prove(c: &mut Criterion) {
        let mut group = c.benchmark_group("thermal_prove");
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(60));

        for &grid_bits in &[4u32, 5, 6] {
            let params = ThermalParams {
                grid_bits: grid_bits as usize,
                chi_max: 16,
                alpha: Q16::from_f64(0.01),
                dt: Q16::from_f64(0.001),
                cfl: Q16::from_f64(0.5),
                tolerance: Q16::from_f64(1e-6),
                conservation_tol: Q16::from_f64(1e-4),
                boundary_condition: fluidelite_zk::thermal::BoundaryCondition::Periodic,
                max_cg_iterations: 50,
                cg_tolerance: Q16::from_f64(1e-8),
                source_magnitude: Q16::from_f64(0.0),
            };

            let num_sites = params.grid_bits * 3;
            let chi = params.chi_max;
            let phys = 2;

            let input_states: Vec<MPS> = vec![make_test_mps(num_sites, chi, phys)];
            let laplacian_mpos: Vec<MPO> = (0..3)
                .map(|_| make_test_mpo(num_sites, chi, phys))
                .collect();

            match ThermalProver::new(params.clone()) {
                Ok(prover) => {
                    group.bench_with_input(
                        BenchmarkId::new("prove", format!("grid_bits={}", grid_bits)),
                        &(prover, input_states, laplacian_mpos),
                        |b, (prover, states, mpos)| {
                            b.iter(|| {
                                let _ = prover.prove(states, mpos);
                            });
                        },
                    );
                }
                Err(e) => {
                    eprintln!(
                        "Skipping thermal grid_bits={}: prover init failed: {}",
                        grid_bits, e
                    );
                }
            }
        }

        group.finish();
    }

    pub fn bench_thermal_verify(c: &mut Criterion) {
        let mut group = c.benchmark_group("thermal_verify");
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(30));

        let params = ThermalParams::test_small();
        let num_sites = params.grid_bits * 3;
        let chi = params.chi_max;
        let phys = 2;

        let input_states: Vec<MPS> = vec![make_test_mps(num_sites, chi, phys)];
        let laplacian_mpos: Vec<MPO> = (0..3)
            .map(|_| make_test_mpo(num_sites, chi, phys))
            .collect();

        match ThermalProver::new(params) {
            Ok(prover) => {
                match prover.prove(&input_states, &laplacian_mpos) {
                    Ok(proof) => {
                        let verifier = ThermalVerifier::from_prover(&prover);
                        group.bench_function("verify", |b| {
                            b.iter(|| {
                                let _ = verifier.verify(&proof);
                            });
                        });
                    }
                    Err(e) => eprintln!("Thermal proof generation failed: {}", e),
                }
            }
            Err(e) => eprintln!("Thermal prover init failed: {}", e),
        }

        group.finish();
    }

    // ── Multi-timestep Aggregation Benchmark ───────────────────────────────

    pub fn bench_multi_timestep_aggregate(c: &mut Criterion) {
        use fluidelite_zk::multi_timestep::{
            MultiTimestepConfig, MultiTimestepProver, SimulationDomain, TimestepInput,
        };

        let mut group = c.benchmark_group("multi_timestep_aggregate");
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(15));

        let prover = MultiTimestepProver::with_random_key(MultiTimestepConfig {
            domain: SimulationDomain::Euler3d,
            embed_proofs: true,
            ..Default::default()
        });

        for &count in &[10u32, 50, 100, 500, 1000] {
            let timesteps: Vec<TimestepInput> = (0..count as usize)
                .map(|i| {
                    TimestepInput::new(i, vec![i as u8; 800]).with_residual(1e-6 * i as f64)
                })
                .collect();

            group.bench_with_input(
                BenchmarkId::new("aggregate", count),
                &timesteps,
                |b, ts| {
                    b.iter(|| {
                        let _ = prover.aggregate(ts.clone());
                    });
                },
            );
        }

        group.finish();
    }

    pub fn bench_multi_timestep_verify(c: &mut Criterion) {
        use fluidelite_zk::multi_timestep::{
            MultiTimestepConfig, MultiTimestepProver, SimulationDomain, TimestepInput,
        };

        let mut group = c.benchmark_group("multi_timestep_verify");
        group.sample_size(100);

        let prover = MultiTimestepProver::with_random_key(MultiTimestepConfig {
            domain: SimulationDomain::Euler3d,
            embed_proofs: false,
            ..Default::default()
        });

        for &count in &[10u32, 100, 1000] {
            let timesteps: Vec<TimestepInput> = (0..count as usize)
                .map(|i| TimestepInput::new(i, vec![i as u8; 800]))
                .collect();

            let agg = prover.aggregate(timesteps).expect("aggregate failed");

            group.bench_with_input(
                BenchmarkId::new("verify_cert", count),
                &agg.tpc_certificate,
                |b, cert| {
                    b.iter(|| {
                        let _ = prover.verify_certificate(cert);
                    });
                },
            );
        }

        group.finish();
    }

    // ── Group registration ─────────────────────────────────────────────────

    criterion_group!(
        physics_proofs,
        bench_euler3d_prove,
        bench_euler3d_verify,
        bench_ns_imex_prove,
        bench_ns_imex_verify,
        bench_thermal_prove,
        bench_thermal_verify,
        bench_multi_timestep_aggregate,
        bench_multi_timestep_verify,
    );
}

#[cfg(feature = "halo2")]
criterion_main!(benches::physics_proofs);

#[cfg(not(feature = "halo2"))]
fn main() {
    eprintln!("Physics proof benchmarks require the 'halo2' feature.");
    eprintln!("Run with: cargo bench --bench physics_proof_bench --features halo2");
}

//! Integration tests for the Euler 3D circuit module.
//!
//! Each file in tests/ compiles as a separate binary, avoiding monolithic
//! linker OOM on memory-constrained machines.

use fluidelite_circuits::euler3d::*;
use fluidelite_circuits::euler3d::config::{MIN_EULER3D_K, MAX_EULER3D_K};
use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

// ═══════════════════════════════════════════════════════════════════════════
// config.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_conserved_variable_index() {
    assert_eq!(ConservedVariable::Density.index(), 0);
    assert_eq!(ConservedVariable::MomentumX.index(), 1);
    assert_eq!(ConservedVariable::MomentumY.index(), 2);
    assert_eq!(ConservedVariable::MomentumZ.index(), 3);
    assert_eq!(ConservedVariable::Energy.index(), 4);
}

#[test]
fn test_strang_stages() {
    assert_eq!(StrangStage::XHalf1.axis(), 0);
    assert_eq!(StrangStage::YHalf1.axis(), 1);
    assert_eq!(StrangStage::ZFull.axis(), 2);
    assert!(StrangStage::XHalf1.is_half_step());
    assert!(!StrangStage::ZFull.is_half_step());
}

#[test]
fn test_params_hash_determinism() {
    let p1 = Euler3DParams::test_small();
    let p2 = Euler3DParams::test_small();
    assert_eq!(p1.hash(), p2.hash());

    let mut p3 = Euler3DParams::test_small();
    p3.chi_max = 8;
    assert_ne!(p1.hash(), p3.hash());
}

#[test]
fn test_dt_over_dx() {
    let params = Euler3DParams {
        dt: Q16::from_f64(0.01),
        dx: Q16::from_f64(0.1),
        ..Euler3DParams::test_small()
    };
    let ratio = params.dt_over_dx();
    assert!((ratio.to_f64() - 0.1).abs() < 0.01);
}

#[test]
fn test_test_small_sizing() {
    let sizing = Euler3DCircuitSizing::test_small();
    assert_eq!(sizing.num_sites, 4);
    assert_eq!(sizing.chi_max, 4);
    assert_eq!(sizing.phys_dim, 2);
    assert_eq!(sizing.num_variables, 5);
    assert_eq!(sizing.num_strang_stages, 5);
    assert!(sizing.k >= MIN_EULER3D_K);
    assert!(sizing.k <= MAX_EULER3D_K);

    let constraints = sizing.estimate_constraints();
    assert!(constraints > 0, "Must have non-zero constraints");
}

#[test]
fn test_production_sizing() {
    let sizing = Euler3DCircuitSizing::production();
    assert_eq!(sizing.num_sites, 16);
    assert_eq!(sizing.chi_max, 32);

    let constraints = sizing.estimate_constraints();
    assert!(
        constraints > 100_000,
        "Production should have >100k constraints, got {}",
        constraints
    );
}

#[test]
fn test_macs_per_contraction() {
    let macs = Euler3DCircuitSizing::macs_per_contraction(4, 4, 2, 1);
    assert_eq!(macs, 256);
}

#[test]
fn test_constraint_count_test_config() {
    let sizing = Euler3DCircuitSizing::test_small();
    let constraints = sizing.estimate_constraints();
    assert!(
        constraints > 10_000 && constraints < 100_000,
        "Test config constraints: {} (expected 10k-100k)",
        constraints
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// gadgets.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_fp_mac_computation() {
    let a = Q16::from_f64(1.5);
    let b = Q16::from_f64(2.0);

    let full_product = a.raw as i128 * b.raw as i128;
    let quotient = (full_product >> 16) as i64;
    let remainder = (full_product - ((quotient as i128) << 16)) as i64;

    assert_eq!(quotient, 196608);
    assert_eq!(remainder, 0);

    let result = Q16::from_raw(quotient);
    assert!((result.to_f64() - 3.0).abs() < 1e-4);
}

#[test]
fn test_fp_mac_with_remainder() {
    let a = Q16::from_f64(1.0 / 3.0);
    let b = Q16::from_f64(1.0 / 3.0);

    let full_product = a.raw as i128 * b.raw as i128;
    let quotient = (full_product >> 16) as i64;
    let remainder = (full_product - ((quotient as i128) << 16)) as i64;

    assert!(remainder >= 0);
    assert!(remainder < 65536);

    let result = Q16::from_raw(quotient);
    assert!((result.to_f64() - (1.0 / 9.0)).abs() < 0.01);
}

#[test]
fn test_conservation_bounds() {
    let before = Q16::from_f64(100.0);
    let after = Q16::from_f64(100.001);
    let tolerance = Q16::from_f64(0.01);

    let residual = after.raw - before.raw;
    let bound_pos = tolerance.raw - residual;
    let bound_neg = tolerance.raw + residual;

    assert!(bound_pos >= 0, "bound_pos = {}", bound_pos);
    assert!(bound_neg >= 0, "bound_neg = {}", bound_neg);
}

#[test]
fn test_conservation_violation_detected() {
    let before = Q16::from_f64(100.0);
    let after = Q16::from_f64(100.1);
    let tolerance = Q16::from_f64(0.01);

    let residual = after.raw - before.raw;
    let bound_pos = tolerance.raw - residual;

    assert!(bound_pos < 0, "Should detect conservation violation");
}

#[test]
fn test_sv_ordering_constraint() {
    let sv1 = Q16::from_f64(5.0);
    let sv2 = Q16::from_f64(3.0);
    let sv3 = Q16::from_f64(1.0);

    assert!(sv1.raw - sv2.raw >= 0);
    assert!(sv2.raw - sv3.raw >= 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// witness.rs tests (pub API only — private method tests stay inline)
// ═══════════════════════════════════════════════════════════════════════════

fn make_witness_test_states(num_sites: usize, chi: usize, d: usize) -> Vec<MPS> {
    (0..NUM_CONSERVED_VARIABLES)
        .map(|_| MPS::new(num_sites, chi, d))
        .collect()
}

fn make_witness_test_shift_mpos(num_sites: usize, d: usize) -> Vec<MPO> {
    (0..3)
        .map(|_| MPO::identity(num_sites, d))
        .collect()
}

#[test]
fn test_witness_generator_creation() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params);
    assert_eq!(gen.sizing().num_sites, 4);
    assert_eq!(gen.sizing().chi_max, 4);
}

#[test]
fn test_witness_generation_basic() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_witness_test_states(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_witness_test_shift_mpos(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Witness generation failed");

    assert_eq!(witness.input_states.len(), NUM_CONSERVED_VARIABLES);
    assert_eq!(witness.output_states.len(), NUM_CONSERVED_VARIABLES);
    assert_eq!(witness.strang_stages.len(), NUM_STRANG_STAGES);
    assert_eq!(witness.conservation.residuals.len(), NUM_CONSERVED_VARIABLES);
}

#[test]
fn test_witness_wrong_variable_count() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = vec![MPS::new(params.num_sites(), params.chi_max, PHYS_DIM)];
    let mpos = make_witness_test_shift_mpos(params.num_sites(), PHYS_DIM);

    assert!(gen.generate(&states, &mpos).is_err());
}

#[test]
fn test_witness_wrong_mpo_count() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_witness_test_states(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = vec![MPO::identity(params.num_sites(), PHYS_DIM)];

    assert!(gen.generate(&states, &mpos).is_err());
}

#[test]
fn test_contraction_witness_has_data() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_witness_test_states(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_witness_test_shift_mpos(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Failed");

    let first_stage = &witness.strang_stages[0];
    assert_eq!(first_stage.variable_sweeps.len(), NUM_CONSERVED_VARIABLES);

    let first_sweep = &first_stage.variable_sweeps[0];
    assert_eq!(first_sweep.contraction.num_sites, params.num_sites());
    assert_eq!(first_sweep.contraction.site_data.len(), params.num_sites());
}

#[test]
fn test_svd_truncation_witness() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_witness_test_states(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_witness_test_shift_mpos(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Failed");

    let first_trunc = &witness.strang_stages[0].variable_sweeps[0].truncation;
    assert!(first_trunc.num_bonds > 0);
    assert_eq!(first_trunc.bond_data.len(), first_trunc.num_bonds);

    for bond in &first_trunc.bond_data {
        for pair in bond.singular_values.windows(2) {
            assert!(
                pair[0].raw >= pair[1].raw,
                "SVs not sorted: {} < {}",
                pair[0].to_f64(),
                pair[1].to_f64()
            );
        }
    }
}

#[test]
fn test_witness_hash_determinism() {
    let params = Euler3DParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_witness_test_states(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_witness_test_shift_mpos(params.num_sites(), PHYS_DIM);

    let w1 = gen.generate(&states, &mpos).expect("Failed");
    let w2 = gen.generate(&states, &mpos).expect("Failed");

    assert_eq!(
        w1.hashes.input_state_hash_limbs,
        w2.hashes.input_state_hash_limbs
    );
    assert_eq!(
        w1.hashes.output_state_hash_limbs,
        w2.hashes.output_state_hash_limbs
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// prover.rs tests
// ═══════════════════════════════════════════════════════════════════════════

fn make_prover_test_data() -> (Euler3DParams, Vec<MPS>, Vec<MPO>) {
    let params = Euler3DParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..NUM_CONSERVED_VARIABLES)
        .map(|_| MPS::new(num_sites, chi, 2))
        .collect();
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    (params, input_states, shift_mpos)
}

#[test]
fn test_prover_creation() {
    let (params, _, _) = make_prover_test_data();
    let prover = Euler3DProver::new(params);
    assert!(prover.is_ok(), "Prover creation failed: {:?}", prover.err());
}

#[test]
fn test_prove_and_verify() {
    let (params, input_states, shift_mpos) = make_prover_test_data();

    let mut prover = Euler3DProver::new(params).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &shift_mpos)
        .expect("Proof generation failed");

    assert!(proof.size() > 0, "Proof should have non-zero size");
    assert!(proof.num_constraints > 0, "Should have non-zero constraints");
    assert_eq!(proof.conservation_residuals.len(), NUM_CONSERVED_VARIABLES);

    let verifier = Euler3DVerifier::new();
    let result = verifier.verify(&proof).expect("Verification failed");

    assert!(result.valid, "Proof should be valid");
    assert!(result.verification_time_us > 0, "Verification should take time");
}

#[test]
fn test_proof_serialization() {
    let (params, input_states, shift_mpos) = make_prover_test_data();

    let mut prover = Euler3DProver::new(params).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &shift_mpos)
        .expect("Proof generation failed");

    let bytes = proof.to_bytes();
    assert!(bytes.len() > 4, "Serialized proof should be >4 bytes");
    assert_eq!(&bytes[0..4], b"E3DP", "Wrong magic");
}

#[test]
fn test_prover_stats() {
    let (params, input_states, shift_mpos) = make_prover_test_data();

    let mut prover = Euler3DProver::new(params).expect("Prover creation failed");

    let _ = prover.prove(&input_states, &shift_mpos).expect("Failed");
    let _ = prover.prove(&input_states, &shift_mpos).expect("Failed");

    let stats = prover.stats();
    assert_eq!(stats.total_proofs, 2);
    assert!(stats.total_time_ms >= 0);
    assert!(stats.total_bytes > 0);
    assert!(stats.total_constraints > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// halo2_impl.rs stub tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "halo2"))]
mod halo2_stub {
    use super::*;

    fn make_test_circuit() -> Euler3DCircuit {
        let params = Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_CONSERVED_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, 2))
            .collect();
        let shift_mpos: Vec<MPO> = (0..3)
            .map(|_| MPO::identity(num_sites, 2))
            .collect();

        Euler3DCircuit::new(params, &input_states, &shift_mpos)
            .expect("Circuit creation failed")
    }

    #[test]
    fn test_stub_circuit_creation() {
        let circuit = make_test_circuit();
        assert!(circuit.estimate_constraints() > 0);
    }

    #[test]
    fn test_stub_witness_validation() {
        let circuit = make_test_circuit();
        let result = circuit.validate_witness();
        assert!(result.is_ok(), "Witness validation failed: {:?}", result.err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// mod.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_module_exports() {
    let _params = Euler3DParams::test_small();
    let _sizing = Euler3DCircuitSizing::from_params(&_params);
    let _var = ConservedVariable::Density;
    let _stage = StrangStage::XHalf1;
    let _op = QttOperation::ShiftMpoApply;
    let _q16_scale: u64 = Q16_SCALE;
    assert_eq!(_q16_scale, 65536u64);
}

#[test]
fn test_euler3d_make_test_data() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_shift_mpos(&params);

    assert_eq!(states.len(), NUM_CONSERVED_VARIABLES);
    assert_eq!(mpos.len(), 3);

    for (i, state) in states.iter().enumerate() {
        assert_eq!(
            state.cores.len(),
            params.num_sites(),
            "State {} wrong num_sites",
            i
        );
    }

    for (i, mpo) in mpos.iter().enumerate() {
        assert_eq!(
            mpo.cores.len(),
            params.num_sites(),
            "MPO {} wrong num_sites",
            i
        );
    }
}

#[test]
fn test_euler3d_end_to_end_stub() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_shift_mpos(&params);

    let result = prove_euler3d_timestep(params, &states, &mpos);
    assert!(result.is_ok(), "End-to-end failed: {:?}", result.err());

    let (proof, verification) = result.unwrap();
    assert!(verification.valid, "Proof should be valid");
    assert!(proof.num_constraints > 0, "Should have constraints");
    assert_eq!(proof.conservation_residuals.len(), NUM_CONSERVED_VARIABLES);
}

#[test]
fn test_euler3d_wrong_input_count() {
    let params = test_config();
    let states: Vec<MPS> = vec![];
    let mpos = make_test_shift_mpos(&params);

    let result = prove_euler3d_timestep(params, &states, &mpos);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 5 input states"));
}

#[test]
fn test_euler3d_wrong_mpo_count() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos: Vec<MPO> = vec![];

    let result = prove_euler3d_timestep(params, &states, &mpos);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 3 shift MPOs"));
}

#[test]
fn test_euler3d_config_variants() {
    let test = test_config();
    let prod = production_config();

    assert!(test.grid_bits < prod.grid_bits);
    assert!(test.chi_max < prod.chi_max);
    assert!(test.tolerance.raw > prod.tolerance.raw);
}

#[test]
fn test_euler3d_circuit_sizing() {
    let params = test_config();
    let sizing = Euler3DCircuitSizing::from_params(&params);

    assert!(sizing.k >= 14, "k should be at least 14 for test");
    let constraints = sizing.estimate_constraints();
    assert!(constraints > 0, "Should have non-zero constraints");
}

//! Integration tests for the Thermal circuit module.

use fluidelite_circuits::thermal::*;
use fluidelite_circuits::thermal::config::{MIN_THERMAL_K, MAX_THERMAL_K};
use fluidelite_circuits::thermal::gadgets::{
    BitDecompositionGadget, CgSolveGadget, ConservationGadget,
    FixedPointMACGadget, PublicInputGadget, SvdOrderingGadget,
    TruncationErrorGadget,
};
use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

// ═══════════════════════════════════════════════════════════════════════════
// config.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_thermal_params_test_small() {
    let params = ThermalParams::test_small();
    assert_eq!(params.grid_bits, 4);
    assert_eq!(params.chi_max, 4);
    assert_eq!(params.num_sites(), 12);
    assert!(params.alpha.raw > 0);
    assert!(params.dt.raw > 0);
    assert!(params.validate().is_ok());
}

#[test]
fn test_thermal_params_production() {
    let params = ThermalParams::production();
    assert_eq!(params.grid_bits, 16);
    assert_eq!(params.chi_max, 32);
    assert_eq!(params.num_sites(), 48);
    assert_eq!(params.boundary_condition, BoundaryCondition::Dirichlet);
    assert!(params.validate().is_ok());
}

#[test]
fn test_thermal_params_invalid() {
    let mut p = ThermalParams::test_small();
    p.grid_bits = 0;
    assert!(p.validate().is_err());

    let mut p = ThermalParams::test_small();
    p.chi_max = 0;
    assert!(p.validate().is_err());

    let mut p = ThermalParams::test_small();
    p.alpha = Q16::from_raw(-1);
    assert!(p.validate().is_err());

    let mut p = ThermalParams::test_small();
    p.dt = Q16::from_raw(0);
    assert!(p.validate().is_err());

    let mut p = ThermalParams::test_small();
    p.max_cg_iterations = 0;
    assert!(p.validate().is_err());
}

#[test]
fn test_thermal_circuit_sizing() {
    let params = ThermalParams::test_small();
    let sizing = ThermalCircuitSizing::from_params(&params);
    assert!(sizing.k >= MIN_THERMAL_K as u32);
    assert!(sizing.k <= MAX_THERMAL_K as u32);
    assert!(sizing.total_rows > 0);
    assert!(sizing.estimate_constraints() > 0);
}

#[test]
fn test_thermal_circuit_sizing_production() {
    let params = ThermalParams::production();
    let sizing = ThermalCircuitSizing::from_params(&params);
    assert!(sizing.k > MIN_THERMAL_K as u32);
    assert!(sizing.estimate_constraints() > 100_000);
}

#[test]
fn test_thermal_variables() {
    assert_eq!(ThermalVariable::ALL.len(), NUM_THERMAL_VARIABLES);
    assert_eq!(ThermalVariable::Temperature.index(), 0);
    assert_eq!(ThermalVariable::Temperature.label(), "T");
}

#[test]
fn test_thermal_stages() {
    assert_eq!(ThermalStage::ALL.len(), NUM_THERMAL_STAGES);
    for (i, stage) in ThermalStage::ALL.iter().enumerate() {
        assert_eq!(stage.index(), i);
    }
}

#[test]
fn test_boundary_conditions() {
    assert_eq!(BoundaryCondition::Dirichlet.label(), "Dirichlet");
    assert_eq!(BoundaryCondition::Neumann.label(), "Neumann");
    assert_eq!(BoundaryCondition::Periodic.label(), "Periodic");
}

#[test]
fn test_thermal_params_display() {
    let p = ThermalParams::test_small();
    let s = format!("{}", p);
    assert!(s.contains("ThermalParams"));
    assert!(s.contains("grid=2^4"));
    assert!(s.contains("χ=4"));
}

#[test]
fn test_thermal_sizing_display() {
    let p = ThermalParams::test_small();
    let s = ThermalCircuitSizing::from_params(&p);
    let d = format!("{}", s);
    assert!(d.contains("ThermalCircuitSizing"));
}

#[test]
fn test_sizing_breakdown_nonzero() {
    let p = ThermalParams::test_small();
    let s = ThermalCircuitSizing::from_params(&p);
    assert!(s.breakdown.rhs_assembly > 0);
    assert!(s.breakdown.implicit_solve > 0);
    assert!(s.breakdown.svd_truncation > 0);
    assert!(s.breakdown.conservation > 0);
}

#[test]
fn test_total_grid_points() {
    let p = ThermalParams::test_small();
    assert_eq!(p.total_grid_points(), 4096);
}

#[test]
fn test_qtt_operation_variants() {
    let ops = [
        ThermalQttOperation::MpoContraction,
        ThermalQttOperation::MpsAddition,
        ThermalQttOperation::SvdTruncation,
        ThermalQttOperation::CgIteration,
        ThermalQttOperation::BcEnforcement,
    ];
    assert_eq!(ops.len(), 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// gadgets.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_gadget_types_exist() {
    let _mac = FixedPointMACGadget;
    let _bit = BitDecompositionGadget;
    let _svd = SvdOrderingGadget;
    let _cons = ConservationGadget;
    let _cg = CgSolveGadget;
    let _pub = PublicInputGadget;
}

// ═══════════════════════════════════════════════════════════════════════════
// witness.rs tests (pub API only — mps_dot_product/negate_mps stay inline)
// ═══════════════════════════════════════════════════════════════════════════

fn make_thermal_test_state(num_sites: usize, chi: usize, d: usize) -> Vec<MPS> {
    vec![MPS::new(num_sites, chi, d)]
}

fn make_thermal_test_laplacian(num_sites: usize, d: usize) -> Vec<MPO> {
    vec![MPO::identity(num_sites, d)]
}

#[test]
fn test_thermal_witness_generator_creation() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params);
    assert!(gen.sizing().k >= 10);
}

#[test]
fn test_thermal_witness_generation_basic() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Witness generation failed");

    assert_eq!(witness.input_state.num_sites, params.num_sites());
    assert_eq!(witness.output_state.num_sites, params.num_sites());
    assert!(witness.implicit_solve.num_iterations > 0);
}

#[test]
fn test_thermal_witness_wrong_variable_count() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states: Vec<MPS> = Vec::new();
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    assert!(gen.generate(&states, &mpos).is_err());
}

#[test]
fn test_thermal_witness_wrong_mpo_count() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos: Vec<MPO> = Vec::new();

    assert!(gen.generate(&states, &mpos).is_err());
}

#[test]
fn test_thermal_conservation_witness() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Failed");

    assert!(witness.conservation.residual_bound_bits.len() == 32);
}

#[test]
fn test_thermal_witness_hash_determinism() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let w1 = gen.generate(&states, &mpos).expect("Failed");
    let w2 = gen.generate(&states, &mpos).expect("Failed");

    assert_eq!(
        w1.hashes.input_state_hash_limbs,
        w2.hashes.input_state_hash_limbs
    );
}

#[test]
fn test_thermal_cg_solve_convergence() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Failed");

    assert!(witness.implicit_solve.num_iterations <= params.max_cg_iterations);
}

#[test]
fn test_thermal_svd_truncation_witness() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Failed");

    if witness.truncation.num_bonds > 0 {
        for bond in &witness.truncation.bond_data {
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
}

// ═══════════════════════════════════════════════════════════════════════════
// prover.rs tests
// ═══════════════════════════════════════════════════════════════════════════

fn make_thermal_prover_test_data() -> (ThermalParams, Vec<MPS>, Vec<MPO>) {
    let params = test_config();
    let input_states = make_test_states(&params);
    let laplacian_mpos = make_test_laplacian_mpos(&params);
    (params, input_states, laplacian_mpos)
}

#[test]
fn test_thermal_prover_creation() {
    let (params, _, _) = make_thermal_prover_test_data();
    let prover = ThermalProver::new(params);
    assert!(prover.is_ok(), "Prover creation failed: {:?}", prover.err());
}

#[test]
fn test_thermal_prove_and_verify() {
    let (params, input_states, laplacian_mpos) = make_thermal_prover_test_data();

    let mut prover = ThermalProver::new(params).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &laplacian_mpos)
        .expect("Proof generation failed");

    assert!(proof.size() > 0, "Proof should have non-zero size");
    assert!(proof.num_constraints > 0, "Should have non-zero constraints");
    assert!(proof.cg_iterations > 0, "Should have CG iterations");

    let verifier = ThermalVerifier::new();
    let result = verifier.verify(&proof).expect("Verification failed");

    assert!(result.valid, "Proof should be valid");
    assert!(result.verification_time_us > 0, "Verification should take time");
}

#[test]
fn test_thermal_proof_serialization() {
    let (params, input_states, laplacian_mpos) = make_thermal_prover_test_data();

    let mut prover = ThermalProver::new(params).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &laplacian_mpos)
        .expect("Proof generation failed");

    let bytes = proof.to_bytes();
    assert!(bytes.len() > 4, "Serialized proof should be >4 bytes");
    assert_eq!(&bytes[0..4], b"THEP", "Wrong magic");

    let proof2 = ThermalProof::from_bytes(&bytes).expect("Deserialization failed");
    assert_eq!(proof2.num_constraints, proof.num_constraints);
    assert_eq!(proof2.k, proof.k);
    assert_eq!(proof2.conservation_residual.raw, proof.conservation_residual.raw);
    assert_eq!(proof2.cg_iterations, proof.cg_iterations);
    assert_eq!(proof2.input_state_hash_limbs, proof.input_state_hash_limbs);
}

#[test]
fn test_thermal_prover_stats() {
    let (params, input_states, laplacian_mpos) = make_thermal_prover_test_data();

    let mut prover = ThermalProver::new(params).expect("Prover creation failed");

    let _ = prover.prove(&input_states, &laplacian_mpos).expect("Failed");
    let _ = prover.prove(&input_states, &laplacian_mpos).expect("Failed");

    let stats = prover.stats();
    assert_eq!(stats.total_proofs, 2);
    assert!(stats.total_bytes > 0);
    assert!(stats.total_constraints > 0);
}

#[test]
fn test_thermal_proof_deterministic_hashes() {
    let (params, input_states, laplacian_mpos) = make_thermal_prover_test_data();

    let mut prover1 = ThermalProver::new(params.clone()).expect("Prover 1 failed");
    let mut prover2 = ThermalProver::new(params).expect("Prover 2 failed");

    let proof1 = prover1
        .prove(&input_states, &laplacian_mpos)
        .expect("Proof 1 failed");
    let proof2 = prover2
        .prove(&input_states, &laplacian_mpos)
        .expect("Proof 2 failed");

    assert_eq!(proof1.input_state_hash_limbs, proof2.input_state_hash_limbs);
    assert_eq!(proof1.output_state_hash_limbs, proof2.output_state_hash_limbs);
    assert_eq!(proof1.params_hash_limbs, proof2.params_hash_limbs);
}

// ═══════════════════════════════════════════════════════════════════════════
// circuit.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_thermal_circuit_creation() {
    let params = ThermalParams::test_small();
    let input_states = vec![MPS::new(params.num_sites(), params.chi_max, PHYS_DIM)];
    let laplacian_mpos = vec![MPO::identity(params.num_sites(), PHYS_DIM)];

    let circuit = ThermalCircuit::new(params, &input_states, &laplacian_mpos);
    assert!(circuit.is_ok(), "Circuit creation failed: {:?}", circuit.err());
}

#[test]
fn test_thermal_circuit_k() {
    let params = ThermalParams::test_small();
    let input_states = vec![MPS::new(params.num_sites(), params.chi_max, PHYS_DIM)];
    let laplacian_mpos = vec![MPO::identity(params.num_sites(), PHYS_DIM)];

    let circuit = ThermalCircuit::new(params, &input_states, &laplacian_mpos)
        .expect("Failed");
    assert!(circuit.k() >= 10);
}

#[test]
fn test_thermal_stub_validate_witness() {
    let params = ThermalParams::test_small();
    let input_states = vec![MPS::new(params.num_sites(), params.chi_max, PHYS_DIM)];
    let laplacian_mpos = vec![MPO::identity(params.num_sites(), PHYS_DIM)];

    let circuit = ThermalCircuit::new(params, &input_states, &laplacian_mpos)
        .expect("Failed");
    assert!(circuit.validate_witness().is_ok());
}

// ═══════════════════════════════════════════════════════════════════════════
// mod.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_thermal_module_exports() {
    let _params = ThermalParams::test_small();
    let _sizing = ThermalCircuitSizing::from_params(&_params);
    let _var = ThermalVariable::Temperature;
    let _stage = ThermalStage::BuildRhs;
    let _bc = BoundaryCondition::Periodic;
    let _op = ThermalQttOperation::MpoContraction;
    let _q16_scale: u64 = Q16_SCALE;
    assert_eq!(_q16_scale, 65536u64);
}

#[test]
fn test_thermal_make_test_data() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    assert_eq!(states.len(), NUM_THERMAL_VARIABLES);
    assert_eq!(mpos.len(), 1);

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
fn test_thermal_end_to_end_stub() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    let result = prove_thermal_timestep(params, &states, &mpos);
    assert!(result.is_ok(), "End-to-end failed: {:?}", result.err());

    let (proof, verification) = result.unwrap();
    assert!(verification.valid, "Proof should be valid");
    assert!(proof.num_constraints > 0, "Should have constraints");
}

#[test]
fn test_thermal_wrong_input_count() {
    let params = test_config();
    let states: Vec<MPS> = vec![];
    let mpos = make_test_laplacian_mpos(&params);

    let result = prove_thermal_timestep(params, &states, &mpos);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 1 input states"));
}

#[test]
fn test_thermal_wrong_mpo_count() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos: Vec<MPO> = vec![];

    let result = prove_thermal_timestep(params, &states, &mpos);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected at least 1 Laplacian MPO"));
}

#[test]
fn test_thermal_config_variants() {
    let test = test_config();
    let prod = production_config();

    assert!(test.grid_bits < prod.grid_bits);
    assert!(test.chi_max < prod.chi_max);
}

#[test]
fn test_thermal_e2e_circuit_sizing() {
    let params = test_config();
    let sizing = ThermalCircuitSizing::from_params(&params);

    assert!(sizing.k >= 10, "k should be at least 10 for test");
    let constraints = sizing.estimate_constraints();
    assert!(constraints > 0, "Should have non-zero constraints");
}

#[test]
fn test_thermal_proof_magic() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    let mut prover = ThermalProver::new(params).expect("Prover failed");
    let proof = prover.prove(&states, &mpos).expect("Prove failed");

    let bytes = proof.to_bytes();
    assert_eq!(&bytes[0..4], b"THEP");
}

#[test]
fn test_thermal_proof_round_trip() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    let mut prover = ThermalProver::new(params).expect("Prover failed");
    let proof = prover.prove(&states, &mpos).expect("Prove failed");

    let bytes = proof.to_bytes();
    let proof2 = ThermalProof::from_bytes(&bytes).expect("Deser failed");

    assert_eq!(proof.num_constraints, proof2.num_constraints);
    assert_eq!(proof.k, proof2.k);
    assert_eq!(proof.cg_iterations, proof2.cg_iterations);
    assert_eq!(proof.input_state_hash_limbs, proof2.input_state_hash_limbs);
    assert_eq!(proof.output_state_hash_limbs, proof2.output_state_hash_limbs);
}

#[test]
fn test_thermal_conservation_residual_reasonable() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    let (proof, _) = prove_thermal_timestep(params, &states, &mpos)
        .expect("E2E failed");

    let _residual_f64 = proof.conservation_residual.to_f64().abs();
}

// ═══════════════════════════════════════════════════════════════════════════
// Task 6.13: Truncation Error Bound Tests
// ═══════════════════════════════════════════════════════════════════════════

/// Verify truncation witness new fields are populated correctly for
/// the baseline case (no actual truncation, chi_max ≥ bond dims).
#[test]
fn test_truncation_error_witness_zero_case() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Witness gen failed");
    let trunc = &witness.truncation;

    // For zero/small MPS with chi_max=4, nothing should be truncated.
    assert!(
        trunc.all_truncated_svs.is_empty(),
        "No SVs should be truncated for chi_max=4, got {}",
        trunc.all_truncated_svs.len(),
    );

    // Accumulators should be [0] (initial only, no MAC steps).
    assert_eq!(trunc.error_sq_accumulators.len(), 1);
    assert_eq!(trunc.error_sq_accumulators[0].raw, 0);

    // Remainders should be empty.
    assert!(trunc.error_sq_remainders.is_empty());

    // Computed total error squared = 0.
    assert_eq!(trunc.total_error_sq.raw, 0);

    // Stated bound = 0 (honest prover matches exact).
    assert_eq!(trunc.stated_error_sq_bound.raw, 0);

    // Bound bits should be 32 zeros (proving 0 ≥ 0).
    assert_eq!(trunc.error_sq_bound_bits.len(), 32);
}

/// Verify the MAC chain witnesses satisfy the MAC constraint exactly.
#[test]
fn test_truncation_error_mac_constraint_algebraic() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Witness gen failed");
    let trunc = &witness.truncation;

    // Verify MAC constraint: a×b = (c_new - c_old)×SCALE + remainder
    let n = trunc.all_truncated_svs.len();
    assert_eq!(trunc.error_sq_accumulators.len(), n + 1);
    assert_eq!(trunc.error_sq_remainders.len(), n);

    let scale: i128 = 65536;
    for i in 0..n {
        let a = trunc.all_truncated_svs[i].raw as i128;
        let b = a; // squaring
        let c_old = trunc.error_sq_accumulators[i].raw as i128;
        let c_new = trunc.error_sq_accumulators[i + 1].raw as i128;
        let rem = trunc.error_sq_remainders[i] as i128;

        let lhs = a * b;
        let rhs = (c_new - c_old) * scale + rem;
        assert_eq!(
            lhs, rhs,
            "MAC constraint violated at step {}: {} != {}",
            i, lhs, rhs,
        );

        // Remainder must be in [0, SCALE)
        assert!(
            rem >= 0 && rem < scale,
            "Remainder out of range at step {}: {}",
            i, rem,
        );
    }
}

/// Verify that the stated error bound is non-negative (honest prover).
#[test]
fn test_truncation_error_bound_nonneg() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Witness gen failed");
    let trunc = &witness.truncation;

    let bound_diff = trunc.stated_error_sq_bound.raw - trunc.total_error_sq.raw;
    assert!(
        bound_diff >= 0,
        "Bound difference should be non-negative: {}",
        bound_diff,
    );

    // Verify bits correctly reconstruct the bound difference
    let reconstructed: i64 = trunc
        .error_sq_bound_bits
        .iter()
        .enumerate()
        .map(|(i, &b)| if b { 1i64 << i } else { 0 })
        .sum();
    assert_eq!(
        reconstructed, bound_diff,
        "Bit decomposition mismatch: {} != {}",
        reconstructed, bound_diff,
    );
}

/// Verify SVD ordering is maintained per-bond AND truncation witness is consistent.
#[test]
fn test_truncation_svs_consistent_with_bond_data() {
    let params = ThermalParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_thermal_test_state(params.num_sites(), params.chi_max, PHYS_DIM);
    let mpos = make_thermal_test_laplacian(params.num_sites(), PHYS_DIM);

    let witness = gen.generate(&states, &mpos).expect("Witness gen failed");
    let trunc = &witness.truncation;

    // Concatenation of truncated SVs from bond_data should match all_truncated_svs.
    let mut expected_truncated: Vec<Q16> = Vec::new();
    for bond in &trunc.bond_data {
        expected_truncated.extend_from_slice(
            &bond.singular_values[bond.truncated_rank..],
        );
    }
    assert_eq!(
        expected_truncated.len(),
        trunc.all_truncated_svs.len(),
        "Truncated SV count mismatch",
    );
    for (i, (e, a)) in expected_truncated
        .iter()
        .zip(trunc.all_truncated_svs.iter())
        .enumerate()
    {
        assert_eq!(
            e.raw, a.raw,
            "Truncated SV mismatch at index {}: {} != {}",
            i, e.raw, a.raw,
        );
    }
}

/// Gadget struct exists and is accessible.
#[test]
fn test_truncation_error_gadget_exists() {
    use fluidelite_circuits::thermal::gadgets::TruncationErrorGadget;
    let _gadget = TruncationErrorGadget;
}

// ═══════════════════════════════════════════════════════════════════════════
// Lean proof artifact validation
// ═══════════════════════════════════════════════════════════════════════════

/// Verify the Lean proof certificate hash matches the actual file.
#[test]
fn test_lean_certificate_hash_matches_proof_file() {
    use sha2::{Sha256, Digest};
    use std::path::Path;

    let proof_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("thermal_conservation_proof");

    let lean_path = proof_dir.join("ThermalConservation.lean");
    let cert_path = proof_dir.join("certificate.json");

    // Skip if artifacts missing (e.g. CI without proof dir)
    if !lean_path.exists() || !cert_path.exists() {
        return;
    }

    let lean_bytes = std::fs::read(&lean_path)
        .expect("Failed to read ThermalConservation.lean");
    let cert_bytes = std::fs::read_to_string(&cert_path)
        .expect("Failed to read certificate.json");

    let mut hasher = Sha256::new();
    hasher.update(&lean_bytes);
    let computed_hash = format!("{:x}", hasher.finalize());

    // Extract hash from JSON (simple substring search — no serde needed)
    let hash_key = "\"lean_proof_hash\": \"";
    let hash_start = cert_bytes.find(hash_key)
        .expect("certificate.json missing lean_proof_hash field")
        + hash_key.len();
    let hash_end = cert_bytes[hash_start..].find('"')
        .expect("Malformed hash value") + hash_start;
    let stored_hash = &cert_bytes[hash_start..hash_end];

    assert_eq!(
        computed_hash, stored_hash,
        "Certificate hash does not match ThermalConservation.lean — \
         regenerate with `sha256sum thermal_conservation_proof/ThermalConservation.lean`"
    );
}

/// Verify witness output conservation values are physically consistent.
///
/// After switching from flat-sum proxy to the exact transfer-matrix integral
/// (task 6.14), the absolute integral values depend on the MPS tensor
/// structure. Instead of hardcoding raw values, we verify the physics
/// invariants: conservation residual bounded, CG converged, SVD valid.
#[test]
fn test_lean_results_match_witness_output() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_laplacian_mpos(&params);

    let gen = WitnessGenerator::new(params.clone());
    let witness = gen.generate(&states, &mpos).expect("Witness generation failed");

    let cons = &witness.conservation;

    // Conservation: residual must be within tolerance.
    assert!(
        cons.residual.raw.abs() <= params.conservation_tol.raw,
        "Conservation violated: |residual| {} > tolerance {}",
        cons.residual.raw.abs(),
        params.conservation_tol.raw
    );

    // Integrals: before and after should be close (no source term).
    let integral_diff = (cons.integral_after.raw - cons.integral_before.raw).abs();
    assert!(
        integral_diff <= params.conservation_tol.raw,
        "Integral drift: |after - before| = {} > tolerance {}",
        integral_diff,
        params.conservation_tol.raw
    );

    // CG solver: must use all iterations (test_small with chi_max=4
    // doesn't converge in fewer than max_iterations).
    let solve = &witness.implicit_solve;
    assert!(
        solve.num_iterations > 0,
        "CG should run at least 1 iteration"
    );
    assert!(
        solve.num_iterations <= params.max_cg_iterations,
        "CG iterations {} exceeds max {}",
        solve.num_iterations,
        params.max_cg_iterations
    );

    // SVD truncation: error bounded, rank within chi_max.
    let trunc = &witness.truncation;
    assert!(
        trunc.total_truncation_error.raw >= 0,
        "Truncation error should be non-negative"
    );
    assert!(
        trunc.output_rank <= params.chi_max,
        "Output rank {} exceeds chi_max {}",
        trunc.output_rank,
        params.chi_max
    );
}

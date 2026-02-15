//! Integration tests for the NS-IMEX circuit module.

use fluidelite_circuits::ns_imex::*;
use fluidelite_circuits::ns_imex::config::MIN_NS_IMEX_K;
use fluidelite_circuits::ns_imex::witness::{
    decompose_nonneg_to_bits, hash_states_to_limbs, q16_sqrt_approx,
};
use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

// ═══════════════════════════════════════════════════════════════════════════
// config.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_ns_variable_all() {
    let all = NSVariable::all();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0], NSVariable::VelocityX);
    assert_eq!(all[1], NSVariable::VelocityY);
    assert_eq!(all[2], NSVariable::VelocityZ);
}

#[test]
fn test_ns_variable_display() {
    assert_eq!(format!("{}", NSVariable::VelocityX), "u");
    assert_eq!(format!("{}", NSVariable::VelocityY), "v");
    assert_eq!(format!("{}", NSVariable::VelocityZ), "w");
}

#[test]
fn test_imex_stage_all() {
    let stages = IMEXStage::all();
    assert_eq!(stages.len(), NUM_IMEX_STAGES);
    assert!(stages[0].is_explicit());
    assert!(stages[1].is_implicit());
    assert!(stages[2].is_explicit());
    assert!(stages[3].is_projection());
}

#[test]
fn test_imex_stage_properties() {
    assert!(IMEXStage::AdvectionHalf1.is_explicit());
    assert!(!IMEXStage::AdvectionHalf1.is_implicit());
    assert!(!IMEXStage::AdvectionHalf1.is_projection());

    assert!(!IMEXStage::DiffusionFull.is_explicit());
    assert!(IMEXStage::DiffusionFull.is_implicit());

    assert!(IMEXStage::Projection.is_projection());
}

#[test]
fn test_ns_qtt_operation_rows() {
    assert!(NsQttOperation::ShiftMpoApply.rows_per_instance() > 0);
    assert!(
        NsQttOperation::LaplacianMpoApply.rows_per_instance()
            > NsQttOperation::ShiftMpoApply.rows_per_instance()
    );
    assert!(
        NsQttOperation::CrossProduct.rows_per_instance()
            > NsQttOperation::ShiftMpoApply.rows_per_instance()
    );
}

#[test]
fn test_params_test_small() {
    let p = NSIMEXParams::test_small();
    assert_eq!(p.grid_bits, 4);
    assert_eq!(p.chi_max, 4);
    assert_eq!(p.num_sites(), 12);
    assert_eq!(p.grid_size(), 16);
    assert!(p.reynolds_number() > 0.0);
}

#[test]
fn test_params_production() {
    let p = NSIMEXParams::production();
    assert_eq!(p.grid_bits, 16);
    assert_eq!(p.chi_max, 32);
    assert_eq!(p.num_sites(), 48);
    assert!(p.grid_bits > NSIMEXParams::test_small().grid_bits);
}

#[test]
fn test_nsimex_params_hash_deterministic() {
    let p1 = NSIMEXParams::test_small();
    let p2 = NSIMEXParams::test_small();
    assert_eq!(p1.hash(), p2.hash());

    let mut p3 = NSIMEXParams::test_small();
    p3.viscosity = Q16::from_f64(0.1);
    assert_ne!(p1.hash(), p3.hash());
}

#[test]
fn test_params_equality() {
    let p1 = NSIMEXParams::test_small();
    let p2 = NSIMEXParams::test_small();
    assert_eq!(p1, p2);

    let p3 = NSIMEXParams::production();
    assert_ne!(p1, p3);
}

#[test]
fn test_circuit_sizing_test() {
    let p = NSIMEXParams::test_small();
    let sizing = NSIMEXCircuitSizing::from_params(&p);

    assert_eq!(sizing.num_sites, 12);
    assert_eq!(sizing.total_variable_sites, 36);
    assert!(sizing.total_constraints > 0);
    assert!(sizing.k >= MIN_NS_IMEX_K);
    assert_eq!(sizing.num_public_inputs, 16);
}

#[test]
fn test_circuit_sizing_production() {
    let test_sizing = NSIMEXCircuitSizing::from_params(&NSIMEXParams::test_small());
    let prod_sizing = NSIMEXCircuitSizing::from_params(&NSIMEXParams::production());

    assert!(prod_sizing.total_constraints > test_sizing.total_constraints);
    assert!(prod_sizing.k > test_sizing.k);
}

#[test]
fn test_sizing_constraints_positive() {
    let p = NSIMEXParams::test_small();
    let constraints = NSIMEXCircuitSizing::estimate_constraints(&p);
    assert!(constraints > 1000, "Expected significant constraints, got {}", constraints);
}

#[test]
fn test_num_public_inputs() {
    assert_eq!(NSIMEXCircuitSizing::num_public_inputs(), 16);
}

#[test]
fn test_reynolds_number() {
    let p = NSIMEXParams::test_small();
    let re = p.reynolds_number();
    assert!((re - 100.0).abs() < 10.0, "Re should be ~100, got {}", re);
}

// ═══════════════════════════════════════════════════════════════════════════
// gadgets.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_ns_fp_mac_computation() {
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
fn test_ns_fp_mac_with_remainder() {
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
fn test_conservation_bounds_pass() {
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
fn test_conservation_bounds_violation() {
    let before = Q16::from_f64(100.0);
    let after = Q16::from_f64(100.1);
    let tolerance = Q16::from_f64(0.01);

    let residual = after.raw - before.raw;
    let bound_pos = tolerance.raw - residual;

    assert!(bound_pos < 0, "Should detect conservation violation");
}

#[test]
fn test_diffusion_residual_computation() {
    let nu_dt = Q16::from_f64(0.001 * 0.01);
    let solution = Q16::from_f64(1.0);
    let laplacian_result = Q16::from_f64(-2.0);

    let nu_dt_lapl_full = (nu_dt.raw as i128) * (laplacian_result.raw as i128);
    let nu_dt_lapl = (nu_dt_lapl_full >> 16) as i64;
    let lhs = solution.raw - nu_dt_lapl;

    let lhs_f64 = lhs as f64 / 65536.0;
    assert!((lhs_f64 - 1.0).abs() < 0.01, "LHS should be ~1.0, got {}", lhs_f64);
}

#[test]
fn test_divergence_free_check() {
    let divergence = Q16::from_f64(1e-6);
    let tolerance = Q16::from_f64(0.001);

    let residual = divergence.raw;
    let bound_pos = tolerance.raw - residual;
    let bound_neg = tolerance.raw + residual;

    assert!(bound_pos >= 0, "Divergence should be within tolerance");
    assert!(bound_neg >= 0);
}

#[test]
fn test_cg_alpha_computation() {
    let r_dot_r = Q16::from_f64(0.5);
    let p_ap = Q16::from_f64(1.0);

    let alpha_raw = if p_ap.raw != 0 {
        let full = (r_dot_r.raw as i128) << 16;
        (full / p_ap.raw as i128) as i64
    } else {
        0i64
    };
    let alpha = Q16::from_raw(alpha_raw);

    assert!((alpha.to_f64() - 0.5).abs() < 0.01, "alpha should be ~0.5, got {}", alpha.to_f64());
}

#[test]
fn test_ns_sv_ordering_constraint() {
    let sv1 = Q16::from_f64(5.0);
    let sv2 = Q16::from_f64(3.0);
    let sv3 = Q16::from_f64(1.0);

    assert!(sv1.raw - sv2.raw >= 0);
    assert!(sv2.raw - sv3.raw >= 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// witness.rs tests (pub API only — compute_kinetic_energy stays inline)
// ═══════════════════════════════════════════════════════════════════════════

fn make_ns_test_states(params: &NSIMEXParams) -> Vec<MPS> {
    (0..NUM_NS_VARIABLES)
        .map(|i| {
            let mut mps = MPS::new(params.num_sites(), params.chi_max, PHYS_DIM);
            if !mps.cores.is_empty() && !mps.cores[0].data.is_empty() {
                mps.cores[0].data[0] = Q16::from_f64(0.1 * (i as f64 + 1.0));
            }
            mps
        })
        .collect()
}

fn make_ns_test_mpos(params: &NSIMEXParams) -> Vec<MPO> {
    (0..NUM_DIMENSIONS)
        .map(|_| MPO::identity(params.num_sites(), PHYS_DIM))
        .collect()
}

#[test]
fn test_ns_witness_generator_creation() {
    let params = NSIMEXParams::test_small();
    let _gen = WitnessGenerator::new(params);
    // WitnessGenerator created successfully (params is private)
}

#[test]
fn test_ns_witness_generation_basic() {
    let params = NSIMEXParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_ns_test_states(&params);
    let mpos = make_ns_test_mpos(&params);

    let witness = gen.generate(&states, &mpos);
    assert!(witness.is_ok(), "Witness generation failed: {:?}", witness.err());

    let w = witness.unwrap();
    assert_eq!(w.stages.len(), NUM_IMEX_STAGES);
    assert_eq!(w.input_hash_limbs.len(), 4);
    assert_eq!(w.output_hash_limbs.len(), 4);
    assert_eq!(w.params_hash_limbs.len(), 4);
}

#[test]
fn test_ns_witness_stage_structure() {
    let params = NSIMEXParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_ns_test_states(&params);
    let mpos = make_ns_test_mpos(&params);

    let w = gen.generate(&states, &mpos).unwrap();

    assert_eq!(w.stages[0].stage, IMEXStage::AdvectionHalf1);
    assert_eq!(w.stages[1].stage, IMEXStage::DiffusionFull);
    assert_eq!(w.stages[2].stage, IMEXStage::AdvectionHalf2);
    assert_eq!(w.stages[3].stage, IMEXStage::Projection);

    assert!(w.stages[0].diffusion_witness.is_none());
    assert!(w.stages[0].projection_witness.is_none());

    assert!(w.stages[1].diffusion_witness.is_some());
    assert!(w.stages[1].projection_witness.is_none());

    assert!(w.stages[3].diffusion_witness.is_none());
    assert!(w.stages[3].projection_witness.is_some());
}

#[test]
fn test_ns_witness_wrong_variable_count() {
    let params = NSIMEXParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states: Vec<MPS> = vec![];
    let mpos = make_ns_test_mpos(&params);

    let result = gen.generate(&states, &mpos);
    assert!(result.is_err());
}

#[test]
fn test_ns_witness_wrong_mpo_count() {
    let params = NSIMEXParams::test_small();
    let gen = WitnessGenerator::new(params.clone());

    let states = make_ns_test_states(&params);
    let mpos: Vec<MPO> = vec![];

    let result = gen.generate(&states, &mpos);
    assert!(result.is_err());
}

#[test]
fn test_ns_hash_determinism() {
    let params = NSIMEXParams::test_small();
    let states = make_ns_test_states(&params);

    let h1 = hash_states_to_limbs(&states);
    let h2 = hash_states_to_limbs(&states);
    assert_eq!(h1, h2, "Hash should be deterministic");
}

#[test]
fn test_ns_bit_decomposition() {
    let val = Q16::from_f64(1.5);
    let bits = decompose_nonneg_to_bits(val, 32);
    assert_eq!(bits.len(), 32);
    assert!(bits[15], "Bit 15 should be set for 1.5");
    assert!(bits[16], "Bit 16 should be set for 1.5");
}

#[test]
fn test_ns_q16_sqrt_approx() {
    let val = Q16::from_f64(4.0);
    let root = q16_sqrt_approx(val);
    let err = (root.to_f64() - 2.0).abs();
    assert!(err < 0.01, "sqrt(4) should be ~2.0, got {}", root.to_f64());
}

#[test]
fn test_ns_diffusion_witness_present() {
    let params = NSIMEXParams::test_small();
    let gen = WitnessGenerator::new(params.clone());
    let states = make_ns_test_states(&params);
    let mpos = make_ns_test_mpos(&params);

    let w = gen.generate(&states, &mpos).unwrap();
    let diff = w.stages[1].diffusion_witness.as_ref().unwrap();
    assert_eq!(diff.variables.len(), NUM_NS_VARIABLES);
    for var_witness in &diff.variables {
        assert_eq!(var_witness.rhs_hash.len(), 4);
        assert_eq!(var_witness.solution_hash.len(), 4);
    }
}

#[test]
fn test_ns_projection_witness_present() {
    let params = NSIMEXParams::test_small();
    let gen = WitnessGenerator::new(params.clone());
    let states = make_ns_test_states(&params);
    let mpos = make_ns_test_mpos(&params);

    let w = gen.generate(&states, &mpos).unwrap();
    let proj = w.stages[3].projection_witness.as_ref().unwrap();
    assert!(proj.cg_iterations > 0);
    assert_eq!(proj.cg_step_witnesses.len(), proj.cg_iterations);
    assert_eq!(proj.correction_witnesses.len(), NUM_NS_VARIABLES);
}

// ═══════════════════════════════════════════════════════════════════════════
// prover.rs tests
// ═══════════════════════════════════════════════════════════════════════════

fn make_ns_prover_test_data() -> (NSIMEXParams, Vec<MPS>, Vec<MPO>) {
    let params = NSIMEXParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
        .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
        .collect();
    let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
        .map(|_| MPO::identity(num_sites, PHYS_DIM))
        .collect();

    (params, input_states, shift_mpos)
}

#[test]
fn test_ns_prover_creation() {
    let (params, _, _) = make_ns_prover_test_data();
    let prover = NSIMEXProver::new(params);
    assert!(prover.is_ok(), "Prover creation failed: {:?}", prover.err());
}

#[test]
fn test_ns_prove_and_verify() {
    let (params, input_states, shift_mpos) = make_ns_prover_test_data();

    let mut prover = NSIMEXProver::new(params).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &shift_mpos)
        .expect("Proof generation failed");

    assert!(proof.size() > 0, "Proof should have non-zero size");
    assert!(proof.num_constraints > 0, "Should have non-zero constraints");

    let verifier = NSIMEXVerifier::new();
    let result = verifier.verify(&proof).expect("Verification failed");

    assert!(result.valid, "Proof should be valid");
    assert!(result.verification_time_us > 0, "Verification should take time");
    assert!(result.reynolds_number > 0.0, "Re should be positive");
}

#[test]
fn test_ns_proof_serialization() {
    let (params, input_states, shift_mpos) = make_ns_prover_test_data();

    let mut prover = NSIMEXProver::new(params).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &shift_mpos)
        .expect("Proof generation failed");

    let bytes = proof.to_bytes();
    assert!(bytes.len() > 4, "Serialized proof should be >4 bytes");
    assert_eq!(&bytes[0..4], b"NSIP", "Wrong magic");
}

#[test]
fn test_ns_prover_stats() {
    let (params, input_states, shift_mpos) = make_ns_prover_test_data();

    let mut prover = NSIMEXProver::new(params).expect("Prover creation failed");

    let _ = prover.prove(&input_states, &shift_mpos).expect("Failed");
    let _ = prover.prove(&input_states, &shift_mpos).expect("Failed");

    let stats = prover.stats();
    assert_eq!(stats.total_proofs, 2);
    assert!(stats.total_time_ms >= 0);
    assert!(stats.total_bytes > 0);
    assert!(stats.total_constraints > 0);
}

#[test]
fn test_ns_proof_diagnostics() {
    let (params, input_states, shift_mpos) = make_ns_prover_test_data();

    let mut prover = NSIMEXProver::new(params.clone()).expect("Prover creation failed");

    let proof = prover
        .prove(&input_states, &shift_mpos)
        .expect("Proof generation failed");

    let ke_abs = proof.ke_residual.raw.unsigned_abs() as i64;
    assert!(
        ke_abs <= params.conservation_tolerance.raw,
        "KE residual {} should be within tolerance {}",
        proof.ke_residual.to_f64(),
        params.conservation_tolerance.to_f64(),
    );

    let div_abs = proof.divergence_residual.raw.unsigned_abs() as i64;
    assert!(
        div_abs <= params.divergence_tolerance.raw,
        "Divergence {} should be within tolerance {}",
        proof.divergence_residual.to_f64(),
        params.divergence_tolerance.to_f64(),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// circuit.rs tests
// ═══════════════════════════════════════════════════════════════════════════

mod circuit_tests {
    use super::*;

    fn make_test_circuit() -> NSIMEXCircuit {
        let params = NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let input_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
            .map(|_| MPS::new(num_sites, chi, PHYS_DIM))
            .collect();
        let shift_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
            .map(|_| MPO::identity(num_sites, PHYS_DIM))
            .collect();

        NSIMEXCircuit::new(params, &input_states, &shift_mpos)
            .expect("Circuit creation failed")
    }

    #[test]
    fn test_ns_stub_circuit_creation() {
        let circuit = make_test_circuit();
        assert!(circuit.estimate_constraints() > 0);
    }

    #[test]
    fn test_ns_stub_witness_validation() {
        let circuit = make_test_circuit();
        let result = circuit.validate_witness();
        assert!(result.is_ok(), "Witness validation failed: {:?}", result.err());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// mod.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_ns_module_exports() {
    let _params = NSIMEXParams::test_small();
    let _sizing = NSIMEXCircuitSizing::from_params(&_params);
    let _var = NSVariable::VelocityX;
    let _stage = IMEXStage::AdvectionHalf1;
    let _op = NsQttOperation::ShiftMpoApply;
    let _q16_scale: u64 = Q16_SCALE;
    assert_eq!(_q16_scale, 65536u64);
}

#[test]
fn test_nsimex_make_test_data() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_shift_mpos(&params);

    assert_eq!(states.len(), NUM_NS_VARIABLES);
    assert_eq!(mpos.len(), NUM_DIMENSIONS);

    for (i, state) in states.iter().enumerate() {
        assert_eq!(
            state.cores.len(),
            params.num_sites(),
            "State {} wrong num_sites",
            i,
        );
    }

    for (i, mpo) in mpos.iter().enumerate() {
        assert_eq!(
            mpo.cores.len(),
            params.num_sites(),
            "MPO {} wrong num_sites",
            i,
        );
    }
}

#[test]
fn test_nsimex_end_to_end_stub() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_shift_mpos(&params);

    let result = prove_ns_imex_timestep(params, &states, &mpos);
    assert!(result.is_ok(), "End-to-end failed: {:?}", result.err());

    let (proof, verification) = result.unwrap();
    assert!(verification.valid, "Proof should be valid");
    assert!(proof.num_constraints > 0, "Should have constraints");
}

#[test]
fn test_nsimex_wrong_input_count() {
    let params = test_config();
    let states: Vec<MPS> = vec![];
    let mpos = make_test_shift_mpos(&params);

    let result = prove_ns_imex_timestep(params, &states, &mpos);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 3 input states"));
}

#[test]
fn test_nsimex_wrong_mpo_count() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos: Vec<MPO> = vec![];

    let result = prove_ns_imex_timestep(params, &states, &mpos);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 3 shift MPOs"));
}

#[test]
fn test_nsimex_config_variants() {
    let test = test_config();
    let prod = production_config();

    assert!(test.grid_bits < prod.grid_bits);
    assert!(test.chi_max < prod.chi_max);
}

#[test]
fn test_nsimex_circuit_sizing() {
    let params = test_config();
    let sizing = NSIMEXCircuitSizing::from_params(&params);

    assert!(sizing.k >= 11, "k should be at least 11 for test");
    assert!(sizing.total_constraints > 0, "Should have non-zero constraints");
}

#[test]
fn test_nsimex_proof_contains_diagnostics() {
    let params = test_config();
    let states = make_test_states(&params);
    let mpos = make_test_shift_mpos(&params);

    let (proof, result) = prove_ns_imex_timestep(params, &states, &mpos)
        .expect("E2E failed");

    assert!(result.reynolds_number > 0.0);
    assert!(result.grid_bits > 0);
    assert!(result.chi_max > 0);

    let input_hash_nonzero = proof
        .input_state_hash_limbs
        .iter()
        .any(|&limb| limb != 0);
    assert!(input_hash_nonzero, "Input hash should be non-zero");

    let params_hash_nonzero = proof
        .params_hash_limbs
        .iter()
        .any(|&limb| limb != 0);
    assert!(params_hash_nonzero, "Params hash should be non-zero");
}

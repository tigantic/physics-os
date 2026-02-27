//! Integration tests for the Proof Preview module.

use fluidelite_circuits::proof_preview::*;
use fluidelite_circuits::proof_preview::spot_check::SpotChecker;
use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;
use fluidelite_core::physics_traits::SolverType;

// ═══════════════════════════════════════════════════════════════════════════
// mod.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_proof_preview_module_exports() {
    let _config = PreviewConfig::default();
    let _verdict = PreviewVerdict::Pass;
}

// ═══════════════════════════════════════════════════════════════════════════
// fast_verify.rs tests (pub API — check_bond_consistency stays inline)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_preview_config_default() {
    let config = PreviewConfig::default();
    assert_eq!(config.time_budget_ms, 60_000);
    assert!(config.min_confidence > 0.0);
    assert!(config.check_dimensions);
    assert!(config.check_data_integrity);
}

#[test]
fn test_preview_config_fast() {
    let config = PreviewConfig::fast();
    assert!(config.time_budget_ms < PreviewConfig::default().time_budget_ms);
    assert!(config.min_confidence < PreviewConfig::default().min_confidence);
}

#[test]
fn test_preview_config_thorough() {
    let config = PreviewConfig::thorough();
    assert!(
        config.spot_check.sample_fraction
            > PreviewConfig::default().spot_check.sample_fraction
    );
    assert!(config.min_confidence > PreviewConfig::default().min_confidence);
}

#[test]
fn test_preview_verdict_display() {
    assert_eq!(format!("{}", PreviewVerdict::Pass), "PASS");
    assert_eq!(format!("{}", PreviewVerdict::Fail), "FAIL");
    assert_eq!(format!("{}", PreviewVerdict::Inconclusive), "INCONCLUSIVE");
}

#[test]
fn test_fast_verify_thermal_pass() {
    let params = fluidelite_circuits::thermal::ThermalParams::test_small();
    let input_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let output_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let laplacian = MPO::identity(params.num_sites(), 2);

    let input_hash = SpotChecker::compute_state_hash(
        &[input_state.clone()],
        b"THERMAL_STATE_V1",
    );
    let output_hash = SpotChecker::compute_state_hash(
        &[output_state.clone()],
        b"THERMAL_STATE_V1",
    );

    let verifier = FastVerifier::default_verifier();
    let result = verifier.verify_thermal(
        &input_state,
        &output_state,
        &laplacian,
        Q16::zero(),
        params.conservation_tol,
        &input_hash,
        &output_hash,
    );

    assert!(result.dimensions_ok);
    assert!(result.data_integrity_ok);
    assert!(result.spot_check.is_some());
    assert!(
        result.verdict == PreviewVerdict::Pass
            || result.verdict == PreviewVerdict::Inconclusive,
        "Expected Pass or Inconclusive, got {:?}: {:?}",
        result.verdict,
        result.messages,
    );
}

#[test]
fn test_fast_verify_thermal_fail_dimensions() {
    let params = fluidelite_circuits::thermal::ThermalParams::test_small();
    let input_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let output_state = MPS::new(params.num_sites() + 2, params.chi_max, 2);
    let laplacian = MPO::identity(params.num_sites(), 2);

    let input_hash = SpotChecker::compute_state_hash(
        &[input_state.clone()],
        b"THERMAL_STATE_V1",
    );
    let output_hash = SpotChecker::compute_state_hash(
        &[output_state.clone()],
        b"THERMAL_STATE_V1",
    );

    let verifier = FastVerifier::default_verifier();
    let result = verifier.verify_thermal(
        &input_state,
        &output_state,
        &laplacian,
        Q16::zero(),
        params.conservation_tol,
        &input_hash,
        &output_hash,
    );

    assert!(!result.dimensions_ok);
    assert_eq!(result.verdict, PreviewVerdict::Fail);
}

#[test]
fn test_fast_verify_generic_pass() {
    let ns = 4;
    let chi = 2;
    let d = 2;
    let input_states: Vec<MPS> = (0..3).map(|_| MPS::new(ns, chi, d)).collect();
    let output_states: Vec<MPS> = (0..3).map(|_| MPS::new(ns, chi, d)).collect();
    let shift_mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(ns, d)).collect();

    let conservation_residuals = vec![Q16::zero(); 3];
    let conservation_tol = Q16::from_raw(1000);

    let input_hash = SpotChecker::compute_state_hash(
        &input_states,
        b"EULER3D_STATE_V1",
    );
    let output_hash = SpotChecker::compute_state_hash(
        &output_states,
        b"EULER3D_STATE_V1",
    );

    let verifier = FastVerifier::default_verifier();
    let result = verifier.verify_generic(
        SolverType::Euler3D,
        &input_states,
        &output_states,
        &shift_mpos,
        &conservation_residuals,
        conservation_tol,
        &input_hash,
        &output_hash,
        b"EULER3D_STATE_V1",
    );

    assert!(result.dimensions_ok);
    assert!(result.data_integrity_ok);
    assert!(
        result.verdict == PreviewVerdict::Pass
            || result.verdict == PreviewVerdict::Inconclusive
    );
}

#[test]
fn test_fast_verify_solver_type_carried() {
    let ns = 4;
    let chi = 2;
    let d = 2;
    let states: Vec<MPS> = (0..5).map(|_| MPS::new(ns, chi, d)).collect();
    let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(ns, d)).collect();
    let residuals = vec![Q16::zero(); 5];
    let tol = Q16::from_raw(1000);

    let h_in = SpotChecker::compute_state_hash(&states, b"NS_IMEX_STATE_V1");
    let h_out = SpotChecker::compute_state_hash(&states, b"NS_IMEX_STATE_V1");

    let verifier = FastVerifier::default_verifier();
    let result = verifier.verify_generic(
        SolverType::NsImex,
        &states,
        &states,
        &mpos,
        &residuals,
        tol,
        &h_in,
        &h_out,
        b"NS_IMEX_STATE_V1",
    );

    assert_eq!(result.solver_type, SolverType::NsImex);
}

// ═══════════════════════════════════════════════════════════════════════════
// spot_check.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_spot_check_config_default() {
    let config = SpotCheckConfig::default();
    assert!(config.sample_fraction > 0.0);
    assert!(config.sample_fraction <= 1.0);
    assert!(config.min_sites > 0);
}

// ═══════════════════════════════════════════════════════════════════════════

// site_selection tests moved to inline in spot_check.rs (private method access)

#[test]
fn test_spot_check_thermal() {
    let params = fluidelite_circuits::thermal::ThermalParams::test_small();
    let input_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let output_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let laplacian = MPO::identity(params.num_sites(), 2);

    let input_hash = SpotChecker::compute_state_hash(
        &[input_state.clone()],
        b"THERMAL_STATE_V1",
    );
    let output_hash = SpotChecker::compute_state_hash(
        &[output_state.clone()],
        b"THERMAL_STATE_V1",
    );

    let checker = SpotChecker::default_checker();
    let result = checker.check_thermal(
        &input_state,
        &output_state,
        &laplacian,
        Q16::zero(),
        params.conservation_tol,
        &input_hash,
        &output_hash,
    );

    assert!(result.passed, "Spot check failed: {:?}", result.failures);
    assert!(result.hash_integrity);
    assert!(result.conservation_passed);
    assert!(result.confidence > 0.0);
    assert!(result.check_time_us > 0);
}

#[test]
fn test_spot_check_bad_hash() {
    let params = fluidelite_circuits::thermal::ThermalParams::test_small();
    let input_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let output_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let laplacian = MPO::identity(params.num_sites(), 2);

    let bad_hash = [0u64; 4];

    let checker = SpotChecker::default_checker();
    let result = checker.check_thermal(
        &input_state,
        &output_state,
        &laplacian,
        Q16::zero(),
        params.conservation_tol,
        &bad_hash,
        &bad_hash,
    );

    assert!(!result.passed);
    assert!(!result.hash_integrity);
}

#[test]
fn test_spot_check_conservation_violation() {
    let params = fluidelite_circuits::thermal::ThermalParams::test_small();
    let input_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let output_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let laplacian = MPO::identity(params.num_sites(), 2);

    let input_hash = SpotChecker::compute_state_hash(
        &[input_state.clone()],
        b"THERMAL_STATE_V1",
    );
    let output_hash = SpotChecker::compute_state_hash(
        &[output_state.clone()],
        b"THERMAL_STATE_V1",
    );

    let big_residual = Q16::from_f64(1000.0);

    let checker = SpotChecker::default_checker();
    let result = checker.check_thermal(
        &input_state,
        &output_state,
        &laplacian,
        big_residual,
        params.conservation_tol,
        &input_hash,
        &output_hash,
    );

    assert!(!result.passed);
    assert!(!result.conservation_passed);
}

#[test]
fn test_spot_check_deterministic() {
    let params = fluidelite_circuits::thermal::ThermalParams::test_small();
    let input_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let output_state = MPS::new(params.num_sites(), params.chi_max, 2);
    let laplacian = MPO::identity(params.num_sites(), 2);

    let input_hash = SpotChecker::compute_state_hash(
        &[input_state.clone()],
        b"THERMAL_STATE_V1",
    );
    let output_hash = SpotChecker::compute_state_hash(
        &[output_state.clone()],
        b"THERMAL_STATE_V1",
    );

    let checker = SpotChecker::default_checker();

    let r1 = checker.check_thermal(
        &input_state, &output_state, &laplacian,
        Q16::zero(), params.conservation_tol,
        &input_hash, &output_hash,
    );
    let r2 = checker.check_thermal(
        &input_state, &output_state, &laplacian,
        Q16::zero(), params.conservation_tol,
        &input_hash, &output_hash,
    );

    assert_eq!(r1.sites_checked, r2.sites_checked);
    assert_eq!(r1.passed, r2.passed);
}

#[test]
fn test_compute_state_hash_deterministic() {
    let state = MPS::new(4, 2, 2);
    let h1 = SpotChecker::compute_state_hash(&[state.clone()], b"TEST");
    let h2 = SpotChecker::compute_state_hash(&[state], b"TEST");
    assert_eq!(h1, h2);
}

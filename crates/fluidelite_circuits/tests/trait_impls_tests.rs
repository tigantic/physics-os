//! Integration tests for trait implementations (PhysicsProof/Prover/Verifier).

use fluidelite_circuits::euler3d;
use fluidelite_circuits::ns_imex;
use fluidelite_circuits::thermal;
use fluidelite_circuits::trait_impls::{euler3d_factory, ns_imex_factory, thermal_factory};
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;
use fluidelite_core::physics_traits::{
    PhysicsProof, PhysicsProver, PhysicsVerifier, SolverType, UnifiedVerificationResult,
};

// ═══════════════════════════════════════════════════════════════════════════
// trait_impls.rs tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_euler3d_physics_proof_trait() {
    let params = euler3d::Euler3DParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..5)
        .map(|_| MPS::new(num_sites, chi, 2))
        .collect();
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    let mut prover =
        euler3d::Euler3DProver::new(params).expect("Prover failed");
    let proof: euler3d::Euler3DProof =
        PhysicsProver::prove(&mut prover, &input_states, &shift_mpos)
            .expect("Prove failed");

    assert_eq!(PhysicsProof::solver_type(&proof), SolverType::Euler3D);
    assert!(PhysicsProof::proof_size(&proof) > 0);
    assert!(PhysicsProof::num_constraints(&proof) > 0);
    assert_eq!(PhysicsProof::input_hash_limbs(&proof).len(), 4);
}

#[test]
fn test_ns_imex_physics_proof_trait() {
    let params = ns_imex::NSIMEXParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..3)
        .map(|_| MPS::new(num_sites, chi, 2))
        .collect();
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    let mut prover =
        ns_imex::NSIMEXProver::new(params).expect("Prover failed");
    let proof: ns_imex::NSIMEXProof =
        PhysicsProver::prove(&mut prover, &input_states, &shift_mpos)
            .expect("Prove failed");

    assert_eq!(PhysicsProof::solver_type(&proof), SolverType::NsImex);
    assert!(PhysicsProof::proof_size(&proof) > 0);
    assert!(PhysicsProof::num_constraints(&proof) > 0);
}

#[test]
fn test_thermal_physics_proof_trait() {
    let params = thermal::ThermalParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states = vec![MPS::new(num_sites, chi, 2)];
    let laplacian_mpos = vec![MPO::identity(num_sites, 2)];

    let mut prover =
        thermal::ThermalProver::new(params).expect("Prover failed");
    let proof: thermal::ThermalProof =
        PhysicsProver::prove(&mut prover, &input_states, &laplacian_mpos)
            .expect("Prove failed");

    assert_eq!(PhysicsProof::solver_type(&proof), SolverType::Thermal);
    assert!(PhysicsProof::proof_size(&proof) > 0);
    assert!(PhysicsProof::num_constraints(&proof) > 0);
    assert_eq!(PhysicsProof::input_hash_limbs(&proof).len(), 4);
}

#[test]
fn test_euler3d_verifier_trait() {
    let params = euler3d::Euler3DParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..5)
        .map(|_| MPS::new(num_sites, chi, 2))
        .collect();
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    let mut prover =
        euler3d::Euler3DProver::new(params).expect("Prover failed");
    let proof =
        PhysicsProver::prove(&mut prover, &input_states, &shift_mpos)
            .expect("Prove failed");

    let verifier = euler3d::Euler3DVerifier::new();
    let result: UnifiedVerificationResult =
        PhysicsVerifier::verify(&verifier, &proof).expect("Verify failed");

    assert!(result.valid);
    assert_eq!(result.solver_type, SolverType::Euler3D);
}

#[test]
fn test_ns_imex_verifier_trait() {
    let params = ns_imex::NSIMEXParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..3)
        .map(|_| MPS::new(num_sites, chi, 2))
        .collect();
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    let mut prover =
        ns_imex::NSIMEXProver::new(params).expect("Prover failed");
    let proof =
        PhysicsProver::prove(&mut prover, &input_states, &shift_mpos)
            .expect("Prove failed");

    let verifier = ns_imex::NSIMEXVerifier::new();
    let result: UnifiedVerificationResult =
        PhysicsVerifier::verify(&verifier, &proof).expect("Verify failed");

    assert!(result.valid);
    assert_eq!(result.solver_type, SolverType::NsImex);
}

#[test]
fn test_thermal_verifier_trait() {
    let params = thermal::ThermalParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states = vec![MPS::new(num_sites, chi, 2)];
    let laplacian_mpos = vec![MPO::identity(num_sites, 2)];

    let mut prover =
        thermal::ThermalProver::new(params).expect("Prover failed");
    let proof =
        PhysicsProver::prove(&mut prover, &input_states, &laplacian_mpos)
            .expect("Prove failed");

    let verifier = thermal::ThermalVerifier::new();
    let result: UnifiedVerificationResult =
        PhysicsVerifier::verify(&verifier, &proof).expect("Verify failed");

    assert!(result.valid);
    assert_eq!(result.solver_type, SolverType::Thermal);
}

#[test]
fn test_euler3d_factory() {
    let params = euler3d::Euler3DParams::test_small();
    let factory = euler3d_factory(params);
    let prover = factory();
    assert!(prover.is_ok());
}

#[test]
fn test_ns_imex_factory() {
    let params = ns_imex::NSIMEXParams::test_small();
    let factory = ns_imex_factory(params);
    let prover = factory();
    assert!(prover.is_ok());
}

#[test]
fn test_thermal_factory() {
    let params = thermal::ThermalParams::test_small();
    let factory = thermal_factory(params);
    let prover = factory();
    assert!(prover.is_ok());
}

#[test]
fn test_prover_avg_time() {
    let params = euler3d::Euler3DParams::test_small();
    let num_sites = params.num_sites();
    let chi = params.chi_max;

    let input_states: Vec<MPS> = (0..5)
        .map(|_| MPS::new(num_sites, chi, 2))
        .collect();
    let shift_mpos: Vec<MPO> = (0..3)
        .map(|_| MPO::identity(num_sites, 2))
        .collect();

    let mut prover =
        euler3d::Euler3DProver::new(params).expect("Prover failed");

    assert_eq!(PhysicsProver::avg_time_ms(&prover), 0.0);

    let _ = PhysicsProver::prove(&mut prover, &input_states, &shift_mpos)
        .expect("Prove failed");

    assert!(PhysicsProver::total_proofs(&prover) == 1);
}

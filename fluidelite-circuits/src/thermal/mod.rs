//! Thermal/Heat Equation ZK Proof Circuit — Trustless Physics Phase 4
//!
//! Zero-knowledge proof circuit for the heat equation
//!   ∂T/∂t = α∇²T + S(x,t)
//! solved via QTT (Quantized Tensor Train) decomposition with
//! implicit time stepping (Conjugate Gradient).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │               Thermal Proof Circuit                          │
//! ├──────────────────────────────────────────────────────────────┤
//! │  Public Inputs:                                              │
//! │    ├── input_state_hash   (SHA-256 of initial T QTT cores)  │
//! │    ├── output_state_hash  (SHA-256 of final T QTT cores)    │
//! │    ├── params_hash        (solver config commitment)        │
//! │    ├── conservation_residual (energy balance |∫T^{n+1}-∫T^n|)│
//! │    ├── timestep_dt        (time step size, Q16.16)          │
//! │    └── thermal_diffusivity_alpha (Q16.16)                   │
//! │                                                              │
//! │  Private Witnesses:                                          │
//! │    ├── CG iteration residuals and search directions         │
//! │    ├── MPO×MPS contraction intermediates (Laplacian apply)  │
//! │    ├── Singular values from SVD truncation                  │
//! │    ├── Fixed-point remainders for MAC operations            │
//! │    └── Bit decompositions for range proofs                  │
//! │                                                              │
//! │  Constraints:                                                │
//! │    ├── RHS Assembly: T^n + Δt·S^n (source term)             │
//! │    ├── CG Solve: (I - α·Δt·L) T^{n+1} = r (implicit step) │
//! │    ├── SVD ordering: s_i ≥ s_{i+1} ≥ 0                     │
//! │    ├── Rank bound: rank ≤ χ_max                             │
//! │    ├── Conservation: |∫T^{n+1} - ∫T^n - Δt·∫S| ≤ ε_cons   │
//! │    └── Public input binding                                 │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # QTT Advantage for ZK Proofs
//!
//! | Parameter      | Test       | Production |
//! |----------------|------------|------------|
//! | grid_bits (L)  | 4          | 16         |
//! | χ_max          | 4          | 32         |
//! | phys_dim       | 2          | 2          |
//! | k              | 10         | 22         |
//! | Constraints    | ~30K       | ~2M        |
//! | Proof time     | <1s        | ~45s       |
//! | Proof size     | ~800 B     | ~800 B     |
//! | Verify time    | <10ms      | <10ms      |
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod config;
pub mod gadgets;
pub mod halo2_impl;
pub mod prover;
#[cfg(feature = "stark")]
pub mod stark_impl;
pub mod witness;

// ═══════════════════════════════════════════════════════════════════════════
// Re-exports: Types that are always available (no feature gate)
// ═══════════════════════════════════════════════════════════════════════════

pub use config::{
    BoundaryCondition, ThermalCircuitBreakdown, ThermalCircuitSizing,
    ThermalParams, ThermalQttOperation, ThermalStage, ThermalVariable,
    NUM_DIMENSIONS, NUM_THERMAL_STAGES, NUM_THERMAL_VARIABLES, PHYS_DIM,
    Q16_SCALE, ROWS_PER_BC_CHECK, ROWS_PER_CG_SOLVE, ROWS_PER_CONSERVATION,
    ROWS_PER_FP_MAC, ROWS_PER_SV_ORDER,
};

pub use witness::{
    BondSvdData, CgIterationWitness, ConservationWitness, ContractionWitness,
    HashWitness, ImplicitSolveWitness, RhsAssemblyWitness,
    SvdTruncationWitness, ThermalWitness, WitnessError, WitnessGenerator,
};

pub use halo2_impl::ThermalCircuit;

pub use prover::{
    ThermalProof, ThermalProver, ThermalProverStats, ThermalVerificationResult,
    ThermalVerifier,
};

#[cfg(feature = "stark")]
pub use stark_impl::{
    ThermalAir, ThermalStarkInputs, TimestepPhysics,
    prove_thermal_stark, verify_thermal_stark,
    build_trace as build_stark_trace,
};

// ═══════════════════════════════════════════════════════════════════════════
// Module-level convenience functions
// ═══════════════════════════════════════════════════════════════════════════

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

/// Run a complete prove-and-verify cycle for one thermal timestep.
///
/// This is the main entry point for the Trustless Physics thermal pipeline:
/// 1. Generates witness from input state and Laplacian MPO
/// 2. Creates the proof circuit
/// 3. Generates the ZK proof
/// 4. Verifies the proof
///
/// Returns (proof, verification_result) on success.
pub fn prove_thermal_timestep(
    params: ThermalParams,
    input_states: &[MPS],
    laplacian_mpos: &[MPO],
) -> Result<(ThermalProof, ThermalVerificationResult), String> {
    // Validate inputs
    if input_states.len() != NUM_THERMAL_VARIABLES {
        return Err(format!(
            "Expected {} input states (temperature), got {}",
            NUM_THERMAL_VARIABLES,
            input_states.len()
        ));
    }
    if laplacian_mpos.is_empty() {
        return Err("Expected at least 1 Laplacian MPO, got 0".to_string());
    }

    // Generate proof
    let mut prover = ThermalProver::new(params)?;
    let proof = prover.prove(input_states, laplacian_mpos)?;

    // Verify proof
    #[cfg(not(feature = "halo2"))]
    let result = {
        let verifier = ThermalVerifier::new();
        verifier.verify(&proof)?
    };

    #[cfg(feature = "halo2")]
    let result = {
        let verifier = ThermalVerifier::from_prover(&prover);
        verifier.verify(&proof)?
    };

    Ok((proof, result))
}

/// Create a test configuration with small parameters.
pub fn test_config() -> ThermalParams {
    ThermalParams::test_small()
}

/// Create a production configuration.
pub fn production_config() -> ThermalParams {
    ThermalParams::production()
}

/// Create test input states (1 MPS for temperature).
pub fn make_test_states(params: &ThermalParams) -> Vec<MPS> {
    vec![{
        let mut mps = MPS::new(params.num_sites(), params.chi_max, 2);
        // Give the state a small non-zero value so conservation has
        // something to check.
        if !mps.cores.is_empty() && !mps.cores[0].data.is_empty() {
            mps.cores[0].data[0] = Q16::from_f64(0.5);
        }
        mps
    }]
}

/// Create test Laplacian MPOs (1, identity for testing).
pub fn make_test_laplacian_mpos(params: &ThermalParams) -> Vec<MPO> {
    vec![MPO::identity(params.num_sites(), 2)]
}

// ═══════════════════════════════════════════════════════════════════════════
// Module-level tests
// ═══════════════════════════════════════════════════════════════════════════

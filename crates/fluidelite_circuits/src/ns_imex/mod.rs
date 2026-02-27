//! NS-IMEX ZK Proof Circuit — Trustless Physics Phase 2
//!
//! Zero-knowledge proof circuit for the incompressible Navier-Stokes
//! equations solved via QTT with IMEX (Implicit-Explicit) time splitting.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                 NS-IMEX Proof Circuit                        │
//! ├──────────────────────────────────────────────────────────────┤
//! │  Public Inputs:                                              │
//! │    ├── input_state_hash   (SHA-256 of initial QTT cores)    │
//! │    ├── output_state_hash  (SHA-256 of final QTT cores)      │
//! │    ├── params_hash        (solver config commitment)        │
//! │    ├── ke_residual        (kinetic energy conservation)     │
//! │    ├── enstrophy_residual (enstrophy dissipation check)     │
//! │    ├── divergence_residual (‖∇·u‖ < ε_div)                 │
//! │    └── timestep_dt        (time step size, Q16.16)          │
//! │                                                              │
//! │  Private Witnesses:                                          │
//! │    ├── All intermediate MPS core values                     │
//! │    ├── Singular values from each SVD truncation             │
//! │    ├── Diffusion solve: (I - ν·Δt·L)u = u*                 │
//! │    ├── CG pressure Poisson: ∇²p = (1/Δt)·∇·u              │
//! │    ├── Fixed-point remainders for MAC operations            │
//! │    └── Bit decompositions for range proofs                  │
//! │                                                              │
//! │  IMEX Stages:                                                │
//! │    1. Advection half-step: u* = u^n + (Δt/2)·A(u^n)        │
//! │    2. Diffusion full-step: (I - ν·Δt·L) u** = u*           │
//! │    3. Advection half-step: u*** = u** + (Δt/2)·A(u**)      │
//! │    4. Projection: u^{n+1} = u*** - Δt·∇p                   │
//! │                                                              │
//! │  Constraints:                                                │
//! │    ├── Advection: MPO×MPS contraction MAC chains            │
//! │    ├── Diffusion: implicit solve residual ≤ tolerance       │
//! │    ├── Projection: CG iteration fidelity                    │
//! │    ├── SVD: singular value ordering (s_i ≥ s_{i+1} ≥ 0)    │
//! │    ├── Conservation: |ΔKE| ≤ ε_cons, |ΔΩ| ≤ ε_cons        │
//! │    ├── Divergence: ‖∇·u^{n+1}‖ ≤ ε_div                    │
//! │    └── Public input binding                                 │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # QTT Advantage for NS-IMEX
//!
//! The incompressible NS equations with IMEX splitting leverage QTT
//! for O(r³ log N) operations per timestep. The ZK circuit verifies
//! the QTT operations with the same logarithmic advantage.
//!
//! | Parameter      | Test       | Production |
//! |----------------|------------|------------|
//! | grid_bits (L)  | 4          | 16         |
//! | χ_max          | 4          | 32         |
//! | phys_dim       | 2          | 2          |
//! | k              | 14         | 25         |
//! | Constraints    | ~80K       | ~8M        |
//! | Proof time     | <1s        | ~90s       |
//! | Proof size     | ~800 B     | ~800 B     |
//! | Verify time    | <10ms      | <10ms      |
//! | Reynolds       | ~100       | ~1000      |
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod circuit;
pub mod config;
pub mod gadgets;
pub mod prover;
pub mod witness;

// ═══════════════════════════════════════════════════════════════════════════
// Re-exports: Types that are always available (no feature gate)
// ═══════════════════════════════════════════════════════════════════════════

pub use config::{
    IMEXStage, NSIMEXCircuitSizing, NSIMEXParams, NSVariable, NsQttOperation,
    NUM_DIMENSIONS, NUM_IMEX_STAGES, NUM_NS_VARIABLES, PHYS_DIM,
    Q16_SCALE, ROWS_PER_CONSERVATION, ROWS_PER_DIFFUSION_SOLVE,
    ROWS_PER_FP_MAC, ROWS_PER_PROJECTION, ROWS_PER_SV_ORDER,
};

pub use witness::{
    CGStepWitness, ContractionWitness, DiffusionSolveWitness,
    DiffusionVariableWitness, IMEXStageWitness, NSIMEXWitness,
    ProjectionWitness, SvdTruncationWitness, VariableSweepWitness,
    WitnessError, WitnessGenerator,
};

pub use circuit::NSIMEXCircuit;

pub use prover::{
    NSIMEXProof, NSIMEXProver, NSIMEXProverStats, NSIMEXVerificationResult,
    NSIMEXVerifier,
};

// ═══════════════════════════════════════════════════════════════════════════
// Module-level convenience functions
// ═══════════════════════════════════════════════════════════════════════════

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

/// Run a complete prove-and-verify cycle for one NS-IMEX timestep.
///
/// This is the main entry point for the Trustless Physics NS-IMEX pipeline:
/// 1. Generates witness from input velocity states and shift MPOs
/// 2. Creates the proof circuit
/// 3. Generates the ZK proof
/// 4. Verifies the proof
///
/// Returns (proof, verification_result) on success.
pub fn prove_ns_imex_timestep(
    params: NSIMEXParams,
    input_states: &[MPS],
    shift_mpos: &[MPO],
) -> Result<(NSIMEXProof, NSIMEXVerificationResult), String> {
    // Validate inputs
    if input_states.len() != NUM_NS_VARIABLES {
        return Err(format!(
            "Expected {} input states (u, v, w), got {}",
            NUM_NS_VARIABLES,
            input_states.len(),
        ));
    }
    if shift_mpos.len() != NUM_DIMENSIONS {
        return Err(format!(
            "Expected {} shift MPOs (one per axis), got {}",
            NUM_DIMENSIONS,
            shift_mpos.len(),
        ));
    }

    // Generate proof
    let mut prover = NSIMEXProver::new(params)?;
    let proof = prover.prove(input_states, shift_mpos)?;

    // Verify proof
    let result = {
        let verifier = NSIMEXVerifier::new();
        verifier.verify(&proof)?
    };

    Ok((proof, result))
}

/// Create a test configuration for NS-IMEX with small parameters.
pub fn test_config() -> NSIMEXParams {
    NSIMEXParams::test_small()
}

/// Create a production configuration for NS-IMEX.
pub fn production_config() -> NSIMEXParams {
    NSIMEXParams::production()
}

/// Create test input states (3 MPS, one per velocity component).
pub fn make_test_states(params: &NSIMEXParams) -> Vec<MPS> {
    (0..NUM_NS_VARIABLES)
        .map(|i| {
            let mut mps = MPS::new(params.num_sites(), params.chi_max, PHYS_DIM);
            if !mps.cores.is_empty() && !mps.cores[0].data.is_empty() {
                let base_val = Q16::from_f64(0.1 * (i as f64 + 1.0));
                mps.cores[0].data[0] = base_val;
            }
            mps
        })
        .collect()
}

/// Create test shift MPOs (3, one per spatial axis).
pub fn make_test_shift_mpos(params: &NSIMEXParams) -> Vec<MPO> {
    (0..NUM_DIMENSIONS)
        .map(|_| MPO::identity(params.num_sites(), PHYS_DIM))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Module-level tests
// ═══════════════════════════════════════════════════════════════════════════

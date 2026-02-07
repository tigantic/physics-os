//! Euler 3D ZK Proof Circuit — Trustless Physics Phase 1
//!
//! Zero-knowledge proof circuit for the compressible Euler equations
//! solved via QTT (Quantized Tensor Train) decomposition with
//! Strang splitting.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                 Euler 3D Proof Circuit                       │
//! ├──────────────────────────────────────────────────────────────┤
//! │  Public Inputs:                                              │
//! │    ├── input_state_hash   (SHA-256 of initial QTT cores)    │
//! │    ├── output_state_hash  (SHA-256 of final QTT cores)      │
//! │    ├── params_hash        (solver config commitment)        │
//! │    ├── conservation_residuals (mass, mom_x/y/z, energy)     │
//! │    └── timestep_dt        (time step size, Q16.16)          │
//! │                                                              │
//! │  Private Witnesses:                                          │
//! │    ├── All intermediate MPS core values                     │
//! │    ├── Singular values from each SVD truncation             │
//! │    ├── Fixed-point remainders for MAC operations            │
//! │    └── Bit decompositions for range proofs                  │
//! │                                                              │
//! │  Constraints:                                                │
//! │    ├── MPO×MPS contraction MAC chains (Q16 fixed-point)     │
//! │    ├── SVD singular value ordering (s_i ≥ s_{i+1} ≥ 0)     │
//! │    ├── Rank bound (rank ≤ χ_max)                            │
//! │    ├── Q16.16 remainder range checks (bit decomposition)    │
//! │    ├── Conservation |∫ρ_new - ∫ρ_old| ≤ ε_cons             │
//! │    └── Public input binding                                 │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # QTT Advantage for ZK Proofs
//!
//! Because QTT operations are O(r³ log N) instead of O(N³), the
//! circuit size is O(r³ log N) — exponentially smaller than a
//! dense CFD proof circuit.
//!
//! | Parameter      | Test       | Production |
//! |----------------|------------|------------|
//! | grid_bits (L)  | 4          | 16         |
//! | χ_max          | 4          | 32         |
//! | phys_dim       | 2          | 2          |
//! | k              | 14         | 23         |
//! | Constraints    | ~50K       | ~4M        |
//! | Proof time     | <1s        | ~60s       |
//! | Proof size     | ~800 B     | ~800 B     |
//! | Verify time    | <10ms      | <10ms      |
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod config;
pub mod gadgets;
pub mod halo2_impl;
pub mod prover;
pub mod witness;

// ═══════════════════════════════════════════════════════════════════════════
// Re-exports: Types that are always available (no feature gate)
// ═══════════════════════════════════════════════════════════════════════════

pub use config::{
    ConservedVariable, Euler3DCircuitSizing, Euler3DParams, QttOperation,
    StrangStage, NUM_CONSERVED_VARIABLES, NUM_STRANG_STAGES, PHYS_DIM,
    Q16_SCALE, ROWS_PER_CONSERVATION, ROWS_PER_FP_MAC, ROWS_PER_SV_ORDER,
};

pub use witness::{
    ConservationWitness, ContractionWitness, Euler3DWitness, HashWitness,
    StrangStageWitness, SvdTruncationWitness, VariableSweepWitness,
    WitnessError, WitnessGenerator,
};

pub use halo2_impl::Euler3DCircuit;

pub use prover::{
    Euler3DProof, Euler3DProver, Euler3DProverStats, Euler3DVerificationResult,
    Euler3DVerifier,
};

// ═══════════════════════════════════════════════════════════════════════════
// Module-level convenience functions
// ═══════════════════════════════════════════════════════════════════════════

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

/// Run a complete prove-and-verify cycle for one Euler 3D timestep.
///
/// This is the main entry point for the Trustless Physics pipeline:
/// 1. Generates witness from input states and shift MPOs
/// 2. Creates the proof circuit
/// 3. Generates the ZK proof
/// 4. Verifies the proof
///
/// Returns (proof, verification_result) on success.
pub fn prove_euler3d_timestep(
    params: Euler3DParams,
    input_states: &[MPS],
    shift_mpos: &[MPO],
) -> Result<(Euler3DProof, Euler3DVerificationResult), String> {
    // Validate inputs
    if input_states.len() != NUM_CONSERVED_VARIABLES {
        return Err(format!(
            "Expected {} input states, got {}",
            NUM_CONSERVED_VARIABLES,
            input_states.len()
        ));
    }
    if shift_mpos.len() != 3 {
        return Err(format!(
            "Expected 3 shift MPOs (one per axis), got {}",
            shift_mpos.len()
        ));
    }

    // Generate proof
    let mut prover = Euler3DProver::new(params)?;
    let proof = prover.prove(input_states, shift_mpos)?;

    // Verify proof
    #[cfg(not(feature = "halo2"))]
    let result = {
        let verifier = Euler3DVerifier::new();
        verifier.verify(&proof)?
    };

    #[cfg(feature = "halo2")]
    let result = {
        let verifier = Euler3DVerifier::from_prover(&prover);
        let public_inputs = proof.reconstruct_public_inputs();
        verifier.verify(&proof, &public_inputs)?
    };

    Ok((proof, result))
}

/// Create a test configuration with small parameters suitable for
/// unit tests and CI.
pub fn test_config() -> Euler3DParams {
    Euler3DParams::test_small()
}

/// Create a production configuration for real-world proving.
pub fn production_config() -> Euler3DParams {
    Euler3DParams::production()
}

/// Create test input states (5 MPS, one per conserved variable).
pub fn make_test_states(params: &Euler3DParams) -> Vec<MPS> {
    (0..NUM_CONSERVED_VARIABLES)
        .map(|i| {
            let mut mps = MPS::new(params.num_sites(), params.chi_max, 2);
            // Give each state a small non-zero value so conservation has
            // something to check.
            if !mps.cores.is_empty() && !mps.cores[0].data.is_empty() {
                let base_val = Q16::from_f64(0.1 * (i as f64 + 1.0));
                mps.cores[0].data[0] = base_val;
            }
            mps
        })
        .collect()
}

/// Create test shift MPOs (3, one per spatial axis).
pub fn make_test_shift_mpos(params: &Euler3DParams) -> Vec<MPO> {
    (0..3)
        .map(|_| MPO::identity(params.num_sites(), 2))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Module-level tests
// ═══════════════════════════════════════════════════════════════════════════

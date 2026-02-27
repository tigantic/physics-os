//! Prover and verifier for the Euler 3D proof circuit.
//!
//! Provides stub prover/verifier implementation.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::circuit::Euler3DCircuit;
use super::config::{Euler3DParams, NUM_CONSERVED_VARIABLES};

// ═══════════════════════════════════════════════════════════════════════════
// Proof Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A ZK proof for one Euler 3D timestep.
#[derive(Clone, Debug)]
pub struct Euler3DProof {
    /// Raw proof bytes (serialized proof).
    pub proof_bytes: Vec<u8>,

    /// Proof generation time in milliseconds.
    pub generation_time_ms: u64,

    /// Number of constraints in the circuit.
    pub num_constraints: usize,

    /// Circuit k parameter (2^k rows).
    pub k: u32,

    /// Physics parameters used for the proof.
    pub params: Euler3DParams,

    /// Conservation residuals (public outputs).
    pub conservation_residuals: Vec<Q16>,

    /// Input state hash limbs (public input).
    pub input_state_hash_limbs: [u64; 4],

    /// Output state hash limbs (public input).
    pub output_state_hash_limbs: [u64; 4],

    /// Parameters hash limbs (public input).
    pub params_hash_limbs: [u64; 4],
}

impl Euler3DProof {
    /// Serialize proof to bytes for transmission/storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic + version
        bytes.extend(b"E3DP");
        bytes.extend(1u32.to_le_bytes());

        // Proof data
        bytes.extend((self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend(&self.proof_bytes);

        // Timing
        bytes.extend(self.generation_time_ms.to_le_bytes());

        // Constraints + k
        bytes.extend((self.num_constraints as u64).to_le_bytes());
        bytes.extend(self.k.to_le_bytes());

        // Conservation residuals
        bytes.extend((self.conservation_residuals.len() as u32).to_le_bytes());
        for r in &self.conservation_residuals {
            bytes.extend(r.raw.to_le_bytes());
        }

        // Hashes
        for limb in &self.input_state_hash_limbs {
            bytes.extend(limb.to_le_bytes());
        }
        for limb in &self.output_state_hash_limbs {
            bytes.extend(limb.to_le_bytes());
        }
        for limb in &self.params_hash_limbs {
            bytes.extend(limb.to_le_bytes());
        }

        bytes
    }

    /// Proof size in bytes.
    pub fn size(&self) -> usize {
        self.proof_bytes.len()
    }
}

/// Verification result for an Euler 3D proof.
#[derive(Debug, Clone)]
pub struct Euler3DVerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,

    /// Verification time in microseconds.
    pub verification_time_us: u64,

    /// Number of constraints verified.
    pub num_constraints: usize,

    /// Conservation residuals from the proof.
    pub conservation_residuals: Vec<Q16>,

    /// Grid configuration.
    pub grid_bits: usize,
    /// Bond dimension.
    pub chi_max: usize,
}

/// Statistics for prover performance tracking.
#[derive(Debug, Default, Clone)]
pub struct Euler3DProverStats {
    /// Total proofs generated.
    pub total_proofs: usize,

    /// Total proving time in ms.
    pub total_time_ms: u64,

    /// Average proof time in ms.
    pub avg_time_ms: f64,

    /// Total proof bytes generated.
    pub total_bytes: usize,

    /// Total constraints proved.
    pub total_constraints: usize,
}

impl Euler3DProverStats {
    /// Record a new proof in the statistics.
    pub fn record(&mut self, proof: &Euler3DProof) {
        self.total_proofs += 1;
        self.total_time_ms += proof.generation_time_ms;
        self.avg_time_ms = self.total_time_ms as f64 / self.total_proofs as f64;
        self.total_bytes += proof.size();
        self.total_constraints += proof.num_constraints;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Prover/Verifier
// ═══════════════════════════════════════════════════════════════════════════

pub mod stub_prover {
    //! Euler 3D prover/verifier (structural validation, no cryptographic proof).
    use super::*;

    /// Euler 3D prover (structural validation, no cryptographic proof).
    pub struct Euler3DProver {
        /// Physics parameters.
        euler_params: Euler3DParams,

        /// Statistics.
        stats: Euler3DProverStats,
    }

    impl Euler3DProver {
        /// Create a stub prover (no key generation needed).
        pub fn new(euler_params: Euler3DParams) -> Result<Self, String> {
            Ok(Self {
                euler_params,
                stats: Euler3DProverStats::default(),
            })
        }

        /// Generate a simulated proof (validates witness, no ZK).
        pub fn prove(
            &mut self,
            input_states: &[MPS],
            shift_mpos: &[MPO],
        ) -> Result<Euler3DProof, String> {
            let start = Instant::now();

            let circuit = Euler3DCircuit::new(
                self.euler_params.clone(),
                input_states,
                shift_mpos,
            )?;

            // Validate witness constraints
            circuit.validate_witness()?;

            let num_constraints = circuit.estimate_constraints();
            let generation_time_ms = start.elapsed().as_millis() as u64;

            // Simulated proof (800 bytes, typical KZG proof size)
            let proof = Euler3DProof {
                proof_bytes: vec![0u8; 800],
                generation_time_ms,
                num_constraints,
                k: circuit.sizing.k,
                params: self.euler_params.clone(),
                conservation_residuals: circuit
                    .witness
                    .conservation
                    .residuals
                    .clone(),
                input_state_hash_limbs: circuit
                    .witness
                    .hashes
                    .input_state_hash_limbs,
                output_state_hash_limbs: circuit
                    .witness
                    .hashes
                    .output_state_hash_limbs,
                params_hash_limbs: circuit.witness.hashes.params_hash_limbs,
            };

            self.stats.record(&proof);

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &Euler3DProverStats {
            &self.stats
        }
    }

    /// Euler 3D verifier (structural validation, no cryptographic proof).
    pub struct Euler3DVerifier {
        /// Simulated verification delay in microseconds.
        pub simulated_delay_us: u64,
    }

    impl Default for Euler3DVerifier {
        fn default() -> Self {
            Self {
                simulated_delay_us: 1000,
            }
        }
    }

    impl Euler3DVerifier {
        /// Create a new stub verifier.
        pub fn new() -> Self {
            Self::default()
        }

        /// Verify a proof (stub: checks proof structure, returns valid).
        pub fn verify(
            &self,
            proof: &Euler3DProof,
        ) -> Result<Euler3DVerificationResult, String> {
            let start = Instant::now();

            // Structural validation
            if proof.proof_bytes.is_empty() {
                return Err("Empty proof bytes".to_string());
            }
            if proof.conservation_residuals.len() != NUM_CONSERVED_VARIABLES {
                return Err(format!(
                    "Wrong conservation residual count: {} (expected {})",
                    proof.conservation_residuals.len(),
                    NUM_CONSERVED_VARIABLES,
                ));
            }

            // Check conservation residuals within tolerance
            for (i, residual) in proof.conservation_residuals.iter().enumerate() {
                if residual.raw.abs() > proof.params.conservation_tolerance.raw {
                    return Err(format!(
                        "Conservation violation for variable {}: |{}| > {}",
                        i,
                        residual.to_f64(),
                        proof.params.conservation_tolerance.to_f64(),
                    ));
                }
            }

            // Simulate verification delay
            std::thread::sleep(std::time::Duration::from_micros(
                self.simulated_delay_us,
            ));

            let verification_time_us = start.elapsed().as_micros() as u64;

            Ok(Euler3DVerificationResult {
                valid: true,
                verification_time_us,
                num_constraints: proof.num_constraints,
                conservation_residuals: proof.conservation_residuals.clone(),
                grid_bits: proof.params.grid_bits,
                chi_max: proof.params.chi_max,
            })
        }
    }
}

pub use stub_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

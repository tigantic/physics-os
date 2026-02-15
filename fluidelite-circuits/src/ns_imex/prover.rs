//! Prover and verifier for the NS-IMEX proof circuit.
//!
//! Provides stub prover/verifier implementation.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::circuit::NSIMEXCircuit;
use super::config::NSIMEXParams;

// ═══════════════════════════════════════════════════════════════════════════
// Proof Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A ZK proof for one NS-IMEX timestep.
#[derive(Clone, Debug)]
pub struct NSIMEXProof {
    /// Raw proof bytes (serialized proof).
    pub proof_bytes: Vec<u8>,

    /// Proof generation time in milliseconds.
    pub generation_time_ms: u64,

    /// Number of constraints in the circuit.
    pub num_constraints: usize,

    /// Circuit k parameter (2^k rows).
    pub k: u32,

    /// Solver parameters used for the proof.
    pub params: NSIMEXParams,

    /// Kinetic energy residual (public output).
    pub ke_residual: Q16,

    /// Enstrophy residual (public output).
    pub enstrophy_residual: Q16,

    /// Divergence residual (public output).
    pub divergence_residual: Q16,

    /// Input state hash limbs (public input).
    pub input_state_hash_limbs: [u64; 4],

    /// Output state hash limbs (public input).
    pub output_state_hash_limbs: [u64; 4],

    /// Parameters hash limbs (public input).
    pub params_hash_limbs: [u64; 4],
}

impl NSIMEXProof {
    /// Serialize proof to bytes for transmission/storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic + version
        bytes.extend(b"NSIP");
        bytes.extend(1u32.to_le_bytes());

        // Proof data
        bytes.extend((self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend(&self.proof_bytes);

        // Timing
        bytes.extend(self.generation_time_ms.to_le_bytes());

        // Constraints + k
        bytes.extend((self.num_constraints as u64).to_le_bytes());
        bytes.extend(self.k.to_le_bytes());

        // Diagnostics
        bytes.extend(self.ke_residual.raw.to_le_bytes());
        bytes.extend(self.enstrophy_residual.raw.to_le_bytes());
        bytes.extend(self.divergence_residual.raw.to_le_bytes());

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

    /// Deserialize proof from bytes (inverse of `to_bytes`).
    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut pos = 0;

        // Magic
        if data.len() < 4 || &data[0..4] != b"NSIP" {
            return Err("Invalid NS-IMEX proof magic".into());
        }
        pos += 4;

        // Version
        if data.len() < pos + 4 {
            return Err("Truncated proof: missing version".into());
        }
        let _version = u32::from_le_bytes(data[pos..pos + 4].try_into()?);
        pos += 4;

        // Proof data length
        if data.len() < pos + 4 {
            return Err("Truncated proof: missing proof length".into());
        }
        let proof_len = u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
        pos += 4;

        // Proof bytes
        if data.len() < pos + proof_len {
            return Err("Truncated proof: missing proof bytes".into());
        }
        let proof_bytes = data[pos..pos + proof_len].to_vec();
        pos += proof_len;

        // Timing
        if data.len() < pos + 8 {
            return Err("Truncated proof: missing timing".into());
        }
        let generation_time_ms = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
        pos += 8;

        // Constraints
        if data.len() < pos + 8 {
            return Err("Truncated proof: missing constraints".into());
        }
        let num_constraints = u64::from_le_bytes(data[pos..pos + 8].try_into()?) as usize;
        pos += 8;

        // k
        if data.len() < pos + 4 {
            return Err("Truncated proof: missing k".into());
        }
        let k = u32::from_le_bytes(data[pos..pos + 4].try_into()?);
        pos += 4;

        // Diagnostics: ke_residual, enstrophy_residual, divergence_residual
        if data.len() < pos + 24 {
            return Err("Truncated proof: missing diagnostics".into());
        }
        let ke_raw = i64::from_le_bytes(data[pos..pos + 8].try_into()?);
        pos += 8;
        let enstrophy_raw = i64::from_le_bytes(data[pos..pos + 8].try_into()?);
        pos += 8;
        let divergence_raw = i64::from_le_bytes(data[pos..pos + 8].try_into()?);
        pos += 8;

        // Hash limbs: 3 × 4 × 8 = 96 bytes
        if data.len() < pos + 96 {
            return Err("Truncated proof: missing hash limbs".into());
        }

        let mut input_state_hash_limbs = [0u64; 4];
        for limb in &mut input_state_hash_limbs {
            *limb = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
        }

        let mut output_state_hash_limbs = [0u64; 4];
        for limb in &mut output_state_hash_limbs {
            *limb = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
        }

        let mut params_hash_limbs = [0u64; 4];
        for limb in &mut params_hash_limbs {
            *limb = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
        }

        Ok(Self {
            proof_bytes,
            generation_time_ms,
            num_constraints,
            k,
            params: NSIMEXParams::test_small(), // Params not serialized; caller provides
            ke_residual: Q16::from_raw(ke_raw),
            enstrophy_residual: Q16::from_raw(enstrophy_raw),
            divergence_residual: Q16::from_raw(divergence_raw),
            input_state_hash_limbs,
            output_state_hash_limbs,
            params_hash_limbs,
        })
    }

}

/// Verification result for an NS-IMEX proof.
#[derive(Debug, Clone)]
pub struct NSIMEXVerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,

    /// Verification time in microseconds.
    pub verification_time_us: u64,

    /// Number of constraints verified.
    pub num_constraints: usize,

    /// KE residual from the proof.
    pub ke_residual: Q16,

    /// Enstrophy residual from the proof.
    pub enstrophy_residual: Q16,

    /// Divergence residual from the proof.
    pub divergence_residual: Q16,

    /// Grid configuration.
    pub grid_bits: usize,

    /// Bond dimension.
    pub chi_max: usize,

    /// Reynolds number.
    pub reynolds_number: f64,
}

/// Statistics for prover performance tracking.
#[derive(Debug, Default, Clone)]
pub struct NSIMEXProverStats {
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

impl NSIMEXProverStats {
    /// Record a new proof in the statistics.
    pub fn record(&mut self, proof: &NSIMEXProof) {
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
    //! NS-IMEX prover/verifier (structural validation, no cryptographic proof).
    use super::*;

    /// NS-IMEX prover (structural validation, no cryptographic proof).
    pub struct NSIMEXProver {
        ns_params: NSIMEXParams,
        stats: NSIMEXProverStats,
    }

    impl NSIMEXProver {
        /// Create a stub prover (no key generation needed).
        pub fn new(ns_params: NSIMEXParams) -> Result<Self, String> {
            Ok(Self {
                ns_params,
                stats: NSIMEXProverStats::default(),
            })
        }

        /// Generate a simulated proof (validates witness, no ZK).
        pub fn prove(
            &mut self,
            input_states: &[MPS],
            shift_mpos: &[MPO],
        ) -> Result<NSIMEXProof, String> {
            let start = Instant::now();

            let circuit = NSIMEXCircuit::new(
                self.ns_params.clone(),
                input_states,
                shift_mpos,
            )?;

            // Validate witness constraints
            circuit.validate_witness()?;

            let num_constraints = circuit.estimate_constraints();
            let generation_time_ms = start.elapsed().as_millis() as u64;

            let ke_residual = Q16::from_raw(
                circuit.witness.kinetic_energy_after.raw
                    - circuit.witness.kinetic_energy_before.raw,
            );
            let ens_residual = Q16::from_raw(
                circuit.witness.enstrophy_after.raw
                    - circuit.witness.enstrophy_before.raw,
            );

            let proof = NSIMEXProof {
                proof_bytes: vec![0u8; 800],
                generation_time_ms,
                num_constraints,
                k: circuit.k(),
                params: self.ns_params.clone(),
                ke_residual,
                enstrophy_residual: ens_residual,
                divergence_residual: circuit.witness.divergence_residual,
                input_state_hash_limbs: circuit.witness.input_hash_limbs,
                output_state_hash_limbs: circuit.witness.output_hash_limbs,
                params_hash_limbs: circuit.witness.params_hash_limbs,
            };

            self.stats.record(&proof);

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &NSIMEXProverStats {
            &self.stats
        }
    }

    /// NS-IMEX verifier (structural validation, no cryptographic proof).
    pub struct NSIMEXVerifier {
        /// Simulated verification delay in microseconds.
        pub simulated_delay_us: u64,
    }

    impl Default for NSIMEXVerifier {
        fn default() -> Self {
            Self {
                simulated_delay_us: 1000,
            }
        }
    }

    impl NSIMEXVerifier {
        /// Create a new verifier.
        pub fn new() -> Self {
            Self::default()
        }

        /// Verify a proof (checks proof structure and conservation bounds).
        pub fn verify(
            &self,
            proof: &NSIMEXProof,
        ) -> Result<NSIMEXVerificationResult, String> {
            let start = Instant::now();

            // Structural validation
            if proof.proof_bytes.is_empty() {
                return Err("Empty proof bytes".to_string());
            }

            // Check conservation bounds
            if proof.ke_residual.raw.unsigned_abs() as i64
                > proof.params.conservation_tolerance.raw
            {
                return Err(format!(
                    "KE conservation violation: |{}| > {}",
                    proof.ke_residual.to_f64(),
                    proof.params.conservation_tolerance.to_f64(),
                ));
            }

            // Check divergence bound
            if proof.divergence_residual.raw.unsigned_abs() as i64
                > proof.params.divergence_tolerance.raw
            {
                return Err(format!(
                    "Divergence violation: |{}| > {}",
                    proof.divergence_residual.to_f64(),
                    proof.params.divergence_tolerance.to_f64(),
                ));
            }

            // Simulate verification delay
            std::thread::sleep(std::time::Duration::from_micros(
                self.simulated_delay_us,
            ));

            let verification_time_us = start.elapsed().as_micros() as u64;

            Ok(NSIMEXVerificationResult {
                valid: true,
                verification_time_us,
                num_constraints: proof.num_constraints,
                ke_residual: proof.ke_residual,
                enstrophy_residual: proof.enstrophy_residual,
                divergence_residual: proof.divergence_residual,
                grid_bits: proof.params.grid_bits,
                chi_max: proof.params.chi_max,
                reynolds_number: proof.params.reynolds_number(),
            })
        }
    }
}

pub use stub_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

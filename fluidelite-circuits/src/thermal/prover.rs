//! Prover and verifier for the Thermal/Heat Equation proof circuit.
//!
//! Two-backend implementation (selected at compile time via features):
//!   - **`stark`** (default): Winterfell STARK — transparent, post-quantum, no trusted setup
//!   - **stub**: Structural validation only (no cryptographic proof)
//!
//! Priority: stark > stub.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::circuit::ThermalCircuit;
use super::config::ThermalParams;

// ═══════════════════════════════════════════════════════════════════════════
// Proof Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A ZK proof for one thermal timestep.
#[derive(Clone, Debug)]
pub struct ThermalProof {
    /// Raw proof bytes (serialized proof).
    pub proof_bytes: Vec<u8>,

    /// Proof generation time in milliseconds.
    pub generation_time_ms: u64,

    /// Number of constraints in the circuit.
    pub num_constraints: usize,

    /// Circuit k parameter (2^k rows).
    pub k: u32,

    /// Physics parameters used for the proof.
    pub params: ThermalParams,

    /// Energy conservation residual (public output).
    pub conservation_residual: Q16,

    /// CG solve final residual norm (diagnostic).
    pub cg_residual_norm: Q16,

    /// Number of CG iterations performed.
    pub cg_iterations: usize,

    /// Input state hash limbs (public input).
    pub input_state_hash_limbs: [u64; 4],

    /// Output state hash limbs (public input).
    pub output_state_hash_limbs: [u64; 4],

    /// Parameters hash limbs (public input).
    pub params_hash_limbs: [u64; 4],

    /// Global step index for this proof (used as STARK public input).
    pub step_index: u64,

    /// Initial energy (Q16 raw) — needed for exact STARK public input reconstruction.
    pub initial_energy_raw: i64,

    /// Final energy (Q16 raw) — needed for exact STARK public input reconstruction.
    pub final_energy_raw: i64,
}

impl ThermalProof {
    /// Serialize proof to bytes for transmission/storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic + version
        bytes.extend(b"THEP");
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
        bytes.extend(self.conservation_residual.raw.to_le_bytes());
        bytes.extend(self.cg_residual_norm.raw.to_le_bytes());
        bytes.extend((self.cg_iterations as u32).to_le_bytes());

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

        // Step index
        bytes.extend(self.step_index.to_le_bytes());

        // Energy (exact values for STARK public input reconstruction)
        bytes.extend(self.initial_energy_raw.to_le_bytes());
        bytes.extend(self.final_energy_raw.to_le_bytes());

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
        if data.len() < 4 || &data[0..4] != b"THEP" {
            return Err("Invalid Thermal proof magic".into());
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

        // Diagnostics
        if data.len() < pos + 8 + 8 + 4 {
            return Err("Truncated proof: missing diagnostics".into());
        }
        let conservation_residual =
            Q16::from_raw(i64::from_le_bytes(data[pos..pos + 8].try_into()?));
        pos += 8;
        let cg_residual_norm =
            Q16::from_raw(i64::from_le_bytes(data[pos..pos + 8].try_into()?));
        pos += 8;
        let cg_iterations =
            u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
        pos += 4;

        // Hashes
        if data.len() < pos + 96 {
            return Err("Truncated proof: missing hashes".into());
        }
        let mut input_state_hash_limbs = [0u64; 4];
        let mut output_state_hash_limbs = [0u64; 4];
        let mut params_hash_limbs = [0u64; 4];

        for limb in &mut input_state_hash_limbs {
            *limb = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
        }
        for limb in &mut output_state_hash_limbs {
            *limb = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
        }
        for limb in &mut params_hash_limbs {
            *limb = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
        }

        // Step index (optional for backward compat)
        let step_index = if data.len() >= pos + 8 {
            let v = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
            v
        } else {
            0
        };

        // Energy (optional for backward compat)
        let initial_energy_raw = if data.len() >= pos + 8 {
            let v = i64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
            v
        } else {
            0
        };
        let final_energy_raw = if data.len() >= pos + 8 {
            let v = i64::from_le_bytes(data[pos..pos + 8].try_into()?);
            pos += 8;
            let _ = pos;
            v
        } else {
            0
        };

        Ok(Self {
            proof_bytes,
            generation_time_ms,
            num_constraints,
            k,
            params: ThermalParams::test_small(), // Params not stored in serialized form for now
            conservation_residual,
            cg_residual_norm,
            cg_iterations,
            input_state_hash_limbs,
            output_state_hash_limbs,
            params_hash_limbs,
            step_index,
            initial_energy_raw,
            final_energy_raw,
        })
    }

}

/// Verification result for a thermal proof.
#[derive(Debug, Clone)]
pub struct ThermalVerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,

    /// Verification time in microseconds.
    pub verification_time_us: u64,

    /// Number of constraints verified.
    pub num_constraints: usize,

    /// Conservation residual from the proof.
    pub conservation_residual: Q16,

    /// CG iterations used.
    pub cg_iterations: usize,

    /// Grid configuration.
    pub grid_bits: usize,
    /// Bond dimension.
    pub chi_max: usize,
}

/// Statistics for prover performance tracking.
#[derive(Debug, Default, Clone)]
pub struct ThermalProverStats {
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

impl ThermalProverStats {
    /// Record a new proof in the statistics.
    pub fn record(&mut self, proof: &ThermalProof) {
        self.total_proofs += 1;
        self.total_time_ms += proof.generation_time_ms;
        self.avg_time_ms = self.total_time_ms as f64 / self.total_proofs as f64;
        self.total_bytes += proof.size();
        self.total_constraints += proof.num_constraints;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stub Prover/Verifier (without STARK)
// ═══════════════════════════════════════════════════════════════════════════

pub mod stub_prover {
    //! Thermal prover/verifier (structural validation, no cryptographic proof).
    use super::*;

    /// Thermal prover (structural validation, no cryptographic proof).
    pub struct ThermalProver {
        /// Physics parameters.
        thermal_params: ThermalParams,

        /// Statistics.
        stats: ThermalProverStats,
    }

    impl ThermalProver {
        /// Create a stub prover (no key generation needed).
        pub fn new(thermal_params: ThermalParams) -> Result<Self, String> {
            Ok(Self {
                thermal_params,
                stats: ThermalProverStats::default(),
            })
        }

        /// Generate a simulated proof (validates witness, no ZK).
        pub fn prove(
            &mut self,
            input_states: &[MPS],
            laplacian_mpos: &[MPO],
        ) -> Result<ThermalProof, String> {
            let start = Instant::now();

            let circuit = ThermalCircuit::new(
                self.thermal_params.clone(),
                input_states,
                laplacian_mpos,
            )?;

            // Validate witness constraints
            circuit.validate_witness()?;

            let num_constraints = circuit.estimate_constraints();
            let generation_time_ms = start.elapsed().as_millis() as u64;

            // Deterministic simulated proof bytes (800 bytes)
            let mut proof_bytes = vec![0u8; 800];
            // FNV-like hash for deterministic but non-trivial proof bytes
            let mut hash: u64 = 0xcbf29ce484222325;
            for val in circuit.witness.input_state.flat_data() {
                hash ^= val.raw as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            for (i, byte) in proof_bytes.iter_mut().enumerate() {
                *byte = ((hash.wrapping_mul(i as u64 + 1)) >> (i % 8 * 8)) as u8;
            }

            let proof = ThermalProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
                k: circuit.sizing.k,
                params: self.thermal_params.clone(),
                conservation_residual: circuit.witness.conservation.residual,
                cg_residual_norm: circuit.witness.implicit_solve.final_residual_norm,
                cg_iterations: circuit.witness.implicit_solve.num_iterations,
                input_state_hash_limbs: circuit
                    .witness
                    .hashes
                    .input_state_hash_limbs,
                output_state_hash_limbs: circuit
                    .witness
                    .hashes
                    .output_state_hash_limbs,
                params_hash_limbs: circuit.witness.hashes.params_hash_limbs,
                step_index: 0,
                initial_energy_raw: circuit.witness.conservation.integral_before.raw,
                final_energy_raw: circuit.witness.conservation.integral_after.raw,
            };

            self.stats.record(&proof);

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &ThermalProverStats {
            &self.stats
        }
    }

    /// Thermal verifier (structural validation, no cryptographic proof).
    pub struct ThermalVerifier {
        /// Simulated verification delay in microseconds.
        pub simulated_delay_us: u64,
    }

    impl Default for ThermalVerifier {
        fn default() -> Self {
            Self {
                simulated_delay_us: 1000,
            }
        }
    }

    impl ThermalVerifier {
        /// Create a new stub verifier.
        pub fn new() -> Self {
            Self::default()
        }

        /// Verify a proof (stub: checks proof structure, returns valid).
        pub fn verify(
            &self,
            proof: &ThermalProof,
        ) -> Result<ThermalVerificationResult, String> {
            let start = Instant::now();

            // Structural validation
            if proof.proof_bytes.is_empty() {
                return Err("Empty proof bytes".to_string());
            }

            // Check conservation residual within tolerance
            if proof.conservation_residual.raw.abs()
                > proof.params.conservation_tol.raw
            {
                return Err(format!(
                    "Conservation violation: |{}| > {}",
                    proof.conservation_residual.to_f64(),
                    proof.params.conservation_tol.to_f64(),
                ));
            }

            // Simulate verification delay
            std::thread::sleep(std::time::Duration::from_micros(
                self.simulated_delay_us,
            ));

            let verification_time_us = start.elapsed().as_micros() as u64;

            Ok(ThermalVerificationResult {
                valid: true,
                verification_time_us,
                num_constraints: proof.num_constraints,
                conservation_residual: proof.conservation_residual,
                cg_iterations: proof.cg_iterations,
                grid_bits: proof.params.grid_bits,
                chi_max: proof.params.chi_max,
            })
        }
    }
}

#[cfg(not(feature = "stark"))]
pub use stub_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// STARK Prover/Verifier (Winterfell, no trusted setup)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "stark")]
pub mod stark_prover {
    //! STARK-based thermal prover/verifier using Winterfell.
    //!
    //! Uses a transparent STARK:
    //! - No trusted setup (hash-based commitment)
    //! - Post-quantum secure (no discrete log)
    //! - GPU-parallelizable (FFT + Merkle hashing)

    use super::*;
    use crate::thermal::stark_impl::{
        self, prove_thermal_stark, verify_thermal_stark, TimestepPhysics,
        ThermalStarkInputs,
    };

    /// STARK thermal prover using Winterfell (Goldilocks + FRI).
    pub struct ThermalProver {
        /// Physics parameters.
        thermal_params: ThermalParams,

        /// Statistics.
        stats: ThermalProverStats,

        /// Accumulated timestep physics for multi-step batch proving.
        timestep_buffer: Vec<TimestepPhysics>,

        /// Cached STARK proof bytes (from most recent batch prove).
        cached_stark_proof: Option<Vec<u8>>,

        /// Cached public inputs (from most recent batch prove).
        cached_pub_inputs: Option<ThermalStarkInputs>,

        /// Output state from the most recent `prove()` call.
        /// Used for state chaining: output of step N → input of step N+1.
        last_output_state: Option<Vec<MPS>>,

        /// Global step counter for the simulation sequence.
        /// Monotonically increasing, ensures each per-step trace is unique.
        global_step_counter: u64,
    }

    impl ThermalProver {
        /// Create a new STARK prover. No key generation needed (transparent setup).
        pub fn new(thermal_params: ThermalParams) -> Result<Self, String> {
            eprintln!("[Thermal-STARK] Prover initialized (no trusted setup)");
            eprintln!(
                "  Field: Goldilocks (p = 2^64 - 2^32 + 1)");
            eprintln!(
                "  Commitment: FRI + Blake3 Merkle (post-quantum)");
            eprintln!(
                "  Params: grid_bits={}, chi_max={}, dt={:.6}, alpha={:.6}",
                thermal_params.grid_bits,
                thermal_params.chi_max,
                thermal_params.dt.to_f64(),
                thermal_params.alpha.to_f64(),
            );

            Ok(Self {
                thermal_params,
                last_output_state: None,
                stats: ThermalProverStats::default(),
                timestep_buffer: Vec::new(),
                cached_stark_proof: None,
                cached_pub_inputs: None,
                global_step_counter: 0,
            })
        }

        /// Generate a proof for one timestep.
        ///
        /// Internally generates the witness via `ThermalCircuit`, extracts physics
        /// summary data, and produces a per-step STARK proof.
        pub fn prove(
            &mut self,
            input_states: &[MPS],
            laplacian_mpos: &[MPO],
        ) -> Result<ThermalProof, String> {
            let start = Instant::now();

            // Generate witness using the existing ThermalCircuit infrastructure.
            let circuit = ThermalCircuit::new(
                self.thermal_params.clone(),
                input_states,
                laplacian_mpos,
            )?;
            circuit.validate_witness()?;

            // Capture the evolved output state for state chaining.
            // The thin `Mps` type is converted back to full `MPS` so the
            // caller can feed it as input to the next timestep.
            self.last_output_state = Some(vec![circuit.witness.output_state.to_full()]);

            // Extract physics summary from witness and stamp with global step.
            let mut physics = extract_physics_from_witness(&circuit);
            let step_idx = self.global_step_counter;
            physics.global_step = step_idx;
            self.global_step_counter += 1;

            // Build single-step trace and prove.
            let (proof_bytes, pub_inputs, trace_len, _gen_ms) =
                prove_thermal_stark(
                    &[physics.clone()],
                    self.thermal_params.dt,
                    self.thermal_params.alpha,
                )?;

            let generation_time_ms = start.elapsed().as_millis() as u64;
            let num_constraints = trace_len * stark_impl::NUM_CONSTRAINTS;

            // Buffer this step for potential batch proving later.
            self.timestep_buffer.push(physics);

            let proof = ThermalProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
                k: (trace_len as f64).log2().ceil() as u32,
                params: self.thermal_params.clone(),
                conservation_residual: circuit.witness.conservation.residual,
                cg_residual_norm: circuit.witness.implicit_solve.final_residual_norm,
                cg_iterations: circuit.witness.implicit_solve.num_iterations,
                input_state_hash_limbs: circuit
                    .witness
                    .hashes
                    .input_state_hash_limbs,
                output_state_hash_limbs: circuit
                    .witness
                    .hashes
                    .output_state_hash_limbs,
                params_hash_limbs: circuit.witness.hashes.params_hash_limbs,
                step_index: step_idx,
                initial_energy_raw: stark_impl::felt_to_q16(pub_inputs.initial_energy).raw,
                final_energy_raw: stark_impl::felt_to_q16(pub_inputs.final_energy).raw,
            };

            self.stats.record(&proof);

            eprintln!(
                "[Thermal-STARK] Proof generated: {} constraints, {} bytes, {:.1}ms, {} CG iters",
                num_constraints,
                proof.size(),
                generation_time_ms as f64,
                proof.cg_iterations,
            );

            Ok(proof)
        }

        /// Generate a STARK proof covering ALL buffered timesteps at once.
        ///
        /// This is the efficient path: one STARK proof for the entire simulation
        /// sequence, rather than one per timestep. Call after all individual
        /// `prove()` calls to get the batch proof.
        pub fn prove_batch(&mut self) -> Result<ThermalProof, String> {
            if self.timestep_buffer.is_empty() {
                return Err("No timesteps buffered for batch proving".to_string());
            }

            let start = Instant::now();

            let (proof_bytes, pub_inputs, trace_len, _gen_ms) =
                prove_thermal_stark(
                    &self.timestep_buffer,
                    self.thermal_params.dt,
                    self.thermal_params.alpha,
                )?;

            let generation_time_ms = start.elapsed().as_millis() as u64;
            let num_constraints = trace_len * stark_impl::NUM_CONSTRAINTS;
            let num_steps = self.timestep_buffer.len();

            // Cache for verification.
            self.cached_stark_proof = Some(proof_bytes.clone());
            self.cached_pub_inputs = Some(pub_inputs.clone());

            // Use first/last timestep for the proof metadata.
            let first = &self.timestep_buffer[0];
            let last = &self.timestep_buffer[num_steps - 1];

            let proof = ThermalProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
                k: (trace_len as f64).log2().ceil() as u32,
                params: self.thermal_params.clone(),
                step_index: first.global_step,
                conservation_residual: last.conservation_residual,
                cg_residual_norm: last.cg_residual,
                cg_iterations: 0, // batch proof — individual CG iters not tracked
                input_state_hash_limbs: first.input_hash_limbs,
                output_state_hash_limbs: last.output_hash_limbs,
                params_hash_limbs: [0; 4],
                initial_energy_raw: stark_impl::felt_to_q16(pub_inputs.initial_energy).raw,
                final_energy_raw: stark_impl::felt_to_q16(pub_inputs.final_energy).raw,
            };

            eprintln!(
                "[Thermal-STARK] Batch proof: {} steps → {} constraints, {} bytes, {:.1}ms",
                num_steps,
                num_constraints,
                proof.size(),
                generation_time_ms as f64,
            );

            Ok(proof)
        }

        /// Get the output state from the most recent `prove()` call.
        ///
        /// Returns the evolved temperature state T^{n+1} as a full `MPS`,
        /// ready to be used as the input state for the next timestep.
        /// This enables state chaining: each proof's output becomes the
        /// next proof's input, producing unique proofs per timestep.
        pub fn last_output_state(&self) -> Option<&Vec<MPS>> {
            self.last_output_state.as_ref()
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &ThermalProverStats {
            &self.stats
        }

        /// Get the cached batch proof public inputs (if available).
        pub fn batch_pub_inputs(&self) -> Option<&ThermalStarkInputs> {
            self.cached_pub_inputs.as_ref()
        }
    }

    /// STARK thermal verifier using Winterfell.
    pub struct ThermalVerifier;

    impl Default for ThermalVerifier {
        fn default() -> Self {
            Self
        }
    }

    impl ThermalVerifier {
        /// Create a new STARK verifier. No setup material needed.
        pub fn new() -> Self {
            Self
        }

        /// Verify a thermal proof using the STARK verifier.
        ///
        /// For single-step proofs, reconstructs public inputs from the proof data.
        /// For batch proofs, requires external public inputs.
        pub fn verify(
            &self,
            proof: &ThermalProof,
        ) -> Result<ThermalVerificationResult, String> {
            let start = Instant::now();

            // Structural validation.
            if proof.proof_bytes.is_empty() {
                return Err("Empty proof bytes".to_string());
            }

            // Check conservation residual within tolerance.
            if proof.conservation_residual.raw.abs()
                > proof.params.conservation_tol.raw
            {
                return Err(format!(
                    "Conservation violation: |{}| > {}",
                    proof.conservation_residual.to_f64(),
                    proof.params.conservation_tol.to_f64(),
                ));
            }

            // Reconstruct public inputs from proof data for STARK verification.
            let trace_length = 1usize << proof.k;
            let pub_inputs = ThermalStarkInputs {
                initial_energy: stark_impl::q16_to_felt(
                    Q16::from_raw(proof.initial_energy_raw),
                ),
                final_energy: stark_impl::q16_to_felt(
                    Q16::from_raw(proof.final_energy_raw),
                ),
                dt: stark_impl::q16_to_felt(proof.params.dt),
                alpha: stark_impl::q16_to_felt(proof.params.alpha),
                num_steps: trace_length - 1,
                trace_length,
                initial_input_hash: [
                    stark_impl::u64_to_felt(proof.input_state_hash_limbs[0]),
                    stark_impl::u64_to_felt(proof.input_state_hash_limbs[1]),
                    stark_impl::u64_to_felt(proof.input_state_hash_limbs[2]),
                    stark_impl::u64_to_felt(proof.input_state_hash_limbs[3]),
                ],
                initial_step: stark_impl::u64_to_felt(proof.step_index),
                final_output_hash: [
                    stark_impl::u64_to_felt(proof.output_state_hash_limbs[0]),
                    stark_impl::u64_to_felt(proof.output_state_hash_limbs[1]),
                    stark_impl::u64_to_felt(proof.output_state_hash_limbs[2]),
                    stark_impl::u64_to_felt(proof.output_state_hash_limbs[3]),
                ],
            };

            // STARK verification — no fallback. Failure is failure.
            let valid = verify_thermal_stark(&proof.proof_bytes, pub_inputs)
                .map_err(|e| format!("STARK verification failed: {e}"))?;

            let verification_time_us = start.elapsed().as_micros() as u64;

            Ok(ThermalVerificationResult {
                valid,
                verification_time_us,
                num_constraints: proof.num_constraints,
                conservation_residual: proof.conservation_residual,
                cg_iterations: proof.cg_iterations,
                grid_bits: proof.params.grid_bits,
                chi_max: proof.params.chi_max,
            })
        }

        /// Verify a batch STARK proof with explicit public inputs.
        pub fn verify_with_pub_inputs(
            &self,
            proof: &ThermalProof,
            pub_inputs: ThermalStarkInputs,
        ) -> Result<ThermalVerificationResult, String> {
            let start = Instant::now();

            let valid = verify_thermal_stark(&proof.proof_bytes, pub_inputs)?;
            let verification_time_us = start.elapsed().as_micros() as u64;

            Ok(ThermalVerificationResult {
                valid,
                verification_time_us,
                num_constraints: proof.num_constraints,
                conservation_residual: proof.conservation_residual,
                cg_iterations: proof.cg_iterations,
                grid_bits: proof.params.grid_bits,
                chi_max: proof.params.chi_max,
            })
        }
    }

    /// Extract physics summary from a completed ThermalCircuit witness.
    fn extract_physics_from_witness(circuit: &ThermalCircuit) -> TimestepPhysics {
        let w = &circuit.witness;

        // Energy from conservation witness.
        let energy = w.conservation.integral_before;

        // L2 norm: sum of squares of all core data.
        let energy_sq = {
            let total: i64 = w.input_state.flat_data().iter()
                .map(|v| {
                    let val = v.raw;
                    // Q16 * Q16 → Q32 — shift down by 16 to stay in Q16.
                    (val.saturating_mul(val)) >> 16
                })
                .sum();
            Q16::from_raw(total)
        };

        // Temperature bounds from input state core data.
        let (max_raw, min_raw) = {
            let data = w.input_state.flat_data();
            if data.is_empty() {
                (0i64, 0i64)
            } else {
                let max_v = data.iter().map(|v| v.raw).max().unwrap_or(0);
                let min_v = data.iter().map(|v| v.raw).min().unwrap_or(0);
                (max_v, min_v)
            }
        };

        // SVD max and rank from truncation witness.
        let (sv_max, rank) = if w.truncation.bond_data.is_empty() {
            (Q16::ZERO, 0)
        } else {
            let max_sv = w.truncation.bond_data.iter()
                .flat_map(|b| b.singular_values.iter())
                .map(|v| v.raw)
                .max()
                .unwrap_or(0);
            (Q16::from_raw(max_sv), w.truncation.output_rank)
        };

        TimestepPhysics {
            energy,
            energy_sq,
            max_temp: Q16::from_raw(max_raw),
            min_temp: Q16::from_raw(min_raw),
            source_energy: Q16::ZERO, // Source = 0 for default thermal config.
            cg_residual: w.implicit_solve.final_residual_norm,
            sv_max,
            global_step: 0, // Overwritten by caller with monotonic counter.
            rank,
            conservation_residual: w.conservation.residual,
            input_hash_limbs: w.hashes.input_state_hash_limbs,
            output_hash_limbs: w.hashes.output_state_hash_limbs,
        }
    }
}

/// Priority: stark > stub.
#[cfg(feature = "stark")]
pub use stark_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

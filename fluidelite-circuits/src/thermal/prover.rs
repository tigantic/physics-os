//! Prover and verifier for the Thermal/Heat Equation proof circuit.
//!
//! Wraps the Halo2 proving/verification API for the thermal circuit.
//! Provides both Halo2 and stub implementations.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::config::ThermalParams;
#[cfg(feature = "halo2")]
use super::config::ThermalCircuitSizing;
use super::halo2_impl::ThermalCircuit;

// ═══════════════════════════════════════════════════════════════════════════
// Proof Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A ZK proof for one thermal timestep.
#[derive(Clone, Debug)]
pub struct ThermalProof {
    /// Raw proof bytes (Halo2/KZG serialized proof).
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
        })
    }

    /// Reconstruct the public inputs vector from proof data.
    #[cfg(feature = "halo2")]
    pub fn reconstruct_public_inputs(&self) -> Vec<halo2_axiom::halo2curves::bn256::Fr> {
        use halo2_axiom::halo2curves::bn256::Fr;

        let mut inputs = Vec::new();

        // Input state hash (4 limbs)
        for limb in &self.input_state_hash_limbs {
            inputs.push(Fr::from(*limb));
        }

        // Output state hash (4 limbs)
        for limb in &self.output_state_hash_limbs {
            inputs.push(Fr::from(*limb));
        }

        // Params hash (4 limbs)
        for limb in &self.params_hash_limbs {
            inputs.push(Fr::from(*limb));
        }

        // Conservation residual
        if self.conservation_residual.raw >= 0 {
            inputs.push(Fr::from(self.conservation_residual.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.conservation_residual.raw) as u64));
        }

        // dt, alpha, chi_max, grid_bits
        if self.params.dt.raw >= 0 {
            inputs.push(Fr::from(self.params.dt.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.params.dt.raw) as u64));
        }
        if self.params.alpha.raw >= 0 {
            inputs.push(Fr::from(self.params.alpha.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.params.alpha.raw) as u64));
        }
        inputs.push(Fr::from(self.params.chi_max as u64));
        inputs.push(Fr::from(self.params.grid_bits as u64));

        inputs
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
// Halo2 Prover/Verifier
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
/// Halo2/KZG prover and verifier for the thermal proof circuit.
pub mod halo2_prover {
    use super::*;
    use halo2_axiom::{
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ProvingKey, VerifyingKey},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::SingleStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer,
            TranscriptWriterBuffer,
        },
    };
    use rand::rngs::OsRng;

    /// Thermal ZK Prover using Halo2/KZG.
    pub struct ThermalProver {
        /// KZG parameters.
        params_kzg: ParamsKZG<Bn256>,

        /// Proving key.
        pk: ProvingKey<G1Affine>,

        /// Verifying key.
        vk: VerifyingKey<G1Affine>,

        /// Physics parameters.
        thermal_params: ThermalParams,

        /// Accumulated statistics.
        stats: ThermalProverStats,
    }

    impl ThermalProver {
        /// Create a new prover. Performs one-time key generation.
        pub fn new(thermal_params: ThermalParams) -> Result<Self, String> {
            println!("[Thermal] Generating proving keys (one-time setup)...");
            let start = Instant::now();

            let sizing = ThermalCircuitSizing::from_params(&thermal_params);
            let k = sizing.k.max(14);

            let params_kzg = ParamsKZG::<Bn256>::setup(k, OsRng);

            let empty_states: Vec<MPS> = vec![
                MPS::new(thermal_params.num_sites(), thermal_params.chi_max, 2),
            ];
            let empty_mpos: Vec<MPO> = vec![
                MPO::identity(thermal_params.num_sites(), 2),
            ];

            let empty_circuit = ThermalCircuit::new(
                thermal_params.clone(),
                &empty_states,
                &empty_mpos,
            )
            .map_err(|e| format!("Empty circuit creation failed: {}", e))?;

            let vk = keygen_vk(&params_kzg, &empty_circuit)
                .expect("keygen_vk failed");
            let pk = keygen_pk(&params_kzg, vk.clone(), &empty_circuit)
                .expect("keygen_pk failed");

            println!(
                "[Thermal] Key generation complete in {:?} (k={})",
                start.elapsed(),
                k
            );

            Ok(Self {
                params_kzg,
                pk,
                vk,
                thermal_params,
                stats: ThermalProverStats::default(),
            })
        }

        /// Generate a proof for one timestep.
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

            let public_inputs = circuit.public_inputs();
            let num_constraints = circuit.sizing.estimate_constraints();
            let k = circuit.k().max(14);

            let mut transcript =
                Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

            create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
                &self.params_kzg,
                &self.pk,
                &[circuit.clone()],
                &[&[&public_inputs]],
                OsRng,
                &mut transcript,
            )
            .map_err(|e| format!("Proof generation failed: {:?}", e))?;

            let proof_bytes = transcript.finalize();
            let generation_time_ms = start.elapsed().as_millis() as u64;

            let proof = ThermalProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
                k,
                params: self.thermal_params.clone(),
                conservation_residual: circuit.witness.conservation.residual,
                cg_residual_norm: circuit.witness.implicit_solve.final_residual_norm,
                cg_iterations: circuit.witness.implicit_solve.num_iterations,
                input_state_hash_limbs: circuit.witness.hashes.input_state_hash_limbs,
                output_state_hash_limbs: circuit.witness.hashes.output_state_hash_limbs,
                params_hash_limbs: circuit.witness.hashes.params_hash_limbs,
            };

            self.stats.record(&proof);

            println!(
                "[Thermal] Proof generated: {} constraints, {} bytes, {:.1}ms, {} CG iters",
                num_constraints,
                proof.size(),
                generation_time_ms as f64,
                proof.cg_iterations,
            );

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &ThermalProverStats {
            &self.stats
        }

        /// Get the verifying key for deployment.
        pub fn verifying_key(&self) -> &VerifyingKey<G1Affine> {
            &self.vk
        }

        /// Get the KZG parameters for deployment.
        pub fn kzg_params(&self) -> &ParamsKZG<Bn256> {
            &self.params_kzg
        }
    }

    /// Thermal ZK Verifier using Halo2/KZG.
    pub struct ThermalVerifier {
        /// KZG parameters.
        params_kzg: ParamsKZG<Bn256>,

        /// Verifying key.
        vk: VerifyingKey<G1Affine>,
    }

    impl ThermalVerifier {
        /// Create a verifier from KZG parameters and verifying key.
        pub fn new(params_kzg: ParamsKZG<Bn256>, vk: VerifyingKey<G1Affine>) -> Self {
            Self { params_kzg, vk }
        }

        /// Create a verifier from a prover (extracts the verifying key).
        pub fn from_prover(prover: &ThermalProver) -> Self {
            Self {
                params_kzg: prover.params_kzg.clone(),
                vk: prover.vk.clone(),
            }
        }

        /// Verify a thermal proof.
        ///
        /// Reconstructs public inputs from the proof data and verifies
        /// the Halo2/KZG proof against them.
        pub fn verify(
            &self,
            proof: &ThermalProof,
        ) -> Result<ThermalVerificationResult, String> {
            let public_inputs = Self::reconstruct_public_inputs(proof);
            self.verify_with_public_inputs(proof, &public_inputs)
        }

        /// Verify with explicitly provided public inputs.
        pub fn verify_with_public_inputs(
            &self,
            proof: &ThermalProof,
            public_inputs: &[Fr],
        ) -> Result<ThermalVerificationResult, String> {
            let start = Instant::now();

            let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(
                &proof.proof_bytes[..],
            );

            let strategy = SingleStrategy::new(&self.params_kzg);

            let valid =
                verify_proof::<KZGCommitmentScheme<Bn256>, VerifierGWC<_>, _, _, _>(
                    &self.params_kzg,
                    &self.vk,
                    strategy,
                    &[&[public_inputs]],
                    &mut transcript,
                )
                .is_ok();

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

        /// Reconstruct the public inputs vector from proof data.
        fn reconstruct_public_inputs(proof: &ThermalProof) -> Vec<Fr> {
            let mut inputs = Vec::new();

            // Input state hash (4 limbs)
            for limb in &proof.input_state_hash_limbs {
                inputs.push(Fr::from(*limb));
            }
            // Output state hash (4 limbs)
            for limb in &proof.output_state_hash_limbs {
                inputs.push(Fr::from(*limb));
            }
            // Params hash (4 limbs)
            for limb in &proof.params_hash_limbs {
                inputs.push(Fr::from(*limb));
            }

            // Conservation residual (signed Q16)
            let r = proof.conservation_residual;
            if r.raw >= 0 {
                inputs.push(Fr::from(r.raw as u64));
            } else {
                inputs.push(-Fr::from((-r.raw) as u64));
            }

            // dt (signed Q16)
            if proof.params.dt.raw >= 0 {
                inputs.push(Fr::from(proof.params.dt.raw as u64));
            } else {
                inputs.push(-Fr::from((-proof.params.dt.raw) as u64));
            }

            // alpha (signed Q16)
            if proof.params.alpha.raw >= 0 {
                inputs.push(Fr::from(proof.params.alpha.raw as u64));
            } else {
                inputs.push(-Fr::from((-proof.params.alpha.raw) as u64));
            }

            // chi_max, grid_bits
            inputs.push(Fr::from(proof.params.chi_max as u64));
            inputs.push(Fr::from(proof.params.grid_bits as u64));

            inputs
        }
    }
}

#[cfg(feature = "halo2")]
pub use halo2_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Stub Prover/Verifier (without Halo2)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "halo2"))]
pub mod stub_prover {
    //! Stub thermal prover/verifier for builds without the Halo2 backend.
    use super::*;

    /// Stub thermal prover for builds without Halo2.
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
            };

            self.stats.record(&proof);

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &ThermalProverStats {
            &self.stats
        }
    }

    /// Stub thermal verifier for builds without Halo2.
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

#[cfg(not(feature = "halo2"))]
pub use stub_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

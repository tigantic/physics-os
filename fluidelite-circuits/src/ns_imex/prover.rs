//! Prover and verifier for the NS-IMEX proof circuit.
//!
//! Wraps the Halo2 proving/verification API for the NS-IMEX circuit.
//! Provides both Halo2 and stub implementations.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::config::NSIMEXParams;
use super::halo2_impl::NSIMEXCircuit;

// ═══════════════════════════════════════════════════════════════════════════
// Proof Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A ZK proof for one NS-IMEX timestep.
#[derive(Clone, Debug)]
pub struct NSIMEXProof {
    /// Raw proof bytes (Halo2/KZG serialized proof).
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

    /// Reconstruct the public inputs vector from proof data.
    #[cfg(feature = "halo2")]
    pub fn reconstruct_public_inputs(&self) -> Vec<halo2_axiom::halo2curves::bn256::Fr> {
        use halo2_axiom::halo2curves::bn256::Fr;

        let mut inputs = Vec::new();

        for limb in &self.input_state_hash_limbs {
            inputs.push(Fr::from(*limb));
        }
        for limb in &self.output_state_hash_limbs {
            inputs.push(Fr::from(*limb));
        }
        for limb in &self.params_hash_limbs {
            inputs.push(Fr::from(*limb));
        }

        // KE residual
        if self.ke_residual.raw >= 0 {
            inputs.push(Fr::from(self.ke_residual.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.ke_residual.raw) as u64));
        }

        // Enstrophy residual
        if self.enstrophy_residual.raw >= 0 {
            inputs.push(Fr::from(self.enstrophy_residual.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.enstrophy_residual.raw) as u64));
        }

        // Divergence residual
        if self.divergence_residual.raw >= 0 {
            inputs.push(Fr::from(self.divergence_residual.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.divergence_residual.raw) as u64));
        }

        // dt
        if self.params.dt.raw >= 0 {
            inputs.push(Fr::from(self.params.dt.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.params.dt.raw) as u64));
        }

        inputs
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
// Halo2 Prover/Verifier
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
/// Halo2/KZG prover and verifier for the NS-IMEX proof circuit.
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

    use super::super::config::{NSIMEXCircuitSizing, NUM_DIMENSIONS, NUM_NS_VARIABLES, PHYS_DIM};

    /// NS-IMEX ZK Prover using Halo2/KZG.
    pub struct NSIMEXProver {
        params_kzg: ParamsKZG<Bn256>,
        pk: ProvingKey<G1Affine>,
        vk: VerifyingKey<G1Affine>,
        ns_params: NSIMEXParams,
        stats: NSIMEXProverStats,
    }

    impl NSIMEXProver {
        /// Create a new prover. Performs one-time key generation.
        pub fn new(ns_params: NSIMEXParams) -> Result<Self, String> {
            println!("[NS-IMEX] Generating proving keys (one-time setup)...");
            let start = Instant::now();

            let sizing = NSIMEXCircuitSizing::from_params(&ns_params);
            let k = (sizing.k as u32).max(14);

            let params_kzg = ParamsKZG::<Bn256>::setup(k, OsRng);

            let empty_states: Vec<MPS> = (0..NUM_NS_VARIABLES)
                .map(|_| MPS::new(ns_params.num_sites(), ns_params.chi_max, PHYS_DIM))
                .collect();
            let empty_mpos: Vec<MPO> = (0..NUM_DIMENSIONS)
                .map(|_| MPO::identity(ns_params.num_sites(), PHYS_DIM))
                .collect();

            let empty_circuit =
                NSIMEXCircuit::new(ns_params.clone(), &empty_states, &empty_mpos)
                    .map_err(|e| format!("Empty circuit creation failed: {}", e))?;

            let vk = keygen_vk(&params_kzg, &empty_circuit)
                .expect("keygen_vk failed");
            let pk = keygen_pk(&params_kzg, vk.clone(), &empty_circuit)
                .expect("keygen_pk failed");

            println!(
                "[NS-IMEX] Key generation complete in {:?} (k={})",
                start.elapsed(),
                k,
            );

            Ok(Self {
                params_kzg,
                pk,
                vk,
                ns_params,
                stats: NSIMEXProverStats::default(),
            })
        }

        /// Generate a proof for one NS-IMEX timestep.
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

            let public_inputs = circuit.public_inputs();
            let num_constraints = circuit.sizing.total_constraints;
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

            let ke_residual = Q16::from_raw(
                circuit.witness.kinetic_energy_after.raw
                    - circuit.witness.kinetic_energy_before.raw,
            );
            let ens_residual = Q16::from_raw(
                circuit.witness.enstrophy_after.raw
                    - circuit.witness.enstrophy_before.raw,
            );

            let proof = NSIMEXProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
                k,
                params: self.ns_params.clone(),
                ke_residual,
                enstrophy_residual: ens_residual,
                divergence_residual: circuit.witness.divergence_residual,
                input_state_hash_limbs: circuit.witness.input_hash_limbs,
                output_state_hash_limbs: circuit.witness.output_hash_limbs,
                params_hash_limbs: circuit.witness.params_hash_limbs,
            };

            self.stats.record(&proof);

            println!(
                "[NS-IMEX] Proof generated: {} constraints, {} bytes, {:.1}ms",
                num_constraints,
                proof.size(),
                generation_time_ms as f64,
            );

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &NSIMEXProverStats {
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

    /// NS-IMEX ZK Verifier using Halo2/KZG.
    pub struct NSIMEXVerifier {
        params_kzg: ParamsKZG<Bn256>,
        vk: VerifyingKey<G1Affine>,
    }

    impl NSIMEXVerifier {
        /// Create a verifier from KZG parameters and verifying key.
        pub fn new(params_kzg: ParamsKZG<Bn256>, vk: VerifyingKey<G1Affine>) -> Self {
            Self { params_kzg, vk }
        }

        /// Create a verifier from a prover (extracts the verifying key).
        pub fn from_prover(prover: &NSIMEXProver) -> Self {
            Self {
                params_kzg: prover.params_kzg.clone(),
                vk: prover.vk.clone(),
            }
        }

        /// Verify an NS-IMEX proof.
        ///
        /// Reconstructs public inputs from the proof data and verifies
        /// the Halo2/KZG proof against them.
        pub fn verify(
            &self,
            proof: &NSIMEXProof,
        ) -> Result<NSIMEXVerificationResult, String> {
            let public_inputs = Self::reconstruct_public_inputs(proof);
            self.verify_with_public_inputs(proof, &public_inputs)
        }

        /// Verify with explicitly provided public inputs.
        pub fn verify_with_public_inputs(
            &self,
            proof: &NSIMEXProof,
            public_inputs: &[Fr],
        ) -> Result<NSIMEXVerificationResult, String> {
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

            Ok(NSIMEXVerificationResult {
                valid,
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

        /// Reconstruct the public inputs vector from proof data.
        fn reconstruct_public_inputs(proof: &NSIMEXProof) -> Vec<Fr> {
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

            // KE residual (signed Q16)
            if proof.ke_residual.raw >= 0 {
                inputs.push(Fr::from(proof.ke_residual.raw as u64));
            } else {
                inputs.push(-Fr::from((-proof.ke_residual.raw) as u64));
            }

            // Enstrophy residual (signed Q16)
            if proof.enstrophy_residual.raw >= 0 {
                inputs.push(Fr::from(proof.enstrophy_residual.raw as u64));
            } else {
                inputs.push(-Fr::from((-proof.enstrophy_residual.raw) as u64));
            }

            // Divergence residual (signed Q16)
            if proof.divergence_residual.raw >= 0 {
                inputs.push(Fr::from(proof.divergence_residual.raw as u64));
            } else {
                inputs.push(-Fr::from((-proof.divergence_residual.raw) as u64));
            }

            // dt (signed Q16)
            if proof.params.dt.raw >= 0 {
                inputs.push(Fr::from(proof.params.dt.raw as u64));
            } else {
                inputs.push(-Fr::from((-proof.params.dt.raw) as u64));
            }

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
    //! Stub NS-IMEX prover/verifier for builds without the Halo2 backend.
    use super::*;

    /// Stub NS-IMEX prover for builds without Halo2.
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
                proof_bytes: vec![0u8; 800], // Simulated KZG proof
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

    /// Stub NS-IMEX verifier for builds without Halo2.
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
        /// Create a new stub verifier.
        pub fn new() -> Self {
            Self::default()
        }

        /// Verify a proof (stub: checks proof structure, returns valid).
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

#[cfg(not(feature = "halo2"))]
pub use stub_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

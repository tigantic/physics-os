//! Prover and verifier for the Euler 3D proof circuit.
//!
//! Wraps the Halo2 proving/verification API for the Euler 3D circuit.
//! Provides both Halo2 and stub implementations.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::time::Instant;

use fluidelite_core::field::Q16;
use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;

use super::config::{Euler3DParams, NUM_CONSERVED_VARIABLES};
use super::halo2_impl::Euler3DCircuit;

// ═══════════════════════════════════════════════════════════════════════════
// Proof Data Structure
// ═══════════════════════════════════════════════════════════════════════════

/// A ZK proof for one Euler 3D timestep.
#[derive(Clone, Debug)]
pub struct Euler3DProof {
    /// Raw proof bytes (Halo2/KZG serialized proof).
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

    /// Reconstruct the public inputs vector from proof data.
    ///
    /// Used by the verifier to recover the public inputs without
    /// needing the original circuit.
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

        // Conservation residuals
        for residual in &self.conservation_residuals {
            if residual.raw >= 0 {
                inputs.push(Fr::from(residual.raw as u64));
            } else {
                inputs.push(-Fr::from((-residual.raw) as u64));
            }
        }

        // dt, chi_max, grid_bits
        if self.params.dt.raw >= 0 {
            inputs.push(Fr::from(self.params.dt.raw as u64));
        } else {
            inputs.push(-Fr::from((-self.params.dt.raw) as u64));
        }
        inputs.push(Fr::from(self.params.chi_max as u64));
        inputs.push(Fr::from(self.params.grid_bits as u64));

        inputs
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
// Halo2 Prover/Verifier
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "halo2")]
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

    /// Euler 3D ZK Prover using Halo2/KZG.
    pub struct Euler3DProver {
        /// KZG parameters.
        params_kzg: ParamsKZG<Bn256>,

        /// Proving key.
        pk: ProvingKey<G1Affine>,

        /// Verifying key.
        vk: VerifyingKey<G1Affine>,

        /// Physics parameters.
        euler_params: Euler3DParams,

        /// Accumulated statistics.
        stats: Euler3DProverStats,
    }

    impl Euler3DProver {
        /// Create a new prover. Performs one-time key generation.
        pub fn new(euler_params: Euler3DParams) -> Result<Self, String> {
            println!("[Euler3D] Generating proving keys (one-time setup)...");
            let start = Instant::now();

            let sizing = Euler3DCircuitSizing::from_params(&euler_params);
            let k = sizing.k.max(14); // Minimum k for meaningful circuits

            // Generate KZG parameters
            let params_kzg = ParamsKZG::<Bn256>::setup(k, OsRng);

            // Create empty circuit for key generation
            let empty_states: Vec<MPS> = (0..NUM_CONSERVED_VARIABLES)
                .map(|_| {
                    MPS::new(euler_params.num_sites(), euler_params.chi_max, 2)
                })
                .collect();
            let empty_mpos: Vec<MPO> = (0..3)
                .map(|_| MPO::identity(euler_params.num_sites(), 2))
                .collect();

            let empty_circuit = Euler3DCircuit::new(
                euler_params.clone(),
                &empty_states,
                &empty_mpos,
            )
            .map_err(|e| format!("Empty circuit creation failed: {}", e))?;

            let vk = keygen_vk(&params_kzg, &empty_circuit)
                .expect("keygen_vk failed");
            let pk = keygen_pk(&params_kzg, vk.clone(), &empty_circuit)
                .expect("keygen_pk failed");

            println!(
                "[Euler3D] Key generation complete in {:?} (k={})",
                start.elapsed(),
                k
            );

            Ok(Self {
                params_kzg,
                pk,
                vk,
                euler_params,
                stats: Euler3DProverStats::default(),
            })
        }

        /// Generate a proof for one timestep.
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

            let proof = Euler3DProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
                k,
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

            println!(
                "[Euler3D] Proof generated: {} constraints, {} bytes, {:.1}ms",
                num_constraints,
                proof.size(),
                generation_time_ms as f64,
            );

            Ok(proof)
        }

        /// Get accumulated statistics.
        pub fn stats(&self) -> &Euler3DProverStats {
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

    /// Euler 3D ZK Verifier using Halo2/KZG.
    pub struct Euler3DVerifier {
        /// KZG parameters.
        params_kzg: ParamsKZG<Bn256>,

        /// Verifying key.
        vk: VerifyingKey<G1Affine>,
    }

    impl Euler3DVerifier {
        /// Create a verifier from KZG parameters and verifying key.
        pub fn new(params_kzg: ParamsKZG<Bn256>, vk: VerifyingKey<G1Affine>) -> Self {
            Self { params_kzg, vk }
        }

        /// Create a verifier from a prover (extracts the verifying key).
        pub fn from_prover(prover: &Euler3DProver) -> Self {
            Self {
                params_kzg: prover.params_kzg.clone(),
                vk: prover.vk.clone(),
            }
        }

        /// Verify an Euler 3D proof.
        pub fn verify(
            &self,
            proof: &Euler3DProof,
            public_inputs: &[Fr],
        ) -> Result<Euler3DVerificationResult, String> {
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

            Ok(Euler3DVerificationResult {
                valid,
                verification_time_us,
                num_constraints: proof.num_constraints,
                conservation_residuals: proof.conservation_residuals.clone(),
                grid_bits: proof.params.grid_bits,
                chi_max: proof.params.chi_max,
            })
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
    //! Stub Euler 3D prover/verifier for builds without the Halo2 backend.
    use super::*;

    /// Stub Euler 3D prover for builds without Halo2.
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

    /// Stub Euler 3D verifier for builds without Halo2.
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

#[cfg(not(feature = "halo2"))]
pub use stub_prover::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

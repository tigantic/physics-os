//! Prover implementation for FluidElite ZK
//!
//! Handles proof generation using Halo2's proving system.

use std::time::Instant;

use crate::circuit::config::CircuitConfig;
use crate::field::Q16;
use crate::mpo::MPO;
use crate::mps::MPS;

/// A ZK proof for FluidElite inference
#[derive(Clone, Debug)]
pub struct FluidEliteProof {
    /// Raw proof bytes
    pub proof_bytes: Vec<u8>,

    /// Proof generation time in milliseconds
    pub generation_time_ms: u64,

    /// Number of constraints
    pub num_constraints: usize,
}

impl FluidEliteProof {
    /// Serialize proof to bytes for transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Proof length + data
        bytes.extend(&(self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend(&self.proof_bytes);

        // Generation time
        bytes.extend(&self.generation_time_ms.to_le_bytes());

        // Constraints
        bytes.extend(&(self.num_constraints as u64).to_le_bytes());

        bytes
    }

    /// Proof size in bytes
    pub fn size(&self) -> usize {
        self.proof_bytes.len()
    }
}

/// Statistics for prover performance
#[derive(Debug, Default, Clone)]
pub struct ProverStats {
    /// Total proofs generated
    pub total_proofs: usize,

    /// Total proving time in ms
    pub total_time_ms: u64,

    /// Average proof time in ms
    pub avg_time_ms: f64,

    /// Total proof bytes generated
    pub total_bytes: usize,

    /// Total constraints proved
    pub total_constraints: usize,
}

impl ProverStats {
    /// Update stats with a new proof
    pub fn record(&mut self, proof: &FluidEliteProof) {
        self.total_proofs += 1;
        self.total_time_ms += proof.generation_time_ms;
        self.avg_time_ms = self.total_time_ms as f64 / self.total_proofs as f64;
        self.total_bytes += proof.size();
        self.total_constraints += proof.num_constraints;
    }
}

// ============================================================================
// Halo2-based prover (requires halo2 feature)
// ============================================================================

#[cfg(feature = "halo2")]
mod halo2_prover {
    use super::*;
    use halo2_axiom::{
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{create_proof, keygen_pk, keygen_vk, ProvingKey, VerifyingKey},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverGWC,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use rand::rngs::OsRng;

    use crate::circuit::FluidEliteCircuit;

    /// Full proof with public inputs (Halo2-specific)
    #[derive(Clone, Debug)]
    pub struct Halo2Proof {
        /// Base proof
        pub inner: FluidEliteProof,
        /// Public inputs
        pub public_inputs: Vec<Fr>,
    }

    impl Halo2Proof {
        /// Returns the number of constraints in the circuit
        pub fn num_constraints(&self) -> usize {
            self.inner.num_constraints
        }

        /// Returns the serialized size of the proof in bytes
        pub fn size(&self) -> usize {
            self.inner.size()
        }
    }

    /// FluidElite ZK Prover using Halo2
    pub struct FluidEliteProver {
        /// KZG parameters
        params: ParamsKZG<Bn256>,

        /// Proving key
        pk: ProvingKey<G1Affine>,

        /// Verifying key
        vk: VerifyingKey<G1Affine>,

        /// Model weights: W_hidden
        w_hidden: MPO,

        /// Model weights: W_input
        w_input: MPO,

        /// Readout weights
        readout_weights: Vec<Q16>,

        /// Circuit configuration
        config: CircuitConfig,

        /// Accumulated stats
        stats: ProverStats,
    }

    impl FluidEliteProver {
        /// Create a new prover with given weights
        pub fn new(
            w_hidden: MPO,
            w_input: MPO,
            readout_weights: Vec<Q16>,
            config: CircuitConfig,
        ) -> Self {
            println!("Generating proving keys (one-time setup)...");
            let start = Instant::now();

            // Generate KZG parameters
            let params = ParamsKZG::<Bn256>::setup(config.k, OsRng);

            // Create empty circuit for key generation
            let empty_context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);
            let empty_circuit = FluidEliteCircuit::new(
                0,
                empty_context,
                w_hidden.clone(),
                w_input.clone(),
                readout_weights.clone(),
            );

            // Generate verifying and proving keys
            let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk failed");
            let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk failed");

            println!("Key generation complete in {:?}", start.elapsed());

            Self {
                params,
                pk,
                vk,
                w_hidden,
                w_input,
                readout_weights,
                config,
                stats: ProverStats::default(),
            }
        }

        /// Generate a proof for a single inference step
        pub fn prove(&mut self, context: &MPS, token_id: u64) -> Result<Halo2Proof, String> {
            let start = Instant::now();

            let circuit = FluidEliteCircuit::new(
                token_id,
                context.clone(),
                self.w_hidden.clone(),
                self.w_input.clone(),
                self.readout_weights.clone(),
            );

            let public_inputs = circuit.public_inputs();
            let num_constraints = self.config.estimate_constraints();

            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

            create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
                &self.params,
                &self.pk,
                &[circuit],
                &[&[&public_inputs]],
                OsRng,
                &mut transcript,
            )
            .map_err(|e| format!("Proof generation failed: {:?}", e))?;

            let proof_bytes = transcript.finalize();
            let generation_time_ms = start.elapsed().as_millis() as u64;

            let inner = FluidEliteProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
            };

            self.stats.record(&inner);

            Ok(Halo2Proof {
                inner,
                public_inputs,
            })
        }

        /// Get accumulated statistics
        pub fn stats(&self) -> &ProverStats {
            &self.stats
        }

        /// Get the circuit configuration
        pub fn config(&self) -> &CircuitConfig {
            &self.config
        }

        /// Get the verifying key for deployment
        pub fn verifying_key(&self) -> &VerifyingKey<G1Affine> {
            &self.vk
        }

        /// Get the KZG parameters (needed by the verifier)
        pub fn params(&self) -> &ParamsKZG<Bn256> {
            &self.params
        }
    }
}

#[cfg(feature = "halo2")]
pub use halo2_prover::*;

// ============================================================================
// Stub prover (development-only fallback when halo2 is not enabled)
//
// WARNING: This stub runs the real computation but returns ZERO-FILLED
// proof bytes (vec![0u8; 800]). These are not cryptographic proofs and
// will not verify. A compile_error! in lib.rs prevents this from being
// included in production/enterprise builds.
// ============================================================================

#[cfg(not(feature = "halo2"))]
mod stub_prover {
    use super::*;

    /// Simulated prover for testing without Halo2
    pub struct FluidEliteProver {
        w_hidden: MPO,
        w_input: MPO,
        #[allow(dead_code)]
        readout_weights: Vec<Q16>,
        config: CircuitConfig,
        stats: ProverStats,
    }

    impl FluidEliteProver {
        /// Create a stub prover
        pub fn new(
            w_hidden: MPO,
            w_input: MPO,
            readout_weights: Vec<Q16>,
            config: CircuitConfig,
        ) -> Self {
            Self {
                w_hidden,
                w_input,
                readout_weights,
                config,
                stats: ProverStats::default(),
            }
        }

        /// Generate a simulated proof
        pub fn prove(&mut self, context: &MPS, token_id: u64) -> Result<FluidEliteProof, String> {
            let start = Instant::now();

            // Run actual computation to verify correctness
            let _new_context = crate::ops::fluidelite_step(
                context,
                token_id as usize,
                &self.w_hidden,
                &self.w_input,
                self.config.chi_max,
            );

            let num_constraints = self.config.estimate_constraints();
            
            // Simulate proof bytes (800 bytes is typical Halo2/KZG proof size)
            let proof_bytes = vec![0u8; 800];
            let generation_time_ms = start.elapsed().as_millis() as u64;

            let proof = FluidEliteProof {
                proof_bytes,
                generation_time_ms,
                num_constraints,
            };

            self.stats.record(&proof);

            Ok(proof)
        }

        /// Get accumulated statistics
        pub fn stats(&self) -> &ProverStats {
            &self.stats
        }

        /// Get the circuit configuration
        pub fn config(&self) -> &CircuitConfig {
            &self.config
        }
    }
}

#[cfg(not(feature = "halo2"))]
pub use stub_prover::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_serialization() {
        let proof = FluidEliteProof {
            proof_bytes: vec![1, 2, 3, 4],
            generation_time_ms: 100,
            num_constraints: 131_000,
        };

        let bytes = proof.to_bytes();
        assert!(bytes.len() > 4);
    }

    #[test]
    fn test_stats() {
        let mut stats = ProverStats::default();

        let mock_proof = FluidEliteProof {
            proof_bytes: vec![0u8; 800],
            generation_time_ms: 10,
            num_constraints: 131_000,
        };

        stats.record(&mock_proof);
        stats.record(&mock_proof);

        assert_eq!(stats.total_proofs, 2);
        assert_eq!(stats.total_time_ms, 20);
        assert_eq!(stats.avg_time_ms, 10.0);
        assert_eq!(stats.total_bytes, 1600);
    }

    #[test]
    fn test_stub_prover() {
        let config = CircuitConfig::test();
        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout_weights = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];

        let mut prover = FluidEliteProver::new(w_hidden, w_input, readout_weights, config.clone());

        let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);
        let proof = prover.prove(&context, 42).expect("Proof generation failed");

        // Halo2Proof has num_constraints() method, FluidEliteProof has it as field
        #[cfg(feature = "halo2")]
        let num_constraints = proof.num_constraints();
        #[cfg(not(feature = "halo2"))]
        let num_constraints = proof.num_constraints;

        assert!(num_constraints > 0);
        println!(
            "Simulated proof: {} constraints, {} bytes",
            num_constraints,
            proof.size()
        );
    }
}

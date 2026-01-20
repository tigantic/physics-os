//! Verifier implementation for FluidElite ZK
//!
//! Handles proof verification using Halo2's verification system.
//! Can be deployed on-chain or used off-chain.

#[cfg(not(feature = "halo2"))]
use crate::prover::FluidEliteProof;

/// Verification result with additional metadata
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the proof is valid
    pub valid: bool,

    /// Token ID that was proven
    pub token_id: u64,

    /// Verification time in microseconds
    pub verification_time_us: u64,

    /// Number of constraints verified
    pub num_constraints: usize,
}

/// On-chain verifier interface
///
/// This would be compiled to a Solidity verifier for Ethereum
/// or native contract for other chains.
pub trait OnChainVerifier {
    /// Verify proof given raw bytes
    fn verify_raw(&self, proof_bytes: &[u8], public_inputs: &[u8]) -> bool;

    /// Gas cost estimate
    fn estimate_gas(&self) -> u64;
}

/// Mock on-chain verifier for testing
#[derive(Clone, Debug)]
pub struct MockOnChainVerifier {
    /// Expected gas cost per verification
    pub gas_cost: u64,
}

impl Default for MockOnChainVerifier {
    fn default() -> Self {
        Self {
            // Typical Halo2 verification on Ethereum: ~300k gas
            gas_cost: 300_000,
        }
    }
}

impl OnChainVerifier for MockOnChainVerifier {
    fn verify_raw(&self, _proof_bytes: &[u8], _public_inputs: &[u8]) -> bool {
        // In production, this would call the actual verifier contract
        true
    }

    fn estimate_gas(&self) -> u64 {
        self.gas_cost
    }
}

// ============================================================================
// Halo2-based verifier (requires halo2 feature)
// ============================================================================

#[cfg(feature = "halo2")]
mod halo2_verifier {
    use super::*;
    use halo2_axiom::{
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{verify_proof, VerifyingKey},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::VerifierGWC,
            strategy::SingleStrategy,
        },
        transcript::{Blake2bRead, Challenge255, TranscriptReadBuffer},
    };
    use std::time::Instant;

    use crate::prover::Halo2Proof;

    /// FluidElite ZK Verifier using Halo2
    pub struct FluidEliteVerifier {
        /// KZG parameters (verifier only needs the g1 points)
        params: ParamsKZG<Bn256>,

        /// Verifying key
        vk: VerifyingKey<G1Affine>,
    }

    impl FluidEliteVerifier {
        /// Create verifier from parameters and verifying key
        pub fn new(params: ParamsKZG<Bn256>, vk: VerifyingKey<G1Affine>) -> Self {
            Self { params, vk }
        }

        /// Verify a proof
        pub fn verify(&self, proof: &Halo2Proof) -> Result<VerificationResult, String> {
            let start = Instant::now();

            let mut transcript =
                Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof.inner.proof_bytes[..]);

            let strategy = SingleStrategy::new(&self.params);

            let valid = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierGWC<_>, _, _, _>(
                &self.params,
                &self.vk,
                strategy,
                &[&[&proof.public_inputs]],
                &mut transcript,
            )
            .is_ok();

            let verification_time_us = start.elapsed().as_micros() as u64;

            // Extract token ID from first public input
            let token_id = {
                let bytes = proof.public_inputs[0].to_bytes();
                u64::from_le_bytes(bytes[0..8].try_into().unwrap())
            };

            Ok(VerificationResult {
                valid,
                token_id,
                verification_time_us,
                num_constraints: proof.inner.num_constraints,
            })
        }

        /// Verify a batch of proofs
        pub fn verify_batch(&self, proofs: &[Halo2Proof]) -> Result<Vec<VerificationResult>, String> {
            proofs.iter().map(|p| self.verify(p)).collect()
        }

        /// Extract public outputs from a proof
        pub fn extract_outputs(&self, proof: &Halo2Proof) -> Vec<Fr> {
            // Skip token_id (first element), return logits
            proof.public_inputs[1..].to_vec()
        }
    }
}

#[cfg(feature = "halo2")]
pub use halo2_verifier::*;

// ============================================================================
// Stub verifier (when halo2 is not enabled)
// ============================================================================

#[cfg(not(feature = "halo2"))]
mod stub_verifier {
    use super::*;
    use std::time::Instant;

    /// Simulated verifier for testing without Halo2
    pub struct FluidEliteVerifier {
        /// Simulated verification delay in microseconds
        pub simulated_delay_us: u64,
    }

    impl Default for FluidEliteVerifier {
        fn default() -> Self {
            Self {
                // Typical verification is ~1ms
                simulated_delay_us: 1000,
            }
        }
    }

    impl FluidEliteVerifier {
        /// Create a new stub verifier
        pub fn new() -> Self {
            Self::default()
        }

        /// Verify a proof (always returns valid for stub)
        pub fn verify(&self, proof: &FluidEliteProof, token_id: u64) -> Result<VerificationResult, String> {
            let start = Instant::now();

            // Simulate verification time
            std::thread::sleep(std::time::Duration::from_micros(self.simulated_delay_us));

            let verification_time_us = start.elapsed().as_micros() as u64;

            Ok(VerificationResult {
                valid: true,
                token_id,
                verification_time_us,
                num_constraints: proof.num_constraints,
            })
        }

        /// Verify a batch of proofs
        pub fn verify_batch(&self, proofs: &[(FluidEliteProof, u64)]) -> Result<Vec<VerificationResult>, String> {
            proofs.iter().map(|(p, tid)| self.verify(p, *tid)).collect()
        }
    }
}

#[cfg(not(feature = "halo2"))]
pub use stub_verifier::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_on_chain_verifier() {
        let verifier = MockOnChainVerifier::default();

        assert!(verifier.verify_raw(&[], &[]));
        assert_eq!(verifier.estimate_gas(), 300_000);
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            valid: true,
            token_id: 42,
            verification_time_us: 1000,
            num_constraints: 131_000,
        };

        assert!(result.valid);
        assert_eq!(result.token_id, 42);
        assert_eq!(result.num_constraints, 131_000);
    }

    #[cfg(not(feature = "halo2"))]
    #[test]
    fn test_stub_verifier() {
        let verifier = FluidEliteVerifier::default();
        let proof = FluidEliteProof {
            proof_bytes: vec![0u8; 800],
            generation_time_ms: 100,
            num_constraints: 131_000,
        };

        let result = verifier.verify(&proof, 42).expect("Verification failed");
        assert!(result.valid);
        assert_eq!(result.token_id, 42);
    }
}

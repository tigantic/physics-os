//! End-to-end integration tests for the FluidElite ZK proof pipeline.
//!
//! These tests exercise the full path:
//!   MPS context → FluidEliteCircuit → FluidEliteProver::prove()
//!   → FluidEliteVerifier::verify() → VerificationResult
//!
//! They require the `halo2` feature and are compiled only when it is enabled.
//!
//! Task 0.8:  Positive E2E — correct proof verifies.
//! Task 0.9:  Soundness — tampered proof bytes are rejected.
//! Task 0.10: Wrong VK — proof generated with VK₁ is rejected by VK₂.

#[cfg(all(test, feature = "halo2"))]
mod e2e_tests {
    use halo2_axiom::{
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{create_proof, keygen_pk, keygen_vk, VerifyingKey},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverGWC,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use rand::rngs::OsRng;

    use crate::circuit::config::CircuitConfig;
    use crate::circuit::FluidEliteCircuit;
    use crate::field::Q16;
    use crate::mpo::MPO;
    use crate::mps::MPS;
    use crate::prover::{FluidEliteProof, FluidEliteProver, Halo2Proof};
    use crate::verifier::FluidEliteVerifier;

    /// Build a small test prover/verifier pair using `CircuitConfig::test()`.
    ///
    /// Returns `(prover, verifier)` ready for E2E testing.
    fn build_test_prover_verifier() -> (FluidEliteProver, FluidEliteVerifier) {
        let config = CircuitConfig::test();
        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);
        let readout_weights =
            vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];

        let prover = FluidEliteProver::new(
            w_hidden,
            w_input,
            readout_weights,
            config.clone(),
        );

        let verifier = FluidEliteVerifier::new(
            prover.params().clone(),
            prover.verifying_key().clone(),
        );

        (prover, verifier)
    }

    // ========================================================================
    // Task 0.8: E2E positive test — correct proof verifies
    // ========================================================================

    /// Generate a real Halo2 proof for a FluidElite inference step and verify it.
    ///
    /// This is the core "proof of life" test: the full pipeline must produce
    /// a proof that an independently constructed verifier accepts.
    #[test]
    fn test_e2e_prove_and_verify() {
        let (mut prover, verifier) = build_test_prover_verifier();
        let config = prover.config().clone();

        let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);
        let token_id = 42u64;

        // Generate a real Halo2 proof
        let proof = prover
            .prove(&context, token_id)
            .expect("proof generation must succeed");

        // Proof must be non-trivial (not the old stub's 800 zero bytes)
        assert!(
            proof.inner.proof_bytes.len() > 0,
            "proof bytes must not be empty"
        );
        assert!(
            proof.inner.proof_bytes.iter().any(|&b| b != 0),
            "proof bytes must not be all zeros (would indicate stub)"
        );

        // Verify the proof with an independently constructed verifier
        let result = verifier
            .verify(&proof)
            .expect("verification must not error");

        assert!(result.valid, "correct proof must verify as valid");
        assert_eq!(result.token_id, token_id);
        assert!(result.verification_time_us > 0);
        assert!(result.num_constraints > 0);

        println!(
            "E2E proof of life: {} bytes, {} constraints, verified in {}µs",
            proof.size(),
            proof.num_constraints(),
            result.verification_time_us
        );
    }

    /// Verify multiple different token IDs to ensure the pipeline is general.
    #[test]
    fn test_e2e_multiple_tokens() {
        let (mut prover, verifier) = build_test_prover_verifier();
        let config = prover.config().clone();

        for token_id in [0u64, 1, 7, 15, 255] {
            let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

            let proof = prover
                .prove(&context, token_id)
                .expect("proof generation must succeed");

            let result = verifier
                .verify(&proof)
                .expect("verification must not error");

            assert!(
                result.valid,
                "token_id={token_id}: correct proof must verify"
            );
            assert_eq!(result.token_id, token_id);
        }

        println!("E2E: 5 different tokens all proved + verified correctly");
    }

    // ========================================================================
    // Task 0.9: Soundness test — tampered proof bytes are rejected
    // ========================================================================

    /// Tamper with the proof bytes and verify that the verifier rejects.
    ///
    /// This tests the cryptographic binding: a bit-flip in the proof transcript
    /// must cause verification failure.
    #[test]
    fn test_e2e_tampered_proof_rejected() {
        let (mut prover, verifier) = build_test_prover_verifier();
        let config = prover.config().clone();

        let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

        let proof = prover
            .prove(&context, 42)
            .expect("proof generation must succeed");

        // Sanity: untampered proof verifies
        let ok = verifier
            .verify(&proof)
            .expect("verification must not error");
        assert!(ok.valid, "original proof must verify");

        // Tamper: flip bits at multiple positions in the proof bytes
        for flip_pos in [0, proof.inner.proof_bytes.len() / 2, proof.inner.proof_bytes.len() - 1]
        {
            let mut tampered_bytes = proof.inner.proof_bytes.clone();
            tampered_bytes[flip_pos] ^= 0xFF;

            let tampered_proof = Halo2Proof {
                inner: FluidEliteProof {
                    proof_bytes: tampered_bytes,
                    generation_time_ms: proof.inner.generation_time_ms,
                    num_constraints: proof.inner.num_constraints,
                },
                public_inputs: proof.public_inputs.clone(),
            };

            let result = verifier.verify(&tampered_proof);

            match result {
                Ok(vr) => {
                    assert!(
                        !vr.valid,
                        "tampered proof (flip at byte {flip_pos}) must not verify as valid"
                    );
                }
                Err(_) => {
                    // Deserialization/transcript error is also acceptable —
                    // the verifier correctly rejects corrupt data.
                }
            }
        }

        println!("E2E soundness: all tampered proofs correctly rejected");
    }

    /// Tamper with public inputs — supply wrong logits.
    ///
    /// Even with a valid proof transcript, mismatched public inputs
    /// must cause verification to fail.
    #[test]
    fn test_e2e_wrong_public_inputs_rejected() {
        let (mut prover, verifier) = build_test_prover_verifier();
        let config = prover.config().clone();

        let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);

        let proof = prover
            .prove(&context, 42)
            .expect("proof generation must succeed");

        // Tamper: change the token_id in public inputs
        let mut wrong_inputs = proof.public_inputs.clone();
        wrong_inputs[0] = Fr::from(9999u64);

        let wrong_proof = Halo2Proof {
            inner: proof.inner.clone(),
            public_inputs: wrong_inputs,
        };

        let result = verifier.verify(&wrong_proof);
        match result {
            Ok(vr) => {
                assert!(
                    !vr.valid,
                    "proof with wrong public inputs must not verify"
                );
            }
            Err(_) => {
                // Also acceptable — transcript mismatch
            }
        }

        // Tamper: change a logit value
        if proof.public_inputs.len() > 1 {
            let mut wrong_logits = proof.public_inputs.clone();
            wrong_logits[1] = Fr::from(123456789u64);

            let wrong_proof2 = Halo2Proof {
                inner: proof.inner.clone(),
                public_inputs: wrong_logits,
            };

            let result2 = verifier.verify(&wrong_proof2);
            match result2 {
                Ok(vr) => {
                    assert!(
                        !vr.valid,
                        "proof with tampered logit must not verify"
                    );
                }
                Err(_) => {}
            }
        }

        println!("E2E soundness: wrong public inputs correctly rejected");
    }

    // ========================================================================
    // Task 0.10: Wrong VK test — proof verified with different VK is rejected
    // ========================================================================

    /// Generate a proof with one set of keys and attempt to verify with a
    /// different verifying key. Must be rejected.
    ///
    /// This confirms that the VK is cryptographically bound to the circuit
    /// and that cross-circuit proof reuse is impossible.
    #[test]
    fn test_e2e_wrong_vk_rejected() {
        let config = CircuitConfig::test();
        let w_hidden = MPO::identity(config.num_sites, config.phys_dim);
        let w_input = MPO::identity(config.num_sites, config.phys_dim);

        // Build prover 1 with readout weights A
        let readout_a = vec![Q16::from_f64(0.1); config.chi_max * config.vocab_size];
        let mut prover_a = FluidEliteProver::new(
            w_hidden.clone(),
            w_input.clone(),
            readout_a,
            config.clone(),
        );

        // Build prover 2 with readout weights B (different weights → different VK)
        let readout_b = vec![Q16::from_f64(0.9); config.chi_max * config.vocab_size];
        let prover_b = FluidEliteProver::new(
            w_hidden,
            w_input,
            readout_b,
            config.clone(),
        );

        // Verifier uses VK from prover B
        let verifier_b = FluidEliteVerifier::new(
            prover_b.params().clone(),
            prover_b.verifying_key().clone(),
        );

        // Generate proof with prover A
        let context = MPS::new(config.num_sites, config.chi_max, config.phys_dim);
        let proof_a = prover_a
            .prove(&context, 42)
            .expect("proof_a generation must succeed");

        // Verify with VK_B — must fail
        let result = verifier_b.verify(&proof_a);
        match result {
            Ok(vr) => {
                assert!(
                    !vr.valid,
                    "proof from VK_A must not verify with VK_B"
                );
            }
            Err(_) => {
                // Transcript/structure mismatch error is also acceptable
            }
        }

        // Sanity: proof_a verifies with its own VK
        let verifier_a = FluidEliteVerifier::new(
            prover_a.params().clone(),
            prover_a.verifying_key().clone(),
        );
        let result_a = verifier_a
            .verify(&proof_a)
            .expect("self-verification must not error");
        assert!(
            result_a.valid,
            "proof_a must verify with its own VK"
        );

        println!("E2E wrong VK: cross-VK verification correctly rejected");
    }

    /// A second wrong-VK test: same weights but different k.
    ///
    /// Even with identical circuit logic, different KZG parameters
    /// must prevent cross-verification.
    #[test]
    fn test_e2e_wrong_params_k_rejected() {
        // Build circuit + key gen at k=10
        let config_10 = CircuitConfig::test(); // k=10

        let w_hidden = MPO::identity(config_10.num_sites, config_10.phys_dim);
        let w_input = MPO::identity(config_10.num_sites, config_10.phys_dim);
        let readout = vec![Q16::from_f64(0.1); config_10.chi_max * config_10.vocab_size];

        let context = MPS::new(config_10.num_sites, config_10.chi_max, config_10.phys_dim);

        // Generate proof at k=10 through the raw API so we can also build k=11
        let params_10 = ParamsKZG::<Bn256>::setup(config_10.k, OsRng);
        let circuit = FluidEliteCircuit::new(
            42,
            context.clone(),
            w_hidden.clone(),
            w_input.clone(),
            readout.clone(),
        );
        let vk_10 = keygen_vk(&params_10, &circuit).expect("keygen_vk k=10");
        let pk_10 = keygen_pk(&params_10, vk_10.clone(), &circuit).expect("keygen_pk k=10");

        let public_inputs = circuit.public_inputs();
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
            &params_10,
            &pk_10,
            &[circuit.clone()],
            &[&[&public_inputs]],
            OsRng,
            &mut transcript,
        )
        .expect("proof gen k=10");
        let proof_bytes = transcript.finalize();

        let halo2_proof = Halo2Proof {
            inner: FluidEliteProof {
                proof_bytes,
                generation_time_ms: 0,
                num_constraints: config_10.estimate_constraints(),
            },
            public_inputs: public_inputs.clone(),
        };

        // Build a verifier at k=11 (different params)
        let params_11 = ParamsKZG::<Bn256>::setup(11, OsRng);
        let circuit_11 = FluidEliteCircuit::new(
            42,
            context,
            w_hidden,
            w_input,
            readout,
        );
        let vk_11 = keygen_vk(&params_11, &circuit_11).expect("keygen_vk k=11");
        let verifier_11 = FluidEliteVerifier::new(params_11, vk_11);

        // Verify proof from k=10 with k=11 verifier — must fail
        let result = verifier_11.verify(&halo2_proof);
        match result {
            Ok(vr) => {
                assert!(
                    !vr.valid,
                    "proof from k=10 must not verify with k=11 params"
                );
            }
            Err(_) => {
                // Structural mismatch error also acceptable
            }
        }

        // Sanity: verify with correct k=10 verifier
        let verifier_10 = FluidEliteVerifier::new(params_10, vk_10);
        let result_10 = verifier_10
            .verify(&halo2_proof)
            .expect("self-verification must not error");
        assert!(result_10.valid, "proof must verify with its own params");

        println!("E2E wrong params: cross-k verification correctly rejected");
    }
}

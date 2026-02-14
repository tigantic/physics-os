//! Halo2 Prover for Hybrid Lookup Circuit
//!
//! This is the real ZK prover that generates cryptographic proofs.

#[cfg(feature = "halo2")]
use halo2_axiom::{
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{create_proof, keygen_pk, keygen_vk, ProvingKey, VerifyingKey},
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::ProverGWC,
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};

#[cfg(feature = "halo2")]
use rand::rngs::OsRng;

#[cfg(feature = "halo2")]
use std::time::Instant;

#[cfg(feature = "halo2")]
use crate::circuit::HybridLookupCircuit;
#[cfg(feature = "halo2")]
use crate::circuit::FallbackCircuit;
use crate::hybrid::HybridWeights;
#[cfg(feature = "halo2")]
use crate::hybrid_prover::FeatureExtractor;

/// Cryptographic proof from Halo2
#[cfg(feature = "halo2")]
#[derive(Clone, Debug)]
pub struct Halo2HybridProof {
    /// Raw proof bytes
    pub proof_bytes: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<Fr>,
    /// Generation time in ms
    pub generation_time_ms: u64,
    /// Number of constraints
    pub num_constraints: usize,
    /// Whether lookup was used
    pub lookup_hit: bool,
}

#[cfg(feature = "halo2")]
impl Halo2HybridProof {
    /// Proof size in bytes
    pub fn size(&self) -> usize {
        self.proof_bytes.len()
    }
}

/// Halo2 Prover for Hybrid Model
#[cfg(feature = "halo2")]
pub struct Halo2HybridProver {
    /// KZG parameters for lookup circuit
    params_lookup: ParamsKZG<Bn256>,
    /// Proving key for lookup circuit
    pk_lookup: ProvingKey<G1Affine>,
    /// Verifying key for lookup circuit
    vk_lookup: VerifyingKey<G1Affine>,
    /// KZG parameters for fallback circuit
    params_fallback: ParamsKZG<Bn256>,
    /// Proving key for fallback circuit
    pk_fallback: ProvingKey<G1Affine>,
    /// Verifying key for fallback circuit
    vk_fallback: VerifyingKey<G1Affine>,
    /// Model weights
    weights: HybridWeights,
    /// Feature extractor for fallback path
    feature_extractor: FeatureExtractor,
    /// Lookup table in circuit format: (hash_lo, hash_hi, prediction)
    circuit_table: Vec<(u64, u64, u8)>,
    /// Statistics
    stats: Halo2HybridProverStats,
}

#[cfg(feature = "halo2")]
#[derive(Debug, Default, Clone)]
pub struct Halo2HybridProverStats {
    pub total_proofs: usize,
    pub lookup_proofs: usize,
    pub fallback_proofs: usize,
    pub total_time_ms: u64,
}

#[cfg(feature = "halo2")]
impl Halo2HybridProver {
    /// Create a new prover with the given weights
    /// 
    /// This performs trusted setup (one-time, expensive operation) for both
    /// the lookup circuit and the fallback matmul circuit.
    pub fn new(weights: HybridWeights) -> Self {
        println!("Initializing Halo2 prover...");
        let start = Instant::now();
        
        // Convert lookup table to circuit format
        let circuit_table: Vec<(u64, u64, u8)> = weights
            .lookup_table
            .iter()
            .map(|(&hash, &pred)| (hash, 0u64, pred))
            .collect();
        
        println!("  Lookup table: {} entries", circuit_table.len());
        
        // ── Lookup circuit setup ───────────────────────────────────────────
        let k_lookup = (circuit_table.len() as f64).log2().ceil() as u32 + 2;
        let k_lookup = k_lookup.max(10);
        println!("  Lookup circuit k: {} (2^{} = {} rows)", k_lookup, k_lookup, 1 << k_lookup);
        
        println!("  Generating lookup KZG parameters...");
        let params_lookup = ParamsKZG::<Bn256>::setup(k_lookup, OsRng);
        
        let empty_lookup = HybridLookupCircuit {
            context: vec![0u8; 12],
            hash_lo: 0,
            hash_hi: 0,
            prediction: 0,
            table: circuit_table.clone(),
        };
        
        println!("  Generating lookup proving/verifying keys...");
        let vk_lookup = keygen_vk(&params_lookup, &empty_lookup)
            .expect("keygen_vk (lookup) failed");
        let pk_lookup = keygen_pk(&params_lookup, vk_lookup.clone(), &empty_lookup)
            .expect("keygen_pk (lookup) failed");
        
        // ── Fallback circuit setup ─────────────────────────────────────────
        let empty_fallback = FallbackCircuit {
            feature_indices: vec![],
            feature_values: vec![],
            u_r: weights.u_r.clone(),
            s_r: weights.s_r.clone(),
            vt_r: weights.vt_r.clone(),
            logits: vec![crate::field::Q16::ZERO; weights.config.vocab_size],
            feature_dim: weights.config.feature_dim,
            rank: weights.config.rank,
            vocab: weights.config.vocab_size,
        };
        let k_fallback = empty_fallback.min_k();
        println!("  Fallback circuit k: {} (2^{} = {} rows)", k_fallback, k_fallback, 1 << k_fallback);
        
        println!("  Generating fallback KZG parameters...");
        let params_fallback = ParamsKZG::<Bn256>::setup(k_fallback, OsRng);
        
        println!("  Generating fallback proving/verifying keys...");
        let vk_fallback = keygen_vk(&params_fallback, &empty_fallback)
            .expect("keygen_vk (fallback) failed");
        let pk_fallback = keygen_pk(&params_fallback, vk_fallback.clone(), &empty_fallback)
            .expect("keygen_pk (fallback) failed");
        
        let feature_extractor = FeatureExtractor::new(weights.config.clone());
        println!("  Setup complete in {:?}", start.elapsed());
        
        Self {
            params_lookup,
            pk_lookup,
            vk_lookup,
            params_fallback,
            pk_fallback,
            vk_fallback,
            weights,
            feature_extractor,
            circuit_table,
            stats: Halo2HybridProverStats::default(),
        }
    }
    
    /// Generate a proof for the given context
    pub fn prove(&mut self, context: &[u8]) -> Result<Halo2HybridProof, String> {
        let start = Instant::now();
        
        // Check if context is in lookup table
        let hash = crate::hybrid::HybridWeights::hash_context(context);
        
        if let Some(&prediction) = self.weights.lookup_table.get(&hash) {
            // ── Lookup path ────────────────────────────────────────────────
            let circuit = HybridLookupCircuit::new(
                context.to_vec(),
                prediction,
                self.circuit_table.clone(),
            );
            
            let public_inputs = circuit.public_inputs();
            
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
            
            create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
                &self.params_lookup,
                &self.pk_lookup,
                &[circuit],
                &[&[&public_inputs]],
                OsRng,
                &mut transcript,
            )
            .map_err(|e| format!("Lookup proof generation failed: {:?}", e))?;
            
            let proof_bytes = transcript.finalize();
            let generation_time_ms = start.elapsed().as_millis() as u64;
            
            self.stats.total_proofs += 1;
            self.stats.lookup_proofs += 1;
            self.stats.total_time_ms += generation_time_ms;
            
            Ok(Halo2HybridProof {
                proof_bytes,
                public_inputs,
                generation_time_ms,
                num_constraints: 80,
                lookup_hit: true,
            })
        } else {
            // ── Fallback path: sparse features → matmul → logits ───────────
            let features = self.feature_extractor.extract(context);

            // Collect non-zero features
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for (i, &val) in features.iter().enumerate() {
                if val != crate::field::Q16::ZERO {
                    indices.push(i);
                    values.push(val);
                }
            }

            // Compute logits on CPU
            let logits = self.weights.matmul(&features);

            let circuit = FallbackCircuit {
                feature_indices: indices,
                feature_values: values,
                u_r: self.weights.u_r.clone(),
                s_r: self.weights.s_r.clone(),
                vt_r: self.weights.vt_r.clone(),
                logits,
                feature_dim: self.weights.config.feature_dim,
                rank: self.weights.config.rank,
                vocab: self.weights.config.vocab_size,
            };

            let public_inputs = circuit.public_inputs();
            let num_constraints = circuit.estimate_constraints();

            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

            create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
                &self.params_fallback,
                &self.pk_fallback,
                &[circuit],
                &[&[&public_inputs]],
                OsRng,
                &mut transcript,
            )
            .map_err(|e| format!("Fallback proof generation failed: {:?}", e))?;

            let proof_bytes = transcript.finalize();
            let generation_time_ms = start.elapsed().as_millis() as u64;

            self.stats.total_proofs += 1;
            self.stats.fallback_proofs += 1;
            self.stats.total_time_ms += generation_time_ms;

            Ok(Halo2HybridProof {
                proof_bytes,
                public_inputs,
                generation_time_ms,
                num_constraints,
                lookup_hit: false,
            })
        }
    }
    
    /// Get the verifying key for lookup circuit deployment
    pub fn verifying_key_lookup(&self) -> &VerifyingKey<G1Affine> {
        &self.vk_lookup
    }

    /// Get the verifying key for fallback circuit deployment
    pub fn verifying_key_fallback(&self) -> &VerifyingKey<G1Affine> {
        &self.vk_fallback
    }
    
    /// Get the KZG parameters for lookup verification
    pub fn params_lookup(&self) -> &ParamsKZG<Bn256> {
        &self.params_lookup
    }

    /// Get the KZG parameters for fallback verification
    pub fn params_fallback(&self) -> &ParamsKZG<Bn256> {
        &self.params_fallback
    }
    
    /// Get statistics
    pub fn stats(&self) -> &Halo2HybridProverStats {
        &self.stats
    }
}

// ============================================================================
// Verifier
// ============================================================================

#[cfg(feature = "halo2")]
use halo2_axiom::{
    plonk::verify_proof,
    poly::kzg::{
        multiopen::VerifierGWC,
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, TranscriptReadBuffer},
};

/// Verify a Halo2 hybrid proof
///
/// The caller must provide the correct params and vk for the proof type:
/// - Lookup proofs: use `params_lookup()` and `verifying_key_lookup()`
/// - Fallback proofs: use `params_fallback()` and `verifying_key_fallback()`
#[cfg(feature = "halo2")]
pub fn verify_hybrid_proof(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    proof: &Halo2HybridProof,
) -> Result<bool, String> {
    let strategy = SingleStrategy::new(params);
    
    let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof.proof_bytes[..]);
    
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierGWC<_>, _, _, _>(
        params,
        vk,
        strategy,
        &[&[&proof.public_inputs]],
        &mut transcript,
    )
    .map(|_| true)
    .map_err(|e| format!("Verification failed: {:?}", e))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_stats() {
        #[cfg(feature = "halo2")]
        {
            use super::Halo2HybridProverStats;
            let stats = Halo2HybridProverStats::default();
            assert_eq!(stats.total_proofs, 0);
        }
    }
}

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
use crate::hybrid::{HybridConfig, HybridWeights};

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
    /// KZG parameters
    params: ParamsKZG<Bn256>,
    /// Proving key for lookup circuit
    pk_lookup: ProvingKey<G1Affine>,
    /// Verifying key for lookup circuit
    vk_lookup: VerifyingKey<G1Affine>,
    /// Model weights
    weights: HybridWeights,
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
    /// This performs trusted setup (one-time, expensive operation).
    pub fn new(weights: HybridWeights) -> Self {
        println!("Initializing Halo2 prover...");
        let start = Instant::now();
        
        // Convert lookup table to circuit format
        // Split 128-bit hash into two 64-bit parts
        let circuit_table: Vec<(u64, u64, u8)> = weights
            .lookup_table
            .iter()
            .map(|(&hash, &pred)| {
                // Our hash is 64-bit, so hash_hi = 0
                (hash, 0u64, pred)
            })
            .collect();
        
        println!("  Lookup table: {} entries", circuit_table.len());
        
        // Determine k (circuit size) based on table size
        // 2^k must be >= table_size
        let k = (circuit_table.len() as f64).log2().ceil() as u32 + 2;
        let k = k.max(10); // Minimum k=10
        println!("  Circuit k: {} (2^{} = {} rows)", k, k, 1 << k);
        
        // Generate KZG parameters
        println!("  Generating KZG parameters...");
        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        
        // Create empty circuit for key generation
        let empty_circuit = HybridLookupCircuit {
            context: vec![0u8; 12],
            hash_lo: 0,
            hash_hi: 0,
            prediction: 0,
            table: circuit_table.clone(),
        };
        
        // Generate keys
        println!("  Generating proving/verifying keys...");
        let vk_lookup = keygen_vk(&params, &empty_circuit)
            .expect("keygen_vk failed");
        let pk_lookup = keygen_pk(&params, vk_lookup.clone(), &empty_circuit)
            .expect("keygen_pk failed");
        
        println!("  Setup complete in {:?}", start.elapsed());
        
        Self {
            params,
            pk_lookup,
            vk_lookup,
            weights,
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
            // Lookup path - generate proof
            let circuit = HybridLookupCircuit::new(
                context.to_vec(),
                prediction,
                self.circuit_table.clone(),
            );
            
            let public_inputs = circuit.public_inputs();
            
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
            
            create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, _, _>(
                &self.params,
                &self.pk_lookup,
                &[circuit],
                &[&[&public_inputs]],
                OsRng,
                &mut transcript,
            )
            .map_err(|e| format!("Proof generation failed: {:?}", e))?;
            
            let proof_bytes = transcript.finalize();
            let generation_time_ms = start.elapsed().as_millis() as u64;
            
            self.stats.total_proofs += 1;
            self.stats.lookup_proofs += 1;
            self.stats.total_time_ms += generation_time_ms;
            
            Ok(Halo2HybridProof {
                proof_bytes,
                public_inputs,
                generation_time_ms,
                num_constraints: 80, // Lookup is ~80 constraints
                lookup_hit: true,
            })
        } else {
            // Fallback path - would use FallbackCircuit
            // For now, return a simulated proof
            self.stats.total_proofs += 1;
            self.stats.fallback_proofs += 1;
            
            Err("Fallback circuit not yet implemented - context not in lookup table".to_string())
        }
    }
    
    /// Get the verifying key for deployment
    pub fn verifying_key(&self) -> &VerifyingKey<G1Affine> {
        &self.vk_lookup
    }
    
    /// Get the KZG parameters for verification
    pub fn params(&self) -> &ParamsKZG<Bn256> {
        &self.params
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

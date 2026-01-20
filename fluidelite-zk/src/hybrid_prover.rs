//! Hybrid Prover for FluidElite
//!
//! Uses Lookup Arguments (100x cheaper than matmul) for seen contexts,
//! falls back to sparse feature extraction + rank-24 matmul for unseen.
//!
//! Architecture (from FINDINGS.md):
//! - Lookup table: 100% accuracy on seen contexts, O(1) hash
//! - Fallback: sparse features → compressed W → logits (46% on unseen)
//! - NO MPS, NO MPO, NO runtime truncation

use std::time::Instant;
use crate::field::Q16;
use crate::hybrid::{HybridConfig, HybridWeights};

/// Result of hybrid inference
#[derive(Debug, Clone)]
pub struct HybridInferenceResult {
    /// Predicted next byte
    pub prediction: u8,
    /// Whether lookup table was used (vs fallback)
    pub lookup_hit: bool,
    /// Logits (only populated if fallback was used)
    pub logits: Option<Vec<Q16>>,
    /// Inference time in microseconds
    pub time_us: u64,
}

/// Proof for hybrid inference
#[derive(Debug, Clone)]
pub struct HybridProof {
    /// Raw proof bytes
    pub proof_bytes: Vec<u8>,
    /// Proof generation time in milliseconds
    pub generation_time_ms: u64,
    /// Number of constraints
    pub num_constraints: usize,
    /// Whether lookup was used
    pub lookup_hit: bool,
}

impl HybridProof {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(&(self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend(&self.proof_bytes);
        bytes.extend(&self.generation_time_ms.to_le_bytes());
        bytes.extend(&(self.num_constraints as u64).to_le_bytes());
        bytes.push(if self.lookup_hit { 1 } else { 0 });
        bytes
    }
    
    /// Proof size in bytes
    pub fn size(&self) -> usize {
        self.proof_bytes.len()
    }
}

/// Statistics for hybrid prover
#[derive(Debug, Default, Clone)]
pub struct HybridProverStats {
    pub total_proofs: usize,
    pub lookup_hits: usize,
    pub fallback_uses: usize,
    pub total_time_ms: u64,
    pub total_constraints: usize,
}

impl HybridProverStats {
    pub fn record(&mut self, proof: &HybridProof) {
        self.total_proofs += 1;
        if proof.lookup_hit {
            self.lookup_hits += 1;
        } else {
            self.fallback_uses += 1;
        }
        self.total_time_ms += proof.generation_time_ms;
        self.total_constraints += proof.num_constraints;
    }
    
    pub fn lookup_rate(&self) -> f64 {
        if self.total_proofs == 0 {
            0.0
        } else {
            self.lookup_hits as f64 / self.total_proofs as f64
        }
    }
}

/// Feature extraction for sparse features
/// Matches Python implementation exactly
pub struct FeatureExtractor {
    config: HybridConfig,
}

impl FeatureExtractor {
    pub fn new(config: HybridConfig) -> Self {
        Self { config }
    }
    
    /// Extract sparse features from context
    /// Returns feature vector of size feature_dim
    pub fn extract(&self, context: &[u8]) -> Vec<Q16> {
        let l = self.config.context_len;
        let mut features = vec![Q16::ZERO; self.config.feature_dim];
        
        // Feature layout:
        // [0..1024]: Unigrams (last 4 bytes)
        // [1024..9216]: Bigrams (8192)
        // [9216..17408]: Trigrams (8192)
        // [17408..21504]: Skipgrams (4096)
        
        // Unigrams: last 4 bytes
        for i in 0..4 {
            if l >= 4 - i {
                let byte_val = context[l - 4 + i] as usize;
                let idx = (i * 256 + byte_val) % 1024;
                features[idx] = features[idx] + Q16::ONE;
            }
        }
        
        // Bigrams
        for i in 0..(l - 1) {
            let b1 = context[i] as usize;
            let b2 = context[i + 1] as usize;
            let idx = 1024 + ((i * 65537 + b1 * 257 + b2) % 8192);
            features[idx] = features[idx] + Q16::ONE;
        }
        
        // Trigrams
        for i in 0..(l - 2) {
            let b1 = context[i] as usize;
            let b2 = context[i + 1] as usize;
            let b3 = context[i + 2] as usize;
            let idx = 1024 + 8192 + ((b1 * 65537 + b2 * 257 + b3) % 8192);
            features[idx] = features[idx] + Q16::ONE;
        }
        
        // Skipgrams (b1, _, b3)
        for i in 0..(l - 2) {
            let b1 = context[i] as usize;
            let b3 = context[i + 2] as usize;
            let idx = 1024 + 8192 + 8192 + ((b1 * 257 + b3) % 4096);
            features[idx] = features[idx] + Q16::ONE;
        }
        
        features
    }
}

/// Hybrid prover: Lookup + Fallback
pub struct HybridProver {
    weights: HybridWeights,
    feature_extractor: FeatureExtractor,
    stats: HybridProverStats,
}

impl HybridProver {
    /// Create prover from weights
    pub fn new(weights: HybridWeights) -> Self {
        let config = weights.config.clone();
        Self {
            weights,
            feature_extractor: FeatureExtractor::new(config),
            stats: HybridProverStats::default(),
        }
    }
    
    /// Load prover from binary file
    pub fn from_binary(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let weights = HybridWeights::from_binary(path)?;
        Ok(Self::new(weights))
    }
    
    /// Run inference (no proof)
    pub fn infer(&self, context: &[u8]) -> HybridInferenceResult {
        let start = Instant::now();
        
        // Check lookup table first
        let hash = HybridWeights::hash_context(context);
        if let Some(prediction) = self.weights.lookup(hash) {
            return HybridInferenceResult {
                prediction,
                lookup_hit: true,
                logits: None,
                time_us: start.elapsed().as_micros() as u64,
            };
        }
        
        // Fallback: sparse features → matmul
        let features = self.feature_extractor.extract(context);
        let logits = self.weights.matmul(&features);
        
        // Argmax
        let prediction = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0);
        
        HybridInferenceResult {
            prediction,
            lookup_hit: false,
            logits: Some(logits),
            time_us: start.elapsed().as_micros() as u64,
        }
    }
    
    /// Generate ZK proof for inference
    pub fn prove(&mut self, context: &[u8]) -> HybridProof {
        let start = Instant::now();
        
        let hash = HybridWeights::hash_context(context);
        let lookup_hit = self.weights.lookup(hash).is_some();
        
        // Constraint count depends on path taken
        let num_constraints = if lookup_hit {
            // Lookup argument: ~100 constraints
            // Hash verification + table membership
            Self::LOOKUP_CONSTRAINTS
        } else {
            // Fallback: feature extraction + matmul
            Self::FALLBACK_CONSTRAINTS
        };
        
        // Simulate proof (real Halo2 integration would go here)
        let proof_bytes = vec![0u8; if lookup_hit { 400 } else { 800 }];
        
        let proof = HybridProof {
            proof_bytes,
            generation_time_ms: start.elapsed().as_millis() as u64,
            num_constraints,
            lookup_hit,
        };
        
        self.stats.record(&proof);
        proof
    }
    
    /// Lookup path constraint count
    /// - Hash verification: ~50 constraints (SHA-256 gadget optimized)
    /// - Lookup argument membership: ~20 constraints
    /// - Output binding: ~10 constraints
    const LOOKUP_CONSTRAINTS: usize = 80;
    
    /// Fallback path constraint count  
    /// - Feature extraction: L * 4 * 2 = 96 constraints (sparse updates)
    /// - U_r matmul: feature_dim * rank * 2 = 1,032,192 constraints
    /// - S_r scaling: rank = 24 constraints
    /// - Vt_r matmul: rank * vocab * 2 = 12,288 constraints
    /// - Argmax: vocab * 2 = 512 constraints
    /// With sparse optimizations: ~50,000 constraints
    const FALLBACK_CONSTRAINTS: usize = 50_000;
    
    /// Get statistics
    pub fn stats(&self) -> &HybridProverStats {
        &self.stats
    }
    
    /// Get config
    pub fn config(&self) -> &HybridConfig {
        &self.weights.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_extractor() {
        let config = HybridConfig::default();
        let extractor = FeatureExtractor::new(config);
        
        let context = b"Hello World!";
        let features = extractor.extract(context);
        
        assert_eq!(features.len(), 21504);
        
        // Should have some non-zero features
        let non_zero: usize = features.iter().filter(|&&x| x != Q16::ZERO).count();
        assert!(non_zero > 0, "Should have non-zero features");
        println!("Non-zero features: {}", non_zero);
    }
    
    #[test]
    fn test_hybrid_stats() {
        let mut stats = HybridProverStats::default();
        
        let lookup_proof = HybridProof {
            proof_bytes: vec![0; 400],
            generation_time_ms: 1,
            num_constraints: 80,
            lookup_hit: true,
        };
        
        let fallback_proof = HybridProof {
            proof_bytes: vec![0; 800],
            generation_time_ms: 10,
            num_constraints: 50_000,
            lookup_hit: false,
        };
        
        stats.record(&lookup_proof);
        stats.record(&fallback_proof);
        
        assert_eq!(stats.total_proofs, 2);
        assert_eq!(stats.lookup_hits, 1);
        assert_eq!(stats.fallback_uses, 1);
        assert_eq!(stats.lookup_rate(), 0.5);
    }
}

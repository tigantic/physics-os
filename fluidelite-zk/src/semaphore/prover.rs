//! Zero-Expansion Semaphore Prover
//!
//! Generates Semaphore proofs using Zero-Expansion for tree depths up to 50.

use crate::qtt_native_msm::{QttTrain, FlattenedQtt, BatchedQttBases, qtt_batched_commit};
use crate::qtt_rmt::RmtChallengeGenerator;
use crate::gpu::GpuAccelerator;
use super::pqc::PqcHybridCommitment;

use icicle_bn254::curve::ScalarField;
use std::time::Instant;

/// Semaphore proof using Zero-Expansion
#[derive(Clone, Debug)]
pub struct ZeroExpansionSemaphoreProof {
    /// QTT commitment to Merkle path
    pub qtt_commitment: Vec<u8>,
    /// RMT Fiat-Shamir challenges
    pub challenges: Vec<u64>,
    /// Structure proof (Halo2)
    pub structure_proof: Vec<u8>,
    /// PQC binding (for quantum migration)
    pub pqc_binding: Option<[u8; 32]>,
    /// Tree depth proven
    pub tree_depth: u8,
    /// Proof generation time
    pub generation_time_ms: u64,
}

/// Public inputs for Semaphore verification
#[derive(Clone, Debug)]
pub struct SemaphorePublicInputs {
    /// Merkle root of identity tree
    pub merkle_root: [u8; 32],
    /// Nullifier hash (prevents double-signaling)
    pub nullifier_hash: [u8; 32],
    /// Hash of the signal being signed
    pub signal_hash: [u8; 32],
    /// External nullifier (scope/context)
    pub external_nullifier: [u8; 32],
    /// Tree depth (visible in verification!)
    pub tree_depth: u8,
}

/// Zero-Expansion Semaphore Prover
///
/// Supports tree depths from 16 to 50 (2^50 = 1 quadrillion members).
pub struct ZeroExpansionSemaphoreProver {
    /// Tree depth
    tree_depth: u8,
    /// QTT rank for compression
    qtt_rank: usize,
    /// Precomputed bases (resident in VRAM)
    bases: Option<BatchedQttBases>,
    /// RMT challenge generator
    rmt: RmtChallengeGenerator,
    /// Statistics
    pub stats: SemaphoreProverStats,
}

/// Prover statistics
#[derive(Clone, Debug, Default)]
pub struct SemaphoreProverStats {
    pub total_proofs: usize,
    pub total_time_ms: u64,
    pub avg_proof_ms: f64,
    pub compression_ratio: f64,
    pub vram_mb: f64,
}

impl ZeroExpansionSemaphoreProver {
    /// Create new prover for given tree depth
    ///
    /// # Arguments
    /// * `tree_depth` - Merkle tree depth (16-50)
    /// * `qtt_rank` - QTT rank for compression (default 16)
    ///
    /// # Panics
    /// Panics if tree_depth is not in 16..=50
    pub fn new(tree_depth: u8, qtt_rank: usize) -> Result<Self, String> {
        if tree_depth < 16 || tree_depth > 50 {
            return Err(format!("Tree depth must be 16-50, got {}", tree_depth));
        }
        
        println!("╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║         ZERO-EXPANSION SEMAPHORE PROVER v3.0 (PQC HYBRID)                   ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════╣");
        println!("║  Tree Depth: {} (2^{} = {} members)", tree_depth, tree_depth, 
            if tree_depth <= 63 { 1u64 << tree_depth } else { u64::MAX });
        println!("║  QTT Rank: {}", qtt_rank);
        println!("╚══════════════════════════════════════════════════════════════════════════════╝");
        
        Ok(Self {
            tree_depth,
            qtt_rank,
            bases: None,
            rmt: RmtChallengeGenerator::new(),
            stats: SemaphoreProverStats::default(),
        })
    }
    
    /// Initialize GPU and preload bases to VRAM
    pub fn setup_gpu(&mut self) -> Result<(), String> {
        println!("\n🔧 Initializing GPU...");
        let _gpu = GpuAccelerator::new().map_err(|e| format!("GPU init failed: {}", e))?;
        println!("✓ GPU initialized");
        
        // Create template QTT for Merkle path
        // Merkle path has `tree_depth` siblings, each 32 bytes
        println!("🔧 Generating QTT commitment bases for depth {}...", self.tree_depth);
        let start = Instant::now();
        
        let template = QttTrain::new(self.tree_depth as usize, 2, self.qtt_rank);
        let bases = BatchedQttBases::generate(&template, 10)
            .map_err(|e| format!("Bases generation failed: {}", e))?;
        
        let vram_mb = bases.vram_bytes() as f64 / (1024.0 * 1024.0);
        let compression = template.compression_ratio();
        
        println!("✓ Bases loaded to VRAM in {:.2}s", start.elapsed().as_secs_f64());
        println!("  VRAM usage: {:.2} MB", vram_mb);
        println!("  Compression ratio: {:.0}x", compression);
        
        self.bases = Some(bases);
        self.stats.vram_mb = vram_mb;
        self.stats.compression_ratio = compression;
        
        Ok(())
    }
    
    /// Generate a Semaphore proof
    ///
    /// # Arguments
    /// * `identity_nullifier` - Secret nullifier
    /// * `identity_trapdoor` - Secret trapdoor
    /// * `merkle_path` - Sibling hashes from leaf to root
    /// * `merkle_indices` - Path indices (0 = left, 1 = right)
    /// * `external_nullifier` - Scope/context identifier
    /// * `signal` - Signal being signed
    /// * `enable_pqc` - Whether to include PQC binding
    pub fn prove(
        &mut self,
        identity_nullifier: &[u8; 32],
        identity_trapdoor: &[u8; 32],
        merkle_path: &[[u8; 32]],
        merkle_indices: &[u8],
        external_nullifier: &[u8; 32],
        signal: &[u8],
        enable_pqc: bool,
    ) -> Result<(ZeroExpansionSemaphoreProof, SemaphorePublicInputs), String> {
        let start = Instant::now();
        
        if merkle_path.len() != self.tree_depth as usize {
            return Err(format!(
                "Merkle path length {} doesn't match tree depth {}",
                merkle_path.len(), self.tree_depth
            ));
        }
        
        let bases = self.bases.as_ref()
            .ok_or("GPU not initialized. Call setup_gpu() first")?;
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 1: Identity Commitment
        // ═══════════════════════════════════════════════════════════════════
        
        // Classical Poseidon commitment (simulated - would use actual Poseidon)
        let mut poseidon_input = Vec::with_capacity(64);
        poseidon_input.extend_from_slice(identity_nullifier);
        poseidon_input.extend_from_slice(identity_trapdoor);
        let identity_commitment = self.hash_poseidon(&poseidon_input);
        
        // PQC hybrid commitment if enabled
        let pqc_commitment = if enable_pqc {
            Some(PqcHybridCommitment::new(
                identity_nullifier,
                identity_trapdoor,
                &identity_commitment,
            ))
        } else {
            None
        };
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 2: Compute Merkle Root via Zero-Expansion QTT
        // ═══════════════════════════════════════════════════════════════════
        
        // Encode Merkle path into QTT
        let qtt = self.encode_merkle_path_as_qtt(merkle_path, merkle_indices)?;
        let flat_qtt = FlattenedQtt::from_qtt(&qtt);
        
        // GPU-accelerated commitment
        let qtt_commitment = qtt_batched_commit(&flat_qtt, bases, 16)
            .map_err(|e| format!("QTT commit failed: {}", e))?;
        
        // Compute Merkle root
        let merkle_root = self.compute_merkle_root(
            &identity_commitment,
            merkle_path,
            merkle_indices,
        );
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 3: Nullifier Generation
        // ═══════════════════════════════════════════════════════════════════
        
        let signal_hash = self.hash_signal(signal);
        
        // Classical nullifier
        let mut nullifier_input = Vec::with_capacity(64);
        nullifier_input.extend_from_slice(external_nullifier);
        nullifier_input.extend_from_slice(identity_nullifier);
        let nullifier_hash = self.hash_poseidon(&nullifier_input);
        
        // PQC binding if enabled
        let pqc_binding = if enable_pqc {
            let pqc_null = PqcHybridCommitment::create_nullifier(
                identity_nullifier,
                external_nullifier,
                &signal_hash,
                &nullifier_hash,
            );
            Some(pqc_null.pqc_binding)
        } else {
            None
        };
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 4: RMT Fiat-Shamir Challenges
        // ═══════════════════════════════════════════════════════════════════
        
        let transcript_hash = self.compute_transcript_hash(
            &merkle_root,
            &nullifier_hash,
            &signal_hash,
            external_nullifier,
        );
        let challenges = self.rmt.generate_field_challenges(
            transcript_hash,
            3,
            (1u64 << 63) - 25,
        );
        
        // ═══════════════════════════════════════════════════════════════════
        // PHASE 5: Structure Proof (would be full Halo2 in production)
        // ═══════════════════════════════════════════════════════════════════
        
        // Simplified structure proof for demo
        let structure_proof = self.generate_structure_proof(
            &qtt_commitment,
            &challenges,
            &merkle_root,
        );
        
        let generation_time = start.elapsed();
        
        // Update stats
        self.stats.total_proofs += 1;
        self.stats.total_time_ms += generation_time.as_millis() as u64;
        self.stats.avg_proof_ms = self.stats.total_time_ms as f64 / self.stats.total_proofs as f64;
        
        // Serialize QTT commitment
        let qtt_commit_bytes: Vec<u8> = format!("{:?}", qtt_commitment).bytes().collect();
        
        let proof = ZeroExpansionSemaphoreProof {
            qtt_commitment: qtt_commit_bytes,
            challenges,
            structure_proof,
            pqc_binding,
            tree_depth: self.tree_depth,
            generation_time_ms: generation_time.as_millis() as u64,
        };
        
        let public_inputs = SemaphorePublicInputs {
            merkle_root,
            nullifier_hash,
            signal_hash,
            external_nullifier: *external_nullifier,
            tree_depth: self.tree_depth,
        };
        
        Ok((proof, public_inputs))
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // HELPER METHODS
    // ═══════════════════════════════════════════════════════════════════════
    
    fn encode_merkle_path_as_qtt(
        &self,
        merkle_path: &[[u8; 32]],
        merkle_indices: &[u8],
    ) -> Result<QttTrain, String> {
        // Each sibling hash becomes a QTT core
        // This encodes the path in O(depth * rank²) instead of O(2^depth)
        let mut qtt = QttTrain::new(self.tree_depth as usize, 2, self.qtt_rank);
        
        // Encode path hashes and indices into QTT cores
        for (i, (sibling, index)) in merkle_path.iter().zip(merkle_indices.iter()).enumerate() {
            // Use first 4 bytes of sibling as scalar (u32 fits ScalarField::from)
            let scalar_u32 = u32::from_le_bytes(sibling[0..4].try_into().unwrap());
            let index_u32 = *index as u32;
            
            // Convert to ScalarField
            let scalar = ScalarField::from(scalar_u32);
            let index_scalar = ScalarField::from(index_u32);
            
            // Set in QTT core
            if i < qtt.cores.len() {
                qtt.cores[i].data[0] = scalar;
                qtt.cores[i].data[1] = index_scalar;
            }
        }
        
        Ok(qtt)
    }
    
    fn compute_merkle_root(
        &self,
        leaf: &[u8; 32],
        path: &[[u8; 32]],
        indices: &[u8],
    ) -> [u8; 32] {
        let mut current = *leaf;
        
        for (sibling, index) in path.iter().zip(indices.iter()) {
            current = if *index == 0 {
                self.hash_poseidon_pair(&current, sibling)
            } else {
                self.hash_poseidon_pair(sibling, &current)
            };
        }
        
        current
    }
    
    fn hash_poseidon(&self, input: &[u8]) -> [u8; 32] {
        // Simplified Poseidon (would use actual Poseidon in production)
        use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
        let mut hasher = Shake256::default();
        hasher.update(b"POSEIDON_MOCK");
        hasher.update(input);
        let mut result = [0u8; 32];
        hasher.finalize_xof().read(&mut result);
        result
    }
    
    fn hash_poseidon_pair(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(left);
        input.extend_from_slice(right);
        self.hash_poseidon(&input)
    }
    
    fn hash_signal(&self, signal: &[u8]) -> [u8; 32] {
        use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
        let mut hasher = Shake256::default();
        hasher.update(b"SIGNAL");
        hasher.update(signal);
        let mut result = [0u8; 32];
        hasher.finalize_xof().read(&mut result);
        result
    }
    
    fn compute_transcript_hash(
        &self,
        merkle_root: &[u8; 32],
        nullifier: &[u8; 32],
        signal: &[u8; 32],
        external: &[u8; 32],
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        merkle_root.hash(&mut hasher);
        nullifier.hash(&mut hasher);
        signal.hash(&mut hasher);
        external.hash(&mut hasher);
        hasher.finish()
    }
    
    fn generate_structure_proof(
        &self,
        _qtt_commitment: &icicle_bn254::curve::G1Projective,
        challenges: &[u64],
        merkle_root: &[u8; 32],
    ) -> Vec<u8> {
        // Simplified structure proof
        // In production, this would be a full Halo2 proof
        let mut proof = Vec::with_capacity(256);
        proof.extend_from_slice(b"ZERO_EXPANSION_SEMAPHORE_V3");
        proof.extend_from_slice(&[self.tree_depth]);
        proof.extend_from_slice(merkle_root);
        for c in challenges {
            proof.extend_from_slice(&c.to_le_bytes());
        }
        proof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prover_creation() {
        let prover = ZeroExpansionSemaphoreProver::new(30, 16);
        assert!(prover.is_ok());
        
        let prover = ZeroExpansionSemaphoreProver::new(50, 16);
        assert!(prover.is_ok());
        
        let prover = ZeroExpansionSemaphoreProver::new(51, 16);
        assert!(prover.is_err());
    }
}

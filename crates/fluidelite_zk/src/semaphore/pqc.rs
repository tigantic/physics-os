//! PQC Hybrid Commitment Module
//!
//! Implements post-quantum cryptographic bindings for Semaphore identities.
//! Uses SHAKE256 (SHA-3 family) for quantum-resistant hash binding alongside
//! Poseidon for ZK-efficiency.

use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};

/// PQC Hybrid Identity Commitment
///
/// Combines classical Poseidon commitment with SHAKE256 binding.
/// Even if BN254 is broken by quantum computers, the SHAKE256
/// component provides collision resistance.
#[derive(Clone, Debug)]
pub struct PqcHybridCommitment {
    /// Classical Poseidon commitment (ZK-friendly)
    pub classical: [u8; 32],
    /// SHAKE256 commitment (quantum-resistant)
    pub pqc: [u8; 32],
    /// Combined hybrid commitment
    pub hybrid: [u8; 32],
}

impl PqcHybridCommitment {
    /// Create hybrid commitment from identity secrets
    ///
    /// # Arguments
    /// * `identity_nullifier` - Secret nullifier (32 bytes)
    /// * `identity_trapdoor` - Secret trapdoor (32 bytes)
    /// * `poseidon_commit` - Pre-computed Poseidon commitment
    pub fn new(
        identity_nullifier: &[u8; 32],
        identity_trapdoor: &[u8; 32],
        poseidon_commit: &[u8; 32],
    ) -> Self {
        // PQC commitment via SHAKE256
        let mut hasher = Shake256::default();
        hasher.update(b"SEMAPHORE_PQC_IDENTITY_V1");
        hasher.update(identity_nullifier);
        hasher.update(identity_trapdoor);
        
        let mut pqc = [0u8; 32];
        hasher.finalize_xof().read(&mut pqc);
        
        // Hybrid: combine classical and PQC
        let mut hybrid_hasher = Shake256::default();
        hybrid_hasher.update(b"SEMAPHORE_HYBRID_V1");
        hybrid_hasher.update(poseidon_commit);
        hybrid_hasher.update(&pqc);
        
        let mut hybrid = [0u8; 32];
        hybrid_hasher.finalize_xof().read(&mut hybrid);
        
        Self {
            classical: *poseidon_commit,
            pqc,
            hybrid,
        }
    }
    
    /// Create PQC-hardened nullifier
    ///
    /// Binds the nullifier to both classical and PQC domains.
    pub fn create_nullifier(
        identity_nullifier: &[u8; 32],
        external_nullifier: &[u8; 32],
        signal_hash: &[u8; 32],
        classical_nullifier: &[u8; 32],
    ) -> PqcNullifier {
        // PQC binding
        let mut hasher = Shake256::default();
        hasher.update(b"SEMAPHORE_PQC_NULLIFIER_V1");
        hasher.update(identity_nullifier);
        hasher.update(external_nullifier);
        hasher.update(signal_hash);
        hasher.update(classical_nullifier);
        
        let mut pqc_binding = [0u8; 32];
        hasher.finalize_xof().read(&mut pqc_binding);
        
        PqcNullifier {
            classical: *classical_nullifier,
            pqc_binding,
        }
    }
}

/// PQC-hardened Nullifier
#[derive(Clone, Debug)]
pub struct PqcNullifier {
    /// Classical nullifier (used in ZK proof)
    pub classical: [u8; 32],
    /// PQC binding (stored off-chain for quantum migration)
    pub pqc_binding: [u8; 32],
}

/// SHAKE256-based Merkle tree node hash (PQC variant)
///
/// For use in PQC-native trees (future migration path).
pub fn shake256_merkle_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Shake256::default();
    hasher.update(b"MERKLE_NODE");
    hasher.update(left);
    hasher.update(right);
    
    let mut result = [0u8; 32];
    hasher.finalize_xof().read(&mut result);
    result
}

/// Generate random identity secrets using SHAKE256 DRBG
pub fn generate_identity_secrets(entropy: &[u8]) -> ([u8; 32], [u8; 32]) {
    let mut hasher = Shake256::default();
    hasher.update(b"SEMAPHORE_IDENTITY_GEN_V1");
    hasher.update(entropy);
    
    let mut reader = hasher.finalize_xof();
    
    let mut nullifier = [0u8; 32];
    let mut trapdoor = [0u8; 32];
    reader.read(&mut nullifier);
    reader.read(&mut trapdoor);
    
    (nullifier, trapdoor)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pqc_commitment() {
        let nullifier = [1u8; 32];
        let trapdoor = [2u8; 32];
        let poseidon = [3u8; 32]; // Mock Poseidon output
        
        let commitment = PqcHybridCommitment::new(&nullifier, &trapdoor, &poseidon);
        
        // Verify deterministic
        let commitment2 = PqcHybridCommitment::new(&nullifier, &trapdoor, &poseidon);
        assert_eq!(commitment.hybrid, commitment2.hybrid);
        
        // Verify components differ
        assert_ne!(commitment.classical, commitment.pqc);
        assert_ne!(commitment.pqc, commitment.hybrid);
        
        println!("PQC Hybrid Commitment:");
        println!("  Classical: {:?}", &commitment.classical[..8]);
        println!("  PQC:       {:?}", &commitment.pqc[..8]);
        println!("  Hybrid:    {:?}", &commitment.hybrid[..8]);
    }
    
    #[test]
    fn test_pqc_nullifier() {
        let identity_nullifier = [1u8; 32];
        let external_nullifier = [2u8; 32];
        let signal = [3u8; 32];
        let classical = [4u8; 32];
        
        let nullifier = PqcHybridCommitment::create_nullifier(
            &identity_nullifier,
            &external_nullifier,
            &signal,
            &classical,
        );
        
        assert_ne!(nullifier.classical, nullifier.pqc_binding);
        println!("PQC Nullifier binding: {:?}", &nullifier.pqc_binding[..8]);
    }
    
    #[test]
    fn test_identity_generation() {
        let entropy = b"some random entropy from secure source";
        let (nullifier, trapdoor) = generate_identity_secrets(entropy);
        
        assert_ne!(nullifier, trapdoor);
        assert_ne!(nullifier, [0u8; 32]);
        
        // Deterministic
        let (n2, t2) = generate_identity_secrets(entropy);
        assert_eq!(nullifier, n2);
        assert_eq!(trapdoor, t2);
    }
}

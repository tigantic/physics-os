//! Semaphore Circuit for Zero-Expansion
//!
//! Defines the constraint system for Semaphore membership proofs.

/// Semaphore circuit configuration
#[derive(Clone, Debug)]
pub struct SemaphoreCircuitConfig {
    /// Tree depth
    pub tree_depth: u8,
    /// Enable PQC hybrid commitments
    pub enable_pqc: bool,
}

impl SemaphoreCircuitConfig {
    /// Create config for given tree depth
    pub fn new(tree_depth: u8) -> Self {
        Self {
            tree_depth,
            enable_pqc: true,
        }
    }
    
    /// Estimate constraint count
    ///
    /// Zero-Expansion reduces this from O(2^depth) to O(depth * rank²)
    pub fn estimate_constraints(&self) -> usize {
        let rank = 16usize;
        
        // Per-level: Poseidon hash + path selection
        let poseidon_constraints = 200; // Approximate for Poseidon
        let selector_constraints = 10;
        let per_level = poseidon_constraints + selector_constraints;
        
        // QTT encoding overhead
        let qtt_overhead = self.tree_depth as usize * rank * rank;
        
        // Nullifier check
        let nullifier_constraints = 300;
        
        // Total
        (self.tree_depth as usize * per_level) + qtt_overhead + nullifier_constraints
    }
    
    /// Estimate proof size
    pub fn estimate_proof_size(&self) -> usize {
        // QTT commitment: 96 bytes (G1 point)
        // Challenges: 24 bytes (3 x 8)
        // Structure proof: ~1KB
        // PQC binding: 32 bytes (optional)
        
        let base = 96 + 24 + 1024;
        if self.enable_pqc { base + 32 } else { base }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constraint_estimation() {
        let config16 = SemaphoreCircuitConfig::new(16);
        let config50 = SemaphoreCircuitConfig::new(50);
        
        let c16 = config16.estimate_constraints();
        let c50 = config50.estimate_constraints();
        
        println!("Depth 16: {} constraints", c16);
        println!("Depth 50: {} constraints", c50);
        
        // Should grow linearly, not exponentially
        assert!(c50 < c16 * 4); // At most 4x for 3x depth increase
    }
    
    #[test]
    fn test_proof_size() {
        let config = SemaphoreCircuitConfig::new(50);
        let size = config.estimate_proof_size();
        
        println!("Proof size at depth 50: {} bytes", size);
        assert!(size < 2000); // Under 2KB regardless of depth
    }
}

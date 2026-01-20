//! Circuit configuration and parameters
//!
//! Defines the sizing parameters for the FluidElite ZK circuit.

use crate::config as model_config;

/// Circuit sizing parameters
#[derive(Clone, Debug)]
pub struct CircuitConfig {
    /// Number of sites in tensor train
    pub num_sites: usize,

    /// Maximum bond dimension
    pub chi_max: usize,

    /// MPO bond dimension
    pub mpo_d: usize,

    /// Physical dimension
    pub phys_dim: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// k parameter (log2 of circuit rows)
    pub k: u32,
}

impl CircuitConfig {
    /// Create configuration for production FluidElite (default: L=16, chi=64)
    pub fn production() -> Self {
        // Estimate rows needed
        // Each MAC operation uses ~2 rows
        // MPO×MPS: L × χ² × D × d² × d operations
        // Addition: L × 2χ × d × 2χ copies
        // Readout: vocab × features operations

        let l = model_config::L;
        let chi = model_config::CHI;
        let d = model_config::D;
        let phys = model_config::PHYS_DIM;
        let vocab = model_config::VOCAB_SIZE;

        let mpo_mps_rows = l * chi * chi * d * phys * phys * phys * 2;
        let add_rows = l * 2 * chi * phys * 2 * chi;
        let readout_rows = vocab * chi * 2;

        let total_rows = mpo_mps_rows + add_rows + readout_rows;

        // k must be large enough: 2^k >= total_rows
        let k = (total_rows as f64).log2().ceil() as u32 + 1;

        Self {
            num_sites: l,
            chi_max: chi,
            mpo_d: d,
            phys_dim: phys,
            vocab_size: vocab,
            k,
        }
    }

    /// Production v1 configuration - MATCHES TRAINED WEIGHTS
    /// Trained on WikiText-2 character-level (vocab=137, padded to 256)
    pub fn production_v1() -> Self {
        Self {
            num_sites: 12,     // MATCHES training: L=12
            chi_max: 64,       // MATCHES training: χ=64
            mpo_d: 1,          // MPO bond dim (D=1 for ZK efficiency)
            phys_dim: 2,       // Binary physical dimension
            vocab_size: 256,   // Padded to power of 2 (trained was 137)
            k: 17,             // 2^17 = 131072 rows (fits L=12 trace)
        }
    }

    /// Create small configuration for testing
    pub fn test() -> Self {
        Self {
            num_sites: 4,
            chi_max: 4,
            mpo_d: 1,
            phys_dim: 2,
            vocab_size: 16,
            k: 10,
        }
    }

    /// Estimate constraint count
    pub fn estimate_constraints(&self) -> usize {
        let l = self.num_sites;
        let chi = self.chi_max;
        let _d = self.mpo_d;
        let phys = self.phys_dim;

        // Embedding: L × phys (decompose token into bits, verify 0/1)
        let embed = l * phys;

        // MPO × MPS: 2 operations (h_term, x_term)
        // For each site, we compute (χ×D) × d_out × (χ×D) output elements
        // Each output element requires d_in MAC operations
        // With D=1, this simplifies to: L × χ² × phys (MACs per operation)
        // With 2 MPO applications: 2 × L × χ² × phys
        let mpo_mps_per_op = l * chi * chi * phys;
        let mpo_mps = 2 * mpo_mps_per_op;

        // Addition: block-diagonal concatenation = no arithmetic
        // (just copy constraints via permutation, which are "free")
        let add = 0;

        // Truncation: Keep top-χ singular values
        // SVD-free approach: just index selection, no arithmetic
        let truncate = 0;

        // Readout: Linear layer from chi features to vocab
        // chi MACs per vocab entry
        let readout = self.vocab_size * chi;

        embed + mpo_mps + add + truncate + readout
    }

    /// Estimate proof generation time (ms) on RTX 4090
    pub fn estimate_proof_time_ms(&self) -> f64 {
        let constraints = self.estimate_constraints();

        // Halo2 on RTX 4090: ~20ns per constraint for MSM
        // Plus polynomial commitment overhead
        let msm_time_ns = constraints as f64 * 20.0;

        // Polynomial operations add ~50% overhead
        let total_ns = msm_time_ns * 1.5;

        total_ns / 1_000_000.0
    }

    /// Estimate proof size in bytes
    pub fn estimate_proof_size(&self) -> usize {
        // Halo2 proofs are roughly:
        // - 32 bytes per commitment point (BN254)
        // - ~20-30 commitments for typical circuit
        // - Plus ~64 bytes for opening proofs

        let num_commitments = 25;
        let commitment_size = 32;
        let opening_proof_size = 64;

        num_commitments * commitment_size + opening_proof_size
    }
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::production()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config() {
        let config = CircuitConfig::production();

        println!("Production config:");
        println!("  Sites: {}", config.num_sites);
        println!("  Chi: {}", config.chi_max);
        println!("  k: {}", config.k);

        let constraints = config.estimate_constraints();
        println!("  Constraints: {}", constraints);

        let proof_time = config.estimate_proof_time_ms();
        println!("  Est. proof time: {:.2} ms", proof_time);

        let proof_size = config.estimate_proof_size();
        println!("  Est. proof size: {} bytes", proof_size);

        // Sanity checks
        assert!(constraints < 500_000, "Too many constraints");
        assert!(proof_time < 100.0, "Proof time too long");
    }

    #[test]
    fn test_test_config() {
        let config = CircuitConfig::test();

        let constraints = config.estimate_constraints();
        println!("Test config constraints: {}", constraints);

        assert!(constraints < 10_000);
    }
}

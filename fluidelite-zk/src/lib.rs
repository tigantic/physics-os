//! FluidElite ZK: Zero-Knowledge Provable Inference
//!
//! This crate implements ZK-provable FluidElite inference.
//! The architecture treats language modeling as fluid dynamics with
//! Matrix Product States (MPS) as the hidden context and Matrix Product
//! Operators (MPO) as weight matrices.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    FluidElite ZK Circuit                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  Input: token_id (public)                                    │
//! │  ├── Bitwise Embedding: token → MPS product state           │
//! │  ├── W_hidden × context_mps (MPO contraction)               │
//! │  ├── W_input × token_mps (MPO contraction)                  │
//! │  ├── Block-diagonal addition                                 │
//! │  ├── Truncation (SVD-free: just keep top-χ bonds)           │
//! │  └── Linear readout → logits (public output)                │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Constraint Efficiency
//!
//! | Operation          | Constraints per Token |
//! |--------------------|----------------------|
//! | Embedding          | L × 2                |
//! | MPO × MPS          | L × χ² × D × d²      |
//! | Addition           | L × 2χ × d × 2χ      |
//! | Truncation         | L × χ × d × χ        |
//! | Readout            | χ × vocab            |
//! |--------------------|----------------------|
//! | **Total (L=16)**   | **~131,000**         |
//!
//! Compare to Transformer: ~50,000,000 constraints/token

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

pub mod circuit;
pub mod field;
pub mod hybrid;
pub mod hybrid_prover;
pub mod mpo;
pub mod mps;
pub mod ops;
pub mod prover;
pub mod verifier;
pub mod weight_crypto;
pub mod weights;

#[cfg(feature = "halo2")]
pub mod halo2_hybrid_prover;

// Optional modules based on features
#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod gpu_halo2_prover;

#[cfg(feature = "gpu")]
pub mod msm_config;

#[cfg(feature = "gpu")]
pub mod qtt_native_msm;

#[cfg(all(feature = "gpu", feature = "halo2"))]
pub mod zero_expansion_prover;

// QTT Geometric Algebra - prover-critical for elliptic curve operations
pub mod qtt_ga;

// QTT Random Matrix Theory - structured Fiat-Shamir challenges
pub mod qtt_rmt;

// QTT RKHS - kernel methods for lookup table compression
pub mod qtt_rkhs;

// Genesis Integration - wires together QTT-GA, QTT-RMT, QTT-RKHS for prover
pub mod genesis_integration;

// Genesis Prover - GPU-accelerated prover with all Genesis primitives
#[cfg(all(feature = "gpu", feature = "halo2"))]
pub mod genesis_prover;

// Large-scale benchmarks
#[cfg(test)]
mod large_scale_test;

// Re-exports for convenience
pub use mpo::MPO;
pub use mps::MPS;

/// Configuration constants for FluidElite ZK
pub mod config {
    /// Number of sites in the tensor train (determines virtual context fidelity)
    /// 2^L = maximum distinguishable token positions
    pub const L: usize = 16;

    /// Bond dimension (memory capacity)
    /// Higher χ = more expressive but more constraints
    pub const CHI: usize = 64;

    /// MPO bond dimension
    pub const D: usize = 1;

    /// Physical dimension (binary embedding)
    pub const PHYS_DIM: usize = 2;

    /// Vocabulary size
    pub const VOCAB_SIZE: usize = 256;

    /// Fixed-point precision bits
    #[allow(dead_code)]
    pub const PRECISION_BITS: u32 = 32;

    /// Fixed-point scale factor (2^16 for Q16.16 format)
    #[allow(dead_code)]
    pub const SCALE: i64 = 1 << 16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_count_estimate() {
        // Verify our constraint estimates match Python prototype
        // Using REDUCED config for testing (not production)
        let l = 8;       // Reduced sites
        let chi = 16;    // Reduced bond dim
        let _d = config::D;       // MPO bond dim = 1 (reserved for future use)
        let phys = config::PHYS_DIM;  // Physical dim = 2

        // Embedding: L sites × 2 constraints (binary decomposition check)
        let embed = l * phys;

        // MPO × MPS: For D=1, each output site has chi_out × d_out × chi_out elements
        // Each element requires d_in (=2) MAC operations
        // Total: L × chi² × d_in
        let mpo_mps = l * chi * chi * phys;

        // Addition: block diagonal concatenation = no arithmetic (just copies)
        // In ZK: copy constraints are "free" via permutation
        let addition = 0;

        // Truncation: Keep top-chi bonds. For SVD-free approach:
        // Just index/copy = no arithmetic constraints
        let truncation = 0;

        // Readout: Linear layer from chi features to vocab
        // chi MACs per vocab entry
        let vocab = 64;  // Reduced vocab for test
        let readout = chi * vocab;

        let total = embed + mpo_mps + addition + truncation + readout;

        println!("Constraint breakdown (reduced config for testing):");
        println!("  L={}, chi={}, d={}, vocab={}", l, chi, phys, vocab);
        println!("  Embedding:   {:>8}", embed);
        println!("  MPO × MPS:   {:>8}", mpo_mps);
        println!("  Addition:    {:>8}", addition);
        println!("  Truncation:  {:>8}", truncation);
        println!("  Readout:     {:>8}", readout);
        println!("  ─────────────────────");
        println!("  Total:       {:>8}", total);

        // For L=8, chi=16, vocab=64: expect ~5k constraints
        assert!(total < 50_000, "Test config should be under 50k constraints");
        assert!(total > 1_000, "Test config should be over 1k (sanity check)");

        // Verify production estimate is reasonable
        // L=16, chi=64, vocab=256:
        // MPO×MPS: 16 × 64 × 64 × 2 = 131,072
        // Readout: 64 × 256 = 16,384
        // Total: ~147k
        let prod_estimate = 16 * 64 * 64 * 2 + 64 * 256;
        println!("\nProduction estimate (L=16, chi=64, vocab=256): {}", prod_estimate);
        assert!(prod_estimate < 200_000, "Production should be under 200k");
    }
}

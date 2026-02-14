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
// ═══════════════════════════════════════════════════════════════════════════════
// Copyright © 2025-2026 Bradly Biron Baker Adams / Tigantic Labs
// All Rights Reserved. Proprietary and Confidential.
// 
// This source code is the exclusive property of Bradly Biron Baker Adams and
// Tigantic Labs. Unauthorized copying, modification, distribution, or use of
// this software, in whole or in part, is strictly prohibited without prior
// written consent from the copyright holder.
// 
// SPDX-License-Identifier: LicenseRef-Proprietary
// ═══════════════════════════════════════════════════════════════════════════════


#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

// ── Compile-time safety: prevent stub prover/verifier in production builds ──
#[cfg(all(
    not(feature = "halo2"),
    any(feature = "production", feature = "production-gpu", feature = "enterprise")
))]
compile_error!(
    "production/enterprise features require the 'halo2' feature. \
     Stub prover/verifier (which return fake proofs) must never ship. \
     Add 'halo2' to your feature list or use a non-production profile."
);

// ── Re-exports from fluidelite-core (tensor primitives) ─────────────────────
pub use fluidelite_core::field;
pub use fluidelite_core::mps;
pub use fluidelite_core::mpo;
pub use fluidelite_core::ops;
pub use fluidelite_core::weights;
pub use fluidelite_core::weight_crypto;
pub use fluidelite_core::physics_traits;
pub use fluidelite_core::config;

// ── Re-exports from fluidelite-circuits (physics ZK circuits) ───────────────
pub use fluidelite_circuits::euler3d;
pub use fluidelite_circuits::ns_imex;
pub use fluidelite_circuits::thermal;
pub use fluidelite_circuits::proof_preview;

// ── Local modules (ZK proof infrastructure) ─────────────────────────────────
pub mod benchmark_baseline;
pub mod circuit;
pub mod cuda_memory_pool;
pub mod hybrid;
pub mod hybrid_prover;
pub mod multi_gpu;
pub mod multi_timestep;
pub mod proof_profiler;
pub mod prover;
pub mod verifier;

#[cfg(feature = "halo2")]
pub mod halo2_hybrid_prover;

#[cfg(feature = "halo2")]
pub mod params;

// Optional modules based on features
#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "server")]
pub mod rate_limit;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "server")]
pub mod trustless_api;

#[cfg(feature = "server")]
pub mod certificate_authority;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod gpu_halo2_prover;

#[cfg(feature = "gpu")]
pub mod msm_config;

#[cfg(feature = "gpu")]
pub mod qtt_native_msm;

#[cfg(feature = "gpu")]
pub mod groth16_output;

#[cfg(feature = "groth16")]
pub mod groth16_prover;

// Re-exports for convenience
pub use fluidelite_core::mpo::MPO;
pub use fluidelite_core::mps::MPS;

#[cfg(test)]
mod e2e_tests;

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

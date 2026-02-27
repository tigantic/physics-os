//! Prover pool: unified, batched, incremental, and compressed proving.
//!
//! This module provides a common abstraction over all physics provers
//! (Euler 3D, NS-IMEX, future solvers) and layers optimization strategies
//! on top:
//!
//! - **Traits**: `PhysicsProver`, `PhysicsVerifier`, `PhysicsProof`
//! - **Batch**: Parallel multi-timestep proving with work-stealing
//! - **Incremental**: State-caching for iterative simulations
//! - **Compression**: Zero-strip, RLE, proof bundling
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                   ProverPool                         │
//! ├─────────────────────────────────────────────────────┤
//! │  BatchProver<P>    → parallel multi-timestep        │
//! │  IncrementalProver → cache + delta detection        │
//! │  ProofCompressor   → zero-strip + RLE + bundle      │
//! ├─────────────────────────────────────────────────────┤
//! │  PhysicsProver trait                                 │
//! │  ├── Euler3DProver (stub / halo2)                   │
//! │  └── NSIMEXProver  (stub / halo2)                   │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod batch;
pub mod compressor;
pub mod incremental;

// Re-exports from fluidelite-core physics_traits (trait definitions)
pub use fluidelite_core::physics_traits::{
    PhysicsProof, PhysicsProver, PhysicsVerifier,
    ProverFactory, SolverType, UnifiedVerificationResult,
};

// Re-exports from fluidelite-circuits trait_impls (factory functions)
pub use fluidelite_circuits::trait_impls::{
    euler3d_factory, ns_imex_factory, thermal_factory,
};

pub use batch::{
    BatchConfig, BatchProver, BatchStats, BatchStatsSnapshot, BatchSummary,
    ProveJob, ProveResult,
};

pub use incremental::{
    analyze_delta, CacheKey, DeltaAnalysis, IncrementalConfig,
    IncrementalProver, IncrementalStats,
};

pub use compressor::{
    CompressedProof, CompressionConfig, CompressionMethod, CompressionStats,
    ProofBundle, ProofCompressor,
};

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Create a batch prover for Euler 3D with default configuration.
pub fn euler3d_batch(
    params: fluidelite_circuits::euler3d::Euler3DParams,
    parallelism: usize,
) -> Result<BatchProver<fluidelite_circuits::euler3d::Euler3DProver>, String> {
    let factory = euler3d_factory(params);
    let config = BatchConfig {
        parallelism,
        ..Default::default()
    };
    BatchProver::new(&factory, config)
}

/// Create a batch prover for NS-IMEX with default configuration.
pub fn ns_imex_batch(
    params: fluidelite_circuits::ns_imex::NSIMEXParams,
    parallelism: usize,
) -> Result<BatchProver<fluidelite_circuits::ns_imex::NSIMEXProver>, String> {
    let factory = ns_imex_factory(params);
    let config = BatchConfig {
        parallelism,
        ..Default::default()
    };
    BatchProver::new(&factory, config)
}

/// Create an incremental prover for Euler 3D with default configuration.
pub fn euler3d_incremental(
    params: fluidelite_circuits::euler3d::Euler3DParams,
) -> Result<IncrementalProver<fluidelite_circuits::euler3d::Euler3DProver>, String> {
    let prover = fluidelite_circuits::euler3d::Euler3DProver::new(params)?;
    Ok(IncrementalProver::new(prover, IncrementalConfig::default()))
}

/// Create an incremental prover for NS-IMEX with default configuration.
pub fn ns_imex_incremental(
    params: fluidelite_circuits::ns_imex::NSIMEXParams,
) -> Result<IncrementalProver<fluidelite_circuits::ns_imex::NSIMEXProver>, String> {
    let prover = fluidelite_circuits::ns_imex::NSIMEXProver::new(params)?;
    Ok(IncrementalProver::new(prover, IncrementalConfig::default()))
}

/// Create a default proof compressor.
pub fn default_compressor() -> ProofCompressor {
    ProofCompressor::new(CompressionConfig::default())
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler3d_batch_convenience() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let batch = euler3d_batch(params, 2);
        assert!(batch.is_ok());
        assert_eq!(batch.unwrap().pool_size(), 2);
    }

    #[test]
    fn test_ns_imex_batch_convenience() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let batch = ns_imex_batch(params, 2);
        assert!(batch.is_ok());
        assert_eq!(batch.unwrap().pool_size(), 2);
    }

    #[test]
    fn test_euler3d_incremental_convenience() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let incr = euler3d_incremental(params);
        assert!(incr.is_ok());
        assert_eq!(incr.unwrap().solver_type(), SolverType::Euler3D);
    }

    #[test]
    fn test_ns_imex_incremental_convenience() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let incr = ns_imex_incremental(params);
        assert!(incr.is_ok());
        assert_eq!(incr.unwrap().solver_type(), SolverType::NsImex);
    }

    #[test]
    fn test_default_compressor() {
        let compressor = default_compressor();
        assert_eq!(compressor.stats().total_compressed, 0);
    }

    #[test]
    fn test_full_pipeline_batch_then_compress() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        // Create batch prover
        let batch = euler3d_batch(params, 2).unwrap();

        // Create jobs
        let jobs: Vec<ProveJob> = (0..4)
            .map(|i| {
                let states: Vec<fluidelite_core::mps::MPS> = (0..5)
                    .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
                    .collect();
                let mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
                    .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
                    .collect();
                ProveJob::new(i, states, mpos)
            })
            .collect();

        // Batch prove
        let (results, summary) = batch.prove_batch(jobs).unwrap();
        assert_eq!(summary.succeeded, 4);

        // Compress results
        let mut compressor = default_compressor();
        let proofs: Vec<_> = results
            .into_iter()
            .filter_map(|r| r.outcome.ok())
            .collect();

        let bundle = compressor.bundle(&proofs).unwrap();
        assert_eq!(bundle.proof_count, 4);
        assert!(bundle.compression_ratio() >= 1.0);

        println!(
            "Pipeline: {} proofs, {:.1}x speedup, {:.1}x compression",
            summary.succeeded, summary.speedup, bundle.compression_ratio()
        );
    }

    #[test]
    fn test_full_pipeline_incremental_then_compress() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;

        let mut incr = ns_imex_incremental(params).unwrap();

        let states: Vec<fluidelite_core::mps::MPS> = (0..3)
            .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
            .collect();
        let mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
            .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
            .collect();

        // Prove twice (second should be cache hit)
        let proof1 = incr.prove(&states, &mpos).unwrap();
        let proof2 = incr.prove(&states, &mpos).unwrap();
        assert_eq!(incr.stats().cache_hits, 1);

        // Compress
        let mut compressor = default_compressor();
        let bundle = compressor.bundle(&[proof1, proof2]).unwrap();
        assert_eq!(bundle.proof_count, 2);
    }
}

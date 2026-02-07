//! Gevulot decentralized verification network integration.
//!
//! This module provides the infrastructure for submitting physics proofs
//! to the Gevulot network for public, decentralized verification.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   Gevulot Integration Layer                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  GevulotClient     → submission lifecycle management            │
//! │  ProofRegistry     → hash-indexed audit trail                   │
//! │  SharedGevulotClient → thread-safe wrapper for server routes    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Types: GevulotSubmission, VerificationRecord, GevulotConfig    │
//! │         SubmissionId, SubmissionStatus, GevulotNetwork           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use fluidelite_zk::gevulot::{GevulotClient, GevulotConfig, ProofRegistry};
//! use fluidelite_zk::prover_pool::PhysicsProof;
//!
//! // Create client
//! let mut client = GevulotClient::local();
//!
//! // Submit a proof (any PhysicsProof impl)
//! // let id = client.submit_proof(&proof).unwrap();
//!
//! // Poll for verification
//! // let status = client.check_status(&id).unwrap();
//!
//! // Register in audit trail
//! let registry = ProofRegistry::new();
//! ```
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod types;
pub mod client;
pub mod registry;

// Re-exports
pub use types::{
    GevulotConfig, GevulotNetwork, GevulotStats, GevulotSubmission,
    SubmissionId, SubmissionStatus, VerificationRecord,
    compute_proof_metadata_hash, current_unix_time,
};

pub use client::{GevulotClient, SharedGevulotClient};

pub use registry::{
    ProofRegistry, RegistryEntry, RegistryQuery, RegistrySummary,
};

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Create a local development Gevulot client (no network required).
pub fn local_client() -> GevulotClient {
    GevulotClient::local()
}

/// Create a thread-safe Gevulot client for server integration.
pub fn shared_client(config: GevulotConfig) -> Result<SharedGevulotClient, String> {
    let client = GevulotClient::new(config)?;
    Ok(SharedGevulotClient::new(client))
}

/// Submit, verify, and register a proof in one operation (local mode).
///
/// Simulates the full Gevulot lifecycle for testing and development.
pub fn submit_and_verify_local<P: fluidelite_core::physics_traits::PhysicsProof>(
    client: &mut GevulotClient,
    registry: &mut ProofRegistry,
    proof: &P,
    verifier_count: u32,
    verification_time_ms: u64,
) -> Result<(SubmissionId, VerificationRecord, u64), String> {
    let id = client.submit_proof(proof)?;
    let record = client.mark_verified(&id, verifier_count, verification_time_ms)?;
    let submission = client.get_submission(&id)?;
    let index = registry.register(submission, &record)?;
    Ok((id, record, index))
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fluidelite_circuits::euler3d::{Euler3DParams, Euler3DProver};
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;
    use fluidelite_circuits::ns_imex::{NSIMEXParams, NSIMEXProver};
    use fluidelite_core::physics_traits::{PhysicsProver, SolverType};

    #[test]
    fn test_local_client_convenience() {
        let client = local_client();
        assert_eq!(client.total_submissions(), 0);
    }

    #[test]
    fn test_full_lifecycle_euler3d() {
        let mut client = local_client();
        let mut registry = ProofRegistry::new();

        // Generate proof
        let params = Euler3DParams::test_small();
        let mut prover = Euler3DProver::new(params).unwrap();
        let states: Vec<MPS> = (0..5).map(|_| MPS::new(4, 4, 2)).collect();
        let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        let proof = prover.prove(&states, &mpos).unwrap();

        // Full lifecycle
        let (id, record, index) =
            submit_and_verify_local(&mut client, &mut registry, &proof, 3, 150)
                .unwrap();

        // Verify submission
        assert!(id.0.starts_with("gvlt-euler3d-"));
        assert_eq!(
            client.check_status(&id).unwrap(),
            SubmissionStatus::Verified
        );

        // Verify record
        assert!(record.valid);
        assert_eq!(record.verifier_count, 3);
        assert_eq!(record.solver_type, SolverType::Euler3D);

        // Verify registry
        assert_eq!(index, 0);
        let entry = registry.get(&id).unwrap();
        assert!(entry.valid);
        assert_eq!(entry.solver_type, SolverType::Euler3D);
        assert_eq!(entry.grid_bits, 4);
        assert_eq!(entry.chi_max, 4);
    }

    #[test]
    fn test_full_lifecycle_ns_imex() {
        let mut client = local_client();
        let mut registry = ProofRegistry::new();

        // Generate proof
        let params = NSIMEXParams::test_small();
        let mut prover = NSIMEXProver::new(params).unwrap();
        let states: Vec<MPS> = (0..3).map(|_| MPS::new(4, 4, 2)).collect();
        let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        let proof = prover.prove(&states, &mpos).unwrap();

        let (id, record, index) =
            submit_and_verify_local(&mut client, &mut registry, &proof, 5, 200)
                .unwrap();

        assert!(id.0.starts_with("gvlt-ns_imex-"));
        assert!(record.valid);
        assert_eq!(index, 0);

        let entry = registry.get(&id).unwrap();
        assert_eq!(entry.solver_type, SolverType::NsImex);
    }

    #[test]
    fn test_multi_solver_registry() {
        let mut client = local_client();
        let mut registry = ProofRegistry::new();

        // Euler 3D proof
        let e_params = Euler3DParams::test_small();
        let mut e_prover = Euler3DProver::new(e_params).unwrap();
        let e_states: Vec<MPS> = (0..5).map(|_| MPS::new(4, 4, 2)).collect();
        let e_mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        let e_proof = e_prover.prove(&e_states, &e_mpos).unwrap();

        // NS-IMEX proof
        let n_params = NSIMEXParams::test_small();
        let mut n_prover = NSIMEXProver::new(n_params).unwrap();
        let n_states: Vec<MPS> = (0..3).map(|_| MPS::new(4, 4, 2)).collect();
        let n_mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        let n_proof = n_prover.prove(&n_states, &n_mpos).unwrap();

        // Submit both
        submit_and_verify_local(&mut client, &mut registry, &e_proof, 3, 100)
            .unwrap();
        submit_and_verify_local(&mut client, &mut registry, &n_proof, 5, 200)
            .unwrap();

        // Verify registry
        assert_eq!(registry.total_entries(), 2);
        assert_eq!(registry.count_by_solver(SolverType::Euler3D), 1);
        assert_eq!(registry.count_by_solver(SolverType::NsImex), 1);

        // Query by solver
        let euler_q = RegistryQuery::for_solver(SolverType::Euler3D);
        let euler_results = registry.query(&euler_q);
        assert_eq!(euler_results.len(), 1);
        assert_eq!(euler_results[0].solver_type, SolverType::Euler3D);

        // Summary
        let summary = registry.summary(&RegistryQuery::all());
        assert_eq!(summary.total_entries, 2);
        assert_eq!(summary.valid_entries, 2);
    }

    #[test]
    fn test_registry_query_pipeline() {
        let mut client = local_client();
        let mut registry = ProofRegistry::new();

        // Generate multiple proofs
        let params = Euler3DParams::test_small();

        for i in 0..3u64 {
            let mut prover = Euler3DProver::new(params.clone()).unwrap();
            // Vary states to get unique IDs
            let states: Vec<MPS> = (0..5)
                .map(|j| {
                    let mut mps = MPS::new(4, 4, 2);
                    if let Some(core) = mps.cores.first_mut() {
                        if let Some(d) = core.data.first_mut() {
                            *d = fluidelite_core::field::Q16::from_raw((i * 100 + j) as i64);
                        }
                    }
                    mps
                })
                .collect();
            let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
            let proof = prover.prove(&states, &mpos).unwrap();
            submit_and_verify_local(
                &mut client,
                &mut registry,
                &proof,
                3,
                100,
            )
            .unwrap();
        }

        assert_eq!(registry.total_entries(), 3);

        // Paginated query
        let page1 = registry.query(&RegistryQuery::all().with_limit(2));
        assert_eq!(page1.len(), 2);

        let page2 = registry.query(
            &RegistryQuery::all().with_limit(2).with_offset(2),
        );
        assert_eq!(page2.len(), 1);
    }

    #[test]
    fn test_stats_aggregation() {
        let mut client = local_client();

        let params = Euler3DParams::test_small();
        let mut prover = Euler3DProver::new(params).unwrap();
        let states: Vec<MPS> = (0..5).map(|_| MPS::new(4, 4, 2)).collect();
        let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        let proof = prover.prove(&states, &mpos).unwrap();

        let id = client.submit_proof(&proof).unwrap();
        client.mark_verified(&id, 5, 250).unwrap();

        let stats = client.stats();
        assert_eq!(stats.total_submissions, 1);
        assert_eq!(stats.verified, 1);
        assert_eq!(stats.pending, 0);
        assert!(stats.success_rate() > 0.99);
        assert!((stats.avg_verification_time_ms - 250.0).abs() < 1.0);
        assert!((stats.avg_verifier_count - 5.0).abs() < 0.1);
    }
}

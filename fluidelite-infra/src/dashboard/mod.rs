//! Certificate dashboard module.
//!
//! Provides the backend infrastructure for the proof certificate dashboard,
//! including data models, analytics engine, and certificate store.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  Certificate Dashboard                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  CertificateStore    → indexed storage with query engine        │
//! │  SolverAnalytics     → per-solver metrics and percentiles       │
//! │  CertificateTimeline → time-bucketed activity views             │
//! │  DashboardSummary    → global system overview                   │
//! │  SystemHealth        → operational health metrics                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ProofCertificate    → individual proof records                 │
//! │  CertificateQuery    → filtered/sorted/paginated queries        │
//! │  PaginatedResult     → cursor-based pagination                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod models;
pub mod analytics;

// Re-exports — models
pub use models::{
    CertificateId, CertificateQuery, CertificateTimeline, DashboardSummary,
    PaginatedResult, ProofCertificate, SolverAnalytics, SortField, SortOrder,
    SystemHealth, TimelineBucket,
};

// Re-exports — analytics
pub use analytics::CertificateStore;

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Create a new certificate store.
pub fn new_store() -> CertificateStore {
    CertificateStore::new()
}

/// Generate a unique certificate ID.
pub fn generate_cert_id(
    solver: fluidelite_core::physics_traits::SolverType,
    sequence: u64,
) -> CertificateId {
    let ts = crate::gevulot::current_unix_time();
    CertificateId(format!("cert-{}-{}-{:08x}", solver, sequence, ts))
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fluidelite_core::physics_traits::SolverType;

    #[test]
    fn test_new_store() {
        let store = new_store();
        assert_eq!(store.total(), 0);
    }

    #[test]
    fn test_generate_cert_id() {
        let id = generate_cert_id(SolverType::Euler3D, 42);
        assert!(id.0.starts_with("cert-euler3d-42-"));
    }

    #[test]
    fn test_full_dashboard_pipeline() {
        let mut store = new_store();

        // Insert certificates
        for i in 0..5 {
            let cert = ProofCertificate::from_verification(
                generate_cert_id(SolverType::Euler3D, i),
                SolverType::Euler3D,
                true,
                4,
                4,
                5000,
                14,
                100 + i as u64 * 10,
                500 + i as u64 * 50,
                800,
                &[i as u64, 2, 3, 4],
                &[5, 6, 7, 8],
                &[9, 10, 11, 12],
                0.001,
            );
            store.insert(cert).unwrap();
        }

        for i in 0..3 {
            let cert = ProofCertificate::from_verification(
                generate_cert_id(SolverType::NsImex, i + 100),
                SolverType::NsImex,
                i < 2, // 2 valid, 1 invalid
                6,
                8,
                12000,
                16,
                200 + i as u64 * 20,
                800 + i as u64 * 100,
                1200,
                &[i as u64 + 100, 200, 300, 400],
                &[500, 600, 700, 800],
                &[900, 1000, 1100, 1200],
                0.0005,
            );
            store.insert(cert).unwrap();
        }

        // Dashboard summary
        let summary = store.dashboard_summary();
        assert_eq!(summary.total_certificates, 8);
        assert_eq!(summary.valid_certificates, 7);
        assert_eq!(summary.failed_certificates, 1);

        // Euler analytics
        let euler = store.solver_analytics(SolverType::Euler3D);
        assert_eq!(euler.total_proofs, 5);
        assert_eq!(euler.valid_proofs, 5);
        assert_eq!(euler.success_rate, 1.0);

        // NS-IMEX analytics
        let ns = store.solver_analytics(SolverType::NsImex);
        assert_eq!(ns.total_proofs, 3);
        assert_eq!(ns.valid_proofs, 2);

        // Query by solver
        let q = CertificateQuery {
            solver_type: Some(SolverType::NsImex),
            ..Default::default()
        };
        let result = store.query(&q);
        assert_eq!(result.total, 3);

        // Health
        let health = store.system_health();
        assert!(health.healthy);
        assert!(health.last_proof_at.is_some());
    }
}

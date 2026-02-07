//! Dashboard data models.
//!
//! Defines the types for the certificate dashboard backend,
//! including proof certificates, timeline views, solver analytics,
//! and system health metrics.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use fluidelite_core::physics_traits::SolverType;

// ═══════════════════════════════════════════════════════════════════════════
// Certificate Types
// ═══════════════════════════════════════════════════════════════════════════

/// Unique certificate identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertificateId(pub String);

impl std::fmt::Display for CertificateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A proof certificate displayed in the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// Certificate ID.
    pub id: CertificateId,

    /// Solver type.
    pub solver_type: SolverType,

    /// Whether the proof is valid.
    pub valid: bool,

    /// Grid resolution.
    pub grid_bits: usize,

    /// Bond dimension.
    pub chi_max: usize,

    /// Number of constraints.
    pub num_constraints: usize,

    /// Circuit k parameter.
    pub k: u32,

    /// Proof generation time (ms).
    pub generation_time_ms: u64,

    /// Verification time (µs).
    pub verification_time_us: u64,

    /// Proof size in bytes.
    pub proof_size: usize,

    /// Input state hash (hex).
    pub input_hash: String,

    /// Output state hash (hex).
    pub output_hash: String,

    /// Parameters hash (hex).
    pub params_hash: String,

    /// Creation timestamp (Unix seconds).
    pub created_at: u64,

    /// Gevulot verification status (if submitted).
    pub gevulot_status: Option<String>,

    /// Gevulot transaction hash (if verified).
    pub gevulot_tx_hash: Option<String>,

    /// Number of Gevulot verifiers.
    pub gevulot_verifiers: Option<u32>,

    /// Lean formal proof references.
    pub lean_proofs: Vec<String>,

    /// Tenant ID (for multi-tenant mode).
    pub tenant_id: Option<String>,

    /// Maximum conservation residual.
    pub max_residual: f64,
}

impl ProofCertificate {
    /// Create from a unified verification result and proof metadata.
    pub fn from_verification(
        id: CertificateId,
        solver: SolverType,
        valid: bool,
        grid_bits: usize,
        chi_max: usize,
        num_constraints: usize,
        k: u32,
        generation_time_ms: u64,
        verification_time_us: u64,
        proof_size: usize,
        input_hash_limbs: &[u64; 4],
        output_hash_limbs: &[u64; 4],
        params_hash_limbs: &[u64; 4],
        max_residual: f64,
    ) -> Self {
        Self {
            id,
            solver_type: solver,
            valid,
            grid_bits,
            chi_max,
            num_constraints,
            k,
            generation_time_ms,
            verification_time_us,
            proof_size,
            input_hash: format_hash_limbs(input_hash_limbs),
            output_hash: format_hash_limbs(output_hash_limbs),
            params_hash: format_hash_limbs(params_hash_limbs),
            created_at: crate::gevulot::current_unix_time(),
            gevulot_status: None,
            gevulot_tx_hash: None,
            gevulot_verifiers: None,
            lean_proofs: solver
                .lean_proofs()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            tenant_id: None,
            max_residual,
        }
    }
}

/// Format 4 u64 limbs as a hex string.
fn format_hash_limbs(limbs: &[u64; 4]) -> String {
    format!(
        "{:016x}{:016x}{:016x}{:016x}",
        limbs[0], limbs[1], limbs[2], limbs[3]
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Timeline View
// ═══════════════════════════════════════════════════════════════════════════

/// A time-bucketed view of proof activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineBucket {
    /// Start of bucket (Unix seconds).
    pub start: u64,

    /// End of bucket (Unix seconds).
    pub end: u64,

    /// Number of proofs in this bucket.
    pub proof_count: usize,

    /// Valid proofs.
    pub valid_count: usize,

    /// Failed proofs.
    pub failed_count: usize,

    /// Average generation time (ms).
    pub avg_generation_ms: f64,

    /// Average verification time (µs).
    pub avg_verification_us: f64,

    /// Total proof bytes.
    pub total_bytes: u64,

    /// Breakdown by solver type.
    pub by_solver: HashMap<String, usize>,
}

/// Certificate timeline spanning a time range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateTimeline {
    /// Timeline buckets.
    pub buckets: Vec<TimelineBucket>,

    /// Bucket width in seconds.
    pub bucket_width_secs: u64,

    /// Total proofs across all buckets.
    pub total_proofs: usize,

    /// Time range start.
    pub start: u64,

    /// Time range end.
    pub end: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Solver Analytics
// ═══════════════════════════════════════════════════════════════════════════

/// Per-solver analytics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverAnalytics {
    /// Solver name.
    pub solver: String,

    /// Solver type.
    pub solver_type: SolverType,

    /// Total proofs generated.
    pub total_proofs: usize,

    /// Valid proofs.
    pub valid_proofs: usize,

    /// Success rate.
    pub success_rate: f64,

    /// Average generation time (ms).
    pub avg_generation_ms: f64,

    /// Median generation time (ms).
    pub median_generation_ms: f64,

    /// P95 generation time (ms).
    pub p95_generation_ms: f64,

    /// P99 generation time (ms).
    pub p99_generation_ms: f64,

    /// Average verification time (µs).
    pub avg_verification_us: f64,

    /// Average proof size (bytes).
    pub avg_proof_size: f64,

    /// Total proof bytes.
    pub total_bytes: u64,

    /// Average number of constraints.
    pub avg_constraints: f64,

    /// Throughput (proofs per second).
    pub throughput: f64,

    /// Average max residual.
    pub avg_max_residual: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Dashboard Summary
// ═══════════════════════════════════════════════════════════════════════════

/// Overall dashboard summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    /// Total certificates.
    pub total_certificates: usize,

    /// Valid certificates.
    pub valid_certificates: usize,

    /// Failed certificates.
    pub failed_certificates: usize,

    /// Overall success rate.
    pub success_rate: f64,

    /// Total proof bytes.
    pub total_proof_bytes: u64,

    /// Average generation time (ms).
    pub avg_generation_ms: f64,

    /// Average verification time (µs).
    pub avg_verification_us: f64,

    /// Per-solver breakdown.
    pub solvers: Vec<SolverAnalytics>,

    /// Number of Gevulot-verified proofs.
    pub gevulot_verified: usize,

    /// Number of pending Gevulot submissions.
    pub gevulot_pending: usize,

    /// Number of active tenants.
    pub active_tenants: usize,

    /// System uptime in seconds.
    pub uptime_secs: u64,

    /// Dashboard generation timestamp.
    pub generated_at: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// System Health
// ═══════════════════════════════════════════════════════════════════════════

/// System health metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Whether the system is operational.
    pub healthy: bool,

    /// CPU utilization (0.0-1.0).
    pub cpu_utilization: f64,

    /// Memory usage in bytes.
    pub memory_bytes: u64,

    /// Memory limit in bytes.
    pub memory_limit_bytes: u64,

    /// Number of active prover threads.
    pub active_provers: usize,

    /// Number of queued proof jobs.
    pub queued_jobs: usize,

    /// Gevulot network connectivity.
    pub gevulot_connected: bool,

    /// Time of last successful proof.
    pub last_proof_at: Option<u64>,

    /// Errors in the last hour.
    pub errors_last_hour: usize,

    /// Uptime in seconds.
    pub uptime_secs: u64,
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            healthy: true,
            cpu_utilization: 0.0,
            memory_bytes: 0,
            memory_limit_bytes: 0,
            active_provers: 0,
            queued_jobs: 0,
            gevulot_connected: false,
            last_proof_at: None,
            errors_last_hour: 0,
            uptime_secs: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dashboard Query
// ═══════════════════════════════════════════════════════════════════════════

/// Query parameters for certificate listing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CertificateQuery {
    /// Filter by solver type.
    pub solver_type: Option<SolverType>,

    /// Filter by validity.
    pub valid_only: Option<bool>,

    /// Filter by minimum grid_bits.
    pub min_grid_bits: Option<usize>,

    /// Filter by minimum chi_max.
    pub min_chi_max: Option<usize>,

    /// Filter by tenant ID.
    pub tenant_id: Option<String>,

    /// After timestamp (Unix seconds).
    pub after: Option<u64>,

    /// Before timestamp (Unix seconds).
    pub before: Option<u64>,

    /// Sort field.
    pub sort_by: Option<SortField>,

    /// Sort direction.
    pub sort_order: Option<SortOrder>,

    /// Page size.
    pub limit: Option<usize>,

    /// Page offset.
    pub offset: Option<usize>,
}

/// Sortable fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortField {
    CreatedAt,
    GenerationTime,
    VerificationTime,
    ProofSize,
    Constraints,
}

/// Sort direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Paginated query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResult<T> {
    /// Items on this page.
    pub items: Vec<T>,

    /// Total matching items.
    pub total: usize,

    /// Current page offset.
    pub offset: usize,

    /// Page size.
    pub limit: usize,

    /// Whether there are more items.
    pub has_more: bool,
}

impl<T> PaginatedResult<T> {
    /// Create a paginated result from a full list.
    pub fn from_full(items: Vec<T>, total: usize, offset: usize, limit: usize) -> Self {
        Self {
            has_more: offset + items.len() < total,
            items,
            total,
            offset,
            limit,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_id() {
        let id = CertificateId("cert-001".into());
        assert_eq!(id.to_string(), "cert-001");
    }

    #[test]
    fn test_format_hash_limbs() {
        let limbs = [0u64, 0, 0, 0];
        let hash = format_hash_limbs(&limbs);
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, "0000000000000000000000000000000000000000000000000000000000000000");
    }

    #[test]
    fn test_proof_certificate_creation() {
        let cert = ProofCertificate::from_verification(
            CertificateId("test-001".into()),
            SolverType::Euler3D,
            true,
            4,
            4,
            5000,
            14,
            100,
            500,
            800,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            0.001,
        );
        assert!(cert.valid);
        assert_eq!(cert.solver_type, SolverType::Euler3D);
        assert_eq!(cert.grid_bits, 4);
        assert!(cert.lean_proofs.contains(&"EulerConservation.lean".to_string()));
        assert!(cert.created_at > 0);
    }

    #[test]
    fn test_dashboard_summary() {
        let summary = DashboardSummary {
            total_certificates: 100,
            valid_certificates: 95,
            failed_certificates: 5,
            success_rate: 0.95,
            total_proof_bytes: 80_000,
            avg_generation_ms: 50.0,
            avg_verification_us: 200.0,
            solvers: vec![],
            gevulot_verified: 80,
            gevulot_pending: 5,
            active_tenants: 3,
            uptime_secs: 3600,
            generated_at: crate::gevulot::current_unix_time(),
        };
        assert_eq!(summary.total_certificates, 100);
    }

    #[test]
    fn test_system_health_default() {
        let health = SystemHealth::default();
        assert!(health.healthy);
        assert_eq!(health.active_provers, 0);
    }

    #[test]
    fn test_paginated_result() {
        let result = PaginatedResult::from_full(vec![1, 2, 3], 10, 0, 3);
        assert!(result.has_more);
        assert_eq!(result.items.len(), 3);

        let last = PaginatedResult::from_full(vec![9, 10], 10, 8, 3);
        assert!(!last.has_more);
    }

    #[test]
    fn test_certificate_query_default() {
        let q = CertificateQuery::default();
        assert!(q.solver_type.is_none());
        assert!(q.valid_only.is_none());
    }

    #[test]
    fn test_solver_analytics_default() {
        let analytics = SolverAnalytics::default();
        assert_eq!(analytics.total_proofs, 0);
        assert_eq!(analytics.success_rate, 0.0);
    }

    #[test]
    fn test_certificate_serialization() {
        let cert = ProofCertificate::from_verification(
            CertificateId("ser-001".into()),
            SolverType::NsImex,
            true,
            6,
            8,
            12000,
            16,
            200,
            1000,
            1200,
            &[100, 200, 300, 400],
            &[500, 600, 700, 800],
            &[900, 1000, 1100, 1200],
            0.0005,
        );
        let json = serde_json::to_string(&cert).unwrap();
        assert!(json.contains("ns_imex"));
        assert!(json.contains("ser-001"));

        let deser: ProofCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.solver_type, SolverType::NsImex);
    }
}

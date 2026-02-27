//! Dashboard analytics engine.
//!
//! Computes aggregated statistics, timelines, per-solver analytics,
//! and system health metrics from the certificate store.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::collections::HashMap;

use fluidelite_core::physics_traits::SolverType;

use super::models::{
    CertificateQuery, CertificateTimeline, DashboardSummary, PaginatedResult,
    ProofCertificate, SolverAnalytics, SystemHealth, TimelineBucket,
};

// ═══════════════════════════════════════════════════════════════════════════
// Certificate Store
// ═══════════════════════════════════════════════════════════════════════════

/// In-memory certificate store with indexed lookups.
///
/// Provides the data backing for the dashboard analytics engine.
/// In production, this would be backed by a persistent store.
pub struct CertificateStore {
    /// All certificates in insertion order.
    certificates: Vec<ProofCertificate>,

    /// Index by certificate ID.
    by_id: HashMap<String, usize>,

    /// Index by solver type.
    by_solver: HashMap<SolverType, Vec<usize>>,

    /// Index by tenant.
    by_tenant: HashMap<String, Vec<usize>>,

    /// System start time (Unix seconds).
    start_time: u64,

    /// Error count in the last hour.
    error_count: usize,

    /// Last proof timestamp.
    last_proof_at: Option<u64>,
}

impl CertificateStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            certificates: Vec::new(),
            by_id: HashMap::new(),
            by_solver: HashMap::new(),
            by_tenant: HashMap::new(),
            start_time: crate::gevulot::current_unix_time(),
            error_count: 0,
            last_proof_at: None,
        }
    }

    /// Insert a certificate into the store.
    pub fn insert(&mut self, cert: ProofCertificate) -> Result<(), String> {
        let id_str = cert.id.0.clone();
        if self.by_id.contains_key(&id_str) {
            return Err(format!("Certificate already exists: {}", id_str));
        }

        let idx = self.certificates.len();
        self.by_id.insert(id_str, idx);

        self.by_solver
            .entry(cert.solver_type)
            .or_default()
            .push(idx);

        if let Some(ref tenant) = cert.tenant_id {
            self.by_tenant
                .entry(tenant.clone())
                .or_default()
                .push(idx);
        }

        self.last_proof_at = Some(cert.created_at);
        self.certificates.push(cert);

        Ok(())
    }

    /// Record an error event.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Get a certificate by ID.
    pub fn get(&self, id: &str) -> Option<&ProofCertificate> {
        self.by_id.get(id).map(|&idx| &self.certificates[idx])
    }

    /// Total certificates.
    pub fn total(&self) -> usize {
        self.certificates.len()
    }

    /// Query certificates with filtering, sorting, and pagination.
    pub fn query(
        &self,
        q: &CertificateQuery,
    ) -> PaginatedResult<ProofCertificate> {
        let mut matching: Vec<&ProofCertificate> = self
            .certificates
            .iter()
            .filter(|c| {
                if let Some(solver) = q.solver_type {
                    if c.solver_type != solver {
                        return false;
                    }
                }
                if let Some(valid) = q.valid_only {
                    if c.valid != valid {
                        return false;
                    }
                }
                if let Some(min_gb) = q.min_grid_bits {
                    if c.grid_bits < min_gb {
                        return false;
                    }
                }
                if let Some(min_chi) = q.min_chi_max {
                    if c.chi_max < min_chi {
                        return false;
                    }
                }
                if let Some(ref tenant) = q.tenant_id {
                    if c.tenant_id.as_ref() != Some(tenant) {
                        return false;
                    }
                }
                if let Some(after) = q.after {
                    if c.created_at < after {
                        return false;
                    }
                }
                if let Some(before) = q.before {
                    if c.created_at > before {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Sort
        if let Some(ref field) = q.sort_by {
            let desc = matches!(
                q.sort_order,
                Some(super::models::SortOrder::Descending)
            );
            matching.sort_by(|a, b| {
                let cmp = match field {
                    super::models::SortField::CreatedAt => {
                        a.created_at.cmp(&b.created_at)
                    }
                    super::models::SortField::GenerationTime => {
                        a.generation_time_ms.cmp(&b.generation_time_ms)
                    }
                    super::models::SortField::VerificationTime => {
                        a.verification_time_us.cmp(&b.verification_time_us)
                    }
                    super::models::SortField::ProofSize => {
                        a.proof_size.cmp(&b.proof_size)
                    }
                    super::models::SortField::Constraints => {
                        a.num_constraints.cmp(&b.num_constraints)
                    }
                };
                if desc { cmp.reverse() } else { cmp }
            });
        }

        let total = matching.len();
        let offset = q.offset.unwrap_or(0);
        let limit = q.limit.unwrap_or(100);

        let items: Vec<ProofCertificate> = matching
            .into_iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect();

        PaginatedResult::from_full(items, total, offset, limit)
    }

    /// Generate the full dashboard summary.
    pub fn dashboard_summary(&self) -> DashboardSummary {
        let total = self.certificates.len();
        let valid = self.certificates.iter().filter(|c| c.valid).count();
        let failed = total - valid;

        let success_rate = if total == 0 {
            0.0
        } else {
            valid as f64 / total as f64
        };

        let total_bytes: u64 = self
            .certificates
            .iter()
            .map(|c| c.proof_size as u64)
            .sum();

        let avg_gen = if total == 0 {
            0.0
        } else {
            self.certificates
                .iter()
                .map(|c| c.generation_time_ms as f64)
                .sum::<f64>()
                / total as f64
        };

        let avg_ver = if total == 0 {
            0.0
        } else {
            self.certificates
                .iter()
                .map(|c| c.verification_time_us as f64)
                .sum::<f64>()
                / total as f64
        };

        let gevulot_verified = self
            .certificates
            .iter()
            .filter(|c| c.gevulot_status.as_deref() == Some("verified"))
            .count();

        let gevulot_pending = self
            .certificates
            .iter()
            .filter(|c| {
                matches!(
                    c.gevulot_status.as_deref(),
                    Some("submitted") | Some("pending") | Some("verifying")
                )
            })
            .count();

        let active_tenants: usize = self.by_tenant.len();

        let now = crate::gevulot::current_unix_time();
        let uptime = now.saturating_sub(self.start_time);

        // Per-solver analytics
        let solvers = vec![
            self.solver_analytics(SolverType::Euler3D),
            self.solver_analytics(SolverType::NsImex),
        ];

        DashboardSummary {
            total_certificates: total,
            valid_certificates: valid,
            failed_certificates: failed,
            success_rate,
            total_proof_bytes: total_bytes,
            avg_generation_ms: avg_gen,
            avg_verification_us: avg_ver,
            solvers,
            gevulot_verified,
            gevulot_pending,
            active_tenants,
            uptime_secs: uptime,
            generated_at: now,
        }
    }

    /// Compute analytics for a specific solver.
    pub fn solver_analytics(&self, solver: SolverType) -> SolverAnalytics {
        let certs: Vec<&ProofCertificate> = self
            .certificates
            .iter()
            .filter(|c| c.solver_type == solver)
            .collect();

        let total = certs.len();
        if total == 0 {
            return SolverAnalytics {
                solver: solver.name().to_string(),
                solver_type: solver,
                ..Default::default()
            };
        }

        let valid = certs.iter().filter(|c| c.valid).count();
        let success_rate = valid as f64 / total as f64;

        let mut gen_times: Vec<u64> =
            certs.iter().map(|c| c.generation_time_ms).collect();
        gen_times.sort();

        let avg_gen = gen_times.iter().sum::<u64>() as f64 / total as f64;
        let median_gen = gen_times[total / 2] as f64;
        let p95_gen = gen_times[((total as f64 * 0.95) as usize)
            .min(total.saturating_sub(1))]
            as f64;
        let p99_gen = gen_times[((total as f64 * 0.99) as usize)
            .min(total.saturating_sub(1))]
            as f64;

        let avg_ver = certs
            .iter()
            .map(|c| c.verification_time_us as f64)
            .sum::<f64>()
            / total as f64;

        let avg_size =
            certs.iter().map(|c| c.proof_size as f64).sum::<f64>()
                / total as f64;

        let total_bytes: u64 =
            certs.iter().map(|c| c.proof_size as u64).sum();

        let avg_constraints = certs
            .iter()
            .map(|c| c.num_constraints as f64)
            .sum::<f64>()
            / total as f64;

        let total_time_secs: f64 =
            gen_times.iter().sum::<u64>() as f64 / 1000.0;
        let throughput = if total_time_secs > 0.0 {
            total as f64 / total_time_secs
        } else {
            0.0
        };

        let avg_residual = certs
            .iter()
            .map(|c| c.max_residual)
            .sum::<f64>()
            / total as f64;

        SolverAnalytics {
            solver: solver.name().to_string(),
            solver_type: solver,
            total_proofs: total,
            valid_proofs: valid,
            success_rate,
            avg_generation_ms: avg_gen,
            median_generation_ms: median_gen,
            p95_generation_ms: p95_gen,
            p99_generation_ms: p99_gen,
            avg_verification_us: avg_ver,
            avg_proof_size: avg_size,
            total_bytes,
            avg_constraints,
            throughput,
            avg_max_residual: avg_residual,
        }
    }

    /// Build a timeline of certificates.
    pub fn timeline(
        &self,
        start: u64,
        end: u64,
        bucket_width_secs: u64,
    ) -> CertificateTimeline {
        if bucket_width_secs == 0 || start >= end {
            return CertificateTimeline {
                buckets: Vec::new(),
                bucket_width_secs,
                total_proofs: 0,
                start,
                end,
            };
        }

        let num_buckets =
            ((end - start + bucket_width_secs - 1) / bucket_width_secs) as usize;

        let mut buckets: Vec<TimelineBucket> = (0..num_buckets)
            .map(|i| {
                let b_start = start + (i as u64 * bucket_width_secs);
                let b_end = (b_start + bucket_width_secs).min(end);
                TimelineBucket {
                    start: b_start,
                    end: b_end,
                    proof_count: 0,
                    valid_count: 0,
                    failed_count: 0,
                    avg_generation_ms: 0.0,
                    avg_verification_us: 0.0,
                    total_bytes: 0,
                    by_solver: HashMap::new(),
                }
            })
            .collect();

        let mut total_proofs = 0;

        for cert in &self.certificates {
            if cert.created_at < start || cert.created_at >= end {
                continue;
            }

            let bucket_idx =
                ((cert.created_at - start) / bucket_width_secs) as usize;
            if bucket_idx >= buckets.len() {
                continue;
            }

            let bucket = &mut buckets[bucket_idx];
            bucket.proof_count += 1;
            total_proofs += 1;

            if cert.valid {
                bucket.valid_count += 1;
            } else {
                bucket.failed_count += 1;
            }

            // Running average
            let n = bucket.proof_count as f64;
            bucket.avg_generation_ms = bucket.avg_generation_ms
                * ((n - 1.0) / n)
                + cert.generation_time_ms as f64 / n;
            bucket.avg_verification_us = bucket.avg_verification_us
                * ((n - 1.0) / n)
                + cert.verification_time_us as f64 / n;

            bucket.total_bytes += cert.proof_size as u64;

            *bucket
                .by_solver
                .entry(cert.solver_type.to_string())
                .or_insert(0) += 1;
        }

        CertificateTimeline {
            buckets,
            bucket_width_secs,
            total_proofs,
            start,
            end,
        }
    }

    /// Get system health metrics.
    pub fn system_health(&self) -> SystemHealth {
        let now = crate::gevulot::current_unix_time();
        SystemHealth {
            healthy: true,
            cpu_utilization: 0.0, // Would come from OS metrics
            memory_bytes: 0,
            memory_limit_bytes: 0,
            active_provers: 0,
            queued_jobs: 0,
            gevulot_connected: false,
            last_proof_at: self.last_proof_at,
            errors_last_hour: self.error_count,
            uptime_secs: now.saturating_sub(self.start_time),
        }
    }
}

impl Default for CertificateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dashboard::models::CertificateId;

    fn make_cert(
        id: &str,
        solver: SolverType,
        valid: bool,
        gen_ms: u64,
        ver_us: u64,
        size: usize,
    ) -> ProofCertificate {
        ProofCertificate::from_verification(
            CertificateId(id.into()),
            solver,
            valid,
            4,
            4,
            5000,
            14,
            gen_ms,
            ver_us,
            size,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            0.001,
        )
    }

    #[test]
    fn test_store_creation() {
        let store = CertificateStore::new();
        assert_eq!(store.total(), 0);
    }

    #[test]
    fn test_store_insert() {
        let mut store = CertificateStore::new();
        let cert = make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800);
        store.insert(cert).unwrap();
        assert_eq!(store.total(), 1);
    }

    #[test]
    fn test_store_duplicate() {
        let mut store = CertificateStore::new();
        let cert = make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800);
        store.insert(cert.clone()).unwrap();
        assert!(store.insert(cert).is_err());
    }

    #[test]
    fn test_store_get() {
        let mut store = CertificateStore::new();
        let cert = make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800);
        store.insert(cert).unwrap();

        let found = store.get("c-001").unwrap();
        assert_eq!(found.solver_type, SolverType::Euler3D);
        assert!(found.valid);
    }

    #[test]
    fn test_store_query_all() {
        let mut store = CertificateStore::new();
        store
            .insert(make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800))
            .unwrap();
        store
            .insert(make_cert("c-002", SolverType::NsImex, true, 200, 600, 1200))
            .unwrap();

        let result = store.query(&CertificateQuery::default());
        assert_eq!(result.total, 2);
        assert_eq!(result.items.len(), 2);
    }

    #[test]
    fn test_store_query_by_solver() {
        let mut store = CertificateStore::new();
        store
            .insert(make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800))
            .unwrap();
        store
            .insert(make_cert("c-002", SolverType::NsImex, true, 200, 600, 1200))
            .unwrap();

        let q = CertificateQuery {
            solver_type: Some(SolverType::Euler3D),
            ..Default::default()
        };
        let result = store.query(&q);
        assert_eq!(result.total, 1);
        assert_eq!(result.items[0].solver_type, SolverType::Euler3D);
    }

    #[test]
    fn test_store_query_pagination() {
        let mut store = CertificateStore::new();
        for i in 0..10 {
            store
                .insert(make_cert(
                    &format!("c-{:03}", i),
                    SolverType::Euler3D,
                    true,
                    100,
                    500,
                    800,
                ))
                .unwrap();
        }

        let q = CertificateQuery {
            limit: Some(3),
            offset: Some(0),
            ..Default::default()
        };
        let page1 = store.query(&q);
        assert_eq!(page1.items.len(), 3);
        assert!(page1.has_more);

        let q2 = CertificateQuery {
            limit: Some(3),
            offset: Some(9),
            ..Default::default()
        };
        let last = store.query(&q2);
        assert_eq!(last.items.len(), 1);
        assert!(!last.has_more);
    }

    #[test]
    fn test_dashboard_summary() {
        let mut store = CertificateStore::new();
        store
            .insert(make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800))
            .unwrap();
        store
            .insert(make_cert("c-002", SolverType::NsImex, true, 200, 600, 1200))
            .unwrap();
        store
            .insert(make_cert(
                "c-003",
                SolverType::Euler3D,
                false,
                300,
                700,
                1000,
            ))
            .unwrap();

        let summary = store.dashboard_summary();
        assert_eq!(summary.total_certificates, 3);
        assert_eq!(summary.valid_certificates, 2);
        assert_eq!(summary.failed_certificates, 1);
        assert!((summary.success_rate - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(summary.solvers.len(), 2);
    }

    #[test]
    fn test_solver_analytics() {
        let mut store = CertificateStore::new();
        store
            .insert(make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800))
            .unwrap();
        store
            .insert(make_cert("c-002", SolverType::Euler3D, true, 200, 600, 1000))
            .unwrap();

        let analytics = store.solver_analytics(SolverType::Euler3D);
        assert_eq!(analytics.total_proofs, 2);
        assert_eq!(analytics.valid_proofs, 2);
        assert_eq!(analytics.success_rate, 1.0);
        assert!((analytics.avg_generation_ms - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_solver_analytics_empty() {
        let store = CertificateStore::new();
        let analytics = store.solver_analytics(SolverType::NsImex);
        assert_eq!(analytics.total_proofs, 0);
        assert_eq!(analytics.success_rate, 0.0);
    }

    #[test]
    fn test_timeline() {
        let mut store = CertificateStore::new();
        let now = crate::gevulot::current_unix_time();

        let mut cert1 = make_cert("c-001", SolverType::Euler3D, true, 100, 500, 800);
        cert1.created_at = now - 50;
        store.insert(cert1).unwrap();

        let mut cert2 = make_cert("c-002", SolverType::NsImex, true, 200, 600, 1200);
        cert2.created_at = now - 10;
        store.insert(cert2).unwrap();

        let timeline = store.timeline(now - 120, now, 60);
        assert_eq!(timeline.buckets.len(), 2);
        assert_eq!(timeline.total_proofs, 2);
    }

    #[test]
    fn test_timeline_empty_range() {
        let store = CertificateStore::new();
        let timeline = store.timeline(100, 50, 10); // start >= end
        assert!(timeline.buckets.is_empty());
    }

    #[test]
    fn test_system_health() {
        let store = CertificateStore::new();
        let health = store.system_health();
        assert!(health.healthy);
        assert!(health.uptime_secs < 2); // Just created
    }

    #[test]
    fn test_record_error() {
        let mut store = CertificateStore::new();
        store.record_error();
        store.record_error();
        let health = store.system_health();
        assert_eq!(health.errors_last_hour, 2);
    }
}

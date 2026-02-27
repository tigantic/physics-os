//! Usage metering and billing.
//!
//! Tracks per-tenant resource consumption (proofs generated, bytes processed,
//! compute time) and enforces rate limits.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use fluidelite_core::physics_traits::SolverType;

use super::tenant::{TenantConfig, TenantId};

// ═══════════════════════════════════════════════════════════════════════════
// Usage Record
// ═══════════════════════════════════════════════════════════════════════════

/// A single usage event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Tenant that incurred the usage.
    pub tenant_id: TenantId,

    /// Solver type used.
    pub solver_type: SolverType,

    /// Timestamp (Unix seconds).
    pub timestamp: u64,

    /// Proof generation time (ms).
    pub generation_time_ms: u64,

    /// Proof size (bytes).
    pub proof_bytes: usize,

    /// Number of constraints.
    pub num_constraints: usize,

    /// Grid bits.
    pub grid_bits: usize,

    /// Bond dimension.
    pub chi_max: usize,

    /// Whether the proof was successful.
    pub success: bool,
}

// ═══════════════════════════════════════════════════════════════════════════
// Tenant Usage Summary
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregated usage for a tenant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TenantUsage {
    /// Tenant ID.
    pub tenant_id: TenantId,

    /// Total proofs generated.
    pub total_proofs: u64,

    /// Successful proofs.
    pub successful_proofs: u64,

    /// Failed proofs.
    pub failed_proofs: u64,

    /// Total proof bytes generated.
    pub total_bytes: u64,

    /// Total compute time (ms).
    pub total_compute_ms: u64,

    /// Proofs in the current hour.
    pub proofs_this_hour: u64,

    /// Current concurrent proofs.
    pub concurrent_proofs: usize,

    /// Per-solver breakdown.
    pub by_solver: HashMap<String, u64>,
}

impl Default for TenantId {
    fn default() -> Self {
        TenantId(String::new())
    }
}

impl TenantUsage {
    /// Success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_proofs == 0 {
            0.0
        } else {
            self.successful_proofs as f64 / self.total_proofs as f64
        }
    }

    /// Average proof generation time (ms).
    pub fn avg_generation_ms(&self) -> f64 {
        if self.total_proofs == 0 {
            0.0
        } else {
            self.total_compute_ms as f64 / self.total_proofs as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rate Limit Decision
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a rate limit check.
#[derive(Debug, Clone)]
pub enum RateLimitDecision {
    /// Request is allowed.
    Allowed,
    /// Request is denied: hourly limit exceeded.
    DeniedHourlyLimit { current: u64, limit: u64 },
    /// Request is denied: concurrent limit exceeded.
    DeniedConcurrentLimit { current: usize, limit: usize },
    /// Request is denied: proof size exceeds limit.
    DeniedProofSize { size: usize, limit: usize },
    /// Request is denied: grid_bits exceeds limit.
    DeniedGridBits { requested: usize, limit: usize },
    /// Request is denied: chi_max exceeds limit.
    DeniedChiMax { requested: usize, limit: usize },
    /// Request is denied: solver not allowed.
    DeniedSolverNotAllowed { solver: SolverType },
}

impl RateLimitDecision {
    /// Whether the request is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }

    /// Get human-readable denial reason.
    pub fn denial_reason(&self) -> Option<String> {
        match self {
            Self::Allowed => None,
            Self::DeniedHourlyLimit { current, limit } => {
                Some(format!("Hourly proof limit exceeded: {}/{}", current, limit))
            }
            Self::DeniedConcurrentLimit { current, limit } => {
                Some(format!(
                    "Concurrent proof limit exceeded: {}/{}",
                    current, limit
                ))
            }
            Self::DeniedProofSize { size, limit } => {
                Some(format!("Proof size {} exceeds limit {}", size, limit))
            }
            Self::DeniedGridBits { requested, limit } => {
                Some(format!(
                    "Grid bits {} exceeds limit {}",
                    requested, limit
                ))
            }
            Self::DeniedChiMax { requested, limit } => {
                Some(format!(
                    "Chi max {} exceeds limit {}",
                    requested, limit
                ))
            }
            Self::DeniedSolverNotAllowed { solver } => {
                Some(format!("Solver {} not allowed for this tier", solver))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Usage Meter
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks usage per tenant and enforces rate limits.
pub struct UsageMeter {
    /// Per-tenant usage data.
    usage: HashMap<TenantId, TenantUsage>,

    /// Per-tenant hourly proof timestamps (sliding window).
    hourly_window: HashMap<TenantId, VecDeque<u64>>,

    /// All usage records (for audit).
    records: Vec<UsageRecord>,
}

impl UsageMeter {
    /// Create a new usage meter.
    pub fn new() -> Self {
        Self {
            usage: HashMap::new(),
            hourly_window: HashMap::new(),
            records: Vec::new(),
        }
    }

    /// Check whether a proof request is allowed for a tenant.
    pub fn check_rate_limit(
        &mut self,
        tenant: &TenantConfig,
        solver: SolverType,
        grid_bits: usize,
        chi_max: usize,
    ) -> RateLimitDecision {
        let limits = tenant.effective_limits();
        let now = crate::gevulot::current_unix_time();

        // Check solver allowed
        if !tenant.effective_solvers().contains(&solver) {
            return RateLimitDecision::DeniedSolverNotAllowed { solver };
        }

        // Check grid_bits
        if grid_bits > limits.max_grid_bits {
            return RateLimitDecision::DeniedGridBits {
                requested: grid_bits,
                limit: limits.max_grid_bits,
            };
        }

        // Check chi_max
        if chi_max > limits.max_chi_max {
            return RateLimitDecision::DeniedChiMax {
                requested: chi_max,
                limit: limits.max_chi_max,
            };
        }

        // Check concurrent limit
        let usage = self
            .usage
            .entry(tenant.id.clone())
            .or_insert_with(|| TenantUsage {
                tenant_id: tenant.id.clone(),
                ..Default::default()
            });

        if usage.concurrent_proofs >= limits.max_concurrent_proofs {
            return RateLimitDecision::DeniedConcurrentLimit {
                current: usage.concurrent_proofs,
                limit: limits.max_concurrent_proofs,
            };
        }

        // Check hourly limit (sliding window)
        let window = self
            .hourly_window
            .entry(tenant.id.clone())
            .or_default();

        // Prune entries older than 1 hour
        let hour_ago = now.saturating_sub(3600);
        while window.front().map_or(false, |&t| t < hour_ago) {
            window.pop_front();
        }

        if window.len() as u64 >= limits.max_proofs_per_hour {
            return RateLimitDecision::DeniedHourlyLimit {
                current: window.len() as u64,
                limit: limits.max_proofs_per_hour,
            };
        }

        RateLimitDecision::Allowed
    }

    /// Record the start of a proof generation.
    pub fn start_proof(&mut self, tenant_id: &TenantId) {
        let usage = self
            .usage
            .entry(tenant_id.clone())
            .or_insert_with(|| TenantUsage {
                tenant_id: tenant_id.clone(),
                ..Default::default()
            });
        usage.concurrent_proofs += 1;
    }

    /// Record the completion of a proof generation.
    pub fn finish_proof(&mut self, record: UsageRecord) {
        let tenant_id = record.tenant_id.clone();
        let now = record.timestamp;

        let usage = self
            .usage
            .entry(tenant_id.clone())
            .or_insert_with(|| TenantUsage {
                tenant_id: tenant_id.clone(),
                ..Default::default()
            });

        if usage.concurrent_proofs > 0 {
            usage.concurrent_proofs -= 1;
        }
        usage.total_proofs += 1;
        if record.success {
            usage.successful_proofs += 1;
        } else {
            usage.failed_proofs += 1;
        }
        usage.total_bytes += record.proof_bytes as u64;
        usage.total_compute_ms += record.generation_time_ms;

        *usage
            .by_solver
            .entry(record.solver_type.to_string())
            .or_insert(0) += 1;

        // Add to hourly window
        self.hourly_window
            .entry(tenant_id)
            .or_default()
            .push_back(now);

        self.records.push(record);
    }

    /// Get usage for a specific tenant.
    pub fn get_usage(&self, tenant_id: &TenantId) -> Option<&TenantUsage> {
        self.usage.get(tenant_id)
    }

    /// Get all usage records.
    pub fn records(&self) -> &[UsageRecord] {
        &self.records
    }

    /// Get records for a specific tenant.
    pub fn records_for_tenant(
        &self,
        tenant_id: &TenantId,
    ) -> Vec<&UsageRecord> {
        self.records
            .iter()
            .filter(|r| r.tenant_id == *tenant_id)
            .collect()
    }

    /// Total proofs across all tenants.
    pub fn total_proofs(&self) -> u64 {
        self.usage.values().map(|u| u.total_proofs).sum()
    }

    /// Total compute time across all tenants (ms).
    pub fn total_compute_ms(&self) -> u64 {
        self.usage.values().map(|u| u.total_compute_ms).sum()
    }
}

impl Default for UsageMeter {
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
    use crate::multi_tenant::tenant::{ApiKey, TenantTier};

    fn make_tenant_config(id: &str, tier: TenantTier) -> TenantConfig {
        TenantConfig {
            id: TenantId(id.into()),
            name: format!("Tenant {}", id),
            tier,
            api_key: ApiKey(format!("test-key-{}-1234567890", id)),
            active: true,
            created_at: crate::gevulot::current_unix_time(),
            custom_limits: None,
            contact_email: format!("{}@example.com", id),
            allowed_solvers: None,
        }
    }

    fn make_record(
        tenant: &TenantConfig,
        solver: SolverType,
        success: bool,
    ) -> UsageRecord {
        UsageRecord {
            tenant_id: tenant.id.clone(),
            solver_type: solver,
            timestamp: crate::gevulot::current_unix_time(),
            generation_time_ms: 100,
            proof_bytes: 800,
            num_constraints: 5000,
            grid_bits: 4,
            chi_max: 4,
            success,
        }
    }

    #[test]
    fn test_meter_creation() {
        let meter = UsageMeter::new();
        assert_eq!(meter.total_proofs(), 0);
    }

    #[test]
    fn test_rate_limit_allowed() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Standard);

        let decision = meter.check_rate_limit(
            &tenant,
            SolverType::Euler3D,
            4,
            4,
        );
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_rate_limit_solver_not_allowed() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Free);

        let decision = meter.check_rate_limit(
            &tenant,
            SolverType::NsImex,
            4,
            4,
        );
        assert!(!decision.is_allowed());
        assert!(decision.denial_reason().unwrap().contains("not allowed"));
    }

    #[test]
    fn test_rate_limit_grid_bits() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Free);

        let decision = meter.check_rate_limit(
            &tenant,
            SolverType::Euler3D,
            20, // Free tier limit is 6
            4,
        );
        assert!(!decision.is_allowed());
    }

    #[test]
    fn test_rate_limit_chi_max() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Free);

        let decision = meter.check_rate_limit(
            &tenant,
            SolverType::Euler3D,
            4,
            64, // Free tier limit is 8
        );
        assert!(!decision.is_allowed());
    }

    #[test]
    fn test_rate_limit_concurrent() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Free); // max 1 concurrent

        meter.start_proof(&tenant.id);
        let decision = meter.check_rate_limit(
            &tenant,
            SolverType::Euler3D,
            4,
            4,
        );
        assert!(!decision.is_allowed());
        assert!(
            decision
                .denial_reason()
                .unwrap()
                .contains("Concurrent")
        );
    }

    #[test]
    fn test_finish_proof() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Standard);

        meter.start_proof(&tenant.id);
        meter.finish_proof(make_record(&tenant, SolverType::Euler3D, true));

        let usage = meter.get_usage(&tenant.id).unwrap();
        assert_eq!(usage.total_proofs, 1);
        assert_eq!(usage.successful_proofs, 1);
        assert_eq!(usage.concurrent_proofs, 0);
    }

    #[test]
    fn test_usage_stats() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Standard);

        for _ in 0..5 {
            meter.start_proof(&tenant.id);
            meter.finish_proof(make_record(&tenant, SolverType::Euler3D, true));
        }
        meter.start_proof(&tenant.id);
        meter.finish_proof(make_record(&tenant, SolverType::Euler3D, false));

        let usage = meter.get_usage(&tenant.id).unwrap();
        assert_eq!(usage.total_proofs, 6);
        assert_eq!(usage.successful_proofs, 5);
        assert_eq!(usage.failed_proofs, 1);
        assert!((usage.success_rate() - 5.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_records_for_tenant() {
        let mut meter = UsageMeter::new();
        let t1 = make_tenant_config("t1", TenantTier::Standard);
        let t2 = make_tenant_config("t2", TenantTier::Standard);

        meter.start_proof(&t1.id);
        meter.finish_proof(make_record(&t1, SolverType::Euler3D, true));
        meter.start_proof(&t2.id);
        meter.finish_proof(make_record(&t2, SolverType::NsImex, true));

        let t1_records = meter.records_for_tenant(&t1.id);
        assert_eq!(t1_records.len(), 1);
        assert_eq!(t1_records[0].solver_type, SolverType::Euler3D);
    }

    #[test]
    fn test_total_proofs() {
        let mut meter = UsageMeter::new();
        let t1 = make_tenant_config("t1", TenantTier::Standard);
        let t2 = make_tenant_config("t2", TenantTier::Standard);

        meter.start_proof(&t1.id);
        meter.finish_proof(make_record(&t1, SolverType::Euler3D, true));
        meter.start_proof(&t2.id);
        meter.finish_proof(make_record(&t2, SolverType::NsImex, true));

        assert_eq!(meter.total_proofs(), 2);
    }

    #[test]
    fn test_by_solver_tracking() {
        let mut meter = UsageMeter::new();
        let tenant = make_tenant_config("t1", TenantTier::Standard);

        meter.start_proof(&tenant.id);
        meter.finish_proof(make_record(&tenant, SolverType::Euler3D, true));
        meter.start_proof(&tenant.id);
        meter.finish_proof(make_record(&tenant, SolverType::NsImex, true));

        let usage = meter.get_usage(&tenant.id).unwrap();
        assert_eq!(usage.by_solver.get("euler3d"), Some(&1));
        assert_eq!(usage.by_solver.get("ns_imex"), Some(&1));
    }
}

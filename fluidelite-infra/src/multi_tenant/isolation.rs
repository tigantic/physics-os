//! Compute isolation for multi-tenant proving.
//!
//! Enforces per-tenant resource limits (CPU time, memory, proof count)
//! to prevent noisy-neighbor effects in shared infrastructure.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use fluidelite_core::physics_traits::SolverType;

use super::tenant::{TenantConfig, TenantId};

// ═══════════════════════════════════════════════════════════════════════════
// Isolation Guard
// ═══════════════════════════════════════════════════════════════════════════

/// A guard that tracks an active proof session.
///
/// When dropped, decrements the concurrent proof counter for the tenant.
pub struct IsolationGuard {
    tenant_id: TenantId,
    tracker: std::sync::Arc<IsolationTracker>,
}

impl Drop for IsolationGuard {
    fn drop(&mut self) {
        self.tracker.release(&self.tenant_id);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-Tenant Counters
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic counters for a single tenant.
struct TenantCounters {
    /// Active concurrent proofs.
    active_proofs: AtomicUsize,

    /// Total proofs completed.
    total_proofs: AtomicU64,

    /// Total compute time in milliseconds.
    total_compute_ms: AtomicU64,

    /// Total proof bytes produced.
    total_bytes: AtomicU64,
}

impl TenantCounters {
    fn new() -> Self {
        Self {
            active_proofs: AtomicUsize::new(0),
            total_proofs: AtomicU64::new(0),
            total_compute_ms: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
        }
    }
}

/// Snapshot of a tenant's isolation metrics.
#[derive(Debug, Clone)]
pub struct IsolationSnapshot {
    /// Tenant ID.
    pub tenant_id: TenantId,

    /// Active concurrent proofs.
    pub active_proofs: usize,

    /// Total proofs completed.
    pub total_proofs: u64,

    /// Total compute time (ms).
    pub total_compute_ms: u64,

    /// Total proof bytes.
    pub total_bytes: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Isolation Tracker
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks per-tenant compute isolation.
///
/// Uses atomic counters for lock-free concurrent access.
pub struct IsolationTracker {
    /// Per-tenant counters.
    counters: HashMap<TenantId, TenantCounters>,

    /// Global active proofs.
    global_active: AtomicUsize,

    /// Global total proofs.
    global_total: AtomicU64,
}

impl IsolationTracker {
    /// Create a new tracker with registered tenants.
    pub fn new(tenant_ids: &[TenantId]) -> Self {
        let mut counters = HashMap::new();
        for id in tenant_ids {
            counters.insert(id.clone(), TenantCounters::new());
        }
        Self {
            counters,
            global_active: AtomicUsize::new(0),
            global_total: AtomicU64::new(0),
        }
    }

    /// Register a new tenant dynamically.
    pub fn register_tenant(&mut self, id: TenantId) {
        self.counters
            .entry(id)
            .or_insert_with(TenantCounters::new);
    }

    /// Try to acquire a proof slot for a tenant.
    ///
    /// Returns `Ok(())` if the slot was acquired, `Err` if the limit is reached.
    pub fn try_acquire(
        &self,
        tenant_id: &TenantId,
        max_concurrent: usize,
    ) -> Result<(), String> {
        let counters = self
            .counters
            .get(tenant_id)
            .ok_or_else(|| format!("Tenant not registered: {}", tenant_id))?;

        let current = counters.active_proofs.load(Ordering::Acquire);
        if current >= max_concurrent {
            return Err(format!(
                "Concurrent limit reached for {}: {}/{}",
                tenant_id, current, max_concurrent
            ));
        }

        counters.active_proofs.fetch_add(1, Ordering::Release);
        self.global_active.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Release a proof slot for a tenant.
    pub fn release(&self, tenant_id: &TenantId) {
        if let Some(counters) = self.counters.get(tenant_id) {
            let prev = counters.active_proofs.fetch_sub(1, Ordering::Release);
            if prev == 0 {
                // Underflow protection — restore to 0
                counters.active_proofs.store(0, Ordering::Release);
            }
            let g_prev =
                self.global_active.fetch_sub(1, Ordering::Release);
            if g_prev == 0 {
                self.global_active.store(0, Ordering::Release);
            }
        }
    }

    /// Record a completed proof.
    pub fn record_completion(
        &self,
        tenant_id: &TenantId,
        compute_ms: u64,
        proof_bytes: u64,
    ) {
        if let Some(counters) = self.counters.get(tenant_id) {
            counters.total_proofs.fetch_add(1, Ordering::Relaxed);
            counters
                .total_compute_ms
                .fetch_add(compute_ms, Ordering::Relaxed);
            counters
                .total_bytes
                .fetch_add(proof_bytes, Ordering::Relaxed);
        }
        self.global_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot for a tenant.
    pub fn snapshot(&self, tenant_id: &TenantId) -> Option<IsolationSnapshot> {
        self.counters.get(tenant_id).map(|c| IsolationSnapshot {
            tenant_id: tenant_id.clone(),
            active_proofs: c.active_proofs.load(Ordering::Acquire),
            total_proofs: c.total_proofs.load(Ordering::Acquire),
            total_compute_ms: c.total_compute_ms.load(Ordering::Acquire),
            total_bytes: c.total_bytes.load(Ordering::Acquire),
        })
    }

    /// Get global active proof count.
    pub fn global_active(&self) -> usize {
        self.global_active.load(Ordering::Acquire)
    }

    /// Get global total proof count.
    pub fn global_total(&self) -> u64 {
        self.global_total.load(Ordering::Acquire)
    }

    /// Get snapshots for all tenants.
    pub fn all_snapshots(&self) -> Vec<IsolationSnapshot> {
        self.counters
            .keys()
            .filter_map(|id| self.snapshot(id))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compute Isolator
// ═══════════════════════════════════════════════════════════════════════════

/// High-level compute isolation manager.
///
/// Combines tenant configuration with isolation tracking to provide
/// a unified interface for enforcing resource limits.
pub struct ComputeIsolator {
    tracker: std::sync::Arc<IsolationTracker>,
}

impl ComputeIsolator {
    /// Create a new isolator.
    pub fn new(tenant_ids: &[TenantId]) -> Self {
        Self {
            tracker: std::sync::Arc::new(IsolationTracker::new(tenant_ids)),
        }
    }

    /// Acquire an isolation guard for a proof job.
    ///
    /// The guard automatically releases the slot when dropped.
    pub fn acquire_guard(
        &self,
        tenant: &TenantConfig,
    ) -> Result<IsolationGuard, String> {
        let limits = tenant.effective_limits();
        self.tracker
            .try_acquire(&tenant.id, limits.max_concurrent_proofs)?;

        Ok(IsolationGuard {
            tenant_id: tenant.id.clone(),
            tracker: self.tracker.clone(),
        })
    }

    /// Record completion.
    pub fn record_completion(
        &self,
        tenant_id: &TenantId,
        compute_ms: u64,
        proof_bytes: u64,
    ) {
        self.tracker
            .record_completion(tenant_id, compute_ms, proof_bytes);
    }

    /// Get snapshot.
    pub fn snapshot(
        &self,
        tenant_id: &TenantId,
    ) -> Option<IsolationSnapshot> {
        self.tracker.snapshot(tenant_id)
    }

    /// Global active count.
    pub fn global_active(&self) -> usize {
        self.tracker.global_active()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_tenant::tenant::{ApiKey, TenantTier};

    fn make_tenant_config(
        id: &str,
        tier: TenantTier,
    ) -> TenantConfig {
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

    #[test]
    fn test_tracker_creation() {
        let ids = vec![TenantId("t1".into()), TenantId("t2".into())];
        let tracker = IsolationTracker::new(&ids);
        assert_eq!(tracker.global_active(), 0);
        assert_eq!(tracker.global_total(), 0);
    }

    #[test]
    fn test_acquire_release() {
        let ids = vec![TenantId("t1".into())];
        let tracker = IsolationTracker::new(&ids);
        let t1 = TenantId("t1".into());

        tracker.try_acquire(&t1, 2).unwrap();
        assert_eq!(tracker.global_active(), 1);

        tracker.release(&t1);
        assert_eq!(tracker.global_active(), 0);
    }

    #[test]
    fn test_concurrent_limit() {
        let ids = vec![TenantId("t1".into())];
        let tracker = IsolationTracker::new(&ids);
        let t1 = TenantId("t1".into());

        tracker.try_acquire(&t1, 1).unwrap();
        assert!(tracker.try_acquire(&t1, 1).is_err());

        tracker.release(&t1);
        tracker.try_acquire(&t1, 1).unwrap();
    }

    #[test]
    fn test_record_completion() {
        let ids = vec![TenantId("t1".into())];
        let tracker = IsolationTracker::new(&ids);
        let t1 = TenantId("t1".into());

        tracker.record_completion(&t1, 100, 800);
        tracker.record_completion(&t1, 200, 1200);

        let snap = tracker.snapshot(&t1).unwrap();
        assert_eq!(snap.total_proofs, 2);
        assert_eq!(snap.total_compute_ms, 300);
        assert_eq!(snap.total_bytes, 2000);
    }

    #[test]
    fn test_unknown_tenant() {
        let tracker = IsolationTracker::new(&[]);
        let t1 = TenantId("unknown".into());
        assert!(tracker.try_acquire(&t1, 1).is_err());
    }

    #[test]
    fn test_snapshot() {
        let ids = vec![TenantId("t1".into())];
        let tracker = IsolationTracker::new(&ids);
        let t1 = TenantId("t1".into());

        tracker.try_acquire(&t1, 5).unwrap();
        tracker.record_completion(&t1, 150, 900);

        let snap = tracker.snapshot(&t1).unwrap();
        assert_eq!(snap.active_proofs, 1);
        assert_eq!(snap.total_proofs, 1);
    }

    #[test]
    fn test_all_snapshots() {
        let ids = vec![TenantId("t1".into()), TenantId("t2".into())];
        let tracker = IsolationTracker::new(&ids);

        let snapshots = tracker.all_snapshots();
        assert_eq!(snapshots.len(), 2);
    }

    #[test]
    fn test_compute_isolator_guard() {
        let t1_id = TenantId("t1".into());
        let isolator = ComputeIsolator::new(&[t1_id.clone()]);
        let tenant = make_tenant_config("t1", TenantTier::Free); // max 1 concurrent

        {
            let _guard = isolator.acquire_guard(&tenant).unwrap();
            assert_eq!(isolator.global_active(), 1);

            // Second acquire should fail (Free tier = 1 concurrent)
            assert!(isolator.acquire_guard(&tenant).is_err());
        }

        // Guard dropped → slot released
        assert_eq!(isolator.global_active(), 0);
        let _guard2 = isolator.acquire_guard(&tenant).unwrap();
    }

    #[test]
    fn test_isolator_record() {
        let t1_id = TenantId("t1".into());
        let isolator = ComputeIsolator::new(&[t1_id.clone()]);

        isolator.record_completion(&t1_id, 100, 800);
        let snap = isolator.snapshot(&t1_id).unwrap();
        assert_eq!(snap.total_proofs, 1);
    }

    #[test]
    fn test_register_dynamic() {
        let mut tracker = IsolationTracker::new(&[]);
        let t1 = TenantId("t1".into());

        assert!(tracker.try_acquire(&t1, 1).is_err());

        tracker.register_tenant(t1.clone());
        tracker.try_acquire(&t1, 1).unwrap();
    }
}

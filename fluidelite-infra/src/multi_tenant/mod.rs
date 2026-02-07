//! Multi-tenant operations module.
//!
//! Provides tenant isolation, metering, persistent storage, and compute
//! isolation for shared proof infrastructure.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Multi-Tenant Layer                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  TenantManager     → registration, auth, tier management        │
//! │  UsageMeter        → per-tenant metering and rate limiting      │
//! │  PersistentCertStore → WAL-backed certificate storage           │
//! │  ComputeIsolator   → RAII guards for concurrent proof limits    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  TenantConfig      → tier, limits, API key, solvers             │
//! │  UsageRecord       → per-proof resource consumption             │
//! │  IsolationGuard    → RAII compute slot management               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

pub mod tenant;
pub mod metering;
pub mod store;
pub mod isolation;

// Re-exports — tenant
pub use tenant::{
    ApiKey, ResourceLimits, TenantConfig, TenantId, TenantManager,
    TenantTier,
};

// Re-exports — metering
pub use metering::{
    RateLimitDecision, TenantUsage, UsageMeter, UsageRecord,
};

// Re-exports — store
pub use store::{PersistentCertStore, StoreConfig};

// Re-exports — isolation
pub use isolation::{
    ComputeIsolator, IsolationGuard, IsolationSnapshot, IsolationTracker,
};

// ═══════════════════════════════════════════════════════════════════════════
// Convenience Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Create a complete multi-tenant setup for testing.
pub fn test_setup() -> (TenantManager, UsageMeter, PersistentCertStore) {
    (
        TenantManager::new(),
        UsageMeter::new(),
        PersistentCertStore::memory(),
    )
}

/// Create a tenant config for testing.
pub fn test_tenant(id: &str, tier: TenantTier) -> TenantConfig {
    TenantConfig {
        id: TenantId(id.into()),
        name: format!("Test Tenant {}", id),
        tier,
        api_key: ApiKey(format!("test-key-{}-1234567890", id)),
        active: true,
        created_at: crate::gevulot::current_unix_time(),
        custom_limits: None,
        contact_email: format!("{}@test.com", id),
        allowed_solvers: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dashboard::models::{CertificateId, ProofCertificate};
    use fluidelite_core::physics_traits::SolverType;

    #[test]
    fn test_full_multi_tenant_pipeline() {
        // Setup
        let (mut mgr, mut meter, mut store) = test_setup();

        // Register tenants
        let t1 = test_tenant("t1", TenantTier::Standard);
        let t2 = test_tenant("t2", TenantTier::Professional);
        mgr.register(t1.clone()).unwrap();
        mgr.register(t2.clone()).unwrap();

        // Authenticate
        let auth_t1 = mgr.authenticate(&t1.api_key.0).unwrap();
        assert_eq!(auth_t1.id, TenantId("t1".into()));

        // Check rate limits
        let decision = meter.check_rate_limit(
            &t1,
            SolverType::Euler3D,
            4,
            4,
        );
        assert!(decision.is_allowed());

        // Simulate proof generation
        meter.start_proof(&t1.id);

        let record = UsageRecord {
            tenant_id: t1.id.clone(),
            solver_type: SolverType::Euler3D,
            timestamp: crate::gevulot::current_unix_time(),
            generation_time_ms: 150,
            proof_bytes: 800,
            num_constraints: 5000,
            grid_bits: 4,
            chi_max: 4,
            success: true,
        };
        meter.finish_proof(record);

        // Store certificate
        let mut cert = ProofCertificate::from_verification(
            CertificateId("cert-t1-001".into()),
            SolverType::Euler3D,
            true,
            4,
            4,
            5000,
            14,
            150,
            500,
            800,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            0.001,
        );
        cert.tenant_id = Some("t1".into());
        store.insert(cert).unwrap();

        // Verify results
        let usage = meter.get_usage(&t1.id).unwrap();
        assert_eq!(usage.total_proofs, 1);
        assert_eq!(usage.successful_proofs, 1);

        assert_eq!(store.total(), 1);
        let stored_cert = store.get("cert-t1-001").unwrap();
        assert_eq!(stored_cert.tenant_id.as_deref(), Some("t1"));
    }

    #[test]
    fn test_tier_enforcement() {
        let (mut mgr, mut meter, _) = test_setup();

        let free = test_tenant("free", TenantTier::Free);
        mgr.register(free.clone()).unwrap();

        // Free tier: NS-IMEX not allowed
        let decision = meter.check_rate_limit(
            &free,
            SolverType::NsImex,
            4,
            4,
        );
        assert!(!decision.is_allowed());

        // Free tier: Euler3D allowed
        let decision = meter.check_rate_limit(
            &free,
            SolverType::Euler3D,
            4,
            4,
        );
        assert!(decision.is_allowed());

        // Free tier: too many grid_bits
        let decision = meter.check_rate_limit(
            &free,
            SolverType::Euler3D,
            10, // limit is 6
            4,
        );
        assert!(!decision.is_allowed());
    }

    #[test]
    fn test_isolation_with_tenants() {
        let t1_id = TenantId("t1".into());
        let t2_id = TenantId("t2".into());

        let isolator =
            ComputeIsolator::new(&[t1_id.clone(), t2_id.clone()]);

        let t1 = test_tenant("t1", TenantTier::Free); // max 1 concurrent
        let t2 = test_tenant("t2", TenantTier::Standard); // max 4 concurrent

        // t1 can acquire 1
        let _guard1 = isolator.acquire_guard(&t1).unwrap();
        assert_eq!(isolator.global_active(), 1);

        // t1 can't acquire more
        assert!(isolator.acquire_guard(&t1).is_err());

        // t2 can still acquire
        let _guard2 = isolator.acquire_guard(&t2).unwrap();
        assert_eq!(isolator.global_active(), 2);
    }

    #[test]
    fn test_store_persistence() {
        let dir = std::env::temp_dir().join(format!(
            "fluidelite_mt_test_{}",
            crate::gevulot::current_unix_time()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        {
            let config = StoreConfig::persistent(&dir);
            let mut store = PersistentCertStore::new(config).unwrap();

            let mut cert = ProofCertificate::from_verification(
                CertificateId("mt-001".into()),
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
            cert.tenant_id = Some("t1".into());
            store.insert(cert).unwrap();
        }

        // Recover
        {
            let config = StoreConfig::persistent(&dir);
            let store = PersistentCertStore::new(config).unwrap();
            assert_eq!(store.total(), 1);
            let cert = store.get("mt-001").unwrap();
            assert_eq!(cert.tenant_id.as_deref(), Some("t1"));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}

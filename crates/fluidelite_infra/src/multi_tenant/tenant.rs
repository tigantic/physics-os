//! Multi-tenant configuration and management.
//!
//! Provides tenant isolation, resource quotas, and SLA enforcement
//! for shared proof infrastructure.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use fluidelite_core::physics_traits::SolverType;

// ═══════════════════════════════════════════════════════════════════════════
// Tenant Identity
// ═══════════════════════════════════════════════════════════════════════════

/// Unique tenant identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// API key for tenant authentication.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApiKey(pub String);

impl ApiKey {
    /// Validate API key format (must be non-empty, alphanumeric + hyphens).
    pub fn validate(&self) -> bool {
        !self.0.is_empty()
            && self.0.len() >= 16
            && self
                .0
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tenant Tier
// ═══════════════════════════════════════════════════════════════════════════

/// Service tier determining resource limits and SLA guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TenantTier {
    /// Free tier: limited proofs, best-effort SLA.
    Free,
    /// Standard tier: moderate limits, 99.5% SLA.
    Standard,
    /// Professional tier: high limits, 99.9% SLA.
    Professional,
    /// Enterprise tier: custom limits, 99.99% SLA.
    Enterprise,
}

impl TenantTier {
    /// Maximum concurrent proof jobs.
    pub fn max_concurrent_proofs(&self) -> usize {
        match self {
            Self::Free => 1,
            Self::Standard => 4,
            Self::Professional => 16,
            Self::Enterprise => 64,
        }
    }

    /// Maximum proofs per hour.
    pub fn max_proofs_per_hour(&self) -> u64 {
        match self {
            Self::Free => 10,
            Self::Standard => 100,
            Self::Professional => 1000,
            Self::Enterprise => 10_000,
        }
    }

    /// Maximum proof size in bytes.
    pub fn max_proof_bytes(&self) -> usize {
        match self {
            Self::Free => 1_024 * 1_024,       // 1 MB
            Self::Standard => 10 * 1_024 * 1_024,    // 10 MB
            Self::Professional => 100 * 1_024 * 1_024,   // 100 MB
            Self::Enterprise => 1_024 * 1_024 * 1_024,    // 1 GB
        }
    }

    /// Maximum bond dimension (chi_max).
    pub fn max_chi_max(&self) -> usize {
        match self {
            Self::Free => 8,
            Self::Standard => 32,
            Self::Professional => 128,
            Self::Enterprise => 512,
        }
    }

    /// Maximum grid bits.
    pub fn max_grid_bits(&self) -> usize {
        match self {
            Self::Free => 6,
            Self::Standard => 10,
            Self::Professional => 14,
            Self::Enterprise => 20,
        }
    }

    /// SLA availability target.
    pub fn sla_availability(&self) -> f64 {
        match self {
            Self::Free => 0.95,
            Self::Standard => 0.995,
            Self::Professional => 0.999,
            Self::Enterprise => 0.9999,
        }
    }

    /// Allowed solver types.
    pub fn allowed_solvers(&self) -> Vec<SolverType> {
        match self {
            Self::Free => vec![SolverType::Euler3D],
            _ => vec![SolverType::Euler3D, SolverType::NsImex],
        }
    }
}

impl fmt::Display for TenantTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => write!(f, "free"),
            Self::Standard => write!(f, "standard"),
            Self::Professional => write!(f, "professional"),
            Self::Enterprise => write!(f, "enterprise"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tenant Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Full tenant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Tenant ID.
    pub id: TenantId,

    /// Display name.
    pub name: String,

    /// Service tier.
    pub tier: TenantTier,

    /// API key for authentication.
    pub api_key: ApiKey,

    /// Whether the tenant is active.
    pub active: bool,

    /// Creation timestamp.
    pub created_at: u64,

    /// Custom resource limits (overrides tier defaults).
    pub custom_limits: Option<ResourceLimits>,

    /// Contact email.
    pub contact_email: String,

    /// Allowed solvers (overrides tier defaults if set).
    pub allowed_solvers: Option<Vec<SolverType>>,
}

impl TenantConfig {
    /// Get effective resource limits (custom or tier default).
    pub fn effective_limits(&self) -> ResourceLimits {
        self.custom_limits.clone().unwrap_or_else(|| ResourceLimits {
            max_concurrent_proofs: self.tier.max_concurrent_proofs(),
            max_proofs_per_hour: self.tier.max_proofs_per_hour(),
            max_proof_bytes: self.tier.max_proof_bytes(),
            max_chi_max: self.tier.max_chi_max(),
            max_grid_bits: self.tier.max_grid_bits(),
        })
    }

    /// Get allowed solver types.
    pub fn effective_solvers(&self) -> Vec<SolverType> {
        self.allowed_solvers
            .clone()
            .unwrap_or_else(|| self.tier.allowed_solvers())
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.0.is_empty() {
            return Err("Tenant ID must not be empty".into());
        }
        if self.name.is_empty() {
            return Err("Tenant name must not be empty".into());
        }
        if !self.api_key.validate() {
            return Err(
                "API key must be at least 16 characters, alphanumeric with hyphens"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Resource limits for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum concurrent proof jobs.
    pub max_concurrent_proofs: usize,

    /// Maximum proofs per hour.
    pub max_proofs_per_hour: u64,

    /// Maximum proof size in bytes.
    pub max_proof_bytes: usize,

    /// Maximum bond dimension.
    pub max_chi_max: usize,

    /// Maximum grid bits.
    pub max_grid_bits: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Tenant Manager
// ═══════════════════════════════════════════════════════════════════════════

/// Manages tenant registration, authentication, and configuration.
pub struct TenantManager {
    /// Tenants by ID.
    tenants: HashMap<TenantId, TenantConfig>,

    /// API key → Tenant ID mapping.
    api_key_index: HashMap<String, TenantId>,
}

impl TenantManager {
    /// Create a new tenant manager.
    pub fn new() -> Self {
        Self {
            tenants: HashMap::new(),
            api_key_index: HashMap::new(),
        }
    }

    /// Register a new tenant.
    pub fn register(&mut self, config: TenantConfig) -> Result<(), String> {
        config.validate()?;

        if self.tenants.contains_key(&config.id) {
            return Err(format!("Tenant already exists: {}", config.id));
        }

        if self.api_key_index.contains_key(&config.api_key.0) {
            return Err("API key already in use".into());
        }

        self.api_key_index
            .insert(config.api_key.0.clone(), config.id.clone());
        self.tenants.insert(config.id.clone(), config);
        Ok(())
    }

    /// Authenticate a request by API key.
    pub fn authenticate(&self, api_key: &str) -> Result<&TenantConfig, String> {
        let tenant_id = self
            .api_key_index
            .get(api_key)
            .ok_or_else(|| "Invalid API key".to_string())?;

        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| "Tenant not found".to_string())?;

        if !config.active {
            return Err(format!("Tenant {} is inactive", tenant_id));
        }

        Ok(config)
    }

    /// Get tenant configuration by ID.
    pub fn get(&self, id: &TenantId) -> Option<&TenantConfig> {
        self.tenants.get(id)
    }

    /// Deactivate a tenant.
    pub fn deactivate(&mut self, id: &TenantId) -> Result<(), String> {
        let config = self
            .tenants
            .get_mut(id)
            .ok_or_else(|| format!("Tenant not found: {}", id))?;
        config.active = false;
        Ok(())
    }

    /// Reactivate a tenant.
    pub fn reactivate(&mut self, id: &TenantId) -> Result<(), String> {
        let config = self
            .tenants
            .get_mut(id)
            .ok_or_else(|| format!("Tenant not found: {}", id))?;
        config.active = true;
        Ok(())
    }

    /// Update tenant tier.
    pub fn update_tier(
        &mut self,
        id: &TenantId,
        tier: TenantTier,
    ) -> Result<(), String> {
        let config = self
            .tenants
            .get_mut(id)
            .ok_or_else(|| format!("Tenant not found: {}", id))?;
        config.tier = tier;
        Ok(())
    }

    /// List all tenant IDs.
    pub fn list_tenants(&self) -> Vec<&TenantId> {
        self.tenants.keys().collect()
    }

    /// List active tenants.
    pub fn active_tenants(&self) -> Vec<&TenantConfig> {
        self.tenants.values().filter(|c| c.active).collect()
    }

    /// Total registered tenants.
    pub fn total_tenants(&self) -> usize {
        self.tenants.len()
    }

    /// Check if a solver is allowed for a tenant.
    pub fn is_solver_allowed(
        &self,
        tenant_id: &TenantId,
        solver: SolverType,
    ) -> Result<bool, String> {
        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| format!("Tenant not found: {}", tenant_id))?;
        Ok(config.effective_solvers().contains(&solver))
    }
}

impl Default for TenantManager {
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

    fn make_tenant(id: &str, tier: TenantTier) -> TenantConfig {
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
    fn test_api_key_validation() {
        assert!(!ApiKey("short".into()).validate());
        assert!(ApiKey("valid-key-12345678".into()).validate());
        assert!(!ApiKey("invalid key spaces".into()).validate());
    }

    #[test]
    fn test_tenant_tier_limits() {
        assert_eq!(TenantTier::Free.max_concurrent_proofs(), 1);
        assert_eq!(TenantTier::Enterprise.max_concurrent_proofs(), 64);
        assert_eq!(TenantTier::Free.max_proofs_per_hour(), 10);
        assert!(TenantTier::Enterprise.sla_availability() > 0.999);
    }

    #[test]
    fn test_tier_allowed_solvers() {
        let free = TenantTier::Free.allowed_solvers();
        assert_eq!(free.len(), 1);
        assert_eq!(free[0], SolverType::Euler3D);

        let pro = TenantTier::Professional.allowed_solvers();
        assert_eq!(pro.len(), 2);
    }

    #[test]
    fn test_tenant_config_validate() {
        let valid = make_tenant("t1", TenantTier::Standard);
        assert!(valid.validate().is_ok());

        let empty_id = TenantConfig {
            id: TenantId(String::new()),
            ..make_tenant("t1", TenantTier::Standard)
        };
        assert!(empty_id.validate().is_err());
    }

    #[test]
    fn test_effective_limits() {
        let tenant = make_tenant("t1", TenantTier::Standard);
        let limits = tenant.effective_limits();
        assert_eq!(limits.max_concurrent_proofs, 4);
        assert_eq!(limits.max_proofs_per_hour, 100);
    }

    #[test]
    fn test_custom_limits() {
        let mut tenant = make_tenant("t1", TenantTier::Free);
        tenant.custom_limits = Some(ResourceLimits {
            max_concurrent_proofs: 10,
            max_proofs_per_hour: 50,
            max_proof_bytes: 5_000_000,
            max_chi_max: 64,
            max_grid_bits: 12,
        });
        let limits = tenant.effective_limits();
        assert_eq!(limits.max_concurrent_proofs, 10);
    }

    #[test]
    fn test_manager_register() {
        let mut mgr = TenantManager::new();
        let tenant = make_tenant("t1", TenantTier::Standard);
        mgr.register(tenant).unwrap();
        assert_eq!(mgr.total_tenants(), 1);
    }

    #[test]
    fn test_manager_duplicate() {
        let mut mgr = TenantManager::new();
        mgr.register(make_tenant("t1", TenantTier::Standard))
            .unwrap();
        assert!(mgr.register(make_tenant("t1", TenantTier::Standard)).is_err());
    }

    #[test]
    fn test_manager_authenticate() {
        let mut mgr = TenantManager::new();
        let tenant = make_tenant("t1", TenantTier::Standard);
        let key = tenant.api_key.0.clone();
        mgr.register(tenant).unwrap();

        let config = mgr.authenticate(&key).unwrap();
        assert_eq!(config.id, TenantId("t1".into()));
    }

    #[test]
    fn test_manager_auth_invalid_key() {
        let mgr = TenantManager::new();
        assert!(mgr.authenticate("nonexistent").is_err());
    }

    #[test]
    fn test_manager_auth_inactive() {
        let mut mgr = TenantManager::new();
        let tenant = make_tenant("t1", TenantTier::Standard);
        let key = tenant.api_key.0.clone();
        mgr.register(tenant).unwrap();
        mgr.deactivate(&TenantId("t1".into())).unwrap();

        assert!(mgr.authenticate(&key).is_err());
    }

    #[test]
    fn test_manager_deactivate_reactivate() {
        let mut mgr = TenantManager::new();
        let tenant = make_tenant("t1", TenantTier::Standard);
        let key = tenant.api_key.0.clone();
        mgr.register(tenant).unwrap();

        mgr.deactivate(&TenantId("t1".into())).unwrap();
        assert!(mgr.authenticate(&key).is_err());

        mgr.reactivate(&TenantId("t1".into())).unwrap();
        assert!(mgr.authenticate(&key).is_ok());
    }

    #[test]
    fn test_manager_update_tier() {
        let mut mgr = TenantManager::new();
        mgr.register(make_tenant("t1", TenantTier::Free)).unwrap();

        mgr.update_tier(&TenantId("t1".into()), TenantTier::Enterprise)
            .unwrap();

        let config = mgr.get(&TenantId("t1".into())).unwrap();
        assert_eq!(config.tier, TenantTier::Enterprise);
    }

    #[test]
    fn test_manager_active_tenants() {
        let mut mgr = TenantManager::new();
        mgr.register(make_tenant("t1", TenantTier::Standard))
            .unwrap();
        mgr.register(make_tenant("t2", TenantTier::Professional))
            .unwrap();
        mgr.deactivate(&TenantId("t2".into())).unwrap();

        let active = mgr.active_tenants();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_solver_allowed() {
        let mut mgr = TenantManager::new();
        mgr.register(make_tenant("t1", TenantTier::Free)).unwrap();
        mgr.register(make_tenant("t2", TenantTier::Standard))
            .unwrap();

        let t1 = TenantId("t1".into());
        assert!(mgr.is_solver_allowed(&t1, SolverType::Euler3D).unwrap());
        assert!(!mgr.is_solver_allowed(&t1, SolverType::NsImex).unwrap());

        let t2 = TenantId("t2".into());
        assert!(mgr.is_solver_allowed(&t2, SolverType::NsImex).unwrap());
    }
}

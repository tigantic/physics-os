//! On-chain proof registry with audit trail.
//!
//! Maintains a persistent, hash-indexed registry of all verified proofs
//! for public auditability. Supports querying by solver type, time range,
//! and proof metadata.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use fluidelite_core::physics_traits::SolverType;

use super::types::{
    current_unix_time, GevulotSubmission, SubmissionId, SubmissionStatus,
    VerificationRecord,
};

// ═══════════════════════════════════════════════════════════════════════════
// Registry Entry
// ═══════════════════════════════════════════════════════════════════════════

/// A proof entry in the registry, combining submission and verification data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Submission ID.
    pub submission_id: SubmissionId,

    /// Solver type.
    pub solver_type: SolverType,

    /// Proof metadata hash (deterministic identifier).
    pub proof_metadata_hash: String,

    /// Input state hash limbs.
    pub input_hash_limbs: [u64; 4],

    /// Output state hash limbs.
    pub output_hash_limbs: [u64; 4],

    /// Parameters hash limbs.
    pub params_hash_limbs: [u64; 4],

    /// Number of constraints.
    pub num_constraints: usize,

    /// Circuit k parameter.
    pub k: u32,

    /// Grid resolution.
    pub grid_bits: usize,

    /// Bond dimension.
    pub chi_max: usize,

    /// Size of proof in bytes.
    pub proof_size: usize,

    /// Submission timestamp (Unix seconds).
    pub submitted_at: u64,

    /// Verification timestamp (Unix seconds).
    pub verified_at: u64,

    /// Number of Gevulot verifiers that confirmed.
    pub verifier_count: u32,

    /// Whether the proof was verified as valid.
    pub valid: bool,

    /// Lean formal proof files.
    pub lean_proofs: Vec<String>,

    /// On-chain transaction hash.
    pub tx_hash: String,

    /// Block number on Gevulot.
    pub block_number: u64,

    /// Sequential registry index.
    pub registry_index: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Registry Query
// ═══════════════════════════════════════════════════════════════════════════

/// Query filter for registry searches.
#[derive(Debug, Clone, Default)]
pub struct RegistryQuery {
    /// Filter by solver type.
    pub solver_type: Option<SolverType>,

    /// Filter by minimum timestamp.
    pub after: Option<u64>,

    /// Filter by maximum timestamp.
    pub before: Option<u64>,

    /// Filter by minimum grid_bits.
    pub min_grid_bits: Option<usize>,

    /// Filter by minimum chi_max.
    pub min_chi_max: Option<usize>,

    /// Filter by validity.
    pub valid_only: bool,

    /// Maximum number of results.
    pub limit: Option<usize>,

    /// Offset for pagination.
    pub offset: usize,
}

impl RegistryQuery {
    /// Create a query that matches everything.
    pub fn all() -> Self {
        Self::default()
    }

    /// Create a query filtered by solver type.
    pub fn for_solver(solver: SolverType) -> Self {
        Self {
            solver_type: Some(solver),
            ..Default::default()
        }
    }

    /// Create a query for valid proofs only.
    pub fn valid() -> Self {
        Self {
            valid_only: true,
            ..Default::default()
        }
    }

    /// Set the time window.
    pub fn time_range(mut self, after: u64, before: u64) -> Self {
        self.after = Some(after);
        self.before = Some(before);
        self
    }

    /// Set result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set pagination offset.
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Check if an entry matches this query.
    fn matches(&self, entry: &RegistryEntry) -> bool {
        if let Some(solver) = self.solver_type {
            if entry.solver_type != solver {
                return false;
            }
        }

        if let Some(after) = self.after {
            if entry.verified_at < after {
                return false;
            }
        }

        if let Some(before) = self.before {
            if entry.verified_at > before {
                return false;
            }
        }

        if let Some(min_gb) = self.min_grid_bits {
            if entry.grid_bits < min_gb {
                return false;
            }
        }

        if let Some(min_chi) = self.min_chi_max {
            if entry.chi_max < min_chi {
                return false;
            }
        }

        if self.valid_only && !entry.valid {
            return false;
        }

        true
    }
}

/// Summary statistics from a registry query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistrySummary {
    /// Total matching entries.
    pub total_entries: usize,

    /// Valid entries.
    pub valid_entries: usize,

    /// Invalid entries.
    pub invalid_entries: usize,

    /// Euler 3D entries.
    pub euler3d_count: usize,

    /// NS-IMEX entries.
    pub ns_imex_count: usize,

    /// Thermal entries.
    pub thermal_count: usize,

    /// Average verifier count.
    pub avg_verifier_count: f64,

    /// Total proof bytes.
    pub total_proof_bytes: u64,

    /// Earliest entry timestamp.
    pub earliest_timestamp: Option<u64>,

    /// Latest entry timestamp.
    pub latest_timestamp: Option<u64>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Proof Registry
// ═══════════════════════════════════════════════════════════════════════════

/// Persistent registry of verified proofs for public auditability.
///
/// Maintains hash-indexed entries with support for querying by solver type,
/// time range, and proof metadata. The registry provides an immutable audit
/// trail of all proofs that have passed decentralized verification.
///
/// # Architecture
///
/// ```text
/// ┌──────────────────────────────────────────────────┐
/// │  ProofRegistry                                    │
/// │  ├── by_id: HashMap<SubmissionId, RegistryEntry>  │
/// │  ├── by_hash: HashMap<String, SubmissionId>       │
/// │  ├── by_solver: HashMap<SolverType, Vec<...>>     │
/// │  └── chronological: Vec<SubmissionId>              │
/// └──────────────────────────────────────────────────┘
/// ```
pub struct ProofRegistry {
    /// Primary index: submission ID → entry.
    by_id: HashMap<SubmissionId, RegistryEntry>,

    /// Hash index: proof_metadata_hash → submission ID.
    by_hash: HashMap<String, SubmissionId>,

    /// Solver index: solver type → submission IDs (chronological order).
    by_solver: HashMap<SolverType, Vec<SubmissionId>>,

    /// Chronological order of all entries.
    chronological: Vec<SubmissionId>,

    /// Next registry index.
    next_index: u64,
}

impl ProofRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        let mut by_solver = HashMap::new();
        by_solver.insert(SolverType::Euler3D, Vec::new());
        by_solver.insert(SolverType::NsImex, Vec::new());

        Self {
            by_id: HashMap::new(),
            by_hash: HashMap::new(),
            by_solver,
            chronological: Vec::new(),
            next_index: 0,
        }
    }

    /// Register a verified proof.
    pub fn register(
        &mut self,
        submission: &GevulotSubmission,
        record: &VerificationRecord,
    ) -> Result<u64, String> {
        if submission.status != SubmissionStatus::Verified
            && submission.status != SubmissionStatus::Failed
        {
            return Err(format!(
                "Cannot register proof in state: {}",
                submission.status
            ));
        }

        if self.by_id.contains_key(&submission.id) {
            return Err(format!(
                "Proof already registered: {}",
                submission.id
            ));
        }

        let index = self.next_index;
        self.next_index += 1;

        let entry = RegistryEntry {
            submission_id: submission.id.clone(),
            solver_type: submission.solver_type,
            proof_metadata_hash: record.proof_metadata_hash.clone(),
            input_hash_limbs: submission.input_hash_limbs,
            output_hash_limbs: submission.output_hash_limbs,
            params_hash_limbs: submission.params_hash_limbs,
            num_constraints: submission.num_constraints,
            k: submission.k,
            grid_bits: submission.grid_bits,
            chi_max: submission.chi_max,
            proof_size: submission.proof_size,
            submitted_at: submission.submitted_at,
            verified_at: record.timestamp,
            verifier_count: record.verifier_count,
            valid: record.valid,
            lean_proofs: record.lean_proofs.clone(),
            tx_hash: record.tx_hash.clone(),
            block_number: record.block_number,
            registry_index: index,
        };

        // Index by proof metadata hash
        self.by_hash
            .insert(record.proof_metadata_hash.clone(), submission.id.clone());

        // Index by solver
        self.by_solver
            .entry(submission.solver_type)
            .or_default()
            .push(submission.id.clone());

        // Chronological
        self.chronological.push(submission.id.clone());

        // Primary index
        self.by_id.insert(submission.id.clone(), entry);

        Ok(index)
    }

    /// Look up a registry entry by submission ID.
    pub fn get(&self, id: &SubmissionId) -> Option<&RegistryEntry> {
        self.by_id.get(id)
    }

    /// Look up by proof metadata hash.
    pub fn get_by_hash(&self, hash: &str) -> Option<&RegistryEntry> {
        self.by_hash
            .get(hash)
            .and_then(|id| self.by_id.get(id))
    }

    /// Query the registry with filters.
    pub fn query(&self, q: &RegistryQuery) -> Vec<&RegistryEntry> {
        let iter = self
            .chronological
            .iter()
            .filter_map(|id| self.by_id.get(id))
            .filter(|e| q.matches(e));

        if let Some(limit) = q.limit {
            iter.skip(q.offset).take(limit).collect()
        } else {
            iter.skip(q.offset).collect()
        }
    }

    /// Get summary statistics for entries matching a query.
    pub fn summary(&self, q: &RegistryQuery) -> RegistrySummary {
        let entries = self.query(q);
        let mut summary = RegistrySummary {
            total_entries: entries.len(),
            ..Default::default()
        };

        let mut total_verifiers: u64 = 0;

        for entry in &entries {
            if entry.valid {
                summary.valid_entries += 1;
            } else {
                summary.invalid_entries += 1;
            }

            match entry.solver_type {
                SolverType::Euler3D => summary.euler3d_count += 1,
                SolverType::NsImex => summary.ns_imex_count += 1,
                SolverType::Thermal => summary.thermal_count += 1,
            }

            total_verifiers += entry.verifier_count as u64;
            summary.total_proof_bytes += entry.proof_size as u64;

            match summary.earliest_timestamp {
                None => summary.earliest_timestamp = Some(entry.verified_at),
                Some(t) if entry.verified_at < t => {
                    summary.earliest_timestamp = Some(entry.verified_at)
                }
                _ => {}
            }

            match summary.latest_timestamp {
                None => summary.latest_timestamp = Some(entry.verified_at),
                Some(t) if entry.verified_at > t => {
                    summary.latest_timestamp = Some(entry.verified_at)
                }
                _ => {}
            }
        }

        if !entries.is_empty() {
            summary.avg_verifier_count =
                total_verifiers as f64 / entries.len() as f64;
        }

        summary
    }

    /// Total number of entries.
    pub fn total_entries(&self) -> usize {
        self.by_id.len()
    }

    /// Number of entries for a specific solver.
    pub fn count_by_solver(&self, solver: SolverType) -> usize {
        self.by_solver
            .get(&solver)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get the latest N entries.
    pub fn latest(&self, n: usize) -> Vec<&RegistryEntry> {
        self.chronological
            .iter()
            .rev()
            .take(n)
            .filter_map(|id| self.by_id.get(id))
            .collect()
    }

    /// Export all entries as JSON.
    pub fn export_json(&self) -> Result<String, String> {
        let entries: Vec<&RegistryEntry> =
            self.chronological
                .iter()
                .filter_map(|id| self.by_id.get(id))
                .collect();
        serde_json::to_string_pretty(&entries)
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Check if a proof metadata hash is registered.
    pub fn contains_hash(&self, hash: &str) -> bool {
        self.by_hash.contains_key(hash)
    }

    /// Check if a submission ID is registered.
    pub fn contains(&self, id: &SubmissionId) -> bool {
        self.by_id.contains_key(id)
    }
}

impl Default for ProofRegistry {
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
    use crate::gevulot::types::compute_proof_metadata_hash;

    fn make_submission(
        solver: SolverType,
        input_hash: [u64; 4],
    ) -> GevulotSubmission {
        let id = SubmissionId::from_proof_hash(solver, &input_hash);
        GevulotSubmission {
            id,
            solver_type: solver,
            status: SubmissionStatus::Verified,
            proof_bytes: vec![1, 2, 3, 4],
            proof_size: 4,
            input_hash_limbs: input_hash,
            output_hash_limbs: [50, 60, 70, 80],
            params_hash_limbs: [90, 100, 110, 120],
            num_constraints: 5000,
            k: 14,
            grid_bits: 4,
            chi_max: 4,
            submitted_at: current_unix_time(),
            verified_at: Some(current_unix_time()),
            verification_time_ms: Some(100),
            tx_hash: Some("0xabc".into()),
            error: None,
            verifier_count: 3,
            lean_proofs: solver
                .lean_proofs()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn make_record(
        submission: &GevulotSubmission,
        valid: bool,
    ) -> VerificationRecord {
        let hash = compute_proof_metadata_hash(
            submission.solver_type,
            &submission.input_hash_limbs,
            &submission.output_hash_limbs,
            &submission.params_hash_limbs,
            submission.num_constraints,
        );
        VerificationRecord {
            submission_id: submission.id.clone(),
            solver_type: submission.solver_type,
            valid,
            verifier_count: 3,
            consensus_fraction: 1.0,
            timestamp: current_unix_time(),
            tx_hash: "0xabc".into(),
            block_number: 42,
            proof_metadata_hash: hash,
            grid_bits: submission.grid_bits,
            chi_max: submission.chi_max,
            num_constraints: submission.num_constraints,
            lean_proofs: submission.lean_proofs.clone(),
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = ProofRegistry::new();
        assert_eq!(registry.total_entries(), 0);
    }

    #[test]
    fn test_register_entry() {
        let mut registry = ProofRegistry::new();
        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);

        let index = registry.register(&sub, &rec).unwrap();
        assert_eq!(index, 0);
        assert_eq!(registry.total_entries(), 1);
    }

    #[test]
    fn test_register_duplicate() {
        let mut registry = ProofRegistry::new();
        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);

        registry.register(&sub, &rec).unwrap();
        assert!(registry.register(&sub, &rec).is_err());
    }

    #[test]
    fn test_register_non_terminal() {
        let mut registry = ProofRegistry::new();
        let mut sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        sub.status = SubmissionStatus::Pending;
        let rec = make_record(&sub, true);

        assert!(registry.register(&sub, &rec).is_err());
    }

    #[test]
    fn test_get_by_id() {
        let mut registry = ProofRegistry::new();
        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);
        registry.register(&sub, &rec).unwrap();

        let entry = registry.get(&sub.id).unwrap();
        assert_eq!(entry.solver_type, SolverType::Euler3D);
        assert!(entry.valid);
    }

    #[test]
    fn test_get_by_hash() {
        let mut registry = ProofRegistry::new();
        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);
        let hash = rec.proof_metadata_hash.clone();
        registry.register(&sub, &rec).unwrap();

        let entry = registry.get_by_hash(&hash).unwrap();
        assert_eq!(entry.submission_id, sub.id);
    }

    #[test]
    fn test_query_all() {
        let mut registry = ProofRegistry::new();

        let sub1 = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec1 = make_record(&sub1, true);
        registry.register(&sub1, &rec1).unwrap();

        let sub2 = make_submission(SolverType::NsImex, [5, 6, 7, 8]);
        let rec2 = make_record(&sub2, true);
        registry.register(&sub2, &rec2).unwrap();

        let results = registry.query(&RegistryQuery::all());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_solver() {
        let mut registry = ProofRegistry::new();

        let sub1 = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec1 = make_record(&sub1, true);
        registry.register(&sub1, &rec1).unwrap();

        let sub2 = make_submission(SolverType::NsImex, [5, 6, 7, 8]);
        let rec2 = make_record(&sub2, true);
        registry.register(&sub2, &rec2).unwrap();

        let euler = registry.query(&RegistryQuery::for_solver(SolverType::Euler3D));
        assert_eq!(euler.len(), 1);
        assert_eq!(euler[0].solver_type, SolverType::Euler3D);
    }

    #[test]
    fn test_query_valid_only() {
        let mut registry = ProofRegistry::new();

        let sub1 = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec1 = make_record(&sub1, true);
        registry.register(&sub1, &rec1).unwrap();

        let mut sub2 = make_submission(SolverType::NsImex, [5, 6, 7, 8]);
        sub2.status = SubmissionStatus::Failed;
        let rec2 = make_record(&sub2, false);
        registry.register(&sub2, &rec2).unwrap();

        let valid = registry.query(&RegistryQuery::valid());
        assert_eq!(valid.len(), 1);
    }

    #[test]
    fn test_query_with_limit() {
        let mut registry = ProofRegistry::new();

        for i in 0..5u64 {
            let sub =
                make_submission(SolverType::Euler3D, [i, i + 1, i + 2, i + 3]);
            let rec = make_record(&sub, true);
            registry.register(&sub, &rec).unwrap();
        }

        let limited = registry.query(&RegistryQuery::all().with_limit(3));
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_query_with_offset() {
        let mut registry = ProofRegistry::new();

        for i in 0..5u64 {
            let sub =
                make_submission(SolverType::Euler3D, [i, i + 1, i + 2, i + 3]);
            let rec = make_record(&sub, true);
            registry.register(&sub, &rec).unwrap();
        }

        let offset = registry.query(&RegistryQuery::all().with_offset(3));
        assert_eq!(offset.len(), 2);
    }

    #[test]
    fn test_summary() {
        let mut registry = ProofRegistry::new();

        let sub1 = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec1 = make_record(&sub1, true);
        registry.register(&sub1, &rec1).unwrap();

        let sub2 = make_submission(SolverType::NsImex, [5, 6, 7, 8]);
        let rec2 = make_record(&sub2, true);
        registry.register(&sub2, &rec2).unwrap();

        let summary = registry.summary(&RegistryQuery::all());
        assert_eq!(summary.total_entries, 2);
        assert_eq!(summary.valid_entries, 2);
        assert_eq!(summary.euler3d_count, 1);
        assert_eq!(summary.ns_imex_count, 1);
        assert_eq!(summary.avg_verifier_count, 3.0);
    }

    #[test]
    fn test_count_by_solver() {
        let mut registry = ProofRegistry::new();

        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);
        registry.register(&sub, &rec).unwrap();

        assert_eq!(registry.count_by_solver(SolverType::Euler3D), 1);
        assert_eq!(registry.count_by_solver(SolverType::NsImex), 0);
    }

    #[test]
    fn test_latest() {
        let mut registry = ProofRegistry::new();

        for i in 0..5u64 {
            let sub =
                make_submission(SolverType::Euler3D, [i, i + 1, i + 2, i + 3]);
            let rec = make_record(&sub, true);
            registry.register(&sub, &rec).unwrap();
        }

        let latest = registry.latest(2);
        assert_eq!(latest.len(), 2);
        // Latest should be in reverse chronological order
        assert!(
            latest[0].registry_index > latest[1].registry_index
        );
    }

    #[test]
    fn test_export_json() {
        let mut registry = ProofRegistry::new();
        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);
        registry.register(&sub, &rec).unwrap();

        let json = registry.export_json().unwrap();
        assert!(json.contains("euler3d"));
        assert!(json.contains("registry_index"));
    }

    #[test]
    fn test_contains() {
        let mut registry = ProofRegistry::new();
        let sub = make_submission(SolverType::Euler3D, [1, 2, 3, 4]);
        let rec = make_record(&sub, true);
        let hash = rec.proof_metadata_hash.clone();
        registry.register(&sub, &rec).unwrap();

        assert!(registry.contains(&sub.id));
        assert!(registry.contains_hash(&hash));
        assert!(!registry.contains(&SubmissionId("fake".into())));
        assert!(!registry.contains_hash("fake"));
    }
}

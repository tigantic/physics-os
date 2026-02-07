//! Gevulot network client.
//!
//! Manages the lifecycle of proof submissions to the Gevulot decentralized
//! verification network: submit → poll → retrieve verification records.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use fluidelite_core::physics_traits::{PhysicsProof, SolverType};

use super::types::{
    compute_proof_metadata_hash, current_unix_time, GevulotConfig, GevulotStats,
    GevulotSubmission, SubmissionId, SubmissionStatus, VerificationRecord,
};

// ═══════════════════════════════════════════════════════════════════════════
// Client
// ═══════════════════════════════════════════════════════════════════════════

/// Client for interacting with the Gevulot decentralized verification network.
///
/// Handles proof submission, status polling, and verification record retrieval.
/// The client maintains an in-memory ledger of all submissions and their
/// current states for local tracking.
///
/// # Architecture
///
/// ```text
/// ┌──────────────────────────────────────────────────┐
/// │  GevulotClient                                    │
/// │  ├── config: GevulotConfig                        │
/// │  ├── submissions: HashMap<SubmissionId, ...>      │
/// │  ├── records: Vec<VerificationRecord>             │
/// │  └── stats: GevulotStats                          │
/// └──────────────────────────────────────────────────┘
/// ```
pub struct GevulotClient {
    config: GevulotConfig,
    submissions: HashMap<SubmissionId, GevulotSubmission>,
    records: Vec<VerificationRecord>,
    stats: GevulotStats,
    sequence: u64,
}

impl GevulotClient {
    /// Create a new Gevulot client with the given configuration.
    pub fn new(config: GevulotConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            submissions: HashMap::new(),
            records: Vec::new(),
            stats: GevulotStats::default(),
            sequence: 0,
        })
    }

    /// Create a client for local development/testing (no network).
    pub fn local() -> Self {
        Self {
            config: GevulotConfig::local(),
            submissions: HashMap::new(),
            records: Vec::new(),
            stats: GevulotStats::default(),
            sequence: 0,
        }
    }

    /// Submit a physics proof for decentralized verification.
    ///
    /// Returns the submission ID for status tracking.
    pub fn submit_proof<P: PhysicsProof>(
        &mut self,
        proof: &P,
    ) -> Result<SubmissionId, String> {
        let serialized = proof.to_serialized_bytes();
        if serialized.is_empty() {
            return Err("Cannot submit empty proof".into());
        }

        let id = SubmissionId::from_proof_hash(
            proof.solver_type(),
            proof.input_hash_limbs(),
        );

        // Check for duplicate submission
        if self.submissions.contains_key(&id) {
            return Err(format!("Duplicate submission: {}", id));
        }

        let now = current_unix_time();
        let solver = proof.solver_type();

        let submission = GevulotSubmission {
            id: id.clone(),
            solver_type: solver,
            status: SubmissionStatus::Submitted,
            proof_bytes: serialized.clone(),
            proof_size: serialized.len(),
            input_hash_limbs: *proof.input_hash_limbs(),
            output_hash_limbs: *proof.output_hash_limbs(),
            params_hash_limbs: *proof.params_hash_limbs(),
            num_constraints: proof.num_constraints(),
            k: proof.k(),
            grid_bits: proof.grid_bits(),
            chi_max: proof.chi_max(),
            submitted_at: now,
            verified_at: None,
            verification_time_ms: None,
            tx_hash: Some(format!("0x{:016x}{:016x}", now, self.sequence)),
            error: None,
            verifier_count: 0,
            lean_proofs: solver.lean_proofs().iter().map(|s| s.to_string()).collect(),
        };

        self.submissions.insert(id.clone(), submission);
        self.stats.total_submissions += 1;
        self.stats.total_bytes_submitted += serialized.len() as u64;
        self.stats.pending += 1;
        self.sequence += 1;

        Ok(id)
    }

    /// Submit raw proof bytes for a given solver type.
    pub fn submit_raw(
        &mut self,
        solver_type: SolverType,
        proof_bytes: Vec<u8>,
        input_hash: [u64; 4],
        output_hash: [u64; 4],
        params_hash: [u64; 4],
        num_constraints: usize,
        k: u32,
        grid_bits: usize,
        chi_max: usize,
    ) -> Result<SubmissionId, String> {
        if proof_bytes.is_empty() {
            return Err("Cannot submit empty proof bytes".into());
        }

        let id = SubmissionId::from_proof_hash(solver_type, &input_hash);

        if self.submissions.contains_key(&id) {
            return Err(format!("Duplicate submission: {}", id));
        }

        let now = current_unix_time();
        let proof_size = proof_bytes.len();

        let submission = GevulotSubmission {
            id: id.clone(),
            solver_type,
            status: SubmissionStatus::Submitted,
            proof_bytes,
            proof_size,
            input_hash_limbs: input_hash,
            output_hash_limbs: output_hash,
            params_hash_limbs: params_hash,
            num_constraints,
            k,
            grid_bits,
            chi_max,
            submitted_at: now,
            verified_at: None,
            verification_time_ms: None,
            tx_hash: Some(format!("0x{:016x}{:016x}", now, self.sequence)),
            error: None,
            verifier_count: 0,
            lean_proofs: solver_type
                .lean_proofs()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };

        self.submissions.insert(id.clone(), submission);
        self.stats.total_submissions += 1;
        self.stats.total_bytes_submitted += proof_size as u64;
        self.stats.pending += 1;
        self.sequence += 1;

        Ok(id)
    }

    /// Check the status of a submission.
    pub fn check_status(
        &self,
        id: &SubmissionId,
    ) -> Result<SubmissionStatus, String> {
        self.submissions
            .get(id)
            .map(|s| s.status)
            .ok_or_else(|| format!("Submission not found: {}", id))
    }

    /// Get full submission details.
    pub fn get_submission(
        &self,
        id: &SubmissionId,
    ) -> Result<&GevulotSubmission, String> {
        self.submissions
            .get(id)
            .ok_or_else(|| format!("Submission not found: {}", id))
    }

    /// Mark a submission as verified (used in local/test mode).
    ///
    /// In production, this would be triggered by network callbacks.
    pub fn mark_verified(
        &mut self,
        id: &SubmissionId,
        verifier_count: u32,
        verification_time_ms: u64,
    ) -> Result<VerificationRecord, String> {
        let submission = self
            .submissions
            .get_mut(id)
            .ok_or_else(|| format!("Submission not found: {}", id))?;

        if submission.status.is_terminal() {
            return Err(format!(
                "Submission {} is already in terminal state: {}",
                id, submission.status
            ));
        }

        let now = current_unix_time();
        submission.status = SubmissionStatus::Verified;
        submission.verified_at = Some(now);
        submission.verification_time_ms = Some(verification_time_ms);
        submission.verifier_count = verifier_count;

        // Update stats
        self.stats.verified += 1;
        if self.stats.pending > 0 {
            self.stats.pending -= 1;
        }

        // Compute rolling average verification time
        let n = self.stats.verified as f64;
        self.stats.avg_verification_time_ms = self.stats.avg_verification_time_ms
            * ((n - 1.0) / n)
            + verification_time_ms as f64 / n;

        self.stats.avg_verifier_count = self.stats.avg_verifier_count
            * ((n - 1.0) / n)
            + verifier_count as f64 / n;

        let proof_hash = compute_proof_metadata_hash(
            submission.solver_type,
            &submission.input_hash_limbs,
            &submission.output_hash_limbs,
            &submission.params_hash_limbs,
            submission.num_constraints,
        );

        let record = VerificationRecord {
            submission_id: id.clone(),
            solver_type: submission.solver_type,
            valid: true,
            verifier_count,
            consensus_fraction: 1.0,
            timestamp: now,
            tx_hash: submission
                .tx_hash
                .clone()
                .unwrap_or_else(|| String::new()),
            block_number: self.sequence,
            proof_metadata_hash: proof_hash,
            grid_bits: submission.grid_bits,
            chi_max: submission.chi_max,
            num_constraints: submission.num_constraints,
            lean_proofs: submission.lean_proofs.clone(),
        };

        self.records.push(record.clone());
        Ok(record)
    }

    /// Mark a submission as failed (used in local/test mode).
    pub fn mark_failed(
        &mut self,
        id: &SubmissionId,
        error: String,
    ) -> Result<(), String> {
        let submission = self
            .submissions
            .get_mut(id)
            .ok_or_else(|| format!("Submission not found: {}", id))?;

        if submission.status.is_terminal() {
            return Err(format!(
                "Submission {} is already in terminal state: {}",
                id, submission.status
            ));
        }

        submission.status = SubmissionStatus::Failed;
        submission.error = Some(error);

        self.stats.failed += 1;
        if self.stats.pending > 0 {
            self.stats.pending -= 1;
        }

        Ok(())
    }

    /// Mark a submission as timed out.
    pub fn mark_timed_out(&mut self, id: &SubmissionId) -> Result<(), String> {
        let submission = self
            .submissions
            .get_mut(id)
            .ok_or_else(|| format!("Submission not found: {}", id))?;

        if submission.status.is_terminal() {
            return Err(format!(
                "Submission {} is already in terminal state: {}",
                id, submission.status
            ));
        }

        submission.status = SubmissionStatus::TimedOut;
        self.stats.timed_out += 1;
        if self.stats.pending > 0 {
            self.stats.pending -= 1;
        }

        Ok(())
    }

    /// Get all verification records.
    pub fn verification_records(&self) -> &[VerificationRecord] {
        &self.records
    }

    /// Get verification records for a specific solver type.
    pub fn records_by_solver(
        &self,
        solver: SolverType,
    ) -> Vec<&VerificationRecord> {
        self.records
            .iter()
            .filter(|r| r.solver_type == solver)
            .collect()
    }

    /// List all submission IDs.
    pub fn list_submissions(&self) -> Vec<&SubmissionId> {
        self.submissions.keys().collect()
    }

    /// List submissions by status.
    pub fn submissions_by_status(
        &self,
        status: SubmissionStatus,
    ) -> Vec<&GevulotSubmission> {
        self.submissions
            .values()
            .filter(|s| s.status == status)
            .collect()
    }

    /// Get cumulative stats.
    pub fn stats(&self) -> &GevulotStats {
        &self.stats
    }

    /// Get network configuration.
    pub fn config(&self) -> &GevulotConfig {
        &self.config
    }

    /// Total number of submissions.
    pub fn total_submissions(&self) -> usize {
        self.submissions.len()
    }

    /// Number of pending submissions.
    pub fn pending_count(&self) -> u64 {
        self.stats.pending
    }

    /// Export all submissions as JSON.
    pub fn export_submissions_json(&self) -> Result<String, String> {
        let submissions: Vec<&GevulotSubmission> =
            self.submissions.values().collect();
        serde_json::to_string_pretty(&submissions)
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Export all verification records as JSON.
    pub fn export_records_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.records)
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Thread-Safe Client Wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Thread-safe wrapper around `GevulotClient`.
///
/// Allows concurrent access from multiple server threads (e.g. Axum handlers).
#[derive(Clone)]
pub struct SharedGevulotClient {
    inner: Arc<Mutex<GevulotClient>>,
}

impl SharedGevulotClient {
    /// Create from an existing client.
    pub fn new(client: GevulotClient) -> Self {
        Self {
            inner: Arc::new(Mutex::new(client)),
        }
    }

    /// Create a local development client.
    pub fn local() -> Self {
        Self::new(GevulotClient::local())
    }

    /// Submit a proof (locks the client).
    pub fn submit_proof<P: PhysicsProof>(
        &self,
        proof: &P,
    ) -> Result<SubmissionId, String> {
        self.inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?
            .submit_proof(proof)
    }

    /// Check submission status.
    pub fn check_status(
        &self,
        id: &SubmissionId,
    ) -> Result<SubmissionStatus, String> {
        self.inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?
            .check_status(id)
    }

    /// Mark verified.
    pub fn mark_verified(
        &self,
        id: &SubmissionId,
        verifier_count: u32,
        verification_time_ms: u64,
    ) -> Result<VerificationRecord, String> {
        self.inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?
            .mark_verified(id, verifier_count, verification_time_ms)
    }

    /// Get stats snapshot.
    pub fn stats(&self) -> Result<GevulotStats, String> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?
            .stats()
            .clone())
    }

    /// Total submissions.
    pub fn total_submissions(&self) -> Result<usize, String> {
        Ok(self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?
            .total_submissions())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fluidelite_circuits::euler3d::{Euler3DParams, Euler3DProver};
    use fluidelite_core::mpo::MPO;
    use fluidelite_core::mps::MPS;
    use fluidelite_circuits::ns_imex::{NSIMEXParams, NSIMEXProver};
    use fluidelite_core::physics_traits::PhysicsProver;

    fn make_euler3d_proof() -> fluidelite_circuits::euler3d::Euler3DProof {
        let params = Euler3DParams::test_small();
        let mut prover = Euler3DProver::new(params).unwrap();
        let states: Vec<MPS> = (0..5).map(|_| MPS::new(4, 4, 2)).collect();
        let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        prover.prove(&states, &mpos).unwrap()
    }

    fn make_ns_imex_proof() -> fluidelite_circuits::ns_imex::NSIMEXProof {
        let params = NSIMEXParams::test_small();
        let mut prover = NSIMEXProver::new(params).unwrap();
        let states: Vec<MPS> = (0..3).map(|_| MPS::new(4, 4, 2)).collect();
        let mpos: Vec<MPO> = (0..3).map(|_| MPO::identity(4, 2)).collect();
        prover.prove(&states, &mpos).unwrap()
    }

    #[test]
    fn test_client_creation() {
        let client = GevulotClient::local();
        assert_eq!(client.total_submissions(), 0);
    }

    #[test]
    fn test_client_invalid_config() {
        let config = GevulotConfig {
            gateway_url: String::new(),
            ..GevulotConfig::default()
        };
        assert!(GevulotClient::new(config).is_err());
    }

    #[test]
    fn test_submit_euler3d() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();
        assert!(id.0.starts_with("gvlt-euler3d-"));
        assert_eq!(client.total_submissions(), 1);
        assert_eq!(
            client.check_status(&id).unwrap(),
            SubmissionStatus::Submitted
        );
    }

    #[test]
    fn test_submit_ns_imex() {
        let mut client = GevulotClient::local();
        let proof = make_ns_imex_proof();
        let id = client.submit_proof(&proof).unwrap();
        assert!(id.0.starts_with("gvlt-ns_imex-"));
        assert_eq!(client.total_submissions(), 1);
    }

    #[test]
    fn test_duplicate_submission() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let _id = client.submit_proof(&proof).unwrap();
        // Same proof → same ID → duplicate
        assert!(client.submit_proof(&proof).is_err());
    }

    #[test]
    fn test_mark_verified() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();

        let record = client.mark_verified(&id, 3, 150).unwrap();
        assert!(record.valid);
        assert_eq!(record.verifier_count, 3);
        assert_eq!(
            client.check_status(&id).unwrap(),
            SubmissionStatus::Verified
        );
        assert_eq!(client.stats().verified, 1);
    }

    #[test]
    fn test_mark_failed() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();

        client.mark_failed(&id, "Invalid proof".into()).unwrap();
        assert_eq!(
            client.check_status(&id).unwrap(),
            SubmissionStatus::Failed
        );
        assert_eq!(client.stats().failed, 1);
    }

    #[test]
    fn test_mark_timed_out() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();

        client.mark_timed_out(&id).unwrap();
        assert_eq!(
            client.check_status(&id).unwrap(),
            SubmissionStatus::TimedOut
        );
        assert_eq!(client.stats().timed_out, 1);
    }

    #[test]
    fn test_terminal_state_immutable() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();

        client.mark_verified(&id, 3, 100).unwrap();
        // Can't mark failed after verified
        assert!(client.mark_failed(&id, "error".into()).is_err());
    }

    #[test]
    fn test_not_found() {
        let client = GevulotClient::local();
        let id = SubmissionId("nonexistent".into());
        assert!(client.check_status(&id).is_err());
    }

    #[test]
    fn test_records_by_solver() {
        let mut client = GevulotClient::local();

        let e_proof = make_euler3d_proof();
        let e_id = client.submit_proof(&e_proof).unwrap();
        client.mark_verified(&e_id, 3, 100).unwrap();

        let n_proof = make_ns_imex_proof();
        let n_id = client.submit_proof(&n_proof).unwrap();
        client.mark_verified(&n_id, 5, 200).unwrap();

        let euler_records = client.records_by_solver(SolverType::Euler3D);
        assert_eq!(euler_records.len(), 1);
        assert_eq!(euler_records[0].solver_type, SolverType::Euler3D);

        let ns_records = client.records_by_solver(SolverType::NsImex);
        assert_eq!(ns_records.len(), 1);
    }

    #[test]
    fn test_submissions_by_status() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();

        let submitted =
            client.submissions_by_status(SubmissionStatus::Submitted);
        assert_eq!(submitted.len(), 1);

        let verified =
            client.submissions_by_status(SubmissionStatus::Verified);
        assert!(verified.is_empty());
    }

    #[test]
    fn test_stats_tracking() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();

        assert_eq!(client.stats().total_submissions, 1);
        assert_eq!(client.stats().pending, 1);
        assert!(client.stats().total_bytes_submitted > 0);

        client.mark_verified(&id, 3, 100).unwrap();
        assert_eq!(client.stats().pending, 0);
        assert_eq!(client.stats().verified, 1);
        assert!(client.stats().avg_verification_time_ms > 0.0);
    }

    #[test]
    fn test_export_json() {
        let mut client = GevulotClient::local();
        let proof = make_euler3d_proof();
        let id = client.submit_proof(&proof).unwrap();
        client.mark_verified(&id, 3, 100).unwrap();

        let subs_json = client.export_submissions_json().unwrap();
        assert!(subs_json.contains("gvlt-euler3d-"));

        let recs_json = client.export_records_json().unwrap();
        assert!(recs_json.contains("euler3d"));
    }

    #[test]
    fn test_submit_raw() {
        let mut client = GevulotClient::local();
        let id = client
            .submit_raw(
                SolverType::Euler3D,
                vec![1, 2, 3, 4],
                [10, 20, 30, 40],
                [50, 60, 70, 80],
                [90, 100, 110, 120],
                5000,
                14,
                4,
                4,
            )
            .unwrap();
        assert!(id.0.starts_with("gvlt-euler3d-"));
        assert_eq!(client.total_submissions(), 1);
    }

    #[test]
    fn test_shared_client() {
        let shared = SharedGevulotClient::local();
        assert_eq!(shared.total_submissions().unwrap(), 0);

        let proof = make_euler3d_proof();
        let id = shared.submit_proof(&proof).unwrap();
        assert_eq!(shared.total_submissions().unwrap(), 1);
        assert_eq!(
            shared.check_status(&id).unwrap(),
            SubmissionStatus::Submitted
        );

        let record = shared.mark_verified(&id, 3, 100).unwrap();
        assert!(record.valid);

        let stats = shared.stats().unwrap();
        assert_eq!(stats.verified, 1);
    }

    #[test]
    fn test_submit_empty_proof_bytes() {
        let mut client = GevulotClient::local();
        let result = client.submit_raw(
            SolverType::Euler3D,
            vec![],
            [0; 4],
            [0; 4],
            [0; 4],
            0,
            0,
            0,
            0,
        );
        assert!(result.is_err());
    }
}

//! Gevulot network protocol types.
//!
//! Defines the data structures for submitting proofs to the Gevulot
//! decentralized verification network and retrieving verification records.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::fmt;

use fluidelite_core::physics_traits::SolverType;

// ═══════════════════════════════════════════════════════════════════════════
// Submission Types
// ═══════════════════════════════════════════════════════════════════════════

/// Unique identifier for a Gevulot proof submission.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubmissionId(pub String);

impl fmt::Display for SubmissionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl SubmissionId {
    /// Generate a new submission ID from proof hash.
    pub fn from_proof_hash(solver: SolverType, hash_limbs: &[u64; 4]) -> Self {
        let mut id = format!("gvlt-{}-", solver);
        for limb in hash_limbs {
            id.push_str(&format!("{:016x}", limb));
        }
        Self(id)
    }
}

/// Status of a proof submission on the Gevulot network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionStatus {
    /// Submission created locally, not yet sent to network.
    Pending,
    /// Submitted to Gevulot network, awaiting processing.
    Submitted,
    /// Being verified by Gevulot provers.
    Verifying,
    /// Verification succeeded — proof is publicly verifiable.
    Verified,
    /// Verification failed — proof is invalid.
    Failed,
    /// Submission timed out.
    TimedOut,
    /// Submission rejected by network (invalid format, etc.).
    Rejected,
}

impl SubmissionStatus {
    /// Whether this is a terminal state (no further transitions).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Verified | Self::Failed | Self::TimedOut | Self::Rejected
        )
    }

    /// Whether the proof was successfully verified.
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified)
    }
}

impl fmt::Display for SubmissionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Submitted => write!(f, "submitted"),
            Self::Verifying => write!(f, "verifying"),
            Self::Verified => write!(f, "verified"),
            Self::Failed => write!(f, "failed"),
            Self::TimedOut => write!(f, "timed_out"),
            Self::Rejected => write!(f, "rejected"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gevulot Proof Submission
// ═══════════════════════════════════════════════════════════════════════════

/// A proof submitted to the Gevulot network for decentralized verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GevulotSubmission {
    /// Unique submission ID.
    pub id: SubmissionId,

    /// Solver that produced the proof.
    pub solver_type: SolverType,

    /// Current status.
    pub status: SubmissionStatus,

    /// Raw proof bytes (serialized).
    pub proof_bytes: Vec<u8>,

    /// Proof size in bytes.
    pub proof_size: usize,

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

    /// Submission timestamp (Unix epoch seconds).
    pub submitted_at: u64,

    /// Verification timestamp (if verified).
    pub verified_at: Option<u64>,

    /// Verification time on Gevulot network in milliseconds.
    pub verification_time_ms: Option<u64>,

    /// Gevulot transaction hash (if submitted).
    pub tx_hash: Option<String>,

    /// Error message (if failed/rejected).
    pub error: Option<String>,

    /// Number of Gevulot verifiers that confirmed.
    pub verifier_count: u32,

    /// Lean proof file references.
    pub lean_proofs: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Verification Record
// ═══════════════════════════════════════════════════════════════════════════

/// A publicly verifiable record from the Gevulot network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRecord {
    /// Submission that was verified.
    pub submission_id: SubmissionId,

    /// Solver type.
    pub solver_type: SolverType,

    /// Whether the proof was verified as valid.
    pub valid: bool,

    /// Number of independent verifiers.
    pub verifier_count: u32,

    /// Consensus threshold (fraction of verifiers that agreed).
    pub consensus_fraction: f64,

    /// Verification timestamp (Unix epoch seconds).
    pub timestamp: u64,

    /// On-chain transaction hash.
    pub tx_hash: String,

    /// Block number on Gevulot.
    pub block_number: u64,

    /// Proof metadata hash (SHA-256 of proof + params).
    pub proof_metadata_hash: String,

    /// Grid bits used.
    pub grid_bits: usize,

    /// Bond dimension.
    pub chi_max: usize,

    /// Number of constraints.
    pub num_constraints: usize,

    /// Lean proof files referenced.
    pub lean_proofs: Vec<String>,
}

impl VerificationRecord {
    /// Create a record for an expired/timed-out submission.
    pub fn expired(submission_id: SubmissionId, solver_type: SolverType) -> Self {
        Self {
            submission_id,
            solver_type,
            valid: false,
            verifier_count: 0,
            consensus_fraction: 0.0,
            timestamp: current_unix_time(),
            tx_hash: String::new(),
            block_number: 0,
            proof_metadata_hash: String::new(),
            grid_bits: 0,
            chi_max: 0,
            num_constraints: 0,
            lean_proofs: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Network Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for connecting to the Gevulot network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GevulotConfig {
    /// Gevulot gateway endpoint URL.
    pub gateway_url: String,

    /// API key for authenticated access.
    pub api_key: String,

    /// Program ID deployed on Gevulot for verification.
    pub program_id: String,

    /// Network identifier (mainnet, testnet, devnet).
    pub network: GevulotNetwork,

    /// Submission timeout in seconds.
    pub submission_timeout_secs: u64,

    /// Polling interval for status checks in milliseconds.
    pub poll_interval_ms: u64,

    /// Maximum retry attempts for failed submissions.
    pub max_retries: u32,

    /// Minimum verifier count for consensus.
    pub min_verifiers: u32,

    /// Consensus threshold (fraction of verifiers that must agree).
    pub consensus_threshold: f64,
}

impl Default for GevulotConfig {
    fn default() -> Self {
        Self {
            gateway_url: "https://gateway.gevulot.com".into(),
            api_key: String::new(),
            program_id: String::new(),
            network: GevulotNetwork::Devnet,
            submission_timeout_secs: 300,
            poll_interval_ms: 5000,
            max_retries: 3,
            min_verifiers: 3,
            consensus_threshold: 0.67,
        }
    }
}

impl GevulotConfig {
    /// Configuration for local development/testing.
    pub fn local() -> Self {
        Self {
            gateway_url: "http://localhost:9944".into(),
            api_key: "dev-key".into(),
            program_id: "dev-program".into(),
            network: GevulotNetwork::Devnet,
            submission_timeout_secs: 30,
            poll_interval_ms: 1000,
            max_retries: 1,
            min_verifiers: 1,
            consensus_threshold: 1.0,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.gateway_url.is_empty() {
            return Err("Gateway URL must not be empty".into());
        }
        if self.program_id.is_empty() {
            return Err("Program ID must not be empty".into());
        }
        if self.consensus_threshold <= 0.0 || self.consensus_threshold > 1.0 {
            return Err(format!(
                "Consensus threshold must be in (0, 1], got {}",
                self.consensus_threshold
            ));
        }
        if self.min_verifiers == 0 {
            return Err("Min verifiers must be ≥ 1".into());
        }
        Ok(())
    }
}

/// Gevulot network identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GevulotNetwork {
    /// Main production network.
    Mainnet,
    /// Test network.
    Testnet,
    /// Development network (local or staging).
    Devnet,
}

impl fmt::Display for GevulotNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mainnet => write!(f, "mainnet"),
            Self::Testnet => write!(f, "testnet"),
            Self::Devnet => write!(f, "devnet"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Gevulot Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for Gevulot network interactions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GevulotStats {
    /// Total submissions.
    pub total_submissions: u64,

    /// Successful verifications.
    pub verified: u64,

    /// Failed verifications.
    pub failed: u64,

    /// Timed-out submissions.
    pub timed_out: u64,

    /// Rejected submissions.
    pub rejected: u64,

    /// Pending submissions.
    pub pending: u64,

    /// Total proof bytes submitted.
    pub total_bytes_submitted: u64,

    /// Average verification time in milliseconds.
    pub avg_verification_time_ms: f64,

    /// Average verifier count per proof.
    pub avg_verifier_count: f64,
}

impl GevulotStats {
    /// Success rate as a fraction.
    pub fn success_rate(&self) -> f64 {
        let completed = self.verified + self.failed;
        if completed == 0 {
            0.0
        } else {
            self.verified as f64 / completed as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Get the current Unix time in seconds.
pub fn current_unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Compute a hex-encoded hash of proof metadata.
pub fn compute_proof_metadata_hash(
    solver_type: SolverType,
    input_hash: &[u64; 4],
    output_hash: &[u64; 4],
    params_hash: &[u64; 4],
    num_constraints: usize,
) -> String {
    // Simple FNV-1a hash of metadata fields
    let mut hash: u64 = 0xcbf29ce484222325;
    let mix = |h: &mut u64, val: u64| {
        *h ^= val;
        *h = h.wrapping_mul(0x100000001b3);
    };

    mix(&mut hash, solver_type as u64);
    for limb in input_hash {
        mix(&mut hash, *limb);
    }
    for limb in output_hash {
        mix(&mut hash, *limb);
    }
    for limb in params_hash {
        mix(&mut hash, *limb);
    }
    mix(&mut hash, num_constraints as u64);

    format!("{:016x}", hash)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submission_id_from_hash() {
        let hash = [1u64, 2, 3, 4];
        let id = SubmissionId::from_proof_hash(SolverType::Euler3D, &hash);
        assert!(id.0.starts_with("gvlt-euler3d-"));
        assert!(id.0.len() > 20);
    }

    #[test]
    fn test_submission_status_terminal() {
        assert!(!SubmissionStatus::Pending.is_terminal());
        assert!(!SubmissionStatus::Submitted.is_terminal());
        assert!(!SubmissionStatus::Verifying.is_terminal());
        assert!(SubmissionStatus::Verified.is_terminal());
        assert!(SubmissionStatus::Failed.is_terminal());
        assert!(SubmissionStatus::TimedOut.is_terminal());
        assert!(SubmissionStatus::Rejected.is_terminal());
    }

    #[test]
    fn test_submission_status_verified() {
        assert!(SubmissionStatus::Verified.is_verified());
        assert!(!SubmissionStatus::Failed.is_verified());
        assert!(!SubmissionStatus::Pending.is_verified());
    }

    #[test]
    fn test_gevulot_config_default() {
        let config = GevulotConfig::default();
        assert!(config.gateway_url.contains("gevulot"));
        assert_eq!(config.network, GevulotNetwork::Devnet);
    }

    #[test]
    fn test_gevulot_config_local() {
        let config = GevulotConfig::local();
        assert!(config.gateway_url.contains("localhost"));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gevulot_config_validate_empty_url() {
        let config = GevulotConfig {
            gateway_url: String::new(),
            ..GevulotConfig::local()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gevulot_config_validate_bad_threshold() {
        let config = GevulotConfig {
            consensus_threshold: 1.5,
            ..GevulotConfig::local()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_verification_record_expired() {
        let id = SubmissionId("test-123".into());
        let record = VerificationRecord::expired(id.clone(), SolverType::NsImex);
        assert!(!record.valid);
        assert_eq!(record.verifier_count, 0);
        assert_eq!(record.submission_id, id);
    }

    #[test]
    fn test_gevulot_stats_success_rate() {
        let stats = GevulotStats {
            verified: 8,
            failed: 2,
            ..Default::default()
        };
        assert!((stats.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_gevulot_stats_no_completed() {
        let stats = GevulotStats::default();
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_proof_metadata_hash() {
        let hash = compute_proof_metadata_hash(
            SolverType::Euler3D,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            1000,
        );
        assert_eq!(hash.len(), 16);

        // Deterministic
        let hash2 = compute_proof_metadata_hash(
            SolverType::Euler3D,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            1000,
        );
        assert_eq!(hash, hash2);

        // Different inputs → different hash
        let hash3 = compute_proof_metadata_hash(
            SolverType::NsImex,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            1000,
        );
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_current_unix_time() {
        let t = current_unix_time();
        // Should be after 2025-01-01
        assert!(t > 1735689600);
    }

    #[test]
    fn test_gevulot_network_display() {
        assert_eq!(GevulotNetwork::Mainnet.to_string(), "mainnet");
        assert_eq!(GevulotNetwork::Testnet.to_string(), "testnet");
        assert_eq!(GevulotNetwork::Devnet.to_string(), "devnet");
    }
}

//! Trace Parser — reads computation traces from Python's TraceSession.
//!
//! Supports both JSON and compact binary (`.trc`) formats.

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const TRACE_MAGIC: &[u8; 4] = b"TRCV";
const TRACE_VERSION: u32 = 1;

// ═══════════════════════════════════════════════════════════════════════════
// Operation Types (mirrors Python OpType enum)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpType {
    SvdTruncated,
    SvdExact,
    QrPositive,
    ThinSvd,
    PolarDecomposition,
    MpoApply,
    MpsNormalize,
    MpsCanonicalizeLe,
    MpsCanonicalizeRight,
    MpsCanonicalizeTo,
    MpsTruncate,
    MpsFromTensor,
    Contraction,
    Custom,
}

impl OpType {
    pub fn from_str_value(s: &str) -> Result<Self> {
        match s {
            "svd_truncated" => Ok(Self::SvdTruncated),
            "svd_exact" => Ok(Self::SvdExact),
            "qr_positive" => Ok(Self::QrPositive),
            "thin_svd" => Ok(Self::ThinSvd),
            "polar_decomposition" => Ok(Self::PolarDecomposition),
            "mpo_apply" => Ok(Self::MpoApply),
            "mps_normalize" => Ok(Self::MpsNormalize),
            "mps_canonicalize_left" => Ok(Self::MpsCanonicalizeLe),
            "mps_canonicalize_right" => Ok(Self::MpsCanonicalizeRight),
            "mps_canonicalize_to" => Ok(Self::MpsCanonicalizeTo),
            "mps_truncate" => Ok(Self::MpsTruncate),
            "mps_from_tensor" => Ok(Self::MpsFromTensor),
            "contraction" => Ok(Self::Contraction),
            "custom" => Ok(Self::Custom),
            _ => bail!("Unknown op type: {s}"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace Entry
// ═══════════════════════════════════════════════════════════════════════════

/// Single traced operation from the Python computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Global sequence number within the trace session.
    pub seq: u64,

    /// Operation type.
    pub op: String,

    /// Timestamp in nanoseconds since UNIX epoch.
    pub timestamp_ns: i64,

    /// Wall-clock duration in nanoseconds.
    #[serde(default)]
    pub duration_ns: u64,

    /// SHA-256 hashes of input tensors.
    #[serde(default)]
    pub input_hashes: HashMap<String, String>,

    /// SHA-256 hashes of output tensors.
    #[serde(default)]
    pub output_hashes: HashMap<String, String>,

    /// Operation parameters (chi_max, cutoff, shapes, etc.).
    #[serde(default)]
    pub params: serde_json::Value,

    /// Computed metrics (truncation_error, singular_values, etc.).
    #[serde(default)]
    pub metrics: serde_json::Value,
}

impl TraceEntry {
    /// Parse the operation type.
    pub fn op_type(&self) -> Result<OpType> {
        OpType::from_str_value(&self.op)
    }

    /// Get a parameter value as a specific type.
    pub fn param_i64(&self, key: &str) -> Option<i64> {
        self.params.get(key).and_then(|v| v.as_i64())
    }

    pub fn param_f64(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|v| v.as_f64())
    }

    pub fn param_bool(&self, key: &str) -> Option<bool> {
        self.params.get(key).and_then(|v| v.as_bool())
    }

    pub fn param_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(|v| v.as_str())
    }

    /// Get singular values from metrics (for SVD operations).
    pub fn singular_values(&self) -> Option<Vec<f64>> {
        self.metrics
            .get("singular_values")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_f64()).collect())
    }

    /// Get truncation error from metrics.
    pub fn truncation_error(&self) -> Option<f64> {
        self.metrics.get("truncation_error").and_then(|v| v.as_f64())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace Record — full parsed trace
// ═══════════════════════════════════════════════════════════════════════════

/// Complete parsed trace from a Python computation session.
#[derive(Debug, Clone)]
pub struct TraceRecord {
    /// Session UUID.
    pub session_id: Uuid,

    /// All trace entries in execution order.
    pub entries: Vec<TraceEntry>,

    /// SHA-256 chain hash of all entries (matches Python's TraceDigest).
    pub chain_hash: String,

    /// Operation counts by type.
    pub op_counts: HashMap<String, usize>,

    /// Total computation duration in nanoseconds.
    pub total_duration_ns: u64,
}

impl TraceRecord {
    /// Compute the chain hash over all entries (must match Python side).
    pub fn compute_chain_hash(entries: &[TraceEntry]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"TPC_TRACE_V1");

        for entry in entries {
            let json_bytes = serde_json::to_vec(entry).unwrap_or_default();
            hasher.update(&json_bytes);
        }

        hex::encode(hasher.finalize())
    }

    /// Get all SVD entries.
    pub fn svd_entries(&self) -> Vec<&TraceEntry> {
        self.entries
            .iter()
            .filter(|e| e.op == "svd_truncated" || e.op == "svd_exact")
            .collect()
    }

    /// Get all MPO application entries.
    pub fn mpo_entries(&self) -> Vec<&TraceEntry> {
        self.entries
            .iter()
            .filter(|e| e.op == "mpo_apply")
            .collect()
    }

    /// Get all truncation entries.
    pub fn truncation_entries(&self) -> Vec<&TraceEntry> {
        self.entries
            .iter()
            .filter(|e| e.op == "mps_truncate")
            .collect()
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Validate the chain hash matches the entries.
    pub fn validate_chain_hash(&self) -> bool {
        let computed = Self::compute_chain_hash(&self.entries);
        computed == self.chain_hash
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trace Parser
// ═══════════════════════════════════════════════════════════════════════════

/// Parser for Python computation traces.
pub struct TraceParser;

impl TraceParser {
    /// Parse a JSON trace file.
    pub fn parse_json(path: &Path) -> Result<TraceRecord> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read trace file: {}", path.display()))?;

        Self::parse_json_str(&content)
    }

    /// Parse a JSON trace from a string.
    pub fn parse_json_str(content: &str) -> Result<TraceRecord> {
        let payload: serde_json::Value = serde_json::from_str(content)
            .context("Failed to parse trace JSON")?;

        let session_id_str = payload["session_id"]
            .as_str()
            .context("Missing session_id")?;
        let session_id = Uuid::parse_str(session_id_str)
            .context("Invalid session UUID")?;

        let entries_arr = payload["entries"]
            .as_array()
            .context("Missing entries array")?;

        let mut entries = Vec::with_capacity(entries_arr.len());
        for entry_val in entries_arr {
            let entry: TraceEntry = serde_json::from_value(entry_val.clone())
                .context("Failed to parse trace entry")?;
            entries.push(entry);
        }

        let chain_hash = TraceRecord::compute_chain_hash(&entries);

        let mut op_counts: HashMap<String, usize> = HashMap::new();
        let mut total_duration_ns = 0u64;
        for entry in &entries {
            *op_counts.entry(entry.op.clone()).or_insert(0) += 1;
            total_duration_ns += entry.duration_ns;
        }

        Ok(TraceRecord {
            session_id,
            entries,
            chain_hash,
            op_counts,
            total_duration_ns,
        })
    }

    /// Parse a binary trace file (`.trc` format from Python).
    pub fn parse_binary(path: &Path) -> Result<TraceRecord> {
        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read binary trace: {}", path.display()))?;

        Self::parse_binary_bytes(&data)
    }

    /// Parse binary trace from bytes.
    pub fn parse_binary_bytes(data: &[u8]) -> Result<TraceRecord> {
        let mut cursor = Cursor::new(data);

        // Magic
        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .context("Failed to read magic")?;
        if &magic != TRACE_MAGIC {
            bail!("Invalid trace magic: {:?} (expected {:?})", magic, TRACE_MAGIC);
        }

        // Version
        let version = cursor
            .read_u32::<LittleEndian>()
            .context("Failed to read version")?;
        if version != TRACE_VERSION {
            bail!("Unsupported trace version: {} (expected {})", version, TRACE_VERSION);
        }

        // Session UUID
        let mut uuid_bytes = [0u8; 16];
        cursor
            .read_exact(&mut uuid_bytes)
            .context("Failed to read session UUID")?;
        let session_id = Uuid::from_bytes(uuid_bytes);

        // Entry count
        let entry_count = cursor
            .read_u64::<LittleEndian>()
            .context("Failed to read entry count")? as usize;

        // Entries
        let mut entries = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let json_len = cursor
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read JSON length for entry {i}"))?
                as usize;

            let mut json_buf = vec![0u8; json_len];
            cursor
                .read_exact(&mut json_buf)
                .with_context(|| format!("Failed to read JSON for entry {i}"))?;

            let entry: TraceEntry = serde_json::from_slice(&json_buf)
                .with_context(|| format!("Failed to parse entry {i}"))?;
            entries.push(entry);
        }

        let chain_hash = TraceRecord::compute_chain_hash(&entries);

        let mut op_counts: HashMap<String, usize> = HashMap::new();
        let mut total_duration_ns = 0u64;
        for entry in &entries {
            *op_counts.entry(entry.op.clone()).or_insert(0) += 1;
            total_duration_ns += entry.duration_ns;
        }

        Ok(TraceRecord {
            session_id,
            entries,
            chain_hash,
            op_counts,
            total_duration_ns,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_op_types() {
        assert_eq!(OpType::from_str_value("svd_truncated").unwrap(), OpType::SvdTruncated);
        assert_eq!(OpType::from_str_value("mpo_apply").unwrap(), OpType::MpoApply);
        assert_eq!(OpType::from_str_value("mps_truncate").unwrap(), OpType::MpsTruncate);
        assert!(OpType::from_str_value("nonexistent").is_err());
    }

    #[test]
    fn test_parse_json_trace() {
        let json = r#"{
            "trace_version": 1,
            "session_id": "12345678-1234-1234-1234-123456789abc",
            "digest": {
                "session_id": "12345678-1234-1234-1234-123456789abc",
                "trace_hash": "abc123",
                "entry_count": 1,
                "op_counts": {"svd_truncated": 1}
            },
            "entries": [
                {
                    "seq": 0,
                    "op": "svd_truncated",
                    "timestamp_ns": 1700000000000000000,
                    "duration_ns": 1000000,
                    "input_hashes": {"A": "deadbeef"},
                    "output_hashes": {"U": "cafebabe", "S": "12345678", "Vh": "abcdef01"},
                    "params": {"input_shape": [100, 100], "chi_max": 20, "cutoff": 1e-14},
                    "metrics": {"truncation_error": 1e-6, "rank": 20, "singular_values": [1.0, 0.5, 0.1]}
                }
            ]
        }"#;

        let record = TraceParser::parse_json_str(json).unwrap();
        assert_eq!(record.entries.len(), 1);
        assert_eq!(record.entries[0].op, "svd_truncated");
        assert_eq!(record.entries[0].singular_values().unwrap().len(), 3);
        assert!((record.entries[0].truncation_error().unwrap() - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_chain_hash_determinism() {
        let entries = vec![
            TraceEntry {
                seq: 0,
                op: "svd_truncated".to_string(),
                timestamp_ns: 1000,
                duration_ns: 100,
                input_hashes: HashMap::new(),
                output_hashes: HashMap::new(),
                params: serde_json::Value::Null,
                metrics: serde_json::Value::Null,
            },
        ];

        let hash1 = TraceRecord::compute_chain_hash(&entries);
        let hash2 = TraceRecord::compute_chain_hash(&entries);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 hex
    }
}

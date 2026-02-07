//! Circuit Builder — transforms computation traces into ZK circuit inputs.
//!
//! Takes a `TraceRecord` and produces `CircuitInputs` that can be fed
//! directly to the `fluidelite-zk` prover.
//!
//! # Constraint Types
//!
//! 1. **SVD Integrity**: For each SVD truncation, verify that:
//!    - U is semi-unitary: U^T U ≈ I (within declared tolerance)
//!    - Singular values are non-negative and ordered
//!    - Truncation error matches declared value
//!    - Rank does not exceed chi_max
//!
//! 2. **Hash Chain**: The trace hash commits to all operations.
//!
//! 3. **MPO Contraction**: Bond dimensions match.
//!
//! 4. **Conservation**: Energy/mass conservation within tolerance.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
// sha2 reserved for Phase 1 hash-commitment constraints
use std::collections::HashMap;

use crate::trace_parser::{OpType, TraceEntry, TraceRecord};

// ═══════════════════════════════════════════════════════════════════════════
// Circuit Constraint Types
// ═══════════════════════════════════════════════════════════════════════════

/// A single constraint that must be satisfied in the ZK circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConstraint {
    /// Constraint type identifier.
    pub kind: ConstraintKind,

    /// Human-readable label.
    pub label: String,

    /// Which trace entry this constraint comes from.
    pub source_seq: u64,

    /// Constraint-specific data.
    pub data: ConstraintData,
}

/// Enumeration of constraint types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    /// Singular values are non-negative and descending.
    SvOrdering,
    /// Truncation rank ≤ chi_max.
    RankBound,
    /// Truncation error matches declared value.
    TruncationError,
    /// Hash of input tensor matches declared hash.
    InputHash,
    /// Hash of output tensor matches declared hash.
    OutputHash,
    /// Bond dimensions are compatible between operations.
    BondDimCompatibility,
    /// Overall trace chain hash.
    TraceChainHash,
    /// Conservation law (mass, energy, etc.).
    Conservation,
}

/// Data associated with a constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConstraintData {
    /// Singular value ordering: s[i] >= s[i+1] >= 0.
    SvOrdering {
        /// Number of singular values.
        count: usize,
        /// The singular values (public inputs to the circuit).
        values: Vec<f64>,
    },
    /// Rank is bounded: rank <= chi_max.
    RankBound {
        rank: u64,
        chi_max: u64,
    },
    /// Truncation error ≤ declared tolerance.
    TruncationError {
        declared_error: f64,
        tolerance: f64,
    },
    /// SHA-256 hash commitment.
    HashCommitment {
        name: String,
        expected_hash: String,
    },
    /// Bond dimension compatibility between adjacent operations.
    BondDimCompatibility {
        left_seq: u64,
        right_seq: u64,
        left_dim: u64,
        right_dim: u64,
    },
    /// Trace chain hash.
    TraceChainHash {
        expected_hash: String,
    },
    /// Conservation law.
    Conservation {
        quantity: String,
        before: f64,
        after: f64,
        tolerance: f64,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// Circuit Inputs — what the prover needs
// ═══════════════════════════════════════════════════════════════════════════

/// Complete set of inputs for the ZK circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInputs {
    /// Session ID of the source trace.
    pub session_id: String,

    /// Trace chain hash (becomes a public input to the ZK proof).
    pub trace_hash: String,

    /// All constraints that the circuit must enforce.
    pub constraints: Vec<CircuitConstraint>,

    /// Public inputs: values visible to the verifier.
    pub public_inputs: HashMap<String, serde_json::Value>,

    /// Private witnesses: values only the prover knows.
    pub private_witnesses: HashMap<String, serde_json::Value>,

    /// Total number of trace operations.
    pub trace_entry_count: usize,

    /// Summary statistics for the verifier.
    pub summary: CircuitSummary,
}

/// Summary statistics embedded in the circuit inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitSummary {
    pub total_svd_ops: usize,
    pub total_mpo_ops: usize,
    pub total_truncations: usize,
    pub max_rank: u64,
    pub max_bond_dim: u64,
    pub total_constraints: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Circuit Builder
// ═══════════════════════════════════════════════════════════════════════════

/// Builds ZK circuit inputs from a computation trace.
pub struct CircuitBuilder {
    /// Maximum allowed truncation error for constraint generation.
    pub max_truncation_error: f64,

    /// Whether to include hash commitments for every tensor.
    pub include_hash_commitments: bool,

    /// Whether to verify singular value ordering.
    pub verify_sv_ordering: bool,
}

impl Default for CircuitBuilder {
    fn default() -> Self {
        Self {
            max_truncation_error: 1e-6,
            include_hash_commitments: true,
            verify_sv_ordering: true,
        }
    }
}

impl CircuitBuilder {
    /// Create a new circuit builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build circuit inputs from a trace record.
    pub fn build(&self, trace: &TraceRecord) -> Result<CircuitInputs> {
        let mut constraints = Vec::new();
        let mut public_inputs = HashMap::new();
        let mut private_witnesses = HashMap::new();

        let mut max_rank: u64 = 0;
        let mut max_bond_dim: u64 = 0;
        let mut total_svd = 0usize;
        let mut total_mpo = 0usize;
        let mut total_truncations = 0usize;

        // Process each trace entry
        for entry in &trace.entries {
            let op = entry.op_type()
                .with_context(|| format!("Unknown op in entry seq={}", entry.seq))?;

            match op {
                OpType::SvdTruncated | OpType::SvdExact => {
                    total_svd += 1;
                    self.process_svd(entry, &mut constraints, &mut public_inputs, &mut private_witnesses)?;

                    if let Some(rank) = entry.param_i64("rank").or_else(|| {
                        entry.metrics.get("rank").and_then(|v| v.as_i64())
                    }) {
                        max_rank = max_rank.max(rank as u64);
                    }
                }
                OpType::MpoApply => {
                    total_mpo += 1;
                    self.process_mpo_apply(entry, &mut constraints)?;

                    if let Some(result_dims) = entry.params.get("result_bond_dims")
                        .and_then(|v| v.as_array())
                    {
                        for dim in result_dims {
                            if let Some(d) = dim.as_u64() {
                                max_bond_dim = max_bond_dim.max(d);
                            }
                        }
                    }
                }
                OpType::MpsTruncate => {
                    total_truncations += 1;
                    self.process_truncation(entry, &mut constraints)?;
                }
                OpType::QrPositive => {
                    self.process_qr(entry, &mut constraints)?;
                }
                _ => {
                    // Hash commitments for all other operations
                    if self.include_hash_commitments {
                        self.add_hash_constraints(entry, &mut constraints);
                    }
                }
            }
        }

        // Add trace chain hash constraint
        constraints.push(CircuitConstraint {
            kind: ConstraintKind::TraceChainHash,
            label: "Trace chain hash integrity".to_string(),
            source_seq: 0,
            data: ConstraintData::TraceChainHash {
                expected_hash: trace.chain_hash.clone(),
            },
        });

        // Public inputs include the trace hash
        public_inputs.insert(
            "trace_hash".to_string(),
            serde_json::Value::String(trace.chain_hash.clone()),
        );
        public_inputs.insert(
            "session_id".to_string(),
            serde_json::Value::String(trace.session_id.to_string()),
        );
        public_inputs.insert(
            "entry_count".to_string(),
            serde_json::json!(trace.entries.len()),
        );

        let summary = CircuitSummary {
            total_svd_ops: total_svd,
            total_mpo_ops: total_mpo,
            total_truncations,
            max_rank,
            max_bond_dim,
            total_constraints: constraints.len(),
        };

        Ok(CircuitInputs {
            session_id: trace.session_id.to_string(),
            trace_hash: trace.chain_hash.clone(),
            constraints,
            public_inputs,
            private_witnesses,
            trace_entry_count: trace.entries.len(),
            summary,
        })
    }

    /// Process an SVD entry: generate ordering and rank constraints.
    fn process_svd(
        &self,
        entry: &TraceEntry,
        constraints: &mut Vec<CircuitConstraint>,
        public_inputs: &mut HashMap<String, serde_json::Value>,
        _private_witnesses: &mut HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let seq = entry.seq;

        // Singular value ordering constraint
        if self.verify_sv_ordering {
            if let Some(svs) = entry.singular_values() {
                constraints.push(CircuitConstraint {
                    kind: ConstraintKind::SvOrdering,
                    label: format!("SVD[{seq}]: singular values non-negative and descending"),
                    source_seq: seq,
                    data: ConstraintData::SvOrdering {
                        count: svs.len(),
                        values: svs.clone(),
                    },
                });

                // Singular values are public inputs (verifier sees the spectrum)
                public_inputs.insert(
                    format!("svd_{seq}_sv"),
                    serde_json::json!(svs),
                );
            }
        }

        // Rank bound constraint
        if let (Some(rank), Some(chi_max)) = (
            entry.metrics.get("rank").and_then(|v| v.as_u64()),
            entry.param_i64("chi_max"),
        ) {
            constraints.push(CircuitConstraint {
                kind: ConstraintKind::RankBound,
                label: format!("SVD[{seq}]: rank ≤ chi_max"),
                source_seq: seq,
                data: ConstraintData::RankBound {
                    rank,
                    chi_max: chi_max as u64,
                },
            });
        }

        // Truncation error constraint
        if let Some(err) = entry.truncation_error() {
            constraints.push(CircuitConstraint {
                kind: ConstraintKind::TruncationError,
                label: format!("SVD[{seq}]: truncation error within tolerance"),
                source_seq: seq,
                data: ConstraintData::TruncationError {
                    declared_error: err,
                    tolerance: self.max_truncation_error,
                },
            });
        }

        // Hash commitments
        if self.include_hash_commitments {
            self.add_hash_constraints(entry, constraints);
        }

        Ok(())
    }

    /// Process an MPO × MPS application entry.
    fn process_mpo_apply(
        &self,
        entry: &TraceEntry,
        constraints: &mut Vec<CircuitConstraint>,
    ) -> Result<()> {
        let seq = entry.seq;

        // Hash commitments
        if self.include_hash_commitments {
            self.add_hash_constraints(entry, constraints);
        }

        // Bond dimension compatibility: result_dim = mps_dim * mpo_dim
        if let (Some(mps_dims), Some(mpo_dims), Some(result_dims)) = (
            entry.params.get("mps_bond_dims").and_then(|v| v.as_array()),
            entry.params.get("mpo_bond_dims").and_then(|v| v.as_array()),
            entry.params.get("result_bond_dims").and_then(|v| v.as_array()),
        ) {
            let n = mps_dims.len().min(mpo_dims.len()).min(result_dims.len());
            for i in 0..n {
                if let (Some(mps_d), Some(mpo_d), Some(res_d)) = (
                    mps_dims[i].as_u64(),
                    mpo_dims[i].as_u64(),
                    result_dims[i].as_u64(),
                ) {
                    let expected = mps_d * mpo_d;
                    constraints.push(CircuitConstraint {
                        kind: ConstraintKind::BondDimCompatibility,
                        label: format!(
                            "MPO[{seq}]: bond {i} result dim = mps_dim × mpo_dim ({mps_d}×{mpo_d}={expected})"
                        ),
                        source_seq: seq,
                        data: ConstraintData::BondDimCompatibility {
                            left_seq: seq,
                            right_seq: seq,
                            left_dim: expected,
                            right_dim: res_d,
                        },
                    });
                }
            }
        }

        Ok(())
    }

    /// Process an MPS truncation entry.
    fn process_truncation(
        &self,
        entry: &TraceEntry,
        constraints: &mut Vec<CircuitConstraint>,
    ) -> Result<()> {
        let seq = entry.seq;

        // Bond dimension bound after truncation
        if let Some(chi_max) = entry.param_i64("chi_max") {
            if let Some(after_dims) = entry.params.get("bond_dims_after")
                .and_then(|v| v.as_array())
            {
                for (i, dim) in after_dims.iter().enumerate() {
                    if let Some(d) = dim.as_u64() {
                        constraints.push(CircuitConstraint {
                            kind: ConstraintKind::RankBound,
                            label: format!("Truncate[{seq}]: bond {i} ≤ chi_max={chi_max}"),
                            source_seq: seq,
                            data: ConstraintData::RankBound {
                                rank: d,
                                chi_max: chi_max as u64,
                            },
                        });
                    }
                }
            }
        }

        if self.include_hash_commitments {
            self.add_hash_constraints(entry, constraints);
        }

        Ok(())
    }

    /// Process a QR decomposition entry.
    fn process_qr(
        &self,
        entry: &TraceEntry,
        constraints: &mut Vec<CircuitConstraint>,
    ) -> Result<()> {
        if self.include_hash_commitments {
            self.add_hash_constraints(entry, constraints);
        }
        Ok(())
    }

    /// Add hash commitment constraints for all inputs and outputs.
    fn add_hash_constraints(
        &self,
        entry: &TraceEntry,
        constraints: &mut Vec<CircuitConstraint>,
    ) {
        let seq = entry.seq;

        for (name, hash) in &entry.input_hashes {
            constraints.push(CircuitConstraint {
                kind: ConstraintKind::InputHash,
                label: format!("Op[{seq}]: input '{name}' hash commitment"),
                source_seq: seq,
                data: ConstraintData::HashCommitment {
                    name: format!("input_{name}"),
                    expected_hash: hash.clone(),
                },
            });
        }

        for (name, hash) in &entry.output_hashes {
            constraints.push(CircuitConstraint {
                kind: ConstraintKind::OutputHash,
                label: format!("Op[{seq}]: output '{name}' hash commitment"),
                source_seq: seq,
                data: ConstraintData::HashCommitment {
                    name: format!("output_{name}"),
                    expected_hash: hash.clone(),
                },
            });
        }
    }

    /// Serialize circuit inputs to JSON for the prover.
    pub fn to_json(inputs: &CircuitInputs) -> Result<String> {
        serde_json::to_string_pretty(inputs)
            .context("Failed to serialize circuit inputs")
    }

    /// Write circuit inputs to a file.
    pub fn write_to_file(inputs: &CircuitInputs, path: &std::path::Path) -> Result<()> {
        let json = Self::to_json(inputs)?;
        std::fs::write(path, json)
            .with_context(|| format!("Failed to write circuit inputs to {}", path.display()))?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_parser::TraceParser;

    fn sample_trace_json() -> &'static str {
        r#"{
            "trace_version": 1,
            "session_id": "12345678-1234-1234-1234-123456789abc",
            "digest": {},
            "entries": [
                {
                    "seq": 0,
                    "op": "svd_truncated",
                    "timestamp_ns": 1700000000000000000,
                    "duration_ns": 1000000,
                    "input_hashes": {"A": "aabbccdd"},
                    "output_hashes": {"U": "11223344", "S": "55667788", "Vh": "99aabbcc"},
                    "params": {"input_shape": [100, 100], "chi_max": 20, "cutoff": 1e-14},
                    "metrics": {
                        "truncation_error": 1e-8,
                        "rank": 20,
                        "original_rank": 100,
                        "singular_values": [5.0, 4.0, 3.0, 2.0, 1.0]
                    }
                },
                {
                    "seq": 1,
                    "op": "mpo_apply",
                    "timestamp_ns": 1700000001000000000,
                    "duration_ns": 2000000,
                    "input_hashes": {"mpo": "deadbeef", "mps": "cafebabe"},
                    "output_hashes": {"result": "12345678"},
                    "params": {
                        "L": 10,
                        "mps_bond_dims": [4, 8, 16, 16, 16, 16, 16, 8, 4],
                        "mpo_bond_dims": [3, 5, 5, 5, 5, 5, 5, 5, 3],
                        "result_bond_dims": [12, 40, 80, 80, 80, 80, 80, 40, 12]
                    },
                    "metrics": {}
                }
            ]
        }"#
    }

    #[test]
    fn test_build_circuit_from_trace() {
        let trace = TraceParser::parse_json_str(sample_trace_json()).unwrap();
        let builder = CircuitBuilder::new();
        let inputs = builder.build(&trace).unwrap();

        assert_eq!(inputs.trace_entry_count, 2);
        assert!(inputs.constraints.len() > 0);
        assert!(inputs.public_inputs.contains_key("trace_hash"));
        assert_eq!(inputs.summary.total_svd_ops, 1);
        assert_eq!(inputs.summary.total_mpo_ops, 1);
    }

    #[test]
    fn test_sv_ordering_constraint() {
        let trace = TraceParser::parse_json_str(sample_trace_json()).unwrap();
        let builder = CircuitBuilder::new();
        let inputs = builder.build(&trace).unwrap();

        let sv_constraints: Vec<_> = inputs.constraints.iter()
            .filter(|c| c.kind == ConstraintKind::SvOrdering)
            .collect();
        assert_eq!(sv_constraints.len(), 1);

        if let ConstraintData::SvOrdering { count, values } = &sv_constraints[0].data {
            assert_eq!(*count, 5);
            assert!((values[0] - 5.0).abs() < 1e-10);
            // Verify ordering
            for i in 1..values.len() {
                assert!(values[i] <= values[i - 1]);
            }
        } else {
            panic!("Expected SvOrdering constraint data");
        }
    }

    #[test]
    fn test_rank_bound_constraint() {
        let trace = TraceParser::parse_json_str(sample_trace_json()).unwrap();
        let builder = CircuitBuilder::new();
        let inputs = builder.build(&trace).unwrap();

        let rank_constraints: Vec<_> = inputs.constraints.iter()
            .filter(|c| c.kind == ConstraintKind::RankBound)
            .collect();
        assert!(rank_constraints.len() >= 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let trace = TraceParser::parse_json_str(sample_trace_json()).unwrap();
        let builder = CircuitBuilder::new();
        let inputs = builder.build(&trace).unwrap();

        let json = CircuitBuilder::to_json(&inputs).unwrap();
        let parsed: CircuitInputs = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.session_id, inputs.session_id);
        assert_eq!(parsed.constraints.len(), inputs.constraints.len());
    }
}

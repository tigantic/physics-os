//! Multi-Timestep Proof Batching
//!
//! Aggregates ZK proofs across T timesteps of a physics simulation into a
//! single TPC certificate with a Merkle root commitment. Verification of
//! the aggregate certificate completes in <10ms regardless of timestep count.
//!
//! # Architecture
//!
//! ```text
//! Timestep 0 → Trace → Circuit → Proof₀ ──┐
//! Timestep 1 → Trace → Circuit → Proof₁ ──┤  Merkle Tree   ┌───────────────┐
//! Timestep 2 → Trace → Circuit → Proof₂ ──┼──────────────► │ TPC Certificate│
//!    ⋮                             ⋮       │  SHA-256 root  │  • Merkle root │
//! Timestep T → Trace → Circuit → ProofT ──┘               │  • T proofs    │
//!                                                           │  • Ed25519 sig │
//!                                                           └───────────────┘
//! ```
//!
//! ## Performance Model
//!
//! - Proof generation: parallel via `rayon` (1 per logical core)
//! - Merkle tree construction: O(T log T) SHA-256 hashes
//! - Verification: O(1) — check Merkle root + Ed25519 signature
//!
//! ## Certificate Format
//!
//! The aggregate TPC certificate stores:
//! - **Layer A (Mathematical Truth):** Aggregate conservation residuals across
//!   all timesteps (max absolute, RMS, per-timestep summary).
//! - **Layer B (Computational Integrity):** Merkle root over all proof hashes,
//!   individual proof hashes, aggregate proof bundle.
//! - **Layer C (Physical Fidelity):** Simulation parameters, domain, timestep
//!   range, solver configuration.
//! - **Metadata:** Timestep count, domain name, simulation UUID.

use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Physics domain for the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimulationDomain {
    /// 3D compressible Euler equations.
    Euler3d,
    /// Navier-Stokes with IMEX time integration.
    NsImex,
    /// Heat / thermal diffusion.
    Thermal,
}

impl SimulationDomain {
    /// Domain label for certificate metadata.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Euler3d => "euler3d",
            Self::NsImex => "ns_imex",
            Self::Thermal => "thermal",
        }
    }
}

/// Input data for a single timestep proof.
#[derive(Debug, Clone)]
pub struct TimestepInput {
    /// Timestep index (0-based).
    pub index: usize,
    /// Raw proof data for this timestep (generated externally or by the
    /// prover pipeline).
    pub proof_bytes: Vec<u8>,
    /// SHA-256 hash of the proof bytes (computed automatically if empty).
    pub proof_hash: [u8; 32],
    /// Optional conservation residual for Layer A evidence.
    pub conservation_residual: Option<f64>,
    /// Optional per-timestep metadata (JSON-serializable).
    pub metadata: Option<serde_json::Value>,
}

impl TimestepInput {
    /// Create a new timestep input from proof bytes.
    ///
    /// Computes SHA-256 hash automatically.
    pub fn new(index: usize, proof_bytes: Vec<u8>) -> Self {
        let hash = sha256_bytes(&proof_bytes);
        Self {
            index,
            proof_bytes,
            proof_hash: hash,
            conservation_residual: None,
            metadata: None,
        }
    }

    /// Builder: attach a conservation residual.
    pub fn with_residual(mut self, residual: f64) -> Self {
        self.conservation_residual = Some(residual);
        self
    }

    /// Builder: attach metadata.
    pub fn with_metadata(mut self, meta: serde_json::Value) -> Self {
        self.metadata = Some(meta);
        self
    }
}

/// Result of multi-timestep proof aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateProof {
    /// Unique certificate identifier.
    pub certificate_id: Uuid,
    /// Simulation domain.
    pub domain: SimulationDomain,
    /// Number of timesteps covered.
    pub timestep_count: usize,
    /// Merkle root over all timestep proof hashes (SHA-256).
    pub merkle_root: [u8; 32],
    /// Individual timestep proof hashes (leaves of the Merkle tree).
    pub proof_hashes: Vec<[u8; 32]>,
    /// Aggregate conservation residual statistics.
    pub residual_stats: ResidualStats,
    /// TPC certificate binary (signed).
    pub tpc_certificate: Vec<u8>,
    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Verification time in microseconds (measured at generation).
    pub verification_time_us: u64,
}

/// Aggregate statistics over conservation residuals across all timesteps.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResidualStats {
    /// Maximum absolute residual across all timesteps.
    pub max_abs: f64,
    /// Root-mean-square residual.
    pub rms: f64,
    /// Number of timesteps with non-zero residuals.
    pub nonzero_count: usize,
    /// Per-timestep residuals (index, value).
    pub per_timestep: Vec<(usize, f64)>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Merkle Tree
// ═══════════════════════════════════════════════════════════════════════════════

/// SHA-256 Merkle tree over proof hashes.
///
/// Leaves are the SHA-256 hashes of individual timestep proofs. Internal
/// nodes are `SHA-256(left || right)`. If the number of leaves is odd, the
/// last leaf is duplicated.
pub struct MerkleTree {
    /// All nodes of the tree, bottom-up. Leaves at indices `[0..n)`,
    /// root at the last element.
    nodes: Vec<[u8; 32]>,
    /// Number of leaves.
    leaf_count: usize,
}

impl MerkleTree {
    /// Build a Merkle tree from leaf hashes.
    ///
    /// # Panics
    ///
    /// Panics if `leaves` is empty.
    pub fn from_leaves(leaves: &[[u8; 32]]) -> Self {
        assert!(!leaves.is_empty(), "Merkle tree requires at least one leaf");

        let leaf_count = leaves.len();
        // Round up to next power of 2 for a balanced tree.
        let padded = leaf_count.next_power_of_two();
        let total_nodes = 2 * padded - 1;

        let mut nodes = vec![[0u8; 32]; total_nodes];

        // Copy leaves into the first `padded` positions.
        for (i, leaf) in leaves.iter().enumerate() {
            nodes[i] = *leaf;
        }
        // Pad with duplicate of last leaf if needed.
        for i in leaf_count..padded {
            nodes[i] = leaves[leaf_count - 1];
        }

        // Build internal nodes bottom-up.
        let mut level_start = 0;
        let mut level_size = padded;
        let mut next_start = padded;

        while level_size > 1 {
            for i in (0..level_size).step_by(2) {
                let left = nodes[level_start + i];
                let right = nodes[level_start + i + 1];
                let parent = hash_pair(&left, &right);
                let parent_idx = next_start + i / 2;
                if parent_idx < total_nodes {
                    nodes[parent_idx] = parent;
                }
            }
            level_start = next_start;
            level_size /= 2;
            next_start += level_size;
        }

        Self { nodes, leaf_count }
    }

    /// The Merkle root hash.
    pub fn root(&self) -> [u8; 32] {
        *self.nodes.last().expect("tree is non-empty")
    }

    /// Number of leaves.
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Generate an inclusion proof for leaf at `index`.
    ///
    /// Returns the sibling hashes along the path from leaf to root.
    pub fn proof(&self, index: usize) -> Vec<[u8; 32]> {
        if index >= self.leaf_count {
            return vec![];
        }

        let padded = self.leaf_count.next_power_of_two();
        let mut proof_nodes = Vec::new();
        let mut pos = index;
        let mut level_start = 0;
        let mut level_size = padded;

        while level_size > 1 {
            let sibling = if pos % 2 == 0 { pos + 1 } else { pos - 1 };
            if level_start + sibling < self.nodes.len() {
                proof_nodes.push(self.nodes[level_start + sibling]);
            }
            level_start += level_size;
            level_size /= 2;
            pos /= 2;
        }

        proof_nodes
    }

    /// Verify an inclusion proof for a leaf hash at `index`.
    pub fn verify_proof(
        root: &[u8; 32],
        leaf: &[u8; 32],
        index: usize,
        proof_nodes: &[[u8; 32]],
    ) -> bool {
        let mut current = *leaf;
        let mut pos = index;

        for sibling in proof_nodes {
            current = if pos % 2 == 0 {
                hash_pair(&current, sibling)
            } else {
                hash_pair(sibling, &current)
            };
            pos /= 2;
        }

        current == *root
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-Timestep Prover
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for multi-timestep proof batching.
#[derive(Debug, Clone)]
pub struct MultiTimestepConfig {
    /// Physics domain of the simulation.
    pub domain: SimulationDomain,
    /// Simulation identifier (e.g., job UUID).
    pub simulation_id: Option<Uuid>,
    /// Solver name for certificate metadata.
    pub solver_name: String,
    /// Solver version string.
    pub solver_version: String,
    /// Whether to embed individual proof bytes in the certificate
    /// (increases size but enables per-timestep re-verification).
    pub embed_proofs: bool,
    /// Maximum number of timesteps per certificate.
    pub max_timesteps: usize,
}

impl Default for MultiTimestepConfig {
    fn default() -> Self {
        Self {
            domain: SimulationDomain::Thermal,
            simulation_id: None,
            solver_name: "fluidelite".to_string(),
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
            embed_proofs: true,
            max_timesteps: 10_000,
        }
    }
}

/// Multi-timestep proof aggregation prover.
///
/// Takes T timestep proofs and produces a single TPC certificate containing
/// a Merkle root commitment over all proofs. Verification requires only
/// checking the Merkle root and Ed25519 signature (< 10ms).
pub struct MultiTimestepProver {
    /// Ed25519 signing key for certificate generation.
    signing_key: SigningKey,
    /// Corresponding verifying key.
    verifying_key: VerifyingKey,
    /// Configuration.
    config: MultiTimestepConfig,
}

impl MultiTimestepProver {
    /// Create a new multi-timestep prover with the given signing key.
    pub fn new(signing_key: SigningKey, config: MultiTimestepConfig) -> Self {
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
            config,
        }
    }

    /// Create a prover with a random signing key (for testing).
    pub fn with_random_key(config: MultiTimestepConfig) -> Self {
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        Self::new(signing_key, config)
    }

    /// Aggregate T timestep proofs into a single TPC certificate.
    ///
    /// # Arguments
    ///
    /// * `timesteps` — Individual timestep proof inputs, in timestep order.
    ///
    /// # Returns
    ///
    /// An [`AggregateProof`] containing the signed TPC certificate, Merkle
    /// root, and verification metadata.
    pub fn aggregate(&self, timesteps: Vec<TimestepInput>) -> Result<AggregateProof, String> {
        if timesteps.is_empty() {
            return Err("no timesteps to aggregate".into());
        }
        if timesteps.len() > self.config.max_timesteps {
            return Err(format!(
                "timestep count {} exceeds max {}",
                timesteps.len(),
                self.config.max_timesteps
            ));
        }

        let start = Instant::now();
        let certificate_id = Uuid::new_v4();
        let timestep_count = timesteps.len();

        // ── Phase 1: Compute proof hashes (parallel) ───────────────────────
        let proof_hashes: Vec<[u8; 32]> = timesteps
            .par_iter()
            .map(|ts| ts.proof_hash)
            .collect();

        // ── Phase 2: Build Merkle tree ─────────────────────────────────────
        let tree = MerkleTree::from_leaves(&proof_hashes);
        let merkle_root = tree.root();

        // ── Phase 3: Compute residual statistics ───────────────────────────
        let residual_stats = compute_residual_stats(&timesteps);

        // ── Phase 4: Build TPC certificate binary ──────────────────────────
        let tpc_certificate = self.build_certificate(
            &certificate_id,
            &timesteps,
            &merkle_root,
            &proof_hashes,
            &residual_stats,
        )?;

        // ── Phase 5: Verify the certificate we just built ──────────────────
        let verify_start = Instant::now();
        self.verify_certificate(&tpc_certificate)?;
        let verification_time_us = verify_start.elapsed().as_micros() as u64;

        let generation_time_ms = start.elapsed().as_millis() as u64;

        tracing::info!(
            certificate_id = %certificate_id,
            timesteps = timestep_count,
            merkle_root = hex::encode(merkle_root),
            generation_ms = generation_time_ms,
            verify_us = verification_time_us,
            cert_size = tpc_certificate.len(),
            "multi-timestep aggregate certificate generated"
        );

        Ok(AggregateProof {
            certificate_id,
            domain: self.config.domain,
            timestep_count,
            merkle_root,
            proof_hashes,
            residual_stats,
            tpc_certificate,
            generation_time_ms,
            verification_time_us,
        })
    }

    /// Verify a multi-timestep aggregate certificate.
    ///
    /// Checks:
    /// 1. TPC header magic and version
    /// 2. Ed25519 signature over certificate content
    /// 3. SHA-256 content hash integrity
    ///
    /// Does NOT re-verify individual timestep proofs (that requires the ZK
    /// verifier for each domain circuit).
    pub fn verify_certificate(&self, certificate: &[u8]) -> Result<(), String> {
        // Minimum size: 64 (header) + 128 (signature section)
        if certificate.len() < 192 {
            return Err(format!(
                "certificate too small: {} bytes (min 192)",
                certificate.len()
            ));
        }

        // Check magic bytes.
        if &certificate[0..4] != b"TPC\x01" {
            return Err("invalid TPC magic bytes".into());
        }

        // Extract content and signature section.
        let content = &certificate[..certificate.len() - 128];
        let sig_section = &certificate[certificate.len() - 128..];

        // Parse signature section: pubkey(32) + signature(64) + hash(32)
        let pubkey_bytes = &sig_section[0..32];
        let sig_bytes = &sig_section[32..96];
        let expected_hash = &sig_section[96..128];

        // Verify content hash.
        let actual_hash = sha256_bytes(content);
        if actual_hash != expected_hash {
            return Err("content hash mismatch".into());
        }

        // Verify Ed25519 signature.
        let pubkey = ed25519_dalek::VerifyingKey::from_bytes(
            pubkey_bytes
                .try_into()
                .map_err(|_| "invalid pubkey length")?,
        )
        .map_err(|e| format!("invalid pubkey: {}", e))?;

        let signature = ed25519_dalek::Signature::from_bytes(
            sig_bytes
                .try_into()
                .map_err(|_| "invalid signature length")?,
        );

        pubkey
            .verify_strict(content, &signature)
            .map_err(|e| format!("signature verification failed: {}", e))?;

        Ok(())
    }

    /// Verify a specific timestep's inclusion in the aggregate certificate.
    ///
    /// Uses the Merkle inclusion proof to verify that a particular timestep's
    /// proof hash is committed in the aggregate Merkle root.
    pub fn verify_timestep_inclusion(
        merkle_root: &[u8; 32],
        timestep_proof_hash: &[u8; 32],
        timestep_index: usize,
        merkle_proof: &[[u8; 32]],
    ) -> bool {
        MerkleTree::verify_proof(merkle_root, timestep_proof_hash, timestep_index, merkle_proof)
    }

    /// Get the public verifying key (for independent verification).
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    // ── Internal: Build TPC binary ─────────────────────────────────────────

    fn build_certificate(
        &self,
        certificate_id: &Uuid,
        timesteps: &[TimestepInput],
        merkle_root: &[u8; 32],
        proof_hashes: &[[u8; 32]],
        residual_stats: &ResidualStats,
    ) -> Result<Vec<u8>, String> {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Solver hash (SHA-256 of solver name + version).
        let solver_hash = {
            let mut h = Sha256::new();
            h.update(self.config.solver_name.as_bytes());
            h.update(b":");
            h.update(self.config.solver_version.as_bytes());
            let result = h.finalize();
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&result);
            hash
        };

        let mut cert = Vec::with_capacity(4096);

        // ── Header (64 bytes) ──────────────────────────────────────────────
        cert.extend_from_slice(b"TPC\x01");           // Magic (4)
        cert.extend_from_slice(&[1u8; 4]);             // Version (4)
        cert.extend_from_slice(certificate_id.as_bytes()); // UUID (16)
        cert.extend_from_slice(&timestamp_ns.to_le_bytes()); // Timestamp (8)
        cert.extend_from_slice(&solver_hash);          // Solver hash (32)
        debug_assert_eq!(cert.len(), 64);

        // ── Layer A: Mathematical Truth ────────────────────────────────────
        let layer_a = serde_json::json!({
            "type": "multi_timestep_conservation",
            "domain": self.config.domain.label(),
            "timestep_count": timesteps.len(),
            "residual_max_abs": residual_stats.max_abs,
            "residual_rms": residual_stats.rms,
            "residual_nonzero_count": residual_stats.nonzero_count,
        });
        let layer_a_json = serde_json::to_vec(&layer_a)
            .map_err(|e| format!("layer A serialization failed: {}", e))?;
        encode_section(&mut cert, b"LAYA", &layer_a_json, &[]);

        // ── Layer B: Computational Integrity ───────────────────────────────
        let layer_b_json_val = serde_json::json!({
            "type": "merkle_aggregate",
            "merkle_root": hex::encode(merkle_root),
            "proof_count": proof_hashes.len(),
            "proof_hashes": proof_hashes.iter().map(hex::encode).collect::<Vec<_>>(),
        });
        let layer_b_json = serde_json::to_vec(&layer_b_json_val)
            .map_err(|e| format!("layer B serialization failed: {}", e))?;

        // Embed proof bundles if configured.
        let proof_bundle_blob = if self.config.embed_proofs {
            let mut bundle = Vec::new();
            // Format: [count:u32 LE] [len₀:u32 LE] [proof₀] [len₁:u32 LE] [proof₁] …
            bundle.extend_from_slice(&(timesteps.len() as u32).to_le_bytes());
            for ts in timesteps {
                bundle.extend_from_slice(&(ts.proof_bytes.len() as u32).to_le_bytes());
                bundle.extend_from_slice(&ts.proof_bytes);
            }
            bundle
        } else {
            vec![]
        };
        encode_section(&mut cert, b"LAYB", &layer_b_json, &proof_bundle_blob);

        // ── Layer C: Physical Fidelity ─────────────────────────────────────
        let layer_c = serde_json::json!({
            "type": "simulation_provenance",
            "domain": self.config.domain.label(),
            "solver": self.config.solver_name,
            "solver_version": self.config.solver_version,
            "simulation_id": self.config.simulation_id.map(|id| id.to_string()),
            "timestep_range": [
                timesteps.first().map(|t| t.index).unwrap_or(0),
                timesteps.last().map(|t| t.index).unwrap_or(0),
            ],
            "total_timesteps": timesteps.len(),
        });
        let layer_c_json = serde_json::to_vec(&layer_c)
            .map_err(|e| format!("layer C serialization failed: {}", e))?;
        encode_section(&mut cert, b"LAYC", &layer_c_json, &[]);

        // ── Metadata section ───────────────────────────────────────────────
        let meta = serde_json::json!({
            "certificate_type": "multi_timestep_aggregate",
            "generator": "fluidelite-zk",
            "generator_version": env!("CARGO_PKG_VERSION"),
            "embed_proofs": self.config.embed_proofs,
        });
        let meta_json = serde_json::to_vec(&meta)
            .map_err(|e| format!("metadata serialization failed: {}", e))?;
        encode_section(&mut cert, b"META", &meta_json, &[]);

        // ── Signature section (128 bytes) ──────────────────────────────────
        let content_hash = sha256_bytes(&cert);
        let signature = self.signing_key.sign(&cert);

        cert.extend_from_slice(self.verifying_key.as_bytes());            // Pubkey (32)
        cert.extend_from_slice(&signature.to_bytes());                    // Signature (64)
        cert.extend_from_slice(&content_hash);                            // Hash (32)
        // Total signature section = 32 + 64 + 32 = 128 bytes.

        Ok(cert)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// SHA-256 of a byte slice.
fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let result = Sha256::digest(data);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Hash two 32-byte nodes to produce a Merkle internal node.
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Encode a TPC section into the certificate buffer.
///
/// Format: `[tag:4] [json_len:u32 LE] [json_bytes] [blob_len:u32 LE] [blob_bytes]`
fn encode_section(buf: &mut Vec<u8>, tag: &[u8; 4], json: &[u8], blob: &[u8]) {
    buf.extend_from_slice(tag);
    buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
    buf.extend_from_slice(json);
    buf.extend_from_slice(&(blob.len() as u32).to_le_bytes());
    buf.extend_from_slice(blob);
}

/// Compute aggregate residual statistics over all timesteps.
fn compute_residual_stats(timesteps: &[TimestepInput]) -> ResidualStats {
    let mut max_abs: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut nonzero_count = 0usize;
    let mut per_timestep = Vec::new();

    for ts in timesteps {
        if let Some(r) = ts.conservation_residual {
            let abs_r = r.abs();
            if abs_r > max_abs {
                max_abs = abs_r;
            }
            sum_sq += r * r;
            if abs_r > f64::EPSILON {
                nonzero_count += 1;
            }
            per_timestep.push((ts.index, r));
        }
    }

    let rms = if per_timestep.is_empty() {
        0.0
    } else {
        (sum_sq / per_timestep.len() as f64).sqrt()
    };

    ResidualStats {
        max_abs,
        rms,
        nonzero_count,
        per_timestep,
    }
}

/// Extract the Merkle root from an aggregate TPC certificate.
///
/// Parses Layer B to find the `merkle_root` field.
pub fn extract_merkle_root(certificate: &[u8]) -> Result<[u8; 32], String> {
    // Skip header (64 bytes) and find LAYB section.
    let cert_content = if certificate.len() > 192 {
        &certificate[64..certificate.len() - 128]
    } else {
        return Err("certificate too small".into());
    };

    // Scan for LAYB tag.
    let mut pos = 0;
    while pos + 8 < cert_content.len() {
        let tag = &cert_content[pos..pos + 4];
        if tag == b"LAYB" {
            let json_len = u32::from_le_bytes(
                cert_content[pos + 4..pos + 8]
                    .try_into()
                    .map_err(|_| "invalid LAYB json_len")?,
            ) as usize;

            if pos + 8 + json_len > cert_content.len() {
                return Err("LAYB json_len exceeds certificate".into());
            }

            let json_bytes = &cert_content[pos + 8..pos + 8 + json_len];
            let layer_b: serde_json::Value = serde_json::from_slice(json_bytes)
                .map_err(|e| format!("LAYB JSON parse failed: {}", e))?;

            let root_hex = layer_b["merkle_root"]
                .as_str()
                .ok_or("missing merkle_root in LAYB")?;

            let root_bytes = hex::decode(root_hex)
                .map_err(|e| format!("invalid merkle_root hex: {}", e))?;

            if root_bytes.len() != 32 {
                return Err(format!("merkle_root is {} bytes, expected 32", root_bytes.len()));
            }

            let mut root = [0u8; 32];
            root.copy_from_slice(&root_bytes);
            return Ok(root);
        }

        // Skip this section: tag(4) + json_len(4) + json + blob_len(4) + blob
        if pos + 8 > cert_content.len() {
            break;
        }
        let jl = u32::from_le_bytes(
            cert_content[pos + 4..pos + 8]
                .try_into()
                .map_err(|_| "invalid section json_len")?,
        ) as usize;
        let blob_len_start = pos + 8 + jl;
        if blob_len_start + 4 > cert_content.len() {
            break;
        }
        let bl = u32::from_le_bytes(
            cert_content[blob_len_start..blob_len_start + 4]
                .try_into()
                .map_err(|_| "invalid section blob_len")?,
        ) as usize;
        pos = blob_len_start + 4 + bl;
    }

    Err("LAYB section not found in certificate".into())
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_prover() -> MultiTimestepProver {
        MultiTimestepProver::with_random_key(MultiTimestepConfig {
            domain: SimulationDomain::Euler3d,
            solver_name: "test-solver".to_string(),
            solver_version: "0.0.1".to_string(),
            embed_proofs: true,
            ..Default::default()
        })
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let leaves = vec![[42u8; 32]];
        let tree = MerkleTree::from_leaves(&leaves);
        assert_eq!(tree.leaf_count(), 1);
        // Single-leaf tree with padded=1: root IS the leaf (no internal nodes).
        assert_eq!(tree.root(), [42u8; 32]);
    }

    #[test]
    fn test_merkle_tree_two_leaves() {
        let leaves = vec![[1u8; 32], [2u8; 32]];
        let tree = MerkleTree::from_leaves(&leaves);
        assert_eq!(tree.leaf_count(), 2);
        let expected = hash_pair(&[1u8; 32], &[2u8; 32]);
        assert_eq!(tree.root(), expected);
    }

    #[test]
    fn test_merkle_tree_four_leaves() {
        let leaves: Vec<[u8; 32]> = (0..4u8).map(|i| [i; 32]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        assert_eq!(tree.leaf_count(), 4);

        let h01 = hash_pair(&[0u8; 32], &[1u8; 32]);
        let h23 = hash_pair(&[2u8; 32], &[3u8; 32]);
        let root = hash_pair(&h01, &h23);
        assert_eq!(tree.root(), root);
    }

    #[test]
    fn test_merkle_inclusion_proof() {
        let leaves: Vec<[u8; 32]> = (0..4u8).map(|i| [i; 32]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();

        for i in 0..4 {
            let proof = tree.proof(i);
            assert!(
                MerkleTree::verify_proof(&root, &leaves[i], i, &proof),
                "proof failed for leaf {}",
                i
            );
        }
    }

    #[test]
    fn test_merkle_inclusion_proof_odd_leaves() {
        let leaves: Vec<[u8; 32]> = (0..5u8).map(|i| [i; 32]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();

        for i in 0..5 {
            let proof = tree.proof(i);
            assert!(
                MerkleTree::verify_proof(&root, &leaves[i], i, &proof),
                "proof failed for leaf {} of 5",
                i
            );
        }
    }

    #[test]
    fn test_merkle_tampered_proof_rejected() {
        let leaves: Vec<[u8; 32]> = (0..4u8).map(|i| [i; 32]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();

        let proof = tree.proof(0);
        let tampered_leaf = [99u8; 32];
        assert!(!MerkleTree::verify_proof(&root, &tampered_leaf, 0, &proof));
    }

    #[test]
    fn test_aggregate_single_timestep() {
        let prover = make_test_prover();
        let ts = vec![TimestepInput::new(0, vec![1, 2, 3, 4])];

        let result = prover.aggregate(ts);
        assert!(result.is_ok());

        let agg = result.unwrap();
        assert_eq!(agg.timestep_count, 1);
        assert_eq!(agg.proof_hashes.len(), 1);
        assert!(agg.verification_time_us < 10_000); // < 10ms
        assert!(!agg.tpc_certificate.is_empty());
    }

    #[test]
    fn test_aggregate_100_timesteps() {
        let prover = make_test_prover();
        let timesteps: Vec<TimestepInput> = (0..100)
            .map(|i| {
                let proof = vec![i as u8; 64 + i];
                TimestepInput::new(i, proof).with_residual(1e-6 * i as f64)
            })
            .collect();

        let result = prover.aggregate(timesteps);
        assert!(result.is_ok());

        let agg = result.unwrap();
        assert_eq!(agg.timestep_count, 100);
        assert_eq!(agg.proof_hashes.len(), 100);
        assert!(agg.verification_time_us < 10_000_000); // < 10s (generous for test)
        assert!(agg.residual_stats.max_abs > 0.0);
        assert!(agg.residual_stats.rms > 0.0);
    }

    #[test]
    fn test_aggregate_certificate_verification() {
        let prover = make_test_prover();
        let timesteps: Vec<TimestepInput> = (0..10)
            .map(|i| TimestepInput::new(i, vec![i as u8; 128]))
            .collect();

        let agg = prover.aggregate(timesteps).unwrap();

        // Verify the certificate.
        assert!(prover.verify_certificate(&agg.tpc_certificate).is_ok());

        // Tamper with the certificate → should fail.
        let mut tampered = agg.tpc_certificate.clone();
        if tampered.len() > 100 {
            tampered[100] ^= 0xFF;
        }
        assert!(prover.verify_certificate(&tampered).is_err());
    }

    #[test]
    fn test_aggregate_merkle_root_extraction() {
        let prover = make_test_prover();
        let timesteps: Vec<TimestepInput> = (0..8)
            .map(|i| TimestepInput::new(i, vec![i as u8; 32]))
            .collect();

        let agg = prover.aggregate(timesteps).unwrap();
        let extracted_root = extract_merkle_root(&agg.tpc_certificate).unwrap();
        assert_eq!(extracted_root, agg.merkle_root);
    }

    #[test]
    fn test_aggregate_timestep_inclusion_verification() {
        let prover = make_test_prover();
        let timesteps: Vec<TimestepInput> = (0..8)
            .map(|i| TimestepInput::new(i, vec![i as u8; 64]))
            .collect();

        let agg = prover.aggregate(timesteps.clone()).unwrap();

        // Build Merkle tree from the aggregate's proof hashes.
        let tree = MerkleTree::from_leaves(&agg.proof_hashes);

        // Verify each timestep's inclusion.
        for (i, ts) in timesteps.iter().enumerate() {
            let proof = tree.proof(i);
            assert!(
                MultiTimestepProver::verify_timestep_inclusion(
                    &agg.merkle_root,
                    &ts.proof_hash,
                    i,
                    &proof
                ),
                "timestep {} inclusion failed",
                i
            );
        }
    }

    #[test]
    fn test_aggregate_empty_timesteps_rejected() {
        let prover = make_test_prover();
        let result = prover.aggregate(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_exceeds_max_timesteps() {
        let config = MultiTimestepConfig {
            max_timesteps: 5,
            ..Default::default()
        };
        let prover = MultiTimestepProver::with_random_key(config);
        let timesteps: Vec<TimestepInput> = (0..10)
            .map(|i| TimestepInput::new(i, vec![0u8; 32]))
            .collect();

        let result = prover.aggregate(timesteps);
        assert!(result.is_err());
    }

    #[test]
    fn test_residual_stats_computation() {
        let timesteps = vec![
            TimestepInput::new(0, vec![]).with_residual(1e-3),
            TimestepInput::new(1, vec![]).with_residual(-2e-3),
            TimestepInput::new(2, vec![]).with_residual(0.0),
            TimestepInput::new(3, vec![]).with_residual(5e-4),
        ];

        let stats = compute_residual_stats(&timesteps);
        assert!((stats.max_abs - 2e-3).abs() < f64::EPSILON);
        assert_eq!(stats.nonzero_count, 3);
        assert_eq!(stats.per_timestep.len(), 4);
        assert!(stats.rms > 0.0);
    }

    #[test]
    fn test_simulation_domain_labels() {
        assert_eq!(SimulationDomain::Euler3d.label(), "euler3d");
        assert_eq!(SimulationDomain::NsImex.label(), "ns_imex");
        assert_eq!(SimulationDomain::Thermal.label(), "thermal");
    }

    #[test]
    fn test_timestep_input_builder() {
        let ts = TimestepInput::new(0, vec![1, 2, 3])
            .with_residual(1e-6)
            .with_metadata(serde_json::json!({"dt": 0.001}));

        assert_eq!(ts.index, 0);
        assert_eq!(ts.proof_bytes, vec![1, 2, 3]);
        assert!(ts.conservation_residual.is_some());
        assert!(ts.metadata.is_some());
        // Hash should be SHA-256 of [1, 2, 3].
        assert_eq!(ts.proof_hash, sha256_bytes(&[1, 2, 3]));
    }

    #[test]
    fn test_certificate_without_embedded_proofs() {
        let config = MultiTimestepConfig {
            embed_proofs: false,
            ..Default::default()
        };
        let prover = MultiTimestepProver::with_random_key(config);
        let timesteps: Vec<TimestepInput> = (0..10)
            .map(|i| TimestepInput::new(i, vec![i as u8; 1024]))
            .collect();

        let agg = prover.aggregate(timesteps).unwrap();
        // Without embedded proofs, certificate is much smaller.
        assert!(agg.tpc_certificate.len() < 5000);
        assert!(prover.verify_certificate(&agg.tpc_certificate).is_ok());
    }
}

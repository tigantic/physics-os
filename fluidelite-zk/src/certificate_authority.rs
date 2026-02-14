//! TPC Certificate Authority Service
//!
//! Microservice that receives proofs, validates them, signs TPC certificates
//! with Ed25519, stores them, and returns certificate IDs.
//!
//! # Architecture
//!
//! ```text
//! Client → POST /v1/certificates/issue
//!        → Validate proof (call prover /verify or local check)
//!        → Build TPC binary certificate (Layers A/B/C + metadata)
//!        → Sign with Ed25519 signing key
//!        → Store certificate (filesystem + optional on-chain registration)
//!        → Return certificate ID + status
//! ```
//!
//! # Endpoints
//!
//! - `POST /v1/certificates/issue`   — Issue a new certificate from proof data
//! - `GET  /v1/certificates/:id`     — Retrieve a certificate by UUID
//! - `POST /v1/certificates/verify`  — Verify a certificate's integrity
//! - `GET  /v1/certificates/stats`   — CA statistics
//! - `GET  /health`                  — Health check
//!
//! # Configuration
//!
//! Environment variables:
//! - `CA_SIGNING_KEY`    — Hex-encoded Ed25519 private key (32 bytes)
//! - `CA_STORAGE_DIR`    — Directory for certificate storage (default: ./certificates)
//! - `CA_PROVER_URL`     — URL of the prover service for proof validation
//! - `CA_LISTEN_ADDR`    — Listen address (default: 0.0.0.0:8444)
//! - `CA_API_KEY`        — Required API key for certificate issuance

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Physics simulation domain for certificate issuance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Domain {
    Thermal,
    Euler3d,
    NsImex,
    Fluidelite,
}

impl Domain {
    /// On-chain domain ID (matches TPCCertificateRegistry.sol)
    pub fn chain_id(&self) -> u8 {
        match self {
            Domain::Thermal => 0,
            Domain::Euler3d => 1,
            Domain::NsImex => 2,
            Domain::Fluidelite => 3,
        }
    }

    /// Display name for the domain
    pub fn name(&self) -> &'static str {
        match self {
            Domain::Thermal => "thermal",
            Domain::Euler3d => "euler3d",
            Domain::NsImex => "ns_imex",
            Domain::Fluidelite => "fluidelite",
        }
    }
}

/// Request to issue a new TPC certificate.
#[derive(Debug, Deserialize)]
pub struct IssueCertificateRequest {
    /// The domain of the physics simulation
    pub domain: Domain,
    /// Serialized proof bytes (hex-encoded or base64)
    pub proof: String,
    /// Public inputs for verification
    pub public_inputs: Vec<String>,
    /// Solver binary hash (SHA-256, hex)
    pub solver_hash: Option<String>,
    /// Additional metadata key-value pairs
    pub metadata: Option<HashMap<String, String>>,
}

/// Response from certificate issuance.
#[derive(Debug, Serialize)]
pub struct IssueCertificateResponse {
    /// UUID of the issued certificate
    pub certificate_id: String,
    /// SHA-256 hash of the certificate binary
    pub content_hash: String,
    /// Ed25519 public key of the CA signer (hex)
    pub signer_pubkey: String,
    /// Domain of the certificate
    pub domain: String,
    /// Certificate size in bytes
    pub size_bytes: usize,
    /// Issuance timestamp (UTC ISO 8601)
    pub issued_at: String,
    /// On-chain registration status
    pub on_chain_status: String,
}

/// Request to verify a certificate.
#[derive(Debug, Deserialize)]
pub struct VerifyCertificateRequest {
    /// Certificate bytes (hex-encoded or base64)
    pub certificate: Option<String>,
    /// Or lookup by certificate ID
    pub certificate_id: Option<String>,
}

/// Response from certificate verification.
#[derive(Debug, Serialize)]
pub struct VerifyCertificateResponse {
    pub valid: bool,
    pub hash_valid: bool,
    pub signature_valid: bool,
    pub signer_pubkey: String,
    pub certificate_id: String,
    pub domain: Option<String>,
    pub error: Option<String>,
}

/// Certificate Authority statistics.
#[derive(Debug, Serialize)]
pub struct CAStats {
    pub total_issued: u64,
    pub total_verified: u64,
    pub total_failed: u64,
    pub uptime_seconds: f64,
    pub signer_pubkey: String,
    pub storage_dir: String,
    pub certificates_by_domain: HashMap<String, u64>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Certificate Authority State
// ═══════════════════════════════════════════════════════════════════════════

/// Core certificate authority state, shared across handlers via `Arc`.
pub struct CertificateAuthority {
    /// Ed25519 signing key
    signing_key: SigningKey,
    /// Ed25519 verifying (public) key
    verifying_key: VerifyingKey,
    /// Storage directory for TPC binaries
    storage_dir: PathBuf,
    /// URL of the upstream prover for proof validation
    prover_url: Option<String>,
    /// API key for access control
    api_key: Option<String>,
    /// Statistics
    total_issued: AtomicU64,
    total_verified: AtomicU64,
    total_failed: AtomicU64,
    domain_counts: RwLock<HashMap<String, u64>>,
    /// Server start time
    start_time: Instant,
    /// In-memory index: certificate_id → (content_hash, domain, path)
    index: RwLock<HashMap<String, CertificateRecord>>,
}

/// Record in the in-memory certificate index.
#[derive(Debug, Clone)]
struct CertificateRecord {
    content_hash: String,
    domain: String,
    file_path: PathBuf,
    issued_at: String,
    size_bytes: usize,
}

impl CertificateAuthority {
    /// Create a new Certificate Authority from configuration.
    pub fn new(
        signing_key_bytes: &[u8; 32],
        storage_dir: PathBuf,
        prover_url: Option<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let signing_key = SigningKey::from_bytes(signing_key_bytes);
        let verifying_key = signing_key.verifying_key();

        // Ensure storage directory exists
        std::fs::create_dir_all(&storage_dir)
            .with_context(|| format!("Failed to create storage dir: {}", storage_dir.display()))?;

        // Load existing certificate index from storage
        let index = Self::load_index(&storage_dir)?;
        let total_issued = index.len() as u64;

        let mut domain_counts = HashMap::new();
        for record in index.values() {
            *domain_counts.entry(record.domain.clone()).or_insert(0u64) += 1;
        }

        Ok(Self {
            signing_key,
            verifying_key,
            storage_dir,
            prover_url,
            api_key,
            total_issued: AtomicU64::new(total_issued),
            total_verified: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            domain_counts: RwLock::new(domain_counts),
            start_time: Instant::now(),
            index: RwLock::new(index),
        })
    }

    /// Load certificate index from the storage directory by scanning .tpc files.
    fn load_index(storage_dir: &Path) -> Result<HashMap<String, CertificateRecord>> {
        let mut index = HashMap::new();

        if !storage_dir.exists() {
            return Ok(index);
        }

        for entry in std::fs::read_dir(storage_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("tpc") {
                continue;
            }

            // Extract UUID from filename: <uuid>.tpc
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };

            // Read just enough to get the header (metadata from filename + hash)
            let data = match std::fs::read(&path) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let content_hash = hex::encode(Sha256::digest(&data));
            let size_bytes = data.len();

            index.insert(stem.clone(), CertificateRecord {
                content_hash,
                domain: "unknown".to_string(), // Would need to parse header for real domain
                file_path: path,
                issued_at: String::new(),
                size_bytes,
            });
        }

        Ok(index)
    }

    /// Get the hex-encoded public key of the CA signer.
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.verifying_key.as_bytes())
    }

    /// Issue a new TPC certificate.
    pub async fn issue_certificate(
        &self,
        req: &IssueCertificateRequest,
    ) -> Result<IssueCertificateResponse> {
        // 1. Decode proof bytes
        let proof_bytes = hex::decode(&req.proof)
            .or_else(|_| {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.decode(&req.proof)
            })
            .context("Failed to decode proof (expected hex or base64)")?;

        // 2. Validate proof against upstream prover (if configured)
        if let Some(ref prover_url) = self.prover_url {
            self.validate_proof_upstream(prover_url, &req.domain, &proof_bytes, &req.public_inputs)
                .await
                .context("Proof validation failed")?;
        }

        // 3. Build TPC certificate binary
        let certificate_id = Uuid::new_v4();
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;

        let solver_hash = req.solver_hash.as_deref()
            .and_then(|h| hex::decode(h).ok())
            .unwrap_or_else(|| vec![0u8; 32]);

        let mut cert_data = Vec::new();

        // TPC Header (64 bytes)
        cert_data.extend_from_slice(b"TPC\x01");                    // magic
        cert_data.extend_from_slice(&1u32.to_le_bytes());           // version
        cert_data.extend_from_slice(certificate_id.as_bytes());     // UUID (16 bytes)
        cert_data.extend_from_slice(&timestamp_ns.to_le_bytes());   // timestamp (8 bytes)
        let mut solver_hash_bytes = [0u8; 32];
        solver_hash_bytes[..solver_hash.len().min(32)]
            .copy_from_slice(&solver_hash[..solver_hash.len().min(32)]);
        cert_data.extend_from_slice(&solver_hash_bytes);            // solver hash (32 bytes)

        // Layer A: Mathematical Truth
        let layer_a = serde_json::json!({
            "type": "mathematical_truth",
            "domain": req.domain.name(),
            "public_inputs": req.public_inputs,
            "proof_system": "halo2_kzg",
            "curve": "BN254",
        });
        let layer_a_json = serde_json::to_vec(&layer_a)?;
        cert_data.extend_from_slice(&(layer_a_json.len() as u32).to_le_bytes());
        cert_data.extend_from_slice(&layer_a_json);
        cert_data.extend_from_slice(&0u32.to_le_bytes()); // 0 blobs

        // Layer B: Computational Integrity
        let layer_b = serde_json::json!({
            "type": "computational_integrity",
            "proof_size_bytes": proof_bytes.len(),
            "proof_hash": hex::encode(Sha256::digest(&proof_bytes)),
        });
        let layer_b_json = serde_json::to_vec(&layer_b)?;
        cert_data.extend_from_slice(&(layer_b_json.len() as u32).to_le_bytes());
        cert_data.extend_from_slice(&layer_b_json);
        // Blob: the proof itself
        cert_data.extend_from_slice(&1u32.to_le_bytes()); // 1 blob
        let blob_name = b"proof";
        cert_data.extend_from_slice(&(blob_name.len() as u16).to_le_bytes());
        cert_data.extend_from_slice(blob_name);
        cert_data.extend_from_slice(&(proof_bytes.len() as u32).to_le_bytes());
        cert_data.extend_from_slice(&proof_bytes);

        // Layer C: Physical Fidelity
        let layer_c = serde_json::json!({
            "type": "physical_fidelity",
            "domain": req.domain.name(),
        });
        let layer_c_json = serde_json::to_vec(&layer_c)?;
        cert_data.extend_from_slice(&(layer_c_json.len() as u32).to_le_bytes());
        cert_data.extend_from_slice(&layer_c_json);
        cert_data.extend_from_slice(&0u32.to_le_bytes()); // 0 blobs

        // Metadata
        let metadata = serde_json::json!({
            "ca_version": "1.0.0",
            "issued_by": "fluidelite-ca",
            "metadata": req.metadata,
        });
        let metadata_json = serde_json::to_vec(&metadata)?;
        cert_data.extend_from_slice(&(metadata_json.len() as u32).to_le_bytes());
        cert_data.extend_from_slice(&metadata_json);
        cert_data.extend_from_slice(&0u32.to_le_bytes()); // 0 blobs

        // 4. Compute content hash and sign
        let content_hash = Sha256::digest(&cert_data);
        let signature = self.signing_key.sign(&content_hash);

        // Signature section (128 bytes): pubkey(32) + signature(64) + hash(32)
        cert_data.extend_from_slice(self.verifying_key.as_bytes());
        cert_data.extend_from_slice(&signature.to_bytes());
        cert_data.extend_from_slice(&content_hash);

        let content_hash_hex = hex::encode(content_hash);

        // 5. Store certificate
        let filename = format!("{}.tpc", certificate_id);
        let file_path = self.storage_dir.join(&filename);
        tokio::fs::write(&file_path, &cert_data)
            .await
            .with_context(|| format!("Failed to write certificate to {}", file_path.display()))?;

        let issued_at = chrono::Utc::now().to_rfc3339();

        // 6. Update index
        {
            let mut index = self.index.write().await;
            index.insert(certificate_id.to_string(), CertificateRecord {
                content_hash: content_hash_hex.clone(),
                domain: req.domain.name().to_string(),
                file_path: file_path.clone(),
                issued_at: issued_at.clone(),
                size_bytes: cert_data.len(),
            });
        }

        // 7. Update statistics
        self.total_issued.fetch_add(1, Ordering::Relaxed);
        {
            let mut counts = self.domain_counts.write().await;
            *counts.entry(req.domain.name().to_string()).or_insert(0) += 1;
        }

        Ok(IssueCertificateResponse {
            certificate_id: certificate_id.to_string(),
            content_hash: content_hash_hex,
            signer_pubkey: self.pubkey_hex(),
            domain: req.domain.name().to_string(),
            size_bytes: cert_data.len(),
            issued_at,
            on_chain_status: "pending".to_string(),
        })
    }

    /// Validate a proof against the upstream prover service.
    async fn validate_proof_upstream(
        &self,
        prover_url: &str,
        domain: &Domain,
        proof_bytes: &[u8],
        _public_inputs: &[String],
    ) -> Result<()> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/verify", prover_url))
            .json(&serde_json::json!({
                "domain": domain.name(),
                "proof": hex::encode(proof_bytes),
            }))
            .send()
            .await
            .context("Failed to contact prover for proof validation")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("Proof validation failed: {}", body);
        }

        Ok(())
    }

    /// Retrieve a certificate by ID.
    pub async fn get_certificate(&self, id: &str) -> Result<Vec<u8>> {
        let index = self.index.read().await;
        let record = index.get(id).context("Certificate not found")?;
        let data = tokio::fs::read(&record.file_path)
            .await
            .context("Failed to read certificate file")?;
        Ok(data)
    }

    /// Verify a certificate's integrity (hash + signature).
    pub async fn verify_certificate(
        &self,
        cert_data: &[u8],
    ) -> VerifyCertificateResponse {
        if cert_data.len() < 64 + 128 {
            self.total_failed.fetch_add(1, Ordering::Relaxed);
            return VerifyCertificateResponse {
                valid: false,
                hash_valid: false,
                signature_valid: false,
                signer_pubkey: String::new(),
                certificate_id: String::new(),
                domain: None,
                error: Some("Certificate too short".to_string()),
            };
        }

        // Check magic
        if &cert_data[..4] != b"TPC\x01" {
            self.total_failed.fetch_add(1, Ordering::Relaxed);
            return VerifyCertificateResponse {
                valid: false,
                hash_valid: false,
                signature_valid: false,
                signer_pubkey: String::new(),
                certificate_id: String::new(),
                domain: None,
                error: Some("Invalid TPC magic bytes".to_string()),
            };
        }

        // Extract certificate ID (bytes 8..24)
        let cert_id_bytes = &cert_data[8..24];
        let certificate_id = Uuid::from_bytes(
            cert_id_bytes.try_into().unwrap_or([0u8; 16])
        ).to_string();

        // Split content from signature section (last 128 bytes)
        let content = &cert_data[..cert_data.len() - 128];
        let sig_section = &cert_data[cert_data.len() - 128..];

        // Parse signature section
        let pubkey_bytes: [u8; 32] = sig_section[..32].try_into().unwrap_or([0u8; 32]);
        let sig_bytes: [u8; 64] = sig_section[32..96].try_into().unwrap_or([0u8; 64]);
        let stored_hash: [u8; 32] = sig_section[96..128].try_into().unwrap_or([0u8; 32]);

        let signer_pubkey = hex::encode(pubkey_bytes);

        // Verify content hash
        let computed_hash = Sha256::digest(content);
        let hash_valid = computed_hash.as_slice() == stored_hash;

        // Verify Ed25519 signature
        let signature_valid = if let Ok(vk) = VerifyingKey::from_bytes(&pubkey_bytes) {
            if let Ok(sig) = ed25519_dalek::Signature::from_bytes(&sig_bytes) {
                vk.verify_strict(&computed_hash, &sig).is_ok()
            } else {
                false
            }
        } else {
            false
        };

        let valid = hash_valid && signature_valid;

        if valid {
            self.total_verified.fetch_add(1, Ordering::Relaxed);
        } else {
            self.total_failed.fetch_add(1, Ordering::Relaxed);
        }

        VerifyCertificateResponse {
            valid,
            hash_valid,
            signature_valid,
            signer_pubkey,
            certificate_id,
            domain: None,
            error: if valid { None } else { Some("Verification failed".to_string()) },
        }
    }

    /// Get CA statistics.
    pub async fn get_stats(&self) -> CAStats {
        let domain_counts = self.domain_counts.read().await.clone();

        CAStats {
            total_issued: self.total_issued.load(Ordering::Relaxed),
            total_verified: self.total_verified.load(Ordering::Relaxed),
            total_failed: self.total_failed.load(Ordering::Relaxed),
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            signer_pubkey: self.pubkey_hex(),
            storage_dir: self.storage_dir.display().to_string(),
            certificates_by_domain: domain_counts,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_ca() -> (CertificateAuthority, TempDir) {
        let dir = TempDir::new().unwrap();
        let key_bytes = [42u8; 32]; // deterministic test key
        let ca = CertificateAuthority::new(
            &key_bytes,
            dir.path().to_path_buf(),
            None,
            None,
        ).unwrap();
        (ca, dir)
    }

    #[tokio::test]
    async fn test_issue_and_verify_certificate() {
        let (ca, _dir) = test_ca();

        let req = IssueCertificateRequest {
            domain: Domain::Thermal,
            proof: hex::encode(vec![0xde, 0xad, 0xbe, 0xef]),
            public_inputs: vec!["0x31".to_string()],
            solver_hash: None,
            metadata: None,
        };

        let resp = ca.issue_certificate(&req).await.unwrap();

        assert!(!resp.certificate_id.is_empty());
        assert!(!resp.content_hash.is_empty());
        assert_eq!(resp.domain, "thermal");
        assert!(resp.size_bytes > 64); // At least header + sig section

        // Retrieve the certificate
        let cert_data = ca.get_certificate(&resp.certificate_id).await.unwrap();
        assert_eq!(cert_data.len(), resp.size_bytes);

        // Verify the certificate
        let verify_resp = ca.verify_certificate(&cert_data).await;
        assert!(verify_resp.valid, "Certificate should be valid");
        assert!(verify_resp.hash_valid, "Hash should be valid");
        assert!(verify_resp.signature_valid, "Signature should be valid");
        assert_eq!(verify_resp.certificate_id, resp.certificate_id);
    }

    #[tokio::test]
    async fn test_verify_tampered_certificate() {
        let (ca, _dir) = test_ca();

        let req = IssueCertificateRequest {
            domain: Domain::Euler3d,
            proof: hex::encode(vec![0x01, 0x02, 0x03]),
            public_inputs: vec![],
            solver_hash: None,
            metadata: None,
        };

        let resp = ca.issue_certificate(&req).await.unwrap();
        let mut cert_data = ca.get_certificate(&resp.certificate_id).await.unwrap();

        // Tamper with a byte in the content
        cert_data[20] ^= 0xFF;

        let verify_resp = ca.verify_certificate(&cert_data).await;
        assert!(!verify_resp.valid, "Tampered certificate should be invalid");
        assert!(!verify_resp.hash_valid, "Hash should fail for tampered content");
    }

    #[tokio::test]
    async fn test_verify_too_short() {
        let (ca, _dir) = test_ca();
        let verify_resp = ca.verify_certificate(&[0u8; 10]).await;
        assert!(!verify_resp.valid);
        assert!(verify_resp.error.unwrap().contains("too short"));
    }

    #[tokio::test]
    async fn test_verify_invalid_magic() {
        let (ca, _dir) = test_ca();
        let data = vec![0u8; 256]; // All zeros, wrong magic
        let verify_resp = ca.verify_certificate(&data).await;
        assert!(!verify_resp.valid);
        assert!(verify_resp.error.unwrap().contains("magic"));
    }

    #[tokio::test]
    async fn test_stats_increment() {
        let (ca, _dir) = test_ca();

        let req = IssueCertificateRequest {
            domain: Domain::NsImex,
            proof: hex::encode(vec![0xAB]),
            public_inputs: vec![],
            solver_hash: None,
            metadata: None,
        };

        ca.issue_certificate(&req).await.unwrap();
        ca.issue_certificate(&IssueCertificateRequest {
            domain: Domain::Thermal,
            proof: hex::encode(vec![0xCD]),
            public_inputs: vec![],
            solver_hash: None,
            metadata: None,
        }).await.unwrap();

        let stats = ca.get_stats().await;
        assert_eq!(stats.total_issued, 2);
        assert_eq!(stats.certificates_by_domain.get("ns_imex"), Some(&1));
        assert_eq!(stats.certificates_by_domain.get("thermal"), Some(&1));
    }

    #[tokio::test]
    async fn test_multiple_domains() {
        let (ca, _dir) = test_ca();

        for domain in [Domain::Thermal, Domain::Euler3d, Domain::NsImex, Domain::Fluidelite] {
            let req = IssueCertificateRequest {
                domain,
                proof: hex::encode(vec![domain.chain_id()]),
                public_inputs: vec![],
                solver_hash: None,
                metadata: None,
            };
            let resp = ca.issue_certificate(&req).await.unwrap();
            assert_eq!(resp.domain, domain.name());
        }

        let stats = ca.get_stats().await;
        assert_eq!(stats.total_issued, 4);
    }

    #[test]
    fn test_domain_chain_ids() {
        assert_eq!(Domain::Thermal.chain_id(), 0);
        assert_eq!(Domain::Euler3d.chain_id(), 1);
        assert_eq!(Domain::NsImex.chain_id(), 2);
        assert_eq!(Domain::Fluidelite.chain_id(), 3);
    }

    #[test]
    fn test_pubkey_hex() {
        let key_bytes = [42u8; 32];
        let dir = TempDir::new().unwrap();
        let ca = CertificateAuthority::new(
            &key_bytes,
            dir.path().to_path_buf(),
            None,
            None,
        ).unwrap();
        let pubkey = ca.pubkey_hex();
        assert_eq!(pubkey.len(), 64); // 32 bytes = 64 hex chars
    }
}

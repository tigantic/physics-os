//! Certificate Writer — generates .tpc binary certificates from Rust.
//!
//! This mirrors the Python `tpc.format` module's binary layout so that
//! certificates can be produced from either Python or Rust.

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature, Verifier};
use sha2::{Digest, Sha256};
use std::io::{Cursor, Read};
use std::path::Path;
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════
// Constants (must match Python tpc/constants.py)
// ═══════════════════════════════════════════════════════════════════════════

/// Magic bytes identifying a TPC file.
pub const TPC_MAGIC: &[u8; 4] = b"TPC\x01";

/// Current TPC format version.
pub const TPC_VERSION: u32 = 1;

/// Fixed header size in bytes.
pub const TPC_HEADER_SIZE: usize = 64;

/// Ed25519 public key size.
pub const PUBLIC_KEY_SIZE: usize = 32;

/// Ed25519 signature size.
pub const SIGNATURE_SIZE: usize = 64;

/// SHA-256 hash size.
pub const HASH_SIZE: usize = 32;

/// Total signature section size.
pub const SIGNATURE_SECTION_SIZE: usize = PUBLIC_KEY_SIZE + SIGNATURE_SIZE + HASH_SIZE;

/// Maximum certificate size (256 MB).
pub const MAX_CERTIFICATE_SIZE: usize = 256 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════
// Header
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed 64-byte TPC certificate header.
#[derive(Debug, Clone)]
pub struct TpcHeader {
    /// Format version.
    pub version: u32,
    /// Unique certificate identifier.
    pub certificate_id: Uuid,
    /// Timestamp in nanoseconds since UNIX epoch.
    pub timestamp_ns: i64,
    /// SHA-256 of the solver binary/source.
    pub solver_hash: [u8; HASH_SIZE],
}

impl TpcHeader {
    /// Create a new header with current timestamp.
    pub fn new() -> Self {
        Self {
            version: TPC_VERSION,
            certificate_id: Uuid::new_v4(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as i64,
            solver_hash: [0u8; HASH_SIZE],
        }
    }

    /// Serialize to exactly 64 bytes.
    pub fn pack(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(TPC_HEADER_SIZE);
        buf.extend_from_slice(TPC_MAGIC);
        buf.write_u32::<LittleEndian>(self.version).unwrap();
        buf.extend_from_slice(self.certificate_id.as_bytes());
        buf.write_i64::<LittleEndian>(self.timestamp_ns).unwrap();
        buf.extend_from_slice(&self.solver_hash);
        assert_eq!(buf.len(), TPC_HEADER_SIZE);
        buf
    }

    /// Deserialize from 64 bytes.
    pub fn unpack(data: &[u8]) -> Result<Self> {
        if data.len() < TPC_HEADER_SIZE {
            bail!("Header too short: {} < {}", data.len(), TPC_HEADER_SIZE);
        }

        let mut cursor = Cursor::new(data);

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).context("Read magic")?;
        if &magic != TPC_MAGIC {
            bail!("Invalid TPC magic: {:?}", magic);
        }

        let version = cursor.read_u32::<LittleEndian>().context("Read version")?;
        let mut uuid_bytes = [0u8; 16];
        cursor.read_exact(&mut uuid_bytes).context("Read UUID")?;
        let certificate_id = Uuid::from_bytes(uuid_bytes);
        let timestamp_ns = cursor.read_i64::<LittleEndian>().context("Read timestamp")?;
        let mut solver_hash = [0u8; HASH_SIZE];
        cursor.read_exact(&mut solver_hash).context("Read solver hash")?;

        Ok(Self {
            version,
            certificate_id,
            timestamp_ns,
            solver_hash,
        })
    }
}

impl Default for TpcHeader {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section Encoding (matches Python _encode_section / _decode_section)
// ═══════════════════════════════════════════════════════════════════════════

/// Encode a JSON dict + optional binary blobs into the TPC section format.
pub fn encode_section(json_data: &serde_json::Value, blobs: &[(&str, &[u8])]) -> Result<Vec<u8>> {
    let json_bytes = serde_json::to_vec(json_data).context("Serialize section JSON")?;

    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(json_bytes.len() as u32)?;
    buf.extend_from_slice(&json_bytes);

    buf.write_u32::<LittleEndian>(blobs.len() as u32)?;
    for (name, blob) in blobs {
        let name_bytes = name.as_bytes();
        buf.write_u16::<LittleEndian>(name_bytes.len() as u16)?;
        buf.extend_from_slice(name_bytes);
        buf.write_u32::<LittleEndian>(blob.len() as u32)?;
        buf.extend_from_slice(blob);
    }

    Ok(buf)
}

/// Decode a TPC section. Returns (json_dict, blobs, bytes_consumed).
pub fn decode_section(data: &[u8], offset: usize) -> Result<(serde_json::Value, Vec<(String, Vec<u8>)>, usize)> {
    let mut pos = offset;

    if pos + 4 > data.len() {
        bail!("Truncated section at offset {pos}");
    }
    let json_len = (&data[pos..pos + 4]).read_u32::<LittleEndian>()? as usize;
    pos += 4;

    if pos + json_len > data.len() {
        bail!("Truncated JSON at offset {pos}");
    }
    let json_val: serde_json::Value = serde_json::from_slice(&data[pos..pos + json_len])
        .context("Parse section JSON")?;
    pos += json_len;

    if pos + 4 > data.len() {
        bail!("Truncated blob count at offset {pos}");
    }
    let blob_count = (&data[pos..pos + 4]).read_u32::<LittleEndian>()? as usize;
    pos += 4;

    let mut blobs = Vec::with_capacity(blob_count);
    for _ in 0..blob_count {
        if pos + 2 > data.len() {
            bail!("Truncated blob name length at offset {pos}");
        }
        let name_len = (&data[pos..pos + 2]).read_u16::<LittleEndian>()? as usize;
        pos += 2;

        if pos + name_len > data.len() {
            bail!("Truncated blob name at offset {pos}");
        }
        let name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .context("Invalid blob name UTF-8")?;
        pos += name_len;

        if pos + 4 > data.len() {
            bail!("Truncated blob length at offset {pos}");
        }
        let blob_len = (&data[pos..pos + 4]).read_u32::<LittleEndian>()? as usize;
        pos += 4;

        if pos + blob_len > data.len() {
            bail!("Truncated blob data at offset {pos}");
        }
        blobs.push((name, data[pos..pos + blob_len].to_vec()));
        pos += blob_len;
    }

    Ok((json_val, blobs, pos))
}

// ═══════════════════════════════════════════════════════════════════════════
// Certificate Writer
// ═══════════════════════════════════════════════════════════════════════════

/// Writes .tpc certificate files from Rust.
pub struct CertificateWriter {
    header: TpcHeader,
    layer_a_json: serde_json::Value,
    layer_a_blobs: Vec<(String, Vec<u8>)>,
    layer_b_json: serde_json::Value,
    layer_b_blobs: Vec<(String, Vec<u8>)>,
    layer_c_json: serde_json::Value,
    layer_c_blobs: Vec<(String, Vec<u8>)>,
    metadata_json: serde_json::Value,
}

impl CertificateWriter {
    /// Create a new certificate writer with a fresh header.
    pub fn new() -> Self {
        Self {
            header: TpcHeader::new(),
            layer_a_json: serde_json::json!({}),
            layer_a_blobs: Vec::new(),
            layer_b_json: serde_json::json!({}),
            layer_b_blobs: Vec::new(),
            layer_c_json: serde_json::json!({}),
            layer_c_blobs: Vec::new(),
            metadata_json: serde_json::json!({}),
        }
    }

    /// Set the certificate header.
    pub fn with_header(mut self, header: TpcHeader) -> Self {
        self.header = header;
        self
    }

    /// Set Layer A (Mathematical Truth).
    pub fn with_layer_a(mut self, json: serde_json::Value, blobs: Vec<(String, Vec<u8>)>) -> Self {
        self.layer_a_json = json;
        self.layer_a_blobs = blobs;
        self
    }

    /// Set Layer B (Computational Integrity).
    pub fn with_layer_b(mut self, json: serde_json::Value, blobs: Vec<(String, Vec<u8>)>) -> Self {
        self.layer_b_json = json;
        self.layer_b_blobs = blobs;
        self
    }

    /// Set Layer C (Physical Fidelity).
    pub fn with_layer_c(mut self, json: serde_json::Value, blobs: Vec<(String, Vec<u8>)>) -> Self {
        self.layer_c_json = json;
        self.layer_c_blobs = blobs;
        self
    }

    /// Set metadata.
    pub fn with_metadata(mut self, json: serde_json::Value) -> Self {
        self.metadata_json = json;
        self
    }

    /// Build the unsigned certificate bytes.
    fn build_content(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();

        // Header (64 bytes)
        buf.extend_from_slice(&self.header.pack());

        // Layer A
        let a_blob_refs: Vec<(&str, &[u8])> = self.layer_a_blobs.iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        buf.extend_from_slice(&encode_section(&self.layer_a_json, &a_blob_refs)?);

        // Layer B
        let b_blob_refs: Vec<(&str, &[u8])> = self.layer_b_blobs.iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        buf.extend_from_slice(&encode_section(&self.layer_b_json, &b_blob_refs)?);

        // Layer C
        let c_blob_refs: Vec<(&str, &[u8])> = self.layer_c_blobs.iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        buf.extend_from_slice(&encode_section(&self.layer_c_json, &c_blob_refs)?);

        // Metadata
        buf.extend_from_slice(&encode_section(&self.metadata_json, &[])?);

        Ok(buf)
    }

    /// Build the certificate with an Ed25519 signature.
    pub fn build_signed(self, signing_key: &SigningKey) -> Result<Vec<u8>> {
        let content = self.build_content()?;

        // Hash the content
        let content_hash = Sha256::digest(&content);

        // Sign
        let signature = signing_key.sign(&content_hash);
        let verifying_key = signing_key.verifying_key();

        // Append signature section (128 bytes)
        let mut result = content;
        result.extend_from_slice(verifying_key.as_bytes());
        result.extend_from_slice(&signature.to_bytes());
        result.extend_from_slice(&content_hash);

        if result.len() > MAX_CERTIFICATE_SIZE {
            bail!(
                "Certificate too large: {} > {}",
                result.len(),
                MAX_CERTIFICATE_SIZE
            );
        }

        Ok(result)
    }

    /// Build an unsigned certificate (zero-key signature section).
    pub fn build_unsigned(self) -> Result<Vec<u8>> {
        let content = self.build_content()?;

        let content_hash = Sha256::digest(&content);

        let mut result = content;
        result.extend_from_slice(&[0u8; PUBLIC_KEY_SIZE]);   // zero public key
        result.extend_from_slice(&[0u8; SIGNATURE_SIZE]);    // zero signature
        result.extend_from_slice(&content_hash);

        Ok(result)
    }

    /// Write the signed certificate to a file.
    pub fn write_signed(self, path: &Path, signing_key: &SigningKey) -> Result<()> {
        let data = self.build_signed(signing_key)?;
        std::fs::write(path, &data)
            .with_context(|| format!("Failed to write certificate to {}", path.display()))?;
        tracing::info!(
            path = %path.display(),
            size = data.len(),
            "Certificate written"
        );
        Ok(())
    }

    /// Write an unsigned certificate to a file.
    pub fn write_unsigned(self, path: &Path) -> Result<()> {
        let data = self.build_unsigned()?;
        std::fs::write(path, &data)
            .with_context(|| format!("Failed to write certificate to {}", path.display()))?;
        Ok(())
    }
}

impl Default for CertificateWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Certificate Verifier (lightweight, for use in proof_bridge)
// ═══════════════════════════════════════════════════════════════════════════

/// Verify a .tpc certificate's structural integrity and signature.
pub fn verify_certificate(data: &[u8]) -> Result<CertificateVerification> {
    if data.len() < TPC_HEADER_SIZE + SIGNATURE_SECTION_SIZE {
        bail!("Certificate too short: {} bytes", data.len());
    }
    if data.len() > MAX_CERTIFICATE_SIZE {
        bail!("Certificate too large: {} bytes", data.len());
    }

    // Parse header
    let header = TpcHeader::unpack(&data[..TPC_HEADER_SIZE])?;

    // The signature section is the last SIGNATURE_SECTION_SIZE bytes
    let sig_offset = data.len() - SIGNATURE_SECTION_SIZE;
    let content = &data[..sig_offset];

    let pub_key_bytes = &data[sig_offset..sig_offset + PUBLIC_KEY_SIZE];
    let sig_bytes = &data[sig_offset + PUBLIC_KEY_SIZE..sig_offset + PUBLIC_KEY_SIZE + SIGNATURE_SIZE];
    let stored_hash = &data[sig_offset + PUBLIC_KEY_SIZE + SIGNATURE_SIZE..];

    // Verify content hash
    let computed_hash = Sha256::digest(content);
    let hash_valid = computed_hash.as_slice() == stored_hash;

    // Verify signature (skip for unsigned/zero-key certificates)
    let is_unsigned = pub_key_bytes.iter().all(|&b| b == 0);
    let signature_valid = if is_unsigned {
        true // Unsigned certificate
    } else {
        let pub_key = VerifyingKey::from_bytes(
            pub_key_bytes.try_into().context("Invalid public key length")?
        ).context("Invalid Ed25519 public key")?;

        let signature = Signature::from_bytes(
            sig_bytes.try_into().context("Invalid signature length")?
        );

        pub_key.verify(&computed_hash, &signature).is_ok()
    };

    Ok(CertificateVerification {
        header,
        hash_valid,
        signature_valid,
        is_unsigned,
        certificate_size: data.len(),
        content_hash: hex::encode(computed_hash),
    })
}

/// Result of certificate verification.
#[derive(Debug)]
pub struct CertificateVerification {
    pub header: TpcHeader,
    pub hash_valid: bool,
    pub signature_valid: bool,
    pub is_unsigned: bool,
    pub certificate_size: usize,
    pub content_hash: String,
}

impl CertificateVerification {
    /// Whether the certificate passes all checks.
    pub fn is_valid(&self) -> bool {
        self.hash_valid && self.signature_valid
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = TpcHeader::new();
        let packed = header.pack();
        assert_eq!(packed.len(), TPC_HEADER_SIZE);

        let unpacked = TpcHeader::unpack(&packed).unwrap();
        assert_eq!(unpacked.version, header.version);
        assert_eq!(unpacked.certificate_id, header.certificate_id);
        assert_eq!(unpacked.timestamp_ns, header.timestamp_ns);
    }

    #[test]
    fn test_section_roundtrip() {
        let json = serde_json::json!({
            "proof_system": "lean4",
            "theorems": [],
            "coverage": "partial"
        });
        let blobs = vec![("proof_objects", b"hello world" as &[u8])];

        let encoded = encode_section(&json, &blobs).unwrap();
        let (decoded_json, decoded_blobs, _) = decode_section(&encoded, 0).unwrap();

        assert_eq!(decoded_json["proof_system"], "lean4");
        assert_eq!(decoded_blobs.len(), 1);
        assert_eq!(decoded_blobs[0].0, "proof_objects");
        assert_eq!(decoded_blobs[0].1, b"hello world");
    }

    #[test]
    fn test_unsigned_certificate() {
        let writer = CertificateWriter::new()
            .with_layer_a(serde_json::json!({"proof_system": "lean4"}), vec![])
            .with_layer_b(serde_json::json!({"proof_system": "stark"}), vec![])
            .with_layer_c(serde_json::json!({"benchmarks": []}), vec![])
            .with_metadata(serde_json::json!({"domain": "cfd", "solver": "euler3d"}));

        let data = writer.build_unsigned().unwrap();
        assert!(data.len() > TPC_HEADER_SIZE + SIGNATURE_SECTION_SIZE);

        let verification = verify_certificate(&data).unwrap();
        assert!(verification.is_valid());
        assert!(verification.is_unsigned);
        assert!(verification.hash_valid);
    }

    #[test]
    fn test_signed_certificate() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);

        let writer = CertificateWriter::new()
            .with_layer_a(serde_json::json!({"proof_system": "lean4"}), vec![])
            .with_layer_b(serde_json::json!({"proof_system": "halo2"}), vec![])
            .with_layer_c(serde_json::json!({"benchmarks": []}), vec![])
            .with_metadata(serde_json::json!({"domain": "cfd"}));

        let data = writer.build_signed(&signing_key).unwrap();

        let verification = verify_certificate(&data).unwrap();
        assert!(verification.is_valid());
        assert!(!verification.is_unsigned);
        assert!(verification.hash_valid);
        assert!(verification.signature_valid);
    }

    #[test]
    fn test_tampered_certificate_fails() {
        let writer = CertificateWriter::new()
            .with_layer_a(serde_json::json!({}), vec![])
            .with_layer_b(serde_json::json!({}), vec![])
            .with_layer_c(serde_json::json!({}), vec![])
            .with_metadata(serde_json::json!({}));

        let mut data = writer.build_unsigned().unwrap();

        // Tamper with a byte in the content
        if data.len() > TPC_HEADER_SIZE + 10 {
            data[TPC_HEADER_SIZE + 5] ^= 0xFF;
        }

        let verification = verify_certificate(&data).unwrap();
        assert!(!verification.hash_valid);
        assert!(!verification.is_valid());
    }
}

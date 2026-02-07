//! TPC certificate parser and verifier logic.
//!
//! This is the core verification module. It parses the .tpc binary format,
//! verifies structural integrity, hash commitments, and Ed25519 signatures.

use byteorder::{LittleEndian, ReadBytesExt};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Cursor, Read};
use uuid::Uuid;

// ═══════════════════════════════════════════════════════════════════════════
// Constants (must match Python tpc/constants.py and Rust proof_bridge)
// ═══════════════════════════════════════════════════════════════════════════

const TPC_MAGIC: &[u8; 4] = b"TPC\x01";
const TPC_VERSION: u32 = 1;
const TPC_HEADER_SIZE: usize = 64;
const PUBLIC_KEY_SIZE: usize = 32;
const SIGNATURE_SIZE: usize = 64;
const HASH_SIZE: usize = 32;
const SIG_SECTION_SIZE: usize = PUBLIC_KEY_SIZE + SIGNATURE_SIZE + HASH_SIZE;
const MAX_CERTIFICATE_SIZE: usize = 256 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════
// Parsed Header
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct TpcHeader {
    pub version: u32,
    pub certificate_id: Uuid,
    pub timestamp_ns: i64,
    pub solver_hash: [u8; HASH_SIZE],
}

impl TpcHeader {
    fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < TPC_HEADER_SIZE {
            return Err(format!("Header too short: {} < {}", data.len(), TPC_HEADER_SIZE));
        }

        let mut cursor = Cursor::new(data);

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| format!("Read magic: {e}"))?;
        if &magic != TPC_MAGIC {
            return Err(format!("Invalid TPC magic: {:02x?} (expected {:02x?})", magic, TPC_MAGIC));
        }

        let version = cursor.read_u32::<LittleEndian>().map_err(|e| format!("Read version: {e}"))?;
        if version != TPC_VERSION {
            return Err(format!("Unsupported version: {version} (expected {TPC_VERSION})"));
        }

        let mut uuid_bytes = [0u8; 16];
        cursor.read_exact(&mut uuid_bytes).map_err(|e| format!("Read UUID: {e}"))?;
        let certificate_id = Uuid::from_bytes(uuid_bytes);

        let timestamp_ns = cursor.read_i64::<LittleEndian>().map_err(|e| format!("Read timestamp: {e}"))?;

        let mut solver_hash = [0u8; HASH_SIZE];
        cursor.read_exact(&mut solver_hash).map_err(|e| format!("Read solver hash: {e}"))?;

        Ok(Self {
            version,
            certificate_id,
            timestamp_ns,
            solver_hash,
        })
    }

    pub fn timestamp_s(&self) -> f64 {
        self.timestamp_ns as f64 / 1_000_000_000.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Section Parser
// ═══════════════════════════════════════════════════════════════════════════

fn parse_section(data: &[u8], offset: usize) -> Result<(serde_json::Value, HashMap<String, Vec<u8>>, usize), String> {
    let mut pos = offset;

    if pos + 4 > data.len() {
        return Err(format!("Truncated section JSON length at offset {pos}"));
    }
    let json_len = (&data[pos..pos + 4]).read_u32::<LittleEndian>()
        .map_err(|e| format!("Read JSON length: {e}"))? as usize;
    pos += 4;

    if pos + json_len > data.len() {
        return Err(format!("Truncated section JSON at offset {pos}, need {json_len} bytes"));
    }
    let json_val: serde_json::Value = serde_json::from_slice(&data[pos..pos + json_len])
        .map_err(|e| format!("Parse section JSON: {e}"))?;
    pos += json_len;

    if pos + 4 > data.len() {
        return Err(format!("Truncated blob count at offset {pos}"));
    }
    let blob_count = (&data[pos..pos + 4]).read_u32::<LittleEndian>()
        .map_err(|e| format!("Read blob count: {e}"))? as usize;
    pos += 4;

    let mut blobs = HashMap::new();
    for i in 0..blob_count {
        if pos + 2 > data.len() {
            return Err(format!("Truncated blob {i} name length at offset {pos}"));
        }
        let name_len = (&data[pos..pos + 2]).read_u16::<LittleEndian>()
            .map_err(|e| format!("Read blob {i} name length: {e}"))? as usize;
        pos += 2;

        if pos + name_len > data.len() {
            return Err(format!("Truncated blob {i} name at offset {pos}"));
        }
        let name = String::from_utf8(data[pos..pos + name_len].to_vec())
            .map_err(|e| format!("Blob {i} name UTF-8: {e}"))?;
        pos += name_len;

        if pos + 4 > data.len() {
            return Err(format!("Truncated blob '{name}' data length at offset {pos}"));
        }
        let blob_len = (&data[pos..pos + 4]).read_u32::<LittleEndian>()
            .map_err(|e| format!("Read blob '{name}' length: {e}"))? as usize;
        pos += 4;

        if pos + blob_len > data.len() {
            return Err(format!("Truncated blob '{name}' data at offset {pos}"));
        }
        blobs.insert(name, data[pos..pos + blob_len].to_vec());
        pos += blob_len;
    }

    Ok((json_val, blobs, pos))
}

// ═══════════════════════════════════════════════════════════════════════════
// Verification Result
// ═══════════════════════════════════════════════════════════════════════════

/// Full verification result.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub valid: bool,
    pub certificate_id: String,
    pub timestamp: String,
    pub version: u32,

    pub header_valid: bool,
    pub structure_valid: bool,
    pub hash_valid: bool,
    pub signature_valid: bool,
    pub is_unsigned: bool,

    pub layer_a_summary: String,
    pub layer_b_summary: String,
    pub layer_c_summary: String,
    pub metadata_summary: String,

    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub certificate_size: usize,
}

impl VerifyResult {
    fn new() -> Self {
        Self {
            valid: false,
            certificate_id: String::new(),
            timestamp: String::new(),
            version: 0,
            header_valid: false,
            structure_valid: false,
            hash_valid: false,
            signature_valid: false,
            is_unsigned: false,
            layer_a_summary: String::new(),
            layer_b_summary: String::new(),
            layer_c_summary: String::new(),
            metadata_summary: String::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            certificate_size: 0,
        }
    }

    pub fn to_text(&self, elapsed_s: f64) -> String {
        let verdict = if self.valid { "VALID ✅" } else { "INVALID ❌" };
        let sig_status = if self.is_unsigned {
            "UNSIGNED"
        } else if self.signature_valid {
            "VERIFIED ✅"
        } else {
            "FAILED ❌"
        };

        let mut lines = vec![
            String::new(),
            "TRUSTLESS PHYSICS VERIFICATION REPORT".to_string(),
            "═".repeat(50),
            format!("Certificate ID:  {}", self.certificate_id),
            format!("Version:         {}", self.version),
            format!("Timestamp:       {}", self.timestamp),
            format!("Size:            {} bytes", self.certificate_size),
            String::new(),
            "Structural Checks".to_string(),
            format!("  Header:        {}", if self.header_valid { "✅" } else { "❌" }),
            format!("  Structure:     {}", if self.structure_valid { "✅" } else { "❌" }),
            format!("  Hash:          {}", if self.hash_valid { "✅" } else { "❌" }),
            format!("  Signature:     {sig_status}"),
            String::new(),
            format!("Layer A: {}", self.layer_a_summary),
            format!("Layer B: {}", self.layer_b_summary),
            format!("Layer C: {}", self.layer_c_summary),
            format!("Metadata: {}", self.metadata_summary),
            String::new(),
            format!("VERDICT: {verdict}"),
            format!("Verification time: {elapsed_s:.3}s"),
        ];

        if !self.errors.is_empty() {
            lines.push(String::new());
            lines.push("ERRORS:".to_string());
            for e in &self.errors {
                lines.push(format!("  ✗ {e}"));
            }
        }

        if !self.warnings.is_empty() {
            lines.push(String::new());
            lines.push("WARNINGS:".to_string());
            for w in &self.warnings {
                lines.push(format!("  ⚠ {w}"));
            }
        }

        lines.join("\n")
    }

    pub fn to_json(&self, elapsed_s: f64) -> String {
        let json = serde_json::json!({
            "valid": self.valid,
            "certificate_id": self.certificate_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "header_valid": self.header_valid,
            "structure_valid": self.structure_valid,
            "hash_valid": self.hash_valid,
            "signature_valid": self.signature_valid,
            "is_unsigned": self.is_unsigned,
            "layer_a": self.layer_a_summary,
            "layer_b": self.layer_b_summary,
            "layer_c": self.layer_c_summary,
            "metadata": self.metadata_summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "certificate_size": self.certificate_size,
            "verification_time_s": elapsed_s,
        });
        serde_json::to_string_pretty(&json).unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Inspect Result
// ═══════════════════════════════════════════════════════════════════════════

/// Lightweight inspection result (no verification).
#[derive(Debug, Clone)]
pub struct InspectResult {
    pub certificate_id: String,
    pub version: u32,
    pub timestamp: String,
    pub solver_hash: String,
    pub size: usize,
    pub is_signed: bool,
    pub layer_a: serde_json::Value,
    pub layer_b: serde_json::Value,
    pub layer_c: serde_json::Value,
    pub metadata: serde_json::Value,
}

impl InspectResult {
    pub fn to_text(&self) -> String {
        let lines = vec![
            String::new(),
            "TRUSTLESS PHYSICS CERTIFICATE".to_string(),
            "═".repeat(50),
            format!("Certificate ID:  {}", self.certificate_id),
            format!("Version:         {}", self.version),
            format!("Timestamp:       {}", self.timestamp),
            format!("Solver hash:     {}", &self.solver_hash[..32]),
            format!("Size:            {} bytes", self.size),
            format!("Signed:          {}", self.is_signed),
            String::new(),
            "Layer A (Mathematical Truth):".to_string(),
            format!("  {}", serde_json::to_string_pretty(&self.layer_a).unwrap_or_default()),
            String::new(),
            "Layer B (Computational Integrity):".to_string(),
            format!("  {}", serde_json::to_string_pretty(&self.layer_b).unwrap_or_default()),
            String::new(),
            "Layer C (Physical Fidelity):".to_string(),
            format!("  {}", serde_json::to_string_pretty(&self.layer_c).unwrap_or_default()),
            String::new(),
            "Metadata:".to_string(),
            format!("  {}", serde_json::to_string_pretty(&self.metadata).unwrap_or_default()),
        ];

        lines.join("\n")
    }

    pub fn to_json(&self) -> String {
        let json = serde_json::json!({
            "certificate_id": self.certificate_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "solver_hash": self.solver_hash,
            "size": self.size,
            "is_signed": self.is_signed,
            "layer_a": self.layer_a,
            "layer_b": self.layer_b,
            "layer_c": self.layer_c,
            "metadata": self.metadata,
        });
        serde_json::to_string_pretty(&json).unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verifier
// ═══════════════════════════════════════════════════════════════════════════

/// Standalone TPC certificate verifier.
pub struct TpcVerifier;

impl TpcVerifier {
    pub fn new() -> Self {
        Self
    }

    /// Full verification of a .tpc certificate.
    pub fn verify(&self, data: &[u8], check_signature: bool) -> VerifyResult {
        let mut result = VerifyResult::new();
        result.certificate_size = data.len();

        // Size checks
        if data.len() < TPC_HEADER_SIZE + SIG_SECTION_SIZE {
            result.errors.push(format!(
                "Certificate too short: {} bytes (minimum {})",
                data.len(),
                TPC_HEADER_SIZE + SIG_SECTION_SIZE
            ));
            return result;
        }
        if data.len() > MAX_CERTIFICATE_SIZE {
            result.errors.push(format!(
                "Certificate too large: {} bytes (maximum {})",
                data.len(),
                MAX_CERTIFICATE_SIZE
            ));
            return result;
        }

        // Parse header
        let header = match TpcHeader::parse(data) {
            Ok(h) => h,
            Err(e) => {
                result.errors.push(format!("Header parse failed: {e}"));
                return result;
            }
        };
        result.header_valid = true;
        result.version = header.version;
        result.certificate_id = header.certificate_id.to_string();

        // Format timestamp
        let ts_s = header.timestamp_s();
        result.timestamp = format_timestamp(ts_s);

        // Parse sections
        let sig_offset = data.len() - SIG_SECTION_SIZE;
        let content = &data[..sig_offset];

        let mut offset = TPC_HEADER_SIZE;

        // Layer A
        let layer_a_json = match parse_section(data, offset) {
            Ok((json, _blobs, next_offset)) => {
                offset = next_offset;
                json
            }
            Err(e) => {
                result.errors.push(format!("Layer A parse failed: {e}"));
                return result;
            }
        };

        // Layer B
        let layer_b_json = match parse_section(data, offset) {
            Ok((json, _blobs, next_offset)) => {
                offset = next_offset;
                json
            }
            Err(e) => {
                result.errors.push(format!("Layer B parse failed: {e}"));
                return result;
            }
        };

        // Layer C
        let layer_c_json = match parse_section(data, offset) {
            Ok((json, _blobs, next_offset)) => {
                offset = next_offset;
                json
            }
            Err(e) => {
                result.errors.push(format!("Layer C parse failed: {e}"));
                return result;
            }
        };

        // Metadata
        let metadata_json = match parse_section(data, offset) {
            Ok((json, _blobs, next_offset)) => {
                offset = next_offset;
                json
            }
            Err(e) => {
                result.errors.push(format!("Metadata parse failed: {e}"));
                return result;
            }
        };

        // Verify offset aligns with signature section
        if offset != sig_offset {
            result.warnings.push(format!(
                "Content/signature boundary mismatch: sections end at {offset}, signature at {sig_offset}"
            ));
        }

        result.structure_valid = true;

        // Summarize layers
        result.layer_a_summary = summarize_layer_a(&layer_a_json);
        result.layer_b_summary = summarize_layer_b(&layer_b_json);
        result.layer_c_summary = summarize_layer_c(&layer_c_json);
        result.metadata_summary = summarize_metadata(&metadata_json);

        // Verify content hash
        let pub_key_bytes = &data[sig_offset..sig_offset + PUBLIC_KEY_SIZE];
        let sig_bytes = &data[sig_offset + PUBLIC_KEY_SIZE..sig_offset + PUBLIC_KEY_SIZE + SIGNATURE_SIZE];
        let stored_hash = &data[sig_offset + PUBLIC_KEY_SIZE + SIGNATURE_SIZE..sig_offset + SIG_SECTION_SIZE];

        let computed_hash = Sha256::digest(content);
        result.hash_valid = computed_hash.as_slice() == stored_hash;

        if !result.hash_valid {
            result.errors.push("Content hash mismatch — certificate may be tampered".to_string());
        }

        // Signature verification
        result.is_unsigned = pub_key_bytes.iter().all(|&b| b == 0);

        if result.is_unsigned {
            result.signature_valid = true;
            result.warnings.push("Certificate is unsigned (self-attested)".to_string());
        } else if check_signature {
            match verify_ed25519(pub_key_bytes, sig_bytes, &computed_hash) {
                Ok(valid) => {
                    result.signature_valid = valid;
                    if !valid {
                        result.errors.push("Ed25519 signature verification failed".to_string());
                    }
                }
                Err(e) => {
                    result.errors.push(format!("Signature verification error: {e}"));
                }
            }
        } else {
            result.signature_valid = true;
            result.warnings.push("Signature verification skipped".to_string());
        }

        // Layer-specific validation
        validate_layer_a(&layer_a_json, &mut result);
        validate_layer_b(&layer_b_json, &mut result);
        validate_layer_c(&layer_c_json, &mut result);

        // Overall verdict
        result.valid = result.header_valid
            && result.structure_valid
            && result.hash_valid
            && result.signature_valid
            && result.errors.is_empty();

        result
    }

    /// Inspect a certificate without full verification.
    pub fn inspect(&self, data: &[u8]) -> Result<InspectResult, String> {
        if data.len() < TPC_HEADER_SIZE + SIG_SECTION_SIZE {
            return Err(format!("Certificate too short: {} bytes", data.len()));
        }

        let header = TpcHeader::parse(data)?;

        let mut offset = TPC_HEADER_SIZE;

        let (layer_a, _, next) = parse_section(data, offset)?;
        offset = next;
        let (layer_b, _, next) = parse_section(data, offset)?;
        offset = next;
        let (layer_c, _, next) = parse_section(data, offset)?;
        offset = next;
        let (metadata, _, _) = parse_section(data, offset)?;

        let sig_offset = data.len() - SIG_SECTION_SIZE;
        let pub_key_bytes = &data[sig_offset..sig_offset + PUBLIC_KEY_SIZE];
        let is_signed = !pub_key_bytes.iter().all(|&b| b == 0);

        Ok(InspectResult {
            certificate_id: header.certificate_id.to_string(),
            version: header.version,
            timestamp: format_timestamp(header.timestamp_s()),
            solver_hash: hex::encode(header.solver_hash),
            size: data.len(),
            is_signed,
            layer_a,
            layer_b,
            layer_c,
            metadata,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn verify_ed25519(pub_key_bytes: &[u8], sig_bytes: &[u8], message: &[u8]) -> Result<bool, String> {
    let pub_key_arr: [u8; 32] = pub_key_bytes
        .try_into()
        .map_err(|_| "Invalid public key length")?;

    let verifying_key = VerifyingKey::from_bytes(&pub_key_arr)
        .map_err(|e| format!("Invalid Ed25519 public key: {e}"))?;

    let sig_arr: [u8; 64] = sig_bytes
        .try_into()
        .map_err(|_| "Invalid signature length")?;

    let signature = Signature::from_bytes(&sig_arr);

    Ok(verifying_key.verify(message, &signature).is_ok())
}

fn format_timestamp(ts_s: f64) -> String {
    let secs = ts_s as i64;
    let _nanos = ((ts_s - secs as f64) * 1e9) as u32;

    // Simple UTC formatting without chrono dependency
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Simplified date calculation (valid from 1970-2100)
    let mut y = 1970i64;
    let mut remaining_days = days_since_epoch;

    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }

    let is_leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [
        31,
        if is_leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];

    let mut m = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md as i64 {
            m = i;
            break;
        }
        remaining_days -= md as i64;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m + 1,
        remaining_days + 1,
        hours,
        minutes,
        seconds
    )
}

fn summarize_layer_a(json: &serde_json::Value) -> String {
    let system = json["proof_system"].as_str().unwrap_or("unknown");
    let theorem_count = json["theorems"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let coverage = json["coverage"].as_str().unwrap_or("unknown");
    let pct = json["coverage_pct"].as_f64().unwrap_or(0.0);
    format!("{system}, {theorem_count} theorems, coverage={coverage} ({pct:.1}%)")
}

fn summarize_layer_b(json: &serde_json::Value) -> String {
    let system = json["proof_system"].as_str().unwrap_or("unknown");
    let proof_size = json["proof_size"].as_u64().unwrap_or(0);
    let constraints = json["circuit_constraints"].as_u64().unwrap_or(0);
    format!("{system}, proof={proof_size} bytes, constraints={constraints}")
}

fn summarize_layer_c(json: &serde_json::Value) -> String {
    let total = json["total_benchmarks"].as_u64()
        .or_else(|| json["benchmarks"].as_array().map(|a| a.len() as u64))
        .unwrap_or(0);
    let passed = json["pass_count"].as_u64().unwrap_or(0);
    let all_passed = json["all_passed"].as_bool().unwrap_or(false);
    let status = if all_passed { "✅" } else { "❌" };
    format!("{passed}/{total} benchmarks passed {status}")
}

fn summarize_metadata(json: &serde_json::Value) -> String {
    let domain = json["domain"].as_str().unwrap_or("unknown");
    let solver = json["solver"].as_str().unwrap_or("unknown");
    format!("domain={domain}, solver={solver}")
}

fn validate_layer_a(json: &serde_json::Value, result: &mut VerifyResult) {
    let system = json["proof_system"].as_str().unwrap_or("none");
    if system == "none" {
        result.warnings.push("Layer A: No formal proof system declared".to_string());
    }

    if let Some(theorems) = json["theorems"].as_array() {
        for (i, t) in theorems.iter().enumerate() {
            if t["name"].as_str().unwrap_or("").is_empty() {
                result.errors.push(format!("Layer A: Theorem {i} has empty name"));
            }
        }
    }
}

fn validate_layer_b(json: &serde_json::Value, result: &mut VerifyResult) {
    let system = json["proof_system"].as_str().unwrap_or("none");
    if system == "none" {
        result.warnings.push("Layer B: No ZK proof system declared".to_string());
    }

    let proof_size = json["proof_size"].as_u64().unwrap_or(0);
    if system != "none" && proof_size == 0 {
        result.warnings.push("Layer B: ZK proof system declared but proof is empty".to_string());
    }
}

fn validate_layer_c(json: &serde_json::Value, result: &mut VerifyResult) {
    if let Some(benchmarks) = json["benchmarks"].as_array() {
        if benchmarks.is_empty() {
            result.warnings.push("Layer C: No benchmarks provided".to_string());
        }

        for (i, b) in benchmarks.iter().enumerate() {
            let passed = b["passed"].as_bool().unwrap_or(false);
            if !passed {
                let name = b["name"].as_str().unwrap_or("unnamed");
                result.warnings.push(format!("Layer C: Benchmark '{name}' (#{i}) FAILED"));
            }
        }
    } else {
        result.warnings.push("Layer C: No benchmarks array found".to_string());
    }

    let all_passed = json["all_passed"].as_bool().unwrap_or(true);
    if !all_passed {
        let pass_count = json["pass_count"].as_u64().unwrap_or(0);
        let total = json["total_benchmarks"].as_u64().unwrap_or(0);
        result.errors.push(format!(
            "Layer C: Not all benchmarks passed ({pass_count}/{total})"
        ));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp() {
        // 2024-01-01 00:00:00 UTC
        let ts = 1704067200.0;
        let formatted = format_timestamp(ts);
        assert_eq!(formatted, "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_invalid_magic() {
        let data = vec![0u8; 256];
        let verifier = TpcVerifier::new();
        let result = verifier.verify(&data, false);
        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }
}

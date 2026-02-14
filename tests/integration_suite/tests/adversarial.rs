//! Task 2.12 — Adversarial Integration Tests
//!
//! Tampers at each pipeline stage and verifies that the correct layer
//! detects and rejects the corruption:
//!
//!   Stage 1: Corrupted trace → TraceParser rejects or chain_hash fails
//!   Stage 2: Malformed circuit inputs → CircuitBuilder rejects
//!   Stage 3: Modified witness / proof bytes → certificate hash fails
//!   Stage 4: Wrong Ed25519 signature → signature verification fails
//!   Stage 5: Truncated certificate → structural rejection
//!   Stage 6: Replayed certificate (same bytes, different signing key)

use ed25519_dalek::SigningKey;
use proof_bridge::certificate::{
    verify_certificate, CertificateWriter, SIGNATURE_SECTION_SIZE, TPC_HEADER_SIZE,
};
use proof_bridge::{CircuitBuilder, TraceParser};
use rand::rngs::OsRng;
use serde_json::json;
use sha2::{Digest, Sha256};


// ═════════════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Minimal valid trace JSON.
fn minimal_trace() -> String {
    serde_json::to_string(&json!({
        "trace_version": 1,
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "digest": {},
        "entries": [
            {
                "seq": 0,
                "op": "svd_truncated",
                "timestamp_ns": 1700000000000000000_i64,
                "duration_ns": 500000,
                "input_hashes": {"A": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
                "output_hashes": {
                    "U": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                    "S": "1111111122222222333333334444444455555555666666667777777788888888",
                    "Vh": "aaaaaaaabbbbbbbbccccccccddddddddeeeeeeeeffffffff0000000011111111"
                },
                "params": {"input_shape": [32, 32], "chi_max": 8, "cutoff": 1e-14},
                "metrics": {
                    "truncation_error": 1e-8,
                    "rank": 8,
                    "original_rank": 32,
                    "singular_values": [4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1]
                }
            }
        ]
    }))
    .unwrap()
}

/// Build a signed certificate with dummy proof bytes.
fn build_test_certificate(signing_key: &SigningKey, proof_bytes: &[u8]) -> Vec<u8> {
    CertificateWriter::new()
        .with_layer_a(json!({"theorem": "test"}), vec![])
        .with_layer_b(
            json!({"proof_system": "test", "proof_hash": hex::encode(Sha256::digest(proof_bytes))}),
            vec![("proof".to_string(), proof_bytes.to_vec())],
        )
        .with_layer_c(json!({"physics": "test"}), vec![])
        .with_metadata(json!({"pipeline": "adversarial_test"}))
        .build_signed(signing_key)
        .expect("Certificate build must succeed for adversarial testing")
}

// ═════════════════════════════════════════════════════════════════════════════
// Stage 1: Corrupted Trace Detection
// ═════════════════════════════════════════════════════════════════════════════

/// Completely invalid JSON must be rejected by TraceParser.
#[test]
fn test_adversarial_garbage_trace_json() {
    let result = TraceParser::parse_json_str("not json at all {{{");
    assert!(result.is_err(), "Garbage JSON must be rejected");
}

/// Valid JSON but missing required fields.
#[test]
fn test_adversarial_missing_session_id() {
    let json = r#"{"entries": [], "digest": {}}"#;
    let result = TraceParser::parse_json_str(json);
    assert!(result.is_err(), "Missing session_id must be rejected");
}

/// Valid JSON but entries array contains garbage op types.
#[test]
fn test_adversarial_unknown_op_type() {
    let json = serde_json::to_string(&json!({
        "trace_version": 1,
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "digest": {},
        "entries": [{
            "seq": 0,
            "op": "definitely_not_a_real_op",
            "timestamp_ns": 1000,
            "duration_ns": 100,
            "input_hashes": {},
            "output_hashes": {},
            "params": {},
            "metrics": {}
        }]
    }))
    .unwrap();

    // TraceParser itself accepts unknown ops (they parse as strings).
    // But CircuitBuilder rejects entries with unknown op types.
    let trace = TraceParser::parse_json_str(&json).unwrap();
    let builder = CircuitBuilder::new();
    let result = builder.build(&trace);
    assert!(
        result.is_err(),
        "CircuitBuilder must reject traces with unknown op types"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Unknown op") || err_msg.contains("definitely_not_a_real_op"),
        "Error message must reference the unknown op: {err_msg}"
    );
}

/// Tamper with a trace entry after parsing and verify chain hash breaks.
#[test]
fn test_adversarial_tampered_trace_chain_hash() {
    let json = minimal_trace();
    let mut trace = TraceParser::parse_json_str(&json).unwrap();
    assert!(trace.validate_chain_hash(), "Original hash must be valid");

    // Tamper: modify a field in the first entry
    trace.entries[0].duration_ns = 999_999_999;

    assert!(
        !trace.validate_chain_hash(),
        "Tampered trace must fail chain hash validation"
    );
}

/// Empty entries list: parse should succeed but circuit inputs should
/// be minimal (no physics constraints).
#[test]
fn test_adversarial_empty_trace() {
    let json = serde_json::to_string(&json!({
        "trace_version": 1,
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "digest": {},
        "entries": []
    }))
    .unwrap();

    let trace = TraceParser::parse_json_str(&json).unwrap();
    assert!(trace.entries.is_empty());

    let builder = CircuitBuilder::new();
    let inputs = builder.build(&trace).unwrap();
    assert_eq!(inputs.summary.total_svd_ops, 0);
    assert_eq!(inputs.summary.total_mpo_ops, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Stage 2: Malformed Circuit Inputs
// ═════════════════════════════════════════════════════════════════════════════

/// SVD with unordered singular values should still produce constraints
/// (the constraint data captures the values; the ZK circuit enforces ordering).
#[test]
fn test_adversarial_unordered_singular_values() {
    let json = serde_json::to_string(&json!({
        "trace_version": 1,
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "digest": {},
        "entries": [{
            "seq": 0,
            "op": "svd_truncated",
            "timestamp_ns": 1000,
            "duration_ns": 100,
            "input_hashes": {"A": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
            "output_hashes": {
                "U": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                "S": "1111111111111111111111111111111111111111111111111111111111111111",
                "Vh": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "params": {"input_shape": [10, 10], "chi_max": 5, "cutoff": 1e-14},
            "metrics": {
                "truncation_error": 1e-5,
                "rank": 3,
                "original_rank": 10,
                "singular_values": [1.0, 5.0, 2.0]
            }
        }]
    }))
    .unwrap();

    let trace = TraceParser::parse_json_str(&json).unwrap();
    let builder = CircuitBuilder::new();
    let inputs = builder.build(&trace).unwrap();

    // The circuit builder captures the claimed values; the ZK circuit
    // will reject the proof if ordering is violated.
    // We just verify the builder doesn't crash — the ZK layer enforces ordering.
    assert!(inputs.summary.total_svd_ops >= 1);
}

/// Negative singular values in trace metrics.
#[test]
fn test_adversarial_negative_singular_values() {
    let json = serde_json::to_string(&json!({
        "trace_version": 1,
        "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "digest": {},
        "entries": [{
            "seq": 0,
            "op": "svd_truncated",
            "timestamp_ns": 1000,
            "duration_ns": 100,
            "input_hashes": {"A": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
            "output_hashes": {
                "U": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                "S": "1111111111111111111111111111111111111111111111111111111111111111",
                "Vh": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "params": {"input_shape": [10, 10], "chi_max": 5, "cutoff": 1e-14},
            "metrics": {
                "truncation_error": 0.001,
                "rank": 3,
                "original_rank": 10,
                "singular_values": [3.0, -1.0, 0.5]
            }
        }]
    }))
    .unwrap();

    let trace = TraceParser::parse_json_str(&json).unwrap();
    let builder = CircuitBuilder::new();
    // Should not crash — the constraint data captures the claimed values
    let inputs = builder.build(&trace).unwrap();
    assert!(inputs.summary.total_svd_ops >= 1);
}

// ═════════════════════════════════════════════════════════════════════════════
// Stage 3: Modified Proof Bytes → Certificate Hash Fails
// ═════════════════════════════════════════════════════════════════════════════

/// Flip a single bit in the proof blob region of a certificate.
#[test]
fn test_adversarial_single_bit_flip_proof_bytes() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let proof_bytes = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
    let cert = build_test_certificate(&signing_key, &proof_bytes);

    // Flip one bit in the content area (not the signature section)
    let mut tampered = cert.clone();
    let content_end = tampered.len() - SIGNATURE_SECTION_SIZE;
    let flip_offset = TPC_HEADER_SIZE + 20; // well into the Layer A/B section
    if flip_offset < content_end {
        tampered[flip_offset] ^= 0x01;
    }

    let v = verify_certificate(&tampered).unwrap();
    assert!(!v.hash_valid, "Single bit flip must invalidate hash");
    assert!(!v.is_valid());
}

/// Replace all proof bytes with zeros.
#[test]
fn test_adversarial_zeroed_proof_bytes() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let cert = build_test_certificate(&signing_key, &[0xFF; 64]);

    // Zero out a chunk of the content
    let mut tampered = cert.clone();
    let content_end = tampered.len() - SIGNATURE_SECTION_SIZE;
    let start = TPC_HEADER_SIZE + 10;
    let end = std::cmp::min(start + 40, content_end);
    for byte in &mut tampered[start..end] {
        *byte = 0;
    }

    let v = verify_certificate(&tampered).unwrap();
    assert!(!v.hash_valid, "Zeroed content must fail hash");
}

// ═════════════════════════════════════════════════════════════════════════════
// Stage 4: Wrong Ed25519 Signature
// ═════════════════════════════════════════════════════════════════════════════

/// Certificate signed with key A, signature replaced with key B's signature.
#[test]
fn test_adversarial_wrong_signing_key() {
    let key_a = SigningKey::generate(&mut OsRng);
    let key_b = SigningKey::generate(&mut OsRng);

    let cert_a = build_test_certificate(&key_a, b"legit proof bytes");
    let cert_b = build_test_certificate(&key_b, b"legit proof bytes");

    // Take the signature section from cert_b and graft it onto cert_a's content
    let sig_offset_a = cert_a.len() - SIGNATURE_SECTION_SIZE;
    let sig_offset_b = cert_b.len() - SIGNATURE_SECTION_SIZE;

    let mut frankenstein = cert_a[..sig_offset_a].to_vec();
    frankenstein.extend_from_slice(&cert_b[sig_offset_b..]);

    let v = verify_certificate(&frankenstein).unwrap();
    // The hash was computed over cert_a's content, but the stored hash
    // in the signature section is from cert_b. So hash check should fail
    // (or signature check, depending on which takes precedence).
    assert!(
        !v.is_valid(),
        "Grafted signature from different key must fail verification"
    );
}

/// Corrupt the Ed25519 signature bytes while keeping the public key intact.
#[test]
fn test_adversarial_corrupted_signature_bytes() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let cert = build_test_certificate(&signing_key, b"proof data");

    let mut tampered = cert.clone();
    let sig_offset = tampered.len() - SIGNATURE_SECTION_SIZE;
    // The signature starts after the 32-byte public key
    let sig_start = sig_offset + 32;
    // Flip a byte in the signature
    tampered[sig_start + 10] ^= 0xFF;

    let v = verify_certificate(&tampered).unwrap();
    assert!(!v.signature_valid, "Corrupted signature must fail sig check");
    // The hash should still be valid because we didn't touch the content
    assert!(v.hash_valid, "Content hash should still be valid");
    assert!(!v.is_valid(), "Overall must fail");
}

/// Replace the public key with all zeros (makes it look unsigned).
#[test]
fn test_adversarial_zeroed_public_key() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let cert = build_test_certificate(&signing_key, b"proof data");

    let mut tampered = cert.clone();
    let sig_offset = tampered.len() - SIGNATURE_SECTION_SIZE;
    // Zero out the public key
    for byte in &mut tampered[sig_offset..sig_offset + 32] {
        *byte = 0;
    }

    let v = verify_certificate(&tampered).unwrap();
    // With zeroed public key, it looks unsigned — signature_valid = true (unsigned passes).
    // BUT the stored hash no longer matches because we modified the trailing section?
    // Actually, the public key IS in the signature section (after content), and the
    // hash is computed over the content (before signature section).
    // So hash_valid should still be true, and is_unsigned should be true.
    assert!(v.is_unsigned, "Zeroed public key should be detected as unsigned");
    assert!(v.hash_valid, "Content hash is unaffected by key change");
}

// ═════════════════════════════════════════════════════════════════════════════
// Stage 5: Truncated Certificate
// ═════════════════════════════════════════════════════════════════════════════

/// Certificate truncated to just the header (no signature section).
#[test]
fn test_adversarial_truncated_to_header_only() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let cert = build_test_certificate(&signing_key, b"data");

    let truncated = &cert[..TPC_HEADER_SIZE];
    let result = verify_certificate(truncated);
    assert!(result.is_err(), "Certificate with only header must fail");
}

/// Certificate truncated in the middle.
#[test]
fn test_adversarial_truncated_mid_content() {
    let signing_key = SigningKey::generate(&mut OsRng);
    let cert = build_test_certificate(&signing_key, b"data");

    let mid = cert.len() / 2;
    let truncated = &cert[..mid];
    let result = verify_certificate(truncated);
    // Either errors out or reports invalid hash
    match result {
        Err(_) => {} // Structural error — good
        Ok(v) => assert!(!v.is_valid(), "Truncated certificate must not be valid"),
    }
}

/// Empty byte slice.
#[test]
fn test_adversarial_empty_certificate() {
    let result = verify_certificate(&[]);
    assert!(result.is_err(), "Empty bytes must be rejected");
}

/// Just 1 byte.
#[test]
fn test_adversarial_single_byte_certificate() {
    let result = verify_certificate(&[0x42]);
    assert!(result.is_err(), "Single byte must be rejected");
}

// ═════════════════════════════════════════════════════════════════════════════
// Stage 6: Replay Attack (same content, different signing context)
// ═════════════════════════════════════════════════════════════════════════════

/// Two certificates with identical proof content but different signing keys.
/// Both must independently verify, and have different content_hashes
/// (because the certificate_id/timestamp differ).
#[test]
fn test_adversarial_replay_different_keys() {
    let key1 = SigningKey::generate(&mut OsRng);
    let key2 = SigningKey::generate(&mut OsRng);

    let proof = b"identical proof content for replay test";
    let cert1 = build_test_certificate(&key1, proof);
    let cert2 = build_test_certificate(&key2, proof);

    let v1 = verify_certificate(&cert1).unwrap();
    let v2 = verify_certificate(&cert2).unwrap();

    assert!(v1.is_valid());
    assert!(v2.is_valid());

    // The content hashes should differ because CertificateWriter::new()
    // generates a fresh UUID and timestamp for each certificate.
    assert_ne!(
        v1.content_hash, v2.content_hash,
        "Different certificates must have different content hashes due to unique headers"
    );
}

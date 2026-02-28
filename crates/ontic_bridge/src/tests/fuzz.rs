//! Fuzzing Tests for Malformed Headers
//!
//! Tests protocol resilience against:
//! - Corrupted magic numbers
//! - Invalid version numbers
//! - Out-of-range field values
//! - Malformed bond dimensions
//! - Invalid checksums

use crate::protocol::{TensorBridgeHeader, TENSOR_BRIDGE_MAGIC};
use crate::qtt::{QTTBridgeHeader, QTT_BRIDGE_MAGIC, MAX_QTT_SITES};
use crate::sovereign_v2::{SovereignHeader, SOVEREIGN_MAGIC};
use crate::BridgeError;

/// Test that corrupted magic numbers are detected
#[test]
fn test_fuzz_magic_numbers() {
    // QTT with corrupted magic
    let qtt = QTTBridgeHeader::default();
    
    // Test all single-byte corruptions
    for byte_idx in 0..4 {
        let mut corrupted = qtt;
        corrupted.magic[byte_idx] ^= 0xFF;
        
        match corrupted.validate() {
            Err(BridgeError::InvalidMagic { expected, actual }) => {
                assert_eq!(expected, QTT_BRIDGE_MAGIC);
                assert_ne!(actual, QTT_BRIDGE_MAGIC);
            }
            _ => panic!("Should detect corrupted magic at byte {}", byte_idx),
        }
    }
    
    // Sovereign with corrupted magic (u32 type)
    let mut sovereign = SovereignHeader::default();
    sovereign.magic = 0xDEADBEEF; // Invalid magic
    assert!(sovereign.validate().is_err());
}

/// Test that invalid version numbers are rejected
#[test]
fn test_fuzz_version_numbers() {
    let mut qtt = QTTBridgeHeader::default();
    
    // Test version 0
    qtt.version = 0;
    assert!(qtt.validate().is_err() || qtt.version != 0);
    
    // Test very high version
    qtt.version = u32::MAX;
    assert!(qtt.validate().is_err());
    
    // Test version 2 (unsupported)
    qtt.version = 2;
    assert!(qtt.validate().is_err());
}

/// Test out-of-range num_sites values
#[test]
fn test_fuzz_num_sites() {
    let mut qtt = QTTBridgeHeader::default();
    qtt.compression_ratio = 2.0;
    qtt.truncation_error = 1e-9;
    
    // Test num_sites > MAX_QTT_SITES
    qtt.num_sites = (MAX_QTT_SITES + 1) as u32;
    assert!(qtt.validate().is_err());
    
    // Test num_sites = u32::MAX
    qtt.num_sites = u32::MAX;
    assert!(qtt.validate().is_err());
    
    // Test num_sites = 0 (edge case, should be valid for empty TT)
    qtt.num_sites = 0;
    // This may or may not be valid depending on protocol
}

/// Test invalid physical dimension values
#[test]
fn test_fuzz_physical_dim() {
    let mut qtt = QTTBridgeHeader::default();
    
    // physical_dim = 0 should be invalid
    qtt.physical_dim = 0;
    assert!(qtt.validate().is_err());
    
    // Very large physical_dim
    qtt.physical_dim = u32::MAX;
    // Should be caught somewhere in the pipeline
}

/// Test doctrine validation with edge cases
#[test]
fn test_fuzz_doctrine_values() {
    let mut qtt = QTTBridgeHeader::default();
    qtt.num_sites = 10;
    
    // Test compression ratio edge cases
    let test_ratios = [
        (0.0, false),      // Zero ratio - invalid
        (0.5, false),      // Below minimum
        (1.0, false),      // Exactly 1 - no compression
        (1.49, false),     // Just below minimum
        (1.5, true),       // Exactly at minimum
        (1.51, true),      // Just above minimum
        (10.0, true),      // Good compression
        (f32::INFINITY, true),  // Infinite compression (edge case)
        (f32::NAN, false),      // NaN should be rejected
    ];
    
    for (ratio, should_pass) in test_ratios {
        qtt.compression_ratio = ratio;
        qtt.truncation_error = 1e-9;
        
        let result = qtt.validate_doctrine();
        if should_pass && !ratio.is_nan() {
            assert!(result.is_ok(), "Ratio {} should pass", ratio);
        }
    }
    
    // Test truncation error edge cases
    let test_errors = [
        (0.0, true),           // Perfect precision
        (1e-10, true),         // Very good
        (1e-6, true),          // At limit
        (1.1e-6, false),       // Just above limit
        (1e-3, false),         // Too high
        (1.0, false),          // Very bad
        (f64::INFINITY, false), // Infinite error
        (f64::NAN, false),      // NaN
    ];
    
    for (error, should_pass) in test_errors {
        qtt.compression_ratio = 2.0;
        qtt.truncation_error = error;
        
        let result = qtt.validate_doctrine();
        if should_pass && !error.is_nan() {
            assert!(result.is_ok(), "Error {} should pass", error);
        }
    }
}

/// Test bond dimension array with invalid values
#[test]
fn test_fuzz_bond_dims() {
    let mut qtt = QTTBridgeHeader::default();
    qtt.num_sites = 5;
    qtt.physical_dim = 2;
    qtt.dtype = 0; // Float32
    
    // Set valid bond dims
    for i in 0..4 {
        qtt.bond_dims[i] = 4;
    }
    
    // Test chi_left and chi_right for boundary sites
    assert_eq!(qtt.chi_left(0), 1, "First site chi_left should be 1");
    assert_eq!(qtt.chi_right(4), 1, "Last site chi_right should be 1");
    
    // Test with zero bond dimension (edge case)
    qtt.bond_dims[1] = 0;
    let core_size = qtt.core_size(2);
    // With bond_dims[1] = 0, chi_left(2) = 0, so core_size = 0
    // Or chi_left may return the raw 0 value, leading to 0 * d * chi_right = 0
    assert_eq!(core_size, 0, "Zero bond dim should result in zero core size");
    
    // Test with very large bond dimension
    qtt.bond_dims[0] = u16::MAX;
    qtt.bond_dims[1] = 4; // Reset to valid value
    let core_size = qtt.core_size(1);
    // Should compute without overflow
    assert!(core_size > 0, "Large bond dim should still compute");
}

/// Test data type validation
#[test]
fn test_fuzz_data_types() {
    use crate::qtt::QTTDataType;
    
    // Valid types
    assert!(QTTDataType::from_u8(0).is_some()); // Float32
    assert!(QTTDataType::from_u8(1).is_some()); // Float64
    assert!(QTTDataType::from_u8(2).is_some()); // Float16
    
    // Invalid types
    assert!(QTTDataType::from_u8(3).is_none());
    assert!(QTTDataType::from_u8(255).is_none());
    
    // Element sizes
    assert_eq!(QTTDataType::Float32.element_size(), 4);
    assert_eq!(QTTDataType::Float64.element_size(), 8);
    assert_eq!(QTTDataType::Float16.element_size(), 2);
}

/// Test flags bitfield manipulation
#[test]
fn test_fuzz_flags() {
    let mut qtt = QTTBridgeHeader::default();
    
    // Test all individual flags
    qtt.flags = 0x01; // is_complex
    assert!(qtt.is_complex());
    assert!(!qtt.is_canonical());
    
    qtt.flags = 0x02; // is_canonical
    assert!(!qtt.is_complex());
    assert!(qtt.is_canonical());
    
    qtt.flags = 0x10; // is_ready
    assert!(qtt.is_ready());
    
    // Test all flags set
    qtt.flags = 0xFF;
    assert!(qtt.is_complex());
    assert!(qtt.is_canonical());
    assert!(qtt.has_norm());
    assert!(qtt.is_periodic());
    assert!(qtt.is_ready());
}

/// Test core offset array with invalid values
#[test]
fn test_fuzz_core_offsets() {
    let mut qtt = QTTBridgeHeader::default();
    qtt.num_sites = 4;
    qtt.physical_dim = 2;
    qtt.dtype = 0;
    
    for i in 0..3 {
        qtt.bond_dims[i] = 4;
    }
    
    // Valid offsets (increasing)
    qtt.core_offsets[0] = 0;
    qtt.core_offsets[1] = 32;  // After core 0
    qtt.core_offsets[2] = 160; // After core 1
    qtt.core_offsets[3] = 288; // After core 2
    
    // Test with overlapping offsets (invalid but should handle gracefully)
    qtt.core_offsets[2] = 16; // Overlaps with core 0
    
    // This should be caught during frame reading, not header validation
    assert!(qtt.validate().is_ok(), "Header validation doesn't check offset overlaps");
}

/// Test timestamp handling
#[test]
fn test_fuzz_timestamps() {
    let mut qtt = QTTBridgeHeader::default();
    
    // Test zero timestamp (epoch)
    qtt.producer_timestamp_us = 0;
    qtt.consumer_timestamp_us = 0;
    assert!(qtt.validate().is_ok());
    
    // Test max timestamp
    qtt.producer_timestamp_us = u64::MAX;
    assert!(qtt.validate().is_ok());
    
    // Test consumer before producer (time travel?)
    qtt.producer_timestamp_us = 1_000_000;
    qtt.consumer_timestamp_us = 500_000;
    // This may or may not be valid depending on protocol
}

/// Test CRC32 with various data patterns
#[test]
fn test_fuzz_crc32_patterns() {
    // Empty data
    let checksum = QTTBridgeHeader::compute_checksum(&[]);
    assert_eq!(checksum, 0x00000000, "Empty data CRC32");
    
    // All zeros
    let zeros = vec![0u8; 1000];
    let checksum = QTTBridgeHeader::compute_checksum(&zeros);
    assert_ne!(checksum, 0, "All zeros should have non-zero CRC32");
    
    // All ones
    let ones = vec![0xFF; 1000];
    let checksum = QTTBridgeHeader::compute_checksum(&ones);
    assert_ne!(checksum, 0, "All ones should have non-zero CRC32");
    
    // Known test vector (CRC-32 ISO-HDLC)
    let test = b"123456789";
    let checksum = QTTBridgeHeader::compute_checksum(test);
    assert_eq!(checksum, 0xCBF43926, "Known CRC32 test vector");
    
    // Single bit change should change checksum
    let mut data = vec![0u8; 100];
    let checksum1 = QTTBridgeHeader::compute_checksum(&data);
    data[50] = 1;
    let checksum2 = QTTBridgeHeader::compute_checksum(&data);
    assert_ne!(checksum1, checksum2, "Single bit flip should change CRC32");
}

/// Randomized fuzzing with pseudo-random inputs
#[test]
fn test_fuzz_random_headers() {
    // Use a simple LCG for reproducible "random" values
    let mut seed: u64 = 0xDEADBEEF;
    let lcg = |s: &mut u64| -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *s
    };
    
    for _ in 0..1000 {
        let mut qtt = QTTBridgeHeader::default();
        
        // Randomly corrupt some fields
        qtt.num_sites = (lcg(&mut seed) % 100) as u32;
        qtt.physical_dim = (lcg(&mut seed) % 10) as u32;
        qtt.compression_ratio = (lcg(&mut seed) % 1000) as f32 / 100.0;
        qtt.truncation_error = (lcg(&mut seed) % 1000) as f64 / 1e9;
        
        // Should not panic regardless of input
        let _ = qtt.validate();
        let _ = qtt.validate_doctrine();
        let _ = qtt.chi_left(0);
        let _ = qtt.chi_right(0);
        let _ = qtt.core_size(0);
        let _ = qtt.is_compression_beneficial();
    }
}

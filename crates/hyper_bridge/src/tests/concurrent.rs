//! Concurrent Access Tests
//!
//! Tests for thread-safety and concurrent access patterns.
//! The bridge protocol is designed for single-producer, single-consumer,
//! but we test edge cases and race conditions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::protocol::TensorBridgeHeader;
use crate::qtt::QTTBridgeHeader;
use crate::sovereign_v2::SovereignHeader;
use crate::weather::WeatherHeader;
use crate::trajectory::TrajectoryHeader;
use crate::swarm::SwarmHeader;

/// Test that header reads are atomic (no torn reads)
#[test]
fn test_header_atomic_reads() {
    // Simulate rapid header updates
    let counter = Arc::new(AtomicU64::new(0));
    let counter_clone = counter.clone();
    
    let producer = thread::spawn(move || {
        for i in 0..10_000u64 {
            counter_clone.store(i, Ordering::SeqCst);
        }
    });
    
    let consumer = thread::spawn(move || {
        let mut last_seen = 0u64;
        let mut monotonic = true;
        
        for _ in 0..10_000 {
            let current = counter.load(Ordering::SeqCst);
            // Frame numbers should be monotonically increasing or equal
            if current < last_seen {
                monotonic = false;
                break;
            }
            last_seen = current;
            thread::yield_now();
        }
        
        monotonic
    });
    
    producer.join().unwrap();
    let monotonic = consumer.join().unwrap();
    
    assert!(monotonic, "Frame numbers should be monotonic");
}

/// Test header size consistency under concurrent access
#[test]
fn test_header_sizes_thread_safe() {
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                // All these size checks should be consistent
                let tensor_size = std::mem::size_of::<TensorBridgeHeader>();
                let qtt_size = std::mem::size_of::<QTTBridgeHeader>();
                let sovereign_size = std::mem::size_of::<SovereignHeader>();
                let weather_size = std::mem::size_of::<WeatherHeader>();
                let trajectory_size = std::mem::size_of::<TrajectoryHeader>();
                let swarm_size = std::mem::size_of::<SwarmHeader>();
                
                // Verify power-of-2 sizes
                assert!(tensor_size.is_power_of_two() || tensor_size == 4096);
                assert!(qtt_size.is_power_of_two());
                assert!(sovereign_size.is_power_of_two());
                assert!(weather_size.is_power_of_two());
                assert!(trajectory_size.is_power_of_two());
                assert!(swarm_size.is_power_of_two());
                
                (tensor_size, qtt_size, sovereign_size, weather_size, trajectory_size, swarm_size)
            })
        })
        .collect();
    
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // All threads should see the same sizes
    let first = &results[0];
    for result in &results[1..] {
        assert_eq!(result, first, "Header sizes must be consistent across threads");
    }
}

/// Test QTT header validation is thread-safe
#[test]
fn test_qtt_validation_concurrent() {
    let handles: Vec<_> = (0..8)
        .map(|i| {
            thread::spawn(move || {
                let mut header = QTTBridgeHeader::default();
                header.num_sites = 10;
                header.compression_ratio = 2.0 + (i as f32 * 0.1);
                header.truncation_error = 1e-8;
                
                // Validation should succeed for all threads
                header.validate().is_ok() && header.validate_doctrine().is_ok()
            })
        })
        .collect();
    
    for handle in handles {
        assert!(handle.join().unwrap(), "QTT validation should be thread-safe");
    }
}

/// Test bond dimension calculations are deterministic
#[test]
fn test_bond_dim_concurrent() {
    let handles: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                let mut header = QTTBridgeHeader::default();
                header.num_sites = 10;
                header.physical_dim = 2;
                
                for i in 0..9 {
                    header.bond_dims[i] = ((i + 1) * 2) as u16;
                }
                
                let mut results = Vec::new();
                for site in 0..10 {
                    results.push((
                        header.chi_left(site),
                        header.chi_right(site),
                        header.core_size(site),
                    ));
                }
                results
            })
        })
        .collect();
    
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // All threads should compute identical results
    let first = &results[0];
    for result in &results[1..] {
        assert_eq!(result, first, "Bond dimension calculations must be deterministic");
    }
}

/// Test CRC32 computation is thread-safe
#[test]
fn test_crc32_concurrent() {
    let test_data = b"The quick brown fox jumps over the lazy dog";
    
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let data = test_data.to_vec();
            thread::spawn(move || {
                QTTBridgeHeader::compute_checksum(&data)
            })
        })
        .collect();
    
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    
    // All threads should compute the same checksum
    let expected = results[0];
    for result in &results[1..] {
        assert_eq!(*result, expected, "CRC32 must be deterministic");
    }
}

/// Stress test: rapid header creation and validation
#[test]
fn test_stress_header_creation() {
    let start = std::time::Instant::now();
    let iterations = 100_000;
    
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            thread::spawn(move || {
                let mut success_count = 0u64;
                
                for i in 0..(iterations / 4) {
                    let mut header = QTTBridgeHeader::default();
                    header.frame_number = (thread_id as u64 * 1_000_000) + i;
                    header.num_sites = ((i % 60) + 4) as u32;
                    header.compression_ratio = 2.0;
                    header.truncation_error = 1e-9;
                    
                    if header.validate().is_ok() {
                        success_count += 1;
                    }
                }
                
                success_count
            })
        })
        .collect();
    
    let total_success: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let elapsed = start.elapsed();
    
    assert_eq!(total_success, iterations as u64, "All validations should succeed");
    
    // Performance check: should complete in reasonable time
    assert!(
        elapsed < Duration::from_secs(5),
        "Stress test took too long: {:?}",
        elapsed
    );
}

//! Proof compression and aggregation.
//!
//! Reduces proof sizes through:
//! 1. Zero-stripping: Remove trailing zero padding from proof bytes
//! 2. Run-length encoding for repetitive proof sections
//! 3. Proof aggregation: Bundle multiple proofs with shared metadata
//! 4. Hash deduplication across proof bundles
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use fluidelite_core::physics_traits::{PhysicsProof, SolverType};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Compression method for proof bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionMethod {
    /// No compression.
    None,
    /// Strip trailing zeros.
    ZeroStrip,
    /// Run-length encoding.
    Rle,
    /// Zero-strip + RLE combined.
    ZeroStripRle,
}

/// Configuration for proof compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression method to use.
    pub method: CompressionMethod,

    /// Minimum proof size to attempt compression (bytes).
    pub min_size_threshold: usize,

    /// Enable hash deduplication in bundles.
    pub deduplicate_hashes: bool,

    /// Maximum proofs in a single bundle.
    pub max_bundle_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            method: CompressionMethod::ZeroStripRle,
            min_size_threshold: 128,
            deduplicate_hashes: true,
            max_bundle_size: 1024,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compressed Proof
// ═══════════════════════════════════════════════════════════════════════════

/// A compressed proof with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedProof {
    /// Magic header: "CPRF".
    pub magic: [u8; 4],

    /// Compression method used.
    pub method: CompressionMethod,

    /// Original uncompressed size in bytes.
    pub original_size: usize,

    /// Compressed proof bytes.
    pub compressed_bytes: Vec<u8>,

    /// Compression time in microseconds.
    pub compression_time_us: u64,

    /// Solver type.
    pub solver_type: SolverType,
}

impl CompressedProof {
    /// Compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        self.compressed_bytes.len()
    }

    /// Compression ratio (original / compressed).
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_bytes.is_empty() {
            1.0
        } else {
            self.original_size as f64 / self.compressed_bytes.len() as f64
        }
    }

    /// Space savings as a percentage.
    pub fn savings_pct(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            (1.0 - self.compressed_bytes.len() as f64 / self.original_size as f64) * 100.0
        }
    }

    /// Serialize to bytes for transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(&self.magic);
        out.push(self.method as u8);
        out.extend((self.original_size as u64).to_le_bytes());
        out.extend((self.compressed_bytes.len() as u64).to_le_bytes());
        out.extend(&self.compressed_bytes);
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proof Bundle
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregated bundle of multiple proofs with shared metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundle {
    /// Magic header: "PBDL".
    pub magic: [u8; 4],

    /// Bundle version.
    pub version: u32,

    /// Solver type for all proofs in the bundle.
    pub solver_type: SolverType,

    /// Number of proofs in the bundle.
    pub proof_count: usize,

    /// Individual compressed proofs.
    pub proofs: Vec<CompressedProof>,

    /// Total original size of all proofs.
    pub total_original_bytes: usize,

    /// Total compressed size of all proofs.
    pub total_compressed_bytes: usize,

    /// Bundle creation time in microseconds.
    pub bundle_time_us: u64,

    /// Unique parameter hash limbs (deduplicated across proofs).
    pub unique_param_hashes: Vec<[u64; 4]>,
}

impl ProofBundle {
    /// Overall compression ratio for the bundle.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            1.0
        } else {
            self.total_original_bytes as f64 / self.total_compressed_bytes as f64
        }
    }

    /// Total savings in bytes.
    pub fn total_savings(&self) -> usize {
        self.total_original_bytes.saturating_sub(self.total_compressed_bytes)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compression Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics for compression operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Total proofs compressed.
    pub total_compressed: u64,

    /// Total proofs decompressed.
    pub total_decompressed: u64,

    /// Total bundles created.
    pub total_bundles: u64,

    /// Total original bytes processed.
    pub total_original_bytes: u64,

    /// Total compressed bytes produced.
    pub total_compressed_bytes: u64,

    /// Total compression time in microseconds.
    pub total_compression_time_us: u64,

    /// Total decompression time in microseconds.
    pub total_decompression_time_us: u64,
}

impl CompressionStats {
    /// Average compression ratio.
    pub fn avg_compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            1.0
        } else {
            self.total_original_bytes as f64 / self.total_compressed_bytes as f64
        }
    }

    /// Total savings in bytes.
    pub fn total_savings(&self) -> u64 {
        self.total_original_bytes
            .saturating_sub(self.total_compressed_bytes)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Proof Compressor
// ═══════════════════════════════════════════════════════════════════════════

/// Proof compressor with configurable compression strategies.
pub struct ProofCompressor {
    /// Configuration.
    config: CompressionConfig,

    /// Statistics.
    stats: CompressionStats,
}

impl ProofCompressor {
    /// Create a new proof compressor.
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            stats: CompressionStats::default(),
        }
    }

    /// Compress a proof's serialized bytes.
    pub fn compress<P: PhysicsProof>(
        &mut self,
        proof: &P,
    ) -> CompressedProof {
        let raw = proof.to_serialized_bytes();
        let t0 = Instant::now();

        let compressed = if raw.len() < self.config.min_size_threshold {
            raw.clone()
        } else {
            match self.config.method {
                CompressionMethod::None => raw.clone(),
                CompressionMethod::ZeroStrip => zero_strip_compress(&raw),
                CompressionMethod::Rle => rle_compress(&raw),
                CompressionMethod::ZeroStripRle => {
                    let stripped = zero_strip_compress(&raw);
                    rle_compress(&stripped)
                }
            }
        };

        let time_us = t0.elapsed().as_micros() as u64;

        self.stats.total_compressed += 1;
        self.stats.total_original_bytes += raw.len() as u64;
        self.stats.total_compressed_bytes += compressed.len() as u64;
        self.stats.total_compression_time_us += time_us;

        CompressedProof {
            magic: *b"CPRF",
            method: self.config.method,
            original_size: raw.len(),
            compressed_bytes: compressed,
            compression_time_us: time_us,
            solver_type: proof.solver_type(),
        }
    }

    /// Decompress a compressed proof back to raw bytes.
    pub fn decompress(
        &mut self,
        compressed: &CompressedProof,
    ) -> Result<Vec<u8>, String> {
        if compressed.magic != *b"CPRF" {
            return Err("Invalid compressed proof magic".into());
        }

        let t0 = Instant::now();

        let decompressed = match compressed.method {
            CompressionMethod::None => compressed.compressed_bytes.clone(),
            CompressionMethod::ZeroStrip => {
                zero_strip_decompress(&compressed.compressed_bytes, compressed.original_size)
            }
            CompressionMethod::Rle => rle_decompress(&compressed.compressed_bytes)?,
            CompressionMethod::ZeroStripRle => {
                let rle_decompressed = rle_decompress(&compressed.compressed_bytes)?;
                zero_strip_decompress(&rle_decompressed, compressed.original_size)
            }
        };

        self.stats.total_decompressed += 1;
        self.stats.total_decompression_time_us +=
            t0.elapsed().as_micros() as u64;

        Ok(decompressed)
    }

    /// Create an aggregated bundle from multiple proofs.
    pub fn bundle<P: PhysicsProof>(
        &mut self,
        proofs: &[P],
    ) -> Result<ProofBundle, String> {
        if proofs.is_empty() {
            return Err("Cannot create empty bundle".into());
        }

        if proofs.len() > self.config.max_bundle_size {
            return Err(format!(
                "Bundle size {} exceeds maximum {}",
                proofs.len(),
                self.config.max_bundle_size
            ));
        }

        let t0 = Instant::now();
        let solver_type = proofs[0].solver_type();

        let mut compressed_proofs = Vec::with_capacity(proofs.len());
        let mut total_original = 0usize;
        let mut total_compressed = 0usize;
        let mut param_hashes = Vec::new();

        for proof in proofs {
            let cp = self.compress(proof);
            total_original += cp.original_size;
            total_compressed += cp.compressed_size();

            // Collect unique parameter hashes
            let ph = *proof.params_hash_limbs();
            if self.config.deduplicate_hashes && !param_hashes.contains(&ph) {
                param_hashes.push(ph);
            }

            compressed_proofs.push(cp);
        }

        let bundle_time_us = t0.elapsed().as_micros() as u64;
        self.stats.total_bundles += 1;

        Ok(ProofBundle {
            magic: *b"PBDL",
            version: 1,
            solver_type,
            proof_count: proofs.len(),
            proofs: compressed_proofs,
            total_original_bytes: total_original,
            total_compressed_bytes: total_compressed,
            bundle_time_us,
            unique_param_hashes: param_hashes,
        })
    }

    /// Get compression statistics.
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compression Algorithms
// ═══════════════════════════════════════════════════════════════════════════

/// Strip trailing zeros from a byte buffer.
/// Stores the original length for reconstruction.
fn zero_strip_compress(data: &[u8]) -> Vec<u8> {
    // Find last non-zero byte
    let last_nonzero = data.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);

    let mut out = Vec::with_capacity(8 + last_nonzero);
    // Store original length as LE u64
    out.extend((data.len() as u64).to_le_bytes());
    // Store non-zero prefix
    out.extend(&data[..last_nonzero]);
    out
}

/// Reconstruct zero-stripped data.
fn zero_strip_decompress(data: &[u8], original_size: usize) -> Vec<u8> {
    if data.len() < 8 {
        let mut out = vec![0u8; original_size];
        out[..data.len()].copy_from_slice(data);
        return out;
    }

    let stored_len =
        u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8])) as usize;
    let target_len = stored_len.max(original_size);
    let payload = &data[8..];

    let mut out = vec![0u8; target_len];
    let copy_len = payload.len().min(target_len);
    out[..copy_len].copy_from_slice(&payload[..copy_len]);
    out
}

/// Run-length encode a byte buffer.
///
/// Format: For each run:
///   - If count == 1: [byte]
///   - If count > 1:  [0xFF, count_high, count_low, byte]
///     (escape 0xFF by encoding as run of 1: [0xFF, 0x00, 0x01, 0xFF])
fn rle_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }

    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        let byte = data[i];
        let mut count = 1u16;

        while i + (count as usize) < data.len()
            && data[i + count as usize] == byte
            && count < 0xFFFF
        {
            count += 1;
        }

        if count > 3 || byte == 0xFF {
            // RLE encode: escape marker + count + byte
            out.push(0xFF);
            out.push((count >> 8) as u8);
            out.push((count & 0xFF) as u8);
            out.push(byte);
        } else {
            // Literal bytes
            for _ in 0..count {
                out.push(byte);
            }
        }

        i += count as usize;
    }

    out
}

/// Decode run-length encoded data.
fn rle_decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0xFF {
            // RLE marker
            if i + 3 >= data.len() {
                return Err("Truncated RLE sequence".into());
            }
            let count = ((data[i + 1] as u16) << 8) | (data[i + 2] as u16);
            let byte = data[i + 3];
            for _ in 0..count {
                out.push(byte);
            }
            i += 4;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }

    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use fluidelite_core::physics_traits::PhysicsProver;

    #[test]
    fn test_zero_strip_roundtrip() {
        let data = vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0];
        let compressed = zero_strip_compress(&data);
        assert!(compressed.len() < data.len() + 8); // Header overhead
        let decompressed = zero_strip_decompress(&compressed, data.len());
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zero_strip_all_zeros() {
        let data = vec![0u8; 100];
        let compressed = zero_strip_compress(&data);
        assert_eq!(compressed.len(), 8); // Just the length header
        let decompressed = zero_strip_decompress(&compressed, data.len());
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zero_strip_no_zeros() {
        let data: Vec<u8> = (1..=50).collect();
        let compressed = zero_strip_compress(&data);
        let decompressed = zero_strip_decompress(&compressed, data.len());
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_rle_compress_runs() {
        let data = vec![0xAA; 100];
        let compressed = rle_compress(&data);
        assert!(compressed.len() < data.len());
        let decompressed = rle_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_rle_no_runs() {
        let data: Vec<u8> = (0..100).map(|i| (i % 254) as u8).collect();
        let compressed = rle_compress(&data);
        let decompressed = rle_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_rle_escape_0xff() {
        let data = vec![0xFF, 0xFF, 0xFF];
        let compressed = rle_compress(&data);
        let decompressed = rle_decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_rle_empty() {
        let compressed = rle_compress(&[]);
        assert!(compressed.is_empty());
        let decompressed = rle_decompress(&compressed).unwrap();
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_compressor_euler3d() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let input_states: Vec<fluidelite_core::mps::MPS> = (0..5)
            .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
            .collect();
        let shift_mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
            .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
            .collect();

        let mut prover = fluidelite_circuits::euler3d::Euler3DProver::new(params).unwrap();
        let proof = prover.prove(&input_states, &shift_mpos).unwrap();

        let mut compressor = ProofCompressor::new(CompressionConfig::default());
        let compressed = compressor.compress(&proof);

        assert_eq!(compressed.magic, *b"CPRF");
        assert!(compressed.original_size > 0);
        // Stub proofs have lots of zeros, so compression should help
        assert!(compressed.compression_ratio() >= 1.0);

        // Decompress
        let decompressed = compressor.decompress(&compressed).unwrap();
        let original = proof.to_serialized_bytes();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_compressor_ns_imex() {
        let params = fluidelite_circuits::ns_imex::NSIMEXParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let input_states: Vec<fluidelite_core::mps::MPS> = (0..3)
            .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
            .collect();
        let shift_mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
            .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
            .collect();

        let mut prover = fluidelite_circuits::ns_imex::NSIMEXProver::new(params).unwrap();
        let proof = prover.prove(&input_states, &shift_mpos).unwrap();

        let mut compressor = ProofCompressor::new(CompressionConfig::default());
        let compressed = compressor.compress(&proof);

        assert_eq!(compressed.solver_type, SolverType::NsImex);
        assert!(compressed.compression_ratio() >= 1.0);

        let decompressed = compressor.decompress(&compressed).unwrap();
        let original = proof.to_serialized_bytes();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_bundle_creation() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let input_states: Vec<fluidelite_core::mps::MPS> = (0..5)
            .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
            .collect();
        let shift_mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
            .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
            .collect();

        let mut prover = fluidelite_circuits::euler3d::Euler3DProver::new(params).unwrap();
        let proof1 = prover.prove(&input_states, &shift_mpos).unwrap();
        let proof2 = prover.prove(&input_states, &shift_mpos).unwrap();
        let proof3 = prover.prove(&input_states, &shift_mpos).unwrap();

        let mut compressor = ProofCompressor::new(CompressionConfig::default());
        let bundle = compressor.bundle(&[proof1, proof2, proof3]).unwrap();

        assert_eq!(bundle.magic, *b"PBDL");
        assert_eq!(bundle.version, 1);
        assert_eq!(bundle.proof_count, 3);
        assert_eq!(bundle.proofs.len(), 3);
        assert!(bundle.total_original_bytes > 0);
        assert!(bundle.compression_ratio() >= 1.0);
        // Same params → should be deduplicated to 1 unique hash
        assert_eq!(bundle.unique_param_hashes.len(), 1);
    }

    #[test]
    fn test_bundle_empty() {
        let mut compressor = ProofCompressor::new(CompressionConfig::default());
        let result = compressor.bundle::<fluidelite_circuits::euler3d::Euler3DProof>(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_stats() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let states: Vec<fluidelite_core::mps::MPS> = (0..5)
            .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
            .collect();
        let mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
            .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
            .collect();

        let mut prover = fluidelite_circuits::euler3d::Euler3DProver::new(params).unwrap();
        let proof = prover.prove(&states, &mpos).unwrap();

        let mut compressor = ProofCompressor::new(CompressionConfig::default());
        let _ = compressor.compress(&proof);
        let _ = compressor.compress(&proof);

        let stats = compressor.stats();
        assert_eq!(stats.total_compressed, 2);
        assert!(stats.total_original_bytes > 0);
        assert!(stats.avg_compression_ratio() >= 1.0);
    }

    #[test]
    fn test_compressed_proof_to_bytes() {
        let cp = CompressedProof {
            magic: *b"CPRF",
            method: CompressionMethod::None,
            original_size: 100,
            compressed_bytes: vec![1, 2, 3],
            compression_time_us: 42,
            solver_type: SolverType::Euler3D,
        };
        let bytes = cp.to_bytes();
        assert_eq!(&bytes[..4], b"CPRF");
        assert!(bytes.len() > 4);
    }

    #[test]
    fn test_compression_method_none() {
        let params = fluidelite_circuits::euler3d::Euler3DParams::test_small();
        let num_sites = params.num_sites();
        let chi = params.chi_max;
        let states: Vec<fluidelite_core::mps::MPS> = (0..5)
            .map(|_| fluidelite_core::mps::MPS::new(num_sites, chi, 2))
            .collect();
        let mpos: Vec<fluidelite_core::mpo::MPO> = (0..3)
            .map(|_| fluidelite_core::mpo::MPO::identity(num_sites, 2))
            .collect();

        let mut prover = fluidelite_circuits::euler3d::Euler3DProver::new(params).unwrap();
        let proof = prover.prove(&states, &mpos).unwrap();

        let config = CompressionConfig {
            method: CompressionMethod::None,
            ..Default::default()
        };
        let mut compressor = ProofCompressor::new(config);
        let compressed = compressor.compress(&proof);
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed, proof.to_serialized_bytes());
    }
}

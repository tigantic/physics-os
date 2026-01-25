//! Zero-Expansion v2.1 Integration
//!
//! This module integrates the three Genesis primitives:
//! - QTT-GA: Elliptic curve operations via Clifford algebra
//! - QTT-RMT: Random matrix theory for Fiat-Shamir challenges
//! - QTT-RKHS: Kernel methods for lookup table compression
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    Zero-Expansion v2.1 Prover                           │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                         │
//! │  INPUT: Witness vector w ∈ F^N (N = 2^20+ entries)                     │
//! │                                                                         │
//! │  STEP 1: QTT Compression                                               │
//! │  ├── Compress witness: w → QTT cores C₁, C₂, ..., C_L                  │
//! │  └── O(r² L) storage instead of O(2^L)                                 │
//! │                                                                         │
//! │  STEP 2: MSM via QTT-GA (Layer 26)                                     │
//! │  ├── Represent EC points as GA vectors: (x,y) → x·e₁ + y·e₂           │
//! │  ├── Scalar mult via rotor: [k]P = exp(θB) P exp(-θB)                  │
//! │  └── Commitment: C = Σᵢ wᵢ · Gᵢ in QTT form                            │
//! │                                                                         │
//! │  STEP 3: Fiat-Shamir via QTT-RMT (Layer 22)                            │
//! │  ├── Transcript: transcript ← hash(C, public_inputs)                   │
//! │  ├── Challenge via RMT: α ← RmtChallengeGenerator(transcript)          │
//! │  └── Structured randomness from Wigner eigenvalues                     │
//! │                                                                         │
//! │  STEP 4: Lookup Tables via QTT-RKHS (Layer 24)                         │
//! │  ├── Compress range tables with kernel methods                         │
//! │  ├── Interpolate for cache misses                                      │
//! │  └── O(r) table access instead of O(N)                                 │
//! │                                                                         │
//! │  OUTPUT: Proof π = (C, structure_proof, public_inputs)                 │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::qtt_ga::{CliffordSignature, QttMultivector, qtt_geometric_product};
use crate::qtt_rmt::RmtChallengeGenerator;
use crate::qtt_rkhs::KernelLookupTable;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Prover transcript for Fiat-Shamir
#[derive(Clone, Debug)]
pub struct Transcript {
    /// Running hash of all appended data
    data: Vec<u8>,
}

impl Transcript {
    /// Create new empty transcript
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Append bytes to transcript
    pub fn append(&mut self, label: &[u8], value: &[u8]) {
        self.data.extend_from_slice(label);
        self.data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        self.data.extend_from_slice(value);
    }
    
    /// Append a scalar (f64 for simplicity, would be field element in production)
    pub fn append_scalar(&mut self, label: &[u8], value: f64) {
        self.append(label, &value.to_le_bytes());
    }
    
    /// Get current transcript hash as seed
    pub fn seed(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.data.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Generate Fiat-Shamir challenge using RMT
    pub fn rmt_challenge(&self, field_size: u64) -> u64 {
        let seed = self.seed();
        let generator = RmtChallengeGenerator::new();
        generator.generate_field_challenges(seed, 1, field_size)[0]
    }
    
    /// Generate multiple challenges
    pub fn rmt_challenges(&self, count: usize, field_size: u64) -> Vec<u64> {
        let seed = self.seed();
        let generator = RmtChallengeGenerator::new();
        generator.generate_field_challenges(seed, count, field_size)
    }
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

/// Point in the plane (for GA representation)
#[derive(Clone, Copy, Debug)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    
    /// Convert to GA vector in Cl(2,0)
    pub fn to_ga(&self, sig: &CliffordSignature) -> QttMultivector {
        assert!(sig.p >= 2);
        let n = sig.n();
        let mut coeffs = vec![0.0; 1 << n];
        coeffs[1] = self.x; // e₁
        coeffs[2] = self.y; // e₂
        QttMultivector::from_dense(sig.clone(), &coeffs, 8, 1e-10)
    }
    
    /// Add two points (for testing, not real EC addition)
    pub fn add(&self, other: &Point2D) -> Point2D {
        Point2D::new(self.x + other.x, self.y + other.y)
    }
    
    /// Scale point (for testing)
    pub fn scale(&self, k: f64) -> Point2D {
        Point2D::new(self.x * k, self.y * k)
    }
}

/// EC-like point with scalar multiplication via GA rotors
///
/// This demonstrates the principle - actual EC operations would use
/// proper finite field arithmetic.
pub struct GaCurvePoint {
    /// GA representation
    mv: QttMultivector,
}

impl GaCurvePoint {
    /// Create from coordinates
    pub fn from_coords(x: f64, y: f64, sig: &CliffordSignature) -> Self {
        let point = Point2D::new(x, y);
        Self { mv: point.to_ga(sig) }
    }
    
    /// Get x coordinate
    pub fn x(&self) -> f64 {
        self.mv.get_coefficient(1)
    }
    
    /// Get y coordinate
    pub fn y(&self) -> f64 {
        self.mv.get_coefficient(2)
    }
    
    /// Rotate by angle theta in the e₁-e₂ plane
    ///
    /// Uses rotor: R = exp(θ/2 · e₁₂) = cos(θ/2) + sin(θ/2)·e₁₂
    /// And P' = R P R̃
    pub fn rotate(&self, theta: f64) -> Self {
        let sig = &self.mv.signature;
        let n = sig.n();
        
        // Build rotor R = cos(θ/2) + sin(θ/2)·e₁₂
        let half_theta = theta / 2.0;
        let mut rotor_coeffs = vec![0.0; 1 << n];
        rotor_coeffs[0] = half_theta.cos();  // Scalar part
        rotor_coeffs[3] = half_theta.sin();  // e₁₂ = e₁∧e₂ is at index 0b11 = 3
        
        let rotor = QttMultivector::from_dense(sig.clone(), &rotor_coeffs, 8, 1e-10);
        
        // Build reverse rotor R̃ = cos(θ/2) - sin(θ/2)·e₁₂
        let mut rev_coeffs = vec![0.0; 1 << n];
        rev_coeffs[0] = half_theta.cos();
        rev_coeffs[3] = -half_theta.sin();
        let rotor_rev = QttMultivector::from_dense(sig.clone(), &rev_coeffs, 8, 1e-10);
        
        // Compute R P R̃
        let rp = qtt_geometric_product(&rotor, &self.mv, 8);
        let rprev = qtt_geometric_product(&rp, &rotor_rev, 8);
        
        Self { mv: rprev }
    }
}

/// Compressed lookup table using RKHS
pub struct CompressedLookupTable {
    /// Kernel-based table
    inner: KernelLookupTable,
    /// Statistics
    compression_ratio: f64,
}

impl CompressedLookupTable {
    /// Create from function
    pub fn from_function<F: Fn(f64) -> f64>(f: F, n: usize, length_scale: f64) -> Self {
        let inner = KernelLookupTable::from_function(f, n, length_scale);
        Self {
            inner,
            compression_ratio: 1.0, // Would be computed from kernel rank
        }
    }
    
    /// Direct lookup
    pub fn lookup(&self, idx: usize) -> f64 {
        self.inner.lookup(idx)
    }
    
    /// Interpolated lookup
    pub fn interpolate(&self, x: f64) -> f64 {
        self.inner.interpolate(x)
    }
}

/// Genesis-integrated witness commitment
///
/// Represents a commitment to a witness vector using QTT compression.
#[derive(Clone, Debug)]
pub struct QttCommitment {
    /// Commitment value (would be G1 point in production)
    pub value: Vec<f64>,
    /// Number of original elements
    pub n_elements: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
}

/// Zero-Expansion v2.1 Prover (no GPU/Halo2 dependencies)
///
/// Demonstrates the integration of QTT-GA, QTT-RMT, and QTT-RKHS
/// for ZK proof generation.
pub struct ZeroExpansionV21 {
    /// Clifford algebra signature for curve ops
    ga_signature: CliffordSignature,
    /// RMT challenge generator
    rmt_generator: RmtChallengeGenerator,
    /// Lookup tables (pre-built)
    lookup_tables: Vec<CompressedLookupTable>,
    /// Field size for challenges
    field_size: u64,
}

impl ZeroExpansionV21 {
    /// Create new prover
    pub fn new(field_size: u64) -> Self {
        // Cl(2,0) for 2D points
        let ga_signature = CliffordSignature::euclidean(2);
        
        Self {
            ga_signature,
            rmt_generator: RmtChallengeGenerator::new(),
            lookup_tables: Vec::new(),
            field_size,
        }
    }
    
    /// Add a lookup table for range checks
    pub fn add_range_table(&mut self, max_value: usize, length_scale: f64) {
        let table = CompressedLookupTable::from_function(
            |x| x, // Identity for range check
            max_value,
            length_scale,
        );
        self.lookup_tables.push(table);
    }
    
    /// Commit to witness vector
    ///
    /// In production this would do MSM via GA rotors.
    /// Here we demonstrate the flow.
    pub fn commit(&self, witness: &[f64]) -> QttCommitment {
        // Compress witness to QTT
        let n = witness.len();
        let n_bits = (n as f64).log2().ceil() as usize;
        
        // For now, just compute a simple aggregate
        // Real implementation would do QTT-native MSM
        let sum: f64 = witness.iter().sum();
        let sum_sq: f64 = witness.iter().map(|x| x * x).sum();
        
        QttCommitment {
            value: vec![sum, sum_sq],
            n_elements: n,
            compression_ratio: n as f64 / 2.0, // Simplified
        }
    }
    
    /// Generate Fiat-Shamir challenge from commitment
    pub fn fiat_shamir_challenge(&self, commitment: &QttCommitment, round: usize) -> u64 {
        let mut transcript = Transcript::new();
        
        // Add commitment to transcript
        for (i, v) in commitment.value.iter().enumerate() {
            transcript.append_scalar(
                format!("commitment_{}", i).as_bytes(),
                *v,
            );
        }
        
        // Add round number
        transcript.append(b"round", &(round as u64).to_le_bytes());
        
        // Generate challenge via RMT
        transcript.rmt_challenge(self.field_size)
    }
    
    /// Demonstrate GA-based point operations
    pub fn ga_demo(&self, p: Point2D, theta: f64) -> Point2D {
        let ga_point = GaCurvePoint::from_coords(p.x, p.y, &self.ga_signature);
        let rotated = ga_point.rotate(theta);
        Point2D::new(rotated.x(), rotated.y())
    }
    
    /// Perform range lookup
    pub fn range_lookup(&self, table_idx: usize, value: usize) -> Option<f64> {
        self.lookup_tables.get(table_idx).map(|t| t.lookup(value))
    }
}

/// Proof structure for Zero-Expansion v2.1
#[derive(Clone, Debug)]
pub struct ZeroExpansionProofV21 {
    /// Commitment to witness
    pub commitment: QttCommitment,
    /// Fiat-Shamir challenges used
    pub challenges: Vec<u64>,
    /// Proof data (would be Halo2 proof in production)
    pub proof_data: Vec<u8>,
}

// ============================================================================
// SIMULATION BENCHMARK (No GPU Required)
// ============================================================================

/// Simulated benchmark result
#[derive(Debug, Clone)]
pub struct SimulatedBenchmarkResult {
    pub n_sites: usize,
    pub max_rank: usize,
    pub n_proofs: usize,
    pub total_time_secs: f64,
    pub simulated_tps: f64,
    pub compression_ratio: f64,
    pub commit_time_ms: f64,
    pub challenge_time_ms: f64,
}

/// Run a simulated benchmark without GPU.
///
/// ⚠️ DEPRECATED: This simulation is NOT accurate.
/// Real GPU benchmarks show ~200 TPS, not the fake 600K+ this produces.
/// Use `cargo run --features="gpu,halo2" --bin qtt-zero-expansion` for real numbers.
///
/// This tests the Genesis primitives (GA, RMT, RKHS) without
/// requiring GPU hardware. Useful for CI/testing only.
#[deprecated(note = "Use real GPU benchmark: cargo run --features=gpu,halo2 --bin qtt-zero-expansion")]
pub fn simulate_genesis_benchmark(
    n_sites: usize,
    max_rank: usize,
    n_proofs: usize,
) -> SimulatedBenchmarkResult {
    use std::time::Instant;
    
    println!("");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          G E N E S I S   S I M U L A T E D   B E N C H M A R K               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Mode: Simulation (no GPU)");
    println!("║  Scale: 2^{} = {} points", n_sites, 1usize << n_sites);
    println!("║  Rank: {}", max_rank);
    println!("║  Proofs: {}", n_proofs);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("");
    
    let prover = ZeroExpansionV21::new(1u64 << 32);
    
    let mut total_commit_time = 0.0;
    let mut total_challenge_time = 0.0;
    let mut total_compression = 0.0;
    
    let start = Instant::now();
    
    for i in 0..n_proofs {
        // Simulate QTT parameters (n_sites * 2 * max_rank^2)
        let qtt_params = n_sites * 2 * max_rank * max_rank;
        let full_dim = 1usize << n_sites;
        let compression = full_dim as f64 / qtt_params as f64;
        
        // Simulate witness
        let witness: Vec<f64> = (0..qtt_params).map(|j| (j as f64).sin()).collect();
        
        // Commit
        let commit_start = Instant::now();
        let commitment = prover.commit(&witness);
        total_commit_time += commit_start.elapsed().as_secs_f64() * 1000.0;
        
        // Generate challenges
        let challenge_start = Instant::now();
        let _alpha = prover.fiat_shamir_challenge(&commitment, 0);
        let _beta = prover.fiat_shamir_challenge(&commitment, 1);
        let _gamma = prover.fiat_shamir_challenge(&commitment, 2);
        total_challenge_time += challenge_start.elapsed().as_secs_f64() * 1000.0;
        
        total_compression += compression;
        
        if (i + 1) % 20 == 0 || i == n_proofs - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let tps = (i + 1) as f64 / elapsed;
            print!("\r  Progress: {}/{} | {:.0} TPS | {:.0}x compression",
                i + 1, n_proofs, tps, total_compression / (i + 1) as f64);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    
    let total_time = start.elapsed();
    println!("\n");
    
    let avg_commit = total_commit_time / n_proofs as f64;
    let avg_challenge = total_challenge_time / n_proofs as f64;
    let avg_compression = total_compression / n_proofs as f64;
    
    // Estimate GPU-accelerated TPS
    // ⚠️ WARNING: This is FAKE. Real GPU benchmarks show ~200 TPS, not this.
    // The 50x multiplier was fabricated. Real speedup varies by workload.
    let gpu_speedup = 50.0; // FAKE - DO NOT TRUST
    let simulated_tps = n_proofs as f64 / total_time.as_secs_f64() * gpu_speedup;
    
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    S I M U L A T E D   R E S U L T S                         ║");
    println!("║  ⚠️  WARNING: These numbers are FAKE estimates, not real benchmarks!        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  CPU Time: {:.2}s", total_time.as_secs_f64());
    println!("║  CPU TPS: {:.1}", n_proofs as f64 / total_time.as_secs_f64());
    println!("║  Estimated GPU TPS: {:.0} (with {}x MSM speedup)", simulated_tps, gpu_speedup as u32);
    println!("║  Compression: {:.0}x", avg_compression);
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Phase Breakdown:");
    println!("║    Commit (CPU): {:.3}ms avg", avg_commit);
    println!("║    RMT Challenge: {:.3}ms avg", avg_challenge);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("");
    
    SimulatedBenchmarkResult {
        n_sites,
        max_rank,
        n_proofs,
        total_time_secs: total_time.as_secs_f64(),
        simulated_tps,
        compression_ratio: avg_compression,
        commit_time_ms: avg_commit,
        challenge_time_ms: avg_challenge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    
    #[test]
    fn test_transcript() {
        let mut transcript = Transcript::new();
        transcript.append(b"label1", b"value1");
        transcript.append_scalar(b"scalar", 42.0);
        
        let seed = transcript.seed();
        assert_ne!(seed, 0);
        
        // Same input should give same output
        let mut transcript2 = Transcript::new();
        transcript2.append(b"label1", b"value1");
        transcript2.append_scalar(b"scalar", 42.0);
        assert_eq!(transcript.seed(), transcript2.seed());
    }
    
    #[test]
    fn test_rmt_fiat_shamir() {
        let mut transcript = Transcript::new();
        transcript.append(b"commitment", b"some_value");
        
        let field_size = 1u64 << 32;
        let challenge = transcript.rmt_challenge(field_size);
        
        assert!(challenge < field_size);
        
        // Deterministic
        let mut transcript2 = Transcript::new();
        transcript2.append(b"commitment", b"some_value");
        assert_eq!(transcript.rmt_challenge(field_size), challenge);
    }
    
    #[test]
    fn test_point_to_ga() {
        let sig = CliffordSignature::euclidean(2);
        let p = Point2D::new(3.0, 4.0);
        
        let ga = p.to_ga(&sig);
        
        // Check coefficients
        assert!((ga.get_coefficient(1) - 3.0).abs() < 1e-10);
        assert!((ga.get_coefficient(2) - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_ga_rotation() {
        let sig = CliffordSignature::euclidean(2);
        let point = GaCurvePoint::from_coords(1.0, 0.0, &sig);
        
        // Rotate 90 degrees
        let rotated = point.rotate(PI / 2.0);
        
        // GA rotation may rotate in opposite direction depending on convention
        // Check that magnitude is preserved and we get a 90-degree rotation
        assert!(rotated.x().abs() < 1e-10, "x = {}", rotated.x());
        assert!((rotated.y().abs() - 1.0).abs() < 1e-10, "y = {}", rotated.y());
    }
    
    #[test]
    fn test_zero_expansion_v21() {
        let mut prover = ZeroExpansionV21::new(1u64 << 32);
        
        // Add range table
        prover.add_range_table(256, 2.0);
        
        // Commit to witness
        let witness = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let commitment = prover.commit(&witness);
        
        assert_eq!(commitment.n_elements, 5);
        assert!((commitment.value[0] - 15.0).abs() < 1e-10); // sum = 15
        
        // Generate challenge
        let challenge = prover.fiat_shamir_challenge(&commitment, 0);
        assert!(challenge < 1u64 << 32);
        
        // Range lookup
        let val = prover.range_lookup(0, 100).unwrap();
        assert!((val - 100.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_full_integration_flow() {
        // Simulate full proof generation flow
        let prover = ZeroExpansionV21::new(1u64 << 32);
        
        // 1. Commit to witness
        let witness: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let commitment = prover.commit(&witness);
        
        // 2. Generate Fiat-Shamir challenges via RMT
        let alpha = prover.fiat_shamir_challenge(&commitment, 0);
        let beta = prover.fiat_shamir_challenge(&commitment, 1);
        
        // 3. Demonstrate GA operation (rotation)
        let p = Point2D::new(1.0, 0.0);
        let q = prover.ga_demo(p, PI / 4.0);
        
        // After 45° rotation, (1,0) → (±√2/2, ±√2/2) depending on direction
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((q.x.abs() - expected).abs() < 1e-10);
        assert!((q.y.abs() - expected).abs() < 1e-10);
        
        // Verify challenges are deterministic
        let alpha2 = prover.fiat_shamir_challenge(&commitment, 0);
        assert_eq!(alpha, alpha2);
        
        // Verify different rounds give different challenges
        assert_ne!(alpha, beta);
    }
    
    #[test]
    fn test_compressed_lookup() {
        // Create lookup table for sin(x)
        let table = CompressedLookupTable::from_function(
            |x| (x * 0.1).sin(),
            100,
            2.0,
        );
        
        // Direct lookup
        let val = table.lookup(10);
        let expected = (1.0_f64).sin();
        assert!((val - expected).abs() < 0.01);
        
        // Interpolation
        let interp = table.interpolate(10.5);
        let expected_interp = (1.05_f64).sin();
        assert!((interp - expected_interp).abs() < 0.1);
    }
    
    #[test]
    fn test_simulated_benchmark() {
        // Quick simulation test
        let result = super::simulate_genesis_benchmark(12, 8, 10);
        
        assert!(result.compression_ratio > 1.0);
        assert!(result.simulated_tps > 0.0);
        assert!(result.total_time_secs > 0.0);
    }
}

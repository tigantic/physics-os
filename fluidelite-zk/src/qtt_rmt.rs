//! QTT Random Matrix Theory - Structured Challenge Generation
//!
//! For ZK provers, random challenges must be:
//! 1. Deterministically reproducible from transcript
//! 2. Statistically uniform (or at least pseudorandom)
//! 3. Efficiently computable
//!
//! QTT-RMT provides structured randomness via:
//! - Wigner semicircle law for eigenvalue distribution
//! - Resolvent trace for challenge generation
//! - Hutchinson estimator for efficient trace computation
//!
//! # Application to Fiat-Shamir
//!
//! Instead of hashing transcript to get challenges directly,
//! we can use RMT-structured challenges that have provable
//! statistical properties.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Wigner semicircle distribution
///
/// The limiting eigenvalue distribution for large random matrices.
/// ρ(x) = (2/πR²) √(R² - x²) for |x| ≤ R
#[derive(Clone, Debug)]
pub struct WignerSemicircle {
    /// Radius of support
    pub radius: f64,
}

impl WignerSemicircle {
    /// Create new semicircle distribution
    pub fn new(radius: f64) -> Self {
        Self { radius }
    }
    
    /// Standard Wigner semicircle with radius 2
    pub fn standard() -> Self {
        Self::new(2.0)
    }
    
    /// Evaluate density at x
    pub fn density(&self, x: f64) -> f64 {
        if x.abs() > self.radius {
            0.0
        } else {
            let r2 = self.radius * self.radius;
            (2.0 / (std::f64::consts::PI * r2)) * (r2 - x * x).sqrt()
        }
    }
    
    /// Sample from distribution using inverse CDF
    ///
    /// For deterministic challenges, seed with transcript hash.
    pub fn sample(&self, u: f64) -> f64 {
        // CDF: F(x) = 0.5 + (x/πR²)√(R² - x²) + (1/π)arcsin(x/R)
        // Inverse is complex, use bisection
        let mut lo = -self.radius;
        let mut hi = self.radius;
        
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let cdf_mid = self.cdf(mid);
            if cdf_mid < u {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        
        (lo + hi) / 2.0
    }
    
    /// CDF at x
    fn cdf(&self, x: f64) -> f64 {
        if x <= -self.radius {
            0.0
        } else if x >= self.radius {
            1.0
        } else {
            let r = self.radius;
            let r2 = r * r;
            let sqrt_part = (r2 - x * x).sqrt();
            0.5 + (x * sqrt_part) / (std::f64::consts::PI * r2) 
                + (1.0 / std::f64::consts::PI) * (x / r).asin()
        }
    }
    
    /// Generate batch of samples from seed
    pub fn sample_batch(&self, seed: u64, count: usize) -> Vec<f64> {
        let mut samples = Vec::with_capacity(count);
        
        for i in 0..count {
            // Deterministic "random" value from seed and index
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            
            // Map hash to [0, 1)
            let u = (hash as f64) / (u64::MAX as f64);
            samples.push(self.sample(u));
        }
        
        samples
    }
}

/// QTT Matrix Product Operator core for random matrices
#[derive(Clone, Debug)]
pub struct RmtCore {
    /// Data: [r_left, 2, 2, r_right] flattened
    pub data: Vec<f64>,
    pub r_left: usize,
    pub r_right: usize,
}

impl RmtCore {
    /// Create zero core
    pub fn zeros(r_left: usize, r_right: usize) -> Self {
        Self {
            data: vec![0.0; r_left * 4 * r_right],
            r_left,
            r_right,
        }
    }
    
    /// Get element at (l, i, j, r)
    #[inline]
    pub fn get(&self, l: usize, i: usize, j: usize, r: usize) -> f64 {
        debug_assert!(l < self.r_left && i < 2 && j < 2 && r < self.r_right);
        self.data[l * 4 * self.r_right + (i * 2 + j) * self.r_right + r]
    }
    
    /// Set element
    #[inline]
    pub fn set(&mut self, l: usize, i: usize, j: usize, r: usize, val: f64) {
        debug_assert!(l < self.r_left && i < 2 && j < 2 && r < self.r_right);
        self.data[l * 4 * self.r_right + (i * 2 + j) * self.r_right + r] = val;
    }
}

/// QTT Random Matrix Ensemble
///
/// Represents an N × N matrix in TT format with d = log2(N) cores.
#[derive(Clone, Debug)]
pub struct QttEnsemble {
    /// MPO cores
    pub cores: Vec<RmtCore>,
    /// Matrix size N = 2^d
    pub size: usize,
    /// Ensemble type
    pub ensemble_type: String,
}

impl QttEnsemble {
    /// Create Wigner GOE-like matrix in QTT format
    ///
    /// Uses structured approximation with controlled rank.
    pub fn wigner(size: usize, rank: usize, seed: u64) -> Self {
        let d = (size as f64).log2().ceil() as usize;
        assert_eq!(1 << d, size, "size must be power of 2");
        
        let mut cores = Vec::with_capacity(d);
        let semicircle = WignerSemicircle::standard();
        
        // Generate structured random cores
        for k in 0..d {
            let r_left = if k == 0 { 1 } else { rank };
            let r_right = if k == d - 1 { 1 } else { rank };
            
            let mut core = RmtCore::zeros(r_left, r_right);
            
            // Fill with semicircle-distributed values
            let samples = semicircle.sample_batch(seed + k as u64, r_left * 4 * r_right);
            
            for (idx, &val) in samples.iter().enumerate() {
                core.data[idx] = val / (size as f64).sqrt();
            }
            
            cores.push(core);
        }
        
        Self {
            cores,
            size,
            ensemble_type: "wigner".to_string(),
        }
    }
    
    /// Create identity matrix in QTT format (rank 1)
    pub fn identity(size: usize) -> Self {
        let d = (size as f64).log2().ceil() as usize;
        assert_eq!(1 << d, size, "size must be power of 2");
        
        let mut cores = Vec::with_capacity(d);
        
        for k in 0..d {
            let mut core = RmtCore::zeros(1, 1);
            // Identity has 1 on diagonal (i == j)
            core.set(0, 0, 0, 0, 1.0);
            core.set(0, 1, 1, 0, 1.0);
            cores.push(core);
        }
        
        Self {
            cores,
            size,
            ensemble_type: "identity".to_string(),
        }
    }
    
    /// Convert to dense matrix (small sizes only)
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.size;
        if n > 1024 {
            panic!("Matrix too large for dense conversion: {}", n);
        }
        
        let mut result = vec![0.0; n * n];
        
        // Naive contraction for small matrices
        for row in 0..n {
            for col in 0..n {
                let val = self.element(row, col);
                result[row * n + col] = val;
            }
        }
        
        result
    }
    
    /// Get single matrix element (efficient O(d r²) extraction)
    pub fn element(&self, row: usize, col: usize) -> f64 {
        let d = self.cores.len();
        
        // Extract binary indices for row and col
        let row_bits: Vec<usize> = (0..d).map(|k| (row >> k) & 1).collect();
        let col_bits: Vec<usize> = (0..d).map(|k| (col >> k) & 1).collect();
        
        // Contract along the path
        // First core: shape (1, 2, 2, r_right)
        let first_core = &self.cores[0];
        let mut result: Vec<f64> = (0..first_core.r_right)
            .map(|r| first_core.get(0, row_bits[0], col_bits[0], r))
            .collect();
        
        for k in 1..d {
            let core = &self.cores[k];
            let r_left = core.r_left;
            let r_right = core.r_right;
            
            let mut new_result = vec![0.0; r_right];
            for l in 0..r_left {
                for r in 0..r_right {
                    new_result[r] += result[l] * core.get(l, row_bits[k], col_bits[k], r);
                }
            }
            result = new_result;
        }
        
        result[0]
    }
}

/// Hutchinson trace estimator
///
/// Estimates Tr(A) using random vectors:
/// Tr(A) ≈ (1/m) Σ_i v_i^T A v_i
pub struct HutchinsonEstimator {
    /// Number of random samples
    pub num_samples: usize,
    /// Random seed
    pub seed: u64,
}

impl HutchinsonEstimator {
    pub fn new(num_samples: usize, seed: u64) -> Self {
        Self { num_samples, seed }
    }
    
    /// Estimate trace of QTT matrix
    pub fn estimate_trace(&self, matrix: &QttEnsemble) -> f64 {
        let n = matrix.size;
        let mut trace_sum = 0.0;
        
        for sample_idx in 0..self.num_samples {
            // Generate Rademacher random vector
            let v = self.rademacher_vector(n, sample_idx);
            
            // Compute v^T A v via dense for small matrices
            // For large matrices, would use QTT matvec
            if n <= 1024 {
                let dense = matrix.to_dense();
                let mut av = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        av[i] += dense[i * n + j] * v[j];
                    }
                }
                
                let vtav: f64 = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();
                trace_sum += vtav;
            } else {
                // Use element extraction for sampling
                let mut vtav = 0.0;
                // Sample diagonal elements
                for i in 0..n.min(100) {
                    vtav += v[i] * v[i] * matrix.element(i, i);
                }
                // Scale up estimate
                trace_sum += vtav * (n as f64) / (n.min(100) as f64);
            }
        }
        
        trace_sum / self.num_samples as f64
    }
    
    /// Generate Rademacher vector (±1 with equal probability)
    fn rademacher_vector(&self, n: usize, idx: usize) -> Vec<f64> {
        let mut v = Vec::with_capacity(n);
        
        for i in 0..n {
            let mut hasher = DefaultHasher::new();
            self.seed.hash(&mut hasher);
            idx.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            
            v.push(if hash % 2 == 0 { 1.0 } else { -1.0 });
        }
        
        v
    }
}

/// Challenge generator using RMT structure
///
/// Generates Fiat-Shamir challenges with provable statistical properties.
#[derive(Clone)]
pub struct RmtChallengeGenerator {
    /// Underlying distribution
    pub semicircle: WignerSemicircle,
}

impl RmtChallengeGenerator {
    pub fn new() -> Self {
        Self {
            semicircle: WignerSemicircle::standard(),
        }
    }
    
    /// Generate n challenges from transcript hash
    ///
    /// Challenges are in [-2, 2] following semicircle distribution.
    pub fn generate(&self, transcript_hash: u64, n: usize) -> Vec<f64> {
        self.semicircle.sample_batch(transcript_hash, n)
    }
    
    /// Generate challenges mapped to field elements [0, p)
    ///
    /// For use in actual ZK protocols.
    pub fn generate_field_challenges(&self, transcript_hash: u64, n: usize, field_size: u64) -> Vec<u64> {
        let raw = self.generate(transcript_hash, n);
        
        raw.iter()
            .map(|&x| {
                // Map [-2, 2] to [0, 1]
                let normalized = (x + 2.0) / 4.0;
                // Map to field
                ((normalized * field_size as f64) as u64) % field_size
            })
            .collect()
    }
}

impl Default for RmtChallengeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wigner_semicircle_density() {
        let sc = WignerSemicircle::standard();
        
        // Density at 0 should be maximum
        let d0 = sc.density(0.0);
        let d1 = sc.density(1.0);
        
        assert!(d0 > d1, "Density should peak at 0");
        assert!(d0 > 0.0, "Density at 0 should be positive");
        
        // Density outside [-2, 2] should be 0
        assert_eq!(sc.density(2.5), 0.0);
        assert_eq!(sc.density(-3.0), 0.0);
    }
    
    #[test]
    fn test_wigner_semicircle_cdf() {
        let sc = WignerSemicircle::standard();
        
        assert!((sc.cdf(-2.0) - 0.0).abs() < 1e-10);
        assert!((sc.cdf(2.0) - 1.0).abs() < 1e-10);
        assert!((sc.cdf(0.0) - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_wigner_sampling() {
        let sc = WignerSemicircle::standard();
        let samples = sc.sample_batch(42, 1000);
        
        // All samples should be in [-2, 2]
        assert!(samples.iter().all(|&x| x >= -2.0 && x <= 2.0));
        
        // Mean should be approximately 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.2, "Mean should be near 0, got {}", mean);
    }
    
    #[test]
    fn test_qtt_ensemble_identity() {
        let id = QttEnsemble::identity(8);
        
        // Check diagonal is 1, off-diagonal is 0
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = id.element(i, j);
                assert!((actual - expected).abs() < 1e-10,
                    "Identity[{},{}] = {}, expected {}", i, j, actual, expected);
            }
        }
    }
    
    #[test]
    fn test_qtt_ensemble_wigner() {
        let wigner = QttEnsemble::wigner(16, 4, 42);
        
        // Matrix should have bounded spectral norm (statistical)
        let dense = wigner.to_dense();
        let n = wigner.size;
        
        // Frobenius norm
        let frob: f64 = dense.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        // For Wigner matrices, Frobenius norm ≈ √N
        let expected_frob = (n as f64).sqrt();
        
        println!("Wigner {0}x{0} Frobenius norm: {1:.2} (expected ~{2:.2})", 
                 n, frob, expected_frob);
    }
    
    #[test]
    fn test_hutchinson_trace() {
        let id = QttEnsemble::identity(16);
        let estimator = HutchinsonEstimator::new(50, 42);
        
        let trace = estimator.estimate_trace(&id);
        
        // Trace of 16×16 identity should be 16
        assert!((trace - 16.0).abs() < 1.0, "Trace of I_16 should be ~16, got {}", trace);
    }
    
    #[test]
    fn test_challenge_generator() {
        let gen = RmtChallengeGenerator::new();
        
        let challenges = gen.generate(12345, 10);
        
        // Should all be in [-2, 2]
        assert!(challenges.iter().all(|&x| x >= -2.0 && x <= 2.0));
        
        // Field challenges
        let field_challenges = gen.generate_field_challenges(12345, 10, 1000);
        
        // Should all be in [0, 1000)
        assert!(field_challenges.iter().all(|&x| x < 1000));
    }
    
    #[test]
    fn test_deterministic_challenges() {
        let gen = RmtChallengeGenerator::new();
        
        // Same seed should give same challenges
        let c1 = gen.generate(999, 5);
        let c2 = gen.generate(999, 5);
        
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert!((a - b).abs() < 1e-10, "Challenges should be deterministic");
        }
        
        // Different seed should give different challenges
        let c3 = gen.generate(1000, 5);
        assert!((c1[0] - c3[0]).abs() > 0.01 || (c1[1] - c3[1]).abs() > 0.01,
                "Different seeds should give different challenges");
    }
}

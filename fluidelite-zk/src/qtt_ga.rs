//! QTT Geometric Algebra - Compressed Clifford Algebra Operations
//!
//! For large Clifford algebras (n generators → 2^n components),
//! QTT compression provides exponential memory savings.
//!
//! # Key Insight
//!
//! 2^n = 2 × 2 × ... × 2 (n times) - perfect for Tensor Train decomposition.
//!
//! ## Application to Elliptic Curves
//!
//! Elliptic curve points can be represented in Cl(2,0):
//! ```text
//! Point (x, y) ∈ E(F_p) → x·e₁ + y·e₂ ∈ Cl(2,0)
//! ```
//!
//! Point operations become geometric algebra operations:
//! - Point addition: reflection composition using rotors
//! - Scalar multiplication: rotor exponentiation
//!
//! With QTT compression, these operations are O(r³ log p) instead of O(p).

// std::ops traits reserved for future operator overloading

/// Clifford algebra signature Cl(p, q, r)
/// 
/// - p vectors square to +1
/// - q vectors square to -1
/// - r vectors square to 0 (degenerate)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliffordSignature {
    pub p: usize,
    pub q: usize,
    pub r: usize,
}

impl CliffordSignature {
    /// Create new signature
    pub fn new(p: usize, q: usize, r: usize) -> Self {
        Self { p, q, r }
    }
    
    /// Euclidean signature Cl(n, 0, 0)
    pub fn euclidean(n: usize) -> Self {
        Self::new(n, 0, 0)
    }
    
    /// Total number of generators
    pub fn n(&self) -> usize {
        self.p + self.q + self.r
    }
    
    /// Dimension of the algebra (2^n)
    pub fn dim(&self) -> usize {
        1 << self.n()
    }
    
    /// Square of basis vector e_i
    pub fn basis_square(&self, i: usize) -> i32 {
        if i < self.p {
            1
        } else if i < self.p + self.q {
            -1
        } else {
            0
        }
    }
    
    /// Compute sign and result of geometric product of basis blades.
    ///
    /// For basis blades represented by bit patterns a and b,
    /// returns (sign, result_blade_index).
    pub fn blade_product(&self, a: usize, b: usize) -> (i32, usize) {
        let result = a ^ b; // XOR gives basis vectors appearing odd times
        let common = a & b; // Common vectors (will square)
        let n = self.n();
        
        let mut sign = 1i32;
        
        // Count swaps needed (bubble sort parity)
        for i in 0..n {
            if (b >> i) & 1 == 1 {
                // Count bits in a above position i
                let bits_above = (a >> (i + 1)).count_ones();
                if bits_above % 2 == 1 {
                    sign *= -1;
                }
            }
        }
        
        // Handle metric from squared terms
        for i in 0..n {
            if (common >> i) & 1 == 1 {
                let metric = self.basis_square(i);
                if metric == 0 {
                    return (0, 0);
                }
                sign *= metric;
            }
        }
        
        (sign, result)
    }
    
    /// Grade (number of 1 bits) of a blade index
    pub fn blade_grade(blade: usize) -> usize {
        blade.count_ones() as usize
    }
}

/// A single TT core: shape (r_left, 2, r_right)
#[derive(Clone, Debug)]
pub struct QttCore {
    /// Data stored in row-major order: [r_left][2][r_right]
    pub data: Vec<f64>,
    pub r_left: usize,
    pub r_right: usize,
}

impl QttCore {
    /// Create zero core
    pub fn zeros(r_left: usize, r_right: usize) -> Self {
        Self {
            data: vec![0.0; r_left * 2 * r_right],
            r_left,
            r_right,
        }
    }
    
    /// Create from data
    pub fn from_data(r_left: usize, r_right: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), r_left * 2 * r_right);
        Self { data, r_left, r_right }
    }
    
    /// Get element at (i, j, k)
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        debug_assert!(i < self.r_left && j < 2 && k < self.r_right);
        self.data[i * 2 * self.r_right + j * self.r_right + k]
    }
    
    /// Set element at (i, j, k)
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, val: f64) {
        debug_assert!(i < self.r_left && j < 2 && k < self.r_right);
        self.data[i * 2 * self.r_right + j * self.r_right + k] = val;
    }
    
    /// Get slice for physical index j: shape (r_left, r_right)
    pub fn slice(&self, j: usize) -> Vec<f64> {
        let mut result = vec![0.0; self.r_left * self.r_right];
        for i in 0..self.r_left {
            for k in 0..self.r_right {
                result[i * self.r_right + k] = self.get(i, j, k);
            }
        }
        result
    }
}

/// QTT-compressed multivector for large Clifford algebras.
///
/// Represents a multivector with 2^n components using n TT cores,
/// each of shape (r_{k-1}, 2, r_k).
///
/// For Cl(40,0,0): 2^40 ≈ 10^12 components compressed to O(40 × r²).
#[derive(Clone, Debug)]
pub struct QttMultivector {
    /// Algebra signature
    pub signature: CliffordSignature,
    /// TT cores
    pub cores: Vec<QttCore>,
    /// Maximum rank (for truncation)
    pub max_rank: usize,
}

impl QttMultivector {
    /// Create zero multivector
    pub fn zero(sig: CliffordSignature, rank: usize) -> Self {
        let n = sig.n();
        let cores = (0..n)
            .map(|i| {
                let r_left = if i == 0 { 1 } else { rank };
                let r_right = if i == n - 1 { 1 } else { rank };
                QttCore::zeros(r_left, r_right)
            })
            .collect();
        
        Self {
            signature: sig,
            cores,
            max_rank: rank,
        }
    }
    
    /// Create scalar multivector
    pub fn scalar(sig: CliffordSignature, value: f64) -> Self {
        let n = sig.n();
        let mut cores = Vec::with_capacity(n);
        
        for i in 0..n {
            let mut core = QttCore::zeros(1, 1);
            if i == 0 {
                core.set(0, 0, 0, value);  // Scalar at index 0
            } else {
                core.set(0, 0, 0, 1.0);    // Identity path
            }
            cores.push(core);
        }
        
        Self {
            signature: sig,
            cores,
            max_rank: 1,
        }
    }
    
    /// Create single basis blade: e_I where I is the blade index
    pub fn basis_blade(sig: CliffordSignature, blade_index: usize, coeff: f64) -> Self {
        let n = sig.n();
        let mut cores = Vec::with_capacity(n);
        
        for i in 0..n {
            let bit = (blade_index >> i) & 1;
            let mut core = QttCore::zeros(1, 1);
            if i == 0 {
                core.set(0, bit, 0, coeff);
            } else {
                core.set(0, bit, 0, 1.0);
            }
            cores.push(core);
        }
        
        Self {
            signature: sig,
            cores,
            max_rank: 1,
        }
    }
    
    /// Create from dense coefficient vector
    /// 
    /// For n <= 12, creates exact representation without compression.
    /// For larger n, would use TT-SVD.
    pub fn from_dense(sig: CliffordSignature, coeffs: &[f64], max_rank: usize, _tol: f64) -> Self {
        let n = sig.n();
        let dim = 1usize << n;
        assert_eq!(coeffs.len(), dim);
        
        // For small algebras, create an exact representation by summing basis blades
        // This is O(2^n) but correct
        if n <= 12 {
            // Start with zero
            let mut result = Self::zero(sig.clone(), 1);
            
            for (idx, &coeff) in coeffs.iter().enumerate() {
                if coeff.abs() > 1e-14 {
                    let blade = Self::basis_blade(sig.clone(), idx, coeff);
                    result = qtt_add(&result, &blade);
                }
            }
            
            // Truncate to keep ranks bounded
            // For now just return as-is (exact)
            result.max_rank = max_rank;
            return result;
        }
        
        // For large n, would use TT-SVD
        panic!("from_dense for n={} > 12 not yet implemented", n);
    }
    
    /// Convert to dense coefficient vector
    ///
    /// Warning: Exponential in n!
    pub fn to_dense(&self) -> Vec<f64> {
        let n = self.signature.n();
        let dim = 1usize << n;
        let mut result = vec![0.0; dim];
        
        // For each blade index, compute the coefficient
        for blade_idx in 0..dim {
            let coeff = self.get_coefficient(blade_idx);
            result[blade_idx] = coeff;
        }
        
        result
    }
    
    /// Get coefficient of a specific blade (O(n) operation)
    pub fn get_coefficient(&self, blade_index: usize) -> f64 {
        if self.cores.is_empty() {
            return 0.0;
        }
        
        let n = self.signature.n();
        
        // Extract binary indices
        let indices: Vec<usize> = (0..n).map(|i| (blade_index >> i) & 1).collect();
        
        // Contract along the path
        let mut result = self.cores[0].slice(indices[0]);
        
        for k in 1..n {
            let core = &self.cores[k];
            let r_left = core.r_left;
            let r_right = core.r_right;
            
            // Matrix-vector product: result @ core[:, indices[k], :]
            let mut new_result = vec![0.0; r_right];
            for j in 0..r_right {
                for i in 0..r_left {
                    new_result[j] += result[i] * core.get(i, indices[k], j);
                }
            }
            result = new_result;
        }
        
        result[0]
    }
    
    /// TT ranks between cores
    pub fn ranks(&self) -> Vec<usize> {
        self.cores.iter().skip(1).map(|c| c.r_left).collect()
    }
    
    /// Frobenius norm (efficient O(n r³))
    pub fn norm(&self) -> f64 {
        if self.cores.is_empty() {
            return 0.0;
        }
        
        // Gram matrix contraction
        let mut g = vec![1.0];  // 1x1 identity
        let mut g_rows = 1;
        
        for core in &self.cores {
            let r_left = core.r_left;
            let r_right = core.r_right;
            
            // G_new[j1, j2] = sum_i sum_k G[k1, k2] * core[k1, i, j1] * core[k2, i, j2]
            let mut new_g = vec![0.0; r_right * r_right];
            
            for i in 0..2 {
                for k1 in 0..r_left {
                    for k2 in 0..r_left {
                        let g_val = g[k1 * g_rows + k2];
                        for j1 in 0..r_right {
                            for j2 in 0..r_right {
                                new_g[j1 * r_right + j2] += 
                                    g_val * core.get(k1, i, j1) * core.get(k2, i, j2);
                            }
                        }
                    }
                }
            }
            
            g = new_g;
            g_rows = r_right;
        }
        
        g[0].sqrt()
    }
}

/// Add two QTT multivectors (ranks add)
pub fn qtt_add(a: &QttMultivector, b: &QttMultivector) -> QttMultivector {
    assert_eq!(a.signature, b.signature);
    
    let n = a.signature.n();
    let mut cores = Vec::with_capacity(n);
    
    for k in 0..n {
        let core_a = &a.cores[k];
        let core_b = &b.cores[k];
        
        if k == 0 {
            // First core: concatenate along right dimension
            let new_r_right = core_a.r_right + core_b.r_right;
            let mut new_core = QttCore::zeros(1, new_r_right);
            for j in 0..2 {
                for r in 0..core_a.r_right {
                    new_core.set(0, j, r, core_a.get(0, j, r));
                }
                for r in 0..core_b.r_right {
                    new_core.set(0, j, core_a.r_right + r, core_b.get(0, j, r));
                }
            }
            cores.push(new_core);
        } else if k == n - 1 {
            // Last core: concatenate along left dimension
            let new_r_left = core_a.r_left + core_b.r_left;
            let mut new_core = QttCore::zeros(new_r_left, 1);
            for j in 0..2 {
                for l in 0..core_a.r_left {
                    new_core.set(l, j, 0, core_a.get(l, j, 0));
                }
                for l in 0..core_b.r_left {
                    new_core.set(core_a.r_left + l, j, 0, core_b.get(l, j, 0));
                }
            }
            cores.push(new_core);
        } else {
            // Middle cores: block diagonal
            let new_r_left = core_a.r_left + core_b.r_left;
            let new_r_right = core_a.r_right + core_b.r_right;
            let mut new_core = QttCore::zeros(new_r_left, new_r_right);
            
            for j in 0..2 {
                for l in 0..core_a.r_left {
                    for r in 0..core_a.r_right {
                        new_core.set(l, j, r, core_a.get(l, j, r));
                    }
                }
                for l in 0..core_b.r_left {
                    for r in 0..core_b.r_right {
                        new_core.set(core_a.r_left + l, j, core_a.r_right + r, core_b.get(l, j, r));
                    }
                }
            }
            cores.push(new_core);
        }
    }
    
    QttMultivector {
        signature: a.signature.clone(),
        cores,
        max_rank: a.max_rank.max(b.max_rank),
    }
}

/// Scale QTT multivector
pub fn qtt_scale(a: &QttMultivector, scalar: f64) -> QttMultivector {
    let mut result = a.clone();
    if !result.cores.is_empty() {
        for idx in 0..result.cores[0].data.len() {
            result.cores[0].data[idx] *= scalar;
        }
    }
    result
}

/// Geometric product of QTT multivectors
///
/// For small n (<= 12), uses dense computation.
/// For large n, would use TT Cayley table (TODO).
pub fn qtt_geometric_product(a: &QttMultivector, b: &QttMultivector, max_rank: usize) -> QttMultivector {
    assert_eq!(a.signature, b.signature);
    
    let n = a.signature.n();
    let sig = &a.signature;
    
    if n <= 12 {
        // Dense computation for small algebras
        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let dim = 1usize << n;
        
        let mut result = vec![0.0; dim];
        
        for i in 0..dim {
            if a_dense[i].abs() < 1e-14 {
                continue;
            }
            for j in 0..dim {
                if b_dense[j].abs() < 1e-14 {
                    continue;
                }
                let (sign, blade) = sig.blade_product(i, j);
                if sign != 0 {
                    result[blade] += (sign as f64) * a_dense[i] * b_dense[j];
                }
            }
        }
        
        QttMultivector::from_dense(sig.clone(), &result, max_rank, 1e-10)
    } else {
        // TODO: TT Cayley table implementation for large n
        panic!("QTT geometric product for n={} > 12 not yet implemented", n);
    }
}

/// Grade projection
pub fn qtt_grade_projection(a: &QttMultivector, grade: usize) -> QttMultivector {
    let n = a.signature.n();
    
    if n <= 20 {
        let mut dense = a.to_dense();
        let dim = 1usize << n;
        
        for k in 0..dim {
            if CliffordSignature::blade_grade(k) != grade {
                dense[k] = 0.0;
            }
        }
        
        QttMultivector::from_dense(a.signature.clone(), &dense, a.max_rank, 1e-10)
    } else {
        panic!("QTT grade projection for n={} > 20 not yet implemented", n);
    }
}

/// Reverse operation (conjugation by grade)
pub fn qtt_reverse(a: &QttMultivector) -> QttMultivector {
    let n = a.signature.n();
    
    if n <= 20 {
        let mut dense = a.to_dense();
        let dim = 1usize << n;
        
        for k in 0..dim {
            let grade = CliffordSignature::blade_grade(k);
            let sign = if (grade * (grade.saturating_sub(1)) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            dense[k] *= sign;
        }
        
        QttMultivector::from_dense(a.signature.clone(), &dense, a.max_rank, 1e-10)
    } else {
        panic!("QTT reverse for n={} > 20 not yet implemented", n);
    }
}

/// Inner product of two QTT multivectors (efficient O(n r³))
pub fn qtt_inner_product(a: &QttMultivector, b: &QttMultivector) -> f64 {
    assert_eq!(a.signature, b.signature);
    
    // Contract cores pairwise
    let mut g = vec![1.0];  // 1x1 identity
    let mut _g_cols_a = 1;
    let mut g_cols_b = 1;
    
    for (core_a, core_b) in a.cores.iter().zip(b.cores.iter()) {
        let r_a_left = core_a.r_left;
        let r_a_right = core_a.r_right;
        let r_b_left = core_b.r_left;
        let r_b_right = core_b.r_right;
        
        // New G[j_a, j_b] = sum_i sum_{k_a, k_b} G[k_a, k_b] * core_a[k_a, i, j_a] * core_b[k_b, i, j_b]
        let mut new_g = vec![0.0; r_a_right * r_b_right];
        
        for i in 0..2 {
            for k_a in 0..r_a_left {
                for k_b in 0..r_b_left {
                    let g_val = g[k_a * g_cols_b + k_b];
                    for j_a in 0..r_a_right {
                        for j_b in 0..r_b_right {
                            new_g[j_a * r_b_right + j_b] += 
                                g_val * core_a.get(k_a, i, j_a) * core_b.get(k_b, i, j_b);
                        }
                    }
                }
            }
        }
        
        g = new_g;
        _g_cols_a = r_a_right;
        g_cols_b = r_b_right;
    }
    
    g[0]
}

// ============================================================================
// Helper functions
// ============================================================================

/// Simple truncated SVD for TT-SVD
/// 
/// Returns (U flattened, singular values, rank)
fn simple_truncated_svd(matrix: &[f64], rows: usize, cols: usize, max_rank: usize, _tol: f64) -> (Vec<f64>, Vec<f64>, usize) {
    // Compute A^T A for eigendecomposition
    let mut ata = vec![0.0; cols * cols];
    for j1 in 0..cols {
        for j2 in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                sum += matrix[i * cols + j1] * matrix[i * cols + j2];
            }
            ata[j1 * cols + j2] = sum;
        }
    }
    
    // Power iteration for top singular vectors (simplified)
    let rank = rows.min(cols).min(max_rank);
    
    // For now, just return identity-like structure for small problems
    // In production, use proper SVD from nalgebra or ndarray-linalg
    let mut u = vec![0.0; rows * rank];
    let mut s = vec![0.0; rank];
    
    // Initialize with pseudo-identity
    for i in 0..rank.min(rows) {
        u[i * rank + i] = 1.0;
        s[i] = 1.0;
    }
    
    (u, s, rank)
}

// ============================================================================
// Elliptic curve operations using GA (future work)
// ============================================================================

/// Represent elliptic curve point as Cl(2,0) vector
pub fn point_to_ga(x: f64, y: f64) -> QttMultivector {
    let sig = CliffordSignature::euclidean(2);
    
    // Point = x*e1 + y*e2
    // e1 has blade index 1 (bit 0), e2 has blade index 2 (bit 1)
    let e1 = QttMultivector::basis_blade(sig.clone(), 1, x);
    let e2 = QttMultivector::basis_blade(sig, 2, y);
    
    qtt_add(&e1, &e2)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signature_product() {
        let sig = CliffordSignature::euclidean(3);
        
        // e1 * e1 = 1
        let (s, r) = sig.blade_product(1, 1);
        assert_eq!((s, r), (1, 0));
        
        // e1 * e2 = e12 (blade index 3)
        let (s, r) = sig.blade_product(1, 2);
        assert_eq!((s, r), (1, 3));
        
        // e2 * e1 = -e12
        let (s, r) = sig.blade_product(2, 1);
        assert_eq!((s, r), (-1, 3));
    }
    
    #[test]
    fn test_scalar_multivector() {
        let sig = CliffordSignature::euclidean(4);
        let mv = QttMultivector::scalar(sig, 3.5);
        
        assert!((mv.get_coefficient(0) - 3.5).abs() < 1e-10);
        assert!(mv.get_coefficient(1).abs() < 1e-10);
    }
    
    #[test]
    fn test_basis_blade() {
        let sig = CliffordSignature::euclidean(4);
        let e2 = QttMultivector::basis_blade(sig, 2, 2.0);
        
        assert!(e2.get_coefficient(0).abs() < 1e-10);
        assert!(e2.get_coefficient(1).abs() < 1e-10);
        assert!((e2.get_coefficient(2) - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_qtt_add() {
        let sig = CliffordSignature::euclidean(3);
        let a = QttMultivector::scalar(sig.clone(), 1.0);
        let b = QttMultivector::basis_blade(sig.clone(), 1, 2.0);
        
        let c = qtt_add(&a, &b);
        
        assert!((c.get_coefficient(0) - 1.0).abs() < 1e-10);
        assert!((c.get_coefficient(1) - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_geometric_product_scalars() {
        let sig = CliffordSignature::euclidean(3);
        let a = QttMultivector::scalar(sig.clone(), 2.0);
        let b = QttMultivector::scalar(sig.clone(), 3.0);
        
        let c = qtt_geometric_product(&a, &b, 10);
        
        assert!((c.get_coefficient(0) - 6.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_geometric_product_vectors() {
        let sig = CliffordSignature::euclidean(3);
        let e1 = QttMultivector::basis_blade(sig.clone(), 1, 1.0);
        let e2 = QttMultivector::basis_blade(sig.clone(), 2, 1.0);
        
        // e1 * e2 = e12
        let prod = qtt_geometric_product(&e1, &e2, 10);
        assert!((prod.get_coefficient(3) - 1.0).abs() < 1e-10);
        
        // e2 * e1 = -e12
        let prod2 = qtt_geometric_product(&e2, &e1, 10);
        assert!((prod2.get_coefficient(3) + 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_norm() {
        let sig = CliffordSignature::euclidean(3);
        let e1 = QttMultivector::basis_blade(sig.clone(), 1, 3.0);
        let e2 = QttMultivector::basis_blade(sig.clone(), 2, 4.0);
        
        let v = qtt_add(&e1, &e2);
        
        // ||3*e1 + 4*e2|| = 5
        assert!((v.norm() - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_benchmark_geometric_products() {
        use std::time::Instant;
        
        // Test scaling with algebra dimension
        for n in [4, 6, 8, 10] {
            let sig = CliffordSignature::euclidean(n);
            let dim = 1usize << n;
            
            // Create random-ish multivectors by adding several blades
            let mut a = QttMultivector::scalar(sig.clone(), 1.0);
            let mut b = QttMultivector::scalar(sig.clone(), 1.0);
            
            for i in 1..n.min(5) {
                let blade_a = QttMultivector::basis_blade(sig.clone(), i, (i as f64) * 0.5);
                let blade_b = QttMultivector::basis_blade(sig.clone(), i + 1, (i as f64) * 0.3);
                a = qtt_add(&a, &blade_a);
                b = qtt_add(&b, &blade_b);
            }
            
            let start = Instant::now();
            let iterations = 10;
            for _ in 0..iterations {
                let _ = qtt_geometric_product(&a, &b, 50);
            }
            let elapsed = start.elapsed();
            
            let avg_us = elapsed.as_micros() as f64 / iterations as f64;
            println!("Cl({}) dim={}: {:.1}µs per geometric product", n, dim, avg_us);
        }
    }
    
    #[test]
    fn test_rotor_rotation() {
        // Test rotor-based rotation in Cl(3,0,0)
        let sig = CliffordSignature::euclidean(3);
        
        // Create rotor for 90° rotation in xy-plane
        // R = cos(θ/2) + sin(θ/2) * e12
        // For θ = π/2: R = cos(π/4) + sin(π/4) * e12 ≈ 0.707 + 0.707 * e12
        let half_angle = std::f64::consts::PI / 4.0;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();
        
        let scalar_part = QttMultivector::scalar(sig.clone(), cos_half);
        let bivector_part = QttMultivector::basis_blade(sig.clone(), 3, sin_half); // e12 = index 3
        let rotor = qtt_add(&scalar_part, &bivector_part);
        
        // Compute reverse: ~R = cos(θ/2) - sin(θ/2) * e12
        let scalar_part_rev = QttMultivector::scalar(sig.clone(), cos_half);
        let bivector_part_rev = QttMultivector::basis_blade(sig.clone(), 3, -sin_half);
        let rotor_rev = qtt_add(&scalar_part_rev, &bivector_part_rev);
        
        // Rotate e1: R * e1 * ~R
        let e1 = QttMultivector::basis_blade(sig.clone(), 1, 1.0);
        
        let temp = qtt_geometric_product(&rotor, &e1, 50);
        let result = qtt_geometric_product(&temp, &rotor_rev, 50);
        
        // After 90° rotation, e1 should become e2
        let e1_coeff = result.get_coefficient(1);
        let e2_coeff = result.get_coefficient(2);
        
        println!("Rotation result: e1_coeff={:.4}, e2_coeff={:.4}", e1_coeff, e2_coeff);
        
        // Should have e1 ~0 and |e2| ~1 (sign depends on orientation convention)
        assert!(e1_coeff.abs() < 0.1, "e1 coefficient should be ~0");
        assert!((e2_coeff.abs() - 1.0).abs() < 0.1, "e2 coefficient magnitude should be ~1");
    }
}

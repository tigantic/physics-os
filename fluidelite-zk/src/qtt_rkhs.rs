//! QTT RKHS - Kernel Methods for Lookup Table Compression
//!
//! For ZK provers, lookup tables can be large. RKHS methods provide:
//! 1. Smooth function approximation via kernel interpolation
//! 2. Low-rank kernel matrix structure in QTT format
//! 3. Fast table lookups via QTT-compressed representations
//!
//! # Application to Prover
//!
//! The hybrid lookup circuit uses hash-based caching.
//! For cache-miss cases, we need fast table lookups.
//! QTT-RKHS compresses lookup tables via kernel interpolation.

use std::f64::consts::PI;

/// RBF (Gaussian) Kernel
///
/// k(x, y) = σ² exp(-||x - y||² / (2ℓ²))
#[derive(Clone, Debug)]
pub struct RbfKernel {
    /// Length scale ℓ
    pub length_scale: f64,
    /// Output variance σ²
    pub variance: f64,
}

impl RbfKernel {
    pub fn new(length_scale: f64, variance: f64) -> Self {
        Self { length_scale, variance }
    }
    
    /// Standard kernel with ℓ=1, σ²=1
    pub fn standard() -> Self {
        Self::new(1.0, 1.0)
    }
    
    /// Evaluate kernel between two scalars
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let dist_sq = (x - y).powi(2);
        self.variance * (-dist_sq / (2.0 * self.length_scale.powi(2))).exp()
    }
    
    /// Evaluate kernel between two vectors
    pub fn eval_vec(&self, x: &[f64], y: &[f64]) -> f64 {
        let dist_sq: f64 = x.iter().zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        self.variance * (-dist_sq / (2.0 * self.length_scale.powi(2))).exp()
    }
    
    /// Compute full kernel matrix K[i,j] = k(X[i], X[j])
    pub fn matrix(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let mut k = vec![0.0; n * n];
        
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] = self.eval(x[i], x[j]);
            }
        }
        
        k
    }
}

/// Polynomial kernel
///
/// k(x, y) = (γ⟨x, y⟩ + c)^d
#[derive(Clone, Debug)]
pub struct PolynomialKernel {
    pub degree: u32,
    pub gamma: f64,
    pub coef0: f64,
}

impl PolynomialKernel {
    pub fn new(degree: u32, gamma: f64, coef0: f64) -> Self {
        Self { degree, gamma, coef0 }
    }
    
    pub fn linear() -> Self {
        Self::new(1, 1.0, 0.0)
    }
    
    pub fn quadratic() -> Self {
        Self::new(2, 1.0, 0.0)
    }
    
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        (self.gamma * x * y + self.coef0).powi(self.degree as i32)
    }
}

/// QTT Kernel Matrix core
#[derive(Clone, Debug)]
pub struct QttKernelCore {
    /// Data: [r_left, 2, 2, r_right] flattened
    pub data: Vec<f64>,
    pub r_left: usize,
    pub r_right: usize,
}

impl QttKernelCore {
    pub fn zeros(r_left: usize, r_right: usize) -> Self {
        Self {
            data: vec![0.0; r_left * 4 * r_right],
            r_left,
            r_right,
        }
    }
    
    #[inline]
    pub fn get(&self, l: usize, i: usize, j: usize, r: usize) -> f64 {
        self.data[l * 4 * self.r_right + (i * 2 + j) * self.r_right + r]
    }
    
    #[inline]
    pub fn set(&mut self, l: usize, i: usize, j: usize, r: usize, val: f64) {
        self.data[l * 4 * self.r_right + (i * 2 + j) * self.r_right + r] = val;
    }
}

/// QTT-compressed kernel matrix
///
/// For N = 2^L points, stores K as TT with L cores.
#[derive(Clone, Debug)]
pub struct QttKernelMatrix {
    pub cores: Vec<QttKernelCore>,
    pub size: usize,
}

impl QttKernelMatrix {
    /// Create identity-like kernel matrix (rank 1)
    pub fn identity(size: usize) -> Self {
        let l = (size as f64).log2().ceil() as usize;
        assert_eq!(1 << l, size);
        
        let mut cores = Vec::with_capacity(l);
        for _ in 0..l {
            let mut core = QttKernelCore::zeros(1, 1);
            // Identity has 1 on diagonal
            core.set(0, 0, 0, 0, 1.0);
            core.set(0, 1, 1, 0, 1.0);
            cores.push(core);
        }
        
        Self { cores, size }
    }
    
    /// Get matrix element K[row, col]
    pub fn element(&self, row: usize, col: usize) -> f64 {
        let l = self.cores.len();
        
        let row_bits: Vec<usize> = (0..l).map(|k| (row >> k) & 1).collect();
        let col_bits: Vec<usize> = (0..l).map(|k| (col >> k) & 1).collect();
        
        // Contract along path
        let first_core = &self.cores[0];
        let mut result: Vec<f64> = (0..first_core.r_right)
            .map(|r| first_core.get(0, row_bits[0], col_bits[0], r))
            .collect();
        
        for k in 1..l {
            let core = &self.cores[k];
            let r_left = core.r_left;
            let r_right = core.r_right;
            
            let mut new_result = vec![0.0; r_right];
            for left in 0..r_left {
                for right in 0..r_right {
                    new_result[right] += result[left] * core.get(left, row_bits[k], col_bits[k], right);
                }
            }
            result = new_result;
        }
        
        result[0]
    }
    
    /// Matrix-vector product K @ v
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        let n = self.size;
        assert_eq!(v.len(), n);
        
        let mut result = vec![0.0; n];
        
        // For small matrices, use element-wise
        if n <= 1024 {
            for i in 0..n {
                for j in 0..n {
                    result[i] += self.element(i, j) * v[j];
                }
            }
        } else {
            // For large matrices, use QTT contraction (TODO)
            panic!("Large QTT matvec not yet implemented");
        }
        
        result
    }
}

/// Lookup table with kernel interpolation
///
/// Stores a function f: [0, N) → ℝ and provides fast lookups.
#[derive(Clone, Debug)]
pub struct KernelLookupTable {
    /// Table values f(0), f(1), ..., f(N-1)
    values: Vec<f64>,
    /// Kernel for interpolation
    kernel: RbfKernel,
    /// Precomputed kernel weights (for interpolation)
    weights: Option<Vec<f64>>,
}

impl KernelLookupTable {
    /// Create table from explicit values
    pub fn from_values(values: Vec<f64>, length_scale: f64) -> Self {
        Self {
            kernel: RbfKernel::new(length_scale, 1.0),
            values,
            weights: None,
        }
    }
    
    /// Create table from function
    pub fn from_function<F: Fn(f64) -> f64>(f: F, n: usize, length_scale: f64) -> Self {
        let values: Vec<f64> = (0..n).map(|i| f(i as f64)).collect();
        Self::from_values(values, length_scale)
    }
    
    /// Direct lookup (no interpolation)
    pub fn lookup(&self, idx: usize) -> f64 {
        self.values[idx]
    }
    
    /// Interpolated lookup (for non-integer indices)
    pub fn interpolate(&self, x: f64) -> f64 {
        let n = self.values.len();
        
        // Kernel interpolation: f(x) = Σ_i w_i k(x, i)
        // where w = K^{-1} y and y = [f(0), ..., f(N-1)]
        
        // For efficiency, use local interpolation
        let center = x.round() as usize;
        let window = 5;
        
        let lo = center.saturating_sub(window);
        let hi = (center + window + 1).min(n);
        
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        
        for i in lo..hi {
            let k = self.kernel.eval(x, i as f64);
            sum += k * self.values[i];
            weight_sum += k;
        }
        
        if weight_sum > 1e-10 {
            sum / weight_sum
        } else {
            self.values[center.min(n - 1)]
        }
    }
    
    /// Batch lookup
    pub fn lookup_batch(&self, indices: &[usize]) -> Vec<f64> {
        indices.iter().map(|&i| self.lookup(i)).collect()
    }
}

/// Kernel Ridge Regression for function approximation
///
/// Useful for compressing lookup tables.
pub struct KernelRidgeRegressor {
    kernel: RbfKernel,
    regularization: f64,
    /// Training points
    x_train: Vec<f64>,
    /// Learned weights (K + λI)^{-1} y
    weights: Vec<f64>,
}

impl KernelRidgeRegressor {
    pub fn new(kernel: RbfKernel, regularization: f64) -> Self {
        Self {
            kernel,
            regularization,
            x_train: Vec::new(),
            weights: Vec::new(),
        }
    }
    
    /// Fit to training data
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len();
        assert_eq!(y.len(), n);
        
        self.x_train = x.to_vec();
        
        // Build kernel matrix K + λI
        let mut k = self.kernel.matrix(x);
        for i in 0..n {
            k[i * n + i] += self.regularization;
        }
        
        // Solve (K + λI) w = y via Cholesky or direct
        // Simple implementation: Gaussian elimination
        self.weights = solve_linear_system(&k, y, n);
    }
    
    /// Predict at new points
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let n = self.x_train.len();
        let m = x.len();
        
        let mut predictions = vec![0.0; m];
        
        for i in 0..m {
            for j in 0..n {
                let k_val = self.kernel.eval(x[i], self.x_train[j]);
                predictions[i] += self.weights[j] * k_val;
            }
        }
        
        predictions
    }
}

/// Simple linear system solver (for small systems)
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // LU decomposition with partial pivoting
    let mut aug = vec![0.0; n * (n + 1)];
    
    // Build augmented matrix [A | b]
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    
    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_val = aug[k * (n + 1) + k].abs();
        let mut max_row = k;
        for i in k + 1..n {
            let val = aug[i * (n + 1) + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }
        
        // Swap rows
        if max_row != k {
            for j in 0..=n {
                let tmp = aug[k * (n + 1) + j];
                aug[k * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }
        
        // Eliminate
        let pivot = aug[k * (n + 1) + k];
        if pivot.abs() < 1e-14 {
            continue;
        }
        
        for i in k + 1..n {
            let factor = aug[i * (n + 1) + k] / pivot;
            for j in k..=n {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }
    
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in i + 1..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        if diag.abs() > 1e-14 {
            x[i] = sum / diag;
        }
    }
    
    x
}

/// Maximum Mean Discrepancy for distribution comparison
///
/// MMD(P, Q) = ||μ_P - μ_Q||_H where H is RKHS
pub fn mmd_squared(x: &[f64], y: &[f64], kernel: &RbfKernel) -> f64 {
    let n = x.len();
    let m = y.len();
    
    if n == 0 || m == 0 {
        return 0.0;
    }
    
    // E[k(X, X')]
    let mut xx_sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            xx_sum += kernel.eval(x[i], x[j]);
        }
    }
    let xx_term = xx_sum / (n * n) as f64;
    
    // E[k(Y, Y')]
    let mut yy_sum = 0.0;
    for i in 0..m {
        for j in 0..m {
            yy_sum += kernel.eval(y[i], y[j]);
        }
    }
    let yy_term = yy_sum / (m * m) as f64;
    
    // E[k(X, Y)]
    let mut xy_sum = 0.0;
    for i in 0..n {
        for j in 0..m {
            xy_sum += kernel.eval(x[i], y[j]);
        }
    }
    let xy_term = xy_sum / (n * m) as f64;
    
    xx_term + yy_term - 2.0 * xy_term
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rbf_kernel() {
        let k = RbfKernel::standard();
        
        // k(x, x) = 1
        assert!((k.eval(1.0, 1.0) - 1.0).abs() < 1e-10);
        
        // k(x, y) < 1 for x ≠ y
        assert!(k.eval(0.0, 1.0) < 1.0);
        
        // Symmetric
        assert!((k.eval(0.0, 1.0) - k.eval(1.0, 0.0)).abs() < 1e-10);
    }
    
    #[test]
    fn test_kernel_matrix() {
        let k = RbfKernel::new(1.0, 1.0);
        let x = vec![0.0, 1.0, 2.0];
        
        let mat = k.matrix(&x);
        
        // Diagonal should be 1
        assert!((mat[0] - 1.0).abs() < 1e-10);
        assert!((mat[4] - 1.0).abs() < 1e-10);
        assert!((mat[8] - 1.0).abs() < 1e-10);
        
        // Symmetric
        assert!((mat[1] - mat[3]).abs() < 1e-10);
    }
    
    #[test]
    fn test_lookup_table() {
        // Table for f(x) = sin(x)
        let table = KernelLookupTable::from_function(
            |x| (x * 0.1).sin(), 
            100, 
            2.0
        );
        
        // Direct lookup
        let val = table.lookup(10);
        assert!((val - (1.0_f64).sin()).abs() < 0.01);
        
        // Interpolation
        let interp = table.interpolate(10.5);
        let expected = (1.05_f64).sin();
        assert!((interp - expected).abs() < 0.1);
    }
    
    #[test]
    fn test_kernel_ridge_regression() {
        let kernel = RbfKernel::new(0.5, 1.0);
        let mut krr = KernelRidgeRegressor::new(kernel, 0.01);
        
        // Fit to sin function
        let x_train: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
        let y_train: Vec<f64> = x_train.iter().map(|&x| x.sin()).collect();
        
        krr.fit(&x_train, &y_train);
        
        // Predict
        let x_test = vec![0.5, 1.5, 2.5];
        let predictions = krr.predict(&x_test);
        
        for (i, &x) in x_test.iter().enumerate() {
            let expected = x.sin();
            let error = (predictions[i] - expected).abs();
            assert!(error < 0.2, "Prediction error too large: {} vs {}", predictions[i], expected);
        }
    }
    
    #[test]
    fn test_mmd() {
        let kernel = RbfKernel::new(1.0, 1.0);
        
        // Same distribution should have MMD ≈ 0
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y = x.clone();
        
        let mmd_same = mmd_squared(&x, &y, &kernel);
        assert!(mmd_same < 0.01, "MMD of same samples should be ~0: {}", mmd_same);
        
        // Different distributions should have MMD > 0
        let z: Vec<f64> = (0..50).map(|i| i as f64 * 0.1 + 5.0).collect();
        let mmd_diff = mmd_squared(&x, &z, &kernel);
        assert!(mmd_diff > 0.1, "MMD of different samples should be >0: {}", mmd_diff);
    }
    
    #[test]
    fn test_qtt_kernel_identity() {
        let id = QttKernelMatrix::identity(8);
        
        for i in 0..8 {
            for j in 0..8 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = id.element(i, j);
                assert!((actual - expected).abs() < 1e-10);
            }
        }
    }
}

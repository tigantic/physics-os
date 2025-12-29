//! Skeleton matrix operations for TCI
//!
//! TCI builds tensor decompositions by sampling "skeleton" matrices.
//! For a matrix A, the skeleton is A ≈ C @ A[I,J]^{-1} @ R where:
//!   - I, J are pivot row and column indices  
//!   - C = A[:, J] (columns at pivot positions)
//!   - R = A[I, :] (rows at pivot positions)
//!   - A[I,J] is the pivot submatrix (intersection)
//!
//! For QTT-TCI, we build skeletons at each qubit level.

use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use nalgebra::{DMatrix, SVD};
use pyo3::prelude::*;

use crate::maxvol::{maxvol, regularized_pinv, select_rows, MaxVolResult};
use crate::types::{MaxVolConfig, TCIError, TruncationPolicy};

/// Skeleton decomposition result
#[derive(Debug, Clone)]
pub struct SkeletonDecomp {
    /// Column submatrix C = A[:, J]
    pub c_matrix: Array2<f64>,
    /// Pivot submatrix inverse A[I,J]^{-1}
    pub pivot_inv: Array2<f64>,
    /// Row submatrix R = A[I, :]
    pub r_matrix: Array2<f64>,
    /// Row pivot indices
    pub row_pivots: Vec<usize>,
    /// Column pivot indices  
    pub col_pivots: Vec<usize>,
    /// Approximation error estimate
    pub error_estimate: f64,
}

impl SkeletonDecomp {
    /// Reconstruct the approximated matrix: C @ pivot_inv @ R
    pub fn reconstruct(&self) -> Array2<f64> {
        self.c_matrix.dot(&self.pivot_inv).dot(&self.r_matrix)
    }
    
    /// Get the effective rank
    pub fn rank(&self) -> usize {
        self.row_pivots.len()
    }
}

/// Build skeleton decomposition from sampled matrix entries
///
/// This is the core TCI operation. Given a "black-box" function f(i,j)
/// that returns matrix entries, we build a low-rank approximation
/// using only O(r * (m + n)) samples instead of O(m * n).
///
/// # Arguments
/// * `eval_fn` - Function to evaluate matrix entries
/// * `m` - Number of rows
/// * `n` - Number of columns  
/// * `initial_rank` - Starting rank guess
/// * `maxvol_config` - MaxVol algorithm configuration
/// * `truncation` - Rank truncation policy
///
/// # Returns
/// Skeleton decomposition or error
pub fn build_skeleton<F>(
    eval_fn: F,
    m: usize,
    n: usize,
    initial_rank: usize,
    maxvol_config: &MaxVolConfig,
    truncation: &TruncationPolicy,
) -> Result<SkeletonDecomp, TCIError>
where
    F: Fn(usize, usize) -> f64,
{
    let rank = initial_rank.min(m).min(n).min(truncation.hard_cap);
    
    // Initialize with first rank rows and columns
    let mut row_pivots: Vec<usize> = (0..rank).collect();
    let mut col_pivots: Vec<usize> = (0..rank).collect();
    
    // Alternating optimization: rows then columns
    for _iter in 0..maxvol_config.max_iterations {
        // Build column submatrix C = A[:, col_pivots]
        let c_matrix = build_column_submatrix(&eval_fn, m, &col_pivots);
        
        // Run MaxVol on C to select best rows
        let row_result = maxvol(&c_matrix, maxvol_config, Some(&row_pivots))?;
        row_pivots = row_result.pivots;
        
        // Build row submatrix R = A[row_pivots, :]
        let r_matrix = build_row_submatrix(&eval_fn, &row_pivots, n);
        
        // Run MaxVol on R^T to select best columns
        let r_t = r_matrix.t().to_owned();
        let col_result = maxvol(&r_t, maxvol_config, Some(&col_pivots))?;
        col_pivots = col_result.pivots;
        
        // Check convergence
        if row_result.converged && col_result.converged {
            break;
        }
    }
    
    // Build final submatrices
    let c_matrix = build_column_submatrix(&eval_fn, m, &col_pivots);
    let r_matrix = build_row_submatrix(&eval_fn, &row_pivots, n);
    
    // Build pivot submatrix and its inverse
    let pivot_matrix = build_pivot_submatrix(&eval_fn, &row_pivots, &col_pivots);
    let pivot_inv = regularized_pinv(&pivot_matrix, maxvol_config.regularization)?;
    
    // Estimate error (sample a few random entries)
    let error_estimate = estimate_error(&eval_fn, &c_matrix, &pivot_inv, &r_matrix, m, n);
    
    Ok(SkeletonDecomp {
        c_matrix,
        pivot_inv,
        r_matrix,
        row_pivots,
        col_pivots,
        error_estimate,
    })
}

/// Build column submatrix by evaluating at pivot columns
fn build_column_submatrix<F>(eval_fn: &F, m: usize, col_pivots: &[usize]) -> Array2<f64>
where
    F: Fn(usize, usize) -> f64,
{
    let r = col_pivots.len();
    let mut c = Array2::zeros((m, r));
    
    for i in 0..m {
        for (k, &j) in col_pivots.iter().enumerate() {
            c[[i, k]] = eval_fn(i, j);
        }
    }
    
    c
}

/// Build row submatrix by evaluating at pivot rows
fn build_row_submatrix<F>(eval_fn: &F, row_pivots: &[usize], n: usize) -> Array2<f64>
where
    F: Fn(usize, usize) -> f64,
{
    let r = row_pivots.len();
    let mut r_mat = Array2::zeros((r, n));
    
    for (k, &i) in row_pivots.iter().enumerate() {
        for j in 0..n {
            r_mat[[k, j]] = eval_fn(i, j);
        }
    }
    
    r_mat
}

/// Build pivot submatrix (intersection of pivot rows and columns)
fn build_pivot_submatrix<F>(
    eval_fn: &F,
    row_pivots: &[usize],
    col_pivots: &[usize],
) -> Array2<f64>
where
    F: Fn(usize, usize) -> f64,
{
    let r = row_pivots.len();
    let mut pivot = Array2::zeros((r, r));
    
    for (ki, &i) in row_pivots.iter().enumerate() {
        for (kj, &j) in col_pivots.iter().enumerate() {
            pivot[[ki, kj]] = eval_fn(i, j);
        }
    }
    
    pivot
}

/// Estimate approximation error by random sampling
fn estimate_error<F>(
    eval_fn: &F,
    c: &Array2<f64>,
    pivot_inv: &Array2<f64>,
    r: &Array2<f64>,
    m: usize,
    n: usize,
) -> f64
where
    F: Fn(usize, usize) -> f64,
{
    use rand::prelude::*;
    
    let mut rng = thread_rng();
    let num_samples = 100.min(m * n / 10).max(10);
    
    let mut max_error = 0.0f64;
    let reconstructed = c.dot(pivot_inv).dot(r);
    
    for _ in 0..num_samples {
        let i = rng.gen_range(0..m);
        let j = rng.gen_range(0..n);
        
        let true_val = eval_fn(i, j);
        let approx_val = reconstructed[[i, j]];
        
        let error = (true_val - approx_val).abs();
        max_error = max_error.max(error);
    }
    
    max_error
}

/// Truncate skeleton decomposition to lower rank via SVD
///
/// If the skeleton has grown too large, this truncates it back
/// to a target rank while preserving the best approximation.
pub fn truncate_skeleton(
    skeleton: &SkeletonDecomp,
    policy: &TruncationPolicy,
) -> Result<SkeletonDecomp, TCIError> {
    let current_rank = skeleton.rank();
    
    // Check if truncation needed
    if current_rank <= policy.target_rank {
        return Ok(skeleton.clone());
    }
    
    // Reconstruct and do truncated SVD
    let full = skeleton.reconstruct();
    let (m, n) = full.dim();
    
    // Convert to nalgebra for SVD
    let full_nalgebra = DMatrix::from_fn(m, n, |i, j| full[[i, j]]);
    let svd = SVD::new(full_nalgebra, true, true);
    
    let u = svd.u.ok_or_else(|| TCIError::LinAlgError {
        message: "SVD truncation: no U matrix".to_string(),
    })?;
    let vt = svd.v_t.ok_or_else(|| TCIError::LinAlgError {
        message: "SVD truncation: no Vt matrix".to_string(),
    })?;
    let s = &svd.singular_values;
    
    // Determine truncation rank
    let new_rank = policy.target_rank.min(s.len());
    
    // Build new skeleton: C = U * sqrt(S), pivot_inv = I, R = sqrt(S) * Vt
    // C = U[:, :new_rank] * diag(sqrt(s[:new_rank]))
    let mut c_new = Array2::zeros((m, new_rank));
    for j in 0..new_rank {
        let sqrt_s = s[j].sqrt();
        for i in 0..m {
            c_new[[i, j]] = u[(i, j)] * sqrt_s;
        }
    }
    
    // R = diag(sqrt(s[:new_rank])) * Vt[:new_rank, :]
    let mut r_new = Array2::zeros((new_rank, n));
    for i in 0..new_rank {
        let sqrt_s = s[i].sqrt();
        for j in 0..n {
            r_new[[i, j]] = sqrt_s * vt[(i, j)];
        }
    }
    
    // pivot_inv = identity (since we absorbed singular values into C and R)
    let pivot_inv = Array2::eye(new_rank);
    
    Ok(SkeletonDecomp {
        c_matrix: c_new,
        pivot_inv,
        r_matrix: r_new,
        row_pivots: (0..new_rank).collect(),
        col_pivots: (0..new_rank).collect(),
        error_estimate: skeleton.error_estimate,
    })
}

/// QTT core from skeleton decomposition
///
/// Converts a skeleton at qubit level k into a QTT core tensor G_k.
/// The core has shape (r_{k-1}, 2, r_k) in standard QTT format.
#[derive(Debug, Clone)]
pub struct QTTCore {
    /// Core tensor data, shape (r_left, 2, r_right)
    pub data: Array2<f64>,
    /// Left bond dimension
    pub r_left: usize,
    /// Right bond dimension
    pub r_right: usize,
}

impl QTTCore {
    /// Create new QTT core from skeleton
    ///
    /// The skeleton C @ pivot_inv @ R is reshaped into a 3-tensor.
    pub fn from_skeleton(skeleton: &SkeletonDecomp) -> Self {
        let r = skeleton.rank();
        
        // For single-qubit core: shape is (r, 2, r)
        // We interpret C as mapping from left bond to physical index
        // and R as mapping from physical index to right bond
        
        // Simplified: use C as the core data for now
        // Full implementation would properly combine C, pivot_inv, R
        QTTCore {
            data: skeleton.c_matrix.clone(),
            r_left: r,
            r_right: r,
        }
    }
    
    /// Contract with left vector
    pub fn contract_left(&self, left: &Array1<f64>, phys_idx: usize) -> Array1<f64> {
        // Simplified contraction
        let r_right = self.data.ncols();
        let mut result = Array1::zeros(r_right);
        
        // This is a placeholder - real implementation depends on
        // how we reshape the skeleton into 3-tensor form
        result
    }
}

/// Batch skeleton evaluation for GPU efficiency
///
/// Instead of evaluating one entry at a time, collect entries
/// needed and evaluate in batches.
pub struct BatchSkeletonBuilder {
    /// Row indices to evaluate
    row_indices: Vec<usize>,
    /// Column indices to evaluate
    col_indices: Vec<usize>,
    /// Number of rows in full matrix
    m: usize,
    /// Number of columns in full matrix
    n: usize,
}

impl BatchSkeletonBuilder {
    pub fn new(m: usize, n: usize) -> Self {
        BatchSkeletonBuilder {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            m,
            n,
        }
    }
    
    /// Add entries needed for column submatrix
    pub fn add_column_entries(&mut self, col_pivots: &[usize]) {
        for i in 0..self.m {
            for &j in col_pivots {
                self.row_indices.push(i);
                self.col_indices.push(j);
            }
        }
    }
    
    /// Add entries needed for row submatrix
    pub fn add_row_entries(&mut self, row_pivots: &[usize]) {
        for &i in row_pivots {
            for j in 0..self.n {
                self.row_indices.push(i);
                self.col_indices.push(j);
            }
        }
    }
    
    /// Add entries needed for pivot submatrix
    pub fn add_pivot_entries(&mut self, row_pivots: &[usize], col_pivots: &[usize]) {
        for &i in row_pivots {
            for &j in col_pivots {
                self.row_indices.push(i);
                self.col_indices.push(j);
            }
        }
    }
    
    /// Get all indices as flat arrays for batch evaluation
    pub fn get_indices(&self) -> (&[usize], &[usize]) {
        (&self.row_indices, &self.col_indices)
    }
    
    /// Get number of entries to evaluate
    pub fn num_entries(&self) -> usize {
        self.row_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_skeleton_low_rank() {
        // Create a rank-2 matrix via outer product
        let u = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let v = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        
        let eval_fn = |i: usize, j: usize| -> f64 {
            u[i] * v[j]
        };
        
        let config = MaxVolConfig::default();
        let truncation = TruncationPolicy::default();
        
        let skeleton = build_skeleton(eval_fn, 5, 3, 2, &config, &truncation).unwrap();
        
        // Rank should be at most 2 (matrix is rank 1)
        assert!(skeleton.rank() <= 2);
        
        // Reconstruction should be close to original
        let recon = skeleton.reconstruct();
        for i in 0..5 {
            for j in 0..3 {
                let expected = u[i] * v[j];
                assert!((recon[[i, j]] - expected).abs() < 1e-6,
                    "Entry ({},{}) mismatch: {} vs {}", i, j, recon[[i, j]], expected);
            }
        }
    }
    
    #[test]
    fn test_batch_builder() {
        let mut builder = BatchSkeletonBuilder::new(10, 8);
        
        let row_pivots = vec![0, 3, 7];
        let col_pivots = vec![1, 4];
        
        builder.add_column_entries(&col_pivots);
        assert_eq!(builder.num_entries(), 10 * 2);  // m * |col_pivots|
        
        builder.add_row_entries(&row_pivots);
        assert_eq!(builder.num_entries(), 10 * 2 + 3 * 8);  // + |row_pivots| * n
        
        builder.add_pivot_entries(&row_pivots, &col_pivots);
        assert_eq!(builder.num_entries(), 10 * 2 + 3 * 8 + 3 * 2);  // + |I| * |J|
    }
}

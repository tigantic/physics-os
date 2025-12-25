//! MaxVol algorithm for optimal pivot selection
//!
//! The MaxVol algorithm finds a subset of rows from a tall matrix A ∈ ℝ^{m×r}
//! that maximizes the volume (absolute determinant) of the selected submatrix.
//!
//! This is critical for TCI convergence — bad pivots lead to poor approximation
//! or non-convergence.
//!
//! Reference: Goreinov, Tyrtyshnikov - "The maximal-volume concept in approximation"

use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use nalgebra::{DMatrix, SVD};
use rustc_hash::FxHashSet;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::types::{MaxVolConfig, TCIError};

/// Result of MaxVol algorithm
#[derive(Debug, Clone)]
pub struct MaxVolResult {
    /// Selected pivot row indices
    pub pivots: Vec<usize>,
    /// Final maximum |C[i,j]| value (should be < 1 + tolerance)
    pub final_max_c: f64,
    /// Number of iterations taken
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Run the MaxVol algorithm to find optimal pivot rows
///
/// # Arguments
/// * `a` - Input matrix of shape (m, r) where m >= r
/// * `config` - MaxVol configuration parameters
///
/// # Returns
/// * `MaxVolResult` containing pivot indices and convergence info
///
/// # Algorithm
/// 1. Initialize with first r rows (or provided initial pivots)
/// 2. Compute B = A[pivots, :] and B_inv via SVD pseudo-inverse
/// 3. Compute C = A @ B_inv
/// 4. Find max |C[i,j]| for i not in pivots
/// 5. If max < 1 + tolerance, converged
/// 6. Otherwise, swap row i into pivot set at position j
/// 7. Repeat until converged or max_iterations
pub fn maxvol(
    a: &Array2<f64>,
    config: &MaxVolConfig,
    initial_pivots: Option<&[usize]>,
) -> Result<MaxVolResult, TCIError> {
    let (m, r) = a.dim();
    
    if m < r {
        return Err(TCIError::NotEnoughRows { m, r });
    }
    
    // Initialize pivots
    let mut pivots: Vec<usize> = match initial_pivots {
        Some(p) if p.len() == r => p.to_vec(),
        _ => (0..r).collect(),
    };
    
    let mut best_result = MaxVolResult {
        pivots: pivots.clone(),
        final_max_c: f64::INFINITY,
        iterations: 0,
        converged: false,
    };
    
    // Try with random restarts if needed
    for restart in 0..=config.random_restarts {
        if restart > 0 {
            // Random initialization for restart
            let mut all_indices: Vec<usize> = (0..m).collect();
            all_indices.shuffle(&mut thread_rng());
            pivots = all_indices[..r].to_vec();
        }
        
        let result = maxvol_inner(a, &mut pivots, config)?;
        
        if result.final_max_c < best_result.final_max_c {
            best_result = result;
        }
        
        if best_result.converged {
            break;
        }
    }
    
    Ok(best_result)
}

/// Inner MaxVol loop (single run without restarts)
fn maxvol_inner(
    a: &Array2<f64>,
    pivots: &mut Vec<usize>,
    config: &MaxVolConfig,
) -> Result<MaxVolResult, TCIError> {
    let (m, r) = a.dim();
    let mut iterations = 0;
    let mut final_max_c = f64::INFINITY;
    
    for iter in 0..config.max_iterations {
        iterations = iter + 1;
        
        // Extract pivot submatrix B = A[pivots, :]
        let b = select_rows(a, pivots);
        
        // Compute regularized pseudo-inverse
        let b_inv = regularized_pinv(&b, config.regularization)?;
        
        // Compute C = A @ B_inv
        let c = a.dot(&b_inv);
        
        // Find maximum |C[i,j]| outside current pivots
        let pivot_set: FxHashSet<usize> = pivots.iter().cloned().collect();
        let (max_i, max_j, max_val) = find_max_outside(&c, &pivot_set);
        
        final_max_c = max_val;
        
        // Check convergence
        if max_val < 1.0 + config.tolerance {
            return Ok(MaxVolResult {
                pivots: pivots.clone(),
                final_max_c,
                iterations,
                converged: true,
            });
        }
        
        // Swap: row max_i replaces pivot at position max_j
        pivots[max_j] = max_i;
    }
    
    Ok(MaxVolResult {
        pivots: pivots.clone(),
        final_max_c,
        iterations,
        converged: false,
    })
}

/// Select rows from matrix by indices
pub fn select_rows(a: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let r = indices.len();
    let n = a.ncols();
    let mut result = Array2::zeros((r, n));
    
    for (out_row, &in_row) in indices.iter().enumerate() {
        result.row_mut(out_row).assign(&a.row(in_row));
    }
    
    result
}

/// Compute regularized pseudo-inverse using SVD
///
/// This is more stable than direct inverse when matrix is nearly singular.
/// Uses truncated SVD: s_inv[i] = 1/s[i] if s[i] > epsilon, else 0
pub fn regularized_pinv(a: &Array2<f64>, epsilon: f64) -> Result<Array2<f64>, TCIError> {
    let (m, n) = a.dim();
    
    // Convert ndarray to nalgebra DMatrix
    let a_nalgebra = ndarray_to_nalgebra(a);
    
    // Compute SVD: A = U @ diag(s) @ Vt
    let svd = SVD::new(a_nalgebra, true, true);
    
    let u = svd.u.ok_or_else(|| TCIError::LinAlgError {
        message: "SVD did not return U matrix".to_string(),
    })?;
    
    let vt = svd.v_t.ok_or_else(|| TCIError::LinAlgError {
        message: "SVD did not return Vt matrix".to_string(),
    })?;
    
    let s = &svd.singular_values;
    
    // Compute regularized inverse of singular values
    let k = s.len();
    let mut s_inv = DMatrix::zeros(k, k);
    for i in 0..k {
        if s[i].abs() > epsilon {
            s_inv[(i, i)] = 1.0 / s[i];
        }
    }
    
    // Compute pseudo-inverse: A+ = V @ diag(s_inv) @ U^T
    // vt is V^T, so V = vt^T
    let v = vt.transpose();
    let result_nalgebra = v * s_inv * u.transpose();
    
    // Convert back to ndarray
    Ok(nalgebra_to_ndarray(&result_nalgebra))
}

/// Convert ndarray to nalgebra DMatrix
fn ndarray_to_nalgebra(a: &Array2<f64>) -> DMatrix<f64> {
    let (m, n) = a.dim();
    DMatrix::from_fn(m, n, |i, j| a[[i, j]])
}

/// Convert nalgebra DMatrix to ndarray
fn nalgebra_to_ndarray(a: &DMatrix<f64>) -> Array2<f64> {
    let (m, n) = a.shape();
    Array2::from_shape_fn((m, n), |(i, j)| a[(i, j)])
}

/// Find maximum absolute value in C outside the pivot set
fn find_max_outside(c: &Array2<f64>, pivot_set: &FxHashSet<usize>) -> (usize, usize, f64) {
    let (m, r) = c.dim();
    let mut max_val = 0.0f64;
    let mut max_i = 0usize;
    let mut max_j = 0usize;
    
    for i in 0..m {
        if pivot_set.contains(&i) {
            continue;
        }
        for j in 0..r {
            let val = c[[i, j]].abs();
            if val > max_val {
                max_val = val;
                max_i = i;
                max_j = j;
            }
        }
    }
    
    (max_i, max_j, max_val)
}

/// Compute the volume (absolute determinant) of a square matrix
pub fn matrix_volume(a: &Array2<f64>) -> Result<f64, TCIError> {
    let a_nalgebra = ndarray_to_nalgebra(a);
    let svd = SVD::new(a_nalgebra, false, false);
    
    // Volume = product of singular values
    let volume: f64 = svd.singular_values.iter().product();
    Ok(volume.abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::Rng;
    
    fn random_matrix(m: usize, n: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        Array2::from_shape_fn((m, n), |_| rng.gen_range(-1.0..1.0))
    }
    
    #[test]
    fn test_maxvol_identity() {
        // For identity matrix, any r rows should work
        let a = Array2::eye(5);
        let config = MaxVolConfig::default();
        
        let result = maxvol(&a, &config, None).unwrap();
        assert!(result.converged);
        assert_eq!(result.pivots.len(), 5);
    }
    
    #[test]
    fn test_maxvol_low_rank() {
        // Create a rank-3 matrix of size 10x3
        let u = random_matrix(10, 3);
        let v = random_matrix(3, 3);
        let a = u.dot(&v);
        
        let config = MaxVolConfig::default();
        let result = maxvol(&a, &config, None).unwrap();
        
        assert!(result.converged);
        assert_eq!(result.pivots.len(), 3);
        
        // Selected submatrix should have non-zero volume
        let submatrix = select_rows(&a, &result.pivots);
        let vol = matrix_volume(&submatrix).unwrap();
        assert!(vol > 1e-10);
    }
    
    #[test]
    fn test_regularized_pinv() {
        let a = Array2::eye(3);
        let config = MaxVolConfig::default();
        
        let a_inv = regularized_pinv(&a, config.regularization).unwrap();
        
        // For identity, pseudo-inverse should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (a_inv[[i, j]] - expected).abs();
                assert!(diff < 1e-10, "Expected {}, got {} at ({},{})", expected, a_inv[[i,j]], i, j);
            }
        }
    }
    
    #[test]
    fn test_maxvol_handles_nearly_singular() {
        // Create a nearly singular matrix
        let mut a = Array2::eye(5);
        a[[4, 4]] = 1e-14;  // Nearly zero
        
        let config = MaxVolConfig::default();
        let result = maxvol(&a, &config, None);
        
        // Should not panic, should handle gracefully
        assert!(result.is_ok());
    }
}

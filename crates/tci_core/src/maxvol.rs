//! MaxVol Algorithm: Maximum Volume Submatrix Selection
//!
//! MaxVol finds the r×r submatrix with maximum determinant (volume) from an m×r matrix.
//! This is critical for pivot selection in TCI to ensure numerical stability.
//!
//! Algorithm:
//! 1. Start with initial row selection
//! 2. Compute coefficient matrix C = A[rows]^{-1} @ A
//! 3. Find (i,j) with max |C[i,j]| where i not in rows
//! 4. If |C[i,j]| > 1 + tol, swap row j with row i
//! 5. Repeat until no improvement
//!
//! Complexity: O(r² × iterations) per call

use ndarray::{Array1, Array2, s};

/// MaxVol result
pub struct MaxVolResult {
    /// Selected row indices
    pub indices: Vec<usize>,
    /// Final coefficient matrix (for rank estimation)
    pub coefficients: Array2<f64>,
    /// Number of iterations
    pub iterations: usize,
}

/// Run MaxVol algorithm to find maximum volume r×r submatrix
/// 
/// # Arguments
/// * `matrix` - m×r matrix (m >= r)
/// * `initial_rows` - Initial row selection (length r)
/// * `max_iter` - Maximum iterations
/// * `tolerance` - Convergence tolerance (stops if max coeff < 1 + tol)
/// 
/// # Returns
/// * Indices of selected rows
pub fn maxvol(
    matrix: &Array2<f64>,
    initial_rows: Option<&[usize]>,
    max_iter: usize,
    tolerance: f64,
) -> MaxVolResult {
    let (m, r) = matrix.dim();
    
    if m < r {
        panic!("Matrix must have at least as many rows as columns");
    }
    
    // Initialize row selection
    let mut rows: Vec<usize> = if let Some(init) = initial_rows {
        assert_eq!(init.len(), r, "Initial rows must have length r");
        init.to_vec()
    } else {
        // Use first r rows as initial guess
        (0..r).collect()
    };
    
    // Compute A[rows] and its inverse
    let mut a_rows = extract_rows(matrix, &rows);
    let mut a_rows_inv = invert_matrix(&a_rows);
    
    // Coefficient matrix C = A @ A[rows]^{-1}
    let mut coeff = matrix.dot(&a_rows_inv);
    
    let mut iterations = 0;
    
    for iter in 0..max_iter {
        iterations = iter + 1;
        
        // Find max |C[i,j]| where i not in rows
        let (best_i, best_j, best_val) = find_max_coeff(&coeff, &rows);
        
        if best_val <= 1.0 + tolerance {
            // Converged
            break;
        }
        
        // Swap: replace row best_j with row best_i
        let old_row = rows[best_j];
        rows[best_j] = best_i;
        
        // Update A[rows] efficiently (rank-1 update)
        // For simplicity, recompute (production would use Sherman-Morrison)
        a_rows = extract_rows(matrix, &rows);
        a_rows_inv = invert_matrix(&a_rows);
        coeff = matrix.dot(&a_rows_inv);
    }
    
    MaxVolResult {
        indices: rows,
        coefficients: coeff,
        iterations,
    }
}

/// Extract rows from matrix
fn extract_rows(matrix: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let r = rows.len();
    let n = matrix.ncols();
    let mut result = Array2::zeros((r, n));
    
    for (i, &row) in rows.iter().enumerate() {
        result.row_mut(i).assign(&matrix.row(row));
    }
    
    result
}

/// Invert a square matrix (Gaussian elimination)
fn invert_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "Matrix must be square");
    
    // Augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    aug.slice_mut(s![.., ..n]).assign(matrix);
    for i in 0..n {
        aug[[i, n + i]] = 1.0;
    }
    
    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_row = row;
                max_val = aug[[row, col]].abs();
            }
        }
        
        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        
        // Normalize pivot row
        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-14 {
            // Nearly singular, return identity as fallback
            return Array2::eye(n);
        }
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }
        
        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }
    
    // Extract inverse
    aug.slice(s![.., n..]).to_owned()
}

/// Find maximum coefficient |C[i,j]| where i not in rows
fn find_max_coeff(coeff: &Array2<f64>, rows: &[usize]) -> (usize, usize, f64) {
    let rows_set: std::collections::HashSet<_> = rows.iter().copied().collect();
    
    let mut best_i = 0;
    let mut best_j = 0;
    let mut best_val = 0.0f64;
    
    for i in 0..coeff.nrows() {
        if rows_set.contains(&i) {
            continue;
        }
        for j in 0..coeff.ncols() {
            let val = coeff[[i, j]].abs();
            if val > best_val {
                best_i = i;
                best_j = j;
                best_val = val;
            }
        }
    }
    
    (best_i, best_j, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_maxvol_identity() {
        // For identity matrix extended, should select first 3 rows
        let matrix = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]);
        
        let result = maxvol(&matrix, None, 100, 0.01);
        
        // Should converge to rows 0, 1, 2
        assert!(result.indices.contains(&0));
        assert!(result.indices.contains(&1));
        assert!(result.indices.contains(&2));
    }
}

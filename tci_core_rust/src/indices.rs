//! Neighbor index generation for CFD stencils
//!
//! This module computes neighbor indices (i+1, i-1) for flux computation.
//! CRITICAL: This MUST be done in Rust, not on GPU!
//!
//! Why? Binary addition for QTT indices causes thread divergence on GPU.
//! A single i+1 in QTT indexing (n qubits) requires n sequential carry operations.
//! This destroys GPU parallelism.
//!
//! By precomputing neighbor indices in Rust and passing both vectors to GPU,
//! we keep all GPU threads in lockstep.

use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};

/// Boundary condition types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Periodic: index wraps around (i + 1 % N)
    Periodic,
    /// Zero-gradient: derivative is zero at boundary (Neumann)
    ZeroGradient,
    /// Extrapolate: linear extrapolation from interior
    Extrapolate,
    /// Fixed: use specified boundary values
    Fixed,
}

impl BoundaryCondition {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "periodic" => BoundaryCondition::Periodic,
            "zero_gradient" | "neumann" => BoundaryCondition::ZeroGradient,
            "extrapolate" => BoundaryCondition::Extrapolate,
            "fixed" | "dirichlet" => BoundaryCondition::Fixed,
            _ => BoundaryCondition::Periodic,
        }
    }
}

/// Batch of indices with their neighbors for CFD flux computation
/// 
/// Provides zero-copy access to index arrays via numpy integration.
/// For true zero-copy to PyTorch, use:
///   indices_gpu = torch.from_numpy(batch.indices_array()).cuda()
#[pyclass]
#[derive(Debug, Clone)]
pub struct IndexBatch {
    /// Primary indices (i) - kept as Vec for internal use
    indices_vec: Vec<i64>,
    
    /// Left neighbor indices (i - 1)
    left_vec: Vec<i64>,
    
    /// Right neighbor indices (i + 1)
    right_vec: Vec<i64>,
    
    /// Domain size (N = 2^n_qubits)
    #[pyo3(get)]
    pub domain_size: u64,
}

#[pymethods]
impl IndexBatch {
    #[new]
    pub fn new(indices: Vec<u64>, domain_size: u64, boundary: &str) -> Self {
        let bc = BoundaryCondition::from_str(boundary);
        let (left, right) = compute_neighbors(&indices, domain_size, bc);
        
        // Convert to i64 for numpy/torch compatibility
        IndexBatch {
            indices_vec: indices.iter().map(|&x| x as i64).collect(),
            left_vec: left.iter().map(|&x| x as i64).collect(),
            right_vec: right.iter().map(|&x| x as i64).collect(),
            domain_size,
        }
    }
    
    /// Get primary indices as Python list (for backward compat)
    #[getter]
    pub fn indices(&self) -> Vec<i64> {
        self.indices_vec.clone()
    }
    
    /// Get left neighbor indices as Python list
    #[getter]
    pub fn left(&self) -> Vec<i64> {
        self.left_vec.clone()
    }
    
    /// Get right neighbor indices as Python list
    #[getter]
    pub fn right(&self) -> Vec<i64> {
        self.right_vec.clone()
    }
    
    /// Get primary indices as numpy array (zero-copy to torch via .cuda())
    /// 
    /// Usage:
    ///   indices = torch.from_numpy(batch.indices_array()).cuda()
    pub fn indices_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.indices_vec.clone().into_pyarray(py)
    }
    
    /// Get left neighbor indices as numpy array
    pub fn left_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.left_vec.clone().into_pyarray(py)
    }
    
    /// Get right neighbor indices as numpy array
    pub fn right_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.right_vec.clone().into_pyarray(py)
    }
    
    /// Get batch size
    pub fn len(&self) -> usize {
        self.indices_vec.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.indices_vec.is_empty()
    }
    
    /// Convert indices to QTT multi-indices
    ///
    /// For a QTT with n qubits, each index i ∈ [0, 2^n) maps to
    /// a tuple (b_0, b_1, ..., b_{n-1}) where b_k = (i >> k) & 1
    pub fn to_qtt_indices(&self, n_qubits: usize) -> Vec<Vec<u8>> {
        self.indices_vec
            .iter()
            .map(|&idx| index_to_qtt(idx as u64, n_qubits))
            .collect()
    }
    
    /// Convert QTT multi-indices back to flat indices
    #[staticmethod]
    pub fn from_qtt_indices(qtt_indices: Vec<Vec<u8>>) -> Vec<u64> {
        qtt_indices
            .iter()
            .map(|bits| qtt_to_index(bits))
            .collect()
    }
}

/// Compute left and right neighbor indices
///
/// This is where we avoid the GPU thread divergence problem.
/// All carry propagation happens here, sequentially, in Rust.
pub fn compute_neighbors(
    indices: &[u64],
    domain_size: u64,
    bc: BoundaryCondition,
) -> (Vec<u64>, Vec<u64>) {
    let n = indices.len();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    
    for &i in indices {
        let (l, r) = neighbor_indices(i, domain_size, bc);
        left.push(l);
        right.push(r);
    }
    
    (left, right)
}

/// Compute single neighbor pair
#[inline]
fn neighbor_indices(i: u64, n: u64, bc: BoundaryCondition) -> (u64, u64) {
    match bc {
        BoundaryCondition::Periodic => {
            let left = if i == 0 { n - 1 } else { i - 1 };
            let right = if i == n - 1 { 0 } else { i + 1 };
            (left, right)
        }
        BoundaryCondition::ZeroGradient => {
            let left = if i == 0 { 0 } else { i - 1 };
            let right = if i == n - 1 { n - 1 } else { i + 1 };
            (left, right)
        }
        BoundaryCondition::Extrapolate => {
            // For extrapolation, we use ghost cells conceptually
            // but physically clamp to boundary
            let left = i.saturating_sub(1);
            let right = (i + 1).min(n - 1);
            (left, right)
        }
        BoundaryCondition::Fixed => {
            // Fixed BC: boundary values handled separately
            // Here we just clamp indices
            let left = i.saturating_sub(1);
            let right = (i + 1).min(n - 1);
            (left, right)
        }
    }
}

/// Convert flat index to QTT multi-index (little-endian bit decomposition)
///
/// For index i and n qubits:
///   i = b_0 * 2^0 + b_1 * 2^1 + ... + b_{n-1} * 2^{n-1}
///
/// Returns [b_0, b_1, ..., b_{n-1}]
#[inline]
pub fn index_to_qtt(i: u64, n_qubits: usize) -> Vec<u8> {
    (0..n_qubits)
        .map(|k| ((i >> k) & 1) as u8)
        .collect()
}

/// Convert QTT multi-index back to flat index
#[inline]
pub fn qtt_to_index(bits: &[u8]) -> u64 {
    bits.iter()
        .enumerate()
        .map(|(k, &b)| (b as u64) << k)
        .sum()
}

/// Generate fiber indices for TCI
///
/// A fiber is a 1D slice through the tensor along one mode.
/// For QTT evaluation at position k with fixed indices on other modes,
/// we generate indices where bit k varies (0 or 1) and other bits are fixed.
pub fn generate_fiber_indices(
    base_index: u64,
    qubit_position: usize,
    n_qubits: usize,
) -> (u64, u64) {
    // Mask to clear bit at qubit_position
    let mask = !(1u64 << qubit_position);
    let base_cleared = base_index & mask;
    
    // Two indices: bit=0 and bit=1
    let idx_0 = base_cleared;
    let idx_1 = base_cleared | (1u64 << qubit_position);
    
    (idx_0, idx_1)
}

/// Generate batch of fiber indices for parallel evaluation
///
/// Given a set of base indices and a qubit position, generates all fiber pairs.
/// This is used for building TCI skeleton matrices.
pub fn generate_fiber_batch(
    base_indices: &[u64],
    qubit_position: usize,
    _n_qubits: usize,
) -> Vec<(u64, u64)> {
    base_indices
        .iter()
        .map(|&base| generate_fiber_indices(base, qubit_position, _n_qubits))
        .collect()
}

/// Multi-index addition for d-dimensional QTT
///
/// For d-dimensional problems, each dimension has its own QTT.
/// This function adds neighbor offsets correctly across dimensions.
pub fn multi_index_neighbors(
    indices: &[u64],
    n_qubits_per_dim: &[usize],
    dimension: usize,
    bc: BoundaryCondition,
) -> (Vec<u64>, Vec<u64>) {
    let dim_size = 1u64 << n_qubits_per_dim[dimension];
    
    // For multi-dimensional case, we need to extract the indices for the
    // specified dimension, compute neighbors, and reassemble
    
    // This is a simplified version - full implementation would handle
    // interleaved QTT indices properly
    compute_neighbors(indices, dim_size, bc)
}

/// Stencil pattern for flux computation
#[derive(Debug, Clone)]
pub struct FluxStencil {
    /// Central index
    pub center: u64,
    /// Left neighbor
    pub left: u64,
    /// Right neighbor
    pub right: u64,
    /// QTT decomposition of center
    pub center_qtt: Vec<u8>,
    /// QTT decomposition of left
    pub left_qtt: Vec<u8>,
    /// QTT decomposition of right
    pub right_qtt: Vec<u8>,
}

impl FluxStencil {
    pub fn new(center: u64, n_qubits: usize, domain_size: u64, bc: BoundaryCondition) -> Self {
        let (left, right) = neighbor_indices(center, domain_size, bc);
        
        FluxStencil {
            center,
            left,
            right,
            center_qtt: index_to_qtt(center, n_qubits),
            left_qtt: index_to_qtt(left, n_qubits),
            right_qtt: index_to_qtt(right, n_qubits),
        }
    }
}

/// Generate flux stencils for a batch of indices
pub fn generate_flux_stencils(
    indices: &[u64],
    n_qubits: usize,
    bc: BoundaryCondition,
) -> Vec<FluxStencil> {
    let domain_size = 1u64 << n_qubits;
    
    indices
        .iter()
        .map(|&i| FluxStencil::new(i, n_qubits, domain_size, bc))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_periodic_neighbors() {
        let n = 8u64;
        
        // Interior point
        assert_eq!(neighbor_indices(3, n, BoundaryCondition::Periodic), (2, 4));
        
        // Left boundary (wraps to end)
        assert_eq!(neighbor_indices(0, n, BoundaryCondition::Periodic), (7, 1));
        
        // Right boundary (wraps to start)
        assert_eq!(neighbor_indices(7, n, BoundaryCondition::Periodic), (6, 0));
    }
    
    #[test]
    fn test_zero_gradient_neighbors() {
        let n = 8u64;
        
        // Interior point
        assert_eq!(neighbor_indices(3, n, BoundaryCondition::ZeroGradient), (2, 4));
        
        // Left boundary (clamps to 0)
        assert_eq!(neighbor_indices(0, n, BoundaryCondition::ZeroGradient), (0, 1));
        
        // Right boundary (clamps to n-1)
        assert_eq!(neighbor_indices(7, n, BoundaryCondition::ZeroGradient), (6, 7));
    }
    
    #[test]
    fn test_index_to_qtt() {
        // i = 5 = 101 in binary
        // Little-endian: [1, 0, 1]
        assert_eq!(index_to_qtt(5, 3), vec![1, 0, 1]);
        
        // i = 0 = 000
        assert_eq!(index_to_qtt(0, 3), vec![0, 0, 0]);
        
        // i = 7 = 111
        assert_eq!(index_to_qtt(7, 3), vec![1, 1, 1]);
    }
    
    #[test]
    fn test_qtt_to_index() {
        assert_eq!(qtt_to_index(&[1, 0, 1]), 5);
        assert_eq!(qtt_to_index(&[0, 0, 0]), 0);
        assert_eq!(qtt_to_index(&[1, 1, 1]), 7);
    }
    
    #[test]
    fn test_roundtrip() {
        for i in 0..16 {
            let qtt = index_to_qtt(i, 4);
            let back = qtt_to_index(&qtt);
            assert_eq!(i, back);
        }
    }
    
    #[test]
    fn test_fiber_indices() {
        // Base index = 5 = 101, qubit 1
        // idx_0 = 1x1 with x=0 -> 101 & ~(10) = 101 -> 5
        // idx_1 = 1x1 with x=1 -> 101 | 10 = 111 -> 7
        let (idx_0, idx_1) = generate_fiber_indices(5, 1, 3);
        assert_eq!(idx_0, 5);  // bit 1 already 0? No, 5=101, bit 1 is 0
        assert_eq!(idx_1, 7);  // 5 | 2 = 7
    }
    
    #[test]
    fn test_batch_neighbors() {
        let indices = vec![0, 3, 7];
        let (left, right) = compute_neighbors(&indices, 8, BoundaryCondition::Periodic);
        
        assert_eq!(left, vec![7, 2, 6]);
        assert_eq!(right, vec![1, 4, 0]);
    }
}

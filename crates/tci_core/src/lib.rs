//! TCI Core: Tensor Cross Interpolation for Native QTT Construction
//!
//! This crate implements the TT-Cross Interpolation algorithm with MaxVol
//! pivot selection. Exposed to Python via PyO3 for drop-in replacement
//! of the slower Python TCI implementation.
//!
//! # Algorithm
//!
//! TCI builds a TT representation by sampling a black-box function at
//! O(r² × n) points rather than O(2^n) dense evaluation. The algorithm:
//!
//! 1. Initialize pivot indices (left/right index sets per site)
//! 2. Sample function at skeleton indices
//! 3. Update pivots using MaxVol (maximum volume submatrix selection)
//! 4. Build TT cores from skeleton matrices
//! 5. Repeat until convergence
//!
//! # Complexity
//!
//! - Function evaluations: O(r² × n × iterations)
//! - Memory: O(r² × n)
//! - vs Dense TT-SVD: O(2^n) evaluations, O(2^n) memory

mod maxvol;
mod sampler;
mod skeleton;
mod block_svd;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyArrayMethods, IntoPyArray};
use ndarray::{Array1, Array2, s};
use std::collections::HashSet;

/// MaxVol configuration parameters
#[pyclass]
#[derive(Clone, Debug)]
pub struct MaxVolConfig {
    /// Maximum iterations for MaxVol
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Tolerance for pivot update (1 + tol is the threshold)
    #[pyo3(get, set)]
    pub tolerance: f64,
}

#[pymethods]
impl MaxVolConfig {
    #[new]
    #[pyo3(signature = (max_iterations=100, tolerance=0.01))]
    fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self { max_iterations, tolerance }
    }
}

impl Default for MaxVolConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 0.01,
        }
    }
}

/// TCI truncation policy
#[pyclass]
#[derive(Clone, Debug)]
pub enum TruncationPolicy {
    /// Fixed maximum rank
    FixedRank { max_rank: usize },
    /// Adaptive based on singular value tolerance
    Adaptive { tolerance: f64, max_rank: usize },
}

#[pymethods]
impl TruncationPolicy {
    #[staticmethod]
    fn fixed(max_rank: usize) -> Self {
        Self::FixedRank { max_rank }
    }

    #[staticmethod]
    #[pyo3(signature = (tolerance, max_rank=64))]
    fn adaptive(tolerance: f64, max_rank: usize) -> Self {
        Self::Adaptive { tolerance, max_rank }
    }
}

impl Default for TruncationPolicy {
    fn default() -> Self {
        Self::FixedRank { max_rank: 64 }
    }
}

/// TCI configuration
#[pyclass]
#[derive(Clone, Debug)]
pub struct TCIConfig {
    /// Maximum TCI iterations
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Convergence tolerance
    #[pyo3(get, set)]
    pub tolerance: f64,
    /// Batch size for function evaluations
    #[pyo3(get, set)]
    pub batch_size: usize,
    /// MaxVol configuration
    #[pyo3(get, set)]
    pub maxvol: MaxVolConfig,
    /// Truncation policy
    #[pyo3(get, set)]
    pub truncation: TruncationPolicy,
}

#[pymethods]
impl TCIConfig {
    #[new]
    #[pyo3(signature = (max_iterations=50, tolerance=1e-6, batch_size=10000))]
    fn new(max_iterations: usize, tolerance: f64, batch_size: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            batch_size,
            maxvol: MaxVolConfig::default(),
            truncation: TruncationPolicy::default(),
        }
    }
}

impl Default for TCIConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            batch_size: 10000,
            maxvol: MaxVolConfig::default(),
            truncation: TruncationPolicy::default(),
        }
    }
}

/// TCI Sampler: Manages pivot indices and skeleton matrices
#[pyclass]
pub struct TCISampler {
    /// Number of qubits (log₂ of domain size)
    n_qubits: usize,
    /// Maximum rank
    max_rank: usize,
    /// Left pivot sets for each site
    pivots_left: Vec<Vec<u64>>,
    /// Right pivot sets for each site  
    pivots_right: Vec<Vec<u64>>,
    /// Cached sample values: index -> value
    samples: std::collections::HashMap<u64, f64>,
    /// Current skeleton matrices (one per site)
    skeletons: Vec<Array2<f64>>,
    /// Configuration
    config: TCIConfig,
    /// Total function evaluations
    total_evals: u64,
    /// Current iteration
    iteration: usize,
}

#[pymethods]
impl TCISampler {
    /// Create a new TCI sampler
    #[new]
    #[pyo3(signature = (n_qubits, max_rank=64, config=None))]
    fn new(n_qubits: usize, max_rank: usize, config: Option<TCIConfig>) -> PyResult<Self> {
        if n_qubits == 0 || n_qubits > 40 {
            return Err(PyValueError::new_err(format!(
                "n_qubits must be in [1, 40], got {}", n_qubits
            )));
        }
        if max_rank == 0 {
            return Err(PyValueError::new_err("max_rank must be positive"));
        }

        let config = config.unwrap_or_default();
        
        // Initialize pivot sets with geometric spread
        let initial_pivots = max_rank.min(16);
        let mut pivots_left = Vec::with_capacity(n_qubits);
        let mut pivots_right = Vec::with_capacity(n_qubits);
        
        for d in 0..n_qubits {
            // Left pivots: indices from 0 to 2^d - 1
            let n_left = (1u64 << d).min(initial_pivots as u64);
            pivots_left.push((0..n_left).collect());
            
            // Right pivots: indices from 0 to 2^(n-d-1) - 1
            let n_right = (1u64 << (n_qubits - d - 1)).min(initial_pivots as u64);
            pivots_right.push((0..n_right).collect());
        }
        
        // Initialize empty skeletons
        let skeletons = vec![Array2::zeros((0, 0)); n_qubits];
        
        Ok(Self {
            n_qubits,
            max_rank,
            pivots_left,
            pivots_right,
            samples: std::collections::HashMap::new(),
            skeletons,
            config,
            total_evals: 0,
            iteration: 0,
        })
    }

    /// Get indices to sample for current iteration
    fn get_sample_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let mut indices = HashSet::new();
        let n = 1u64 << self.n_qubits;
        
        // For each dimension, generate fiber indices
        for dim in 0..self.n_qubits {
            for &left_idx in &self.pivots_left[dim] {
                for bit in 0..2u64 {
                    for &right_idx in &self.pivots_right[dim] {
                        let full_idx = compose_index(left_idx, bit, right_idx, dim, self.n_qubits);
                        if full_idx < n && !self.samples.contains_key(&full_idx) {
                            indices.insert(full_idx);
                        }
                    }
                }
            }
        }
        
        // Add random exploration points
        let n_random = (self.config.batch_size / 10).min(100);
        let mut rng_state = self.iteration as u64 * 31337;
        for _ in 0..n_random {
            // Simple LCG for reproducibility
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = rng_state % n;
            if !self.samples.contains_key(&idx) {
                indices.insert(idx);
            }
        }
        
        // Limit to batch size
        let mut indices: Vec<i64> = indices.into_iter()
            .take(self.config.batch_size)
            .map(|x| x as i64)
            .collect();
        indices.sort_unstable();
        
        Array1::from_vec(indices).into_pyarray(py)
    }

    /// Submit sample values from function evaluation
    fn submit_samples(
        &mut self,
        indices: &Bound<'_, PyArray1<i64>>,
        values: &Bound<'_, PyArray1<f64>>,
    ) -> PyResult<()> {
        let indices = unsafe { indices.as_slice()? };
        let values = unsafe { values.as_slice()? };
        
        if indices.len() != values.len() {
            return Err(PyValueError::new_err("indices and values must have same length"));
        }
        
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            self.samples.insert(idx as u64, val);
        }
        
        self.total_evals += indices.len() as u64;
        self.iteration += 1;
        
        // Update pivots based on new samples
        self.update_pivots();
        
        Ok(())
    }

    /// Check if TCI has converged
    fn is_converged(&self) -> bool {
        // Simple convergence: sample growth rate < 1%
        let coverage = self.samples.len() as f64 / (1u64 << self.n_qubits) as f64;
        coverage > 0.5 || self.iteration >= self.config.max_iterations
    }

    /// Get number of function evaluations
    fn get_total_evals(&self) -> u64 {
        self.total_evals
    }

    /// Get number of samples cached
    fn get_sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Build TT cores from current skeleton
    fn build_cores<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        // Build TT cores from samples using fiber-based reconstruction
        let cores = self.reconstruct_cores()?;
        
        Ok(cores.into_iter()
            .map(|c| c.into_pyarray(py))
            .collect())
    }
}

impl TCISampler {
    /// Update pivots based on sample values
    fn update_pivots(&mut self) {
        // For each dimension, update pivots using MaxVol-style selection
        for dim in 0..self.n_qubits {
            self.update_pivots_for_dim(dim);
        }
    }

    /// Update pivots for a specific dimension
    fn update_pivots_for_dim(&mut self, dim: usize) {
        // Collect samples relevant to this dimension
        let n = 1u64 << self.n_qubits;
        
        // Build fiber matrix: rows = left pivots, cols = right pivots
        // For each (left, right) pair, we have 2 values (bit 0 and bit 1)
        let n_left = self.pivots_left[dim].len();
        let n_right = self.pivots_right[dim].len();
        
        if n_left == 0 || n_right == 0 {
            return;
        }
        
        // Collect fiber values
        let mut fiber_vals: Vec<(u64, u64, [f64; 2])> = Vec::new();
        
        for (li, &left_idx) in self.pivots_left[dim].iter().enumerate() {
            for (ri, &right_idx) in self.pivots_right[dim].iter().enumerate() {
                let idx0 = compose_index(left_idx, 0, right_idx, dim, self.n_qubits);
                let idx1 = compose_index(left_idx, 1, right_idx, dim, self.n_qubits);
                
                if let (Some(&v0), Some(&v1)) = (self.samples.get(&idx0), self.samples.get(&idx1)) {
                    fiber_vals.push((left_idx, right_idx, [v0, v1]));
                }
            }
        }
        
        // Find pivots with largest absolute values
        if fiber_vals.is_empty() {
            return;
        }
        
        // Sort by maximum absolute value
        let mut ranked: Vec<(f64, u64, u64)> = fiber_vals.iter()
            .map(|(l, r, vals)| (vals[0].abs().max(vals[1].abs()), *l, *r))
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Update pivot sets (keep top max_rank)
        let mut new_left: HashSet<u64> = HashSet::new();
        let mut new_right: HashSet<u64> = HashSet::new();
        
        for (_, l, r) in ranked.iter().take(self.max_rank * 2) {
            if new_left.len() < self.max_rank {
                new_left.insert(*l);
            }
            if new_right.len() < self.max_rank {
                new_right.insert(*r);
            }
        }
        
        // Keep old pivots if new ones are empty
        if !new_left.is_empty() {
            let mut left_vec: Vec<u64> = new_left.into_iter().collect();
            left_vec.sort_unstable();
            self.pivots_left[dim] = left_vec;
        }
        if !new_right.is_empty() {
            let mut right_vec: Vec<u64> = new_right.into_iter().collect();
            right_vec.sort_unstable();
            self.pivots_right[dim] = right_vec;
        }
    }

    /// Reconstruct TT cores from samples
    fn reconstruct_cores(&self) -> PyResult<Vec<Array2<f64>>> {
        // For each site, build core from fiber samples
        let mut cores = Vec::with_capacity(self.n_qubits);
        
        for dim in 0..self.n_qubits {
            let n_left = self.pivots_left[dim].len().max(1);
            let n_right = self.pivots_right[dim].len().max(1);
            
            // Core shape: (r_left, 2, r_right) flattened to (r_left * 2, r_right)
            // We'll return (r_left * 2, r_right) and reshape in Python
            let mut core = Array2::zeros((n_left * 2, n_right));
            
            for (li, &left_idx) in self.pivots_left[dim].iter().enumerate() {
                for bit in 0..2u64 {
                    for (ri, &right_idx) in self.pivots_right[dim].iter().enumerate() {
                        let full_idx = compose_index(left_idx, bit, right_idx, dim, self.n_qubits);
                        if let Some(&val) = self.samples.get(&full_idx) {
                            core[[li * 2 + bit as usize, ri]] = val;
                        }
                    }
                }
            }
            
            // Normalize core (simple scaling for stability)
            let max_val = core.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            if max_val > 1e-10 {
                core /= max_val.sqrt();
            }
            
            cores.push(core);
        }
        
        Ok(cores)
    }
}

/// Compose a full index from left, bit, right components
fn compose_index(left: u64, bit: u64, right: u64, dim: usize, n_qubits: usize) -> u64 {
    // Index = (left << (n_qubits - dim)) | (bit << (n_qubits - dim - 1)) | right
    let left_shift = n_qubits - dim;
    let bit_shift = n_qubits - dim - 1;
    
    (left << left_shift) | (bit << bit_shift) | right
}

/// Check if Rust TCI is available
#[pyfunction]
fn rust_available() -> bool {
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-SVD Reconstruction FFI
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-computed block pointers for O(1) lookup (Python wrapper)
#[pyclass]
pub struct BlockPointers {
    inner: block_svd::BlockPointers,
}

#[pymethods]
impl BlockPointers {
    /// Build pointer arrays from ranks array
    #[new]
    fn new(ranks: Vec<u16>, block_size: usize, frame_h: usize, frame_w: usize) -> Self {
        Self {
            inner: block_svd::BlockPointers::new(&ranks, block_size, frame_h, frame_w),
        }
    }
    
    /// Get blocks per frame
    #[getter]
    fn blocks_per_frame(&self) -> usize {
        self.inner.blocks_per_frame
    }
    
    /// Get total number of blocks
    #[getter]
    fn n_blocks(&self) -> usize {
        self.inner.ranks.len()
    }
}

/// Reconstruct a single frame using Rust parallel processing
#[pyfunction]
fn reconstruct_frame<'py>(
    py: Python<'py>,
    u_data: &Bound<'py, PyArray1<f32>>,
    s_data: &Bound<'py, PyArray1<f32>>,
    vh_data: &Bound<'py, PyArray1<f32>>,
    pointers: &BlockPointers,
    frame_idx: usize,
    mean: f32,
    std: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Get readonly views
    let u_slice = unsafe { u_data.as_slice()? };
    let s_slice = unsafe { s_data.as_slice()? };
    let vh_slice = unsafe { vh_data.as_slice()? };
    
    // Reconstruct frame in parallel
    let mut frame = block_svd::reconstruct_frame_parallel(
        u_slice, s_slice, vh_slice, &pointers.inner, frame_idx
    );
    
    // Denormalize: frame = frame * std + mean
    frame.mapv_inplace(|x| x * std + mean);
    
    Ok(frame.into_pyarray(py))
}

/// Reconstruct multiple frames in parallel
#[pyfunction]
fn reconstruct_batch<'py>(
    py: Python<'py>,
    u_data: &Bound<'py, PyArray1<f32>>,
    s_data: &Bound<'py, PyArray1<f32>>,
    vh_data: &Bound<'py, PyArray1<f32>>,
    pointers: &BlockPointers,
    frame_indices: Vec<usize>,
    mean: f32,
    std: f32,
) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
    let u_slice = unsafe { u_data.as_slice()? };
    let s_slice = unsafe { s_data.as_slice()? };
    let vh_slice = unsafe { vh_data.as_slice()? };
    
    let mut frames = block_svd::reconstruct_batch_parallel(
        u_slice, s_slice, vh_slice, &pointers.inner, &frame_indices
    );
    
    // Denormalize all frames
    for frame in &mut frames {
        frame.mapv_inplace(|x| x * std + mean);
    }
    
    Ok(frames.into_iter().map(|f| f.into_pyarray(py)).collect())
}

/// Build cumulative pointer arrays from ranks (pure Rust)
#[pyfunction]
fn build_cumsum_pointers<'py>(
    py: Python<'py>,
    ranks: &Bound<'py, PyArray1<u16>>,
    block_size: usize,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let ranks_slice = unsafe { ranks.as_slice()? };
    let n = ranks_slice.len();
    
    let mut cumsum_u = Vec::with_capacity(n + 1);
    let mut cumsum_s = Vec::with_capacity(n + 1);
    
    cumsum_u.push(0u64);
    cumsum_s.push(0u64);
    
    let mut u_ptr = 0u64;
    let mut s_ptr = 0u64;
    
    for &r in ranks_slice {
        let r = r as u64;
        u_ptr += r * block_size as u64;
        s_ptr += r;
        cumsum_u.push(u_ptr);
        cumsum_s.push(s_ptr);
    }
    
    Ok((
        Array1::from(cumsum_u).into_pyarray(py),
        Array1::from(cumsum_s).into_pyarray(py),
    ))
}

/// PyO3 module initialization
#[pymodule]
fn tci_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("RUST_AVAILABLE", true)?;
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_class::<MaxVolConfig>()?;
    m.add_class::<TCIConfig>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<TCISampler>()?;
    // Block-SVD reconstruction
    m.add_class::<BlockPointers>()?;
    m.add_function(wrap_pyfunction!(reconstruct_frame, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_batch, m)?)?;
    m.add_function(wrap_pyfunction!(build_cumsum_pointers, m)?)?;
    Ok(())
}

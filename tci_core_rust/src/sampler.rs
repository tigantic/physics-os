//! TCI Sampler: Fiber-based and adaptive sampling strategies
//!
//! The TCI sampler generates batches of sample points for flux evaluation.
//! Two strategies:
//!
//! 1. **Fiber sampling**: Sample along 1D fibers through the tensor.
//!    Used for building skeleton matrices in cross-interpolation.
//!
//! 2. **Adaptive sampling**: Focus samples in regions of high error.
//!    Used for shock capturing and refinement.
//!
//! CRITICAL: Batch size must be >= 10,000 for GPU efficiency.

use pyo3::prelude::*;
use rand::prelude::*;
use rand::distributions::Uniform;
use rustc_hash::FxHashSet;

use crate::indices::{IndexBatch, BoundaryCondition};

/// Sampling strategy enum
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Fiber-based sampling for TCI skeleton construction
    Fiber,
    /// Random sampling for initial exploration
    Random,
    /// Adaptive sampling focused on high-error regions
    Adaptive,
    /// Uniform grid sampling
    Uniform,
}

/// TCI Sampler for generating sample batches
#[pyclass]
pub struct TCISampler {
    /// Number of qubits (domain size = 2^n_qubits)
    #[allow(dead_code)]
    n_qubits: usize,
    
    /// Domain size
    domain_size: u64,
    
    /// Boundary condition type
    boundary: String,
    
    /// Minimum batch size (default 10000)
    min_batch_size: usize,
    
    /// Current pivot rows (for fiber sampling)
    pivot_rows: Vec<Vec<u64>>,
    
    /// Current pivot cols (for fiber sampling)
    pivot_cols: Vec<Vec<u64>>,
    
    /// Sampled indices so far (for adaptive deduplication)
    sampled_set: FxHashSet<u64>,
    
    /// Error estimates at sampled points (for adaptive refinement)
    error_map: Vec<(u64, f64)>,
    
    /// Random number generator
    rng: StdRng,
}

#[pymethods]
impl TCISampler {
    #[new]
    pub fn new(n_qubits: usize, boundary: &str, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        TCISampler {
            n_qubits,
            domain_size: 1u64 << n_qubits,
            boundary: boundary.to_string(),
            min_batch_size: 10_000,
            pivot_rows: Vec::new(),
            pivot_cols: Vec::new(),
            sampled_set: FxHashSet::default(),
            error_map: Vec::new(),
            rng,
        }
    }
    
    /// Set minimum batch size (default 10000)
    pub fn set_min_batch_size(&mut self, size: usize) {
        self.min_batch_size = size;
    }
    
    /// Initialize pivot sets for a specific qubit position
    ///
    /// For TCI, we maintain row and column pivot indices at each level.
    /// These define the skeleton structure.
    pub fn init_pivots(&mut self, qubit: usize, row_pivots: Vec<u64>, col_pivots: Vec<u64>) {
        // Ensure we have enough levels
        while self.pivot_rows.len() <= qubit {
            self.pivot_rows.push(Vec::new());
            self.pivot_cols.push(Vec::new());
        }
        
        self.pivot_rows[qubit] = row_pivots;
        self.pivot_cols[qubit] = col_pivots;
    }
    
    /// Generate fiber samples for TCI at a given qubit level
    ///
    /// Fibers are 1D slices through the tensor. For QTT, a fiber at position k
    /// fixes all bits except bit k, which varies (0 or 1).
    pub fn sample_fibers(&mut self, qubit: usize) -> IndexBatch {
        let mut indices = Vec::new();
        
        // If we have pivot sets, sample at pivot combinations
        if qubit < self.pivot_rows.len() && !self.pivot_rows[qubit].is_empty() {
            let rows = &self.pivot_rows[qubit];
            let cols = &self.pivot_cols[qubit];
            
            // Generate fiber indices at pivot intersections
            for &row_base in rows {
                for &col_base in cols {
                    // Combine row and col multi-indices
                    let combined = self.combine_indices(row_base, col_base, qubit);
                    
                    // Fiber: vary bit at position qubit
                    let mask = !(1u64 << qubit);
                    let base_cleared = combined & mask;
                    
                    indices.push(base_cleared);                    // bit k = 0
                    indices.push(base_cleared | (1u64 << qubit)); // bit k = 1
                }
            }
        }
        
        // Pad with random samples to meet minimum batch size
        self.pad_to_minimum(&mut indices);
        
        // Mark as sampled
        for &idx in &indices {
            self.sampled_set.insert(idx);
        }
        
        IndexBatch::new(indices, self.domain_size, &self.boundary)
    }
    
    /// Generate random samples for initial exploration
    pub fn sample_random(&mut self, count: usize) -> IndexBatch {
        let target = count.max(self.min_batch_size);
        let dist = Uniform::new(0u64, self.domain_size);
        
        let mut indices = Vec::with_capacity(target);
        let mut attempts = 0;
        let max_attempts = target * 10;
        
        while indices.len() < target && attempts < max_attempts {
            let idx = dist.sample(&mut self.rng);
            if !self.sampled_set.contains(&idx) {
                indices.push(idx);
                self.sampled_set.insert(idx);
            }
            attempts += 1;
        }
        
        // If we can't find enough unique samples, allow duplicates
        while indices.len() < target {
            indices.push(dist.sample(&mut self.rng));
        }
        
        IndexBatch::new(indices, self.domain_size, &self.boundary)
    }
    
    /// Generate adaptive samples focused on high-error regions
    ///
    /// Uses error estimates from previous iterations to focus sampling
    /// where the approximation is worst (typically at shocks).
    pub fn sample_adaptive(&mut self, error_threshold: f64) -> IndexBatch {
        let mut indices = Vec::new();
        
        // Sort error map by error (descending)
        // Handle NaN gracefully - treat NaN as less than all valid values
        let mut sorted_errors: Vec<_> = self.error_map.clone();
        sorted_errors.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Sample around high-error points
        for (idx, err) in sorted_errors.iter() {
            if *err < error_threshold {
                break;
            }
            
            // Add the high-error point
            if !self.sampled_set.contains(idx) {
                indices.push(*idx);
                self.sampled_set.insert(*idx);
            }
            
            // Add neighbors (shock fronts often need neighbor refinement)
            let bc = BoundaryCondition::parse(&self.boundary);
            let (left, right) = crate::indices::compute_neighbors(&[*idx], self.domain_size, bc);
            
            if !self.sampled_set.contains(&left[0]) {
                indices.push(left[0]);
                self.sampled_set.insert(left[0]);
            }
            if !self.sampled_set.contains(&right[0]) {
                indices.push(right[0]);
                self.sampled_set.insert(right[0]);
            }
        }
        
        // Pad to minimum
        self.pad_to_minimum(&mut indices);
        
        IndexBatch::new(indices, self.domain_size, &self.boundary)
    }
    
    /// Generate uniform grid samples
    pub fn sample_uniform(&mut self, stride: usize) -> IndexBatch {
        let stride = stride.max(1) as u64;
        let mut indices = Vec::new();
        
        let mut i = 0u64;
        while i < self.domain_size {
            if !self.sampled_set.contains(&i) {
                indices.push(i);
                self.sampled_set.insert(i);
            }
            i += stride;
        }
        
        self.pad_to_minimum(&mut indices);
        
        IndexBatch::new(indices, self.domain_size, &self.boundary)
    }
    
    /// Update error estimates from flux evaluation results
    pub fn update_errors(&mut self, indices: Vec<u64>, errors: Vec<f64>) {
        for (idx, err) in indices.into_iter().zip(errors.into_iter()) {
            self.error_map.push((idx, err));
        }
        
        // Keep error map bounded (most recent errors)
        if self.error_map.len() > 100_000 {
            self.error_map.drain(0..50_000);
        }
    }
    
    /// Clear sampling history (for new TCI iteration)
    pub fn reset(&mut self) {
        self.sampled_set.clear();
        self.error_map.clear();
        self.pivot_rows.clear();
        self.pivot_cols.clear();
    }
    
    /// Get number of unique samples taken
    pub fn num_samples(&self) -> usize {
        self.sampled_set.len()
    }
    
    /// Get current domain size
    pub fn get_domain_size(&self) -> u64 {
        self.domain_size
    }
}

impl TCISampler {
    /// Combine row and column base indices at a qubit level
    ///
    /// For TCI, row indices cover bits [0, k) and column indices cover bits [k, n).
    /// This combines them into a single index.
    fn combine_indices(&self, row_base: u64, col_base: u64, qubit: usize) -> u64 {
        // Row base has bits [0, qubit), col base has bits [qubit, n_qubits)
        // Mask row to keep only lower bits, shift col to upper bits
        let row_mask = (1u64 << qubit) - 1;
        let row_part = row_base & row_mask;
        let col_part = col_base << qubit;
        
        row_part | col_part
    }
    
    /// Pad sample vector to minimum batch size with random samples
    fn pad_to_minimum(&mut self, indices: &mut Vec<u64>) {
        if indices.len() >= self.min_batch_size {
            return;
        }
        
        let needed = self.min_batch_size - indices.len();
        let dist = Uniform::new(0u64, self.domain_size);
        
        for _ in 0..needed {
            let idx = dist.sample(&mut self.rng);
            indices.push(idx);
        }
    }
}

/// Generate pivot indices for QTT cross-interpolation
///
/// Initial pivots can be:
/// - Uniform spacing
/// - Random
/// - Based on prior knowledge (e.g., expected shock locations)
pub fn generate_initial_pivots(
    n_qubits: usize,
    rank: usize,
    strategy: &str,
) -> (Vec<u64>, Vec<u64>) {
    let domain_size = 1u64 << n_qubits;
    
    match strategy {
        "uniform" => {
            let stride = (domain_size / rank as u64).max(1);
            let row_pivots: Vec<u64> = (0..rank as u64).map(|i| i * stride).collect();
            let col_pivots: Vec<u64> = (0..rank as u64).map(|i| (i * stride + stride / 2) % domain_size).collect();
            (row_pivots, col_pivots)
        }
        "random" => {
            let mut rng = thread_rng();
            let dist = Uniform::new(0u64, domain_size);
            let row_pivots: Vec<u64> = (0..rank).map(|_| dist.sample(&mut rng)).collect();
            let col_pivots: Vec<u64> = (0..rank).map(|_| dist.sample(&mut rng)).collect();
            (row_pivots, col_pivots)
        }
        _ => {
            // Default to uniform
            generate_initial_pivots(n_qubits, rank, "uniform")
        }
    }
}

/// Batch iterator for processing large sample sets
pub struct SampleBatchIterator {
    indices: Vec<u64>,
    batch_size: usize,
    current: usize,
}

impl SampleBatchIterator {
    pub fn new(indices: Vec<u64>, batch_size: usize) -> Self {
        SampleBatchIterator {
            indices,
            batch_size,
            current: 0,
        }
    }
}

impl Iterator for SampleBatchIterator {
    type Item = Vec<u64>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }
        
        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.current..end].to_vec();
        self.current = end;
        
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampler_creation() {
        let sampler = TCISampler::new(10, "periodic", Some(42));
        assert_eq!(sampler.domain_size, 1024);
        assert_eq!(sampler.n_qubits, 10);
    }
    
    #[test]
    fn test_random_sampling() {
        let mut sampler = TCISampler::new(10, "periodic", Some(42));
        sampler.set_min_batch_size(100);
        
        let batch = sampler.sample_random(50);
        assert!(batch.len() >= 100);  // Padded to min
        
        // All indices should be in range
        for idx in &batch.indices {
            assert!(*idx < 1024);
        }
    }
    
    #[test]
    fn test_uniform_sampling() {
        let mut sampler = TCISampler::new(10, "periodic", Some(42));
        sampler.set_min_batch_size(10);
        
        let batch = sampler.sample_uniform(64);
        
        // Should have ~16 uniformly spaced samples + padding
        assert!(batch.indices.iter().any(|&x| x == 0));
        assert!(batch.indices.iter().any(|&x| x == 64));
    }
    
    #[test]
    fn test_combine_indices() {
        let sampler = TCISampler::new(8, "periodic", Some(42));
        
        // qubit=4: row has bits [0,4), col has bits [4,8)
        let row_base = 0b0101u64;  // 5
        let col_base = 0b1010u64;  // 10, but shifted: 10 << 4 = 160
        
        let combined = sampler.combine_indices(row_base, col_base, 4);
        // row_part = 5 & 0xF = 5
        // col_part = 10 << 4 = 160
        // combined = 5 | 160 = 165
        assert_eq!(combined, 165);
    }
    
    #[test]
    fn test_initial_pivots_uniform() {
        let (rows, cols) = generate_initial_pivots(10, 8, "uniform");
        
        assert_eq!(rows.len(), 8);
        assert_eq!(cols.len(), 8);
        
        // Should be evenly spaced
        let expected_stride = 1024 / 8;
        for (i, &r) in rows.iter().enumerate() {
            assert_eq!(r, i as u64 * expected_stride as u64);
        }
    }
    
    #[test]
    fn test_batch_iterator() {
        let indices: Vec<u64> = (0..100).collect();
        let mut iter = SampleBatchIterator::new(indices, 30);
        
        let b1 = iter.next().unwrap();
        assert_eq!(b1.len(), 30);
        
        let b2 = iter.next().unwrap();
        assert_eq!(b2.len(), 30);
        
        let b3 = iter.next().unwrap();
        assert_eq!(b3.len(), 30);
        
        let b4 = iter.next().unwrap();
        assert_eq!(b4.len(), 10);  // Remaining
        
        assert!(iter.next().is_none());
    }
}

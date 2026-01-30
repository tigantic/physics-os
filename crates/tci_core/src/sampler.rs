//! TCI Sampler: Coordinates function evaluation batches
//!
//! This module manages the batching of function evaluations for TCI.

use crate::skeleton::TCISkeleton;
use ndarray::Array3;

/// Sampler state
pub struct Sampler {
    skeleton: TCISkeleton,
    max_rank: usize,
    batch_size: usize,
    total_evals: u64,
    iteration: usize,
}

impl Sampler {
    /// Create new sampler
    pub fn new(n_sites: usize, max_rank: usize, batch_size: usize) -> Self {
        Self {
            skeleton: TCISkeleton::new(n_sites, max_rank),
            max_rank,
            batch_size,
            total_evals: 0,
            iteration: 0,
        }
    }

    /// Get indices to sample
    pub fn get_indices(&self) -> Vec<u64> {
        self.skeleton.get_sample_indices(self.batch_size)
    }

    /// Submit sample values
    pub fn submit(&mut self, indices: &[u64], values: &[f64]) {
        self.skeleton.submit_samples(indices, values);
        self.total_evals += indices.len() as u64;
        self.iteration += 1;
        self.skeleton.update_pivots(self.max_rank);
    }

    /// Check convergence
    pub fn is_converged(&self, max_iterations: usize) -> bool {
        let n = 1u64 << self.skeleton.n_sites;
        let coverage = self.skeleton.samples.len() as f64 / n as f64;
        coverage > 0.5 || self.iteration >= max_iterations
    }

    /// Get total function evaluations
    pub fn total_evals(&self) -> u64 {
        self.total_evals
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.skeleton.samples.len()
    }

    /// Build TT cores
    pub fn build_cores(&self) -> Vec<Array3<f64>> {
        self.skeleton.build_cores()
    }
}

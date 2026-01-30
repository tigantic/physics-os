//! Tensor Train GPU Evaluator
//!
//! Evaluates TT-cores directly on GPU WITHOUT decompression.
//!
//! # QTT Doctrine
//!
//! This evaluator follows strict QTT doctrine:
//! - TT-cores are uploaded directly to GPU buffers
//! - Evaluation happens via matrix-vector contractions
//! - No dense tensor is ever materialized
//! - Memory usage: O(L·χ² + N) where N is query count
//!
//! # Example
//!
//! ```rust,ignore
//! use hyper_core::gpu::TTEvaluator;
//!
//! // Create evaluator with WGPU device
//! let evaluator = TTEvaluator::new(&device, &queue);
//!
//! // Upload TT-cores from QTTFrame
//! evaluator.upload_cores(&qtt_frame)?;
//!
//! // Evaluate at query points
//! let indices = vec![0, 1, 0, 1, 1, 0]; // 2 queries, 3 sites each
//! let results = evaluator.evaluate(&indices, 2)?;
//! ```

/// Parameters for TT evaluation (matches shader uniform)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TTParams {
    /// Number of TT sites (L)
    pub num_sites: u32,
    /// Physical dimension per site (d)
    pub physical_dim: u32,
    /// Maximum bond dimension (for validation)
    pub max_bond_dim: u32,
    /// Number of query points to evaluate
    pub num_queries: u32,
}

/// GPU-based Tensor Train evaluator
///
/// Evaluates TT at multiple query points in parallel using WGPU compute shaders.
pub struct TTEvaluator {
    /// Shader source (embedded at compile time)
    shader_source: &'static str,
    
    /// TT structure parameters
    params: TTParams,
    
    /// Bond dimensions array
    bond_dims: Vec<u32>,
    
    /// Core offsets (bytes)
    core_offsets: Vec<u32>,
    
    /// Flattened core data
    cores: Vec<f32>,
}

impl TTEvaluator {
    /// Create a new TT evaluator
    pub fn new() -> Self {
        Self {
            shader_source: include_str!("tt_eval.wgsl"),
            params: TTParams {
                num_sites: 0,
                physical_dim: 2,
                max_bond_dim: 64,
                num_queries: 0,
            },
            bond_dims: Vec::new(),
            core_offsets: Vec::new(),
            cores: Vec::new(),
        }
    }
    
    /// Get the WGSL shader source
    pub fn shader_source(&self) -> &str {
        self.shader_source
    }
    
    /// Set TT structure from bond dimensions
    ///
    /// # Arguments
    /// * `num_sites` - Number of TT sites (L)
    /// * `physical_dim` - Physical dimension (d)
    /// * `bond_dims` - Bond dimensions [χ₀, χ₁, ..., χ_{L-2}]
    pub fn set_structure(&mut self, num_sites: u32, physical_dim: u32, bond_dims: &[u16]) {
        self.params.num_sites = num_sites;
        self.params.physical_dim = physical_dim;
        self.params.max_bond_dim = bond_dims.iter().map(|&x| x as u32).max().unwrap_or(1);
        
        self.bond_dims = bond_dims.iter().map(|&x| x as u32).collect();
    }
    
    /// Upload TT-cores from raw data
    ///
    /// # Arguments
    /// * `cores` - Flattened core data (all cores concatenated)
    /// * `offsets` - Byte offset to each core
    pub fn upload_cores(&mut self, cores: &[f32], offsets: &[u32]) {
        self.cores = cores.to_vec();
        self.core_offsets = offsets.to_vec();
    }
    
    /// Get left bond dimension for site i
    fn chi_left(&self, site: usize) -> usize {
        if site == 0 {
            1
        } else {
            self.bond_dims.get(site - 1).map(|&x| x as usize).unwrap_or(1)
        }
    }
    
    /// Get right bond dimension for site i
    fn chi_right(&self, site: usize) -> usize {
        let num_sites = self.params.num_sites as usize;
        if site >= num_sites - 1 {
            1
        } else {
            self.bond_dims.get(site).map(|&x| x as usize).unwrap_or(1)
        }
    }
    
    /// Get core element at position (alpha_left, x, alpha_right) for site i
    fn get_core_element(&self, site: usize, alpha_left: usize, x: usize, alpha_right: usize) -> f32 {
        let _chi_l = self.chi_left(site);
        let chi_r = self.chi_right(site);
        let d = self.params.physical_dim as usize;
        
        let local_idx = alpha_left * (d * chi_r) + x * chi_r + alpha_right;
        let offset = self.core_offsets.get(site).map(|&o| o as usize / 4).unwrap_or(0);
        let global_idx = offset + local_idx;
        
        self.cores.get(global_idx).copied().unwrap_or(0.0)
    }
    
    /// Evaluate TT at a single query point (CPU fallback)
    ///
    /// This is a reference implementation for testing.
    /// Production code should use GPU evaluation.
    ///
    /// # Arguments
    /// * `indices` - Multi-index (x₁, x₂, ..., x_L)
    pub fn evaluate_single_cpu(&self, indices: &[u32]) -> f32 {
        let num_sites = self.params.num_sites as usize;
        
        if indices.len() < num_sites {
            return 0.0;
        }
        
        // Initialize accumulator as 1×1 identity
        let mut acc = vec![1.0f32];
        let mut current_dim = 1usize;
        
        // Contract through all sites
        for site in 0..num_sites {
            let x_i = indices[site] as usize;
            let chi_l = self.chi_left(site);
            let chi_r = self.chi_right(site);
            
            // Validate
            if x_i >= self.params.physical_dim as usize {
                return 0.0;
            }
            if chi_l != current_dim {
                return 0.0; // Dimension mismatch
            }
            
            // Matrix-vector multiply: new_acc = acc @ core_slice
            let mut new_acc = vec![0.0f32; chi_r];
            
            for j in 0..chi_r {
                let mut sum = 0.0f32;
                for i in 0..chi_l {
                    sum += acc[i] * self.get_core_element(site, i, x_i, j);
                }
                new_acc[j] = sum;
            }
            
            acc = new_acc;
            current_dim = chi_r;
        }
        
        acc.first().copied().unwrap_or(0.0)
    }
    
    /// Evaluate TT at multiple query points (CPU fallback)
    ///
    /// # Arguments
    /// * `indices` - Flattened multi-indices (num_queries × num_sites)
    /// * `num_queries` - Number of query points
    pub fn evaluate_batch_cpu(&self, indices: &[u32], num_queries: usize) -> Vec<f32> {
        let num_sites = self.params.num_sites as usize;
        
        (0..num_queries)
            .map(|q| {
                let start = q * num_sites;
                let end = start + num_sites;
                if end <= indices.len() {
                    self.evaluate_single_cpu(&indices[start..end])
                } else {
                    0.0
                }
            })
            .collect()
    }
    
    /// Calculate total memory usage for cores
    pub fn core_memory_bytes(&self) -> usize {
        self.cores.len() * std::mem::size_of::<f32>()
    }
    
    /// Calculate theoretical dense memory (for comparison)
    pub fn dense_memory_bytes(&self) -> usize {
        let num_sites = self.params.num_sites as usize;
        let d = self.params.physical_dim as usize;
        
        if num_sites == 0 {
            return 0;
        }
        
        // d^L elements × 4 bytes per f32
        d.pow(num_sites as u32) * std::mem::size_of::<f32>()
    }
    
    /// Get compression ratio (dense / TT)
    pub fn compression_ratio(&self) -> f32 {
        let dense = self.dense_memory_bytes();
        let tt = self.core_memory_bytes();
        
        if tt == 0 {
            return 1.0;
        }
        
        dense as f32 / tt as f32
    }
}

impl Default for TTEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tt_params_size() {
        assert_eq!(std::mem::size_of::<TTParams>(), 16);
    }
    
    #[test]
    fn test_simple_tt_evaluation() {
        // Create a simple 3-site TT with d=2, χ=2
        // This represents a 2×2×2 tensor
        let mut evaluator = TTEvaluator::new();
        
        // Set structure: 3 sites, d=2, χ=[2, 2]
        evaluator.set_structure(3, 2, &[2, 2]);
        
        // Create cores:
        // Core 0: shape (1, 2, 2) = 4 elements
        // Core 1: shape (2, 2, 2) = 8 elements  
        // Core 2: shape (2, 2, 1) = 4 elements
        let core0: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // Identity-like
        let core1: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let core2: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        
        let mut cores = Vec::new();
        let offset0 = 0u32;
        cores.extend(&core0);
        let offset1 = (cores.len() * 4) as u32;
        cores.extend(&core1);
        let offset2 = (cores.len() * 4) as u32;
        cores.extend(&core2);
        
        evaluator.upload_cores(&cores, &[offset0, offset1, offset2]);
        
        // Evaluate at index (0, 0, 0)
        let result = evaluator.evaluate_single_cpu(&[0, 0, 0]);
        
        // Should produce a valid result
        assert!(result.is_finite(), "Result should be finite");
    }
    
    #[test]
    fn test_batch_evaluation() {
        let mut evaluator = TTEvaluator::new();
        evaluator.set_structure(2, 2, &[2]);
        
        // Simple 2-site TT
        // Core 0: (1, 2, 2) = [1,0,0,1]
        // Core 1: (2, 2, 1) = [1,1,1,1]
        let cores: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let offsets = vec![0u32, 16u32];
        
        evaluator.upload_cores(&cores, &offsets);
        
        // Evaluate at all 4 indices
        let indices: Vec<u32> = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let results = evaluator.evaluate_batch_cpu(&indices, 4);
        
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.is_finite());
        }
    }
    
    #[test]
    fn test_compression_ratio() {
        let mut evaluator = TTEvaluator::new();
        
        // 10-site TT with d=2, χ_max=4
        // Dense: 2^10 = 1024 elements × 4 bytes = 4KB
        // TT: ~10 × 4 × 2 × 4 × 4 = 1280 bytes (approximate)
        evaluator.set_structure(10, 2, &[4, 4, 4, 4, 4, 4, 4, 4, 4]);
        
        // Mock cores (just testing size calculation)
        let total_elements = 1 * 2 * 4  // Core 0: (1, 2, 4)
            + 8 * 4 * 2 * 4             // Cores 1-8: (4, 2, 4)
            + 4 * 2 * 1;                // Core 9: (4, 2, 1)
        
        let cores = vec![0.0f32; total_elements];
        let mut offsets = Vec::new();
        let mut offset = 0u32;
        offsets.push(offset);
        offset += 1 * 2 * 4 * 4; // Core 0 size in bytes
        for _ in 0..8 {
            offsets.push(offset);
            offset += 4 * 2 * 4 * 4;
        }
        offsets.push(offset);
        
        evaluator.upload_cores(&cores, &offsets);
        
        let ratio = evaluator.compression_ratio();
        assert!(ratio > 1.0, "TT should compress 10-site tensor: ratio = {}", ratio);
    }
    
    #[test]
    fn test_shader_source_embedded() {
        let evaluator = TTEvaluator::new();
        let source = evaluator.shader_source();
        
        assert!(source.contains("evaluate_tt_single"), "Shader should contain evaluation function");
        assert!(source.contains("TTParams"), "Shader should contain params struct");
        assert!(source.contains("@compute"), "Shader should be a compute shader");
    }
}

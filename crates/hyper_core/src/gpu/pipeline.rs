//! GPU Pipeline for TT Evaluation
//!
//! Provides WGPU-based compute pipeline for tensor train evaluation.
//!
//! # Feature Flag
//!
//! This module requires the `wgpu` feature to be enabled:
//! ```toml
//! hyper_core = { version = "0.1", features = ["wgpu"] }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                         GPU Pipeline Architecture                         │
//! ├───────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐  │
//! │  │ TT-cores    │───▶│ GPU Buffer  │───▶│  Compute    │───▶│ Results  │  │
//! │  │ (CPU mmap)  │    │ (zero-copy) │    │  Shader     │    │ Buffer   │  │
//! │  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘  │
//! │                                                                           │
//! │  Memory flow: /dev/shm → wgpu::Buffer(MappedAtCreation) → GPU            │
//! │                                                                           │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Zero-Copy Strategy
//!
//! 1. TT-cores reside in shared memory (`/dev/shm/`)
//! 2. WGPU buffer created with `MappedAtCreation` flag
//! 3. Data copied once: shm → GPU buffer staging
//! 4. GPU evaluates TT without decompression
//! 5. Results copied back: GPU → host buffer

use super::tt_eval::TTEvaluator;

/// GPU buffer layout for TT-cores
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoreBufferLayout {
    /// Total size of all cores in bytes
    pub total_bytes: u32,
    /// Number of TT sites
    pub num_sites: u32,
    /// Maximum bond dimension
    pub max_bond_dim: u32,
    /// Physical dimension
    pub physical_dim: u32,
}

const _: () = assert!(std::mem::size_of::<CoreBufferLayout>() == 16);

/// Query buffer layout
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QueryBufferLayout {
    /// Number of query points
    pub num_queries: u32,
    /// Sites per query
    pub sites_per_query: u32,
    /// Padding for alignment
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<QueryBufferLayout>() == 16);

/// GPU Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Workgroup size (must match shader)
    pub workgroup_size: u32,
    /// Maximum queries per dispatch
    pub max_queries_per_dispatch: u32,
    /// Enable async readback
    pub async_readback: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
            max_queries_per_dispatch: 1_048_576, // 1M queries
            async_readback: true,
        }
    }
}

/// GPU Pipeline for TT Evaluation
///
/// This struct is the entry point for GPU-accelerated TT evaluation.
/// Without the `wgpu` feature, it provides a CPU fallback.
pub struct TTPipeline {
    /// CPU fallback evaluator
    evaluator: TTEvaluator,
    
    /// Pipeline configuration
    #[allow(dead_code)]
    config: PipelineConfig,
    
    /// Whether GPU is available
    gpu_available: bool,
}

impl TTPipeline {
    /// Create a new TT pipeline with CPU fallback
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            evaluator: TTEvaluator::new(),
            config,
            gpu_available: false, // Will be true when wgpu feature is enabled
        }
    }
    
    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }
    
    /// Get the underlying CPU evaluator
    pub fn cpu_evaluator(&self) -> &TTEvaluator {
        &self.evaluator
    }
    
    /// Get mutable reference to CPU evaluator
    pub fn cpu_evaluator_mut(&mut self) -> &mut TTEvaluator {
        &mut self.evaluator
    }
    
    /// Set TT structure from bond dimensions
    pub fn set_structure(&mut self, num_sites: u32, physical_dim: u32, bond_dims: &[u16]) {
        self.evaluator.set_structure(num_sites, physical_dim, bond_dims);
    }
    
    /// Upload TT-cores for evaluation
    ///
    /// # Zero-Copy Path (with wgpu feature)
    ///
    /// When GPU is available:
    /// 1. Creates staging buffer with MappedAtCreation
    /// 2. Copies cores directly to GPU memory
    /// 3. Returns immediately (async upload)
    ///
    /// # CPU Path (without wgpu feature)
    ///
    /// Stores cores in CPU memory for evaluate_batch_cpu().
    pub fn upload_cores(&mut self, cores: &[f32], offsets: &[u32]) {
        self.evaluator.upload_cores(cores, offsets);
        
        // When wgpu feature is enabled, this would also:
        // 1. Create GPU buffer
        // 2. Map buffer
        // 3. Copy cores to buffer
        // 4. Unmap buffer
    }
    
    /// Evaluate TT at multiple query points
    ///
    /// Uses GPU if available, otherwise falls back to CPU.
    ///
    /// # Arguments
    /// * `indices` - Flattened multi-indices (num_queries × num_sites)
    /// * `num_queries` - Number of query points
    ///
    /// # Returns
    /// Vector of TT values at each query point
    pub fn evaluate(&self, indices: &[u32], num_queries: usize) -> Vec<f32> {
        if self.gpu_available {
            // GPU path would:
            // 1. Upload indices to GPU buffer
            // 2. Dispatch compute shader
            // 3. Wait for results
            // 4. Read back result buffer
            unimplemented!("GPU evaluation requires wgpu feature")
        } else {
            self.evaluator.evaluate_batch_cpu(indices, num_queries)
        }
    }
    
    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            core_bytes: self.evaluator.core_memory_bytes(),
            dense_bytes: self.evaluator.dense_memory_bytes(),
            compression_ratio: self.evaluator.compression_ratio(),
            gpu_buffer_bytes: 0, // Would be non-zero with wgpu
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Bytes used by TT-cores
    pub core_bytes: usize,
    /// Bytes that would be used by dense tensor
    pub dense_bytes: usize,
    /// Compression ratio (dense / TT)
    pub compression_ratio: f32,
    /// Bytes allocated on GPU
    pub gpu_buffer_bytes: usize,
}

impl MemoryStats {
    /// Memory saved compared to dense representation
    pub fn memory_saved(&self) -> usize {
        if self.dense_bytes > self.core_bytes {
            self.dense_bytes - self.core_bytes
        } else {
            0
        }
    }
    
    /// Percentage of memory saved
    pub fn savings_percent(&self) -> f32 {
        if self.dense_bytes > 0 {
            100.0 * (1.0 - (self.core_bytes as f32 / self.dense_bytes as f32))
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_layout_sizes() {
        assert_eq!(std::mem::size_of::<CoreBufferLayout>(), 16);
        assert_eq!(std::mem::size_of::<QueryBufferLayout>(), 16);
    }
    
    #[test]
    fn test_pipeline_cpu_fallback() {
        let mut pipeline = TTPipeline::new(PipelineConfig::default());
        
        // Set up simple 2-site TT
        pipeline.set_structure(2, 2, &[2]);
        
        // Upload cores
        let cores: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let offsets = vec![0u32, 16u32];
        pipeline.upload_cores(&cores, &offsets);
        
        // Evaluate
        let indices: Vec<u32> = vec![0, 0, 1, 1];
        let results = pipeline.evaluate(&indices, 2);
        
        assert_eq!(results.len(), 2);
        assert!(!pipeline.is_gpu_available());
    }
    
    #[test]
    fn test_memory_stats() {
        let mut pipeline = TTPipeline::new(PipelineConfig::default());
        
        // 10-site TT with χ=4
        pipeline.set_structure(10, 2, &[4, 4, 4, 4, 4, 4, 4, 4, 4]);
        
        // Mock cores
        let cores = vec![0.0f32; 500];
        let offsets = vec![0u32; 10];
        pipeline.upload_cores(&cores, &offsets);
        
        let stats = pipeline.memory_stats();
        
        // 10-site binary TT: 2^10 = 1024 elements = 4096 bytes dense
        assert_eq!(stats.dense_bytes, 4096);
        assert!(stats.compression_ratio >= 1.0);
        assert!(stats.savings_percent() >= 0.0);
    }
}

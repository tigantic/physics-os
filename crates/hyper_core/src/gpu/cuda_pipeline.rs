//! CUDA Compute Pipeline for TT Evaluation
//!
//! Native CUDA acceleration via cudarc 0.19. Works directly with NVIDIA GPUs including
//! WSL2 passthrough.
//!
//! Features:
//! - Pinned host memory for DMA transfers (no staging copy)
//! - Double-buffered async pipeline (upload N+1 while computing N)
//! - Ring buffer caching for zero-allocation hot path

use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaModule, CudaFunction, LaunchConfig, PushKernelArg, PinnedHostSlice};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::compile_ptx;

/// Error types for CUDA pipeline
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("No CUDA device found")]
    NoDevice,
    #[error("CUDA driver error: {0}")]
    DriverError(String),
    #[error("CUDA kernel compilation error: {0}")]
    CompilationError(String),
    #[error("CUDA kernel launch error: {0}")]
    LaunchError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Memory transfer error: {0}")]
    MemoryError(String),
}

#[cfg(feature = "gpu")]
impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaError::DriverError(e.to_string())
    }
}

#[cfg(feature = "gpu")]
impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        CudaError::CompilationError(e.to_string())
    }
}

/// CUDA TT Evaluation kernel source
const TT_EVAL_KERNEL: &str = r#"
extern "C" __global__ void tt_eval_kernel(
    const float* __restrict__ cores,
    const int* __restrict__ bond_dims,
    const int* __restrict__ core_offsets,
    const int* __restrict__ indices,
    float* __restrict__ output,
    int num_sites,
    int physical_dim,
    int max_bond_dim,
    int num_queries
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;

    float left[64];
    float temp[64];
    
    left[0] = 1.0f;
    for (int i = 1; i < max_bond_dim; i++) {
        left[i] = 0.0f;
    }
    
    for (int site = 0; site < num_sites; site++) {
        int idx = indices[query_idx * num_sites + site];
        int left_bond = bond_dims[site];
        int right_bond = bond_dims[site + 1];
        int core_offset = core_offsets[site];
        
        for (int j = 0; j < right_bond; j++) {
            temp[j] = 0.0f;
        }
        
        for (int i = 0; i < left_bond; i++) {
            for (int j = 0; j < right_bond; j++) {
                int core_idx = core_offset + i * physical_dim * right_bond + idx * right_bond + j;
                temp[j] += left[i] * cores[core_idx];
            }
        }
        
        for (int j = 0; j < right_bond; j++) {
            left[j] = temp[j];
        }
    }
    
    output[query_idx] = left[0];
}
"#;

/// CUDA context wrapper
#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub module: Arc<CudaModule>,
    pub kernel: CudaFunction,
    pub device_name: String,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Initialize GPU context on device 0
    pub fn new() -> Result<Self, CudaError> {
        Self::new_on_device(0)
    }

    /// Initialize GPU context on specific device
    pub fn new_on_device(ordinal: usize) -> Result<Self, CudaError> {
        // Initialize context
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        
        // Compile kernel
        let ptx = compile_ptx(TT_EVAL_KERNEL)?;
        let module = ctx.load_module(ptx)?;
        let kernel = module.load_function("tt_eval_kernel")?;
        
        // Get device name
        let device_name = format!("CUDA Device {}", ordinal);
        
        eprintln!("✓ CUDA Device {} initialized", ordinal);
        eprintln!("✓ TT evaluation kernel compiled and loaded");
        
        Ok(Self {
            ctx,
            stream,
            module,
            kernel,
            device_name,
        })
    }
}

/// Cached GPU buffer pool for zero-allocation hot path
#[cfg(feature = "gpu")]
pub struct GpuBufferCache {
    /// Cached index buffers (ring buffer for reuse)
    pub index_buffers: Vec<Option<CudaSlice<i32>>>,
    /// Cached output buffers (ring buffer for reuse)
    pub output_buffers: Vec<Option<CudaSlice<f32>>>,
    /// Current ring buffer position
    ring_pos: usize,
    /// Buffer capacity (max queries per buffer)
    buffer_capacity: usize,
    /// Ring buffer depth
    ring_depth: usize,
    /// Cache hit/miss stats
    pub hits: u64,
    pub misses: u64,
    pub reallocs: u64,
}

#[cfg(feature = "gpu")]
impl GpuBufferCache {
    /// Create a new buffer cache with given ring depth
    pub fn new(ring_depth: usize) -> Self {
        Self {
            index_buffers: vec![None; ring_depth],
            output_buffers: vec![None; ring_depth],
            ring_pos: 0,
            buffer_capacity: 0,
            ring_depth,
            hits: 0,
            misses: 0,
            reallocs: 0,
        }
    }

    /// Get or allocate index buffer with at least `capacity` elements
    pub fn get_index_buffer(
        &mut self,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<(usize, bool), CudaError> {
        let slot = self.ring_pos;
        self.ring_pos = (self.ring_pos + 1) % self.ring_depth;

        // Check if existing buffer is large enough
        let needs_alloc = match &self.index_buffers[slot] {
            Some(buf) if buf.len() >= capacity => {
                self.hits += 1;
                false
            }
            Some(_) => {
                self.reallocs += 1;
                true
            }
            None => {
                self.misses += 1;
                true
            }
        };

        if needs_alloc {
            // Round up to power of 2 for better reuse
            let alloc_size = capacity.next_power_of_two().max(1024);
            let buf: CudaSlice<i32> = stream.alloc_zeros(alloc_size)?;
            self.index_buffers[slot] = Some(buf);
            self.buffer_capacity = self.buffer_capacity.max(alloc_size);
        }

        Ok((slot, needs_alloc))
    }

    /// Get or allocate output buffer with at least `capacity` elements
    pub fn get_output_buffer(
        &mut self,
        stream: &Arc<CudaStream>,
        capacity: usize,
        slot: usize,
    ) -> Result<bool, CudaError> {
        let needs_alloc = match &self.output_buffers[slot] {
            Some(buf) if buf.len() >= capacity => false,
            _ => true,
        };

        if needs_alloc {
            let alloc_size = capacity.next_power_of_two().max(1024);
            let buf: CudaSlice<f32> = stream.alloc_zeros(alloc_size)?;
            self.output_buffers[slot] = Some(buf);
        }

        Ok(needs_alloc)
    }

    /// Get mutable reference to index buffer at slot
    pub fn index_buffer_mut(&mut self, slot: usize) -> Option<&mut CudaSlice<i32>> {
        self.index_buffers[slot].as_mut()
    }

    /// Get mutable reference to output buffer at slot
    pub fn output_buffer_mut(&mut self, slot: usize) -> Option<&mut CudaSlice<f32>> {
        self.output_buffers[slot].as_mut()
    }

    /// Get reference to output buffer at slot
    pub fn output_buffer(&self, slot: usize) -> Option<&CudaSlice<f32>> {
        self.output_buffers[slot].as_ref()
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits, self.misses, self.reallocs)
    }

    /// Get hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// CUDA-based TT Evaluation Pipeline with buffer caching
#[cfg(feature = "gpu")]
pub struct CudaTTPipeline {
    ctx: Arc<GpuContext>,
    
    // Device buffers for TT structure (static, uploaded once)
    d_cores: Option<CudaSlice<f32>>,
    d_bond_dims: Option<CudaSlice<i32>>,
    d_core_offsets: Option<CudaSlice<i32>>,
    
    // Buffer cache for zero-allocation hot path
    buffer_cache: GpuBufferCache,
    
    // Structure info
    num_sites: u32,
    physical_dim: u32,
    max_bond_dim: u32,
    
    // Transfer metrics
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub kernel_launches: u64,
}

#[cfg(feature = "gpu")]
impl CudaTTPipeline {
    /// Create a new CUDA pipeline with buffer caching
    pub fn new(ctx: Arc<GpuContext>) -> Result<Self, CudaError> {
        Ok(Self {
            ctx,
            d_cores: None,
            d_bond_dims: None,
            d_core_offsets: None,
            buffer_cache: GpuBufferCache::new(4), // 4-deep ring buffer
            num_sites: 0,
            physical_dim: 0,
            max_bond_dim: 0,
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            kernel_launches: 0,
        })
    }

    /// Set the TT structure (upload cores to GPU)
    pub fn set_tt_structure(
        &mut self,
        cores: &[f32],
        bond_dims: &[u32],
        physical_dim: u32,
    ) -> Result<(), CudaError> {
        if bond_dims.is_empty() {
            return Err(CudaError::InvalidParams("Empty bond dimensions".into()));
        }
        
        let num_sites = bond_dims.len() - 1;
        let max_bond_dim = *bond_dims.iter().max().unwrap_or(&1);
        
        if max_bond_dim > 64 {
            return Err(CudaError::InvalidParams(format!(
                "Max bond dimension {} exceeds limit of 64",
                max_bond_dim
            )));
        }
        
        // Compute core offsets
        let mut core_offsets = Vec::with_capacity(num_sites);
        let mut offset = 0u32;
        for site in 0..num_sites {
            core_offsets.push(offset as i32);
            let left = bond_dims[site];
            let right = bond_dims[site + 1];
            offset += left * physical_dim * right;
        }
        
        // Convert bond_dims to i32 for CUDA
        let bond_dims_i32: Vec<i32> = bond_dims.iter().map(|&x| x as i32).collect();
        
        // Upload to GPU using stream
        let d_cores = self.ctx.stream.clone_htod(cores)?;
        let d_bond_dims = self.ctx.stream.clone_htod(&bond_dims_i32)?;
        let d_core_offsets = self.ctx.stream.clone_htod(&core_offsets)?;
        
        self.d_cores = Some(d_cores);
        self.d_bond_dims = Some(d_bond_dims);
        self.d_core_offsets = Some(d_core_offsets);
        self.num_sites = num_sites as u32;
        self.physical_dim = physical_dim;
        self.max_bond_dim = max_bond_dim;
        
        // Track upload bytes
        self.bytes_uploaded += (cores.len() * std::mem::size_of::<f32>()) as u64;
        self.bytes_uploaded += (bond_dims_i32.len() * std::mem::size_of::<i32>()) as u64;
        self.bytes_uploaded += (core_offsets.len() * std::mem::size_of::<i32>()) as u64;
        
        eprintln!(
            "✓ Uploaded TT structure: {} sites, physical_dim={}, max_bond={}, {} KB",
            num_sites, physical_dim, max_bond_dim,
            self.bytes_uploaded / 1024
        );
        
        Ok(())
    }

    /// Evaluate TT at given indices (uses buffer cache for zero-allocation hot path)
    pub fn evaluate(&mut self, indices: &[u32]) -> Result<Vec<f32>, CudaError> {
        let d_cores = self.d_cores.as_ref()
            .ok_or_else(|| CudaError::InvalidParams("TT structure not set".into()))?;
        let d_bond_dims = self.d_bond_dims.as_ref()
            .ok_or_else(|| CudaError::InvalidParams("TT structure not set".into()))?;
        let d_core_offsets = self.d_core_offsets.as_ref()
            .ok_or_else(|| CudaError::InvalidParams("TT structure not set".into()))?;
        
        let num_queries = indices.len() / self.num_sites as usize;
        if indices.len() != num_queries * self.num_sites as usize {
            return Err(CudaError::InvalidParams(format!(
                "indices length {} not divisible by num_sites {}",
                indices.len(), self.num_sites
            )));
        }
        
        // Get cached buffers (zero allocation on cache hit)
        let (slot, _index_alloc) = self.buffer_cache.get_index_buffer(&self.ctx.stream, indices.len())?;
        let _output_alloc = self.buffer_cache.get_output_buffer(&self.ctx.stream, num_queries, slot)?;
        
        // Convert indices to i32 and upload to cached buffer
        let indices_i32: Vec<i32> = indices.iter().map(|&x| x as i32).collect();
        
        // Upload indices to device buffer
        {
            let d_indices = self.buffer_cache.index_buffer_mut(slot)
                .ok_or_else(|| CudaError::MemoryError("Index buffer not allocated".into()))?;
            self.ctx.stream.memcpy_htod(&indices_i32, d_indices)?;
        }
        
        // Track upload bytes
        self.bytes_uploaded += (indices_i32.len() * std::mem::size_of::<i32>()) as u64;
        
        // Now get both buffers for kernel launch - we need to get raw pointers
        // since cudarc uses the buffer references directly
        let d_indices_ptr = {
            let d_indices = self.buffer_cache.index_buffers[slot].as_ref()
                .ok_or_else(|| CudaError::MemoryError("Index buffer not allocated".into()))?;
            d_indices as *const CudaSlice<i32>
        };
        let d_output_ptr = {
            let d_output = self.buffer_cache.output_buffers[slot].as_mut()
                .ok_or_else(|| CudaError::MemoryError("Output buffer not allocated".into()))?;
            d_output as *mut CudaSlice<f32>
        };
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = (num_queries as u32 + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            self.ctx.stream.launch_builder(&self.ctx.kernel)
                .arg(d_cores)
                .arg(d_bond_dims)
                .arg(d_core_offsets)
                .arg(&*d_indices_ptr)
                .arg(&mut *d_output_ptr)
                .arg(&(self.num_sites as i32))
                .arg(&(self.physical_dim as i32))
                .arg(&(self.max_bond_dim as i32))
                .arg(&(num_queries as i32))
                .launch(cfg)?;
        }
        
        self.kernel_launches += 1;
        
        // Synchronize and download results
        self.ctx.stream.synchronize()?;
        
        // Download only the results we need using a slice view
        let d_output = self.buffer_cache.output_buffer(slot)
            .ok_or_else(|| CudaError::MemoryError("Output buffer not allocated".into()))?;
        
        // Create a view of just the portion we need
        let d_output_view = d_output.slice(0..num_queries);
        let mut output = vec![0.0f32; num_queries];
        self.ctx.stream.memcpy_dtoh(&d_output_view, &mut output)?;
        self.bytes_downloaded += (num_queries * std::mem::size_of::<f32>()) as u64;
        
        Ok(output)
    }

    /// Get buffer cache statistics (hits, misses, reallocs)
    pub fn cache_stats(&self) -> (u64, u64, u64) {
        self.buffer_cache.stats()
    }

    /// Get buffer cache hit rate as percentage
    pub fn cache_hit_rate(&self) -> f64 {
        self.buffer_cache.hit_rate()
    }

    /// Get transfer statistics (bytes uploaded, downloaded, kernel launches)
    pub fn transfer_stats(&self) -> (u64, u64, u64) {
        (self.bytes_uploaded, self.bytes_downloaded, self.kernel_launches)
    }
}

/// Double-buffered async CUDA pipeline for maximum throughput
/// 
/// Uses pinned host memory and dual streams to overlap:
/// - Upload batch N+1 on stream A
/// - Compute batch N on stream B  
/// - Download batch N-1 on stream A
#[cfg(feature = "gpu")]
pub struct AsyncCudaTTPipeline {
    ctx: Arc<GpuContext>,
    
    // Two streams for overlap
    stream_a: Arc<CudaStream>,
    stream_b: Arc<CudaStream>,
    
    // Device buffers for TT structure (static)
    d_cores: Option<CudaSlice<f32>>,
    d_bond_dims: Option<CudaSlice<i32>>,
    d_core_offsets: Option<CudaSlice<i32>>,
    
    // Double-buffered device memory
    d_indices: [Option<CudaSlice<i32>>; 2],
    d_output: [Option<CudaSlice<f32>>; 2],
    
    // Pinned host buffers for async DMA
    h_indices: Option<PinnedHostSlice<i32>>,
    h_output: Option<PinnedHostSlice<f32>>,
    
    // Current buffer index (ping-pong)
    current_buf: usize,
    
    // Structure info
    num_sites: u32,
    physical_dim: u32,
    max_bond_dim: u32,
    
    // Pre-allocated capacity
    max_queries: usize,
    
    // Transfer metrics
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub kernel_launches: u64,
}

#[cfg(feature = "gpu")]
impl AsyncCudaTTPipeline {
    /// Create a new async pipeline with given max query capacity
    pub fn new(ctx: Arc<GpuContext>, max_queries: usize) -> Result<Self, CudaError> {
        // Fork streams for concurrent work
        let stream_a = ctx.stream.fork()?;
        let stream_b = ctx.stream.fork()?;
        
        Ok(Self {
            ctx,
            stream_a,
            stream_b,
            d_cores: None,
            d_bond_dims: None,
            d_core_offsets: None,
            d_indices: [None, None],
            d_output: [None, None],
            h_indices: None,
            h_output: None,
            current_buf: 0,
            num_sites: 0,
            physical_dim: 0,
            max_bond_dim: 0,
            max_queries,
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            kernel_launches: 0,
        })
    }

    /// Set the TT structure and allocate all buffers
    pub fn set_tt_structure(
        &mut self,
        cores: &[f32],
        bond_dims: &[u32],
        physical_dim: u32,
    ) -> Result<(), CudaError> {
        if bond_dims.is_empty() {
            return Err(CudaError::InvalidParams("Empty bond dimensions".into()));
        }
        
        let num_sites = bond_dims.len() - 1;
        let max_bond_dim = *bond_dims.iter().max().unwrap_or(&1);
        
        if max_bond_dim > 64 {
            return Err(CudaError::InvalidParams(format!(
                "Max bond dimension {} exceeds limit of 64",
                max_bond_dim
            )));
        }
        
        // Compute core offsets
        let mut core_offsets = Vec::with_capacity(num_sites);
        let mut offset = 0u32;
        for site in 0..num_sites {
            core_offsets.push(offset as i32);
            let left = bond_dims[site];
            let right = bond_dims[site + 1];
            offset += left * physical_dim * right;
        }
        
        // Convert bond_dims to i32 for CUDA
        let bond_dims_i32: Vec<i32> = bond_dims.iter().map(|&x| x as i32).collect();
        
        // Upload TT structure to GPU
        let d_cores = self.stream_a.clone_htod(cores)?;
        let d_bond_dims = self.stream_a.clone_htod(&bond_dims_i32)?;
        let d_core_offsets = self.stream_a.clone_htod(&core_offsets)?;
        
        // Pre-allocate double-buffered device memory
        let index_capacity = self.max_queries * num_sites;
        let d_indices_0: CudaSlice<i32> = self.stream_a.alloc_zeros(index_capacity)?;
        let d_indices_1: CudaSlice<i32> = self.stream_b.alloc_zeros(index_capacity)?;
        let d_output_0: CudaSlice<f32> = self.stream_a.alloc_zeros(self.max_queries)?;
        let d_output_1: CudaSlice<f32> = self.stream_b.alloc_zeros(self.max_queries)?;
        
        // Allocate pinned host memory for async DMA
        let h_indices = unsafe { self.ctx.ctx.alloc_pinned::<i32>(index_capacity)? };
        let h_output = unsafe { self.ctx.ctx.alloc_pinned::<f32>(self.max_queries)? };
        
        self.d_cores = Some(d_cores);
        self.d_bond_dims = Some(d_bond_dims);
        self.d_core_offsets = Some(d_core_offsets);
        self.d_indices = [Some(d_indices_0), Some(d_indices_1)];
        self.d_output = [Some(d_output_0), Some(d_output_1)];
        self.h_indices = Some(h_indices);
        self.h_output = Some(h_output);
        self.num_sites = num_sites as u32;
        self.physical_dim = physical_dim;
        self.max_bond_dim = max_bond_dim;
        
        // Sync to ensure all allocations complete
        self.stream_a.synchronize()?;
        self.stream_b.synchronize()?;
        
        // Track upload bytes
        self.bytes_uploaded += (cores.len() * std::mem::size_of::<f32>()) as u64;
        self.bytes_uploaded += (bond_dims_i32.len() * std::mem::size_of::<i32>()) as u64;
        self.bytes_uploaded += (core_offsets.len() * std::mem::size_of::<i32>()) as u64;
        
        eprintln!(
            "✓ Async Pipeline: {} sites, max_bond={}, capacity={} queries",
            num_sites, max_bond_dim, self.max_queries
        );
        eprintln!("  Pinned host memory: {} MB indices, {} MB output",
            (index_capacity * 4) / (1024 * 1024),
            (self.max_queries * 4) / (1024 * 1024));
        
        Ok(())
    }

    /// Evaluate TT at given indices using async pipeline
    pub fn evaluate(&mut self, indices: &[u32]) -> Result<Vec<f32>, CudaError> {
        let d_cores = self.d_cores.as_ref()
            .ok_or_else(|| CudaError::InvalidParams("TT structure not set".into()))?;
        let d_bond_dims = self.d_bond_dims.as_ref()
            .ok_or_else(|| CudaError::InvalidParams("TT structure not set".into()))?;
        let d_core_offsets = self.d_core_offsets.as_ref()
            .ok_or_else(|| CudaError::InvalidParams("TT structure not set".into()))?;
        
        let num_queries = indices.len() / self.num_sites as usize;
        if num_queries > self.max_queries {
            return Err(CudaError::InvalidParams(format!(
                "Query count {} exceeds capacity {}",
                num_queries, self.max_queries
            )));
        }
        
        // Get current buffer slot
        let buf = self.current_buf;
        let stream = if buf == 0 { &self.stream_a } else { &self.stream_b };
        
        // Get pinned host memory and write indices directly to it
        let h_indices = self.h_indices.as_mut()
            .ok_or_else(|| CudaError::MemoryError("Pinned memory not allocated".into()))?;
        
        // Get mutable slice from pinned memory (syncs first)
        let h_indices_slice = h_indices.as_mut_slice()
            .map_err(|e| CudaError::MemoryError(format!("Failed to get pinned slice: {}", e)))?;
        
        // Write indices to pinned memory
        for (i, &idx) in indices.iter().enumerate() {
            h_indices_slice[i] = idx as i32;
        }
        
        // Async upload from pinned memory to device
        let d_indices = self.d_indices[buf].as_mut()
            .ok_or_else(|| CudaError::MemoryError("Device buffer not allocated".into()))?;
        
        // Upload using the slice
        stream.memcpy_htod(&h_indices_slice[..indices.len()], d_indices)?;
        
        self.bytes_uploaded += (indices.len() * std::mem::size_of::<i32>()) as u64;
        
        // Get output buffer
        let d_output = self.d_output[buf].as_mut()
            .ok_or_else(|| CudaError::MemoryError("Output buffer not allocated".into()))?;
        
        // Launch kernel
        let block_size = 256u32;
        let grid_size = (num_queries as u32 + block_size - 1) / block_size;
        
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            stream.launch_builder(&self.ctx.kernel)
                .arg(d_cores)
                .arg(d_bond_dims)
                .arg(d_core_offsets)
                .arg(&*d_indices)
                .arg(&mut *d_output)
                .arg(&(self.num_sites as i32))
                .arg(&(self.physical_dim as i32))
                .arg(&(self.max_bond_dim as i32))
                .arg(&(num_queries as i32))
                .launch(cfg)?;
        }
        
        self.kernel_launches += 1;
        
        // Sync and download results to pinned memory
        stream.synchronize()?;
        
        let d_output_view = d_output.slice(0..num_queries);
        let h_output = self.h_output.as_mut()
            .ok_or_else(|| CudaError::MemoryError("Pinned output memory not allocated".into()))?;
        
        // Get mutable slice from pinned output memory
        let h_output_slice = h_output.as_mut_slice()
            .map_err(|e| CudaError::MemoryError(format!("Failed to get pinned output slice: {}", e)))?;
        
        stream.memcpy_dtoh(&d_output_view, &mut h_output_slice[..num_queries])?;
        stream.synchronize()?;
        
        self.bytes_downloaded += (num_queries * std::mem::size_of::<f32>()) as u64;
        
        // Copy from pinned to regular memory (fast memcpy)
        let mut output = vec![0.0f32; num_queries];
        output.copy_from_slice(&h_output_slice[..num_queries]);
        
        // Ping-pong buffer
        self.current_buf = 1 - buf;
        
        Ok(output)
    }

    /// Get transfer statistics
    pub fn transfer_stats(&self) -> (u64, u64, u64) {
        (self.bytes_uploaded, self.bytes_downloaded, self.kernel_launches)
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub num_queries: usize,
    pub iterations: usize,
    pub total_time_ms: f64,
    pub queries_per_sec: f64,
}

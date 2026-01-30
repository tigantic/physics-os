//! WGPU Compute Pipeline for TT Evaluation
//!
//! This module provides GPU-accelerated tensor train evaluation using WGPU.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                      WGPU Compute Pipeline                                │
//! ├───────────────────────────────────────────────────────────────────────────┤
//! │                                                                           │
//! │  CPU Side                           GPU Side                              │
//! │  ─────────                          ────────                              │
//! │  ┌──────────────┐                   ┌──────────────┐                      │
//! │  │ TT-cores     │───────upload──────▶ cores_buffer │                      │
//! │  └──────────────┘                   └──────┬───────┘                      │
//! │  ┌──────────────┐                          │                              │
//! │  │ Bond dims    │───────upload──────▶ bond_buffer──┐                      │
//! │  └──────────────┘                                  │                      │
//! │  ┌──────────────┐                                  ▼                      │
//! │  │ Core offsets │───────upload──────▶ offset_buf ──┼──▶ ┌────────────┐   │
//! │  └──────────────┘                                  │    │  Compute   │   │
//! │  ┌──────────────┐                                  │    │  Shader    │   │
//! │  │ Query indices│───────upload──────▶ index_buf ───┼──▶ │            │   │
//! │  └──────────────┘                                  │    │ tt_eval    │   │
//! │  ┌──────────────┐                                  │    │   .wgsl    │   │
//! │  │ Params       │───────upload──────▶ param_buf ───┘──▶ └─────┬──────┘   │
//! │  └──────────────┘                                             │          │
//! │                                                               ▼          │
//! │  ┌──────────────┐                   ┌──────────────┐                      │
//! │  │ Results      │◀──────read────────│ output_buf   │                      │
//! │  └──────────────┘                   └──────────────┘                      │
//! │                                                                           │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupLayout, Buffer, BufferUsages,
    ComputePipeline, Device, Queue, ShaderModule,
};

/// TT Parameters uniform buffer (matches shader struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TTParamsGpu {
    pub num_sites: u32,
    pub physical_dim: u32,
    pub max_bond_dim: u32,
    pub num_queries: u32,
}

const _: () = assert!(std::mem::size_of::<TTParamsGpu>() == 16);

/// Error types for GPU pipeline
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,
    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(String),
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),
    #[error("Buffer mapping failed: {0}")]
    BufferMapping(String),
    #[error("GPU not initialized")]
    NotInitialized,
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
}

/// GPU context holding WGPU resources
#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Initialize GPU context (blocking)
    pub fn new_blocking() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    /// Initialize GPU context (async)
    pub async fn new_async() -> Result<Self, GpuError> {
        // Try DX12 first (works best in WSL2), then Vulkan
        let backends = wgpu::Backends::DX12 | wgpu::Backends::VULKAN;
        
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        // Enumerate all adapters and find a REAL GPU (not llvmpipe/lavapipe/CPU)
        let adapters = instance.enumerate_adapters(backends);
        
        eprintln!("Available adapters:");
        let mut discrete_gpu: Option<wgpu::Adapter> = None;
        let mut integrated_gpu: Option<wgpu::Adapter> = None;
        
        for adapter in adapters {
            let info = adapter.get_info();
            let is_software = info.device_type == wgpu::DeviceType::Cpu
                || info.name.to_lowercase().contains("llvmpipe")
                || info.name.to_lowercase().contains("lavapipe")
                || info.name.to_lowercase().contains("software")
                || info.name.to_lowercase().contains("microsoft basic");
            
            let marker = if is_software { "  [SOFTWARE - SKIP]" } else { "  [HARDWARE]" };
            eprintln!("  {} ({:?}) - {:?}{}", info.name, info.backend, info.device_type, marker);
            
            if !is_software {
                match info.device_type {
                    wgpu::DeviceType::DiscreteGpu => {
                        if discrete_gpu.is_none() {
                            discrete_gpu = Some(adapter);
                        }
                    }
                    wgpu::DeviceType::IntegratedGpu => {
                        if integrated_gpu.is_none() {
                            integrated_gpu = Some(adapter);
                        }
                    }
                    wgpu::DeviceType::Other => {
                        // DX12 on WSL sometimes reports as "Other"
                        if discrete_gpu.is_none() && info.backend == wgpu::Backend::Dx12 {
                            discrete_gpu = Some(adapter);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Prefer discrete GPU, fall back to integrated
        let adapter = discrete_gpu
            .or(integrated_gpu)
            .ok_or_else(|| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        eprintln!("\n✓ Selected GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HyperTensor TT Evaluator"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }
}

/// WGPU-based TT Evaluation Pipeline
#[cfg(feature = "gpu")]
pub struct WgpuTTPipeline {
    ctx: Arc<GpuContext>,
    shader: ShaderModule,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,

    // Persistent buffers
    params_buffer: Buffer,
    bond_dims_buffer: Option<Buffer>,
    core_offsets_buffer: Option<Buffer>,
    cores_buffer: Option<Buffer>,

    // Current structure
    num_sites: u32,
    physical_dim: u32,
    max_bond_dim: u32,

    // Workgroup configuration
    workgroup_size: u32,
}

#[cfg(feature = "gpu")]
impl WgpuTTPipeline {
    /// Create a new GPU pipeline
    pub fn new(ctx: Arc<GpuContext>) -> Result<Self, GpuError> {
        let shader_source = include_str!("tt_eval.wgsl");

        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TT Evaluation Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        // Create bind group layout
        let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TT Bind Group Layout"),
            entries: &[
                // 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: bond_dims (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: core_offsets (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: cores (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: indices (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: output (storage read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TT Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TT Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create params buffer (will be updated per dispatch)
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TT Params Buffer"),
            size: std::mem::size_of::<TTParamsGpu>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            ctx,
            shader,
            pipeline,
            bind_group_layout,
            params_buffer,
            bond_dims_buffer: None,
            core_offsets_buffer: None,
            cores_buffer: None,
            num_sites: 0,
            physical_dim: 2,
            max_bond_dim: 0,
            workgroup_size: 256,
        })
    }

    /// Set TT structure and upload persistent buffers
    pub fn set_structure(
        &mut self,
        num_sites: u32,
        physical_dim: u32,
        bond_dims: &[u16],
        core_offsets: &[u32],
        cores: &[f32],
    ) {
        self.num_sites = num_sites;
        self.physical_dim = physical_dim;
        self.max_bond_dim = bond_dims.iter().map(|&x| x as u32).max().unwrap_or(1);

        // Upload bond dimensions (convert to u32 for shader)
        let bond_dims_u32: Vec<u32> = bond_dims.iter().map(|&x| x as u32).collect();
        self.bond_dims_buffer = Some(
            self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bond Dims Buffer"),
                contents: bytemuck::cast_slice(&bond_dims_u32),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
        );

        // Upload core offsets
        self.core_offsets_buffer = Some(
            self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Core Offsets Buffer"),
                contents: bytemuck::cast_slice(core_offsets),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
        );

        // Upload cores (the main data)
        self.cores_buffer = Some(
            self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TT Cores Buffer"),
                contents: bytemuck::cast_slice(cores),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
        );
    }

    /// Evaluate TT at multiple query points
    ///
    /// # Arguments
    /// * `indices` - Flattened multi-indices (num_queries × num_sites)
    /// * `num_queries` - Number of query points
    ///
    /// # Returns
    /// Vector of TT values at each query point
    pub fn evaluate(&self, indices: &[u32], num_queries: usize) -> Result<Vec<f32>, GpuError> {
        if self.bond_dims_buffer.is_none() || self.cores_buffer.is_none() {
            return Err(GpuError::NotInitialized);
        }

        if indices.len() < num_queries * self.num_sites as usize {
            return Err(GpuError::InvalidParams(format!(
                "Expected {} indices, got {}",
                num_queries * self.num_sites as usize,
                indices.len()
            )));
        }

        // Update params
        let params = TTParamsGpu {
            num_sites: self.num_sites,
            physical_dim: self.physical_dim,
            max_bond_dim: self.max_bond_dim,
            num_queries: num_queries as u32,
        };
        self.ctx.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Create indices buffer
        let indices_buffer = self.ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Query Indices Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::STORAGE,
        });

        // Create output buffer
        let output_size = (num_queries * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TT Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bond_dims_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.core_offsets_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.cores_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TT Compute Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TT Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Calculate workgroups
            let workgroups = (num_queries as u32 + self.workgroup_size - 1) / self.workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.ctx.queue.submit(Some(encoder.finish()));

        // Read back results (blocking)
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::BufferMapping("Channel closed".to_string()))?
            .map_err(|e| GpuError::BufferMapping(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Get GPU info
    pub fn gpu_info(&self) -> &wgpu::AdapterInfo {
        &self.ctx.adapter_info
    }

    /// Get core buffer size in bytes
    pub fn core_buffer_bytes(&self) -> usize {
        self.cores_buffer
            .as_ref()
            .map(|b| b.size() as usize)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        // This test requires a GPU, skip if not available
        let ctx = match GpuContext::new_blocking() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping GPU test: {}", e);
                return;
            }
        };

        eprintln!("GPU: {}", ctx.adapter_info.name);
        eprintln!("Backend: {:?}", ctx.adapter_info.backend);
    }

    #[test]
    fn test_simple_tt_evaluation() {
        let ctx = match GpuContext::new_blocking() {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                eprintln!("Skipping GPU test: {}", e);
                return;
            }
        };

        let mut pipeline = WgpuTTPipeline::new(ctx).unwrap();

        // Simple 3-site TT with d=2, χ=[2, 2]
        // Core 0: shape (1, 2, 2) = 4 elements
        // Core 1: shape (2, 2, 2) = 8 elements
        // Core 2: shape (2, 2, 1) = 4 elements
        let bond_dims: Vec<u16> = vec![2, 2];
        let core_offsets: Vec<u32> = vec![0, 16, 48]; // bytes
        let cores: Vec<f32> = vec![
            // Core 0: identity-like
            1.0, 0.0, 0.0, 1.0,
            // Core 1
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            // Core 2
            1.0, 1.0, 1.0, 1.0,
        ];

        pipeline.set_structure(3, 2, &bond_dims, &core_offsets, &cores);

        // Evaluate at (0, 0, 0)
        let indices: Vec<u32> = vec![0, 0, 0];
        let results = pipeline.evaluate(&indices, 1).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].is_finite());
        eprintln!("TT((0,0,0)) = {}", results[0]);
    }

    #[test]
    fn test_batch_evaluation() {
        let ctx = match GpuContext::new_blocking() {
            Ok(ctx) => Arc::new(ctx),
            Err(e) => {
                eprintln!("Skipping GPU test: {}", e);
                return;
            }
        };

        let mut pipeline = WgpuTTPipeline::new(ctx).unwrap();

        // 2-site TT
        let bond_dims: Vec<u16> = vec![2];
        let core_offsets: Vec<u32> = vec![0, 16];
        let cores: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        pipeline.set_structure(2, 2, &bond_dims, &core_offsets, &cores);

        // Evaluate at all 4 indices
        let indices: Vec<u32> = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let results = pipeline.evaluate(&indices, 4).unwrap();

        assert_eq!(results.len(), 4);
        for (i, r) in results.iter().enumerate() {
            assert!(r.is_finite(), "Result {} is not finite", i);
            eprintln!("TT[{}] = {}", i, r);
        }
    }
}

// Tensor Colormap GPU Pipeline
// Converts raw tensor data to RGB using scientific colormaps
// Constitutional compliance: Article II (type safety), Article V (performance)

// Phase 3-5 scaffolding: Tensor visualization colormap pipeline
// This module will be used when tensor field visualization is integrated

use wgpu::util::DeviceExt;

#[allow(dead_code)]
/// Colormap variants
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColormapType {
    Viridis = 0,
    Plasma = 1,
    Turbo = 2,
    Inferno = 3,
    Magma = 4,
}

// Phase 3-5 scaffolding: Colormap type methods for tensor visualization
#[allow(dead_code)]
impl ColormapType {
    /// Get colormap name for UI display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Viridis => "Viridis",
            Self::Plasma => "Plasma",
            Self::Turbo => "Turbo",
            Self::Inferno => "Inferno",
            Self::Magma => "Magma",
        }
    }

    /// Cycle to next colormap (for keyboard shortcuts)
    pub fn next(&self) -> Self {
        match self {
            Self::Viridis => Self::Plasma,
            Self::Plasma => Self::Turbo,
            Self::Turbo => Self::Inferno,
            Self::Inferno => Self::Magma,
            Self::Magma => Self::Viridis,
        }
    }
}

/// Uniforms for colormap shader
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ColormapUniforms {
    tensor_min: f32,
    tensor_max: f32,
    colormap_id: u32,
    _padding: u32,
}

#[allow(dead_code)]
/// Tensor colormap GPU pipeline
pub struct TensorColormap {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    current_colormap: ColormapType,
}

// Phase 3-5 scaffolding: TensorColormap implementation for tensor visualization
#[allow(dead_code)]
impl TensorColormap {
    /// Create new colormap pipeline
    pub fn new(device: &wgpu::Device) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tensor Colormap Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tensor_colormap.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tensor Colormap Bind Group Layout"),
            entries: &[
                // Uniforms (tensor_min, tensor_max, colormap_id)
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
                // Input tensor texture (R32Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output RGBA texture (Rgba8Unorm)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tensor Colormap Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tensor Colormap Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "colormap_main",
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tensor Colormap Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ColormapUniforms {
                tensor_min: 0.0,
                tensor_max: 1.0,
                colormap_id: ColormapType::Viridis as u32,
                _padding: 0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            current_colormap: ColormapType::Viridis,
        }
    }

    /// Apply colormap to tensor texture
    #[allow(clippy::too_many_arguments)]  // wgpu operations require these parameters
    pub fn apply(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_texture: &wgpu::TextureView,
        output_texture: &wgpu::TextureView,
        tensor_min: f32,
        tensor_max: f32,
        width: u32,
        height: u32,
    ) {
        // Update uniforms
        let uniforms = ColormapUniforms {
            tensor_min,
            tensor_max,
            colormap_id: self.current_colormap as u32,
            _padding: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tensor Colormap Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(output_texture),
                },
            ],
        });

        // Dispatch compute shader (8x8 workgroups)
        let workgroup_size = 8;
        let workgroups_x = width.div_ceil(workgroup_size);
        let workgroups_y = height.div_ceil(workgroup_size);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tensor Colormap Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Set active colormap
    pub fn set_colormap(&mut self, colormap: ColormapType) {
        self.current_colormap = colormap;
    }

    /// Get current colormap
    pub fn colormap(&self) -> ColormapType {
        self.current_colormap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colormap_cycle() {
        let mut cm = ColormapType::Viridis;
        assert_eq!(cm, ColormapType::Viridis);
        
        cm = cm.next();
        assert_eq!(cm, ColormapType::Plasma);
        
        cm = cm.next();
        assert_eq!(cm, ColormapType::Turbo);
        
        cm = cm.next();
        assert_eq!(cm, ColormapType::Inferno);
        
        cm = cm.next();
        assert_eq!(cm, ColormapType::Magma);
        
        cm = cm.next();
        assert_eq!(cm, ColormapType::Viridis);  // Cycles back
    }

    #[test]
    fn test_colormap_names() {
        assert_eq!(ColormapType::Viridis.name(), "Viridis");
        assert_eq!(ColormapType::Plasma.name(), "Plasma");
        assert_eq!(ColormapType::Turbo.name(), "Turbo");
        assert_eq!(ColormapType::Inferno.name(), "Inferno");
        assert_eq!(ColormapType::Magma.name(), "Magma");
    }

    #[test]
    fn test_uniforms_layout() {
        let uniforms = ColormapUniforms {
            tensor_min: -1.0,
            tensor_max: 1.0,
            colormap_id: 2,
            _padding: 0,
        };
        
        // Verify size matches WGSL struct (16 bytes)
        assert_eq!(std::mem::size_of::<ColormapUniforms>(), 16);
        
        // Verify bytemuck works
        let _bytes: &[u8] = bytemuck::cast_slice(&[uniforms]);
    }
}

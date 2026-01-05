// Phase 2: Tensor Field GPU Renderer
// Instanced billboarded quads for tensor field visualization
// Constitutional compliance: Doctrine 3 (GPU compute), Doctrine 1 (procedural)

// Phase 3-5 scaffolding: Tensor field GPU rendering pipeline
// Will be used for 3D tensor field visualization with QTT data

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::tensor_field::{TensorField, ColorMode};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct VisualizationParams {
    dimensions: [u32; 3],
    color_mode: u32,
    intensity_scale: f32,
    threshold: f32,
    show_glyphs: u32,
    show_vectors: u32,
}

#[allow(dead_code)]
pub struct TensorRenderer {
    pipeline: wgpu::RenderPipeline,
    tensor_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    tensor_field: TensorField,
    instance_count: u32,
}

// Phase 3-5 scaffolding: TensorRenderer implementation for tensor visualization
#[allow(dead_code)]
impl TensorRenderer {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Self, anyhow::Error> {
        // Create tensor field (16×16×8 grid for Phase 2)
        let mut tensor_field = TensorField::new(16, 16, 8);
        tensor_field.generate_test_pattern();
        
        let instance_count = tensor_field.data.len() as u32;
        
        // Prepare GPU data
        let gpu_data = tensor_field.prepare_gpu_data();
        let tensor_data_bytes: &[u8] = bytemuck::cast_slice(&gpu_data);
        
        // Create tensor data buffer (storage buffer)
        let tensor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tensor Data Buffer"),
            contents: tensor_data_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create visualization parameters buffer
        let params = VisualizationParams {
            dimensions: [
                tensor_field.dimensions.0,
                tensor_field.dimensions.1,
                tensor_field.dimensions.2,
            ],
            color_mode: match tensor_field.vis_params.color_mode {
                ColorMode::Magnitude => 0,
                ColorMode::Trace => 1,
                ColorMode::Direction => 2,
                ColorMode::Heatmap => 3,
            },
            intensity_scale: tensor_field.vis_params.intensity_scale,
            threshold: tensor_field.vis_params.threshold,
            show_glyphs: if tensor_field.vis_params.show_glyphs { 1 } else { 0 },
            show_vectors: if tensor_field.vis_params.show_vectors { 1 } else { 0 },
        };
        
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Visualization Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create bind group layout for tensor data
        let tensor_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Tensor Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tensor Bind Group"),
            layout: &tensor_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tensor Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tensor.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tensor Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &tensor_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tensor Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Billboard, no culling
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        Ok(Self {
            pipeline,
            tensor_buffer,
            params_buffer,
            bind_group,
            tensor_field,
            instance_count,
        })
    }
    
    /// Update tensor field data (for future QTT integration)
    pub fn update_tensor_field(&mut self, queue: &wgpu::Queue, field: TensorField) {
        let gpu_data = field.prepare_gpu_data();
        let tensor_data_bytes: &[u8] = bytemuck::cast_slice(&gpu_data);
        queue.write_buffer(&self.tensor_buffer, 0, tensor_data_bytes);
        
        self.instance_count = field.data.len() as u32;
        self.tensor_field = field;
    }
    
    /// Update visualization parameters
    pub fn update_params(&mut self, queue: &wgpu::Queue) {
        let params = VisualizationParams {
            dimensions: [
                self.tensor_field.dimensions.0,
                self.tensor_field.dimensions.1,
                self.tensor_field.dimensions.2,
            ],
            color_mode: match self.tensor_field.vis_params.color_mode {
                ColorMode::Magnitude => 0,
                ColorMode::Trace => 1,
                ColorMode::Direction => 2,
                ColorMode::Heatmap => 3,
            },
            intensity_scale: self.tensor_field.vis_params.intensity_scale,
            threshold: self.tensor_field.vis_params.threshold,
            show_glyphs: if self.tensor_field.vis_params.show_glyphs { 1 } else { 0 },
            show_vectors: if self.tensor_field.vis_params.show_vectors { 1 } else { 0 },
        };
        
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }
    
    /// Render tensor field overlay
    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        camera_bind_group: &'rpass wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        
        // Draw instanced quads (6 vertices per quad)
        render_pass.draw(0..6, 0..self.instance_count);
    }
    
    /// Get field statistics for telemetry
    pub fn get_statistics(&self) -> crate::tensor_field::FieldStatistics {
        self.tensor_field.statistics()
    }
    
    /// Cycle color mode for testing
    pub fn cycle_color_mode(&mut self, queue: &wgpu::Queue) {
        use ColorMode::*;
        self.tensor_field.vis_params.color_mode = match self.tensor_field.vis_params.color_mode {
            Magnitude => Trace,
            Trace => Direction,
            Direction => Heatmap,
            Heatmap => Magnitude,
        };
        self.update_params(queue);
    }
    
    /// Adjust intensity scale
    pub fn set_intensity(&mut self, queue: &wgpu::Queue, scale: f32) {
        self.tensor_field.vis_params.intensity_scale = scale.clamp(0.1, 10.0);
        self.update_params(queue);
    }
}

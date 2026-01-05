//! Starfield Renderer
//!
//! Phase 7: Procedural starfield background
//! Renders a fullscreen starfield behind the globe for space atmosphere

use glam::Mat4;
use wgpu::util::DeviceExt;

/// Starfield uniform data
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StarfieldUniforms {
    /// Inverse view-projection matrix for ray direction
    pub inv_view_proj: [[f32; 4]; 4],
    /// Camera position
    pub camera_pos: [f32; 3],
    /// Time for twinkling effect
    pub time: f32,
}

/// Starfield renderer - draws procedural stars behind the globe
#[allow(dead_code)]  // render() disabled for globe texture debugging
pub struct StarfieldRenderer {
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Bind group
    bind_group: wgpu::BindGroup,
}

impl StarfieldRenderer {
    /// Create a new starfield renderer
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Starfield Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/starfield.wgsl").into()
            ),
        });
        
        // Create uniform buffer
        let uniforms = StarfieldUniforms {
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 3.0],
            time: 0.0,
        };
        
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Starfield Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Starfield Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Starfield Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Starfield Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Render pipeline - no depth write, renders behind everything (wgpu 0.19 API)
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Starfield Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],  // Fullscreen quad from vertex index
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,  // Fullscreen quad
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            // No depth test - stars are infinitely far
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
        }
    }
    
    /// Update uniforms with current camera state
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        time: f32,
    ) {
        let inv_view_proj = view_proj.inverse();
        
        let uniforms = StarfieldUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos,
            time,
        };
        
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }
    
    /// Render the starfield
    /// Should be called BEFORE the globe pass with depth testing disabled
    #[allow(dead_code)]  // Disabled for globe texture debugging
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..4, 0..1);  // Fullscreen quad from 4 vertices
    }
}

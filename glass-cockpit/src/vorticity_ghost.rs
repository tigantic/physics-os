/*!
 * Vorticity Ghost Renderer - Phase 8: Appendix G
 * 
 * Volumetric smoke-like overlay visualizing the curl of the tensor field.
 * Uses ray marching through a procedural smoke density field colored by
 * vorticity sign (blue = cyclonic, orange = anticyclonic).
 * 
 * Constitutional Compliance:
 * - Article V: GPU-accelerated ray marching
 * - Doctrine 3: Procedural rendering, no texture assets
 */

use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::convergence::{ConvergenceConfig, ConvergenceBridge, GpuConvergenceCell};

/// Maximum cells to sample for vorticity
const MAX_SAMPLE_CELLS: usize = 8192;

/// GPU uniforms for vorticity ghost rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VorticityUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub time: f32,
    pub globe_radius: f32,
    pub vorticity_threshold: f32,
    pub vorticity_max: f32,
    pub max_opacity: f32,
}

/// Vorticity Ghost Renderer
/// Renders volumetric smoke overlay showing curl of tensor field
pub struct VorticityGhostRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    cell_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    enabled: bool,
}

impl VorticityGhostRenderer {
    /// Create new vorticity ghost renderer
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity Ghost Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vorticity_ghost.wgsl").into()),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Uniforms"),
            size: std::mem::size_of::<VorticityUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create cell storage buffer
        let cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vorticity Cells"),
            size: (std::mem::size_of::<GpuConvergenceCell>() * MAX_SAMPLE_CELLS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vorticity Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vorticity Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vorticity Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline with alpha blending for volumetric effect
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vorticity Ghost Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
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
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // Rendered after globe, before UI
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            cell_buffer,
            bind_group,
            enabled: false,
        }
    }

    /// Enable or disable vorticity ghost rendering
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if rendering is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Update uniforms and cell data
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        camera_pos: [f32; 3],
        globe_radius: f32,
        time: f32,
        bridge: &ConvergenceBridge,
    ) {
        if !self.enabled {
            return;
        }

        // Upload uniforms
        let uniforms = VorticityUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos,
            time,
            globe_radius,
            vorticity_threshold: 0.2,
            vorticity_max: 1.0,
            max_opacity: 0.6, // Per Appendix G: max 60% opacity
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Upload cell data for vorticity sampling
        let field = bridge.field();
        let gpu_cells: Vec<GpuConvergenceCell> = field
            .visible_cells()
            .take(MAX_SAMPLE_CELLS)
            .map(GpuConvergenceCell::from)
            .collect();

        if !gpu_cells.is_empty() {
            queue.write_buffer(&self.cell_buffer, 0, bytemuck::cast_slice(&gpu_cells));
        }
    }

    /// Render vorticity ghost overlay
    /// Should be called after globe but before UI elements
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if !self.enabled {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Full-screen triangle
    }
}

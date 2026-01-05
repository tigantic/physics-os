/*!
 * Vorticity Ghost Renderer - Phase 8: Appendix G Implementation
 *
 * Volumetric ray-marching renderer for atmospheric vorticity.
 * Implements the "Vorticity Ghost" layer.
 *
 * Constitutional Compliance:
 * - Doctrine 1: Non-blocking GPU dispatch
 * - Appendix G: Volumetric smoke-like overlay
 */
#![allow(dead_code)] // Slice plane and render methods ready for integration

use wgpu::util::DeviceExt;

/// Slice mode for volumetric slicing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceMode {
    Off = 0,
    Below = 1,
    Above = 2,
    Thin = 3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VorticityUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    globe_radius: f32,
    vorticity_threshold: f32,
    vorticity_max: f32,
    max_opacity: f32,
    // Phase 8: Volumetric slicing (The Big One Phase 4)
    slice_plane: [f32; 4],  // xyz = normal, w = distance
    slice_mode: u32,         // 0 = off, 1 = below, 2 = above, 3 = thin
    slice_thickness: f32,    // For thin slice mode
    _padding: [f32; 2],
}

/// Slice plane state for interactive control
#[derive(Debug, Clone)]
pub struct SlicePlane {
    pub normal: glam::Vec3,
    pub distance: f32,
    pub mode: SliceMode,
    pub thickness: f32,
}

impl Default for SlicePlane {
    fn default() -> Self {
        Self {
            normal: glam::Vec3::Y, // Horizontal slice by default
            distance: 0.0,
            mode: SliceMode::Off,
            thickness: 0.1,
        }
    }
}

impl SlicePlane {
    /// Move the slice plane along its normal
    pub fn move_plane(&mut self, delta: f32) {
        self.distance += delta;
    }
    
    /// Rotate the slice plane (around X axis for now)
    pub fn rotate_x(&mut self, radians: f32) {
        let rotation = glam::Mat3::from_rotation_x(radians);
        self.normal = rotation * self.normal;
    }
    
    /// Rotate the slice plane (around Z axis)
    pub fn rotate_z(&mut self, radians: f32) {
        let rotation = glam::Mat3::from_rotation_z(radians);
        self.normal = rotation * self.normal;
    }
    
    /// Cycle through slice modes
    pub fn cycle_mode(&mut self) {
        self.mode = match self.mode {
            SliceMode::Off => SliceMode::Below,
            SliceMode::Below => SliceMode::Above,
            SliceMode::Above => SliceMode::Thin,
            SliceMode::Thin => SliceMode::Off,
        };
    }
}

pub struct VorticityRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    start_time: std::time::Instant,
    /// Slice plane state
    pub slice_plane: SlicePlane,
}

impl VorticityRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        convergence_buffer: &wgpu::Buffer, // The storage buffer containing grid data
    ) -> Self {
        let start_time = std::time::Instant::now();

        // 1. Create Uniform Buffer
        let uniforms = VorticityUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            time: 0.0,
            globe_radius: 100.0, // Scaled units, adjust to match GlobeRenderer
            vorticity_threshold: 0.2,
            vorticity_max: 5.0,
            max_opacity: 0.2, // Reduced - less distracting
            slice_plane: [0.0, 1.0, 0.0, 0.0], // Default: horizontal plane at origin
            slice_mode: 0,                      // Off by default
            slice_thickness: 0.1,
            _padding: [0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vorticity Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Create Bind Group Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vorticity Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Storage Buffer (Convergence Cells)
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

        // 3. Create Bind Group
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
                    resource: convergence_buffer.as_entire_binding(),
                },
            ],
        });

        // 4. Load Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vorticity Ghost Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vorticity_ghost.wgsl").into()),
        });

        // 5. Create Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vorticity Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vorticity Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // Procedural full-screen quad
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Additive Blending for "Ghost" look
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // Ghost draws over everything
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            start_time,
            slice_plane: SlicePlane::default(),
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, view_proj: glam::Mat4, camera_pos: glam::Vec3) {
        let time = self.start_time.elapsed().as_secs_f32();

        let uniforms = VorticityUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            time,
            globe_radius: 100.0, // Match your main globe radius
            vorticity_threshold: 0.15, // Sensitivity
            vorticity_max: 2.0,
            max_opacity: 0.2, // Reduced - less distracting
            // Phase 8: Slice plane uniforms
            slice_plane: [
                self.slice_plane.normal.x,
                self.slice_plane.normal.y,
                self.slice_plane.normal.z,
                self.slice_plane.distance,
            ],
            slice_mode: self.slice_plane.mode as u32,
            slice_thickness: self.slice_plane.thickness,
            _padding: [0.0, 0.0],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Full screen triangle
    }
}

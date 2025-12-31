//! Convergence Heatmap Renderer
//!
//! Phase 6: GPU rendering of probabilistic convergence zones
//!
//! Features:
//! - Instanced quad rendering for each convergence cell
//! - Spectral/plasma colormap with vorticity influence
//! - Alpha compositing over globe and vector layers
//! - LOD integration for performance maintenance
//!
//! Constitutional: Maintains 60 FPS mandate through budget constraints
#![allow(dead_code)] // Update and render methods ready for integration

use crate::convergence::{ConvergenceBridge, ConvergenceConfig, ConvergenceUniforms, GpuConvergenceCell};
use crate::lod::{LodCuller, LodLevel};
use glam::Mat4;
use wgpu::util::DeviceExt;

/// Maximum cells to render per frame (budget constraint)
const MAX_RENDER_CELLS: usize = 16384;

/// Convergence heatmap renderer
pub struct ConvergenceRenderer {
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Cell storage buffer
    cell_buffer: wgpu::Buffer,
    /// Grid dimensions uniform
    _grid_dims_buffer: wgpu::Buffer,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Current cell count for instanced draw
    cell_count: u32,
    /// Convergence data bridge
    bridge: ConvergenceBridge,
    /// LOD culler reference for budget management
    /// Phase 6: Using instance budget from LOD infrastructure
    current_lod: LodLevel,
}

impl ConvergenceRenderer {
    /// Create a new convergence renderer
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        convergence_config: ConvergenceConfig,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Convergence Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/convergence.wgsl").into()),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Convergence Uniforms"),
            size: std::mem::size_of::<ConvergenceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create cell storage buffer
        let cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Convergence Cells"),
            size: (MAX_RENDER_CELLS * std::mem::size_of::<GpuConvergenceCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Grid dimensions buffer
        let grid_dims = [convergence_config.resolution.0, convergence_config.resolution.1];
        let grid_dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Dimensions"),
            contents: bytemuck::cast_slice(&grid_dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Convergence Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    // Phase 8: Fragment shader now reads cells for hover feedback
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
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
            label: Some("Convergence Bind Group"),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Convergence Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline with alpha blending
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Convergence Pipeline"),
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
                    format: config.format,
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
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Render both sides for visibility
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth (overlay)
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            cell_buffer,
            _grid_dims_buffer: grid_dims_buffer,
            bind_group,
            cell_count: 0,
            bridge: ConvergenceBridge::new(convergence_config),
            current_lod: LodLevel::High,
        }
    }

    /// Update convergence data and upload to GPU
    /// Phase 8: Added hover_state parameter for Appendix D feedback
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        globe_radius: f32,
        camera_pos: [f32; 3],
        time: f32,
        culler: &mut LodCuller,
    ) {
        self.update_with_hover(queue, view_proj, globe_radius, camera_pos, time, culler, None);
    }

    /// Update with optional hover state for Appendix D visual feedback
    #[allow(clippy::too_many_arguments)]  // Render params require all these
    pub fn update_with_hover(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        globe_radius: f32,
        camera_pos: [f32; 3],
        time: f32,
        culler: &mut LodCuller,
        hover_state: Option<(f32, f32, f32)>,  // (lon, lat, intensity)
    ) {
        // Update convergence field from bridge
        self.bridge.update(time);
        let field = self.bridge.field();

        // Determine LOD based on camera distance
        // Phase 6: Use culler for LOD decisions
        let cam_distance = culler.camera_pos.length();
        self.current_lod = culler.config.get_level(cam_distance);

        // DEBUG: Print LOD info
        if cam_distance > 0.0 {
            // println!("DEBUG: cam_distance={:.2}, LOD={:?}", cam_distance, self.current_lod);
        }

        // Calculate cell budget based on LOD
        // NOTE: Camera is viewing globe from distance, don't cull the heatmap itself
        // The LOD should be based on globe detail, not heatmap visibility
        let max_cells = MAX_RENDER_CELLS; // Always allow full budget for heatmap

        // Allocate from budget
        let budget_cells = culler.budget.allocate_heatmap_cells(max_cells as u32) as usize;

        // Collect visible cells (above threshold)
        let gpu_cells: Vec<GpuConvergenceCell> = field
            .visible_cells()
            .take(budget_cells)
            .map(GpuConvergenceCell::from)
            .collect();

        self.cell_count = gpu_cells.len() as u32;

        // Upload cells to GPU
        if !gpu_cells.is_empty() {
            queue.write_buffer(&self.cell_buffer, 0, bytemuck::cast_slice(&gpu_cells));
        }

        // Upload uniforms with hover state
        let (hover_pos, hover_intensity) = match hover_state {
            Some((lon, lat, intensity)) => ([lon, lat], intensity),
            None => ([0.0, 0.0], 0.0),
        };
        
        let uniforms = ConvergenceUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
            globe_radius,
            time,
            visibility_threshold: field.config.visibility_threshold,
            high_intensity_threshold: field.config.high_intensity_threshold,
            pulse_frequency: field.config.pulse_frequency,
            max_intensity: field.max_intensity,
            hover_pos,
            _padding_a: 0.0,
            hover_intensity,
            ghost_mode: 0.0,  // Normal mode by default
            _pad1: 0.0,
            ghost_selected_pos: [0.0, 0.0],
            _pad2: [0.0; 2],
            _padding: [0.0; 4],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Update with ghost mode state for Appendix D causal trace
    /// Phase 8: Enables faded rendering with upstream node highlighting
    #[allow(dead_code)]  // Phase 8: Causal trace mode
    #[allow(clippy::too_many_arguments)]  // Render params require all these
    pub fn update_with_ghost_mode(
        &mut self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        globe_radius: f32,
        camera_pos: [f32; 3],
        time: f32,
        culler: &mut LodCuller,
        ghost_state: Option<(f32, f32)>,  // (selected_lon, selected_lat) if in ghost mode
    ) {
        // Update convergence field from bridge
        self.bridge.update(time);
        let field = self.bridge.field();

        // LOD and culling (same as normal update)
        let cam_distance = culler.camera_pos.length();
        self.current_lod = culler.config.get_level(cam_distance);
        
        let max_cells = MAX_RENDER_CELLS;
        let budget_cells = culler.budget.allocate_heatmap_cells(max_cells as u32) as usize;

        let gpu_cells: Vec<GpuConvergenceCell> = field
            .visible_cells()
            .take(budget_cells)
            .map(GpuConvergenceCell::from)
            .collect();

        self.cell_count = gpu_cells.len() as u32;

        if !gpu_cells.is_empty() {
            queue.write_buffer(&self.cell_buffer, 0, bytemuck::cast_slice(&gpu_cells));
        }

        // Upload uniforms with ghost mode
        let (ghost_mode, ghost_selected_pos) = match ghost_state {
            Some((lon, lat)) => (1.0, [lon, lat]),
            None => (0.0, [0.0, 0.0]),
        };
        
        let uniforms = ConvergenceUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
            globe_radius,
            time,
            visibility_threshold: field.config.visibility_threshold,
            high_intensity_threshold: field.config.high_intensity_threshold,
            pulse_frequency: field.config.pulse_frequency,
            max_intensity: field.max_intensity,
            hover_pos: [0.0, 0.0],  // No hover during ghost mode
            _padding_a: 0.0,
            hover_intensity: 0.0,
            ghost_mode,
            _pad1: 0.0,
            ghost_selected_pos,
            _pad2: [0.0; 2],
            _padding: [0.0; 4],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Render convergence heatmap
    /// Call after globe, before/after vectors depending on desired layering
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.cell_count == 0 || self.current_lod == LodLevel::Culled {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        
        // 6 vertices per quad (2 triangles), instanced per cell
        render_pass.draw(0..6, 0..self.cell_count);
    }

    /// Get current cell count for telemetry
    pub fn cell_count(&self) -> u32 {
        self.cell_count
    }

    /// Get high intensity zone count for telemetry
    pub fn high_intensity_count(&self) -> u32 {
        self.bridge.field().high_intensity_count
    }

    /// Get current LOD level for telemetry
    /// Phase 8: Detailed telemetry display
    #[allow(dead_code)] // Phase 8: Telemetry rails
    pub fn current_lod(&self) -> LodLevel {
        self.current_lod
    }

    /// Get max intensity for telemetry
    /// Phase 8: Detailed telemetry display
    #[allow(dead_code)] // Phase 8: Telemetry rails
    pub fn max_intensity(&self) -> f32 {
        self.bridge.field().max_intensity
    }

    /// Sample convergence intensity at geographic coordinates
    /// Phase 8: Probe hover detection
    /// Returns (intensity, vorticity, confidence) for the sampled cell
    pub fn sample_at(&self, lon: f32, lat: f32) -> (f32, f32, f32) {
        let cell = self.bridge.field().sample(lon, lat);
        (cell.intensity, cell.vorticity, cell.confidence)
    }

    /// Get reference to cell storage buffer for vorticity ghost renderer
    /// Phase 8 Appendix G: Allows VorticityRenderer to bind the same buffer
    pub fn cell_buffer(&self) -> &wgpu::Buffer {
        &self.cell_buffer
    }
}

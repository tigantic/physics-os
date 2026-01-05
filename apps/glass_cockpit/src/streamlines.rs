// Phase 5: Streamline Renderer
// Precomputed streamline curves with magnitude-based thickness
// Constitutional compliance: Doctrine 3 (GPU-rendered), Doctrine 1 (procedural)
#![allow(dead_code)] // Render method ready for integration

use glam::Vec2;
use crate::vector_field::VectorField;

/// Streamline vertex for GPU rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StreamlineVertex {
    /// Position (lon, lat, altitude)
    pub position: [f32; 3],
    /// Tangent direction (normalized)
    pub tangent: [f32; 3],
    /// Properties: (speed, vorticity, arc_length, side)
    pub properties: [f32; 4],
}

/// Streamline configuration
#[derive(Debug, Clone, Copy)]
pub struct StreamlineConfig {
    /// Number of streamlines to generate
    pub count: u32,
    /// Maximum length per streamline (meters)
    pub max_length: f32,
    /// Integration step size (meters)
    pub step_size: f32,
    /// Maximum integration steps
    pub max_steps: u32,
    /// Base line thickness (pixels)
    pub base_thickness: f32,
    /// Thickness multiplier based on speed
    pub speed_thickness_factor: f32,
    /// Minimum separation between streamlines (degrees)
    pub min_separation: f32,
    /// Seed spacing mode
    pub spacing: StreamlineSpacing,
}

/// Streamline seed placement strategy
#[allow(dead_code)] // Phase 5 scaffolding - Random/VorticityGuided for advanced seeding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamlineSpacing {
    /// Regular grid with jitter
    Grid,
    /// Random placement
    Random,
    /// Place along high-vorticity regions
    VorticityGuided,
}

impl Default for StreamlineConfig {
    fn default() -> Self {
        Self {
            count: 500,
            max_length: 2_000_000.0, // 2000 km
            step_size: 10_000.0,      // 10 km steps
            max_steps: 500,
            base_thickness: 1.5,
            speed_thickness_factor: 2.0,
            min_separation: 0.5,
            spacing: StreamlineSpacing::Grid,
        }
    }
}

/// A single streamline (CPU representation)
#[derive(Debug, Clone)]
pub struct Streamline {
    /// Vertices along the streamline
    pub vertices: Vec<StreamlineVertex>,
    /// Total arc length (meters)
    pub arc_length: f32,
    /// Average speed along line
    pub avg_speed: f32,
    /// Maximum vorticity magnitude
    pub max_vorticity: f32,
}

impl Streamline {
    /// Create empty streamline
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            arc_length: 0.0,
            avg_speed: 0.0,
            max_vorticity: 0.0,
        }
    }
    
    /// Check if streamline has enough points
    pub fn is_valid(&self) -> bool {
        self.vertices.len() >= 3
    }
}

impl Default for Streamline {
    fn default() -> Self {
        Self::new()
    }
}

/// Streamline generator (CPU-side)
pub struct StreamlineGenerator {
    config: StreamlineConfig,
    /// Density field to prevent overcrowding
    density_grid: Vec<bool>,
    density_width: u32,
    density_height: u32,
}

impl StreamlineGenerator {
    /// Create new generator
    pub fn new(config: StreamlineConfig) -> Self {
        Self {
            config,
            density_grid: Vec::new(),
            density_width: 0,
            density_height: 0,
        }
    }
    
    /// Generate streamlines from vector field
    pub fn generate(&mut self, field: &VectorField) -> Vec<Streamline> {
        // Initialize density grid
        self.init_density_grid(field);
        
        let seeds = self.generate_seeds(field);
        let mut streamlines = Vec::with_capacity(seeds.len());
        
        for seed in seeds {
            if !self.check_density(seed.x, seed.y) {
                continue;
            }
            
            // Integrate forward
            let forward = self.integrate(field, seed, 1.0);
            
            // Integrate backward
            let mut backward = self.integrate(field, seed, -1.0);
            backward.vertices.reverse();
            
            // Merge (remove duplicate seed point)
            let backward_arc = backward.arc_length;
            let mut merged = backward;
            if !merged.vertices.is_empty() {
                merged.vertices.pop();
            }
            merged.vertices.extend(forward.vertices);
            merged.arc_length = forward.arc_length + backward_arc;
            
            if merged.is_valid() {
                // Mark density
                for v in &merged.vertices {
                    self.mark_density(v.position[0], v.position[1]);
                }
                streamlines.push(merged);
            }
        }
        
        streamlines
    }
    
    /// Initialize density grid
    fn init_density_grid(&mut self, field: &VectorField) {
        let lon_range = field.config.lon_max - field.config.lon_min;
        let lat_range = field.config.lat_max - field.config.lat_min;
        
        self.density_width = (lon_range / self.config.min_separation).ceil() as u32;
        self.density_height = (lat_range / self.config.min_separation).ceil() as u32;
        
        let size = (self.density_width * self.density_height) as usize;
        self.density_grid = vec![false; size];
    }
    
    /// Check if position is valid in density grid
    fn check_density(&self, lon: f32, lat: f32) -> bool {
        if self.density_grid.is_empty() {
            return true;
        }
        let x = ((lon - self.config.min_separation) / self.config.min_separation) as i32;
        let y = ((lat - self.config.min_separation) / self.config.min_separation) as i32;
        if x < 0 || y < 0 || x >= self.density_width as i32 || y >= self.density_height as i32 {
            return true; // Out of bounds = allowed
        }
        let idx = (y as u32 * self.density_width + x as u32) as usize;
        !self.density_grid[idx]
    }
    
    /// Mark position in density grid
    fn mark_density(&mut self, lon: f32, lat: f32) {
        if self.density_grid.is_empty() {
            return;
        }
        let x = ((lon - self.config.min_separation) / self.config.min_separation) as i32;
        let y = ((lat - self.config.min_separation) / self.config.min_separation) as i32;
        if x >= 0 && y >= 0 && x < self.density_width as i32 && y < self.density_height as i32 {
            let idx = (y as u32 * self.density_width + x as u32) as usize;
            self.density_grid[idx] = true;
        }
    }
    
    /// Generate seed points
    fn generate_seeds(&self, field: &VectorField) -> Vec<Vec2> {
        let mut seeds = Vec::new();
        
        match self.config.spacing {
            StreamlineSpacing::Grid => {
                let sqrt_count = (self.config.count as f32).sqrt().ceil() as u32;
                let lon_step = (field.config.lon_max - field.config.lon_min) / sqrt_count as f32;
                let lat_step = (field.config.lat_max - field.config.lat_min) / sqrt_count as f32;
                
                for i in 0..sqrt_count {
                    for j in 0..sqrt_count {
                        let lon = field.config.lon_min + (i as f32 + 0.5) * lon_step;
                        let lat = field.config.lat_min + (j as f32 + 0.5) * lat_step;
                        seeds.push(Vec2::new(lon, lat));
                    }
                }
            }
            StreamlineSpacing::Random => {
                let mut seed: u32 = 12345;
                for _ in 0..self.config.count {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let fx = (seed & 0xFFFF) as f32 / 65535.0;
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let fy = (seed & 0xFFFF) as f32 / 65535.0;
                    
                    let lon = field.config.lon_min + fx * (field.config.lon_max - field.config.lon_min);
                    let lat = field.config.lat_min + fy * (field.config.lat_max - field.config.lat_min);
                    seeds.push(Vec2::new(lon, lat));
                }
            }
            StreamlineSpacing::VorticityGuided => {
                // Prefer high-vorticity regions
                // Position derived from cell index in flattened grid
                let width = field.config.grid_width;
                for (idx, cell) in field.data.iter().enumerate() {
                    if cell.vorticity.abs() > 0.0001 {
                        let ix = idx as u32 % width;
                        let iy = idx as u32 / width;
                        let lon = field.config.lon_min + (ix as f32 + 0.5) * (field.config.lon_max - field.config.lon_min) / width as f32;
                        let lat = field.config.lat_min + (iy as f32 + 0.5) * (field.config.lat_max - field.config.lat_min) / field.config.grid_height as f32;
                        seeds.push(Vec2::new(lon, lat));
                    }
                }
                // Fallback to grid if not enough seeds
                if seeds.len() < self.config.count as usize / 2 {
                    return self.generate_seeds_grid(field);
                }
            }
        }
        
        seeds
    }
    
    fn generate_seeds_grid(&self, field: &VectorField) -> Vec<Vec2> {
        let mut seeds = Vec::new();
        let sqrt_count = (self.config.count as f32).sqrt().ceil() as u32;
        let lon_step = (field.config.lon_max - field.config.lon_min) / sqrt_count as f32;
        let lat_step = (field.config.lat_max - field.config.lat_min) / sqrt_count as f32;
        
        for i in 0..sqrt_count {
            for j in 0..sqrt_count {
                let lon = field.config.lon_min + (i as f32 + 0.5) * lon_step;
                let lat = field.config.lat_min + (j as f32 + 0.5) * lat_step;
                seeds.push(Vec2::new(lon, lat));
            }
        }
        seeds
    }
    
    /// Integrate streamline from seed
    fn integrate(&self, field: &VectorField, seed: Vec2, direction: f32) -> Streamline {
        let mut streamline = Streamline::new();
        let mut pos = seed;
        let mut arc_length = 0.0_f32;
        let mut total_speed = 0.0_f32;
        let mut max_vort = 0.0_f32;
        let mut _prev_tangent = Vec2::ZERO;
        
        for step in 0..self.config.max_steps {
            // Sample velocity
            let cell = field.sample(pos.x, pos.y);
            let vel = Vec2::new(cell.u, cell.v);
            let speed = vel.length();
            
            if speed < 0.01 {
                break; // Stagnation point
            }
            
            let tangent = vel.normalize() * direction;
            
            // RK4 integration
            let k1 = self.velocity_at(field, pos);
            let k2 = self.velocity_at(field, pos + k1 * 0.5);
            let k3 = self.velocity_at(field, pos + k2 * 0.5);
            let k4 = self.velocity_at(field, pos + k3);
            
            let step_meters = self.config.step_size;
            let meters_per_degree = 111000.0; // Approximate
            let step_degrees = step_meters / meters_per_degree;
            
            let delta = (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0 * step_degrees * direction;
            
            // Add vertex
            let side = if step % 2 == 0 { 1.0 } else { -1.0 };
            streamline.vertices.push(StreamlineVertex {
                position: [pos.x, pos.y, 0.0],
                tangent: [tangent.x, tangent.y, 0.0],
                properties: [speed, cell.vorticity, arc_length, side],
            });
            
            // Update position
            pos += delta;
            arc_length += step_meters;
            total_speed += speed;
            max_vort = max_vort.max(cell.vorticity.abs());
            _prev_tangent = tangent;
            
            // Check bounds
            if pos.x < field.config.lon_min || pos.x > field.config.lon_max
                || pos.y < field.config.lat_min || pos.y > field.config.lat_max
            {
                break;
            }
            
            // Check max length
            if arc_length >= self.config.max_length {
                break;
            }
        }
        
        streamline.arc_length = arc_length;
        streamline.avg_speed = if !streamline.vertices.is_empty() {
            total_speed / streamline.vertices.len() as f32
        } else {
            0.0
        };
        streamline.max_vorticity = max_vort;
        
        streamline
    }
    
    /// Get normalized velocity at position
    fn velocity_at(&self, field: &VectorField, pos: Vec2) -> Vec2 {
        let cell = field.sample(pos.x, pos.y);
        let vel = Vec2::new(cell.u, cell.v);
        let len = vel.length();
        if len > 0.01 { vel / len } else { Vec2::ZERO }
    }
}

/// GPU streamline renderer
pub struct StreamlineRenderer {
    /// Vertex buffer
    vertex_buffer: wgpu::Buffer,
    /// Index buffer
    index_buffer: wgpu::Buffer,
    /// Uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Vertex count
    vertex_count: u32,
    /// Index count
    index_count: u32,
    /// Configuration
    config: StreamlineConfig,
}

/// Streamline uniforms
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StreamlineUniforms {
    /// View bounds: (lon_min, lon_max, lat_min, lat_max)
    pub bounds: [f32; 4],
    /// Stats: (max_speed, max_vorticity, base_thickness, speed_factor)
    pub stats: [f32; 4],
    /// Time: (current_time, animation_speed, _, _)
    pub time: [f32; 4],
}

impl StreamlineRenderer {
    /// Maximum vertices
    pub const MAX_VERTICES: u32 = 500_000;
    
    /// Create new renderer
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        config: StreamlineConfig,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let vertex_size = std::mem::size_of::<StreamlineVertex>() as u64;
        
        // Vertex buffer
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Streamline Vertex Buffer"),
            size: vertex_size * Self::MAX_VERTICES as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Index buffer (for triangle strips)
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Streamline Index Buffer"),
            size: 4 * Self::MAX_VERTICES as u64 * 2, // Worst case
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Streamline Uniforms"),
            size: std::mem::size_of::<StreamlineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Streamline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/streamlines.wgsl").into()),
        });
        
        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Streamline Bind Group Layout"),
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
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Streamline Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Streamline Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_streamline",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: vertex_size,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,  // position
                        1 => Float32x3,  // tangent
                        2 => Float32x4,  // properties
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_streamline",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Streamline Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        Self {
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            pipeline,
            bind_group,
            vertex_count: 0,
            index_count: 0,
            config,
        }
    }
    
    /// Get current vertex count
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    
    /// Get current index count
    pub fn index_count(&self) -> u32 {
        self.index_count
    }
    
    /// Upload streamlines to GPU
    pub fn upload(&mut self, queue: &wgpu::Queue, streamlines: &[Streamline]) {
        // Collect all vertices with ribbon expansion
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        for streamline in streamlines {
            if streamline.vertices.len() < 2 {
                continue;
            }
            
            let base_idx = vertices.len() as u32;
            
            // Expand line to ribbon (2 vertices per point)
            for (i, v) in streamline.vertices.iter().enumerate() {
                // Left vertex
                let mut left = *v;
                left.properties[3] = -1.0;
                vertices.push(left);
                
                // Right vertex
                let mut right = *v;
                right.properties[3] = 1.0;
                vertices.push(right);
                
                // Indices for triangle strip
                if i > 0 {
                    let prev_left = base_idx + (i as u32 - 1) * 2;
                    let prev_right = prev_left + 1;
                    let curr_left = base_idx + i as u32 * 2;
                    let curr_right = curr_left + 1;
                    
                    // Two triangles per segment
                    indices.push(prev_left);
                    indices.push(prev_right);
                    indices.push(curr_left);
                    indices.push(curr_right);
                    indices.push(0xFFFFFFFF); // Primitive restart
                }
            }
        }
        
        self.vertex_count = vertices.len() as u32;
        self.index_count = indices.len() as u32;
        
        if self.vertex_count > 0 {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
        if self.index_count > 0 {
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        }
    }
    
    /// Update uniforms
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        field: &VectorField,
        time: f32,
    ) {
        let uniforms = StreamlineUniforms {
            bounds: [
                field.config.lon_min,
                field.config.lon_max,
                field.config.lat_min,
                field.config.lat_max,
            ],
            stats: [
                field.stats.max_speed,
                field.stats.max_vorticity,
                self.config.base_thickness,
                self.config.speed_thickness_factor,
            ],
            time: [time, 1.0, 0.0, 0.0],
        };
        
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }
    
    /// Render streamlines
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, camera_bind_group: &'a wgpu::BindGroup) {
        if self.vertex_count == 0 {
            return;
        }
        
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_field::VectorFieldConfig;
    
    #[test]
    fn test_streamline_creation() {
        let sl = Streamline::new();
        assert!(!sl.is_valid());
    }
    
    #[test]
    fn test_config_default() {
        let config = StreamlineConfig::default();
        assert_eq!(config.count, 500);
        assert!(config.max_length > 0.0);
    }
    
    #[test]
    fn test_generator_seeds() {
        let config = StreamlineConfig {
            count: 16,
            spacing: StreamlineSpacing::Grid,
            ..Default::default()
        };
        let generator = StreamlineGenerator::new(config);
        
        let field_config = VectorFieldConfig::default();
        let field = VectorField::new(field_config);
        
        let seeds = generator.generate_seeds(&field);
        assert_eq!(seeds.len(), 16); // 4x4 grid
    }
}

// Phase 5: GPU-Instanced Particle System
// Flow particles advected along vector field for visualization
// Constitutional compliance: Doctrine 3 (GPU compute), Doctrine 1 (procedural)
#![allow(dead_code)] // Render method ready for integration

use wgpu::util::DeviceExt;
use crate::vector_field::{VectorField, VectorFieldConfig};

/// Maximum number of particles (GPU buffer size)
pub const MAX_PARTICLES: u32 = 100_000;

/// Particle state in GPU buffer
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    /// Position: (lon, lat, altitude, age)
    pub position: [f32; 4],
    /// Velocity: (u, v, w, speed)
    pub velocity: [f32; 4],
    /// Visual properties: (vorticity, lifetime, size, alpha)
    pub properties: [f32; 4],
}

#[allow(dead_code)] // Phase 5 scaffolding - accessors for particle properties in shader
impl Particle {
    /// Create new particle at position
    pub fn new(lon: f32, lat: f32, lifetime: f32) -> Self {
        Self {
            position: [lon, lat, 0.0, 0.0], // age starts at 0
            velocity: [0.0, 0.0, 0.0, 0.0],
            properties: [0.0, lifetime, 1.0, 1.0],
        }
    }
    
    /// Get longitude
    pub fn lon(&self) -> f32 { self.position[0] }
    
    /// Get latitude
    pub fn lat(&self) -> f32 { self.position[1] }
    
    /// Get age (seconds)
    pub fn age(&self) -> f32 { self.position[3] }
    
    /// Get lifetime (seconds)
    pub fn lifetime(&self) -> f32 { self.properties[1] }
    
    /// Get normalized age (0-1)
    pub fn age_normalized(&self) -> f32 {
        let lifetime = self.lifetime();
        if lifetime > 0.0 { self.age() / lifetime } else { 1.0 }
    }
    
    /// Check if particle is alive
    pub fn is_alive(&self) -> bool {
        self.age() < self.lifetime()
    }
}

/// Particle system configuration
#[allow(dead_code)] // Phase 5 scaffolding - dt/speed_multiplier for GPU compute tuning
#[derive(Debug, Clone, Copy)]
pub struct ParticleConfig {
    /// Particle spawn rate (particles per second)
    pub spawn_rate: f32,
    /// Particle base lifetime (seconds)
    pub lifetime: f32,
    /// Lifetime variance (seconds)
    pub lifetime_variance: f32,
    /// Particle base size (pixels)
    pub base_size: f32,
    /// Size multiplier based on speed
    pub speed_size_factor: f32,
    /// Integration time step (seconds)
    pub dt: f32,
    /// Speed multiplier for advection
    pub speed_multiplier: f32,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            spawn_rate: 5000.0,    // 5000 particles/second
            lifetime: 8.0,          // 8 second lifetime
            lifetime_variance: 2.0, // ±2 seconds
            base_size: 2.0,         // 2 pixel base
            speed_size_factor: 0.5, // Size increases with speed
            dt: 1.0 / 165.0,        // 165Hz Sovereign integration
            speed_multiplier: 1.0,  // 1:1 advection speed
        }
    }
}

/// GPU particle system uniforms
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleUniforms {
    /// Time: (current_time, dt, spawn_rate, seed)
    pub time: [f32; 4],
    /// Config: (lifetime, lifetime_var, base_size, speed_size_factor)
    pub config: [f32; 4],
    /// View bounds: (lon_min, lon_max, lat_min, lat_max)
    pub bounds: [f32; 4],
    /// Stats: (max_speed, max_vorticity, particle_count, _)
    pub stats: [f32; 4],
}

/// Particle vertex for rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleVertex {
    /// Screen position (x, y)
    pub position: [f32; 2],
    /// Texture coordinates
    pub uv: [f32; 2],
}

impl ParticleVertex {
    const QUAD: [Self; 6] = [
        Self { position: [-0.5, -0.5], uv: [0.0, 0.0] },
        Self { position: [ 0.5, -0.5], uv: [1.0, 0.0] },
        Self { position: [ 0.5,  0.5], uv: [1.0, 1.0] },
        Self { position: [-0.5, -0.5], uv: [0.0, 0.0] },
        Self { position: [ 0.5,  0.5], uv: [1.0, 1.0] },
        Self { position: [-0.5,  0.5], uv: [0.0, 1.0] },
    ];
}

/// GPU-based particle system
#[allow(dead_code)] // Phase 5 scaffolding - vector_texture_view retained for future GPU texture binding
pub struct ParticleSystem {
    /// Particle buffer (double-buffered for compute)
    particle_buffers: [wgpu::Buffer; 2],
    /// Current read buffer index
    read_buffer: usize,
    /// Vertex buffer for quad rendering
    vertex_buffer: wgpu::Buffer,
    /// Uniform buffer
    uniform_buffer: wgpu::Buffer,
    /// Vector field texture (for GPU sampling)
    vector_texture: wgpu::Texture,
    vector_texture_view: wgpu::TextureView,
    /// Compute pipeline for advection
    advect_pipeline: wgpu::ComputePipeline,
    advect_bind_groups: [wgpu::BindGroup; 2],
    /// Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    /// Particle data bind group (group 1 in shader)
    render_bind_group: wgpu::BindGroup,
    /// Configuration
    config: ParticleConfig,
    /// Current particle count
    particle_count: u32,
    /// Spawn accumulator
    spawn_accumulator: f32,
    /// Current time
    current_time: f32,
    /// RNG seed
    seed: u32,
}

impl ParticleSystem {
    /// Create new particle system
    /// Phase 8: Now accepts camera_bind_group_layout for 3D globe projection
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        field_config: &VectorFieldConfig,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Create particle buffers (double-buffered)
        let particle_size = std::mem::size_of::<Particle>() as u64;
        let buffer_size = particle_size * MAX_PARTICLES as u64;
        
        let particle_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Buffer 0"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Buffer 1"),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];
        
        // Vertex buffer for instanced quads
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Vertex Buffer"),
            contents: bytemuck::cast_slice(&ParticleVertex::QUAD),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Uniforms"),
            size: std::mem::size_of::<ParticleUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Vector field texture (RGBA32Float: u, v, w, vorticity)
        let vector_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vector Field Texture"),
            size: wgpu::Extent3d {
                width: field_config.grid_width,
                height: field_config.grid_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let vector_texture_view = vector_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particles.wgsl").into()),
        });
        
        // Compute pipeline bind group layout
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Compute Bind Group Layout"),
            entries: &[
                // Input particles
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output particles
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Vector field texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        
        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let advect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Particle Advect Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_advect",
        });
        
        // Create compute bind groups (double-buffered)
        let advect_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Advect Bind Group 0"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particle_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: particle_buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&vector_texture_view) },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Advect Bind Group 1"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: particle_buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: particle_buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&vector_texture_view) },
                ],
            }),
        ];
        
        // Render pipeline
        // Group 0: Camera uniforms (shared with globe pipeline)
        // Group 1: Particle data + uniforms
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Render Bind Group Layout"),
            entries: &[
                // Particles
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        
        // Phase 8: Pipeline layout now includes camera at group 0
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle Render Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout, &render_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_particle",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_particle",
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffers[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: uniform_buffer.as_entire_binding() },
            ],
        });
        
        Self {
            particle_buffers,
            read_buffer: 0,
            vertex_buffer,
            uniform_buffer,
            vector_texture,
            vector_texture_view,
            advect_pipeline,
            advect_bind_groups,
            render_pipeline,
            render_bind_group,
            config: ParticleConfig::default(),
            particle_count: 0,
            spawn_accumulator: 0.0,
            current_time: 0.0,
            seed: 42,
        }
    }
    
    /// Upload vector field data to GPU texture
    pub fn upload_vector_field(&self, queue: &wgpu::Queue, field: &VectorField) {
        let data: Vec<f32> = field.data.iter()
            .flat_map(|cell| [cell.u, cell.v, cell.w, cell.vorticity])
            .collect();
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.vector_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(field.config.grid_width * 16), // 4 floats * 4 bytes
                rows_per_image: Some(field.config.grid_height),
            },
            wgpu::Extent3d {
                width: field.config.grid_width,
                height: field.config.grid_height,
                depth_or_array_layers: 1,
            },
        );
    }
    
    /// Update particle system (call each frame)
    pub fn update(&mut self, queue: &wgpu::Queue, dt: f32, field: &VectorField) {
        self.current_time += dt;
        self.spawn_accumulator += self.config.spawn_rate * dt;
        
        // Spawn new particles
        let spawn_count = self.spawn_accumulator.floor() as u32;
        if spawn_count > 0 {
            self.spawn_accumulator -= spawn_count as f32;
            self.spawn_particles(queue, spawn_count, field);
        }
        
        // Update uniforms
        let uniforms = ParticleUniforms {
            time: [self.current_time, dt, self.config.spawn_rate, self.seed as f32],
            config: [
                self.config.lifetime,
                self.config.lifetime_variance,
                self.config.base_size,
                self.config.speed_size_factor,
            ],
            bounds: [
                field.config.lon_min,
                field.config.lon_max,
                field.config.lat_min,
                field.config.lat_max,
            ],
            stats: [
                field.stats.max_speed.max(1.0),
                field.stats.max_vorticity.abs().max(1.0),
                self.particle_count as f32,
                0.0,
            ],
        };
        
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        
        self.seed = self.seed.wrapping_add(1);
    }
    
    /// Spawn new particles randomly in viewport
    fn spawn_particles(&mut self, queue: &wgpu::Queue, count: u32, field: &VectorField) {
        // Cap at 100K initial particles - GPU handles respawning dead ones
        const INITIAL_CAP: u32 = 100_000;
        if self.particle_count >= INITIAL_CAP {
            return; // GPU compute will respawn dead particles
        }
        
        let available = INITIAL_CAP.saturating_sub(self.particle_count);
        let to_spawn = count.min(available);
        
        if to_spawn == 0 {
            return;
        }
        
        // Generate particles on CPU (could move to compute shader for higher rates)
        let mut particles = Vec::with_capacity(to_spawn as usize);
        
        for i in 0..to_spawn {
            // Pseudo-random position within bounds
            let seed = self.seed.wrapping_add(i).wrapping_mul(1103515245).wrapping_add(12345);
            let fx = (seed & 0xFFFF) as f32 / 65535.0;
            let fy = ((seed >> 16) & 0xFFFF) as f32 / 65535.0;
            
            let lon = field.config.lon_min + fx * (field.config.lon_max - field.config.lon_min);
            let lat = field.config.lat_min + fy * (field.config.lat_max - field.config.lat_min);
            
            // Random lifetime variation
            let lifetime_seed = seed.wrapping_mul(48271);
            let lifetime_var = ((lifetime_seed & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0 * self.config.lifetime_variance;
            let lifetime = self.config.lifetime + lifetime_var;
            
            particles.push(Particle::new(lon, lat, lifetime));
        }
        
        // Upload to GPU
        let offset = (self.particle_count as usize) * std::mem::size_of::<Particle>();
        queue.write_buffer(
            &self.particle_buffers[self.read_buffer],
            offset as u64,
            bytemuck::cast_slice(&particles),
        );
        
        self.particle_count += to_spawn;
    }
    
    /// Run advection compute pass
    pub fn advect(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.particle_count == 0 {
            return;
        }
        
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Particle Advect Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.advect_pipeline);
        compute_pass.set_bind_group(0, &self.advect_bind_groups[self.read_buffer], &[]);
        
        let workgroups = (self.particle_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
        
        drop(compute_pass);
        
        // Swap buffers
        self.read_buffer = 1 - self.read_buffer;
    }
    
    /// Render particles
    /// Phase 8: Now accepts camera_bind_group for 3D globe projection
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, camera_bind_group: &'a wgpu::BindGroup) {
        if self.particle_count == 0 {
            return;
        }
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);  // Camera at group 0
        render_pass.set_bind_group(1, &self.render_bind_group, &[]);  // Particles at group 1
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..self.particle_count);
    }
    
    /// Get current particle count
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
    
    /// Set configuration
    pub fn set_config(&mut self, config: ParticleConfig) {
        self.config = config;
    }
    
    /// Clear all particles
    pub fn clear(&mut self) {
        self.particle_count = 0;
        self.spawn_accumulator = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_particle_creation() {
        let p = Particle::new(-120.0, 35.0, 5.0);
        assert!((p.lon() - -120.0).abs() < 1e-6);
        assert!((p.lat() - 35.0).abs() < 1e-6);
        assert!((p.lifetime() - 5.0).abs() < 1e-6);
        assert!(p.is_alive());
    }
    
    #[test]
    fn test_particle_age() {
        let mut p = Particle::new(0.0, 0.0, 10.0);
        p.position[3] = 5.0; // Set age to 5
        
        assert!((p.age_normalized() - 0.5).abs() < 1e-6);
        assert!(p.is_alive());
        
        p.position[3] = 11.0; // Set age past lifetime
        assert!(!p.is_alive());
    }
}

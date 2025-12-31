/*!
 * Phase 7: Telemetry Rails
 * 
 * System Vitality Rail (Left Panel):
 * - P-core/E-core utilization bars
 * - Memory usage gauge
 * - Frame time sparkline
 * - Stability score indicator
 * 
 * Weather Metrics Rail (Right Panel):
 * - Temperature gauge (semicircular)
 * - Wind speed gauge
 * - Pressure reading
 * - Physics field statistics
 */
#![allow(dead_code)] // Telemetry rails ready for integration

use wgpu::util::DeviceExt;
use std::collections::VecDeque;

/// Maximum samples in sparkline history
const SPARKLINE_SAMPLES: usize = 120;

/// System metrics collected each frame
#[derive(Clone, Copy, Default)]
pub struct SystemMetrics {
    /// CPU utilization 0.0-1.0
    pub cpu_usage: f32,
    /// Memory usage in bytes
    pub memory_used: u64,
    /// Total memory in bytes
    pub memory_total: u64,
    /// Frame time in milliseconds
    pub frame_time_ms: f32,
    /// Stability score (max/mean ratio, 1.0 = perfect)
    pub stability_score: f32,
    /// Current FPS
    pub fps: f32,
    /// GPU memory used (if available)
    pub gpu_memory_used: u64,
    /// GPU utilization 0.0-1.0 (if available)
    pub gpu_usage: f32,
}

/// Physics field metrics
#[derive(Clone, Copy, Default)]
pub struct PhysicsMetrics {
    /// Mean temperature in field
    pub temperature: f32,
    /// Temperature range (min, max)
    pub temp_range: (f32, f32),
    /// Mean wind speed
    pub wind_speed: f32,
    /// Max wind speed
    pub wind_max: f32,
    /// Mean pressure
    pub pressure: f32,
    /// Convergence intensity (0.0-1.0)
    pub convergence: f32,
    /// Number of active vortices detected
    pub vortex_count: u32,
}

/// Sparkline data for historical visualization
pub struct Sparkline {
    samples: VecDeque<f32>,
    min_val: f32,
    max_val: f32,
}

impl Sparkline {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(SPARKLINE_SAMPLES),
            min_val: 0.0,
            max_val: 1.0,
        }
    }
    
    pub fn push(&mut self, value: f32) {
        if self.samples.len() >= SPARKLINE_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
        
        // Update bounds
        if let (Some(&min), Some(&max)) = (
            self.samples.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            self.samples.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            self.min_val = min;
            self.max_val = max.max(min + 0.001); // Avoid division by zero
        }
    }
    
    pub fn normalized(&self) -> Vec<f32> {
        let range = self.max_val - self.min_val;
        self.samples
            .iter()
            .map(|&v| (v - self.min_val) / range)
            .collect()
    }
    
    pub fn current(&self) -> f32 {
        self.samples.back().copied().unwrap_or(0.0)
    }
}

/// GPU-rendered telemetry rail
pub struct TelemetryRail {
    // Pipelines
    bar_pipeline: wgpu::RenderPipeline,
    sparkline_pipeline: wgpu::RenderPipeline,
    gauge_pipeline: wgpu::RenderPipeline,
    
    // Uniform buffers
    bar_uniform_buffer: wgpu::Buffer,
    sparkline_uniform_buffer: wgpu::Buffer,
    gauge_uniform_buffer: wgpu::Buffer,
    
    // Bind groups
    bar_bind_group: wgpu::BindGroup,
    sparkline_bind_group: wgpu::BindGroup,
    gauge_bind_group: wgpu::BindGroup,
    
    // Vertex buffers for sparkline
    sparkline_vertex_buffer: wgpu::Buffer,
    
    // Historical data
    frame_time_sparkline: Sparkline,
    fps_sparkline: Sparkline,
    cpu_sparkline: Sparkline,
    memory_sparkline: Sparkline,
    
    // Current metrics
    system_metrics: SystemMetrics,
    physics_metrics: PhysicsMetrics,
    
    // Rail dimensions
    rail_width: f32,
    screen_width: f32,
    screen_height: f32,
}

/// Bar uniform data (progress bars, utilization)
/// WGSL alignment: vec4 needs 16-byte alignment
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BarUniforms {
    rect: [f32; 4],       // x, y, width, height (NDC) - 16 bytes, offset 0
    color: [f32; 4],      // RGBA color (w = unused) - 16 bytes, offset 16
    bg_color: [f32; 4],   // RGBA background - 16 bytes, offset 32
    fill: f32,            // 0.0-1.0 fill level - 4 bytes, offset 48
    border_radius: f32,   // Corner radius in pixels - 4 bytes, offset 52
    _padding: [f32; 2],   // Padding to 64 bytes - 8 bytes, offset 56
}

/// Sparkline uniform data
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SparklineUniforms {
    rect: [f32; 4],       // x, y, width, height (NDC)
    color: [f32; 4],      // RGBA line color
    point_count: u32,     // Number of points
    line_width: f32,      // Line thickness
    _padding: [f32; 2],
}

/// Gauge uniform data (semicircular gauges)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GaugeUniforms {
    center: [f32; 2],     // Center position (NDC)
    radius: f32,          // Outer radius
    thickness: f32,       // Arc thickness
    value: f32,           // 0.0-1.0 value
    angle_start: f32,     // Start angle (radians)
    angle_sweep: f32,     // Sweep angle (radians)
    _padding: f32,
    color_low: [f32; 4],  // Color at 0%
    color_high: [f32; 4], // Color at 100%
}

impl TelemetryRail {
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let screen_width = config.width as f32;
        let screen_height = config.height as f32;
        let rail_width = 200.0; // 200px wide rails
        
        // Create shader modules
        let bar_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bar Shader"),
            source: wgpu::ShaderSource::Wgsl(BAR_SHADER.into()),
        });
        
        let sparkline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sparkline Shader"),
            source: wgpu::ShaderSource::Wgsl(SPARKLINE_SHADER.into()),
        });
        
        let gauge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gauge Shader"),
            source: wgpu::ShaderSource::Wgsl(GAUGE_SHADER.into()),
        });
        
        // Bind group layouts
        let bar_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bar Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let sparkline_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sparkline Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        let gauge_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gauge Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Pipeline layouts
        let bar_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bar Pipeline Layout"),
            bind_group_layouts: &[&bar_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let sparkline_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sparkline Pipeline Layout"),
            bind_group_layouts: &[&sparkline_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let gauge_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gauge Pipeline Layout"),
            bind_group_layouts: &[&gauge_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Render pipelines
        let blend_state = wgpu::BlendState::ALPHA_BLENDING;
        
        let bar_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bar Pipeline"),
            layout: Some(&bar_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bar_shader,
                entry_point: "vs_main",
                buffers: &[],
                
            },
            fragment: Some(wgpu::FragmentState {
                module: &bar_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            
        });
        
        let sparkline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sparkline Pipeline"),
            layout: Some(&sparkline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sparkline_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8, // 2 floats
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x2,
                    }],
                }],
                
            },
            fragment: Some(wgpu::FragmentState {
                module: &sparkline_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            
        });
        
        let gauge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gauge Pipeline"),
            layout: Some(&gauge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &gauge_shader,
                entry_point: "vs_main",
                buffers: &[],
                
            },
            fragment: Some(wgpu::FragmentState {
                module: &gauge_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            
        });
        
        // Uniform buffers
        let bar_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bar Uniform Buffer"),
            contents: bytemuck::cast_slice(&[BarUniforms {
                rect: [0.0; 4],
                color: [0.0; 4],
                bg_color: [0.0; 4],
                fill: 0.0,
                border_radius: 0.0,
                _padding: [0.0; 2],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let sparkline_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sparkline Uniform Buffer"),
            contents: bytemuck::cast_slice(&[SparklineUniforms {
                rect: [0.0; 4],
                color: [0.0; 4],
                point_count: 0,
                line_width: 1.0,
                _padding: [0.0; 2],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let gauge_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gauge Uniform Buffer"),
            contents: bytemuck::cast_slice(&[GaugeUniforms {
                center: [0.0; 2],
                radius: 0.0,
                thickness: 0.0,
                value: 0.0,
                angle_start: 0.0,
                angle_sweep: 0.0,
                _padding: 0.0,
                color_low: [0.0; 4],
                color_high: [0.0; 4],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Sparkline vertex buffer (max 120 points * 2 floats)
        let sparkline_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparkline Vertex Buffer"),
            size: (SPARKLINE_SAMPLES * 2 * 4) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Bind groups
        let bar_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bar Bind Group"),
            layout: &bar_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bar_uniform_buffer.as_entire_binding(),
            }],
        });
        
        let sparkline_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparkline Bind Group"),
            layout: &sparkline_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sparkline_uniform_buffer.as_entire_binding(),
            }],
        });
        
        let gauge_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gauge Bind Group"),
            layout: &gauge_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gauge_uniform_buffer.as_entire_binding(),
            }],
        });
        
        Self {
            bar_pipeline,
            sparkline_pipeline,
            gauge_pipeline,
            bar_uniform_buffer,
            sparkline_uniform_buffer,
            gauge_uniform_buffer,
            bar_bind_group,
            sparkline_bind_group,
            gauge_bind_group,
            sparkline_vertex_buffer,
            frame_time_sparkline: Sparkline::new(),
            fps_sparkline: Sparkline::new(),
            cpu_sparkline: Sparkline::new(),
            memory_sparkline: Sparkline::new(),
            system_metrics: SystemMetrics::default(),
            physics_metrics: PhysicsMetrics::default(),
            rail_width,
            screen_width,
            screen_height,
        }
    }
    
    /// Update system metrics
    pub fn update_system(&mut self, metrics: SystemMetrics) {
        self.system_metrics = metrics;
        self.frame_time_sparkline.push(metrics.frame_time_ms);
        self.fps_sparkline.push(metrics.fps);
        self.cpu_sparkline.push(metrics.cpu_usage);
        self.memory_sparkline.push(metrics.memory_used as f32 / metrics.memory_total.max(1) as f32);
    }
    
    /// Update physics metrics
    pub fn update_physics(&mut self, metrics: PhysicsMetrics) {
        self.physics_metrics = metrics;
    }
    
    /// Render left rail (system vitality) with provided metrics
    pub fn render_left_rail<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        system: &SystemMetrics,
    ) {
        let rail_x = 20.0; // 20px from left edge
        let bar_width = 160.0; // Fixed width
        let bar_height = 35.0; // Bigger bars
        let mut y = 60.0;  // Start below title area
        
        // CPU Usage Bar - bright cyan
        let cpu_fill = system.cpu_usage.max(0.1);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            cpu_fill,
            [0.0, 0.8, 1.0], // Cyan
        );
        y += 50.0;
        
        // Memory Usage Bar - orange
        let mem_usage = (system.memory_used as f32 
            / system.memory_total.max(1) as f32).max(0.1);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            mem_usage,
            [1.0, 0.5, 0.0], // Orange
        );
        y += 50.0;
        
        // FPS Bar - green
        let fps_fill = (system.fps / 200.0).clamp(0.1, 1.0);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            fps_fill,
            [0.0, 1.0, 0.3], // Green
        );
        y += 50.0;
        
        // Frame Time Bar - yellow
        let frame_fill = (system.frame_time_ms / 32.0).clamp(0.1, 1.0);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            frame_fill,
            [1.0, 1.0, 0.0], // Yellow
        );
    }
    
    /// Render right rail (weather/physics metrics) with provided metrics
    pub fn render_right_rail<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        physics: &PhysicsMetrics,
    ) {
        let rail_x = self.screen_width - 180.0; // 180px from right edge
        let bar_width = 160.0;
        let bar_height = 35.0;
        let mut y = 60.0;
        
        // Temperature Bar - red/orange gradient
        let temp_normalized = ((physics.temperature - physics.temp_range.0)
            / (physics.temp_range.1 - physics.temp_range.0).max(0.001))
            .clamp(0.1, 1.0);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            temp_normalized,
            [1.0, 0.3, 0.1], // Red-orange
        );
        y += 50.0;
        
        // Wind Speed Bar - light blue
        let wind_normalized = (physics.wind_speed / physics.wind_max.max(0.001))
            .clamp(0.1, 1.0);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            wind_normalized,
            [0.3, 0.7, 1.0], // Light blue
        );
        y += 50.0;
        
        // Convergence Bar - magenta
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            physics.convergence.max(0.1),
            [1.0, 0.2, 0.8], // Magenta
        );
        y += 50.0;
        
        // Pressure Bar - purple
        let pressure_normalized = ((physics.pressure - 980.0) / 60.0).clamp(0.1, 1.0);
        self.render_bar(
            render_pass,
            queue,
            rail_x,
            y,
            bar_width,
            bar_height,
            pressure_normalized,
            [0.6, 0.3, 1.0], // Purple
        );
    }
    
    #[allow(clippy::too_many_arguments)]  // Render geometry params
    fn render_bar<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        fill: f32,
        color: [f32; 3],
    ) {
        // Convert pixel coords to NDC
        let ndc_x = (x / self.screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / self.screen_height) * 2.0;
        let ndc_w = (width / self.screen_width) * 2.0;
        let ndc_h = (height / self.screen_height) * 2.0;
        
        let uniforms = BarUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            color: [color[0], color[1], color[2], 1.0],
            bg_color: [0.1, 0.1, 0.15, 0.8],
            fill,
            border_radius: 4.0,
            _padding: [0.0; 2],
        };
        
        queue.write_buffer(&self.bar_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        render_pass.set_pipeline(&self.bar_pipeline);
        render_pass.set_bind_group(0, &self.bar_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
    
    #[allow(clippy::too_many_arguments)]  // Render geometry params
    fn render_sparkline<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        sparkline: &Sparkline,
        color: [f32; 4],
    ) {
        let samples = sparkline.normalized();
        if samples.is_empty() {
            return;
        }
        
        // Convert to vertex positions
        let mut vertices: Vec<f32> = Vec::with_capacity(samples.len() * 2);
        for (i, &val) in samples.iter().enumerate() {
            let px = x + (i as f32 / samples.len() as f32) * width;
            let py = y + height - val * height;
            
            // Convert to NDC
            let ndc_x = (px / self.screen_width) * 2.0 - 1.0;
            let ndc_y = 1.0 - (py / self.screen_height) * 2.0;
            vertices.push(ndc_x);
            vertices.push(ndc_y);
        }
        
        queue.write_buffer(&self.sparkline_vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        
        let ndc_x = (x / self.screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / self.screen_height) * 2.0;
        let ndc_w = (width / self.screen_width) * 2.0;
        let ndc_h = (height / self.screen_height) * 2.0;
        
        let uniforms = SparklineUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            color,
            point_count: samples.len() as u32,
            line_width: 2.0,
            _padding: [0.0; 2],
        };
        
        queue.write_buffer(&self.sparkline_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        render_pass.set_pipeline(&self.sparkline_pipeline);
        render_pass.set_bind_group(0, &self.sparkline_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.sparkline_vertex_buffer.slice(..));
        render_pass.draw(0..samples.len() as u32, 0..1);
    }
    
    #[allow(clippy::too_many_arguments)]  // Render geometry params
    fn render_gauge<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        center_x: f32,
        center_y: f32,
        radius: f32,
        thickness: f32,
        value: f32,
    ) {
        // Convert to NDC
        let ndc_cx = (center_x / self.screen_width) * 2.0 - 1.0;
        let ndc_cy = 1.0 - (center_y / self.screen_height) * 2.0;
        let ndc_r = (radius / self.screen_width) * 2.0;
        let ndc_t = (thickness / self.screen_width) * 2.0;
        
        let uniforms = GaugeUniforms {
            center: [ndc_cx, ndc_cy],
            radius: ndc_r,
            thickness: ndc_t,
            value,
            angle_start: std::f32::consts::PI,       // Start at left (180°)
            angle_sweep: std::f32::consts::PI,       // Sweep 180° (semicircle)
            _padding: 0.0,
            color_low: [1.0, 0.2, 0.2, 1.0],         // Red at low
            color_high: [0.2, 1.0, 0.2, 1.0],        // Green at high
        };
        
        queue.write_buffer(&self.gauge_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        render_pass.set_pipeline(&self.gauge_pipeline);
        render_pass.set_bind_group(0, &self.gauge_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
    
    /// Get current system metrics (for display)
    pub fn system_metrics(&self) -> &SystemMetrics {
        &self.system_metrics
    }
    
    /// Get current physics metrics (for display)
    pub fn physics_metrics(&self) -> &PhysicsMetrics {
        &self.physics_metrics
    }
}

// ============================================================================
// WGSL Shaders
// ============================================================================

const BAR_SHADER: &str = r#"
struct BarUniforms {
    rect: vec4<f32>,      // x, y, width, height (NDC)
    color: vec4<f32>,     // RGBA color (w unused)
    bg_color: vec4<f32>,
    fill: f32,
    border_radius: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: BarUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    
    let pos = positions[vertex_index];
    let x = uniforms.rect.x + pos.x * uniforms.rect.z;
    let y = uniforms.rect.y - pos.y * uniforms.rect.w;
    
    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = pos;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    
    // Solid dark background
    var color = vec4<f32>(0.05, 0.05, 0.08, 0.95);
    
    // Bright fill bar - SOLID color, no gradient bullshit
    if uv.x < uniforms.fill {
        color = vec4<f32>(uniforms.color.rgb, 1.0);
    }
    
    // Sharp white border on all edges
    let border = 0.02;
    if uv.x < border || uv.x > (1.0 - border) || uv.y < border || uv.y > (1.0 - border) {
        color = vec4<f32>(0.8, 0.8, 0.9, 1.0);
    }
    
    return color;
}
"#;

const SPARKLINE_SHADER: &str = r#"
struct SparklineUniforms {
    rect: vec4<f32>,
    color: vec4<f32>,
    point_count: u32,
    line_width: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: SparklineUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.color = uniforms.color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

const GAUGE_SHADER: &str = r#"
struct GaugeUniforms {
    center: vec2<f32>,
    radius: f32,
    thickness: f32,
    value: f32,
    angle_start: f32,
    angle_sweep: f32,
    _padding: f32,
    color_low: vec4<f32>,
    color_high: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: GaugeUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Create a quad covering the gauge area
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    
    let pos = positions[vertex_index];
    let scaled = pos * (uniforms.radius + uniforms.thickness);
    
    var output: VertexOutput;
    output.position = vec4<f32>(uniforms.center + scaled, 0.0, 1.0);
    output.uv = pos;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let dist = length(uv);
    
    // Check if within ring
    let inner_r = 1.0 - uniforms.thickness / uniforms.radius;
    if dist < inner_r || dist > 1.0 {
        discard;
    }
    
    // Calculate angle (atan2 gives -PI to PI)
    var angle = atan2(uv.y, uv.x);
    if angle < 0.0 {
        angle += 2.0 * 3.14159265;
    }
    
    // Check if within arc
    let normalized_angle = (angle - uniforms.angle_start) / uniforms.angle_sweep;
    if normalized_angle < 0.0 || normalized_angle > 1.0 {
        discard;
    }
    
    // Color based on whether filled
    if normalized_angle <= uniforms.value {
        // Interpolate between low and high colors
        return mix(uniforms.color_low, uniforms.color_high, uniforms.value);
    } else {
        // Unfilled portion - dark
        return vec4<f32>(0.15, 0.15, 0.2, 0.6);
    }
}
"#;

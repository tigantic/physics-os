/*!
 * Probe Panel - Phase 6b: Tensor Inspection UI
 * 
 * Displays tensor data for selected convergence zone nodes:
 * - Contribution weights (which variables drive convergence)
 * - Temporal probability curve (confidence over frames)
 * - Tensor spine (parallel coordinates visualization)
 * 
 * Constitutional Compliance:
 * - Article V: GPU-rendered procedural graphics
 * - Doctrine 6: Mathematical transparency for users
 */
#![allow(dead_code)] // Probe panel API ready for integration

use glam::Vec3;
use wgpu::util::DeviceExt;

/// Probe state machine (per Appendix D specification)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbeState {
    /// No interaction - globe rotates freely
    Idle,
    /// Mouse hovering over convergence zone
    Hover { zone_intensity: f32 },
    /// Click locked - computing PCA
    Anchor { node_id: u32, world_pos: Vec3 },
    /// Panel visible with tensor data
    Probe { node_id: u32, world_pos: Vec3 },
    /// Causal trace mode - ghost rendering
    CausalTrace { node_id: u32, variable_idx: u32 },
}

impl Default for ProbeState {
    fn default() -> Self {
        ProbeState::Idle
    }
}

/// Contribution weight for a single variable
#[derive(Clone, Copy, Debug)]
pub struct ContributionWeight {
    pub name: &'static str,
    pub weight: f32,  // 0.0 to 1.0
    pub color: [f32; 4],
}

/// Tensor data extracted from RAM bridge node
#[derive(Clone, Debug)]
pub struct TensorProbeData {
    /// Node ID from spatial lookup
    pub node_id: u32,
    /// World position (ECEF coordinates)
    pub world_pos: Vec3,
    /// Geographic coordinates (lat, lon)
    pub geo_coords: (f32, f32),
    /// Convergence intensity at this node
    pub intensity: f32,
    /// Variable contributions (sorted by weight)
    pub contributions: Vec<ContributionWeight>,
    /// Temporal confidence curve (frame_idx, confidence)
    pub temporal_curve: Vec<(u32, f32)>,
    /// Raw tensor values for spine visualization
    pub tensor_spine: Vec<f32>,
}

impl Default for TensorProbeData {
    fn default() -> Self {
        Self {
            node_id: 0,
            world_pos: Vec3::ZERO,
            geo_coords: (0.0, 0.0),
            intensity: 0.0,
            contributions: vec![
                ContributionWeight { name: "Temperature", weight: 0.0, color: [1.0, 0.4, 0.2, 1.0] },
                ContributionWeight { name: "Vorticity", weight: 0.0, color: [0.4, 0.8, 1.0, 1.0] },
                ContributionWeight { name: "Moisture", weight: 0.0, color: [0.2, 0.8, 0.4, 1.0] },
                ContributionWeight { name: "Pressure", weight: 0.0, color: [0.8, 0.4, 1.0, 1.0] },
            ],
            temporal_curve: Vec::new(),
            tensor_spine: Vec::new(),
        }
    }
}

/// Probe Panel renderer - slides from right rail
pub struct ProbePanel {
    state: ProbeState,
    data: TensorProbeData,
    slide_animation: f32,  // 0.0 = hidden, 1.0 = fully visible
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProbePanelUniforms {
    screen_size: [f32; 2],
    panel_pos: [f32; 2],      // Top-left corner in pixels
    panel_size: [f32; 2],     // Width, height in pixels
    slide_progress: f32,      // 0.0 to 1.0 for animation
    time: f32,
    // Contribution bars
    bar_weights: [f32; 4],    // Temperature, Vorticity, Moisture, Pressure
    bar_colors: [[f32; 4]; 4],
    // Tether line
    tether_screen: [f32; 2],  // Screen position of click
    tether_active: f32,
    _padding: f32,
    // Phase 8: Appendix D - Temporal curve samples packed as vec4s (9 samples + padding)
    temporal_samples_0: [f32; 4],  // samples 0-3
    temporal_samples_1: [f32; 4],  // samples 4-7
    temporal_samples_2: [f32; 4],  // sample 8 + padding
    // Phase 8: Appendix D - Tensor spine values packed as vec4s (8 rank indices R0-R7)
    tensor_spine_0: [f32; 4],  // R0-R3
    tensor_spine_1: [f32; 4],  // R4-R7
}

impl ProbePanel {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        // Shader for panel background, bars, tether, temporal curve, and tensor spine
        let shader_src = r#"
struct Uniforms {
    screen_size: vec2<f32>,
    panel_pos: vec2<f32>,
    panel_size: vec2<f32>,
    slide_progress: f32,
    time: f32,
    bar_weights: vec4<f32>,
    bar_colors: array<vec4<f32>, 4>,
    tether_screen: vec2<f32>,
    tether_active: f32,
    _padding: f32,
    // Phase 8: Temporal samples packed as vec4s for 16-byte alignment
    temporal_samples_0: vec4<f32>,  // samples 0-3
    temporal_samples_1: vec4<f32>,  // samples 4-7
    temporal_samples_2: vec4<f32>,  // sample 8 + padding
    // Phase 8: Tensor spine packed as vec4s
    tensor_spine_0: vec4<f32>,  // R0-R3
    tensor_spine_1: vec4<f32>,  // R4-R7
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) screen_pos: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen quad
    var positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
    );
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0)
    );
    
    var out: VertexOutput;
    out.position = vec4(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    out.screen_pos = (positions[vertex_index] * 0.5 + 0.5) * u.screen_size;
    return out;
}

// SDF for rounded rectangle
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Helper to get temporal sample by index (unpacks from vec4s)
fn get_temporal_sample(i: u32) -> f32 {
    if i < 4u {
        return u.temporal_samples_0[i];
    } else if i < 8u {
        return u.temporal_samples_1[i - 4u];
    } else {
        return u.temporal_samples_2[i - 8u];
    }
}

// Helper to get tensor spine value by index
fn get_tensor_spine(i: u32) -> f32 {
    if i < 4u {
        return u.tensor_spine_0[i];
    } else {
        return u.tensor_spine_1[i - 4u];
    }
}

// Interpolate temporal samples at normalized x position
// Unrolled for WGSL constant-indexing requirement
fn sample_temporal_curve(x: f32) -> f32 {
    // 9 logarithmic sample points mapped to [0, 1]: 0, 100, 1k, 5k, 10k, 25k, 50k, 75k, 100k
    // Positions: 0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0
    
    if x <= 0.001 {
        let t = x / 0.001;
        return mix(get_temporal_sample(0u), get_temporal_sample(1u), t);
    } else if x <= 0.01 {
        let t = (x - 0.001) / (0.01 - 0.001);
        return mix(get_temporal_sample(1u), get_temporal_sample(2u), t);
    } else if x <= 0.05 {
        let t = (x - 0.01) / (0.05 - 0.01);
        return mix(get_temporal_sample(2u), get_temporal_sample(3u), t);
    } else if x <= 0.1 {
        let t = (x - 0.05) / (0.1 - 0.05);
        return mix(get_temporal_sample(3u), get_temporal_sample(4u), t);
    } else if x <= 0.25 {
        let t = (x - 0.1) / (0.25 - 0.1);
        return mix(get_temporal_sample(4u), get_temporal_sample(5u), t);
    } else if x <= 0.5 {
        let t = (x - 0.25) / (0.5 - 0.25);
        return mix(get_temporal_sample(5u), get_temporal_sample(6u), t);
    } else if x <= 0.75 {
        let t = (x - 0.5) / (0.75 - 0.5);
        return mix(get_temporal_sample(6u), get_temporal_sample(7u), t);
    } else {
        let t = (x - 0.75) / (1.0 - 0.75);
        return mix(get_temporal_sample(7u), get_temporal_sample(8u), t);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let slide_offset = (1.0 - u.slide_progress) * u.panel_size.x;
    let panel_center = u.panel_pos + u.panel_size * 0.5 + vec2(slide_offset, 0.0);
    
    // Panel background
    let p = in.screen_pos - panel_center;
    let d = sd_rounded_box(p, u.panel_size * 0.5 - 8.0, 12.0);
    
    // Glass panel effect
    let panel_alpha = smoothstep(2.0, 0.0, d) * 0.85 * u.slide_progress;
    let panel_color = vec3(0.08, 0.10, 0.14);
    
    // Border glow
    let border = smoothstep(3.0, 0.0, abs(d)) * 0.6;
    let border_color = vec3(0.0, 0.6, 1.0);
    
    // Inside panel content
    var content_color = vec4(0.0);
    if d < 0.0 {
        // Header bar
        let header_y = u.panel_pos.y + 30.0 + slide_offset * 0.0;
        if in.screen_pos.y < header_y && in.screen_pos.y > u.panel_pos.y {
            content_color = vec4(0.0, 0.5, 0.8, 0.3);
        }
        
        // Contribution bars (4 bars, spaced vertically)
        let bar_start_y = u.panel_pos.y + 60.0;
        let bar_height = 20.0;
        let bar_spacing = 35.0;
        let bar_margin = 20.0;
        let max_bar_width = u.panel_size.x - bar_margin * 2.0 - 100.0;  // Leave room for labels
        
        for (var i = 0u; i < 4u; i++) {
            let bar_y = bar_start_y + f32(i) * bar_spacing;
            let bar_x = u.panel_pos.x + bar_margin + slide_offset;
            
            // Bar background
            if in.screen_pos.y > bar_y && in.screen_pos.y < bar_y + bar_height {
                if in.screen_pos.x > bar_x && in.screen_pos.x < bar_x + max_bar_width {
                    // Background track
                    content_color = vec4(0.15, 0.15, 0.2, 0.8);
                    
                    // Filled portion
                    let fill_width = max_bar_width * u.bar_weights[i];
                    if in.screen_pos.x < bar_x + fill_width {
                        content_color = vec4(u.bar_colors[i].rgb, 0.9);
                    }
                }
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // TEMPORAL PROBABILITY CURVE (Appendix D.3.4)
        // Shows confidence over 100k frames with logarithmic sampling
        // ═══════════════════════════════════════════════════════════════════
        let curve_y = bar_start_y + 4.0 * bar_spacing + 20.0;
        let curve_height = 60.0;
        if in.screen_pos.y > curve_y && in.screen_pos.y < curve_y + curve_height {
            let curve_x = u.panel_pos.x + bar_margin + slide_offset;
            let curve_width = u.panel_size.x - bar_margin * 2.0;
            if in.screen_pos.x > curve_x && in.screen_pos.x < curve_x + curve_width {
                let local_x = (in.screen_pos.x - curve_x) / curve_width;
                let local_y = 1.0 - (in.screen_pos.y - curve_y) / curve_height;
                
                // Grid lines (4x4)
                let grid_x = fract(local_x * 4.0);
                let grid_y = fract(local_y * 4.0);
                let grid = max(
                    smoothstep(0.02, 0.0, abs(grid_x - 0.5) - 0.48),
                    smoothstep(0.02, 0.0, abs(grid_y - 0.5) - 0.48)
                ) * 0.15;
                
                // Sample the actual temporal curve data
                let curve_val = sample_temporal_curve(local_x);
                let on_curve = smoothstep(0.04, 0.0, abs(local_y - curve_val));
                
                // Fill area under curve with gradient
                let under_curve = smoothstep(curve_val + 0.02, curve_val - 0.02, local_y) * 0.3;
                
                content_color = vec4(
                    grid * 0.3 + on_curve * 0.2 + under_curve * 0.0,
                    grid * 0.4 + on_curve * 0.9 + under_curve * 0.4,
                    grid * 0.5 + on_curve * 1.0 + under_curve * 0.6,
                    0.5 + on_curve * 0.5 + under_curve
                );
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // TENSOR SPINE - 3D Parallel Coordinates (Appendix D.3.4)
        // Shows 8 rank indices R0-R7 as connected line segments
        // ═══════════════════════════════════════════════════════════════════
        let spine_y = curve_y + curve_height + 15.0;
        let spine_height = 50.0;
        if in.screen_pos.y > spine_y && in.screen_pos.y < spine_y + spine_height {
            let spine_x = u.panel_pos.x + bar_margin + slide_offset;
            let spine_width = u.panel_size.x - bar_margin * 2.0;
            if in.screen_pos.x > spine_x && in.screen_pos.x < spine_x + spine_width {
                let local_x = (in.screen_pos.x - spine_x) / spine_width;
                let local_y = 1.0 - (in.screen_pos.y - spine_y) / spine_height;
                
                // Vertical axis lines for each rank
                var on_axis = 0.0;
                for (var i = 0u; i < 8u; i++) {
                    let axis_x = (f32(i) + 0.5) / 8.0;
                    on_axis += smoothstep(0.008, 0.0, abs(local_x - axis_x)) * 0.3;
                }
                
                // Connecting lines between ranks
                var on_spine = 0.0;
                for (var i = 0u; i < 7u; i++) {
                    let x0 = (f32(i) + 0.5) / 8.0;
                    let x1 = (f32(i) + 1.5) / 8.0;
                    let y0 = get_tensor_spine(i);
                    let y1 = get_tensor_spine(i + 1u);
                    
                    // Check if we're on this line segment
                    if local_x >= x0 && local_x <= x1 {
                        let t = (local_x - x0) / (x1 - x0);
                        let line_y = mix(y0, y1, t);
                        on_spine = max(on_spine, smoothstep(0.05, 0.0, abs(local_y - line_y)));
                    }
                }
                
                // Node dots at each rank position
                var on_node = 0.0;
                for (var i = 0u; i < 8u; i++) {
                    let node_x = (f32(i) + 0.5) / 8.0;
                    let node_y = get_tensor_spine(i);
                    let dist = length(vec2(local_x - node_x, (local_y - node_y) * 0.5));
                    on_node = max(on_node, smoothstep(0.025, 0.015, dist));
                }
                
                // Combine: axes (dim), spine line (cyan), nodes (bright)
                content_color = vec4(
                    on_axis * 0.3 + on_spine * 0.2 + on_node * 0.4,
                    on_axis * 0.4 + on_spine * 0.7 + on_node * 0.9,
                    on_axis * 0.5 + on_spine * 1.0 + on_node * 1.0,
                    max(on_axis, max(on_spine * 0.8, on_node)) * 0.9
                );
            }
        }
    }
    
    // Tether line (from click point to panel)
    if u.tether_active > 0.5 {
        let tether_start = u.tether_screen;
        let tether_end = vec2(u.panel_pos.x + slide_offset, panel_center.y);
        
        // Line distance
        let line_dir = normalize(tether_end - tether_start);
        let to_pixel = in.screen_pos - tether_start;
        let proj_len = dot(to_pixel, line_dir);
        let line_len = length(tether_end - tether_start);
        
        if proj_len > 0.0 && proj_len < line_len {
            let closest = tether_start + line_dir * proj_len;
            let dist = length(in.screen_pos - closest);
            
            // Animated pulse along line
            let pulse_pos = fract(u.time * 2.0);
            let pulse = smoothstep(0.1, 0.0, abs(proj_len / line_len - pulse_pos));
            
            let tether_alpha = smoothstep(3.0, 0.0, dist) * (0.5 + pulse * 0.5);
            let tether_col = vec3(0.0, 0.7, 1.0);
            
            return mix(
                vec4(panel_color + border_color * border, panel_alpha) + content_color,
                vec4(tether_col, 1.0),
                tether_alpha * u.slide_progress
            );
        }
    }
    
    return vec4(panel_color + border_color * border, panel_alpha) + content_color;
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe_panel_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let uniforms = ProbePanelUniforms {
            screen_size: [1920.0, 1080.0],
            panel_pos: [1920.0 - 350.0, 100.0],  // Right side
            panel_size: [320.0, 400.0],
            slide_progress: 0.0,
            time: 0.0,
            bar_weights: [0.0; 4],
            bar_colors: [
                [1.0, 0.4, 0.2, 1.0],  // Temperature - orange
                [0.4, 0.8, 1.0, 1.0],  // Vorticity - cyan
                [0.2, 0.8, 0.4, 1.0],  // Moisture - green
                [0.8, 0.4, 1.0, 1.0],  // Pressure - purple
            ],
            tether_screen: [0.0, 0.0],
            tether_active: 0.0,
            _padding: 0.0,
            // Phase 8: Appendix D - Temporal curve (9 samples packed as vec4s)
            temporal_samples_0: [0.2, 0.3, 0.45, 0.55],
            temporal_samples_1: [0.65, 0.72, 0.78, 0.82],
            temporal_samples_2: [0.85, 0.0, 0.0, 0.0],
            // Phase 8: Appendix D - Tensor spine (8 rank values packed as vec4s)
            tensor_spine_0: [0.8, 0.6, 0.4, 0.7],
            tensor_spine_1: [0.5, 0.3, 0.6, 0.4],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe_panel_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe_panel_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe_panel_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_panel_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("probe_panel_pipeline"),
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
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_panel_vertex_buffer"),
            size: 64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        Self {
            state: ProbeState::Idle,
            data: TensorProbeData::default(),
            slide_animation: 0.0,
            pipeline,
            uniform_buffer,
            bind_group,
            vertex_buffer,
        }
    }

    /// Get current probe state
    pub fn state(&self) -> ProbeState {
        self.state
    }

    /// Transition to hover state when mouse enters convergence zone
    pub fn on_hover(&mut self, intensity: f32) {
        if intensity > 0.3 && self.state == ProbeState::Idle {
            self.state = ProbeState::Hover { zone_intensity: intensity };
        }
    }

    /// Transition to anchor state on click
    pub fn on_click(&mut self, node_id: u32, world_pos: Vec3, _screen_pos: (f32, f32)) {
        match self.state {
            ProbeState::Hover { .. } | ProbeState::Idle => {
                self.state = ProbeState::Anchor { node_id, world_pos };
                // Start loading tensor data (simulated for now)
                self.load_tensor_data(node_id, world_pos);
                self.data.geo_coords = world_pos_to_geo(world_pos);
            }
            ProbeState::Probe { .. } => {
                // Click on variable would go to CausalTrace
                // For now, just close the panel
                self.state = ProbeState::Idle;
            }
            _ => {}
        }
    }

    /// Exit hover when mouse leaves zone
    pub fn on_hover_exit(&mut self) {
        if let ProbeState::Hover { .. } = self.state {
            self.state = ProbeState::Idle;
        }
    }

    /// Close panel on ESC or click outside
    pub fn close(&mut self) {
        self.state = ProbeState::Idle;
    }

    /// Load tensor data from RAM bridge (or simulate)
    fn load_tensor_data(&mut self, node_id: u32, world_pos: Vec3) {
        // Simulate loading tensor data
        // In production, this would read from RAM bridge
        self.data = TensorProbeData {
            node_id,
            world_pos,
            geo_coords: world_pos_to_geo(world_pos),
            intensity: 0.85,
            contributions: vec![
                ContributionWeight { name: "Temperature", weight: 0.783, color: [1.0, 0.4, 0.2, 1.0] },
                ContributionWeight { name: "Vorticity", weight: 0.521, color: [0.4, 0.8, 1.0, 1.0] },
                ContributionWeight { name: "Moisture", weight: 0.417, color: [0.2, 0.8, 0.4, 1.0] },
                ContributionWeight { name: "Pressure", weight: 0.294, color: [0.8, 0.4, 1.0, 1.0] },
            ],
            temporal_curve: (0..20).map(|i| (i * 5000, 0.5 + 0.3 * (i as f32 * 0.5).sin())).collect(),
            tensor_spine: vec![0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.6, 0.4],
        };
        
        // Transition to probe state after "loading"
        self.state = ProbeState::Probe { node_id, world_pos };
    }

    /// Update animation and uniforms
    pub fn update(&mut self, queue: &wgpu::Queue, screen_size: (u32, u32), time: f32, tether_pos: Option<(f32, f32)>) {
        // Animate slide
        let target = match self.state {
            ProbeState::Probe { .. } | ProbeState::Anchor { .. } => 1.0,
            _ => 0.0,
        };
        self.slide_animation += (target - self.slide_animation) * 0.15;  // Ease-out

        let panel_width = 320.0;
        
        // Phase 8: Appendix D - Extract temporal curve samples (9 logarithmic points)
        let temporal_samples: [f32; 9] = {
            let mut samples = [0.0f32; 9];
            let log_frames = [0u32, 100, 1000, 5000, 10000, 25000, 50000, 75000, 100000];
            for (i, &target_frame) in log_frames.iter().enumerate() {
                let confidence = self.data.temporal_curve.iter()
                    .min_by_key(|(f, _)| (*f as i64 - target_frame as i64).abs())
                    .map(|(_, c)| *c)
                    .unwrap_or(0.5);
                samples[i] = confidence;
            }
            samples
        };
        
        // Phase 8: Appendix D - Extract tensor spine (8 rank values)
        let tensor_spine: [f32; 8] = {
            let mut spine = [0.5f32; 8];
            for (i, val) in self.data.tensor_spine.iter().take(8).enumerate() {
                spine[i] = *val;
            }
            spine
        };
        
        let uniforms = ProbePanelUniforms {
            screen_size: [screen_size.0 as f32, screen_size.1 as f32],
            panel_pos: [screen_size.0 as f32 - panel_width - 30.0, 100.0],
            panel_size: [panel_width, 400.0],
            slide_progress: self.slide_animation,
            time,
            bar_weights: [
                self.data.contributions.get(0).map(|c| c.weight).unwrap_or(0.0),
                self.data.contributions.get(1).map(|c| c.weight).unwrap_or(0.0),
                self.data.contributions.get(2).map(|c| c.weight).unwrap_or(0.0),
                self.data.contributions.get(3).map(|c| c.weight).unwrap_or(0.0),
            ],
            bar_colors: [
                self.data.contributions.get(0).map(|c| c.color).unwrap_or([1.0, 0.4, 0.2, 1.0]),
                self.data.contributions.get(1).map(|c| c.color).unwrap_or([0.4, 0.8, 1.0, 1.0]),
                self.data.contributions.get(2).map(|c| c.color).unwrap_or([0.2, 0.8, 0.4, 1.0]),
                self.data.contributions.get(3).map(|c| c.color).unwrap_or([0.8, 0.4, 1.0, 1.0]),
            ],
            tether_screen: tether_pos.unwrap_or((0.0, 0.0)).into(),
            tether_active: if tether_pos.is_some() && self.slide_animation > 0.1 { 1.0 } else { 0.0 },
            _padding: 0.0,
            // Pack temporal samples into vec4s
            temporal_samples_0: [temporal_samples[0], temporal_samples[1], temporal_samples[2], temporal_samples[3]],
            temporal_samples_1: [temporal_samples[4], temporal_samples[5], temporal_samples[6], temporal_samples[7]],
            temporal_samples_2: [temporal_samples[8], 0.0, 0.0, 0.0],
            // Pack tensor spine into vec4s
            tensor_spine_0: [tensor_spine[0], tensor_spine[1], tensor_spine[2], tensor_spine[3]],
            tensor_spine_1: [tensor_spine[4], tensor_spine[5], tensor_spine[6], tensor_spine[7]],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Render the probe panel
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.slide_animation < 0.01 {
            return;  // Don't render when fully hidden
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }

    /// Check if panel is visible (for input handling)
    pub fn is_visible(&self) -> bool {
        self.slide_animation > 0.1
    }
    
    /// Get probe data for text rendering
    pub fn get_display_data(&self) -> Option<&TensorProbeData> {
        match self.state {
            ProbeState::Probe { .. } | ProbeState::Anchor { .. } => Some(&self.data),
            _ => None,
        }
    }
}

/// Convert world position (approximate ECEF) to lat/lon
fn world_pos_to_geo(pos: Vec3) -> (f32, f32) {
    // Simplified - assumes unit sphere scaled
    let lat = pos.y.asin().to_degrees();
    let lon = pos.z.atan2(pos.x).to_degrees();
    (lat, lon)
}

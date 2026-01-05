/*!
 * Timeline Scrubber - Phase 6b: Temporal Navigation
 * 
 * Navigate 100k-frame simulation timeline:
 * - Seismograph-style progress bar
 * - Instant frame seek (no loading - data in RAM)
 * - Comparison shadow overlay
 * 
 * Constitutional Compliance:
 * - Article V: GPU-rendered procedural graphics
 * - Doctrine 7: User controls time, not just watches it
 */

use wgpu::util::DeviceExt;

/// Timeline state
pub struct TimelineScrubber {
    /// Current frame index (0 to max_frames)
    current_frame: u64,
    /// Maximum frame count
    max_frames: u64,
    /// Is timeline being dragged?
    is_dragging: bool,
    /// Playback state
    playback: PlaybackState,
    /// Animation time for pulse effects
    animation_time: f32,
    /// Pipeline and GPU resources
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaybackState {
    Paused,
    Playing { speed: f32 },  // 1.0 = realtime, 2.0 = 2x, etc.
    Rewinding { speed: f32 },
}

impl Default for PlaybackState {
    fn default() -> Self {
        PlaybackState::Paused
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TimelineUniforms {
    screen_size: [f32; 2],
    bar_pos: [f32; 2],       // Top-left corner
    bar_size: [f32; 2],      // Width, height
    progress: f32,           // 0.0 to 1.0
    time: f32,
    is_dragging: f32,
    heartbeat_phase: f32,
    current_frame: f32,
    max_frames: f32,
    _pad: [f32; 2],
}

impl TimelineScrubber {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader_src = r#"
struct Uniforms {
    screen_size: vec2<f32>,
    bar_pos: vec2<f32>,
    bar_size: vec2<f32>,
    progress: f32,
    time: f32,
    is_dragging: f32,
    heartbeat_phase: f32,
    current_frame: f32,
    max_frames: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) screen_pos: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
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

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let bar_center = u.bar_pos + u.bar_size * 0.5;
    let p = in.screen_pos - bar_center;
    
    // Background track
    let track_height = 12.0;
    let track_d = sd_rounded_box(p, vec2(u.bar_size.x * 0.5, track_height * 0.5), 6.0);
    let track_alpha = smoothstep(1.0, 0.0, track_d) * 0.6;
    var color = vec3(0.1, 0.12, 0.15);
    
    // Progress fill
    let fill_width = u.bar_size.x * u.progress;
    let fill_center_x = u.bar_pos.x + fill_width * 0.5;
    let fill_p = in.screen_pos - vec2(fill_center_x, bar_center.y);
    let fill_d = sd_rounded_box(fill_p, vec2(fill_width * 0.5 - 2.0, track_height * 0.5 - 2.0), 4.0);
    
    if fill_d < 0.0 {
        // Gradient fill with pulse
        let pulse = sin(u.time * 3.0) * 0.1 + 0.9;
        let local_x = (in.screen_pos.x - u.bar_pos.x) / fill_width;
        color = mix(
            vec3(0.0, 0.4, 0.8),  // Blue start
            vec3(0.0, 0.7, 1.0) * pulse,  // Bright cyan end
            local_x
        );
    }
    
    // Playhead (current position marker)
    let playhead_x = u.bar_pos.x + u.bar_size.x * u.progress;
    let playhead_dist = abs(in.screen_pos.x - playhead_x);
    let playhead_glow = smoothstep(20.0, 0.0, playhead_dist);
    
    // Playhead circle
    let playhead_center = vec2(playhead_x, bar_center.y);
    let playhead_radius = 8.0 + u.is_dragging * 4.0;  // Larger when dragging
    let playhead_d = length(in.screen_pos - playhead_center) - playhead_radius;
    
    if playhead_d < 0.0 {
        color = vec3(1.0, 1.0, 1.0);
    } else if playhead_d < 3.0 {
        // Glow ring
        let ring = smoothstep(3.0, 0.0, playhead_d);
        color = mix(color, vec3(0.0, 0.8, 1.0), ring * 0.8);
    }
    
    // Heartbeat pulse (synced to physics engine)
    let heartbeat = sin(u.heartbeat_phase) * 0.5 + 0.5;
    let heartbeat_glow = smoothstep(u.bar_size.x * 0.1, 0.0, abs(in.screen_pos.x - playhead_x)) * heartbeat * 0.3;
    color += vec3(0.0, 0.5, 1.0) * heartbeat_glow;
    
    // Tick marks (every 25k frames)
    let tick_spacing = u.bar_size.x / 4.0;
    for (var i = 1u; i < 4u; i++) {
        let tick_x = u.bar_pos.x + tick_spacing * f32(i);
        let tick_dist = abs(in.screen_pos.x - tick_x);
        if tick_dist < 1.5 && abs(p.y) < track_height * 0.5 + 5.0 {
            color = mix(color, vec3(0.4, 0.5, 0.6), 0.8);
        }
    }
    
    // Frame number labels would be rendered via text system
    
    // Only render if near the bar
    let final_alpha = track_alpha * smoothstep(track_height + 30.0, track_height, abs(p.y));
    
    if final_alpha < 0.01 {
        discard;
    }
    
    return vec4(color, final_alpha);
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("timeline_scrubber_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let uniforms = TimelineUniforms {
            screen_size: [1920.0, 1080.0],
            bar_pos: [200.0, 1080.0 - 80.0],  // Bottom of screen
            bar_size: [1520.0, 40.0],
            progress: 0.0,
            time: 0.0,
            is_dragging: 0.0,
            heartbeat_phase: 0.0,
            current_frame: 0.0,
            max_frames: 100000.0,
            _pad: [0.0; 2],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("timeline_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("timeline_bind_group_layout"),
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
            label: Some("timeline_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("timeline_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("timeline_pipeline"),
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

        Self {
            current_frame: 0,
            max_frames: 100_000,
            is_dragging: false,
            playback: PlaybackState::Paused,
            animation_time: 0.0,
            pipeline,
            uniform_buffer,
            bind_group,
        }
    }

    /// Get current frame
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get progress as 0.0-1.0
    pub fn progress(&self) -> f32 {
        self.current_frame as f32 / self.max_frames as f32
    }

    /// Set frame from external source (e.g., RAM bridge)
    pub fn set_frame(&mut self, frame: u64) {
        self.current_frame = frame.min(self.max_frames);
    }

    /// Set max frames
    pub fn set_max_frames(&mut self, max: u64) {
        self.max_frames = max.max(1);
    }

    /// Start dragging at screen position
    pub fn start_drag(&mut self, screen_x: f32, bar_x: f32, bar_width: f32) {
        self.is_dragging = true;
        self.update_from_drag(screen_x, bar_x, bar_width);
    }

    /// Update position during drag
    pub fn update_from_drag(&mut self, screen_x: f32, bar_x: f32, bar_width: f32) {
        if self.is_dragging {
            let progress = ((screen_x - bar_x) / bar_width).clamp(0.0, 1.0);
            self.current_frame = (progress * self.max_frames as f32) as u64;
        }
    }

    /// End dragging
    pub fn end_drag(&mut self) {
        self.is_dragging = false;
    }

    /// Check if point is within timeline bar
    pub fn hit_test(&self, screen_pos: (f32, f32), screen_size: (u32, u32)) -> bool {
        let bar_y = screen_size.1 as f32 - 80.0;
        let bar_height = 40.0;
        let bar_x = 200.0;
        let bar_width = screen_size.0 as f32 - 400.0;
        
        screen_pos.0 >= bar_x && screen_pos.0 <= bar_x + bar_width &&
        screen_pos.1 >= bar_y && screen_pos.1 <= bar_y + bar_height
    }

    /// Toggle playback
    pub fn toggle_playback(&mut self) {
        self.playback = match self.playback {
            PlaybackState::Paused => PlaybackState::Playing { speed: 1.0 },
            PlaybackState::Playing { .. } => PlaybackState::Paused,
            PlaybackState::Rewinding { .. } => PlaybackState::Paused,
        };
    }

    /// Step forward one frame
    pub fn step_forward(&mut self) {
        if self.current_frame < self.max_frames {
            self.current_frame += 1;
        }
    }

    /// Step backward one frame
    pub fn step_backward(&mut self) {
        if self.current_frame > 0 {
            self.current_frame -= 1;
        }
    }

    /// Jump to start
    pub fn go_to_start(&mut self) {
        self.current_frame = 0;
    }

    /// Jump to end
    pub fn go_to_end(&mut self) {
        self.current_frame = self.max_frames;
    }

    /// Update animation and playback
    pub fn update(&mut self, queue: &wgpu::Queue, screen_size: (u32, u32), dt: f32, heartbeat_phase: f32) {
        self.animation_time += dt;

        // Update playback
        match self.playback {
            PlaybackState::Playing { speed } => {
                let frames_to_add = (speed * 60.0 * dt) as u64;  // Assume 60 FPS base
                self.current_frame = (self.current_frame + frames_to_add).min(self.max_frames);
                if self.current_frame >= self.max_frames {
                    self.playback = PlaybackState::Paused;
                }
            }
            PlaybackState::Rewinding { speed } => {
                let frames_to_sub = (speed * 60.0 * dt) as u64;
                self.current_frame = self.current_frame.saturating_sub(frames_to_sub);
                if self.current_frame == 0 {
                    self.playback = PlaybackState::Paused;
                }
            }
            PlaybackState::Paused => {}
        }

        let bar_width = screen_size.0 as f32 - 400.0;
        let uniforms = TimelineUniforms {
            screen_size: [screen_size.0 as f32, screen_size.1 as f32],
            bar_pos: [200.0, screen_size.1 as f32 - 80.0],
            bar_size: [bar_width, 40.0],
            progress: self.progress(),
            time: self.animation_time,
            is_dragging: if self.is_dragging { 1.0 } else { 0.0 },
            heartbeat_phase,
            current_frame: self.current_frame as f32,
            max_frames: self.max_frames as f32,
            _pad: [0.0; 2],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Render the timeline bar
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }

    /// Get bar bounds for text labels
    pub fn get_bar_bounds(&self, screen_size: (u32, u32)) -> (f32, f32, f32, f32) {
        let bar_width = screen_size.0 as f32 - 400.0;
        (200.0, screen_size.1 as f32 - 80.0, bar_width, 40.0)
    }
    
    /// Get maximum frame count
    pub fn max_frames(&self) -> u64 {
        self.max_frames
    }
}

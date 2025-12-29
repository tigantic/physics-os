/*!
 * Sovereign Glass Chrome - GPU-Rendered HUD Panels
 * 
 * SDF-based glass panels with glowing borders
 * No CSS, no opacity sliders - pure GPU shader aesthetics
 * 
 * Constitutional: Article V GPU mandate, Doctrine 3 procedural rendering
 */

use wgpu::util::DeviceExt;

/// Glass panel configuration
#[derive(Clone, Copy)]
pub struct GlassPanelConfig {
    /// Position (x, y) in pixels from top-left
    pub x: f32,
    pub y: f32,
    /// Size in pixels
    pub width: f32,
    pub height: f32,
    /// Corner radius
    pub corner_radius: f32,
    /// Border width
    pub border_width: f32,
    /// Glow falloff (higher = wider glow)
    pub glow_falloff: f32,
    /// Fill color RGBA (the glass tint)
    pub fill_color: [f32; 4],
    /// Border color RGB
    pub border_color: [f32; 3],
    /// Glow intensity (0.0 - 1.0)
    pub glow_intensity: f32,
}

impl Default for GlassPanelConfig {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 200.0,
            height: 400.0,
            corner_radius: 8.0,
            border_width: 1.5,
            glow_falloff: 6.0,
            fill_color: [0.03, 0.03, 0.05, 0.88],
            border_color: [0.0, 0.5, 0.9], // Sovereign Blue
            glow_intensity: 0.5,
        }
    }
}

/// Uniforms for glass panel shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GlassUniforms {
    // Panel rect in NDC (x, y, w, h)
    rect: [f32; 4],
    // Panel size in pixels for SDF (w, h, corner_radius, border_width)
    panel_params: [f32; 4],
    // Fill color RGBA
    fill_color: [f32; 4],
    // Border color RGB + glow intensity
    border_glow: [f32; 4],
    // Glow falloff + padding
    glow_params: [f32; 4],
}

/// GPU-rendered glass panel chrome
pub struct GlassChrome {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    screen_width: f32,
    screen_height: f32,
}

impl GlassChrome {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let screen_width = config.width as f32;
        let screen_height = config.height as f32;
        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Glass Chrome Shader"),
            source: wgpu::ShaderSource::Wgsl(GLASS_SHADER.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Glass Chrome Layout"),
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
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Glass Chrome Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Glass Chrome Pipeline"),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Glass Chrome Uniforms"),
            contents: bytemuck::cast_slice(&[GlassUniforms {
                rect: [0.0; 4],
                panel_params: [0.0; 4],
                fill_color: [0.0; 4],
                border_glow: [0.0; 4],
                glow_params: [0.0; 4],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Glass Chrome Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            screen_width,
            screen_height,
        }
    }
    
    /// Render a glass panel
    pub fn render_panel<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        panel: &GlassPanelConfig,
    ) {
        // Convert pixel coords to NDC
        let ndc_x = (panel.x / self.screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (panel.y / self.screen_height) * 2.0;
        let ndc_w = (panel.width / self.screen_width) * 2.0;
        let ndc_h = (panel.height / self.screen_height) * 2.0;
        
        let uniforms = GlassUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            panel_params: [panel.width, panel.height, panel.corner_radius, panel.border_width],
            fill_color: panel.fill_color,
            border_glow: [
                panel.border_color[0],
                panel.border_color[1],
                panel.border_color[2],
                panel.glow_intensity,
            ],
            glow_params: [panel.glow_falloff, 0.0, 0.0, 0.0],
        };
        
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
    
    /// Render left telemetry rail (system vitality)
    pub fn render_left_rail<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        screen_width: f32,
        screen_height: f32,
    ) {
        let panel = GlassPanelConfig {
            x: 8.0,
            y: 8.0,
            width: 200.0,
            height: screen_height - 250.0, // Extend to near bottom
            corner_radius: 6.0,
            border_width: 1.0,
            glow_falloff: 8.0,
            fill_color: [0.02, 0.02, 0.05, 0.65],  // Dark void with glass alpha
            border_color: [0.0, 0.5, 0.9],
            glow_intensity: 0.5,
        };
        self.render_panel_sized(render_pass, queue, &panel, screen_width, screen_height);
    }
    
    /// Render right telemetry rail (weather/physics)
    pub fn render_right_rail<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        screen_width: f32,
        screen_height: f32,
    ) {
        let panel = GlassPanelConfig {
            x: screen_width - 208.0,
            y: 8.0,
            width: 200.0,
            height: screen_height - 250.0,
            corner_radius: 6.0,
            border_width: 1.0,
            glow_falloff: 8.0,
            fill_color: [0.02, 0.02, 0.05, 0.65],  // Dark void with glass alpha
            border_color: [0.0, 0.5, 0.9],
            glow_intensity: 0.5,
        };
        self.render_panel_sized(render_pass, queue, &panel, screen_width, screen_height);
    }
    
    /// Render bottom terminal panel
    pub fn render_terminal_panel<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        screen_width: f32,
        screen_height: f32,
    ) {
        let panel = GlassPanelConfig {
            x: 216.0,
            y: screen_height - 228.0,
            width: screen_width - 432.0,
            height: 220.0,
            corner_radius: 6.0,
            border_width: 1.0,
            glow_falloff: 8.0,
            fill_color: [0.02, 0.02, 0.05, 0.70],  // Dark void with glass alpha
            border_color: [0.0, 0.4, 0.8],
            glow_intensity: 0.45,
        };
        self.render_panel_sized(render_pass, queue, &panel, screen_width, screen_height);
    }
    
    /// Render panel with explicit screen dimensions
    fn render_panel_sized<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        panel: &GlassPanelConfig,
        screen_width: f32,
        screen_height: f32,
    ) {
        // Convert pixel coords to NDC
        let ndc_x = (panel.x / screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (panel.y / screen_height) * 2.0;
        let ndc_w = (panel.width / screen_width) * 2.0;
        let ndc_h = (panel.height / screen_height) * 2.0;
        
        let uniforms = GlassUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            panel_params: [panel.width, panel.height, panel.corner_radius, panel.border_width],
            fill_color: panel.fill_color,
            border_glow: [
                panel.border_color[0],
                panel.border_color[1],
                panel.border_color[2],
                panel.glow_intensity,
            ],
            glow_params: [panel.glow_falloff, 0.0, 0.0, 0.0],
        };
        
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

// ============================================================================
// WGSL Shader - Sovereign Glass with SDF
// ============================================================================

const GLASS_SHADER: &str = r#"
struct GlassUniforms {
    rect: vec4<f32>,          // NDC position (x, y, w, h)
    panel_params: vec4<f32>,  // Pixel size (w, h, corner_r, border_w)
    fill_color: vec4<f32>,    // RGBA fill
    border_glow: vec4<f32>,   // RGB + glow_intensity
    glow_params: vec4<f32>,   // glow_falloff + padding
}

@group(0) @binding(0)
var<uniform> uniforms: GlassUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,           // 0-1 UV within panel
    @location(1) local_pos: vec2<f32>,    // Pixel position within panel
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    let pos = positions[vi];
    
    let x = uniforms.rect.x + (pos.x * uniforms.rect.z); 
    let y = uniforms.rect.y - (pos.y * uniforms.rect.w);
    
    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = pos;
    output.local_pos = (pos - 0.5) * vec2<f32>(uniforms.panel_params.x, uniforms.panel_params.y);
    return output;
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let panel_size = vec2<f32>(uniforms.panel_params.x, uniforms.panel_params.y);
    let corner_radius = uniforms.panel_params.z;
    let border_width = uniforms.panel_params.w;
    let glow_falloff = uniforms.glow_params.x;
    let glow_intensity = uniforms.border_glow.w;
    let cyan = vec3<f32>(0.0, 0.8, 1.0);
    
    // SDF: negative inside, positive outside
    let dist = sd_rounded_box(input.local_pos, panel_size * 0.5, corner_radius);
    
    // CASE 1: Outside the panel (dist > 0) - only glow
    if dist > 0.0 {
        let glow = exp(-dist / glow_falloff) * glow_intensity;
        return vec4<f32>(cyan * glow, glow * 0.8);
    }
    
    // CASE 2: On the border (dist between -border_width and 0)
    if dist > -border_width {
        // Sharp cyan border
        return vec4<f32>(cyan, 0.95);
    }
    
    // CASE 3: Inside the panel (dist < -border_width)
    // Very subtle dark tint - mostly transparent
    let deep_void = vec3<f32>(0.02, 0.02, 0.05);
    return vec4<f32>(deep_void, 0.25);  // 25% opacity = see-through glass
}
"#;

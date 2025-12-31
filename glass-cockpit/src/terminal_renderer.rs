/*!
 * Phase 7: Terminal Renderer
 * 
 * GPU-accelerated terminal pane for event log display.
 * Renders scrolling text with color-coded severity levels.
 */
#![allow(dead_code)] // Terminal API ready for integration

use wgpu::util::DeviceExt;
use crate::event_log::EventLog;

/// Terminal configuration
pub struct TerminalConfig {
    /// Terminal position (x, y) in pixels from bottom-left
    pub x: f32,
    pub y: f32,
    /// Terminal size in pixels
    pub width: f32,
    pub height: f32,
    /// Font size in pixels
    pub font_size: f32,
    /// Line height multiplier
    pub line_height: f32,
    /// Background color
    pub bg_color: [f32; 4],
    /// Border color
    pub border_color: [f32; 4],
    /// Padding in pixels
    pub padding: f32,
}

impl Default for TerminalConfig {
    fn default() -> Self {
        Self {
            x: 224.0,           // Inside terminal panel (was 220)
            y: 12.0,            // From bottom
            width: 0.0,         // Will be calculated
            height: 160.0,      // Match HUD panel (was 220)
            font_size: 13.0,    // Slightly smaller for denser text
            line_height: 1.25,
            bg_color: [0.0, 0.0, 0.0, 0.0], // Transparent - HUD provides background
            border_color: [0.0, 0.0, 0.0, 0.0], // No border - HUD provides it
            padding: 8.0,
        }
    }
}

/// Terminal vertex for background/border
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TerminalVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

/// Character instance for instanced text rendering
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CharInstance {
    /// Position (x, y) and size (w, h)
    pub rect: [f32; 4],
    /// Atlas UV (u, v, u_size, v_size)
    pub uv: [f32; 4],
    /// Color RGBA
    pub color: [f32; 4],
}

/// GPU-rendered terminal
pub struct TerminalRenderer {
    // Background rendering
    bg_pipeline: wgpu::RenderPipeline,
    bg_vertex_buffer: wgpu::Buffer,
    bg_uniform_buffer: wgpu::Buffer,
    bg_bind_group: wgpu::BindGroup,
    
    // Text rendering
    text_pipeline: wgpu::RenderPipeline,
    text_instance_buffer: wgpu::Buffer,
    text_bind_group: wgpu::BindGroup,
    atlas_texture: wgpu::Texture,
    
    // Configuration
    config: TerminalConfig,
    screen_width: f32,
    screen_height: f32,
    
    // Text layout cache
    max_chars: usize,
    lines_visible: usize,
}

/// Background uniforms - must match WGSL alignment (64 bytes)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BgUniforms {
    rect: [f32; 4],         // 16 bytes, offset 0
    bg_color: [f32; 4],     // 16 bytes, offset 16
    border_color: [f32; 4], // 16 bytes, offset 32
    border_width: [f32; 4], // 16 bytes, offset 48 (only .x used, rest padding)
}

impl TerminalRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let screen_width = config.width as f32;
        let screen_height = config.height as f32;
        
        // Calculate width to span between rails
        let term_config = TerminalConfig {
            width: screen_width - 440.0, // 220px each side
            ..TerminalConfig::default()
        };
        
        let lines_visible = ((term_config.height - term_config.padding * 2.0) 
            / (term_config.font_size * term_config.line_height)) as usize;
        let max_chars = 150 * lines_visible; // ~150 chars per line max
        
        // Create shaders
        let bg_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terminal BG Shader"),
            source: wgpu::ShaderSource::Wgsl(TERMINAL_BG_SHADER.into()),
        });
        
        let text_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terminal Text Shader"),
            source: wgpu::ShaderSource::Wgsl(TERMINAL_TEXT_SHADER.into()),
        });
        
        // Background pipeline
        let bg_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terminal BG Layout"),
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
        
        let bg_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terminal BG Pipeline Layout"),
            bind_group_layouts: &[&bg_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terminal BG Pipeline"),
            layout: Some(&bg_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bg_shader,
                entry_point: "vs_main",
                buffers: &[],
                
            },
            fragment: Some(wgpu::FragmentState {
                module: &bg_shader,
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
        
        // Background vertex buffer (quad)
        let bg_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terminal BG Vertices"),
            contents: bytemuck::cast_slice(&[
                TerminalVertex { position: [0.0, 0.0], uv: [0.0, 0.0] },
                TerminalVertex { position: [1.0, 0.0], uv: [1.0, 0.0] },
                TerminalVertex { position: [0.0, 1.0], uv: [0.0, 1.0] },
                TerminalVertex { position: [1.0, 1.0], uv: [1.0, 1.0] },
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let bg_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terminal BG Uniforms"),
            contents: bytemuck::cast_slice(&[BgUniforms {
                rect: [0.0; 4],
                bg_color: term_config.bg_color,
                border_color: term_config.border_color,
                border_width: [1.0, 0.0, 0.0, 0.0],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terminal BG Bind Group"),
            layout: &bg_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bg_uniform_buffer.as_entire_binding(),
            }],
        });
        
        // Text atlas texture (256x256, simple bitmap font)
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terminal Font Atlas"),
            size: wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Generate simple font atlas
        let atlas_data = generate_font_atlas();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
        );
        
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Terminal Font Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Text pipeline
        let text_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terminal Text Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        
        let text_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terminal Text Pipeline Layout"),
            bind_group_layouts: &[&text_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let text_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terminal Text Pipeline"),
            layout: Some(&text_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &text_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CharInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 2,
                        },
                    ],
                }],
                
            },
            fragment: Some(wgpu::FragmentState {
                module: &text_shader,
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
        
        let text_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terminal Text Instances"),
            size: (max_chars * std::mem::size_of::<CharInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let text_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terminal Text Bind Group"),
            layout: &text_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });
        
        Self {
            bg_pipeline,
            bg_vertex_buffer,
            bg_uniform_buffer,
            bg_bind_group,
            text_pipeline,
            text_instance_buffer,
            text_bind_group,
            atlas_texture,
            config: term_config,
            screen_width,
            screen_height,
            max_chars,
            lines_visible,
        }
    }
    
    /// Render terminal with event log
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        event_log: &EventLog,
    ) {
        // Update background uniforms
        let ndc_x = (self.config.x / self.screen_width) * 2.0 - 1.0;
        let ndc_y = -1.0 + (self.config.y / self.screen_height) * 2.0;
        let ndc_w = (self.config.width / self.screen_width) * 2.0;
        let ndc_h = (self.config.height / self.screen_height) * 2.0;
        
        let bg_uniforms = BgUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            bg_color: self.config.bg_color,
            border_color: self.config.border_color,
            border_width: [1.0, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.bg_uniform_buffer, 0, bytemuck::cast_slice(&[bg_uniforms]));
        
        // Render background
        render_pass.set_pipeline(&self.bg_pipeline);
        render_pass.set_bind_group(0, &self.bg_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
        
        // Build character instances from visible events
        let visible = event_log.visible_events();
        let mut instances: Vec<CharInstance> = Vec::with_capacity(self.max_chars);
        
        let char_width = self.config.font_size * 0.6;
        let line_height = self.config.font_size * self.config.line_height;
        let start_x = self.config.x + self.config.padding;
        let start_y = self.screen_height - self.config.y - self.config.height + self.config.padding;
        
        for (line_idx, event) in visible.iter().enumerate() {
            let y = start_y + (line_idx as f32) * line_height;
            let text = event.formatted();
            let color = event.level.color();
            
            for (char_idx, ch) in text.chars().enumerate() {
                if char_idx >= 150 { break; } // Max chars per line
                
                let x = start_x + (char_idx as f32) * char_width;
                
                // Skip non-printable chars
                let code = ch as u32;
                if !(32..=126).contains(&code) {
                    continue;
                }
                
                let instance = char_to_instance(
                    x, y,
                    self.config.font_size,
                    ch,
                    color,
                    self.screen_width,
                    self.screen_height,
                );
                instances.push(instance);
            }
        }
        
        if instances.is_empty() {
            return;
        }
        
        // Upload instances
        queue.write_buffer(&self.text_instance_buffer, 0, bytemuck::cast_slice(&instances));
        
        // Render text
        render_pass.set_pipeline(&self.text_pipeline);
        render_pass.set_bind_group(0, &self.text_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.text_instance_buffer.slice(..));
        render_pass.draw(0..4, 0..instances.len() as u32);
    }
    
    /// Set visible lines based on terminal height
    pub fn set_height(&mut self, height: f32) {
        self.config.height = height;
        self.lines_visible = ((height - self.config.padding * 2.0) 
            / (self.config.font_size * self.config.line_height)) as usize;
    }
    
    pub fn lines_visible(&self) -> usize {
        self.lines_visible
    }
}

/// Convert character to instance
fn char_to_instance(
    x: f32, y: f32,
    size: f32,
    ch: char,
    color: [f32; 4],
    screen_w: f32,
    screen_h: f32,
) -> CharInstance {
    // Atlas: 16x16 grid, each cell 16x16 pixels
    let code = (ch as u32).saturating_sub(32).min(94);
    let atlas_x = (code % 16) as f32 * 16.0;
    let atlas_y = (code / 16) as f32 * 16.0;
    
    // Convert to NDC
    let ndc_x = (x / screen_w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (y / screen_h) * 2.0;
    let ndc_w = (size * 0.6 / screen_w) * 2.0;
    let ndc_h = (size / screen_h) * 2.0;
    
    CharInstance {
        rect: [ndc_x, ndc_y, ndc_w, ndc_h],
        uv: [atlas_x / 256.0, atlas_y / 256.0, 16.0 / 256.0, 16.0 / 256.0],
        color,
    }
}

/// Generate simple bitmap font atlas (8x8 glyphs, 16x16 cells)
fn generate_font_atlas() -> Vec<u8> {
    let mut atlas = vec![0u8; 256 * 256];
    
    // Simple 5x7 font glyphs for ASCII 32-126
    // This is a minimal implementation - real apps would use a proper font
    let glyphs = include_font_data();
    
    for (idx, glyph) in glyphs.iter().enumerate() {
        let cell_x = (idx % 16) * 16;
        let cell_y = (idx / 16) * 16;
        
        for (row, &glyph_row) in glyph.iter().enumerate() {
            for col in 0..8 {
                let bit = (glyph_row >> (7 - col)) & 1;
                if bit != 0 {
                    let ax = cell_x + col + 4; // Center in cell
                    let ay = cell_y + row + 4;
                    if ax < 256 && ay < 256 {
                        atlas[ay * 256 + ax] = 255;
                    }
                }
            }
        }
    }
    
    atlas
}

/// Simple 8x8 font data for printable ASCII (32-126)
fn include_font_data() -> [[u8; 8]; 95] {
    // Basic 8x8 font - each byte is a row, MSB is leftmost pixel
    [
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // Space
        [0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00], // !
        [0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00], // "
        [0x6C, 0xFE, 0x6C, 0x6C, 0xFE, 0x6C, 0x00, 0x00], // #
        [0x18, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x18, 0x00], // $
        [0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00], // %
        [0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00], // &
        [0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00], // '
        [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00], // (
        [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00], // )
        [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // *
        [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00], // +
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30], // ,
        [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00], // -
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00], // .
        [0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00], // /
        [0x7C, 0xC6, 0xCE, 0xD6, 0xE6, 0xC6, 0x7C, 0x00], // 0
        [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00], // 1
        [0x7C, 0xC6, 0x06, 0x1C, 0x30, 0x66, 0xFE, 0x00], // 2
        [0x7C, 0xC6, 0x06, 0x3C, 0x06, 0xC6, 0x7C, 0x00], // 3
        [0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x1E, 0x00], // 4
        [0xFE, 0xC0, 0xFC, 0x06, 0x06, 0xC6, 0x7C, 0x00], // 5
        [0x38, 0x60, 0xC0, 0xFC, 0xC6, 0xC6, 0x7C, 0x00], // 6
        [0xFE, 0xC6, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00], // 7
        [0x7C, 0xC6, 0xC6, 0x7C, 0xC6, 0xC6, 0x7C, 0x00], // 8
        [0x7C, 0xC6, 0xC6, 0x7E, 0x06, 0x0C, 0x78, 0x00], // 9
        [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00], // :
        [0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30], // ;
        [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // <
        [0x00, 0x00, 0x7E, 0x00, 0x00, 0x7E, 0x00, 0x00], // =
        [0x60, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x60, 0x00], // >
        [0x7C, 0xC6, 0x0C, 0x18, 0x18, 0x00, 0x18, 0x00], // ?
        [0x7C, 0xC6, 0xDE, 0xDE, 0xDC, 0xC0, 0x7C, 0x00], // @
        [0x38, 0x6C, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0x00], // A
        [0xFC, 0x66, 0x66, 0x7C, 0x66, 0x66, 0xFC, 0x00], // B
        [0x3C, 0x66, 0xC0, 0xC0, 0xC0, 0x66, 0x3C, 0x00], // C
        [0xF8, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0xF8, 0x00], // D
        [0xFE, 0x62, 0x68, 0x78, 0x68, 0x62, 0xFE, 0x00], // E
        [0xFE, 0x62, 0x68, 0x78, 0x68, 0x60, 0xF0, 0x00], // F
        [0x3C, 0x66, 0xC0, 0xC0, 0xCE, 0x66, 0x3E, 0x00], // G
        [0xC6, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00], // H
        [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00], // I
        [0x1E, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78, 0x00], // J
        [0xE6, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0xE6, 0x00], // K
        [0xF0, 0x60, 0x60, 0x60, 0x62, 0x66, 0xFE, 0x00], // L
        [0xC6, 0xEE, 0xFE, 0xFE, 0xD6, 0xC6, 0xC6, 0x00], // M
        [0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00], // N
        [0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00], // O
        [0xFC, 0x66, 0x66, 0x7C, 0x60, 0x60, 0xF0, 0x00], // P
        [0x7C, 0xC6, 0xC6, 0xC6, 0xD6, 0xDE, 0x7C, 0x0E], // Q
        [0xFC, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0xE6, 0x00], // R
        [0x7C, 0xC6, 0x60, 0x38, 0x0C, 0xC6, 0x7C, 0x00], // S
        [0x7E, 0x7E, 0x5A, 0x18, 0x18, 0x18, 0x3C, 0x00], // T
        [0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00], // U
        [0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00], // V
        [0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00], // W
        [0xC6, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0xC6, 0x00], // X
        [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x3C, 0x00], // Y
        [0xFE, 0xC6, 0x8C, 0x18, 0x32, 0x66, 0xFE, 0x00], // Z
        [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00], // [
        [0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00], // \
        [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00], // ]
        [0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00], // ^
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // _
        [0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00], // `
        [0x00, 0x00, 0x78, 0x0C, 0x7C, 0xCC, 0x76, 0x00], // a
        [0xE0, 0x60, 0x7C, 0x66, 0x66, 0x66, 0xDC, 0x00], // b
        [0x00, 0x00, 0x7C, 0xC6, 0xC0, 0xC6, 0x7C, 0x00], // c
        [0x1C, 0x0C, 0x7C, 0xCC, 0xCC, 0xCC, 0x76, 0x00], // d
        [0x00, 0x00, 0x7C, 0xC6, 0xFE, 0xC0, 0x7C, 0x00], // e
        [0x38, 0x6C, 0x60, 0xF8, 0x60, 0x60, 0xF0, 0x00], // f
        [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0xF8], // g
        [0xE0, 0x60, 0x6C, 0x76, 0x66, 0x66, 0xE6, 0x00], // h
        [0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00], // i
        [0x06, 0x00, 0x0E, 0x06, 0x06, 0x66, 0x66, 0x3C], // j
        [0xE0, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0xE6, 0x00], // k
        [0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00], // l
        [0x00, 0x00, 0xEC, 0xFE, 0xD6, 0xD6, 0xD6, 0x00], // m
        [0x00, 0x00, 0xDC, 0x66, 0x66, 0x66, 0x66, 0x00], // n
        [0x00, 0x00, 0x7C, 0xC6, 0xC6, 0xC6, 0x7C, 0x00], // o
        [0x00, 0x00, 0xDC, 0x66, 0x66, 0x7C, 0x60, 0xF0], // p
        [0x00, 0x00, 0x76, 0xCC, 0xCC, 0x7C, 0x0C, 0x1E], // q
        [0x00, 0x00, 0xDC, 0x76, 0x60, 0x60, 0xF0, 0x00], // r
        [0x00, 0x00, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x00], // s
        [0x30, 0x30, 0xFC, 0x30, 0x30, 0x36, 0x1C, 0x00], // t
        [0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00], // u
        [0x00, 0x00, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00], // v
        [0x00, 0x00, 0xC6, 0xD6, 0xD6, 0xFE, 0x6C, 0x00], // w
        [0x00, 0x00, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0x00], // x
        [0x00, 0x00, 0xC6, 0xC6, 0xC6, 0x7E, 0x06, 0xFC], // y
        [0x00, 0x00, 0xFE, 0x8C, 0x18, 0x32, 0xFE, 0x00], // z
        [0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00], // {
        [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // |
        [0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00], // }
        [0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ~
    ]
}

// ============================================================================
// WGSL Shaders
// ============================================================================

const TERMINAL_BG_SHADER: &str = r#"
struct BgUniforms {
    rect: vec4<f32>,
    bg_color: vec4<f32>,
    border_color: vec4<f32>,
    border_width: vec4<f32>,  // Only .x used, rest is padding
}

@group(0) @binding(0)
var<uniform> uniforms: BgUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    
    let pos = positions[vi];
    let x = uniforms.rect.x + pos.x * uniforms.rect.z;
    let y = uniforms.rect.y + pos.y * uniforms.rect.w;
    
    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = pos;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    
    // Border detection (in UV space)
    let border = 0.02;
    let in_border = uv.x < border || uv.x > (1.0 - border) || 
                   uv.y < border || uv.y > (1.0 - border);
    
    if in_border {
        return uniforms.border_color;
    }
    
    return uniforms.bg_color;
}
"#;

const TERMINAL_TEXT_SHADER: &str = r#"
@group(0) @binding(0)
var t_atlas: texture_2d<f32>;
@group(0) @binding(1)
var s_atlas: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,
    @location(1) atlas_uv: vec4<f32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    
    let pos = positions[vi];
    let x = rect.x + pos.x * rect.z;
    let y = rect.y - pos.y * rect.w;
    
    let u = atlas_uv.x + pos.x * atlas_uv.z;
    let v = atlas_uv.y + pos.y * atlas_uv.w;
    
    var output: VertexOutput;
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>(u, v);
    output.color = color;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = textureSample(t_atlas, s_atlas, input.uv).r;
    return vec4<f32>(input.color.rgb, input.color.a * alpha);
}
"#;

/*!
 * HUD Overlay - Unified Heads-Up Display System
 * 
 * Combines all telemetry elements into a cohesive cockpit HUD:
 * - Glass panels with SDF borders
 * - Labeled metric bars with values
 * - Text rendering for labels
 * 
 * Constitutional: Article V GPU mandate, Doctrine 1 procedural
 */
#![allow(dead_code)] // HUD API ready for integration

use wgpu::util::DeviceExt;

/// HUD element types
#[derive(Clone, Copy)]
pub enum HudElement {
    LeftRail,
    RightRail,
    BottomTerminal,
    ProbePanel,
}

/// Unified HUD overlay renderer
/// NOTE: Panel/container rendering is delegated to GlassChrome (Layer 5)
/// HudOverlay only renders DATA elements: bars, crosshair, text (Layer 6)
pub struct HudOverlay {
    // Bar rendering (data elements)
    bar_pipeline: wgpu::RenderPipeline,
    bar_uniform_buffer: wgpu::Buffer,
    bar_bind_group: wgpu::BindGroup,
    
    // Text rendering
    text_pipeline: wgpu::RenderPipeline,
    text_uniform_buffer: wgpu::Buffer,
    text_bind_group: wgpu::BindGroup,
    glyph_texture: wgpu::Texture,
    glyph_sampler: wgpu::Sampler,
    
    screen_width: f32,
    screen_height: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BarUniforms {
    rect: [f32; 4],           // x, y, w, h in NDC
    fill_color: [f32; 4],     // RGBA fill color
    bg_color: [f32; 4],       // RGBA background
    params: [f32; 4],         // fill_amount, corner_radius, 0, 0
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TextUniforms {
    position: [f32; 4],       // x, y, scale, 0
    color: [f32; 4],          // RGBA
    char_data: [u32; 4],      // char_code, 0, 0, 0
}

impl HudOverlay {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: &wgpu::SurfaceConfiguration) -> Self {
        let screen_width = config.width as f32;
        let screen_height = config.height as f32;
        
        // Create glyph atlas texture (16x6 grid of 8x8 characters = 128x48)
        let glyph_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HUD Glyph Atlas"),
            size: wgpu::Extent3d { width: 128, height: 48, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Generate and upload glyph data
        let glyph_data = Self::generate_glyph_atlas();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &glyph_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &glyph_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: Some(48),
            },
            wgpu::Extent3d { width: 128, height: 48, depth_or_array_layers: 1 },
        );
        
        let glyph_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HUD Glyph Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Shared blend state
        let blend_state = wgpu::BlendState {
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
        };
        
        // NOTE: Panel/container rendering is delegated to GlassChrome (Layer 5)
        // HudOverlay only handles data elements (bars, crosshair, text)
        
        // Bar pipeline for data elements
        let bar_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HUD Bar Shader"),
            source: wgpu::ShaderSource::Wgsl(BAR_SHADER.into()),
        });
        
        let bar_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HUD Bar Bind Group Layout"),
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
        
        let bar_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HUD Bar Pipeline Layout"),
            bind_group_layouts: &[&bar_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let bar_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HUD Bar Pipeline"),
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
        
        let bar_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HUD Bar Uniforms"),
            contents: bytemuck::cast_slice(&[BarUniforms {
                rect: [0.0; 4],
                fill_color: [0.0; 4],
                bg_color: [0.0; 4],
                params: [0.0; 4],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bar_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HUD Bar Bind Group"),
            layout: &bar_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bar_uniform_buffer.as_entire_binding(),
            }],
        });
        
        // Text pipeline
        let text_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HUD Text Shader"),
            source: wgpu::ShaderSource::Wgsl(TEXT_SHADER.into()),
        });
        
        let glyph_view = glyph_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let text_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HUD Text Bind Group Layout"),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let text_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HUD Text Pipeline Layout"),
            bind_group_layouts: &[&text_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let text_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HUD Text Pipeline"),
            layout: Some(&text_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &text_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &text_shader,
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
        
        let text_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HUD Text Uniforms"),
            contents: bytemuck::cast_slice(&[TextUniforms {
                position: [0.0; 4],
                color: [0.0; 4],
                char_data: [0; 4],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let text_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HUD Text Bind Group"),
            layout: &text_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: text_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&glyph_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&glyph_sampler),
                },
            ],
        });
        
        Self {
            bar_pipeline,
            bar_uniform_buffer,
            bar_bind_group,
            text_pipeline,
            text_uniform_buffer,
            text_bind_group,
            glyph_texture,
            glyph_sampler,
            screen_width,
            screen_height,
        }
    }
    
    /// Generate 8x8 glyph atlas for ASCII 32-127
    fn generate_glyph_atlas() -> Vec<u8> {
        // 16 chars per row, 6 rows = 96 chars (32-127)
        // Each char is 8x8 pixels
        let mut atlas = vec![0u8; 128 * 48];
        
        // Simple block-based font for key characters
        let font_data: [(char, [u8; 8]); 43] = [
            ('0', [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00]),
            ('1', [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00]),
            ('2', [0x3C, 0x66, 0x60, 0x30, 0x18, 0x0C, 0x7E, 0x00]),
            ('3', [0x3C, 0x66, 0x60, 0x38, 0x60, 0x66, 0x3C, 0x00]),
            ('4', [0x30, 0x38, 0x34, 0x32, 0x7E, 0x30, 0x30, 0x00]),
            ('5', [0x7E, 0x06, 0x3E, 0x60, 0x60, 0x66, 0x3C, 0x00]),
            ('6', [0x38, 0x0C, 0x06, 0x3E, 0x66, 0x66, 0x3C, 0x00]),
            ('7', [0x7E, 0x60, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00]),
            ('8', [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00]),
            ('9', [0x3C, 0x66, 0x66, 0x7C, 0x60, 0x30, 0x1C, 0x00]),
            ('A', [0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x00]),
            ('B', [0x3E, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3E, 0x00]),
            ('C', [0x3C, 0x66, 0x06, 0x06, 0x06, 0x66, 0x3C, 0x00]),
            ('D', [0x1E, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1E, 0x00]),
            ('E', [0x7E, 0x06, 0x06, 0x3E, 0x06, 0x06, 0x7E, 0x00]),
            ('F', [0x7E, 0x06, 0x06, 0x3E, 0x06, 0x06, 0x06, 0x00]),
            ('G', [0x3C, 0x66, 0x06, 0x76, 0x66, 0x66, 0x3C, 0x00]),
            ('M', [0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00]),
            ('P', [0x3E, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x06, 0x00]),
            ('R', [0x3E, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x66, 0x00]),
            ('S', [0x3C, 0x66, 0x06, 0x3C, 0x60, 0x66, 0x3C, 0x00]),
            ('T', [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00]),
            ('U', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00]),
            ('W', [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00]),
            (':', [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00]),
            ('.', [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00]),
            ('%', [0x46, 0xA6, 0x4C, 0x18, 0x32, 0x65, 0x62, 0x00]),
            ('/', [0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00]),
            (' ', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            ('m', [0x00, 0x00, 0x76, 0x7F, 0x6B, 0x6B, 0x6B, 0x00]),
            ('s', [0x00, 0x00, 0x3C, 0x06, 0x3C, 0x60, 0x3E, 0x00]),
            ('I', [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00]),
            ('N', [0x66, 0x6E, 0x7E, 0x7E, 0x76, 0x66, 0x66, 0x00]),
            ('D', [0x1E, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1E, 0x00]),
            ('O', [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00]),
            ('V', [0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00]),
            ('H', [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00]),
            ('K', [0x66, 0x36, 0x1E, 0x0E, 0x1E, 0x36, 0x66, 0x00]),
            ('L', [0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x7E, 0x00]),
            ('b', [0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00]),
            ('a', [0x00, 0x00, 0x3C, 0x60, 0x7C, 0x66, 0x7C, 0x00]),
            ('r', [0x00, 0x00, 0x36, 0x0E, 0x06, 0x06, 0x06, 0x00]),
            ('-', [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00]),
        ];
        
        for (c, glyph) in font_data.iter() {
            let idx = (*c as usize).saturating_sub(32);
            if idx < 96 {
                let col = idx % 16;
                let row = idx / 16;
                for y in 0..8 {
                    for x in 0..8 {
                        if (glyph[y] & (1 << (7 - x))) != 0 {
                            atlas[(row * 8 + y) * 128 + col * 8 + x] = 255;
                        }
                    }
                }
            }
        }
        
        atlas
    }
    
    // NOTE: render_panel has been REMOVED - panel containers are rendered by GlassChrome (Layer 5)
    // HudOverlay only renders data elements (bars, crosshair, text) on Layer 6
    
    /// Render a metric bar with label
    #[allow(clippy::too_many_arguments)]  // Render geometry params
    pub fn render_bar<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        x: f32, y: f32, w: f32, h: f32,
        fill: f32,
        color: [f32; 3],
    ) {
        let ndc_x = (x / self.screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / self.screen_height) * 2.0;
        let ndc_w = (w / self.screen_width) * 2.0;
        let ndc_h = (h / self.screen_height) * 2.0;
        
        let uniforms = BarUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            fill_color: [color[0], color[1], color[2], 1.0],
            bg_color: [0.05, 0.05, 0.08, 0.9],
            params: [fill.clamp(0.0, 1.0), 4.0, 0.0, 0.0],
        };
        
        queue.write_buffer(&self.bar_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        render_pass.set_pipeline(&self.bar_pipeline);
        render_pass.set_bind_group(0, &self.bar_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
    
    /// Render a labeled metric bar (bar with small label above)
    #[allow(clippy::too_many_arguments)]  // Render geometry + text params
    pub fn render_labeled_bar<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        x: f32, y: f32, w: f32, h: f32,
        fill: f32,
        color: [f32; 3],
        label: &str,
        value_str: &str,
    ) {
        // Render bar
        self.render_bar(render_pass, queue, x, y, w, h, fill, color);
        
        // Render label text as small bar segments (ASCII art style)
        // For now, just render a small indicator on the left
        let label_w = 4.0;
        let label_h = h - 4.0;
        let label_x = x + 2.0;
        let label_y = y + 2.0;
        
        // Label indicator - brighter version of bar color
        let label_color = [
            (color[0] * 1.3).min(1.0),
            (color[1] * 1.3).min(1.0),
            (color[2] * 1.3).min(1.0),
        ];
        
        let ndc_x = (label_x / self.screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (label_y / self.screen_height) * 2.0;
        let ndc_w = (label_w / self.screen_width) * 2.0;
        let ndc_h = (label_h / self.screen_height) * 2.0;
        
        let uniforms = BarUniforms {
            rect: [ndc_x, ndc_y, ndc_w, ndc_h],
            fill_color: [label_color[0], label_color[1], label_color[2], 1.0],
            bg_color: [0.0, 0.0, 0.0, 0.0],
            params: [1.0, 0.0, 0.0, 0.0],
        };
        
        queue.write_buffer(&self.bar_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        render_pass.set_pipeline(&self.bar_pipeline);
        render_pass.set_bind_group(0, &self.bar_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
        
        // Note: Actual text rendering would use the text pipeline with glyph atlas
        // For now, the label indicator provides visual distinction
        let _ = (label, value_str); // Suppress unused warnings
    }
    
    /// Render full HUD with all elements
    #[allow(clippy::too_many_arguments)]  // HUD requires all telemetry values
    pub fn render_full_hud<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        screen_w: f32,
        screen_h: f32,
        cpu: f32,
        memory: f32,
        fps: f32,
        frame_ms: f32,
        temp: f32,
        wind: f32,
        convergence: f32,
        pressure: f32,
    ) {
        let left_x = 12.0;
        let right_x = screen_w - 200.0;
        let bar_w = 176.0;
        let bar_h = 26.0;
        let spacing = 40.0;
        
        // === HUD FRAME ELEMENTS ===
        // Top-left corner bracket
        self.render_bar(render_pass, queue, 4.0, 4.0, 40.0, 3.0, 1.0, [0.0, 0.7, 0.9]);
        self.render_bar(render_pass, queue, 4.0, 4.0, 3.0, 40.0, 1.0, [0.0, 0.7, 0.9]);
        
        // Top-right corner bracket
        self.render_bar(render_pass, queue, screen_w - 44.0, 4.0, 40.0, 3.0, 1.0, [0.0, 0.7, 0.9]);
        self.render_bar(render_pass, queue, screen_w - 7.0, 4.0, 3.0, 40.0, 1.0, [0.0, 0.7, 0.9]);
        
        // Bottom-left corner bracket
        self.render_bar(render_pass, queue, 4.0, screen_h - 7.0, 40.0, 3.0, 1.0, [0.0, 0.7, 0.9]);
        self.render_bar(render_pass, queue, 4.0, screen_h - 44.0, 3.0, 40.0, 1.0, [0.0, 0.7, 0.9]);
        
        // Bottom-right corner bracket
        self.render_bar(render_pass, queue, screen_w - 44.0, screen_h - 7.0, 40.0, 3.0, 1.0, [0.0, 0.7, 0.9]);
        self.render_bar(render_pass, queue, screen_w - 7.0, screen_h - 44.0, 3.0, 40.0, 1.0, [0.0, 0.7, 0.9]);
        
        // === PANEL CONTAINERS RENDERED BY GLASS_CHROME (LAYER 5) ===
        // HudOverlay only renders DATA elements on top of glass panels
        // Left Rail, Right Rail, and Terminal Panel are handled by glass_chrome.rs
        
        // Left rail bars
        let mut y = 38.0;
        
        // CPU
        self.render_labeled_bar(render_pass, queue, left_x, y, bar_w, bar_h, cpu, 
            [0.0, 0.85, 1.0], "CPU", &format!("{:.0}%", cpu * 100.0));
        y += spacing;
        
        // Memory
        self.render_labeled_bar(render_pass, queue, left_x, y, bar_w, bar_h, memory, 
            [1.0, 0.5, 0.0], "MEM", &format!("{:.0}%", memory * 100.0));
        y += spacing;
        
        // FPS (normalized to 120)
        self.render_labeled_bar(render_pass, queue, left_x, y, bar_w, bar_h, fps / 120.0, 
            [0.0, 1.0, 0.4], "FPS", &format!("{:.0}", fps));
        y += spacing;
        
        // Frame time (normalized to 33ms)
        self.render_labeled_bar(render_pass, queue, left_x, y, bar_w, bar_h, frame_ms / 33.0, 
            [1.0, 1.0, 0.0], "FRM", &format!("{:.1}ms", frame_ms));
        
        // Right rail bars
        y = 40.0;
        
        // Temperature
        self.render_labeled_bar(render_pass, queue, right_x, y, bar_w, bar_h, temp, 
            [1.0, 0.3, 0.1], "TEMP", "290K");
        y += spacing;
        
        // Wind
        self.render_labeled_bar(render_pass, queue, right_x, y, bar_w, bar_h, wind, 
            [0.4, 0.7, 1.0], "WIND", "25m/s");
        y += spacing;
        
        // Convergence
        self.render_labeled_bar(render_pass, queue, right_x, y, bar_w, bar_h, convergence, 
            [1.0, 0.2, 0.8], "CONV", "0.45");
        y += spacing;
        
        // Pressure
        self.render_labeled_bar(render_pass, queue, right_x, y, bar_w, bar_h, pressure, 
            [0.6, 0.3, 1.0], "PRES", "1013hPa");
    }
    
    /// Render bottom telemetry bar chart (OPERATION VALHALLA style)
    /// Displays historical data as vertical bars across the bottom of the screen
    pub fn render_bottom_telemetry<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        screen_w: f32,
        screen_h: f32,
        time: f32,
    ) {
        // Bar chart configuration
        let bar_count = 48;
        let chart_x = screen_w * 0.25;           // Start at 25% width
        let chart_w = screen_w * 0.50;           // 50% of screen width
        let chart_y = screen_h - 60.0;           // 60px from bottom
        let max_bar_h = 40.0;
        let bar_width = chart_w / bar_count as f32 * 0.75;
        let bar_gap = chart_w / bar_count as f32 * 0.25;
        
        // Generate pseudo-random telemetry data (seeded by time for animation)
        for i in 0..bar_count {
            let phase = (i as f32 * 0.3 + time * 0.5).sin() * 0.5 + 0.5;
            let noise = ((i as f32 * 7.3 + time * 2.0).sin() * 0.2).abs();
            let value = (phase + noise).clamp(0.1, 1.0);
            
            let x = chart_x + i as f32 * (bar_width + bar_gap);
            let bar_h = value * max_bar_h;
            
            // Color gradient: low = cyan, high = orange/yellow
            let color = if value < 0.5 {
                [0.2, 0.6 + value, 0.9 - value * 0.5]  // Cyan to green
            } else {
                [0.5 + value * 0.5, 0.7 - value * 0.3, 0.2]  // Yellow to orange
            };
            
            // Render bar from bottom up
            self.render_bar(render_pass, queue, 
                x, chart_y + max_bar_h - bar_h, 
                bar_width, bar_h, 
                1.0, color);
        }
        
        // Axis line at bottom
        self.render_bar(render_pass, queue, 
            chart_x - 5.0, chart_y + max_bar_h + 2.0, 
            chart_w + 10.0, 1.0, 
            1.0, [0.3, 0.4, 0.5]);
    }
    
    /// Render probe panel (shows on hover/selection)
    #[allow(clippy::too_many_arguments)]  // Probe requires position + data
    pub fn render_probe_panel<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        screen_w: f32,
        screen_h: f32,
        mouse_x: f32,
        mouse_y: f32,
        lat: f32,
        lon: f32,
        value: f32,
        label: &str,
    ) {
        // Position probe panel near cursor but offset to not obscure
        let panel_w = 160.0;
        let panel_h = 80.0;
        let offset_x = 20.0;
        let offset_y = -40.0;
        
        // Keep panel on screen
        let mut x = mouse_x + offset_x;
        let mut y = mouse_y + offset_y;
        if x + panel_w > screen_w - 10.0 {
            x = mouse_x - panel_w - offset_x;
        }
        if y < 10.0 {
            y = 10.0;
        }
        if y + panel_h > screen_h - 10.0 {
            y = screen_h - panel_h - 10.0;
        }
        
        // NOTE: Panel backdrop is rendered by GlassChrome if needed
        // HudOverlay only renders the data elements within the probe area
        
        // Value indicator bar
        let value_norm = value.clamp(0.0, 1.0);
        self.render_bar(render_pass, queue, x + 8.0, y + 50.0, panel_w - 16.0, 20.0, value_norm, [0.0, 0.9, 0.6]);
        
        // Crosshair at cursor position
        let ch_size = 12.0;
        self.render_bar(render_pass, queue, mouse_x - ch_size, mouse_y - 1.0, ch_size * 2.0, 2.0, 1.0, [0.2, 0.9, 1.0]);
        self.render_bar(render_pass, queue, mouse_x - 1.0, mouse_y - ch_size, 2.0, ch_size * 2.0, 1.0, [0.2, 0.9, 1.0]);
        
        // Text rendering pending glyph pipeline integration (Phase 8)
        let _ = (lat, lon, label);
    }
    
    /// Render targeting crosshair at mouse position
    pub fn render_crosshair<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        queue: &wgpu::Queue,
        mouse_x: f32,
        mouse_y: f32,
    ) {
        let ch_size = 16.0;
        let gap = 4.0;
        
        // Horizontal lines with gap
        self.render_bar(render_pass, queue, mouse_x - ch_size, mouse_y - 1.0, ch_size - gap, 2.0, 1.0, [0.0, 0.8, 1.0]);
        self.render_bar(render_pass, queue, mouse_x + gap, mouse_y - 1.0, ch_size - gap, 2.0, 1.0, [0.0, 0.8, 1.0]);
        
        // Vertical lines with gap
        self.render_bar(render_pass, queue, mouse_x - 1.0, mouse_y - ch_size, 2.0, ch_size - gap, 1.0, [0.0, 0.8, 1.0]);
        self.render_bar(render_pass, queue, mouse_x - 1.0, mouse_y + gap, 2.0, ch_size - gap, 1.0, [0.0, 0.8, 1.0]);
        
        // Corner brackets
        let bracket = 6.0;
        // Top-left
        self.render_bar(render_pass, queue, mouse_x - ch_size - 2.0, mouse_y - ch_size - 2.0, bracket, 1.5, 1.0, [0.0, 0.8, 1.0]);
        self.render_bar(render_pass, queue, mouse_x - ch_size - 2.0, mouse_y - ch_size - 2.0, 1.5, bracket, 1.0, [0.0, 0.8, 1.0]);
        // Top-right
        self.render_bar(render_pass, queue, mouse_x + ch_size - bracket + 2.0, mouse_y - ch_size - 2.0, bracket, 1.5, 1.0, [0.0, 0.8, 1.0]);
        self.render_bar(render_pass, queue, mouse_x + ch_size + 0.5, mouse_y - ch_size - 2.0, 1.5, bracket, 1.0, [0.0, 0.8, 1.0]);
        // Bottom-left
        self.render_bar(render_pass, queue, mouse_x - ch_size - 2.0, mouse_y + ch_size + 0.5, bracket, 1.5, 1.0, [0.0, 0.8, 1.0]);
        self.render_bar(render_pass, queue, mouse_x - ch_size - 2.0, mouse_y + ch_size - bracket + 2.0, 1.5, bracket, 1.0, [0.0, 0.8, 1.0]);
        // Bottom-right
        self.render_bar(render_pass, queue, mouse_x + ch_size - bracket + 2.0, mouse_y + ch_size + 0.5, bracket, 1.5, 1.0, [0.0, 0.8, 1.0]);
        self.render_bar(render_pass, queue, mouse_x + ch_size + 0.5, mouse_y + ch_size - bracket + 2.0, 1.5, bracket, 1.0, [0.0, 0.8, 1.0]);
    }
}

// ============================================================================
// WGSL Shaders - DATA ELEMENTS ONLY (bars, text)
// NOTE: Panel shaders removed - containers rendered by GlassChrome (Layer 5)
// ============================================================================

const BAR_SHADER: &str = r#"
struct BarUniforms {
    rect: vec4<f32>,
    fill_color: vec4<f32>,
    bg_color: vec4<f32>,
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: BarUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0)
    );
    let p = pos[idx];
    var out: VertexOutput;
    out.position = vec4(u.rect.x + p.x * u.rect.z, u.rect.y - p.y * u.rect.w, 0.0, 1.0);
    out.uv = p;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let fill = u.params.x;
    
    // Dark background
    var color = u.bg_color;
    
    // Bright fill portion
    if uv.x < fill {
        // Slight gradient for depth
        let brightness = 0.85 + 0.15 * (1.0 - uv.y);
        color = vec4(u.fill_color.rgb * brightness, 1.0);
        
        // Edge highlight on fill end
        if uv.x > fill - 0.02 {
            color = vec4(u.fill_color.rgb * 1.3, 1.0);
        }
    }
    
    // Border
    let border = 0.03;
    if uv.x < border || uv.x > (1.0 - border) || uv.y < border || uv.y > (1.0 - border) {
        color = vec4(0.3, 0.4, 0.5, 1.0);
    }
    
    // Inner glow on fill
    if uv.x < fill && uv.y > 0.1 && uv.y < 0.3 {
        color = vec4(color.rgb * 1.2, color.a);
    }
    
    return color;
}
"#;

const TEXT_SHADER: &str = r#"
struct TextUniforms {
    position: vec4<f32>,
    color: vec4<f32>,
    char_data: vec4<u32>,
}

@group(0) @binding(0) var<uniform> u: TextUniforms;
@group(0) @binding(1) var glyph_tex: texture_2d<f32>;
@group(0) @binding(2) var glyph_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0)
    );
    let p = pos[idx];
    let scale = u.position.z;
    var out: VertexOutput;
    out.position = vec4(u.position.x + p.x * scale * 0.01, u.position.y - p.y * scale * 0.02, 0.0, 1.0);
    out.uv = p;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let char_idx = u.char_data.x;
    let col = char_idx % 16u;
    let row = char_idx / 16u;
    
    let tex_x = (f32(col) + in.uv.x) / 16.0;
    let tex_y = (f32(row) + in.uv.y) / 6.0;
    
    let alpha = textureSample(glyph_tex, glyph_sampler, vec2(tex_x, tex_y)).r;
    return vec4(u.color.rgb, u.color.a * alpha);
}
"#;

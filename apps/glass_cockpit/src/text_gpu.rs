// Phase 2: GPU Text Rendering
// Bitmap atlas with instanced quad rendering for high-performance text display
// Constitutional compliance: Doctrine 1 (GPU rendering), Doctrine 8 (texture atlas)

// Phase 2 scaffolding: GPU-accelerated text rendering for telemetry overlay
// Will be used for high-performance text display in Phase 2+

use anyhow::Result;

#[allow(dead_code)]
/// GPU text rendering system with bitmap atlas
pub struct GpuTextRenderer {
    /// Atlas texture (256x256, 16x16 grid of 8x8 glyphs)
    atlas_texture: wgpu::Texture,
    atlas_bind_group: wgpu::BindGroup,
    
    /// Instance buffer for glyph quads (position, uv, color)
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
}

/// Single glyph instance for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    /// Screen position (x, y) + size (w, h)
    pub position: [f32; 4],
    /// Atlas UV coordinates (u, v, u_size, v_size)
    pub uv: [f32; 4],
    /// Color (r, g, b, a)
    pub color: [f32; 4],
}

impl GlyphInstance {
    pub fn new(x: f32, y: f32, char_code: u8, color: [f32; 4]) -> Self {
        // Atlas layout: 16x16 grid, each cell 16x16 pixels (for 8x8 glyph + padding)
        let atlas_index = (char_code as u32).saturating_sub(32); // ASCII 32-126
        let atlas_x = (atlas_index % 16) as f32 * 16.0;
        let atlas_y = (atlas_index / 16) as f32 * 16.0;
        
        Self {
            position: [x, y, 8.0, 8.0], // 8x8 glyph size
            uv: [atlas_x / 256.0, atlas_y / 256.0, 16.0 / 256.0, 16.0 / 256.0],
            color,
        }
    }
}

// Phase 2 scaffolding: GpuTextRenderer implementation for GPU text rendering
#[allow(dead_code)]
impl GpuTextRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: wgpu::TextureFormat) -> Result<Self> {
        // Create atlas texture (256x256 R8 format)
        let atlas_size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };
        
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Text Atlas"),
            size: atlas_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Generate atlas bitmap data
        let atlas_data = Self::generate_atlas();
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
            atlas_size,
        );
        
        // Create texture view and sampler
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Text Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Bind Group Layout"),
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
        
        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Bind Group"),
            layout: &bind_group_layout,
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
        
        // Create instance buffer (max 1024 glyphs per frame)
        let instance_capacity = 1024;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Text Instance Buffer"),
            size: (instance_capacity * std::mem::size_of::<GlyphInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Text Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/text.wgsl").into()),
        });
        
        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Text Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GlyphInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        // position (vec4)
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        // uv (vec4)
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                        // color (vec4)
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 32,
                            shader_location: 2,
                        },
                    ],
                }],
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        Ok(Self {
            atlas_texture,
            atlas_bind_group,
            instance_buffer,
            instance_capacity,
            pipeline,
        })
    }
    
    /// Generate atlas bitmap from BitmapFont
    fn generate_atlas() -> Vec<u8> {
        use crate::text::BitmapFont;
        
        let font = BitmapFont::new();
        let mut atlas = vec![0u8; 256 * 256];
        
        // Layout: 16x16 grid, each cell 16x16 pixels (8x8 glyph centered with padding)
        for char_code in 32u8..=126 {
            if let Some(glyph) = font.get_glyph(char_code as char) {
                let atlas_index = (char_code - 32) as usize;
                let cell_x = (atlas_index % 16) * 16;
                let cell_y = (atlas_index / 16) * 16;
                
                // Copy 8x8 glyph to center of 16x16 cell (with 4px padding on each side)
                for row in 0..8 {
                    for col in 0..8 {
                        if font.is_pixel_set(glyph, col as u32, row as u32) {
                            let atlas_x = cell_x + col + 4; // 4px left padding
                            let atlas_y = cell_y + row + 4; // 4px top padding
                            atlas[atlas_y * 256 + atlas_x] = 255;
                        }
                    }
                }
            }
        }
        
        atlas
    }
    
    /// Render text instances
    pub fn render<'rpass>(&'rpass self, queue: &wgpu::Queue, render_pass: &mut wgpu::RenderPass<'rpass>, instances: &[GlyphInstance]) {
        if instances.is_empty() {
            return;
        }
        
        let instance_count = instances.len().min(self.instance_capacity);
        
        // Upload instance data
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances[..instance_count]));
        
        // Draw instanced quads (6 vertices per quad)
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.atlas_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        render_pass.draw(0..6, 0..instance_count as u32);
    }
}

// Phase 2 scaffolding: Text building helper for GPU text rendering
#[allow(dead_code)]
/// Helper to build text rendering instances
pub struct TextBuilder {
    instances: Vec<GlyphInstance>,
    cursor_x: f32,
    cursor_y: f32,
    color: [f32; 4],
}

// Phase 2 scaffolding: TextBuilder implementation for GPU text building
#[allow(dead_code)]
impl TextBuilder {
    pub fn new() -> Self {
        Self {
            instances: Vec::with_capacity(256),
            cursor_x: 0.0,
            cursor_y: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
    
    pub fn set_position(&mut self, x: f32, y: f32) {
        self.cursor_x = x;
        self.cursor_y = y;
    }
    
    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color = color;
    }
    
    pub fn add_text(&mut self, text: &str) {
        for ch in text.chars() {
            if ch.is_ascii() && (' '..='~').contains(&ch) {
                self.instances.push(GlyphInstance::new(
                    self.cursor_x,
                    self.cursor_y,
                    ch as u8,
                    self.color,
                ));
                self.cursor_x += 8.0; // Advance by glyph width
            }
        }
    }
    
    pub fn newline(&mut self) {
        self.cursor_x = 0.0;
        self.cursor_y += 10.0; // Line height (8px glyph + 2px spacing)
    }
    
    pub fn build(self) -> Vec<GlyphInstance> {
        self.instances
    }
}

impl Default for TextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

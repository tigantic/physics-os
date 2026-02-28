//! Grayscale Bridge Heatmap Renderer with 1D Colormap Lookup
//!
//! Receives uint8 intensity from Python and applies colormap in fragment shader.
//! 
//! Benefits:
//! - 4x bandwidth reduction (32KB vs 128KB for 256×128)
//! - Dynamic colormap switching in UI (Plasma, Viridis, Inferno, etc.)
//! - GPU-optimized texture lookup
//!
//! Constitutional: Article V GPU mandate, Doctrine 2 RAM Bridge Protocol v2

use std::path::PathBuf;

use crate::ram_bridge_v2::RamBridgeV2;

/// Available colormaps
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Colormap {
    Plasma,
    Viridis,
    Inferno,
    Magma,
    Turbo,
}

impl Colormap {
    pub fn next(self) -> Self {
        match self {
            Colormap::Plasma => Colormap::Viridis,
            Colormap::Viridis => Colormap::Inferno,
            Colormap::Inferno => Colormap::Magma,
            Colormap::Magma => Colormap::Turbo,
            Colormap::Turbo => Colormap::Plasma,
        }
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Colormap::Plasma => "Plasma",
            Colormap::Viridis => "Viridis",
            Colormap::Inferno => "Inferno",
            Colormap::Magma => "Magma",
            Colormap::Turbo => "Turbo",
        }
    }
    
    /// Generate 256-entry RGBA8 colormap
    pub fn generate_lut(&self) -> [u8; 1024] {
        let mut lut = [0u8; 1024];
        
        for i in 0..256 {
            let t = i as f32 / 255.0;
            let (r, g, b) = match self {
                Colormap::Plasma => Self::plasma(t),
                Colormap::Viridis => Self::viridis(t),
                Colormap::Inferno => Self::inferno(t),
                Colormap::Magma => Self::magma(t),
                Colormap::Turbo => Self::turbo(t),
            };
            
            let idx = i * 4;
            lut[idx] = (r * 255.0) as u8;
            lut[idx + 1] = (g * 255.0) as u8;
            lut[idx + 2] = (b * 255.0) as u8;
            lut[idx + 3] = ((t * 2.0).min(1.0) * 255.0) as u8; // Alpha ramps up
        }
        
        lut
    }
    
    fn plasma(t: f32) -> (f32, f32, f32) {
        // Plasma colormap approximation
        let r = (0.5 + 0.5 * (std::f32::consts::PI * (t * 2.0 - 0.5)).sin()).clamp(0.0, 1.0);
        let g = (0.5 + 0.5 * (std::f32::consts::PI * (t * 2.0 - 1.0)).sin()).clamp(0.0, 1.0);
        let b = (1.0 - 0.9 * t).clamp(0.0, 1.0);
        (r, g, b)
    }
    
    fn viridis(t: f32) -> (f32, f32, f32) {
        // Viridis colormap approximation
        let r = (0.267 + t * (0.329 + t * (1.542 - t * 1.138))).clamp(0.0, 1.0);
        let g = (0.004 + t * (1.513 - t * 0.527)).clamp(0.0, 1.0);
        let b = (0.329 + t * (1.288 - t * (2.891 - t * 1.579))).clamp(0.0, 1.0);
        (r, g, b)
    }
    
    fn inferno(t: f32) -> (f32, f32, f32) {
        // Inferno colormap approximation
        let r = (t * (2.871 - t * 1.872)).clamp(0.0, 1.0);
        let g = (t * t * (2.0 - t * 0.7)).clamp(0.0, 1.0);
        let b = (0.5 * (1.0 - (t - 0.5).abs() * 2.0) + 0.1).clamp(0.0, 1.0);
        (r, g, b)
    }
    
    fn magma(t: f32) -> (f32, f32, f32) {
        // Magma colormap approximation
        let r = (t * (2.5 - t * 1.5)).clamp(0.0, 1.0);
        let g = (t * t * 1.5).clamp(0.0, 1.0);
        let b = (0.3 + t * (1.2 - t * 0.7)).clamp(0.0, 1.0);
        (r, g, b)
    }
    
    fn turbo(t: f32) -> (f32, f32, f32) {
        use std::f32::consts::TAU;
        // Turbo colormap (rainbow-like, high contrast)
        let r = (0.5 + 0.5 * (TAU * (t - 0.0)).sin()).clamp(0.0, 1.0);
        let g = (0.5 + 0.5 * (TAU * (t - 0.33)).sin()).clamp(0.0, 1.0);
        let b = (0.5 + 0.5 * (TAU * (t - 0.67)).sin()).clamp(0.0, 1.0);
        (r, g, b)
    }
}

/// Grayscale bridge heatmap renderer with colormap lookup
pub struct GrayscaleBridgeRenderer {
    /// Intensity texture (R8)
    intensity_texture: wgpu::Texture,
    intensity_view: wgpu::TextureView,
    
    /// 1D colormap texture (256 × 1 RGBA8)
    colormap_texture: wgpu::Texture,
    colormap_view: wgpu::TextureView,
    
    /// Samplers
    intensity_sampler: wgpu::Sampler,
    colormap_sampler: wgpu::Sampler,
    
    /// Pipeline
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    
    /// Bridge
    bridge: Option<RamBridgeV2>,
    last_frame: u64,
    
    /// Current colormap
    current_colormap: Colormap,
    
    /// Texture dimensions
    texture_width: u32,
    texture_height: u32,
    
    /// Performance
    pub read_us: u64,
    pub upload_us: u64,
}

impl GrayscaleBridgeRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        width: u32,
        height: u32,
    ) -> Self {
        // Create intensity texture (R8 format)
        let intensity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Intensity Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let intensity_view = intensity_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create 1D colormap texture (256 × 1 RGBA8)
        let colormap_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colormap Texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let colormap_view = colormap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Upload initial colormap
        let colormap = Colormap::Plasma;
        let lut = colormap.generate_lut();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &colormap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lut,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        
        // Samplers
        let intensity_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Intensity Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        let colormap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Colormap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grayscale Colormap Shader"),
            source: wgpu::ShaderSource::Wgsl(GRAYSCALE_COLORMAP_SHADER.into()),
        });
        
        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grayscale Bind Group Layout"),
            entries: &[
                // Intensity texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Intensity sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Colormap texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D1,
                        multisampled: false,
                    },
                    count: None,
                },
                // Colormap sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grayscale Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&intensity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&intensity_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&colormap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&colormap_sampler),
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grayscale Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grayscale Pipeline"),
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        let bridge = Self::try_connect_bridge();
        
        Self {
            intensity_texture,
            intensity_view,
            colormap_texture,
            colormap_view,
            intensity_sampler,
            colormap_sampler,
            pipeline,
            bind_group_layout,
            bind_group,
            bridge,
            last_frame: 0,
            current_colormap: colormap,
            texture_width: width,
            texture_height: height,
            read_us: 0,
            upload_us: 0,
        }
    }
    
    fn try_connect_bridge() -> Option<RamBridgeV2> {
        // Cross-platform path: Windows uses TEMP, Linux uses /dev/shm
        let path = if cfg!(target_os = "windows") {
            let temp = std::env::var("TEMP").unwrap_or_else(|_| "C:\\Temp".to_string());
            PathBuf::from(temp).join("ontic_bridge")
        } else {
            PathBuf::from("/dev/shm/ontic_bridge")
        };
        println!("  Trying grayscale bridge at: {:?}", path);
        match RamBridgeV2::connect(path) {
            Ok(bridge) => {
                println!("✓ Connected to grayscale bridge");
                Some(bridge)
            }
            Err(e) => {
                println!("⚠ Bridge not available: {}", e);
                None
            }
        }
    }
    
    /// Switch colormap dynamically
    pub fn set_colormap(&mut self, queue: &wgpu::Queue, colormap: Colormap) {
        if colormap == self.current_colormap {
            return;
        }
        
        let lut = colormap.generate_lut();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.colormap_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lut,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        
        self.current_colormap = colormap;
        println!("  Colormap: {}", colormap.name());
    }
    
    /// Cycle to next colormap
    pub fn next_colormap(&mut self, queue: &wgpu::Queue) {
        self.set_colormap(queue, self.current_colormap.next());
    }
    
    /// Get current colormap name
    pub fn colormap_name(&self) -> &'static str {
        self.current_colormap.name()
    }
    
    /// Update from bridge
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let Some(bridge) = &mut self.bridge else {
            self.bridge = Self::try_connect_bridge();
            return false;
        };
        
        let t0 = std::time::Instant::now();
        
        match bridge.read_frame() {
            Ok(Some((header, data))) => {
                let t1 = std::time::Instant::now();
                self.read_us = t1.duration_since(t0).as_micros() as u64;
                
                // Check if grayscale (1 channel)
                if header.channels != 1 {
                    // Fallback: might be RGBA, skip
                    return false;
                }
                
                // Resize if needed
                if header.width != self.texture_width || header.height != self.texture_height {
                    self.resize_texture(device, header.width, header.height);
                }
                
                // Upload intensity data
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.intensity_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(header.width),
                        rows_per_image: Some(header.height),
                    },
                    wgpu::Extent3d {
                        width: header.width,
                        height: header.height,
                        depth_or_array_layers: 1,
                    },
                );
                
                let t2 = std::time::Instant::now();
                self.upload_us = t2.duration_since(t1).as_micros() as u64;
                self.last_frame = header.frame_number;
                
                true
            }
            Ok(None) => {
                self.read_us = 0;
                self.upload_us = 0;
                false
            }
            Err(e) => {
                eprintln!("⚠ Bridge error: {}", e);
                self.bridge = None;
                false
            }
        }
    }
    
    fn resize_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        println!("  ℹ Resizing: {}×{} → {}×{}", 
            self.texture_width, self.texture_height, width, height);
        
        self.intensity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Intensity Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.intensity_view = self.intensity_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Recreate bind group
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grayscale Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.intensity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.intensity_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.colormap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.colormap_sampler),
                },
            ],
        });
        
        self.texture_width = width;
        self.texture_height = height;
    }
    
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.bridge.is_none() {
            return;
        }
        
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
    
    pub fn is_connected(&self) -> bool {
        self.bridge.is_some()
    }
}

const GRAYSCALE_COLORMAP_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );
    
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@group(0) @binding(0) var t_intensity: texture_2d<f32>;
@group(0) @binding(1) var s_intensity: sampler;
@group(0) @binding(2) var t_colormap: texture_1d<f32>;
@group(0) @binding(3) var s_colormap: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample intensity (grayscale)
    let intensity = textureSample(t_intensity, s_intensity, in.uv).r;
    
    // Lookup color in 1D colormap
    let color = textureSample(t_colormap, s_colormap, intensity);
    
    return color;
}
"#;

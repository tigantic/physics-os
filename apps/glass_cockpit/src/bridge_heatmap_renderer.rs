//! Bridge Heatmap Renderer
//!
//! Displays pre-rendered heatmap texture from Python/CUDA via RAM bridge.
//! This offloads all computation to GPU, achieving 1000+ FPS.
//!
//! Data Flow:
//!   Python (CUDA) → /dev/shm/ontic_bridge → This renderer → Display
//!
//! Constitutional: Article V GPU mandate, Doctrine 2 RAM Bridge Protocol
#![allow(dead_code)] // Bridge renderer ready for Python backend

use std::path::PathBuf;

use crate::ram_bridge_v2::RamBridgeV2;

/// Bridge heatmap renderer - displays pre-computed RGBA8 texture
pub struct BridgeHeatmapRenderer {
    /// wgpu texture for heatmap display
    texture: wgpu::Texture,
    /// Texture view
    texture_view: wgpu::TextureView,
    /// Sampler
    sampler: wgpu::Sampler,
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout (for recreating bind group on texture change)
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// RAM bridge connection (None if not connected)
    bridge: Option<RamBridgeV2>,
    /// Last frame number read
    last_frame: u64,
    /// Current texture dimensions
    texture_width: u32,
    texture_height: u32,
    /// Performance: microseconds to read last frame
    pub read_us: u64,
    /// Performance: microseconds to upload last frame
    pub upload_us: u64,
}

impl BridgeHeatmapRenderer {
    /// Create a new bridge heatmap renderer
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        width: u32,
        height: u32,
    ) -> Self {
        // Create texture for heatmap
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bridge Heatmap Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bridge Heatmap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Simple fullscreen quad shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bridge Heatmap Shader"),
            source: wgpu::ShaderSource::Wgsl(BRIDGE_HEATMAP_SHADER.into()),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bridge Heatmap Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bridge Heatmap Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bridge Heatmap Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bridge Heatmap Pipeline"),
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

        // Try to connect to bridge
        let bridge = Self::try_connect_bridge();

        Self {
            texture,
            texture_view,
            sampler,
            pipeline,
            bind_group_layout,
            bind_group,
            bridge,
            last_frame: 0,
            texture_width: width,
            texture_height: height,
            read_us: 0,
            upload_us: 0,
        }
    }

    /// Try to connect to RAM bridge
    fn try_connect_bridge() -> Option<RamBridgeV2> {
        // Cross-platform path: Windows uses TEMP, Linux uses /dev/shm
        let path = if cfg!(target_os = "windows") {
            let temp = std::env::var("TEMP").unwrap_or_else(|_| "C:\\Temp".to_string());
            PathBuf::from(temp).join("ontic_bridge")
        } else {
            PathBuf::from("/dev/shm/ontic_bridge")
        };
        println!("  Trying bridge at: {:?}", path);
        match RamBridgeV2::connect(path) {
            Ok(bridge) => {
                println!("✓ Connected to RAM bridge");
                Some(bridge)
            }
            Err(e) => {
                println!("⚠ RAM bridge not available: {}", e);
                None
            }
        }
    }

    /// Update texture from RAM bridge
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        let Some(bridge) = &mut self.bridge else {
            // Try to reconnect periodically
            self.bridge = Self::try_connect_bridge();
            return false;
        };

        let t0 = std::time::Instant::now();

        // Read frame from bridge
        match bridge.read_frame() {
            Ok(Some((header, data))) => {
                let t1 = std::time::Instant::now();
                self.read_us = t1.duration_since(t0).as_micros() as u64;

                // Check if texture needs to be resized
                if header.width != self.texture_width || header.height != self.texture_height {
                    self.resize_texture(device, header.width, header.height);
                }

                // Upload to GPU texture
                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(header.width * 4),
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
                // No new frame (cached)
                self.read_us = 0;
                self.upload_us = 0;
                false
            }
            Err(e) => {
                eprintln!("⚠ Bridge read error: {}", e);
                self.bridge = None;
                false
            }
        }
    }

    /// Resize texture to match new dimensions from bridge
    fn resize_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        println!("  ℹ Resizing bridge texture: {}×{} → {}×{}", 
            self.texture_width, self.texture_height, width, height);

        // Create new texture
        self.texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bridge Heatmap Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.texture_view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate bind group with new texture view
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bridge Heatmap Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.texture_width = width;
        self.texture_height = height;
    }

    /// Render the heatmap texture as fullscreen overlay
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.bridge.is_none() {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1); // Fullscreen quad
    }

    /// Check if bridge is connected
    pub fn is_connected(&self) -> bool {
        self.bridge.is_some()
    }

    /// Get last frame number
    pub fn last_frame(&self) -> u64 {
        self.last_frame
    }
}

/// WGSL shader for fullscreen textured quad
const BRIDGE_HEATMAP_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle strip
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

@group(0) @binding(0) var t_heatmap: texture_2d<f32>;
@group(0) @binding(1) var s_heatmap: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_heatmap, s_heatmap, in.uv);
    return color;
}
"#;

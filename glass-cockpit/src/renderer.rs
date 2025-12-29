/*!
 * Phase 2: Integrated Renderer - Grid + Tensor Field + UI + Text
 * Multi-pipeline rendering: grid → tensor overlay → UI overlay → text overlay
 */

use anyhow::Result;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::bridge::Telemetry;
use crate::camera::Camera;
use crate::layout::ViewLayout;
use crate::overlay::TelemetryOverlay;
use crate::tensor_renderer::TensorRenderer;
use crate::text_gpu::{GpuTextRenderer, TextBuilder};

// Grid shader uniforms
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GridUniforms {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

// UI shader uniforms
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UiUniforms {
    screen_size: [f32; 2],
    time: f32,
    _padding: f32,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    
    // Grid rendering
    grid_pipeline: wgpu::RenderPipeline,
    grid_uniform_buffer: wgpu::Buffer,
    grid_bind_group: wgpu::BindGroup,
    
    // Tensor field rendering (Phase 2)
    tensor_renderer: TensorRenderer,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    
    // UI overlay rendering
    ui_pipeline: wgpu::RenderPipeline,
    ui_uniform_buffer: wgpu::Buffer,
    ui_bind_group: wgpu::BindGroup,
    
    // Text rendering
    text_renderer: GpuTextRenderer,
    
    // State management
    pub camera: Camera,
    pub layout: ViewLayout,
    pub telemetry: TelemetryOverlay,
    
    adapter_info: String,
    start_time: std::time::Instant,
}

impl Renderer {
    pub async fn new(window: &'static Window) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No adapter"))?;
        
        let adapter_info = format!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
        
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            }, None)
            .await?;
        
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied()
            .find(|f| f.is_srgb()).unwrap_or(surface_caps.formats[0]);
        
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        
        // Initialize state managers
        let aspect = size.width as f32 / size.height as f32;
        let camera = Camera::new(aspect);
        let layout = ViewLayout::new(size.width, size.height);
        let telemetry = TelemetryOverlay::new();
        
        // ==== GRID PIPELINE ====
        let grid_uniforms = GridUniforms {
            view_proj: camera.view_proj_matrix().to_cols_array_2d(),
            inv_view_proj: camera.inv_view_proj_matrix().to_cols_array_2d(),
            camera_pos: camera.position.to_array(),
            _padding: 0.0,
        };
        
        let grid_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Uniform Buffer"),
            contents: bytemuck::cast_slice(&[grid_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let grid_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grid Bind Group Layout"),
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
        
        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Bind Group"),
            layout: &grid_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_uniform_buffer.as_entire_binding(),
            }],
        });
        
        let grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid.wgsl").into()),
        });
        
        let grid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid Pipeline Layout"),
            bind_group_layouts: &[&grid_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Pipeline"),
            layout: Some(&grid_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        
        // ==== UI OVERLAY PIPELINE ====
        let ui_uniforms = UiUniforms {
            screen_size: [size.width as f32, size.height as f32],
            time: 0.0,
            _padding: 0.0,
        };
        
        let ui_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ui_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let ui_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UI Bind Group Layout"),
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
        
        let ui_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &ui_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ui_uniform_buffer.as_entire_binding(),
            }],
        });
        
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf.wgsl").into()),
        });
        
        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&ui_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&ui_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
        
        // ==== DEPTH BUFFER FOR TENSOR RENDERING ====
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // ==== TENSOR FIELD RENDERING PIPELINE ====
        let tensor_renderer = TensorRenderer::new(&device, &queue, config.format, &grid_bind_group_layout)?;
        
        // ==== TEXT RENDERING PIPELINE ====
        let text_renderer = GpuTextRenderer::new(&device, &queue, config.format)?;
        
        Ok(Self {
            surface, device, queue, config, size,
            grid_pipeline, grid_uniform_buffer, grid_bind_group,
            tensor_renderer, depth_texture, depth_view,
            ui_pipeline, ui_uniform_buffer, ui_bind_group,
            text_renderer,
            camera, layout, telemetry, adapter_info,
            start_time: std::time::Instant::now(),
        })
    }
    
    pub fn adapter_info(&self) -> &str {
        &self.adapter_info
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.size.width = width;
            self.size.height = height;
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.camera.set_aspect(width as f32 / height as f32);
            self.layout.resize(width, height);
            
            // Recreate depth texture
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            self.update_grid_uniforms();
            self.update_ui_uniforms();
        }
    }
    
    fn update_grid_uniforms(&self) {
        let uniforms = GridUniforms {
            view_proj: self.camera.view_proj_matrix().to_cols_array_2d(),
            inv_view_proj: self.camera.inv_view_proj_matrix().to_cols_array_2d(),
            camera_pos: self.camera.position.to_array(),
            _padding: 0.0,
        };
        self.queue.write_buffer(&self.grid_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }
    
    fn update_ui_uniforms(&self) {
        let time = self.start_time.elapsed().as_secs_f32();
        let uniforms = UiUniforms {
            screen_size: [self.size.width as f32, self.size.height as f32],
            time,
            _padding: 0.0,
        };
        self.queue.write_buffer(&self.ui_uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }
    
    pub fn render(&mut self, frame_duration: std::time::Duration, _telemetry: Option<Telemetry>) -> Result<()> {
        // Update telemetry overlay
        self.telemetry.update(frame_duration);
        
        // Update uniforms
        self.update_grid_uniforms();
        self.update_ui_uniforms();
        
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.071, g: 0.071, b: 0.071, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            // Draw grid (background layer)
            render_pass.set_pipeline(&self.grid_pipeline);
            render_pass.set_bind_group(0, &self.grid_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
            
            // Draw tensor field overlay (Phase 2)
            self.tensor_renderer.render(&mut render_pass, &self.grid_bind_group);
            
            // Draw UI overlay (foreground layer) if visible
            if self.telemetry.visible {
                render_pass.set_pipeline(&self.ui_pipeline);
                render_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }
        
        // Build text instances from telemetry
        let text_instances = self.build_telemetry_text();
        
        // Render text overlay
        if !text_instances.is_empty() {
            let mut text_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Don't clear - composite on top
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            
            self.text_renderer.render(&self.queue, &mut text_pass, &text_instances);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
    
    fn build_telemetry_text(&self) -> Vec<crate::text_gpu::GlyphInstance> {
        let mut builder = TextBuilder::new();
        
        // Left rail telemetry
        let rail_x = 10.0;
        
        // P-Core card
        builder.set_position(rail_x, 60.0);
        builder.set_color([0.9, 0.9, 1.0, 1.0]);
        builder.add_text("P-CORE");
        builder.set_position(rail_x, 75.0);
        builder.set_color([1.0, 1.0, 1.0, 1.0]);
        builder.add_text(&format!("{:.1}%", self.telemetry.current.p_core_usage));
        
        // E-Core card
        builder.set_position(rail_x, 140.0);
        builder.set_color([0.9, 0.9, 1.0, 1.0]);
        builder.add_text("E-CORE");
        builder.set_position(rail_x, 155.0);
        builder.set_color([1.0, 1.0, 1.0, 1.0]);
        builder.add_text(&format!("{:.1}%", self.telemetry.current.e_core_usage));
        
        // FPS card
        builder.set_position(rail_x, 220.0);
        builder.set_color([0.9, 1.0, 0.9, 1.0]);
        builder.add_text("FPS");
        builder.set_position(rail_x, 235.0);
        builder.set_color([1.0, 1.0, 1.0, 1.0]);
        builder.add_text(&format!("{:.1}", self.telemetry.current.fps));
        
        // Right rail telemetry
        let right_rail_x = self.size.width as f32 - 180.0;
        
        // Memory card
        builder.set_position(right_rail_x, 60.0);
        builder.set_color([1.0, 0.9, 0.7, 1.0]);
        builder.add_text("MEMORY");
        builder.set_position(right_rail_x, 75.0);
        builder.set_color([1.0, 1.0, 1.0, 1.0]);
        builder.add_text(&format!("{:.0}MB", self.telemetry.current.memory_mb));
        
        // Stability card
        builder.set_position(right_rail_x, 140.0);
        let stability_color = self.telemetry.stability_color();
        builder.set_color([stability_color[0], stability_color[1], stability_color[2], 1.0]);
        builder.add_text("STABILITY");
        builder.set_position(right_rail_x, 155.0);
        builder.set_color([1.0, 1.0, 1.0, 1.0]);
        builder.add_text(&format!("{:.1}%", self.telemetry.current.stability * 100.0));
        
        builder.build()
    }
}

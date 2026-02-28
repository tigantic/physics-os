/*!
 * Ontic Glass Cockpit - Phase 4: Globe & Satellite Visualization
 * 
 * Orthographic globe rendering with NASA GIBS satellite tiles.
 * 
 * Constitutional Compliance:
 * - Article II: Type-safe Rust with E-core affinity
 * - Article V: GPU-accelerated globe rendering
 * - Article VIII: <5% CPU (GPU handles all rendering)
 * - Doctrine 2: Async tile fetching, never blocks render
 */

use anyhow::Result;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent, ElementState, MouseButton, MouseScrollDelta},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{PhysicalKey, KeyCode},
    window::{Window, WindowBuilder},
};
use wgpu::util::DeviceExt;

mod affinity;
mod globe;
mod tile_fetcher;

use globe::{Icosphere, GlobeConfig, GlobeCamera};
use tile_fetcher::{TileFetcher, GibsConfig};

/// Phase 4 Globe Visualization
fn main() -> Result<()> {
    println!("Ontic Glass Cockpit v0.4.0");
    println!("Phase 4: Globe & Satellite Visualization");
    println!("═══════════════════════════════════════════════");
    
    // STEP 1: Enforce E-core affinity
    println!("[1/4] Enforcing E-core affinity...");
    #[cfg(target_os = "windows")]
    {
        affinity::enforce_e_core_affinity()?;
        println!("  ✓ UI pinned to E-cores (16-31)");
    }
    #[cfg(not(target_os = "windows"))]
    {
        println!("  ⚠ E-core affinity only supported on Windows");
    }
    
    // STEP 2: Initialize tile fetcher
    println!("[2/4] Initializing NASA GIBS tile fetcher...");
    let gibs_config = GibsConfig::default();
    let _tile_fetcher = TileFetcher::new(gibs_config)?;
    println!("  ✓ Tile cache initialized");
    
    // STEP 3: Create event loop and window
    println!("[3/4] Creating window...");
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("Ontic Glass Cockpit - Phase 4")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)?);
    println!("  ✓ Window created (1920×1080)");
    
    // STEP 4: Initialize GPU
    println!("[4/4] Initializing GPU pipeline...");
    let (device, queue, surface, config) = pollster::block_on(init_gpu(window.as_ref()))?;
    let window_clone = Arc::clone(&window);
    println!("  ✓ wgpu initialized");
    
    // Initialize globe
    let globe_config = GlobeConfig::default();
    let icosphere = Icosphere::new(globe_config.clone());
    println!("  ✓ Globe mesh: {} vertices, {} triangles", 
        icosphere.vertex_count(), icosphere.triangle_count());
    
    // Create camera
    let mut camera = GlobeCamera::new();
    
    // Create GPU buffers and pipeline
    let globe_pipeline = create_globe_pipeline(&device, &config, &icosphere)?;
    
    // Mouse state for pan/zoom
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    
    println!("\nPhase 4 Globe Visualization Running");
    println!("Controls:");
    println!("  • Mouse Drag: Pan camera");
    println!("  • Mouse Wheel: Zoom in/out");
    println!("  • ESC: Exit");
    println!("═══════════════════════════════════════════════\n");
    
    // Main event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent { window_id: event_window_id, event } => {
                if event_window_id != window_clone.id() {
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => {
                        println!("\nPhase 4 Shutdown");
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                                elwt.exit();
                            }
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left {
                            mouse_pressed = state == ElementState::Pressed;
                            if !mouse_pressed {
                                last_mouse_pos = None;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if mouse_pressed {
                            if let Some((last_x, last_y)) = last_mouse_pos {
                                let delta_x = (position.x - last_x) as f32 * 0.001;
                                let delta_y = (position.y - last_y) as f32 * 0.001;
                                camera.pan(-delta_x, delta_y);
                            }
                            last_mouse_pos = Some((position.x, position.y));
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let zoom_delta = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                        };
                        camera.zoom(zoom_delta);
                    }
                    WindowEvent::RedrawRequested => {
                        // Update camera (165Hz Sovereign mode)
                        camera.update(1.0 / 165.0);
                        
                        // Render frame
                        match render_frame(&device, &queue, &surface, &globe_pipeline, &camera, &icosphere) {
                            Ok(_) => {},
                            Err(e) => eprintln!("Render error: {}", e),
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                // Request redraw
                window_clone.request_redraw();
            }
            _ => {}
        }
    })?;
    
    Ok(())
}

/// Initialize GPU (wgpu)
async fn init_gpu(window: &Window) -> Result<(wgpu::Device, wgpu::Queue, wgpu::Surface<'_>, wgpu::SurfaceConfiguration)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    let surface = instance.create_surface(window)?;
    
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }).await.ok_or_else(|| anyhow::anyhow!("Failed to find GPU adapter"))?;
    
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Ontic GPU"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ).await?;
    
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_capabilities(&adapter).formats[0],
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox, // 165Hz Sovereign mode
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);
    
    Ok((device, queue, surface, config))
}

/// Globe rendering pipeline
struct GlobePipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

/// Create globe rendering pipeline
fn create_globe_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    icosphere: &Icosphere,
) -> Result<GlobePipeline> {
    // Load shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Globe Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/globe.wgsl").into()),
    });
    
    // Create vertex buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Vertex {
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        lat_lon: [f32; 2],
    }
    
    let vertices: Vec<Vertex> = icosphere.vertices.iter().map(|v| {
        Vertex {
            position: [v.position.x, v.position.y, v.position.z],
            normal: [v.normal.x, v.normal.y, v.normal.z],
            uv: v.uv,
            lat_lon: [v.lat, v.lon],
        }
    }).collect();
    
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globe Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Globe Index Buffer"),
        contents: bytemuck::cast_slice(&icosphere.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    
    // Create camera uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct CameraUniforms {
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        _padding: f32,
        zoom: f32,
        aspect_ratio: f32,
        time: f32,
        _padding2: f32,
    }
    
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Camera Buffer"),
        size: std::mem::size_of::<CameraUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Camera Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera Bind Group"),
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Globe Pipeline Layout"),
        bind_group_layouts: &[&camera_bind_group_layout], // No material bind group for procedural shader
        push_constant_ranges: &[],
    });
    
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Globe Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3,  // position
                    1 => Float32x3,  // normal
                    2 => Float32x2,  // uv
                    3 => Float32x2,  // lat_lon
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_procedural",  // Use procedural shader for now
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });
    
    Ok(GlobePipeline {
        render_pipeline,
        vertex_buffer,
        index_buffer,
        index_count: icosphere.indices.len() as u32,
        camera_buffer,
        camera_bind_group,
    })
}

/// Render a single frame
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &wgpu::Surface,
    pipeline: &GlobePipeline,
    camera: &GlobeCamera,
    _icosphere: &Icosphere,
) -> Result<()> {
    // Update camera uniform buffer
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct CameraUniforms {
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        _padding: f32,
        zoom: f32,
        aspect_ratio: f32,
        time: f32,
        _padding2: f32,
    }
    
    let view_matrix = camera.view_matrix();
    let proj_matrix = camera.projection_matrix(16.0 / 9.0);
    let view_proj = proj_matrix * view_matrix;
    
    let uniforms = CameraUniforms {
        view_proj: view_proj.to_cols_array_2d(),
        camera_pos: [camera.position.x, camera.position.y, camera.position.z],
        _padding: 0.0,
        zoom: camera.zoom,
        aspect_ratio: 16.0 / 9.0,
        time: 0.0,
        _padding2: 0.0,
    };
    
    queue.write_buffer(&pipeline.camera_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    
    // Get surface texture
    let output = surface.get_current_texture()?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Globe Render Encoder"),
    });
    
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Globe Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        render_pass.set_pipeline(&pipeline.render_pipeline);
        render_pass.set_bind_group(0, &pipeline.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));
        render_pass.set_index_buffer(pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..pipeline.index_count, 0, 0..1);
    }
    
    queue.submit(std::iter::once(encoder.finish()));
    output.present();
    
    Ok(())
}

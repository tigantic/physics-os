/*!
 * HyperTensor Glass Cockpit - Phase 5: Vector Field Overlay
 * 
 * GPU-accelerated vector field visualization with particle advection
 * and streamline rendering over the globe.
 * 
 * Constitutional Compliance:
 * - Article II: Type-safe Rust with E-core affinity
 * - Article V: GPU-accelerated particle compute + render
 * - Article VIII: <5% CPU (GPU handles advection + rendering)
 * - Doctrine 3: All particle compute on GPU via compute shaders
 */

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
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
mod vector_field;
mod particle_system;
mod streamlines;

use globe::{Icosphere, GlobeConfig, GlobeCamera};
use tile_fetcher::{TileFetcher, GibsConfig};
use vector_field::{VectorField, VectorFieldConfig};
use particle_system::{ParticleSystem, ParticleConfig};
use streamlines::{StreamlineGenerator, StreamlineRenderer, StreamlineConfig, StreamlineSpacing};

/// Visualization mode
#[derive(Clone, Copy, PartialEq, Eq)]
enum VizMode {
    Particles,
    Streamlines,
    Both,
}

impl VizMode {
    fn next(self) -> Self {
        match self {
            VizMode::Particles => VizMode::Streamlines,
            VizMode::Streamlines => VizMode::Both,
            VizMode::Both => VizMode::Particles,
        }
    }
    
    fn name(&self) -> &'static str {
        match self {
            VizMode::Particles => "Particles",
            VizMode::Streamlines => "Streamlines",
            VizMode::Both => "Both",
        }
    }
}

/// Phase 5 Vector Field Visualization
fn main() -> Result<()> {
    println!("HyperTensor Glass Cockpit v0.5.0");
    println!("Phase 5: Vector Field Overlay");
    println!("═══════════════════════════════════════════════");
    
    // STEP 1: Enforce E-core affinity
    println!("[1/6] Enforcing E-core affinity...");
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
    println!("[2/6] Initializing NASA GIBS tile fetcher...");
    let gibs_config = GibsConfig::default();
    let _tile_fetcher = TileFetcher::new(gibs_config)?;
    println!("  ✓ Tile cache initialized");
    
    // STEP 3: Generate synthetic vector field
    println!("[3/6] Generating vector field...");
    let field_config = VectorFieldConfig::for_zoom_level(5, 0.0, 35.0); // Mesoscale view, centered on central US
    let mut vector_field = VectorField::new(field_config);
    vector_field.generate_test_pattern();
    let stats = vector_field.stats.clone();
    println!("  ✓ Vector field: {}x{} grid", field_config.grid_width, field_config.grid_height);
    println!("  ✓ Max speed: {:.1} m/s, Max vorticity: {:.6}", stats.max_speed, stats.max_vorticity);
    
    // STEP 4: Create event loop and window
    println!("[4/6] Creating window...");
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("HyperTensor Glass Cockpit - Phase 5")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)?);
    println!("  ✓ Window created (1920×1080)");
    
    // STEP 5: Initialize GPU
    println!("[5/6] Initializing GPU pipeline...");
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
    
    // Create globe pipeline
    let globe_pipeline = create_globe_pipeline(&device, &config, &icosphere)?;
    
    // STEP 6: Initialize particle system
    println!("[6/6] Initializing vector visualization...");
    let mut particle_system = ParticleSystem::new(&device, &queue, config.format, &field_config);
    let particle_config = ParticleConfig {
        spawn_rate: 500.0,      // Reduced from 2000 - GPU respawns dead particles
        lifetime: 12.0,          // Longer lifetime for better trails
        lifetime_variance: 4.0,
        base_size: 2.0,
        speed_size_factor: 0.8,
        ..Default::default()
    };
    particle_system.set_config(particle_config);
    particle_system.upload_vector_field(&queue, &vector_field);
    println!("  ✓ Particle system: {} max particles", particle_system::MAX_PARTICLES);
    
    // Generate streamlines
    let streamline_config = StreamlineConfig {
        count: 200,
        max_length: 1_000_000.0,
        step_size: 20_000.0,
        spacing: StreamlineSpacing::Grid,
        ..Default::default()
    };
    let mut streamline_gen = StreamlineGenerator::new(streamline_config);
    let streamlines = streamline_gen.generate(&vector_field);
    println!("  ✓ Streamlines: {} generated", streamlines.len());
    
    let mut streamline_renderer = StreamlineRenderer::new(&device, config.format, streamline_config);
    streamline_renderer.upload(&queue, &streamlines);
    
    // Visualization state
    // Start with both modes for production use
    let mut viz_mode = VizMode::Both;
    let mut show_globe = true;
    let start_time = Instant::now();
    
    // Mouse state
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    
    println!("\nPhase 5 Vector Field Visualization Running");
    println!("Controls:");
    println!("  • Mouse Drag: Pan camera");
    println!("  • Mouse Wheel: Zoom in/out");
    println!("  • V: Toggle visualization mode ({})", viz_mode.name());
    println!("  • G: Toggle globe visibility");
    println!("  • R: Regenerate vector field");
    println!("  • ESC: Exit");
    println!("═══════════════════════════════════════════════\n");
    
    // Frame timing
    let mut last_frame = Instant::now();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    
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
                        println!("\nPhase 5 Shutdown");
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                                PhysicalKey::Code(KeyCode::KeyV) => {
                                    viz_mode = viz_mode.next();
                                    println!("Visualization mode: {}", viz_mode.name());
                                }
                                PhysicalKey::Code(KeyCode::KeyG) => {
                                    show_globe = !show_globe;
                                    println!("Globe: {}", if show_globe { "visible" } else { "hidden" });
                                }
                                PhysicalKey::Code(KeyCode::KeyR) => {
                                    vector_field.generate_test_pattern();
                                    particle_system.upload_vector_field(&queue, &vector_field);
                                    particle_system.clear();
                                    let new_streamlines = streamline_gen.generate(&vector_field);
                                    streamline_renderer.upload(&queue, &new_streamlines);
                                    println!("Vector field regenerated");
                                }
                                _ => {}
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
                        // Calculate delta time
                        let now = Instant::now();
                        let dt = (now - last_frame).as_secs_f32();
                        last_frame = now;
                        
                        let time = start_time.elapsed().as_secs_f32();
                        
                        // Update camera
                        camera.update(dt);
                        
                        // Update particle system
                        if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
                            particle_system.update(&queue, dt, &vector_field);
                        }
                        
                        // Update streamline uniforms
                        if viz_mode == VizMode::Streamlines || viz_mode == VizMode::Both {
                            streamline_renderer.update(&queue, &vector_field, time);
                        }
                        
                        // Render frame
                        match render_frame(
                            &device,
                            &queue,
                            &surface,
                            &globe_pipeline,
                            &camera,
                            &icosphere,
                            &mut particle_system,
                            &streamline_renderer,
                            show_globe,
                            viz_mode,
                        ) {
                            Ok(_) => {},
                            Err(e) => eprintln!("Render error: {}", e),
                        }
                        // FPS counter - note: dt includes event loop + present time, not just GPU
                        frame_count += 1;
                        if fps_timer.elapsed().as_secs() >= 2 {
                            let fps = frame_count as f64 / fps_timer.elapsed().as_secs_f64();
                            println!("FPS: {:.1} | Particles: {}", fps, particle_system.particle_count());
                            frame_count = 0;
                            fps_timer = Instant::now();
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
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
            label: Some("HyperTensor GPU"),
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
        present_mode: wgpu::PresentMode::Mailbox, // Triple buffering, no VSync
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
        bind_group_layouts: &[&camera_bind_group_layout],
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
            entry_point: "fs_procedural",
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
#[allow(clippy::too_many_arguments)] // Phase 5 integration - will refactor in Phase 6
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &wgpu::Surface<'_>,
    globe_pipeline: &GlobePipeline,
    camera: &GlobeCamera,
    _icosphere: &Icosphere,
    particle_system: &mut ParticleSystem,
    streamline_renderer: &StreamlineRenderer,
    show_globe: bool,
    viz_mode: VizMode,
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
    
    queue.write_buffer(&globe_pipeline.camera_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    
    // Get surface texture
    let output = surface.get_current_texture()?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Phase 5 Render Encoder"),
    });
    
    // Run particle advection compute pass
    if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
        particle_system.advect(&mut encoder);
    }
    
    // Render pass
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phase 5 Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.05,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        // Draw globe (if visible)
        if show_globe {
            render_pass.set_pipeline(&globe_pipeline.render_pipeline);
            render_pass.set_bind_group(0, &globe_pipeline.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, globe_pipeline.vertex_buffer.slice(..));
            render_pass.set_index_buffer(globe_pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..globe_pipeline.index_count, 0, 0..1);
        }
        
        // Draw streamlines
        if viz_mode == VizMode::Streamlines || viz_mode == VizMode::Both {
            streamline_renderer.render(&mut render_pass);
        }
        
        // Draw particles
        if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
            particle_system.render(&mut render_pass);
        }
    }
    
    queue.submit(std::iter::once(encoder.finish()));
    output.present();
    
    Ok(())
}

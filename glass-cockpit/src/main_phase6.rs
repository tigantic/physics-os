/*!
 * HyperTensor Glass Cockpit - Phase 6: Convergence Heatmaps
 * 
 * GPU-accelerated convergence zone visualization with:
 * - Globe rendering
 * - Vector field overlay (particles + streamlines)
 * - Probabilistic convergence heatmaps
 * - LOD infrastructure for performance maintenance
 * 
 * Constitutional Compliance:
 * - Article II: Type-safe Rust with E-core affinity
 * - Article V: GPU-accelerated rendering
 * - Article VIII: <5% CPU, 60 FPS mandate
 * - Doctrine 3: Procedural rendering, no assets
 * 
 * Performance: LOD + culling infrastructure maintains 60 FPS
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
mod lod;
mod convergence;
mod convergence_renderer;
mod ram_bridge_v2;
mod bridge_heatmap_renderer;
mod grayscale_bridge_renderer;

use globe::{Icosphere, GlobeConfig, GlobeCamera};
use tile_fetcher::{TileFetcher, GibsConfig};
use vector_field::{VectorField, VectorFieldConfig};
use particle_system::{ParticleSystem, ParticleConfig};
use streamlines::{StreamlineGenerator, StreamlineRenderer, StreamlineConfig, StreamlineSpacing};
use lod::{LodCuller, LodConfig};
use convergence::ConvergenceConfig;
use convergence_renderer::ConvergenceRenderer;
use bridge_heatmap_renderer::BridgeHeatmapRenderer;
#[allow(unused_imports)]  // Colormap API available for future colormapping features
use grayscale_bridge_renderer::{GrayscaleBridgeRenderer, Colormap};

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

/// Phase 6 Convergence Heatmap Visualization
#[allow(unused_assignments)] // current_fps assigned in closure, used for telemetry
fn main() -> Result<()> {
    println!("HyperTensor Glass Cockpit v0.6.0 [Sovereign 165Hz]");
    println!("Phase 6: Convergence Heatmaps");
    println!("═══════════════════════════════════════════════");
    
    // STEP 1: Enforce E-core affinity
    println!("[1/7] Enforcing E-core affinity...");
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
    println!("[2/7] Initializing NASA GIBS tile fetcher...");
    let gibs_config = GibsConfig::default();
    let _tile_fetcher = TileFetcher::new(gibs_config)?;
    println!("  ✓ Tile cache initialized");
    
    // STEP 3: Generate synthetic vector field
    println!("[3/7] Generating vector field...");
    let field_config = VectorFieldConfig::for_zoom_level(5, 0.0, 35.0);
    let mut vector_field = VectorField::new(field_config);
    vector_field.generate_test_pattern();
    let stats = vector_field.stats.clone();
    println!("  ✓ Vector field: {}x{} grid", field_config.grid_width, field_config.grid_height);
    println!("  ✓ Max speed: {:.1} m/s, Max vorticity: {:.6}", stats.max_speed, stats.max_vorticity);
    
    // STEP 4: Create event loop and window
    println!("[4/7] Creating window...");
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("HyperTensor Glass Cockpit - Phase 6")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)?);
    println!("  ✓ Window created (1920×1080)");
    
    // STEP 5: Initialize GPU
    println!("[5/7] Initializing GPU pipeline...");
    let (device, queue, surface, config) = pollster::block_on(init_gpu(window.as_ref()))?;
    let window_clone = Arc::clone(&window);
    println!("  ✓ wgpu initialized");
    
    // Initialize globe
    let globe_config = GlobeConfig::default();
    let icosphere = Icosphere::new(globe_config.clone());
    println!("  ✓ Globe mesh: {} vertices, {} triangles", 
        icosphere.vertex_count(), icosphere.triangle_count());
    
    // Create depth texture for proper layering
    let depth_texture = create_depth_texture(&device, &config);
    
    // Create camera
    let mut camera = GlobeCamera::new();
    
    // Create LOD culler (Phase 6: Performance infrastructure)
    let mut lod_culler = LodCuller::new();
    lod_culler.config = LodConfig::globe_scale();
    
    // Create globe pipeline
    let globe_pipeline = create_globe_pipeline(&device, &config, &icosphere)?;
    
    // STEP 6: Initialize particle system
    println!("[6/7] Initializing vector visualization...");
    // Phase 8: Pass camera bind group layout for 3D globe projection
    let mut particle_system = ParticleSystem::new(
        &device, 
        &queue, 
        config.format, 
        &field_config,
        &globe_pipeline.camera_bind_group_layout,
    );
    let particle_config = ParticleConfig {
        spawn_rate: 500.0,
        lifetime: 12.0,
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
    
    let mut streamline_renderer = StreamlineRenderer::new(
        &device, 
        config.format, 
        streamline_config,
        &globe_pipeline.camera_bind_group_layout,
    );
    streamline_renderer.upload(&queue, &streamlines);
    
    // STEP 7: Initialize convergence heatmap (Phase 6)
    println!("[7/7] Initializing convergence heatmaps...");
    let convergence_config = ConvergenceConfig::default();
    let mut convergence_renderer = ConvergenceRenderer::new(&device, &config, convergence_config);
    println!("  ✓ Convergence field: {}x{} grid", 
        convergence_config.resolution.0, convergence_config.resolution.1);
    
    // BRIDGE MODE: Initialize bridge heatmap renderer for backend compute
    let mut bridge_heatmap = BridgeHeatmapRenderer::new(&device, &config, 1920, 1080);
    
    // GRAYSCALE BRIDGE: v2 with 1D colormap lookup (4x bandwidth reduction)
    let mut grayscale_bridge = GrayscaleBridgeRenderer::new(&device, &queue, &config, 256, 128);
    let use_grayscale = grayscale_bridge.is_connected();
    if use_grayscale {
        println!("  ✓ Grayscale bridge ENABLED - 4x bandwidth reduction");
        println!("  ✓ Colormap: {} (press C to cycle)", grayscale_bridge.colormap_name());
    } else if bridge_heatmap.is_connected() {
        println!("  ✓ RGBA bridge mode - using Python/CUDA backend");
    } else {
        println!("  ⚠ Bridge unavailable - using local compute");
    }
    
    // Visualization state
    let mut viz_mode = VizMode::Both;
    let mut show_globe = true;
    let mut show_heatmap = true;
    let mut use_bridge = use_grayscale || bridge_heatmap.is_connected();
    let use_grayscale_mode = use_grayscale;
    let start_time = Instant::now();
    
    // Mouse state
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    
    // Heatmap caching: only recompute on view change or animation tick
    let mut last_heatmap_cam_pos = glam::Vec3::ZERO;
    let mut last_heatmap_time = 0.0_f32;
    const HEATMAP_ANIMATION_INTERVAL: f32 = 0.1; // 10 Hz animation update
    const HEATMAP_VIEW_THRESHOLD: f32 = 0.01; // Minimum camera movement to trigger recompute
    
    println!("\nPhase 6 Convergence Heatmap Visualization Running");
    println!("Controls:");
    println!("  • Mouse Drag: Pan camera");
    println!("  • Mouse Wheel: Zoom in/out");
    println!("  • V: Toggle vector mode ({})", viz_mode.name());
    println!("  • G: Toggle globe visibility");
    println!("  • H: Toggle heatmap visibility");
    println!("  • B: Toggle bridge mode (Python/CUDA backend)");
    println!("  • C: Cycle colormap (Plasma → Viridis → Inferno → Magma → Turbo)");
    println!("  • R: Regenerate all data");
    println!("  • ESC: Exit");
    println!("═══════════════════════════════════════════════\n");
    
    // Frame timing - Sovereign 165Hz mode
    let mut last_frame = Instant::now();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    let mut current_fps = 165.0f32; // Match physics engine tick rate
    
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
                        println!("\nPhase 6 Shutdown");
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                                PhysicalKey::Code(KeyCode::KeyV) => {
                                    viz_mode = viz_mode.next();
                                    println!("Vector mode: {}", viz_mode.name());
                                }
                                PhysicalKey::Code(KeyCode::KeyG) => {
                                    show_globe = !show_globe;
                                    println!("Globe: {}", if show_globe { "visible" } else { "hidden" });
                                }
                                PhysicalKey::Code(KeyCode::KeyH) => {
                                    show_heatmap = !show_heatmap;
                                    println!("Heatmap: {}", if show_heatmap { "visible" } else { "hidden" });
                                }
                                PhysicalKey::Code(KeyCode::KeyB) => {
                                    use_bridge = !use_bridge && (grayscale_bridge.is_connected() || bridge_heatmap.is_connected());
                                    println!("Bridge mode: {}", if use_bridge { "ENABLED (Python/CUDA)" } else { "DISABLED (local)" });
                                }
                                PhysicalKey::Code(KeyCode::KeyC) => {
                                    // Cycle colormap (only affects grayscale mode)
                                    if use_grayscale_mode {
                                        grayscale_bridge.next_colormap(&queue);
                                    } else {
                                        println!("  (Colormap only available in grayscale bridge mode)");
                                    }
                                }
                                PhysicalKey::Code(KeyCode::KeyR) => {
                                    vector_field.generate_test_pattern();
                                    particle_system.upload_vector_field(&queue, &vector_field);
                                    particle_system.clear();
                                    let new_streamlines = streamline_gen.generate(&vector_field);
                                    streamline_renderer.upload(&queue, &new_streamlines);
                                    println!("All data regenerated");
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
                        
                        // PROFILING: Track each section
                        let profile_start = Instant::now();
                        
                        // Update camera
                        camera.update(dt);
                        
                        // Update LOD culler
                        let view_matrix = camera.view_matrix();
                        let proj_matrix = camera.projection_matrix(16.0 / 9.0);
                        let view_proj = proj_matrix * view_matrix;
                        lod_culler.update(view_proj, camera.position);
                        
                        let after_camera = profile_start.elapsed().as_micros();
                        
                        // Note: Budget adjustment moved to FPS counter block (periodic, not per-frame)
                        
                        // Update particle system
                        if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
                            particle_system.update(&queue, dt, &vector_field);
                        }
                        
                        let after_particles = profile_start.elapsed().as_micros();
                        
                        // Update streamline uniforms
                        if viz_mode == VizMode::Streamlines || viz_mode == VizMode::Both {
                            streamline_renderer.update(&queue, &vector_field, time);
                        }
                        
                        let after_streamlines = profile_start.elapsed().as_micros();
                        
                        // Update convergence heatmap
                        if show_heatmap {
                            if use_bridge && use_grayscale_mode {
                                // GRAYSCALE BRIDGE: 1-byte intensity, colormap in shader
                                grayscale_bridge.update(&device, &queue);
                            } else if use_bridge {
                                // RGBA BRIDGE: Pre-rendered texture from Python/CUDA
                                bridge_heatmap.update(&device, &queue);
                            } else {
                                // LOCAL MODE: Compute on CPU with view-change caching
                                let cam_moved = (camera.position - last_heatmap_cam_pos).length() > HEATMAP_VIEW_THRESHOLD;
                                let anim_tick = (time - last_heatmap_time).abs() > HEATMAP_ANIMATION_INTERVAL;
                                
                                if cam_moved || anim_tick {
                                    convergence_renderer.update(
                                        &queue,
                                        view_proj,
                                        globe_config.radius as f32,
                                        time,
                                        &mut lod_culler,
                                    );
                                    last_heatmap_cam_pos = camera.position;
                                    last_heatmap_time = time;
                                }
                            }
                        }
                        
                        let after_heatmap = profile_start.elapsed().as_micros();
                        
                        // Render frame
                        match render_frame(
                            &device,
                            &queue,
                            &surface,
                            &depth_texture,
                            &globe_pipeline,
                            &camera,
                            &mut particle_system,
                            &streamline_renderer,
                            &convergence_renderer,
                            &bridge_heatmap,
                            &grayscale_bridge,
                            show_globe,
                            show_heatmap,
                            use_bridge,
                            use_grayscale_mode,
                            viz_mode,
                        ) {
                            Ok(_) => {},
                            Err(e) => eprintln!("Render error: {}", e),
                        }
                        
                        let after_render = profile_start.elapsed().as_micros();
                        
                        // FPS counter
                        frame_count += 1;
                        if fps_timer.elapsed().as_secs() >= 2 {
                            current_fps = frame_count as f32 / fps_timer.elapsed().as_secs_f32();
                            
                            // Measure actual render time (not display-blocked)
                            let render_time_ms = dt * 1000.0;
                            let _theoretical_fps = if render_time_ms > 0.0 { 1000.0 / render_time_ms } else { 0.0 };
                            
                            // Adjust budgets based on FPS (periodic, Sovereign 165Hz target)
                            lod_culler.budget.adjust_for_fps(current_fps, 165.0);
                            
                            let cells = convergence_renderer.cell_count();
                            let high_intensity = convergence_renderer.high_intensity_count();
                            let budget_stress = lod_culler.stress_level() * 100.0;
                            
                            // Profile breakdown
                            let camera_us = after_camera;
                            let particles_us = after_particles - after_camera;
                            let streamlines_us = after_streamlines - after_particles;
                            let heatmap_us = after_heatmap - after_streamlines;
                            let render_us = after_render - after_heatmap;
                            
                            println!(
                                "FPS: {:.1} (frame: {:.1}ms) | Profile: cam={}us part={}us stream={}us heat={}us render={}us",
                                current_fps,
                                render_time_ms,
                                camera_us, particles_us, streamlines_us, heatmap_us, render_us
                            );
                            println!(
                                "  Particles: {} | Heatmap: {} cells ({} hot) | Budget: {:.0}%",
                                particle_system.particle_count(),
                                cells,
                                high_intensity,
                                budget_stress,
                            );
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
    
    // List all available adapters
    println!("  Available GPU adapters:");
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        println!("    - {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    }
    
    let surface = instance.create_surface(window)?;
    
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }).await.ok_or_else(|| anyhow::anyhow!("Failed to find GPU adapter"))?;
    
    // Log selected adapter
    let info = adapter.get_info();
    println!("  Selected: {} ({:?})", info.name, info.backend);
    
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

/// Create depth texture for proper layer ordering
fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    
    depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Globe rendering pipeline
struct GlobePipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,  // Phase 8: Exposed for particle system
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
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
        camera_bind_group_layout,  // Phase 8: Exposed for particle system
    })
}

/// Render a single frame
/// Phase 6: Added depth buffer and convergence heatmap layer
/// Phase 7: Added bridge heatmap mode for Python/CUDA backend
#[allow(clippy::too_many_arguments)]
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &wgpu::Surface<'_>,
    depth_view: &wgpu::TextureView,
    globe_pipeline: &GlobePipeline,
    camera: &GlobeCamera,
    particle_system: &mut ParticleSystem,
    streamline_renderer: &StreamlineRenderer,
    convergence_renderer: &ConvergenceRenderer,
    bridge_heatmap: &BridgeHeatmapRenderer,
    grayscale_bridge: &GrayscaleBridgeRenderer,
    show_globe: bool,
    show_heatmap: bool,
    use_bridge: bool,
    use_grayscale_mode: bool,
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
        label: Some("Phase 6 Render Encoder"),
    });
    
    // Run particle advection compute pass
    if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
        particle_system.advect(&mut encoder);
    }
    
    // PASS 1: Globe + Heatmap (with depth buffer)
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phase 6 Depth Pass"),
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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        // LAYER 1: Globe (writes depth)
        if show_globe {
            render_pass.set_pipeline(&globe_pipeline.render_pipeline);
            render_pass.set_bind_group(0, &globe_pipeline.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, globe_pipeline.vertex_buffer.slice(..));
            render_pass.set_index_buffer(globe_pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..globe_pipeline.index_count, 0, 0..1);
        }
        
        // LAYER 2: Convergence heatmap (reads depth, alpha blend)
        if show_heatmap {
            if use_bridge && use_grayscale_mode {
                grayscale_bridge.render(&mut render_pass);
            } else if use_bridge {
                bridge_heatmap.render(&mut render_pass);
            } else {
                convergence_renderer.render(&mut render_pass);
            }
        }
    }
    
    // PASS 2: Streamlines + Particles (no depth buffer - overlay)
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phase 6 Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Keep existing content
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None, // No depth for overlay layers
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        // LAYER 3: Streamlines (Phase 8: Now projected onto globe surface)
        if viz_mode == VizMode::Streamlines || viz_mode == VizMode::Both {
            streamline_renderer.render(&mut render_pass, &globe_pipeline.camera_bind_group);
        }
        
        // LAYER 4: Particles (Phase 8: Now projected onto globe surface)
        if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
            particle_system.render(&mut render_pass, &globe_pipeline.camera_bind_group);
        }
    }
    
    let submit_start = std::time::Instant::now();
    queue.submit(std::iter::once(encoder.finish()));
    let submit_time = submit_start.elapsed().as_micros();
    
    // Measure present time separately (blocks on compositor/vsync)
    let present_start = std::time::Instant::now();
    output.present();
    let present_time = present_start.elapsed().as_micros();
    
    // Log timing breakdown (useful for debugging WSL2 vs native)
    static FRAME_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = FRAME_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame % 120 == 0 {
        println!("  [GPU] submit={}us present={}us", submit_time, present_time);
    }
    
    Ok(())
}

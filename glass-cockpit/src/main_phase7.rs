/*!
 * HyperTensor Glass Cockpit - Phase 7: Telemetry Rails
 * 
 * GPU-accelerated telemetry visualization with:
 * - System Vitality Rail (left): CPU, memory, frame time sparklines
 * - Weather Metrics Rail (right): Temperature, wind, pressure gauges
 * - Terminal Output (bottom): Scrolling event log
 * 
 * Constitutional Compliance:
 * - Article II: Type-safe Rust with E-core affinity
 * - Article V: GPU-accelerated rendering
 * - Article VIII: <5% CPU, 60 FPS mandate
 * - Doctrine 3: Procedural rendering, no assets
 * 
 * Performance: LOD + culling + telemetry maintains 60+ FPS
 */

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use std::io::Write;
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
mod vorticity_renderer;  // Phase 8 Appendix G: Vorticity Ghost
mod ram_bridge_v2;
mod bridge_heatmap_renderer;
mod grayscale_bridge_renderer;
mod telemetry_rail;
mod event_log;
mod system_metrics;
mod terminal_renderer;
mod text;
mod interaction;
mod glass_chrome;
mod hud_overlay;
mod layout;
mod grid_renderer;
mod tensor_field;       // Phase 2: 3D tensor grid infrastructure
mod tensor_renderer;    // Phase 2: GPU tensor voxel cloud
mod text_gpu;           // Phase 2: GPU text atlas renderer
mod probe_panel;        // Phase 6b: Tensor inspection probe
mod timeline_scrubber;  // Phase 6b: Frame timeline navigation

use globe::{Icosphere, GlobeConfig, GlobeCamera};
use tile_fetcher::{TileFetcher, GibsConfig};
use vector_field::{VectorField, VectorFieldConfig};
use particle_system::{ParticleSystem, ParticleConfig};
use streamlines::{StreamlineGenerator, StreamlineRenderer, StreamlineConfig, StreamlineSpacing};
use lod::{LodCuller, LodConfig};
use convergence::ConvergenceConfig;
use convergence_renderer::ConvergenceRenderer;
use vorticity_renderer::VorticityRenderer;  // Phase 8 Appendix G
use bridge_heatmap_renderer::BridgeHeatmapRenderer;
use grayscale_bridge_renderer::GrayscaleBridgeRenderer;
use telemetry_rail::{TelemetryRail, SystemMetrics, PhysicsMetrics};
use event_log::{EventLog, EventCategory, EventLevel};
use system_metrics::MetricsCollector;
use interaction::{InteractionState, Ray, intersect_sphere, world_to_geo};
use glass_chrome::{GlassChrome, GlassPanelConfig};
use hud_overlay::HudOverlay;
use terminal_renderer::TerminalRenderer;
use layout::ViewLayout;
use grid_renderer::GridRenderer;                                 // Phase 1: Procedural Grid
use tensor_renderer::TensorRenderer;                            // Phase 2: 3D Voxel Cloud
use probe_panel::ProbePanel;                                    // Phase 6b: Tensor Probe
use timeline_scrubber::TimelineScrubber;                        // Phase 6b: Timeline
use text_gpu::{GpuTextRenderer, TextBuilder, GlyphInstance};   // Phase 2: GPU Typography

/// Simple file logger for debugging crashes
struct CrashLog {
    file: Option<std::fs::File>,
}

impl CrashLog {
    fn new() -> Self {
        // Write to desktop on Windows, current dir otherwise
        let path = if cfg!(target_os = "windows") {
            if let Ok(userprofile) = std::env::var("USERPROFILE") {
                format!("{}\\Desktop\\phase7_crash.log", userprofile)
            } else {
                "phase7_crash.log".to_string()
            }
        } else {
            "phase7_crash.log".to_string()
        };
        
        let file = std::fs::File::create(&path).ok();
        if file.is_some() {
            println!("Logging to: {}", path);
        }
        Self { file }
    }
    
    fn log(&mut self, msg: &str) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let line = format!("[{}] {}\n", timestamp, msg);
        
        if let Some(ref mut f) = self.file {
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
        println!("{}", msg);
    }
}

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

/// Phase 7 Telemetry Rails Visualization
#[allow(unused_assignments)]
fn main() -> Result<()> {
    // Initialize crash logger first
    let mut crash_log = CrashLog::new();
    crash_log.log("Phase 7 starting...");
    
    // Set panic hook to log panics
    std::panic::set_hook(Box::new(|panic_info| {
        let path = if cfg!(target_os = "windows") {
            if let Ok(userprofile) = std::env::var("USERPROFILE") {
                format!("{}\\Desktop\\phase7_panic.log", userprofile)
            } else {
                "phase7_panic.log".to_string()
            }
        } else {
            "phase7_panic.log".to_string()
        };
        
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = writeln!(f, "PANIC: {}", panic_info);
        }
        eprintln!("PANIC: {}", panic_info);
    }));
    
    crash_log.log("Panic hook installed");
    
    println!("HyperTensor Glass Cockpit v0.7.0 [Sovereign 165Hz]");
    println!("Phase 7: Telemetry Rails");
    println!("═══════════════════════════════════════════════");
    
    // Initialize event log early for logging startup
    let mut event_log = EventLog::new();
    event_log.info(EventCategory::System, "HyperTensor Glass Cockpit v0.7.0 starting");
    crash_log.log("Event log initialized");
    
    // STEP 1: Enforce E-core affinity (Doctrine 1: Computational Sovereignty)
    crash_log.log("[1/8] Enforcing E-core affinity...");
    println!("[1/8] Enforcing E-core affinity...");
    if let Err(e) = affinity::enforce_e_core_affinity() {
        crash_log.log(&format!("  ⚠ E-core affinity failed: {}", e));
        println!("  ⚠ E-core affinity failed: {}", e);
    } else {
        crash_log.log("  ✓ E-core affinity set (or platform handled)");
    }
    event_log.info(EventCategory::System, "E-core affinity check complete");
    
    // STEP 2: Initialize metrics collector
    crash_log.log("[2/8] Initializing system metrics collector...");
    println!("[2/8] Initializing system metrics collector...");
    let mut metrics_collector = MetricsCollector::new();
    metrics_collector.sample_now();
    let mem = metrics_collector.memory();
    crash_log.log(&format!("  Memory: {}/{}", mem.used, mem.total));
    println!("  ✓ Memory: {} / {}", 
        system_metrics::format_bytes(mem.used),
        system_metrics::format_bytes(mem.total));
    event_log.info(EventCategory::System, format!("Memory: {:.1}% used", mem.usage * 100.0));
    
    // STEP 3: Initialize tile fetcher
    crash_log.log("[3/8] Initializing NASA GIBS tile fetcher...");
    println!("[3/8] Initializing NASA GIBS tile fetcher...");
    let gibs_config = GibsConfig::default();
    let _tile_fetcher = match TileFetcher::new(gibs_config) {
        Ok(tf) => {
            crash_log.log("  ✓ Tile fetcher OK");
            tf
        },
        Err(e) => {
            crash_log.log(&format!("  ✗ Tile fetcher FAILED: {}", e));
            return Err(e);
        }
    };
    println!("  ✓ Tile cache initialized");
    event_log.debug(EventCategory::System, "GIBS tile fetcher initialized");
    
    // STEP 4: Generate synthetic vector field
    crash_log.log("[4/8] Generating vector field...");
    println!("[4/8] Generating vector field...");
    let field_config = VectorFieldConfig::for_zoom_level(5, 0.0, 35.0);
    let mut vector_field = VectorField::new(field_config);
    vector_field.generate_test_pattern();
    let stats = vector_field.stats.clone();
    crash_log.log(&format!("  Vector field {}x{}", field_config.grid_width, field_config.grid_height));
    println!("  ✓ Vector field: {}x{} grid", field_config.grid_width, field_config.grid_height);
    println!("  ✓ Max speed: {:.1} m/s, Max vorticity: {:.6}", stats.max_speed, stats.max_vorticity);
    event_log.info(EventCategory::Physics, format!("Vector field {}x{}, max speed {:.1} m/s", 
        field_config.grid_width, field_config.grid_height, stats.max_speed));
    
    // STEP 5: Create event loop and window
    crash_log.log("[5/8] Creating window...");
    println!("[5/8] Creating window...");
    let event_loop = EventLoop::new()?;
    crash_log.log("  EventLoop created");
    let window = Arc::new(WindowBuilder::new()
        .with_title("HyperTensor Glass Cockpit - Phase 7 Telemetry")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)?);
    crash_log.log("  Window created");
    println!("  ✓ Window created (1920×1080)");
    
    // STEP 6: Initialize GPU
    crash_log.log("[6/8] Initializing GPU pipeline...");
    println!("[6/8] Initializing GPU pipeline...");
    let (device, queue, surface, mut config) = match pollster::block_on(init_gpu(window.as_ref())) {
        Ok(result) => {
            crash_log.log("  ✓ GPU initialized");
            result
        },
        Err(e) => {
            crash_log.log(&format!("  ✗ GPU FAILED: {}", e));
            return Err(e);
        }
    };
    let window_clone = Arc::clone(&window);
    println!("  ✓ wgpu initialized");
    event_log.info(EventCategory::Render, "GPU pipeline initialized");
    
    // Initialize globe
    crash_log.log("  Creating globe mesh...");
    let globe_config = GlobeConfig::default();
    let icosphere = Icosphere::new(globe_config.clone());
    crash_log.log(&format!("  Globe: {} vertices", icosphere.vertex_count()));
    println!("  ✓ Globe mesh: {} vertices, {} triangles", 
        icosphere.vertex_count(), icosphere.triangle_count());
    
    // Create depth texture for proper layering
    crash_log.log("  Creating depth texture...");
    let (mut depth_texture, mut depth_view) = create_depth_texture_with_view(&device, &config);
    
    // Create camera
    crash_log.log("  Creating camera...");
    let mut camera = GlobeCamera::new();
    
    // Create LOD culler
    crash_log.log("  Creating LOD culler...");
    let mut lod_culler = LodCuller::new();
    lod_culler.config = LodConfig::globe_scale();
    
    // Create globe pipeline
    crash_log.log("  Creating globe pipeline...");
    let globe_pipeline = match create_globe_pipeline(&device, &config, &icosphere) {
        Ok(p) => {
            crash_log.log("  ✓ Globe pipeline OK");
            p
        },
        Err(e) => {
            crash_log.log(&format!("  ✗ Globe pipeline FAILED: {}", e));
            return Err(e);
        }
    };
    
    // Create tensor renderer (Phase 2: 3D Voxel Cloud)
    crash_log.log("  Creating TensorRenderer...");
    let tensor_renderer = match TensorRenderer::new(
        &device,
        &queue,
        config.format,
        &globe_pipeline.camera_bind_group_layout,
    ) {
        Ok(tr) => {
            crash_log.log("  ✓ TensorRenderer OK (Phase 2 tensor.wgsl integrated)");
            Some(tr)
        },
        Err(e) => {
            crash_log.log(&format!("  ⚠ TensorRenderer failed: {} (continuing without)", e));
            None
        }
    };
    
    // Create GPU text renderer (Phase 2: High-fidelity typography)
    crash_log.log("  Creating GpuTextRenderer...");
    let gpu_text_renderer = match GpuTextRenderer::new(&device, &queue, config.format) {
        Ok(gtr) => {
            crash_log.log("  ✓ GpuTextRenderer OK (Phase 2 text.wgsl integrated)");
            Some(gtr)
        },
        Err(e) => {
            crash_log.log(&format!("  ⚠ GpuTextRenderer failed: {} (continuing without)", e));
            None
        }
    };
    
    // STEP 7: Initialize particle system
    crash_log.log("[7/8] Initializing vector visualization...");
    println!("[7/8] Initializing vector visualization...");
    crash_log.log("  Creating particle system...");
    let mut particle_system = ParticleSystem::new(&device, &queue, config.format, &field_config);
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
    crash_log.log(&format!("  ✓ Particles: {} max", particle_system::MAX_PARTICLES));
    println!("  ✓ Particle system: {} max particles", particle_system::MAX_PARTICLES);
    
    // Generate streamlines
    crash_log.log("  Generating streamlines...");
    let streamline_config = StreamlineConfig {
        count: 200,
        max_length: 1_000_000.0,
        step_size: 20_000.0,
        spacing: StreamlineSpacing::Grid,
        ..Default::default()
    };
    let mut streamline_gen = StreamlineGenerator::new(streamline_config);
    let streamlines = streamline_gen.generate(&vector_field);
    crash_log.log(&format!("  ✓ Streamlines: {}", streamlines.len()));
    println!("  ✓ Streamlines: {} generated", streamlines.len());
    
    crash_log.log("  Creating streamline renderer...");
    let mut streamline_renderer = StreamlineRenderer::new(&device, config.format, streamline_config);
    streamline_renderer.upload(&queue, &streamlines);
    
    // Initialize convergence heatmap
    crash_log.log("  Creating convergence renderer...");
    let convergence_config = ConvergenceConfig::default();
    let mut convergence_renderer = ConvergenceRenderer::new(&device, &config, convergence_config);
    crash_log.log(&format!("  ✓ Convergence: {}x{}", convergence_config.resolution.0, convergence_config.resolution.1));
    println!("  ✓ Convergence field: {}x{} grid", 
        convergence_config.resolution.0, convergence_config.resolution.1);
    
    // Phase 8 Appendix G: Vorticity Ghost (volumetric overlay)
    crash_log.log("  Creating vorticity ghost renderer (Appendix G)...");
    let vorticity_renderer = VorticityRenderer::new(
        &device,
        config.format,
        convergence_renderer.cell_buffer(),
    );
    crash_log.log("  ✓ VorticityRenderer OK (32-step ray march)");
    println!("  ✓ Vorticity Ghost: Appendix G volumetric layer");
    
    // BRIDGE MODE - these are optional, don't fail if bridge not available
    crash_log.log("  Creating bridge renderers (optional)...");
    let mut bridge_heatmap = BridgeHeatmapRenderer::new(&device, &config, 1920, 1080);
    let mut grayscale_bridge = GrayscaleBridgeRenderer::new(&device, &queue, &config, 256, 128);
    let use_grayscale = grayscale_bridge.is_connected();
    crash_log.log(&format!("  Bridge connected: {}", use_grayscale || bridge_heatmap.is_connected()));
    
    // STEP 8: Initialize telemetry rails (Phase 7)
    crash_log.log("[8/8] Initializing telemetry rails...");
    println!("[8/8] Initializing telemetry rails...");
    crash_log.log("  Creating TelemetryRail...");
    let telemetry_rail = TelemetryRail::new(&device, &config);
    crash_log.log("  ✓ TelemetryRail OK");
    crash_log.log("  Creating TerminalRenderer...");
    let terminal_renderer = TerminalRenderer::new(&device, &queue, &config);
    crash_log.log("  ✓ TerminalRenderer OK");
    
    // Initialize glass chrome (SDF-based panels)
    crash_log.log("  Creating GlassChrome...");
    let glass_chrome = GlassChrome::new(&device, &config);
    crash_log.log("  ✓ GlassChrome OK");
    
    // Initialize unified HUD overlay
    crash_log.log("  Creating HudOverlay...");
    let hud_overlay = HudOverlay::new(&device, &queue, &config);
    crash_log.log("  ✓ HudOverlay OK");
    
    // Initialize procedural grid renderer (Phase 1 integration)
    crash_log.log("  Creating GridRenderer...");
    let grid_renderer = GridRenderer::new(&device, config.format);
    crash_log.log("  ✓ GridRenderer OK (grid.wgsl integrated)");
    
    // Initialize interaction state (raycaster)
    let mut interaction_state = InteractionState::new((config.width, config.height));
    crash_log.log("  ✓ InteractionState OK");
    
    // Phase 6b: Initialize probe panel (tensor inspection)
    crash_log.log("  Creating ProbePanel (Phase 6b)...");
    let mut probe_panel = ProbePanel::new(&device, config.format);
    crash_log.log("  ✓ ProbePanel OK");
    
    // Phase 6b: Initialize timeline scrubber (frame navigation)
    crash_log.log("  Creating TimelineScrubber (Phase 6b)...");
    let mut timeline_scrubber = TimelineScrubber::new(&device, config.format);
    crash_log.log("  ✓ TimelineScrubber OK");
    
    // Initialize layout system (Phase 1 integration)
    let mut view_layout = ViewLayout::new(config.width, config.height);
    crash_log.log(&format!("  ✓ ViewLayout OK: rails {}px, canvas {}px", 
        view_layout.left_rail.width as u32, view_layout.canvas.width as u32));
    
    event_log.set_visible_lines(terminal_renderer.lines_visible());
    println!("  ✓ System Vitality Rail (left panel)");
    println!("  ✓ Weather Metrics Rail (right panel)");
    println!("  ✓ Terminal Output (bottom pane, {} lines)", terminal_renderer.lines_visible());
    println!("  ✓ Probe Panel (Phase 6b - tensor inspection)");
    println!("  ✓ Timeline Scrubber (Phase 6b - frame navigation)");
    event_log.info(EventCategory::Render, "Telemetry rails initialized");
    event_log.info(EventCategory::System, "Phase 6b user agency components ready");
    
    crash_log.log("=== ALL INIT COMPLETE - ENTERING EVENT LOOP ===");
    
    // Visualization state
    let mut viz_mode = VizMode::Both;
    let mut show_globe = true;
    let mut show_grid = true;  // Phase 1 procedural grid background
    let mut show_heatmap = true;
    let mut show_telemetry = true;
    let mut show_tensor = true;  // Phase 2 tensor voxel cloud
    let mut use_bridge = use_grayscale || bridge_heatmap.is_connected();
    let mut use_grayscale_mode = use_grayscale;
    let start_time = Instant::now();
    
    // Mouse state
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    
    // Phase 8: Hover state for convergence feedback (lon, lat, intensity)
    let mut hover_state: Option<(f32, f32, f32)> = None;
    
    // Heatmap caching
    let mut last_heatmap_cam_pos = glam::Vec3::ZERO;
    let mut last_heatmap_time = 0.0_f32;
    const HEATMAP_ANIMATION_INTERVAL: f32 = 0.1;
    const HEATMAP_VIEW_THRESHOLD: f32 = 0.01;
    
    println!("\nPhase 7 Telemetry Rails Running");
    println!("Controls:");
    println!("  • Mouse Drag: Pan camera");
    println!("  • Mouse Wheel: Zoom in/out");
    println!("  • V: Toggle vector mode ({})", viz_mode.name());
    println!("  • G: Toggle globe visibility");
    println!("  • X: Toggle grid visibility (Phase 1)");
    println!("  • Z: Toggle tensor cloud (Phase 2)");
    println!("  • H: Toggle heatmap visibility");
    println!("  • T: Toggle telemetry rails");
    println!("  • B: Toggle bridge mode (Python/CUDA backend)");
    println!("  • C: Cycle colormap");
    println!("  • R: Regenerate all data");
    println!("  • Page Up/Down: Scroll terminal");
    println!("Phase 6b User Agency:");
    println!("  • Space: Play/Pause timeline");
    println!("  • Left/Right: Step frames");
    println!("  • Home: Go to start");
    println!("  • Click: Probe tensor node");
    println!("  • P: Close probe panel");
    println!("  • ESC: Exit");
    println!("═══════════════════════════════════════════════\n");
    
    event_log.info(EventCategory::System, "Phase 7 ready - all systems nominal");
    
    // Frame timing
    let mut last_frame = Instant::now();
    let mut frame_count = 0u64;
    let mut fps_timer = Instant::now();
    let mut current_fps = 165.0f32;
    let mut frame_time_ms = 0.0f32;
    
    // Main event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent { window_id: event_window_id, event } => {
                if event_window_id != window_clone.id() {
                    return;
                }
                match event {
                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            config.width = new_size.width;
                            config.height = new_size.height;
                            surface.configure(&device, &config);
                            
                            // Update layout system
                            view_layout.resize(new_size.width, new_size.height);
                            interaction_state.resize((new_size.width, new_size.height));
                            
                            // Recreate depth texture using helper function
                            let (new_tex, new_view) = create_depth_texture_with_view(&device, &config);
                            depth_texture = new_tex;
                            depth_view = new_view;
                            
                            event_log.debug(EventCategory::Render, format!(
                                "Resized to {}x{}", new_size.width, new_size.height
                            ));
                        }
                    }
                    WindowEvent::CloseRequested => {
                        event_log.info(EventCategory::System, "Shutdown requested");
                        println!("\nPhase 7 Shutdown");
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event: key_event, .. } => {
                        if key_event.state == ElementState::Pressed {
                            match key_event.physical_key {
                                PhysicalKey::Code(KeyCode::Escape) => elwt.exit(),
                                PhysicalKey::Code(KeyCode::KeyV) => {
                                    viz_mode = viz_mode.next();
                                    event_log.debug(EventCategory::Render, format!("Vector mode: {}", viz_mode.name()));
                                }
                                PhysicalKey::Code(KeyCode::KeyG) => {
                                    show_globe = !show_globe;
                                    event_log.debug(EventCategory::Render, format!("Globe: {}", if show_globe { "visible" } else { "hidden" }));
                                }
                                PhysicalKey::Code(KeyCode::KeyX) => {
                                    show_grid = !show_grid;
                                    event_log.debug(EventCategory::Render, format!("Grid: {}", if show_grid { "visible" } else { "hidden" }));
                                }
                                PhysicalKey::Code(KeyCode::KeyZ) => {
                                    show_tensor = !show_tensor;
                                    event_log.debug(EventCategory::Render, format!("Tensor cloud: {}", if show_tensor { "visible" } else { "hidden" }));
                                }
                                PhysicalKey::Code(KeyCode::KeyH) => {
                                    show_heatmap = !show_heatmap;
                                    event_log.debug(EventCategory::Render, format!("Heatmap: {}", if show_heatmap { "visible" } else { "hidden" }));
                                }
                                PhysicalKey::Code(KeyCode::KeyT) => {
                                    show_telemetry = !show_telemetry;
                                    event_log.info(EventCategory::Render, format!("Telemetry: {}", if show_telemetry { "visible" } else { "hidden" }));
                                }
                                PhysicalKey::Code(KeyCode::KeyB) => {
                                    use_bridge = !use_bridge && (grayscale_bridge.is_connected() || bridge_heatmap.is_connected());
                                    event_log.info(EventCategory::Bridge, format!("Bridge mode: {}", if use_bridge { "ENABLED" } else { "DISABLED" }));
                                }
                                PhysicalKey::Code(KeyCode::KeyC) => {
                                    if use_grayscale_mode {
                                        grayscale_bridge.next_colormap(&queue);
                                        event_log.debug(EventCategory::Render, format!("Colormap: {}", grayscale_bridge.colormap_name()));
                                    }
                                }
                                PhysicalKey::Code(KeyCode::KeyR) => {
                                    vector_field.generate_test_pattern();
                                    particle_system.upload_vector_field(&queue, &vector_field);
                                    particle_system.clear();
                                    let new_streamlines = streamline_gen.generate(&vector_field);
                                    streamline_renderer.upload(&queue, &new_streamlines);
                                    event_log.info(EventCategory::Physics, "All data regenerated");
                                }
                                PhysicalKey::Code(KeyCode::PageUp) => {
                                    event_log.scroll_up(5);
                                }
                                PhysicalKey::Code(KeyCode::PageDown) => {
                                    event_log.scroll_down(5);
                                }
                                PhysicalKey::Code(KeyCode::End) => {
                                    event_log.scroll_to_bottom();
                                }
                                // Phase 6b: Timeline controls
                                PhysicalKey::Code(KeyCode::Space) => {
                                    timeline_scrubber.toggle_playback();
                                    event_log.info(EventCategory::System, "Timeline playback toggled");
                                }
                                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                    timeline_scrubber.step_backward();
                                }
                                PhysicalKey::Code(KeyCode::ArrowRight) => {
                                    timeline_scrubber.step_forward();
                                }
                                PhysicalKey::Code(KeyCode::Home) => {
                                    timeline_scrubber.go_to_start();
                                    event_log.debug(EventCategory::System, "Timeline: Start");
                                }
                                // Phase 6b: Probe panel
                                PhysicalKey::Code(KeyCode::KeyP) => {
                                    probe_panel.close();
                                    event_log.debug(EventCategory::System, "Probe panel closed");
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        // Phase 6b: Handle probe panel click
                        if button == MouseButton::Left && state == ElementState::Pressed {
                            if let Some(pos) = interaction_state.mouse_pos {
                                let screen_pos = (pos.x as f32, pos.y as f32);
                                let screen_size = (config.width, config.height);
                                
                                // Check if clicking on timeline
                                if timeline_scrubber.hit_test(screen_pos, screen_size) {
                                    let (bar_x, _, bar_width, _) = timeline_scrubber.get_bar_bounds(screen_size);
                                    timeline_scrubber.start_drag(screen_pos.0, bar_x, bar_width);
                                } else if !probe_panel.is_visible() {
                                    // Phase 8: Use proper ray-globe intersection for probe click
                                    let view_proj = camera.projection_matrix(16.0 / 9.0) * camera.view_matrix();
                                    if let Some(ray) = interaction_state.cast_ray(&view_proj, camera.position) {
                                        let globe_radius = 1.0;
                                        if let Some(dist) = intersect_sphere(&ray, glam::Vec3::ZERO, globe_radius) {
                                            let hit_pos = ray.at(dist);
                                            let world_pos = hit_pos * 6371000.0; // Scale to Earth radius
                                            
                                            // Sample convergence at this position for node ID derivation
                                            let (lat, lon) = world_to_geo(hit_pos, globe_radius);
                                            let (intensity, _vorticity, _confidence) = convergence_renderer.sample_at(lon, lat);
                                            
                                            // Generate a node ID based on position
                                            let node_id = ((lat.abs() * 1000.0) as u32) ^ ((lon.abs() * 1000.0) as u32);
                                            
                                            probe_panel.on_click(node_id, world_pos, screen_pos);
                                            event_log.info(EventCategory::System, format!(
                                                "Probe node {} at ({:.2}°, {:.2}°) intensity: {:.2}", 
                                                node_id, lat.to_degrees(), lon.to_degrees(), intensity
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        
                        if button == MouseButton::Left {
                            mouse_pressed = state == ElementState::Pressed;
                            if !mouse_pressed {
                                timeline_scrubber.end_drag();
                            }
                            if !mouse_pressed {
                                last_mouse_pos = None;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        // Update interaction state for raycasting
                        interaction_state.set_mouse_pos(position);
                        
                        // Phase 6b: Update timeline drag
                        let screen_size = (config.width, config.height);
                        let (bar_x, _, bar_width, _) = timeline_scrubber.get_bar_bounds(screen_size);
                        timeline_scrubber.update_from_drag(position.x as f32, bar_x, bar_width);
                        
                        // Phase 8: Probe hover detection via ray-globe intersection
                        if !mouse_pressed && !probe_panel.is_visible() {
                            // Cast ray from mouse position through globe
                            let view_proj = camera.projection_matrix(16.0 / 9.0) * camera.view_matrix();
                            if let Some(ray) = interaction_state.cast_ray(&view_proj, camera.position) {
                                let globe_radius = 1.0; // Normalized globe radius
                                if let Some(dist) = intersect_sphere(&ray, glam::Vec3::ZERO, globe_radius) {
                                    let hit_pos = ray.at(dist);
                                    let (lat, lon) = world_to_geo(hit_pos, globe_radius);
                                    
                                    // Sample convergence field at this position
                                    let (intensity, _vorticity, _confidence) = convergence_renderer.sample_at(lon, lat);
                                    
                                    // Update probe hover state if intensity above threshold
                                    if intensity > 0.3 {
                                        probe_panel.on_hover(intensity);
                                        // Phase 8: Store hover state for convergence shader feedback
                                        hover_state = Some((lon, lat, intensity));
                                    } else {
                                        probe_panel.on_hover_exit();
                                        hover_state = None;
                                    }
                                } else {
                                    probe_panel.on_hover_exit();
                                    hover_state = None;
                                }
                            } else {
                                hover_state = None;
                            }
                        } else {
                            hover_state = None;
                        }
                        
                        if mouse_pressed && !timeline_scrubber.hit_test((position.x as f32, position.y as f32), screen_size) {
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
                        let now = Instant::now();
                        let dt = (now - last_frame).as_secs_f32();
                        last_frame = now;
                        frame_time_ms = dt * 1000.0;
                        
                        let time = start_time.elapsed().as_secs_f32();
                        
                        // Sample system metrics (rate-limited internally)
                        metrics_collector.sample();
                        
                        // Update camera
                        camera.update(dt);
                        
                        // Update LOD culler
                        let view_matrix = camera.view_matrix();
                        let proj_matrix = camera.projection_matrix(16.0 / 9.0);
                        let view_proj = proj_matrix * view_matrix;
                        lod_culler.update(view_proj, camera.position);
                        
                        // Update particle system
                        if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
                            particle_system.update(&queue, dt, &vector_field);
                        }
                        
                        // Update streamline uniforms
                        if viz_mode == VizMode::Streamlines || viz_mode == VizMode::Both {
                            streamline_renderer.update(&queue, &vector_field, time);
                        }
                        
                        // Update convergence heatmap
                        if show_heatmap {
                            if use_bridge && use_grayscale_mode {
                                grayscale_bridge.update(&device, &queue);
                            } else if use_bridge {
                                bridge_heatmap.update(&device, &queue);
                            } else {
                                let cam_moved = (camera.position - last_heatmap_cam_pos).length() > HEATMAP_VIEW_THRESHOLD;
                                let anim_tick = (time - last_heatmap_time).abs() > HEATMAP_ANIMATION_INTERVAL;
                                
                                if cam_moved || anim_tick || hover_state.is_some() {
                                    // Phase 8: Use update_with_hover for Appendix D feedback
                                    convergence_renderer.update_with_hover(
                                        &queue,
                                        view_proj,
                                        globe_config.radius as f32,
                                        time,
                                        &mut lod_culler,
                                        hover_state,
                                    );
                                    last_heatmap_cam_pos = camera.position;
                                    last_heatmap_time = time;
                                }
                            }
                        }
                        
                        // Phase 8 Appendix G: Update vorticity ghost uniforms
                        vorticity_renderer.update(&queue, view_proj, camera.position);
                        
                        // Update grid uniforms from camera
                        grid_renderer.update_uniforms(&queue, view_proj, camera.position);
                        
                        // Phase 6b: Update probe panel and timeline
                        let screen_size = (config.width, config.height);
                        let tether_pos = interaction_state.mouse_pos.map(|p| (p.x as f32, p.y as f32));
                        probe_panel.update(&queue, screen_size, time, tether_pos);
                        timeline_scrubber.update(&queue, screen_size, dt, time * 3.0);  // Heartbeat synced to time
                        
                        // Render frame
                        match render_frame(
                            &device,
                            &queue,
                            &surface,
                            &depth_view,
                            &globe_pipeline,
                            &grid_renderer,
                            &camera,
                            &mut particle_system,
                            &streamline_renderer,
                            &convergence_renderer,
                            &vorticity_renderer,  // Phase 8 Appendix G
                            &bridge_heatmap,
                            &grayscale_bridge,
                            tensor_renderer.as_ref(),  // Phase 2: Tensor Cloud
                            &telemetry_rail,
                            &terminal_renderer,
                            &glass_chrome,
                            gpu_text_renderer.as_ref(),  // Phase 2: GPU Text
                            &hud_overlay,
                            &probe_panel,              // Phase 6b: Probe Panel
                            &timeline_scrubber,        // Phase 6b: Timeline
                            &event_log,
                            &metrics_collector,
                            interaction_state.mouse_pos.map(|p| (p.x as f32, p.y as f32)),
                            show_globe,
                            show_grid,
                            show_heatmap,
                            show_tensor,  // Phase 2
                            show_telemetry,
                            use_bridge,
                            use_grayscale_mode,
                            viz_mode,
                            current_fps,
                            frame_time_ms,
                        ) {
                            Ok(_) => {},
                            Err(e) => {
                                eprintln!("Render error: {}", e);
                                event_log.error(EventCategory::Render, format!("Render error: {}", e));
                            }
                        }
                        
                        // FPS counter
                        frame_count += 1;
                        if fps_timer.elapsed().as_secs() >= 2 {
                            current_fps = frame_count as f32 / fps_timer.elapsed().as_secs_f32();
                            lod_culler.budget.adjust_for_fps(current_fps, 165.0);
                            
                            // Log periodic stats
                            if frame_count > 0 {
                                event_log.debug(EventCategory::Render, format!(
                                    "FPS: {:.1}, Particles: {}, Heatmap cells: {}",
                                    current_fps,
                                    particle_system.particle_count(),
                                    convergence_renderer.cell_count()
                                ));
                            }
                            
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
    let caps = surface.get_capabilities(&adapter);
    
    // Select best present mode: prefer Mailbox (low latency), fall back to Fifo (vsync)
    let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
        println!("  Using Mailbox present mode (low latency)");
        wgpu::PresentMode::Mailbox
    } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
        println!("  Using Immediate present mode");
        wgpu::PresentMode::Immediate
    } else {
        println!("  Using Fifo present mode (vsync)");
        wgpu::PresentMode::Fifo
    };
    
    let format = caps.formats.first().copied()
        .ok_or_else(|| anyhow::anyhow!("No supported surface formats"))?;
    println!("  Surface format: {:?}", format);
    
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);
    
    Ok((device, queue, surface, config))
}

/// Create depth texture (returns both texture and view for resize)
fn create_depth_texture_with_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> (wgpu::Texture, wgpu::TextureView) {
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
    
    let view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    (depth_texture, view)
}

/// Create depth texture (legacy - returns view only)
fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    create_depth_texture_with_view(device, config).1
}

/// Globe rendering pipeline
struct GlobePipeline {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bind_group_layout: wgpu::BindGroupLayout,  // Exposed for tensor_renderer
}

/// Create globe rendering pipeline
fn create_globe_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    icosphere: &Icosphere,
) -> Result<GlobePipeline> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Globe Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/globe.wgsl").into()),
    });
    
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
    
    // Phase 8: Appendix F - Split precision RTE camera uniforms
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct CameraUniforms {
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],      // Legacy camera_pos_ecef
        _padding: f32,
        // Phase 8: Split precision for sub-meter RTE
        camera_pos_high: [f32; 3], // High-order bits of camera position
        _padding1b: f32,
        camera_pos_low: [f32; 3],  // Low-order bits (remainder)
        _padding1c: f32,
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
                    0 => Float32x3,
                    1 => Float32x3,
                    2 => Float32x2,
                    3 => Float32x2,
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
        camera_bind_group_layout,
    })
}

/// Render a single frame with telemetry overlays
#[allow(clippy::too_many_arguments)]
fn render_frame(
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
    surface: &wgpu::Surface<'_>,
    depth_view: &wgpu::TextureView,
    globe_pipeline: &GlobePipeline,
    grid_renderer: &GridRenderer,
    camera: &GlobeCamera,
    particle_system: &mut ParticleSystem,
    streamline_renderer: &StreamlineRenderer,
    convergence_renderer: &ConvergenceRenderer,
    vorticity_renderer: &VorticityRenderer,  // Phase 8 Appendix G
    bridge_heatmap: &BridgeHeatmapRenderer,
    grayscale_bridge: &GrayscaleBridgeRenderer,
    tensor_renderer: Option<&TensorRenderer>,  // Phase 2: 3D Voxel Cloud
    _telemetry_rail: &TelemetryRail,  // Deprecated: superseded by hud_overlay
    terminal_renderer: &TerminalRenderer,
    glass_chrome: &GlassChrome,  // Phase 1: SDF Glass Panels (ACTIVE)
    gpu_text_renderer: Option<&GpuTextRenderer>,  // Phase 2: GPU Typography
    hud_overlay: &HudOverlay,
    probe_panel: &ProbePanel,           // Phase 6b: Tensor Probe Panel
    timeline_scrubber: &TimelineScrubber,  // Phase 6b: Timeline Scrubber
    event_log: &EventLog,
    metrics_collector: &MetricsCollector,
    mouse_pos: Option<(f32, f32)>,  // Mouse position for crosshair
    show_globe: bool,
    show_grid: bool,
    show_heatmap: bool,
    show_tensor: bool,  // Phase 2 tensor voxel cloud
    show_telemetry: bool,
    use_bridge: bool,
    use_grayscale_mode: bool,
    viz_mode: VizMode,
    current_fps: f32,
    frame_time_ms: f32,
) -> Result<()> {
    // Update camera uniform buffer
    // Phase 8: Appendix F - Split precision RTE
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct CameraUniforms {
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        _padding: f32,
        camera_pos_high: [f32; 3],
        _padding1b: f32,
        camera_pos_low: [f32; 3],
        _padding1c: f32,
        zoom: f32,
        aspect_ratio: f32,
        time: f32,
        _padding2: f32,
    }
    
    // Phase 8: Split camera position into high/low for RTE precision
    fn split_f32(val: f32) -> (f32, f32) {
        let high = val;
        let low = 0.0_f32;  // Our camera uses normalized coords, no extra precision needed
        (high, low)
    }
    
    let (cam_x_high, cam_x_low) = split_f32(camera.position.x);
    let (cam_y_high, cam_y_low) = split_f32(camera.position.y);
    let (cam_z_high, cam_z_low) = split_f32(camera.position.z);
    
    let view_matrix = camera.view_matrix();
    let proj_matrix = camera.projection_matrix(16.0 / 9.0);
    let view_proj = proj_matrix * view_matrix;
    
    let uniforms = CameraUniforms {
        view_proj: view_proj.to_cols_array_2d(),
        camera_pos: [camera.position.x, camera.position.y, camera.position.z],
        _padding: 0.0,
        camera_pos_high: [cam_x_high, cam_y_high, cam_z_high],
        _padding1b: 0.0,
        camera_pos_low: [cam_x_low, cam_y_low, cam_z_low],
        _padding1c: 0.0,
        zoom: camera.zoom,
        aspect_ratio: 16.0 / 9.0,
        time: 0.0,
        _padding2: 0.0,
    };
    
    queue.write_buffer(&globe_pipeline.camera_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    
    // Get surface texture
    let output = surface.get_current_texture()?;
    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    let mut encoder = _device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Phase 7 Render Encoder"),
    });
    
    // Run particle advection compute pass
    if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
        particle_system.advect(&mut encoder);
    }
    
    // PASS 1: Globe + Heatmap (with depth buffer)
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phase 7 Depth Pass"),
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
        
        // LAYER 0: Procedural Grid (Phase 1 - background)
        if show_grid {
            grid_renderer.render(&mut render_pass);
        }
        
        // LAYER 1: Globe
        if show_globe {
            render_pass.set_pipeline(&globe_pipeline.render_pipeline);
            render_pass.set_bind_group(0, &globe_pipeline.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, globe_pipeline.vertex_buffer.slice(..));
            render_pass.set_index_buffer(globe_pipeline.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..globe_pipeline.index_count, 0, 0..1);
        }
        
        // LAYER 2: Convergence heatmap (Floor Data)
        if show_heatmap {
            if use_bridge && use_grayscale_mode {
                grayscale_bridge.render(&mut render_pass);
            } else if use_bridge {
                bridge_heatmap.render(&mut render_pass);
            } else {
                convergence_renderer.render(&mut render_pass);
            }
        }
        
        // LAYER 3: Tensor Voxel Cloud (Phase 2 - Air Data)
        if show_tensor {
            if let Some(tr) = tensor_renderer {
                tr.render(&mut render_pass, &globe_pipeline.camera_bind_group);
            }
        }
    }
    
    // PASS 2: Streamlines + Particles (no depth buffer - overlay)
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phase 7 Overlay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        // LAYER 2.5: Vorticity Ghost (Phase 8 Appendix G - Volumetric Smoke)
        // Renders after globe geometry but before streamlines
        vorticity_renderer.render(&mut render_pass);
        
        // LAYER 3: Streamlines
        if viz_mode == VizMode::Streamlines || viz_mode == VizMode::Both {
            streamline_renderer.render(&mut render_pass);
        }
        
        // LAYER 4: Particles
        if viz_mode == VizMode::Particles || viz_mode == VizMode::Both {
            particle_system.render(&mut render_pass);
        }
    }
    
    // PASS 3: Telemetry overlays (UI layer - no depth)
    if show_telemetry {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Phase 7 Telemetry Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        // Build system metrics from collector
        let cpu = metrics_collector.cpu();
        let mem = metrics_collector.memory();
        
        // Get screen dimensions from surface texture
        let width = output.texture.width() as f32;
        let height = output.texture.height() as f32;
        
        // Calculate normalized metrics for HUD
        let cpu_norm = cpu.total.clamp(0.0, 1.0);
        let mem_norm = (mem.used as f32 / mem.total.max(1) as f32).clamp(0.0, 1.0);
        let fps_norm = (current_fps / 120.0).clamp(0.0, 1.0);
        let frame_norm = (frame_time_ms / 33.0).clamp(0.0, 1.0);
        
        // Physics placeholders (normalized)
        let temp_norm = 0.55;  // ~290K in 250-320 range
        let wind_norm = 0.31;  // 25/80
        let conv_norm = 0.45;
        let pres_norm = 0.55;  // 1013 in 980-1040 range
        
        // LAYER 5: Glass Chrome Panels (Phase 1 SDF - the container)
        // Renders semi-transparent glass panels with glowing borders
        glass_chrome.render_left_rail(&mut render_pass, queue, width, height);
        glass_chrome.render_right_rail(&mut render_pass, queue, width, height);
        glass_chrome.render_terminal_panel(&mut render_pass, queue, width, height);
        
        // LAYER 6: Unified HUD with all metrics (ON TOP of glass panels)
        hud_overlay.render_full_hud(
            &mut render_pass,
            queue,
            width,
            height,
            cpu_norm,
            mem_norm,
            current_fps,
            frame_time_ms,
            temp_norm,
            wind_norm,
            conv_norm,
            pres_norm,
        );
        
        // LAYER 7: GPU Text Rendering (Phase 2 - high-fidelity typography)
        if let Some(gtr) = gpu_text_renderer {
            // Build text for HUD display
            let mut builder = TextBuilder::new();
            
            // Left rail text - System Vitality
            builder.set_position(20.0, 30.0);
            builder.set_color([0.0, 0.8, 1.0, 1.0]);  // Sovereign Blue
            builder.add_text("SYSTEM VITALITY");
            
            builder.set_position(20.0, 55.0);
            builder.set_color([0.7, 1.0, 0.9, 1.0]);  // Cyan
            builder.add_text(&format!("CPU: {:.0}%", cpu_norm * 100.0));
            
            builder.set_position(20.0, 70.0);
            builder.set_color([1.0, 0.7, 0.3, 1.0]);  // Orange
            builder.add_text(&format!("MEM: {:.0}%", mem_norm * 100.0));
            
            builder.set_position(20.0, 85.0);
            builder.set_color([0.3, 1.0, 0.5, 1.0]);  // Green
            builder.add_text(&format!("FPS: {:.0}", current_fps));
            
            builder.set_position(20.0, 100.0);
            builder.set_color([1.0, 1.0, 0.5, 1.0]);  // Yellow
            builder.add_text(&format!("FRAME: {:.1}ms", frame_time_ms));
            
            // Right rail text - Weather Metrics
            builder.set_position(width - 190.0, 30.0);
            builder.set_color([0.0, 0.8, 1.0, 1.0]);  // Sovereign Blue
            builder.add_text("WEATHER METRICS");
            
            builder.set_position(width - 190.0, 55.0);
            builder.set_color([1.0, 0.5, 0.3, 1.0]);  // Red-orange
            builder.add_text(&format!("TEMP: {:.0}K", 250.0 + temp_norm * 70.0));
            
            builder.set_position(width - 190.0, 70.0);
            builder.set_color([0.6, 0.9, 1.0, 1.0]);  // Light blue
            builder.add_text(&format!("WIND: {:.0}m/s", wind_norm * 80.0));
            
            builder.set_position(width - 190.0, 85.0);
            builder.set_color([1.0, 0.4, 0.8, 1.0]);  // Magenta
            builder.add_text(&format!("CONV: {:.1}", conv_norm));
            
            builder.set_position(width - 190.0, 100.0);
            builder.set_color([0.7, 0.5, 1.0, 1.0]);  // Purple
            builder.add_text(&format!("PRES: {:.0}hPa", 980.0 + pres_norm * 60.0));
            
            // Phase 6b: Probe Panel Labels (when visible)
            if let Some(probe_data) = probe_panel.get_display_data() {
                let panel_x = width - 320.0 - 30.0;  // Match probe_panel.rs panel_pos
                let panel_y = 100.0;
                
                // Header
                builder.set_position(panel_x + 15.0, panel_y + 15.0);
                builder.set_color([0.0, 0.8, 1.0, 1.0]);  // Sovereign Blue
                builder.add_text("PROBABILITY PROBE");
                
                builder.set_position(panel_x + 15.0, panel_y + 30.0);
                builder.set_color([0.6, 0.7, 0.8, 1.0]);  // Grey
                builder.add_text(&format!("Node {} ({:.2}, {:.2})", 
                    probe_data.node_id, probe_data.geo_coords.0, probe_data.geo_coords.1));
                
                // Contribution bars with labels (the key insight from Appendix D)
                let bar_start_y = panel_y + 60.0;
                let bar_spacing = 35.0;
                
                for (i, contrib) in probe_data.contributions.iter().enumerate() {
                    let bar_y = bar_start_y + (i as f32) * bar_spacing;
                    
                    // Variable name on left
                    builder.set_position(panel_x + 15.0, bar_y + 4.0);
                    builder.set_color(contrib.color);
                    builder.add_text(contrib.name);
                    
                    // Percentage on right (THE CRITICAL LABEL)
                    builder.set_position(panel_x + 240.0, bar_y + 4.0);
                    builder.set_color([1.0, 1.0, 1.0, 1.0]);  // White
                    builder.add_text(&format!("{:.1}%", contrib.weight * 100.0));
                }
                
                // Temporal curve section header
                let curve_y = bar_start_y + 4.0 * bar_spacing + 20.0;
                builder.set_position(panel_x + 15.0, curve_y - 5.0);
                builder.set_color([0.0, 0.8, 1.0, 1.0]);
                builder.add_text("TEMPORAL CONFIDENCE");
                
                // Frame labels under curve
                builder.set_position(panel_x + 15.0, curve_y + 85.0);
                builder.set_color([0.5, 0.6, 0.7, 1.0]);
                builder.add_text("0      25k     50k     75k    100k");
            }
            
            // Phase 6b: Timeline frame counter
            let timeline_frame = timeline_scrubber.current_frame();
            let timeline_max = timeline_scrubber.max_frames();
            builder.set_position(width / 2.0 - 60.0, height - 95.0);
            builder.set_color([0.0, 0.9, 1.0, 1.0]);  // Bright cyan
            builder.add_text(&format!("FRAME {}/{}", timeline_frame, timeline_max));
            
            let instances = builder.build();
            gtr.render(queue, &mut render_pass, &instances);
        }
        
        // Phase 6b: Render timeline scrubber (bottom, above terminal)
        timeline_scrubber.render(&mut render_pass);
        
        // Phase 6b: Render probe panel (right side overlay)
        probe_panel.render(&mut render_pass);
        
        // Render crosshair at mouse position (Phase 8 interaction)
        if let Some((mx, my)) = mouse_pos {
            hud_overlay.render_crosshair(&mut render_pass, queue, mx, my);
        }
        
        // Render terminal (bottom pane)
        terminal_renderer.render(&mut render_pass, queue, event_log);
    }
    
    queue.submit(std::iter::once(encoder.finish()));
    output.present();
    
    Ok(())
}

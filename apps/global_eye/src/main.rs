//! Global Eye - HyperTensor Atmospheric Observation Platform
//!
//! A global-scale weather visualization system that shares the ontic_bridge IPC
//! protocol and ontic_core physics transforms with Glass Cockpit.
//!
//! # Phase 1C-10: Complete Integration
//!
//! - 1A: Python fetches HRRR data → tensors
//! - 1B: Python writes to /dev/shm/hyper_weather_v1
//! - 1C: Rust reads and renders on GPU with winit + wgpu
//!
//! # Usage
//!
//! ```bash
//! # Terminal 1: Start weather stream
//! python -m tensornet.sovereign.weather_stream
//!
//! # Terminal 2: Run visualizer
//! cargo run -p global_eye
//! ```

mod camera;
mod globe_mesh;
mod renderer;
mod wind_texture;

use anyhow::Result;
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

/// Main entry point for Global Eye.
fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║               GLOBAL EYE - Weather Observation                ║");
    println!("║                    HyperTensor Platform                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Check weather bridge status
    let reader = ontic_bridge::WeatherReader::new();
    
    if reader.is_available() {
        println!("  ✓ Weather bridge detected at {}", ontic_bridge::WEATHER_SHM_PATH);
        
        // Try to read current frame
        match reader.read_current() {
            Ok(frame) => {
                print_frame_info(&frame);
            }
            Err(e) => {
                println!("  ⚠ Could not read weather frame: {}", e);
            }
        }
    } else {
        println!("  ⊘ Weather bridge not available (will show demo pattern)");
        println!("    To enable live data: python -m tensornet.sovereign.weather_stream");
    }
    
    println!();
    println!("  Controls:");
    println!("    • Left mouse drag: Rotate globe");
    println!("    • Mouse wheel:     Zoom in/out");
    println!("    • Space:           Toggle auto-rotate");
    println!("    • R:               Reset camera");
    println!("    • Escape:          Exit");
    println!();
    
    // Create window and event loop
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Global Eye - HyperTensor Weather Visualization")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)?;
    
    // Leak window to get 'static lifetime (same pattern as glass_cockpit)
    let window: &'static Window = Box::leak(Box::new(window));
    
    // Create renderer
    println!("  Initializing renderer...");
    let mut renderer = pollster::block_on(renderer::GlobeRenderer::new(window))?;
    println!("  ✓ Renderer initialized");
    
    // Input state
    let mut mouse_pressed = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    let mut auto_rotate = true;
    
    // Run event loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                
                WindowEvent::Resized(size) => {
                    renderer.resize(size.width, size.height);
                }
                
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(key),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => match key {
                    KeyCode::Escape => elwt.exit(),
                    KeyCode::Space => {
                        auto_rotate = !auto_rotate;
                        println!(
                            "  Auto-rotate: {}",
                            if auto_rotate { "ON" } else { "OFF" }
                        );
                    }
                    KeyCode::KeyR => {
                        renderer.camera = camera::Camera::default();
                        println!("  Camera reset");
                    }
                    _ => {}
                },
                
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        mouse_pressed = state == ElementState::Pressed;
                        if mouse_pressed {
                            auto_rotate = false;
                        }
                    }
                }
                
                WindowEvent::CursorMoved { position, .. } => {
                    if mouse_pressed {
                        if let Some((last_x, last_y)) = last_mouse_pos {
                            let dx = (position.x - last_x) as f32;
                            let dy = (position.y - last_y) as f32;
                            renderer.camera.rotate(dx * 0.005, dy * 0.005);
                        }
                    }
                    last_mouse_pos = Some((position.x, position.y));
                }
                
                WindowEvent::MouseWheel { delta, .. } => {
                    let zoom = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 0.3,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    renderer.camera.zoom(-zoom);
                }
                
                WindowEvent::RedrawRequested => {
                    // Auto-rotate when not interacting
                    if auto_rotate {
                        renderer.camera.rotate(0.002, 0.0);
                    }
                    
                    // Check for new weather data
                    renderer.update_weather();
                    
                    // Render frame
                    match renderer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = window.inner_size();
                            renderer.resize(size.width, size.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            eprintln!("Out of memory!");
                            elwt.exit();
                        }
                        Err(e) => {
                            eprintln!("Render error: {:?}", e);
                        }
                    }
                }
                
                _ => {}
            },
            
            Event::AboutToWait => {
                // Request continuous redraws
                window.request_redraw();
            }
            
            _ => {}
        }
    })?;
    
    Ok(())
}

/// Print information about a weather frame.
fn print_frame_info(frame: &ontic_bridge::WeatherFrame) {
    let header = frame.header();
    
    // Copy fields from packed struct to avoid unaligned access
    let grid_w = header.grid_w;
    let grid_h = header.grid_h;
    let frame_number = header.frame_number;
    let timestamp = header.timestamp;
    let max_wind = header.max_wind_speed;
    let mean_wind = header.mean_wind_speed;
    let lat_min = header.lat_min;
    let lat_max = header.lat_max;
    let lon_min = header.lon_min;
    let lon_max = header.lon_max;
    
    println!();
    println!("  ┌─────────────────────────────────────┐");
    println!("  │       Weather Frame Info            │");
    println!("  ├─────────────────────────────────────┤");
    println!("  │ Grid:      {}×{}                ", grid_w, grid_h);
    println!("  │ Frame:     {}                     ", frame_number);
    println!("  │ Timestamp: {}                 ", timestamp);
    println!("  │ Max Wind:  {:.1} m/s            ", max_wind);
    println!("  │ Mean Wind: {:.1} m/s            ", mean_wind);
    println!("  │ Lat Range: {:.2}° to {:.2}°      ", lat_min, lat_max);
    println!("  │ Lon Range: {:.2}° to {:.2}°     ", lon_min, lon_max);
    println!("  └─────────────────────────────────────┘");
    
    // Sample some wind values
    let u = frame.u_field();
    
    if !u.is_empty() {
        let center_idx = u.len() / 2;
        let center_speed = frame.magnitude_at(center_idx);
        println!("  Center wind speed: {:.1} m/s", center_speed);
    }
}

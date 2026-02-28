/*!
 * Ontic Glass Cockpit - Phase 1: Grid Visualization
 * 
 * Sovereign observation layer for atmospheric intelligence.
 * 
 * Constitutional Compliance:
 * - Article II: E-core affinity enforced at startup
 * - Article VIII: <5% CPU target (flexible during development)
 * - Article IX: Zero network activity (satellite tiles disabled by default)
 */

use anyhow::Result;
use winit::{
    event::{Event, WindowEvent, MouseButton, ElementState, MouseScrollDelta},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod affinity;
mod bridge;
mod camera;
mod ghost_plane;
mod layout;
mod overlay;
mod ram_bridge_v2;
mod renderer;
mod swarm_renderer;
mod telemetry;
mod tensor_colormap;
mod tensor_field;
mod tensor_renderer;
mod text;
mod text_gpu;
mod tube_geometry;

use bridge::SovereignBridge;
use renderer::Renderer;
use telemetry::FrameTiming;

/// Phase 1 Entry Point
fn main() -> Result<()> {
    // Initialize logging (only in debug builds)
    #[cfg(feature = "debug-logging")]
    env_logger::init();
    
    println!("Ontic Glass Cockpit v0.1.0");
    println!("Phase 1: Grid Visualization");
    println!("═══════════════════════════════════════════════");
    
    // STEP 1: Enforce E-core affinity (Doctrine 1)
    println!("[1/4] Enforcing E-core affinity...");
    #[cfg(target_os = "windows")]
    {
        affinity::enforce_e_core_affinity()?;
        println!("  ✓ Process pinned to E-cores (16-31)");
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        println!("  ⚠ E-core affinity only supported on Windows");
    }
    
    // STEP 2: Initialize RAM bridge connection
    println!("[2/4] Connecting to RAM bridge...");
    let bridge = match SovereignBridge::connect("/dev/shm/sovereign_bridge") {
        Ok(b) => {
            let frame_idx = b.read_frame_index();
            println!("  ✓ RAM bridge connected (frame {})", frame_idx);
            Some(b)
        }
        Err(e) => {
            println!("  ⚠ RAM bridge not available: {}", e);
            None
        }
    };
    
    // STEP 3: Initialize window and GPU pipeline
    println!("[3/4] Initializing GPU pipeline...");
    let event_loop = EventLoop::new()?;
    let window_obj = WindowBuilder::new()
        .with_title("Ontic Glass Cockpit - Phase 1")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop)?;
    
    let window: &'static Window = Box::leak(Box::new(window_obj));
    let mut renderer = pollster::block_on(Renderer::new(window))?;
    println!("  ✓ wgpu initialized");
    println!("  → GPU: {}", renderer.adapter_info());
    
    // STEP 4: Enter render loop
    println!("[4/4] Starting render loop...");
    println!("═══════════════════════════════════════════════");
    println!("Controls: Left drag=Rotate | Right drag=Pan | Scroll=Zoom | ESC=Exit");
    println!();
    
    let mut frame_timing = FrameTiming::new();
    let mut frame_count = 0u64;
    let mut mouse_pressed_left = false;
    let mut mouse_pressed_right = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;
    
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        
        match event {
            Event::WindowEvent { window_id, event } => {
                if window_id != window.id() {
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => {
                        println!("\nPhase 1 Shutdown - {} frames", frame_count);
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        use winit::keyboard::{PhysicalKey, KeyCode};
                        if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            elwt.exit();
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        match button {
                            MouseButton::Left => {
                                mouse_pressed_left = state == ElementState::Pressed;
                                if !mouse_pressed_left {
                                    last_mouse_pos = None;
                                }
                            }
                            MouseButton::Right => {
                                mouse_pressed_right = state == ElementState::Pressed;
                                if !mouse_pressed_right {
                                    last_mouse_pos = None;
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let current_pos = (position.x, position.y);
                        
                        if let Some(last_pos) = last_mouse_pos {
                            let delta_x = (current_pos.0 - last_pos.0) as f32;
                            let delta_y = (current_pos.1 - last_pos.1) as f32;
                            
                            if mouse_pressed_left {
                                renderer.camera.orbit(delta_x * 0.005, -delta_y * 0.005);
                            } else if mouse_pressed_right {
                                renderer.camera.pan(-delta_x, delta_y);
                            }
                        }
                        
                        last_mouse_pos = Some(current_pos);
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let scroll_amount = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y * 2.0,
                            MouseScrollDelta::PixelDelta(pos) => (pos.y / 50.0) as f32,
                        };
                        renderer.camera.zoom(-scroll_amount);
                    }
                    WindowEvent::Resized(size) => {
                        renderer.resize(size.width, size.height);
                    }
                    WindowEvent::RedrawRequested => {
                        let frame_start = std::time::Instant::now();
                        let telemetry = bridge.as_ref().map(|b| b.read_telemetry());
                        
                        match renderer.render(std::time::Duration::from_millis(0), telemetry) {
                            Ok(_) => {
                                let frame_duration = frame_start.elapsed();
                                frame_count += 1;
                                frame_timing.record(frame_duration);
                                
                                if frame_count % 60 == 0 {
                                    println!(
                                        "Frame {}: {:.2}ms | FPS: {:.1} | Stability: {:.2}",
                                        frame_count,
                                        frame_timing.mean_ms(),
                                        1000.0 / frame_timing.mean_ms(),
                                        frame_timing.stability_score()
                                    );
                                }
                            }
                            Err(e) => eprintln!("Render error: {}", e),
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;
    
    Ok(())
}

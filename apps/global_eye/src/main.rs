//! Global Eye - HyperTensor Atmospheric Observation Platform
//!
//! A global-scale weather visualization system that shares the hyper_bridge IPC
//! protocol and hyper_core physics transforms with Glass Cockpit.
//!
//! # Phase 1 Implementation
//!
//! - 1A: Python fetches HRRR data → tensors
//! - 1B: Python writes to /dev/shm/hyper_weather_v1
//! - 1C: Rust reads and renders on GPU
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

mod wind_texture;

use anyhow::Result;
use hyper_bridge::{WeatherReader, WeatherFrame};

/// Main entry point for Global Eye.
fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║               GLOBAL EYE - Weather Observation                ║");
    println!("║                    HyperTensor Platform                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Check weather bridge status
    let reader = WeatherReader::new();
    
    if reader.is_available() {
        println!("  ✓ Weather bridge detected at {}", hyper_bridge::WEATHER_SHM_PATH);
        
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
        println!("  ⊘ Weather bridge not available");
        println!("    Run: python -m tensornet.sovereign.weather_stream");
    }
    
    println!();
    println!("  Shared Infrastructure:");
    println!("    • hyper_bridge: Weather protocol v{}", hyper_bridge::WEATHER_VERSION);
    println!("    • hyper_core:   Morton encoding ready");
    println!();
    
    // Verify Morton encoding works
    let morton = hyper_core::transforms::morton::encode_2d(100, 200);
    let (x, y) = hyper_core::transforms::morton::decode_2d(morton);
    println!("  Morton test: (100, 200) → {} → ({}, {})", morton, x, y);
    
    println!();
    println!("  Phase 1 Status:");
    println!("    [✓] 1A: HRRR data pipeline (Python)");
    println!("    [✓] 1B: Shared memory bridge (IPC)");
    println!("    [~] 1C: WGPU visualization (in progress)");
    println!();
    
    // TODO: Add actual rendering loop once windowing is set up
    println!("  Next: Add winit window + wgpu renderer");
    
    Ok(())
}

/// Print information about a weather frame.
fn print_frame_info(frame: &WeatherFrame) {
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

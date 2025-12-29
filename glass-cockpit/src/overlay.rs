// Phase 2: Telemetry Overlay System
// Reads RAM bridge telemetry and manages display state
// Constitutional compliance: Doctrine 2 (RAM bridge), Doctrine 3 (explicit state)

use crate::bridge::{Telemetry, SovereignBridge};
use std::time::{Duration, Instant};

/// Simple performance statistics tracker
// Phase 2 scaffolding: Frame timing analysis for telemetry overlay
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    frame_times: Vec<f32>,
    frame_index: usize,
    max_samples: usize,
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            frame_times: vec![0.0; 100],
            frame_index: 0,
            max_samples: 100,
        }
    }

    fn record_frame(&mut self, duration: Duration) {
        let ms = duration.as_secs_f32() * 1000.0;
        self.frame_times[self.frame_index] = ms;
        self.frame_index = (self.frame_index + 1) % self.max_samples;
    }

    fn variance(&self) -> f32 {
        let mean: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        let variance: f32 = self.frame_times.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.frame_times.len() as f32;
        variance
    }
}

/// Telemetry data snapshot for display
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    /// P-core usage percentage (0-100)
    pub p_core_usage: f32,
    
    /// E-core usage percentage (0-100)
    pub e_core_usage: f32,
    
    // Phase 2 scaffolding: Telemetry display fields
    #[allow(dead_code)]
    /// Current frames per second
    pub fps: f32,
    
    #[allow(dead_code)]
    /// Frame time in milliseconds
    pub frame_time_ms: f32,
    
    /// Memory usage in MB
    pub memory_mb: f32,
    
    /// Stability score (0.0-1.0, higher is better)
    pub stability: f32,
    
    #[allow(dead_code)]
    /// Timestamp of this snapshot
    pub timestamp: Instant,
}

impl Default for TelemetrySnapshot {
    fn default() -> Self {
        Self {
            p_core_usage: 0.0,
            e_core_usage: 0.0,
            fps: 0.0,
            frame_time_ms: 0.0,
            memory_mb: 0.0,
            stability: 1.0,
            timestamp: Instant::now(),
        }
    }
}

/// Manages telemetry display and updates
pub struct TelemetryOverlay {
    /// Current telemetry snapshot
    pub current: TelemetrySnapshot,
    
    /// Performance statistics accumulator
    stats: PerformanceStats,
    
    /// Last update time
    last_update: Instant,
    
    /// Update interval (how often to refresh telemetry)
    update_interval: Duration,
    
    /// Frame time history for FPS calculation (ring buffer)
    frame_times: Vec<Duration>,
    frame_time_index: usize,
    
    /// Whether overlay is visible
    pub visible: bool,
    
    /// RAM bridge connection (None if not available)
    bridge: Option<SovereignBridge>,
    
    // Phase 2 scaffolding: Simulated mode tracking for overlay display
    #[allow(dead_code)]
    /// Whether using simulated data (fallback mode)
    simulated: bool,
}

impl TelemetryOverlay {
    /// Create a new telemetry overlay
    pub fn new() -> Self {
        // Attempt to connect to RAM bridge (graceful fallback if unavailable)
        let bridge = SovereignBridge::connect("/dev/shm/sovereign_bridge")
            .inspect_err(|e| {
                eprintln!("[Telemetry] RAM bridge not available: {} - using simulated data", e);
            })
            .ok();
        
        let simulated = bridge.is_none();
        if !simulated {
            println!("[Telemetry] Connected to RAM bridge at /dev/shm/sovereign_bridge");
        }
        
        Self {
            current: TelemetrySnapshot::default(),
            stats: PerformanceStats::new(),
            last_update: Instant::now(),
            update_interval: Duration::from_millis(100), // Update 10x per second
            frame_times: vec![Duration::from_secs(0); 60], // Track 60 frames
            frame_time_index: 0,
            visible: true,
            bridge,
            simulated,
        }
    }

    /// Update telemetry with new frame timing
    pub fn update(&mut self, frame_duration: Duration) {
        // Record frame time
        self.frame_times[self.frame_time_index] = frame_duration;
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.len();
        self.stats.record_frame(frame_duration);

        // Check if it's time to update telemetry snapshot
        let now = Instant::now();
        if now.duration_since(self.last_update) >= self.update_interval {
            self.update_snapshot();
            self.last_update = now;
        }
    }

    /// Update the current telemetry snapshot
    fn update_snapshot(&mut self) {
        // Calculate FPS from frame time history
        let total_time: Duration = self.frame_times.iter().sum();
        let avg_frame_time = total_time.as_secs_f32() / self.frame_times.len() as f32;
        let fps = if avg_frame_time > 0.0 {
            1.0 / avg_frame_time
        } else {
            0.0
        };
        let frame_time_ms = avg_frame_time * 1000.0;

        // Try to read from RAM bridge, fall back to simulation if unavailable
        let (p_core_usage, e_core_usage, memory_mb, stability) = if let Some(bridge_data) = self.read_from_bridge() {
            // Real data from Sovereign Engine
            (
                bridge_data.p_core_utilization * 100.0,
                bridge_data.e_core_utilization * 100.0,
                bridge_data.memory_usage_gb * 1024.0, // Convert GB to MB
                bridge_data.stability_score,
            )
        } else {
            // Simulated data (fallback)
            let p_core = (frame_time_ms / 16.67 * 100.0).clamp(0.0, 100.0); // Relative to 60fps
            let e_core = (p_core * 0.3).clamp(0.0, 100.0); // E-cores at 30% of P-core load
            let memory = 150.0 + (frame_time_ms * 2.0); // Base + load-dependent
            
            // Stability score based on frame time variance
            let variance = self.stats.variance();
            let stab = if variance > 0.0 {
                (1.0 / (1.0 + variance * 100.0)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            
            (p_core, e_core, memory, stab)
        };

        self.current = TelemetrySnapshot {
            p_core_usage,
            e_core_usage,
            fps,
            frame_time_ms,
            memory_mb,
            stability,
            timestamp: Instant::now(),
        };
    }

    /// Read telemetry from RAM bridge
    fn read_from_bridge(&self) -> Option<Telemetry> {
        self.bridge.as_ref().map(|b| b.read_telemetry())
    }
    
    // Phase 2 scaffolding: Overlay status methods for telemetry UI
    #[allow(dead_code)]
    /// Check if using simulated data
    pub fn is_simulated(&self) -> bool {
        self.simulated
    }
    
    #[allow(dead_code)]
    /// Get bridge connection status
    pub fn bridge_status(&self) -> &str {
        if self.simulated {
            "Simulated (RAM bridge unavailable)"
        } else {
            "Connected to /dev/shm/sovereign_bridge"
        }
    }

    #[allow(dead_code)]
    /// Toggle overlay visibility
    pub fn toggle_visibility(&mut self) {
        self.visible = !self.visible;
    }

    #[allow(dead_code)]
    /// Get formatted string for display
    pub fn format_p_core(&self) -> String {
        format!("P-Core: {:.1}%", self.current.p_core_usage)
    }

    #[allow(dead_code)]
    pub fn format_e_core(&self) -> String {
        format!("E-Core: {:.1}%", self.current.e_core_usage)
    }

    #[allow(dead_code)]
    pub fn format_fps(&self) -> String {
        format!("FPS: {:.1}", self.current.fps)
    }

    #[allow(dead_code)]
    pub fn format_frame_time(&self) -> String {
        format!("{:.2}ms", self.current.frame_time_ms)
    }

    #[allow(dead_code)]
    pub fn format_memory(&self) -> String {
        format!("RAM: {:.1}MB", self.current.memory_mb)
    }

    #[allow(dead_code)]
    pub fn format_stability(&self) -> String {
        let percent = self.current.stability * 100.0;
        format!("Stability: {:.1}%", percent)
    }

    #[allow(dead_code)]
    /// Get color for stability display (green=good, yellow=ok, red=bad)
    pub fn stability_color(&self) -> [f32; 3] {
        let s = self.current.stability;
        if s > 0.8 {
            [0.2, 0.8, 0.2] // Green
        } else if s > 0.5 {
            [0.8, 0.8, 0.2] // Yellow
        } else {
            [0.8, 0.2, 0.2] // Red
        }
    }

    #[allow(dead_code)]
    /// Get color for core usage (blue gradient based on load)
    pub fn core_usage_color(&self, usage: f32) -> [f32; 3] {
        let intensity = (usage / 100.0).clamp(0.0, 1.0);
        [0.2, 0.4 + intensity * 0.4, 0.8] // Blue with varying intensity
    }

    #[allow(dead_code)]
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = PerformanceStats::new();
        self.frame_times.fill(Duration::from_secs(0));
        self.frame_time_index = 0;
    }

    #[allow(dead_code)]
    /// Get performance statistics
    pub fn get_stats(&self) -> &PerformanceStats {
        &self.stats
    }

    #[allow(dead_code)]
    /// Set update interval
    pub fn set_update_interval(&mut self, interval: Duration) {
        self.update_interval = interval;
    }
}

impl Default for TelemetryOverlay {
    fn default() -> Self {
        Self::new()
    }
}

// Phase 2 scaffolding: Telemetry card UI layout system
#[allow(dead_code)]
/// Telemetry card layout information
#[derive(Debug, Clone, Copy)]
pub struct TelemetryCard {
    /// Card title
    pub title: &'static str,
    
    /// Y position in rail (pixels from top)
    pub y_offset: f32,
    
    /// Card height
    pub height: f32,
    
    /// Card color (RGB)
    pub color: [f32; 3],
}

// Phase 2 scaffolding: Predefined card layouts for telemetry visualization
#[allow(dead_code)]
/// Predefined telemetry card layouts for left rail
pub const LEFT_RAIL_CARDS: [TelemetryCard; 3] = [
    TelemetryCard {
        title: "P-Core",
        y_offset: 80.0,
        height: 60.0,
        color: [0.2, 0.4, 0.8],
    },
    TelemetryCard {
        title: "E-Core",
        y_offset: 160.0,
        height: 60.0,
        color: [0.4, 0.6, 0.8],
    },
    TelemetryCard {
        title: "FPS",
        y_offset: 240.0,
        height: 60.0,
        color: [0.2, 0.8, 0.4],
    },
];

#[allow(dead_code)]
/// Predefined telemetry card layouts for right rail
pub const RIGHT_RAIL_CARDS: [TelemetryCard; 2] = [
    TelemetryCard {
        title: "Memory",
        y_offset: 80.0,
        height: 60.0,
        color: [0.8, 0.6, 0.2],
    },
    TelemetryCard {
        title: "Stability",
        y_offset: 160.0,
        height: 60.0,
        color: [0.6, 0.2, 0.8],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_creation() {
        let overlay = TelemetryOverlay::new();
        assert_eq!(overlay.current.fps, 0.0);
        assert_eq!(overlay.current.stability, 1.0);
        assert!(overlay.visible);
    }

    #[test]
    fn test_telemetry_update() {
        let mut overlay = TelemetryOverlay::new();
        
        // Simulate 60fps (16.67ms per frame)
        let frame_time = Duration::from_micros(16670);
        for _ in 0..60 {
            overlay.update(frame_time);
        }
        
        // Force snapshot update
        overlay.last_update = Instant::now() - Duration::from_secs(1);
        overlay.update(frame_time);
        
        // FPS should be close to 60
        assert!((overlay.current.fps - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_visibility_toggle() {
        let mut overlay = TelemetryOverlay::new();
        assert!(overlay.visible);
        
        overlay.toggle_visibility();
        assert!(!overlay.visible);
        
        overlay.toggle_visibility();
        assert!(overlay.visible);
    }

    #[test]
    fn test_format_strings() {
        let mut overlay = TelemetryOverlay::new();
        overlay.current.p_core_usage = 75.5;
        overlay.current.e_core_usage = 22.3;
        overlay.current.fps = 58.9;
        
        assert_eq!(overlay.format_p_core(), "P-Core: 75.5%");
        assert_eq!(overlay.format_e_core(), "E-Core: 22.3%");
        assert_eq!(overlay.format_fps(), "FPS: 58.9");
    }

    #[test]
    fn test_stability_colors() {
        let mut overlay = TelemetryOverlay::new();
        
        // Good stability (green)
        overlay.current.stability = 0.9;
        let color = overlay.stability_color();
        assert_eq!(color[1], 0.8); // Green channel
        
        // Medium stability (yellow)
        overlay.current.stability = 0.6;
        let color = overlay.stability_color();
        assert_eq!(color[0], 0.8); // Red channel
        assert_eq!(color[1], 0.8); // Green channel
        
        // Low stability (red)
        overlay.current.stability = 0.3;
        let color = overlay.stability_color();
        assert_eq!(color[0], 0.8); // Red channel
    }
}

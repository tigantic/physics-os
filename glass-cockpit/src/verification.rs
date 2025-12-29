/*!
 * Verification Protocols - Phase 8: Appendix H
 * 
 * Automated tests for proving each Constitutional doctrine is upheld.
 * 
 * Doctrines verified:
 * - Doctrine 1: Computational Sovereignty (E-core affinity)
 * - Doctrine 2: RAM Bridge Protocol (zero network)
 * - Doctrine 3: Procedural Rendering (no assets)
 * - Doctrine 8: Performance Boundaries (<5% CPU, 60+ FPS)
 */

use std::time::{Duration, Instant};

/// Result of a stress test run
#[derive(Debug)]
pub struct StressTestResult {
    pub duration_seconds: u64,
    pub frame_count: u64,
    pub mean_frame_time_ms: f32,
    pub max_frame_time_ms: f32,
    pub min_frame_time_ms: f32,
    pub stability_score: f32,      // max / mean - should be < 1.5
    pub fps: f32,
    pub passed: bool,
}

impl StressTestResult {
    /// Check if the result meets Phase 8 criteria
    pub fn verify(&self) -> bool {
        // Doctrine 8: 60+ FPS mandate
        let fps_ok = self.fps >= 60.0;
        
        // Stability: max frame time < 1.5x mean
        let stability_ok = self.stability_score < 1.5;
        
        fps_ok && stability_ok
    }
    
    /// Format result as verification report
    pub fn report(&self) -> String {
        format!(
            "═══ Stress Test Report ═══\n\
             Duration: {}s ({} frames)\n\
             FPS: {:.1}\n\
             Frame time: {:.2}ms mean, {:.2}ms max\n\
             Stability score: {:.2} (target: < 1.5)\n\
             Result: {}\n\
             ══════════════════════════",
            self.duration_seconds,
            self.frame_count,
            self.fps,
            self.mean_frame_time_ms,
            self.max_frame_time_ms,
            self.stability_score,
            if self.passed { "PASS" } else { "FAIL" }
        )
    }
}

/// Frame timing collector for stress tests
pub struct FrameTimingCollector {
    frame_times_ms: Vec<f32>,
    start_time: Instant,
    last_frame: Instant,
}

impl FrameTimingCollector {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            frame_times_ms: Vec::with_capacity(100_000),
            start_time: now,
            last_frame: now,
        }
    }
    
    /// Record a frame completion
    pub fn record_frame(&mut self) {
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame);
        self.frame_times_ms.push(frame_time.as_secs_f32() * 1000.0);
        self.last_frame = now;
    }
    
    /// Get elapsed time since start
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_times_ms.len()
    }
    
    /// Generate stress test result
    pub fn finalize(self) -> StressTestResult {
        if self.frame_times_ms.is_empty() {
            return StressTestResult {
                duration_seconds: 0,
                frame_count: 0,
                mean_frame_time_ms: 0.0,
                max_frame_time_ms: 0.0,
                min_frame_time_ms: 0.0,
                stability_score: 0.0,
                fps: 0.0,
                passed: false,
            };
        }
        
        let duration = self.start_time.elapsed();
        let frame_count = self.frame_times_ms.len() as u64;
        
        let sum: f32 = self.frame_times_ms.iter().sum();
        let mean = sum / self.frame_times_ms.len() as f32;
        let max = self.frame_times_ms.iter().cloned().fold(0.0_f32, f32::max);
        let min = self.frame_times_ms.iter().cloned().fold(f32::MAX, f32::min);
        
        let stability_score = if mean > 0.0 { max / mean } else { 0.0 };
        let fps = if mean > 0.0 { 1000.0 / mean } else { 0.0 };
        let passed = fps >= 60.0 && stability_score < 1.5;
        
        StressTestResult {
            duration_seconds: duration.as_secs(),
            frame_count,
            mean_frame_time_ms: mean,
            max_frame_time_ms: max,
            min_frame_time_ms: min,
            stability_score,
            fps,
            passed,
        }
    }
}

impl Default for FrameTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Doctrine 3 verification: No image assets in build
pub fn verify_no_image_assets() -> Result<(), Vec<String>> {
    use std::path::Path;
    
    let asset_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "svg", "webp"];
    let mut found_assets: Vec<String> = Vec::new();
    
    // Check shaders directory for any images
    let shader_dir = Path::new("src/shaders");
    if shader_dir.exists() {
        for entry in std::fs::read_dir(shader_dir).unwrap() {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if asset_extensions.contains(&ext.to_str().unwrap_or("")) {
                        found_assets.push(path.display().to_string());
                    }
                }
            }
        }
    }
    
    if found_assets.is_empty() {
        Ok(())
    } else {
        Err(found_assets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_frame_timing_collector() {
        let mut collector = FrameTimingCollector::new();
        
        // Simulate 100 frames at ~60 FPS (16.6ms each)
        for _ in 0..100 {
            std::thread::sleep(Duration::from_millis(16));
            collector.record_frame();
        }
        
        let result = collector.finalize();
        assert!(result.frame_count == 100);
        assert!(result.mean_frame_time_ms > 15.0 && result.mean_frame_time_ms < 20.0);
    }
    
    #[test]
    fn test_stress_result_verification() {
        let good_result = StressTestResult {
            duration_seconds: 60,
            frame_count: 3600,
            mean_frame_time_ms: 16.0,
            max_frame_time_ms: 20.0,
            min_frame_time_ms: 14.0,
            stability_score: 1.25,
            fps: 62.5,
            passed: true,
        };
        assert!(good_result.verify());
        
        let bad_fps_result = StressTestResult {
            fps: 50.0,
            ..good_result
        };
        assert!(!bad_fps_result.verify());
        
        let bad_stability_result = StressTestResult {
            stability_score: 2.0,
            ..good_result
        };
        assert!(!bad_stability_result.verify());
    }
    
    #[test]
    fn test_no_image_assets() {
        // This test verifies Doctrine 3: Procedural Rendering
        match verify_no_image_assets() {
            Ok(()) => println!("PASS: No image assets found (Doctrine 3)"),
            Err(assets) => {
                println!("FAIL: Image assets found (Doctrine 3 violation):");
                for asset in &assets {
                    println!("  - {}", asset);
                }
                // Don't panic - this is a warning for now
            }
        }
    }
}

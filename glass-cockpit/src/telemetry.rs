/*!
 * Frame Timing Telemetry
 * 
 * Doctrine 8: Performance Boundaries
 * 
 * Tracks frame timing to compute stability score and detect performance regressions.
 */

use std::collections::VecDeque;
use std::time::Duration;

const WINDOW_SIZE: usize = 120; // 2 seconds @ 60 FPS

pub struct FrameTiming {
    samples: VecDeque<f32>, // Frame times in milliseconds
}

impl FrameTiming {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(WINDOW_SIZE),
        }
    }
    
    pub fn record(&mut self, frame_time: Duration) {
        let ms = frame_time.as_secs_f32() * 1000.0;
        
        if self.samples.len() >= WINDOW_SIZE {
            self.samples.pop_front();
        }
        
        self.samples.push_back(ms);
    }
    
    pub fn mean_ms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = self.samples.iter().sum();
        sum / self.samples.len() as f32
    }
    
    pub fn max_ms(&self) -> f32 {
        self.samples
            .iter()
            .copied()
            .fold(0.0, f32::max)
    }
    
    pub fn min_ms(&self) -> f32 {
        self.samples
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
    }
    
    pub fn stability_score(&self) -> f32 {
        let mean = self.mean_ms();
        if mean == 0.0 {
            return 1.0;
        }
        
        let max = self.max_ms();
        max / mean
    }
    
    pub fn variance(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        
        let mean = self.mean_ms();
        let sum_squared_diff: f32 = self.samples
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        sum_squared_diff / (self.samples.len() - 1) as f32
    }
    
    pub fn fps(&self) -> f32 {
        let mean = self.mean_ms();
        if mean == 0.0 {
            return 0.0;
        }
        
        1000.0 / mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stable_timing() {
        let mut timing = FrameTiming::new();
        
        // Perfect 60 FPS (16.67ms per frame)
        for _ in 0..120 {
            timing.record(Duration::from_micros(16670));
        }
        
        assert!((timing.mean_ms() - 16.67).abs() < 0.1);
        assert!((timing.stability_score() - 1.0).abs() < 0.01);
        assert!((timing.fps() - 60.0).abs() < 1.0);
    }
    
    #[test]
    fn test_unstable_timing() {
        let mut timing = FrameTiming::new();
        
        // Unstable: alternate 10ms and 30ms
        for i in 0..120 {
            let ms = if i % 2 == 0 { 10000 } else { 30000 };
            timing.record(Duration::from_micros(ms));
        }
        
        assert!((timing.mean_ms() - 20.0).abs() < 0.1);
        assert!(timing.stability_score() > 1.4); // Max/mean = 30/20 = 1.5
    }
}

// Phase 1: Layout System
// Manages viewport regions: 80% center canvas, 10% left rail, 10% right rail
// Constitutional compliance: Doctrine 3 (explicit state), Doctrine 8 (minimal allocation)

use glam::{Vec2, Vec4};

/// Viewport region boundaries in pixel coordinates
#[derive(Debug, Clone, Copy)]
pub struct ViewportRegion {
    /// Left edge (x-coordinate)
    pub x: f32,
    /// Top edge (y-coordinate)
    pub y: f32,
    /// Region width
    pub width: f32,
    /// Region height
    pub height: f32,
}

impl ViewportRegion {
    /// Create a new viewport region
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    // Phase 1-2 scaffolding: Used for mouse interaction and coordinate transforms
    #[allow(dead_code)]
    /// Check if a point (in pixel coordinates) is within this region
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    #[allow(dead_code)]
    /// Convert screen coordinates to region-local coordinates
    pub fn to_local(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(x - self.x, y - self.y)
    }

    #[allow(dead_code)]
    /// Convert region-local coordinates to screen coordinates
    pub fn to_screen(&self, local_x: f32, local_y: f32) -> Vec2 {
        Vec2::new(self.x + local_x, self.y + local_y)
    }

    #[allow(dead_code)]
    /// Get the center point of this region
    pub fn center(&self) -> Vec2 {
        Vec2::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    #[allow(dead_code)]
    /// Get as Vec4 (x, y, width, height) for shader uniforms
    pub fn as_vec4(&self) -> Vec4 {
        Vec4::new(self.x, self.y, self.width, self.height)
    }
}

/// Layout manager for Glass Cockpit viewport
/// Manages three regions: left rail (10%), center canvas (80%), right rail (10%)
#[derive(Debug, Clone)]
pub struct ViewLayout {
    /// Full window dimensions
    pub window_width: u32,
    pub window_height: u32,

    /// Left rail region (10% of width)
    pub left_rail: ViewportRegion,

    /// Center canvas region (80% of width)
    pub canvas: ViewportRegion,

    /// Right rail region (10% of width)
    pub right_rail: ViewportRegion,

    /// Rail width as percentage (0.10 = 10%)
    pub rail_width_percent: f32,
}

impl ViewLayout {
    /// Create a new layout with the given window dimensions
    pub fn new(window_width: u32, window_height: u32) -> Self {
        let mut layout = Self {
            window_width,
            window_height,
            left_rail: ViewportRegion::new(0.0, 0.0, 0.0, 0.0),
            canvas: ViewportRegion::new(0.0, 0.0, 0.0, 0.0),
            right_rail: ViewportRegion::new(0.0, 0.0, 0.0, 0.0),
            rail_width_percent: 0.10, // 10% for each rail
        };
        layout.update_regions();
        layout
    }

    /// Update layout regions after window resize
    pub fn resize(&mut self, window_width: u32, window_height: u32) {
        self.window_width = window_width;
        self.window_height = window_height;
        self.update_regions();
    }

    /// Recalculate all viewport regions based on current window size
    fn update_regions(&mut self) {
        let width = self.window_width as f32;
        let height = self.window_height as f32;

        // Calculate rail widths
        let rail_width = width * self.rail_width_percent;
        let canvas_width = width - (rail_width * 2.0);

        // Left rail: 0 to rail_width
        self.left_rail = ViewportRegion::new(0.0, 0.0, rail_width, height);

        // Canvas: rail_width to (width - rail_width)
        self.canvas = ViewportRegion::new(rail_width, 0.0, canvas_width, height);

        // Right rail: (width - rail_width) to width
        self.right_rail = ViewportRegion::new(width - rail_width, 0.0, rail_width, height);
    }

    // Phase 1-2 scaffolding: Mouse interaction and region detection
    #[allow(dead_code)]
    /// Determine which region a point falls into
    pub fn region_at_point(&self, x: f32, y: f32) -> LayoutRegion {
        if self.left_rail.contains_point(x, y) {
            LayoutRegion::LeftRail
        } else if self.right_rail.contains_point(x, y) {
            LayoutRegion::RightRail
        } else if self.canvas.contains_point(x, y) {
            LayoutRegion::Canvas
        } else {
            LayoutRegion::None
        }
    }

    #[allow(dead_code)]
    /// Get the canvas aspect ratio (width / height)
    pub fn canvas_aspect_ratio(&self) -> f32 {
        self.canvas.width / self.canvas.height
    }

    #[allow(dead_code)]
    /// Set rail width percentage (e.g., 0.10 for 10%)
    pub fn set_rail_width_percent(&mut self, percent: f32) {
        self.rail_width_percent = percent.clamp(0.05, 0.25); // 5% to 25% range
        self.update_regions();
    }

    #[allow(dead_code)]
    /// Get screen size as Vec2 for shader uniforms
    pub fn screen_size(&self) -> Vec2 {
        Vec2::new(self.window_width as f32, self.window_height as f32)
    }
}

// Phase 1-2 scaffolding: Layout region detection for UI interactions
#[allow(dead_code)]
/// Enum representing which layout region a point belongs to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutRegion {
    /// Left side rail (telemetry, controls)
    LeftRail,
    /// Center canvas (main 3D view)
    Canvas,
    /// Right side rail (telemetry, diagnostics)
    RightRail,
    /// Outside all regions (should not happen in normal use)
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_creation() {
        let layout = ViewLayout::new(1920, 1080);
        
        // Check window dimensions
        assert_eq!(layout.window_width, 1920);
        assert_eq!(layout.window_height, 1080);
        
        // Check rail widths (10% of 1920 = 192)
        assert_eq!(layout.left_rail.width, 192.0);
        assert_eq!(layout.right_rail.width, 192.0);
        
        // Check canvas width (80% of 1920 = 1536)
        assert_eq!(layout.canvas.width, 1536.0);
        
        // Check positions
        assert_eq!(layout.left_rail.x, 0.0);
        assert_eq!(layout.canvas.x, 192.0);
        assert_eq!(layout.right_rail.x, 1728.0);
    }

    #[test]
    fn test_region_detection() {
        let layout = ViewLayout::new(1000, 600);
        
        // Left rail: 0-100
        assert_eq!(layout.region_at_point(50.0, 300.0), LayoutRegion::LeftRail);
        
        // Canvas: 100-900
        assert_eq!(layout.region_at_point(500.0, 300.0), LayoutRegion::Canvas);
        
        // Right rail: 900-1000
        assert_eq!(layout.region_at_point(950.0, 300.0), LayoutRegion::RightRail);
    }

    #[test]
    fn test_resize() {
        let mut layout = ViewLayout::new(800, 600);
        assert_eq!(layout.canvas.width, 640.0); // 80% of 800
        
        layout.resize(1600, 900);
        assert_eq!(layout.canvas.width, 1280.0); // 80% of 1600
    }

    #[test]
    fn test_coordinate_transforms() {
        let layout = ViewLayout::new(1000, 600);
        let region = &layout.canvas;
        
        // Screen point (500, 300) in canvas -> local (400, 300) since canvas starts at x=100
        let local = region.to_local(500.0, 300.0);
        assert_eq!(local.x, 400.0);
        assert_eq!(local.y, 300.0);
        
        // Convert back
        let screen = region.to_screen(400.0, 300.0);
        assert_eq!(screen.x, 500.0);
        assert_eq!(screen.y, 300.0);
    }

    #[test]
    fn test_aspect_ratio() {
        let layout = ViewLayout::new(1920, 1080);
        let aspect = layout.canvas_aspect_ratio();
        
        // Canvas is 1536x1080, aspect = 1536/1080 ≈ 1.422
        assert!((aspect - 1.422).abs() < 0.01);
    }

    #[test]
    fn test_rail_width_adjustment() {
        let mut layout = ViewLayout::new(1000, 600);
        
        // Default 10%
        assert_eq!(layout.left_rail.width, 100.0);
        
        // Set to 15%
        layout.set_rail_width_percent(0.15);
        assert_eq!(layout.left_rail.width, 150.0);
        assert_eq!(layout.canvas.width, 700.0); // 70% remains
        
        // Clamp to max 25%
        layout.set_rail_width_percent(0.50); // Try to set 50%
        assert_eq!(layout.left_rail.width, 250.0); // Clamped to 25%
        
        // Clamp to min 5%
        layout.set_rail_width_percent(0.01); // Try to set 1%
        assert_eq!(layout.left_rail.width, 50.0); // Clamped to 5%
    }
}

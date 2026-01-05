//! LOD (Level of Detail) and Culling Infrastructure
//!
//! Phase 6+ Performance Foundation - Constitutional Optimization
//!
//! This module provides:
//! - Frustum culling for discarding off-screen geometry
//! - Distance-based LOD levels
//! - Instance budget management to maintain 60 FPS mandate
//!
//! All subsequent phases (voxels, particles, streamlines) inherit this infrastructure.
#![allow(dead_code)] // Stress level and advanced culling ready for integration

use glam::{Mat4, Vec3, Vec4};

/// Frustum planes for GPU-friendly culling
/// Phase 7: Used for voxel visibility culling
#[allow(dead_code)] // Phase 7: Voxel frustum culling
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// Left, Right, Bottom, Top, Near, Far planes (ax + by + cz + d = 0)
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from view-projection matrix
    /// Using Gribb/Hartmann method for efficient extraction
    pub fn from_view_projection(vp: Mat4) -> Self {
        let row0 = Vec4::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x, vp.w_axis.x);
        let row1 = Vec4::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y, vp.w_axis.y);
        let row2 = Vec4::new(vp.x_axis.z, vp.y_axis.z, vp.z_axis.z, vp.w_axis.z);
        let row3 = Vec4::new(vp.x_axis.w, vp.y_axis.w, vp.z_axis.w, vp.w_axis.w);

        let planes = [
            (row3 + row0).normalize(), // Left
            (row3 - row0).normalize(), // Right
            (row3 + row1).normalize(), // Bottom
            (row3 - row1).normalize(), // Top
            (row3 + row2).normalize(), // Near
            (row3 - row2).normalize(), // Far
        ];

        Self { planes }
    }

    /// Test if a sphere is inside or intersects the frustum
    /// Returns true if visible (should be rendered)
    /// Phase 7: Used for voxel sphere-based culling
    #[inline]
    #[allow(dead_code)] // Phase 7: Voxel culling
    pub fn is_sphere_visible(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let distance = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if distance < -radius {
                return false; // Entirely outside this plane
            }
        }
        true
    }

    /// Test if an AABB is inside or intersects the frustum
    /// Returns true if visible (should be rendered)
    /// Phase 7: Used for voxel AABB culling
    #[inline]
    #[allow(dead_code)] // Phase 7: Voxel culling
    pub fn is_aabb_visible(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Find the corner most aligned with plane normal (p-vertex)
            let p = Vec3::new(
                if plane.x >= 0.0 { max.x } else { min.x },
                if plane.y >= 0.0 { max.y } else { min.y },
                if plane.z >= 0.0 { max.z } else { min.z },
            );

            let distance = plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w;
            if distance < 0.0 {
                return false; // Entirely outside this plane
            }
        }
        true
    }
}

/// LOD level for distance-based quality selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LodLevel {
    /// Full detail - closest to camera
    High = 0,
    /// Medium detail - moderate distance
    Medium = 1,
    /// Low detail - far from camera
    Low = 2,
    /// Culled - beyond visibility threshold
    Culled = 3,
}

impl LodLevel {
    /// Get particle density multiplier for this LOD level
    /// Phase 7: Used for adaptive particle density
    #[allow(dead_code)] // Phase 7: Adaptive particle density
    pub fn particle_density(&self) -> f32 {
        match self {
            LodLevel::High => 1.0,
            LodLevel::Medium => 0.5,
            LodLevel::Low => 0.25,
            LodLevel::Culled => 0.0,
        }
    }

    /// Get streamline sample rate for this LOD level
    /// Phase 7: Used for adaptive streamline sampling
    #[allow(dead_code)] // Phase 7: Adaptive streamlines
    pub fn streamline_samples(&self) -> u32 {
        match self {
            LodLevel::High => 100,
            LodLevel::Medium => 50,
            LodLevel::Low => 25,
            LodLevel::Culled => 0,
        }
    }

    /// Get voxel subdivision level for this LOD level
    /// Phase 7: Core voxel LOD system
    #[allow(dead_code)] // Phase 7: Voxel subdivision
    pub fn voxel_subdivision(&self) -> u32 {
        match self {
            LodLevel::High => 4,
            LodLevel::Medium => 2,
            LodLevel::Low => 1,
            LodLevel::Culled => 0,
        }
    }
}

/// LOD configuration with distance thresholds
#[derive(Debug, Clone, Copy)]
pub struct LodConfig {
    /// Distance threshold for High → Medium transition
    pub high_distance: f32,
    /// Distance threshold for Medium → Low transition
    pub medium_distance: f32,
    /// Distance threshold for Low → Culled transition
    pub cull_distance: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            high_distance: 2.0,    // Within 2 units: full detail
            medium_distance: 5.0,  // 2-5 units: medium detail
            cull_distance: 15.0,   // Beyond 15 units: culled
        }
    }
}

impl LodConfig {
    /// Determine LOD level based on distance from camera
    #[inline]
    pub fn get_level(&self, distance: f32) -> LodLevel {
        if distance < self.high_distance {
            LodLevel::High
        } else if distance < self.medium_distance {
            LodLevel::Medium
        } else if distance < self.cull_distance {
            LodLevel::Low
        } else {
            LodLevel::Culled
        }
    }

    /// Create config for globe-scale rendering (Phase 4+)
    pub fn globe_scale() -> Self {
        Self {
            high_distance: 3.0,
            medium_distance: 8.0,
            cull_distance: 20.0,
        }
    }

    /// Create config for voxel rendering (Phase 7)
    #[allow(dead_code)] // Phase 7: Voxel LOD config
    pub fn voxel_scale() -> Self {
        Self {
            high_distance: 1.0,
            medium_distance: 3.0,
            cull_distance: 10.0,
        }
    }
}

/// Instance budget manager to maintain 60 FPS mandate
#[derive(Debug)]
pub struct InstanceBudget {
    /// Maximum particles allowed
    pub max_particles: u32,
    /// Maximum streamline vertices allowed
    pub max_streamline_vertices: u32,
    /// Maximum voxel instances allowed
    pub max_voxels: u32,
    /// Maximum heatmap cells (Phase 6)
    pub max_heatmap_cells: u32,
    /// Current particle count
    current_particles: u32,
    /// Current streamline vertex count
    current_streamline_vertices: u32,
    /// Current voxel count
    current_voxels: u32,
    /// Current heatmap cells
    current_heatmap_cells: u32,
}

impl Default for InstanceBudget {
    fn default() -> Self {
        Self {
            // Particle budget for RTX 5070 - 100k is easily achievable
            max_particles: 100_000,
            max_streamline_vertices: 50_000,
            max_voxels: 100_000,
            max_heatmap_cells: 65_536, // 256×256 grid
            current_particles: 0,
            current_streamline_vertices: 0,
            current_voxels: 0,
            current_heatmap_cells: 0,
        }
    }
}

impl InstanceBudget {
    /// Reset all current counts (call at frame start)
    pub fn reset(&mut self) {
        self.current_particles = 0;
        self.current_streamline_vertices = 0;
        self.current_voxels = 0;
        self.current_heatmap_cells = 0;
    }

    /// Request particle allocation, returns actual allowed count
    /// Phase 7: Particle budget enforcement
    #[allow(dead_code)] // Phase 7: Particle budget
    pub fn allocate_particles(&mut self, requested: u32) -> u32 {
        let available = self.max_particles.saturating_sub(self.current_particles);
        let granted = requested.min(available);
        self.current_particles += granted;
        granted
    }

    /// Request streamline vertex allocation
    /// Phase 7: Streamline budget enforcement
    #[allow(dead_code)] // Phase 7: Streamline budget
    pub fn allocate_streamline_vertices(&mut self, requested: u32) -> u32 {
        let available = self.max_streamline_vertices.saturating_sub(self.current_streamline_vertices);
        let granted = requested.min(available);
        self.current_streamline_vertices += granted;
        granted
    }

    /// Request voxel allocation
    /// Phase 7: Voxel budget enforcement
    #[allow(dead_code)] // Phase 7: Voxel budget
    pub fn allocate_voxels(&mut self, requested: u32) -> u32 {
        let available = self.max_voxels.saturating_sub(self.current_voxels);
        let granted = requested.min(available);
        self.current_voxels += granted;
        granted
    }

    /// Request heatmap cell allocation
    pub fn allocate_heatmap_cells(&mut self, requested: u32) -> u32 {
        let available = self.max_heatmap_cells.saturating_sub(self.current_heatmap_cells);
        let granted = requested.min(available);
        self.current_heatmap_cells += granted;
        granted
    }

    /// Get utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        let particle_util = self.current_particles as f32 / self.max_particles as f32;
        let streamline_util = self.current_streamline_vertices as f32 / self.max_streamline_vertices as f32;
        let voxel_util = self.current_voxels as f32 / self.max_voxels as f32;
        let heatmap_util = self.current_heatmap_cells as f32 / self.max_heatmap_cells as f32;

        // Return max utilization across all categories
        particle_util.max(streamline_util).max(voxel_util).max(heatmap_util)
    }

    /// Check if we're approaching budget limit (>80%)
    /// Phase 8: Performance warning telemetry
    #[allow(dead_code)] // Phase 8: Performance telemetry
    pub fn is_stressed(&self) -> bool {
        self.utilization() > 0.8
    }

    /// Adjust budgets based on current FPS
    /// Call this to dynamically scale quality for 165Hz Sovereign mandate
    /// Note: Should be called periodically (every few seconds), not every frame
    pub fn adjust_for_fps(&mut self, current_fps: f32, target_fps: f32) {
        let ratio = current_fps / target_fps;

        // Minimum floors to prevent budget starvation
        const MIN_PARTICLES: u32 = 1000;
        const MIN_STREAMLINE_VERTICES: u32 = 5000;
        const MIN_VOXELS: u32 = 10000;
        const MIN_HEATMAP_CELLS: u32 = 4096;

        if ratio < 0.9 {
            // FPS too low, reduce budgets by 10% (with minimums)
            self.max_particles = ((self.max_particles as f32 * 0.9) as u32).max(MIN_PARTICLES);
            self.max_streamline_vertices = ((self.max_streamline_vertices as f32 * 0.9) as u32).max(MIN_STREAMLINE_VERTICES);
            self.max_voxels = ((self.max_voxels as f32 * 0.9) as u32).max(MIN_VOXELS);
            self.max_heatmap_cells = ((self.max_heatmap_cells as f32 * 0.9) as u32).max(MIN_HEATMAP_CELLS);
        } else if ratio > 1.2 && self.max_particles < 100_000 {
            // FPS healthy, can increase budgets by 5%
            self.max_particles = (self.max_particles as f32 * 1.05) as u32;
            self.max_streamline_vertices = (self.max_streamline_vertices as f32 * 1.05) as u32;
            self.max_voxels = (self.max_voxels as f32 * 1.05) as u32;
            self.max_heatmap_cells = (self.max_heatmap_cells as f32 * 1.05) as u32;
        }
    }
}

/// Visibility result from culling pass
/// Phase 7: Return type for voxel visibility queries
#[allow(dead_code)] // Phase 7: Voxel visibility
#[derive(Debug, Clone, Copy)]
pub struct VisibilityResult {
    /// Is the object visible at all?
    pub visible: bool,
    /// What LOD level should be used?
    pub lod: LodLevel,
    /// Distance from camera (for sorting)
    pub distance: f32,
}

/// Combined LOD and culling system
pub struct LodCuller {
    /// Current frustum for culling
    pub frustum: Frustum,
    /// LOD configuration
    pub config: LodConfig,
    /// Camera position for distance calculations
    pub camera_pos: Vec3,
    /// Instance budget manager
    pub budget: InstanceBudget,
}

impl LodCuller {
    /// Create a new LOD culler with default settings
    pub fn new() -> Self {
        Self {
            frustum: Frustum::from_view_projection(Mat4::IDENTITY),
            config: LodConfig::default(),
            camera_pos: Vec3::ZERO,
            budget: InstanceBudget::default(),
        }
    }

    /// Update frustum and camera position from view-projection matrix
    pub fn update(&mut self, view_proj: Mat4, camera_pos: Vec3) {
        self.frustum = Frustum::from_view_projection(view_proj);
        self.camera_pos = camera_pos;
        self.budget.reset();
    }

    /// Test visibility and LOD for a sphere
    /// Phase 7: Sphere-based voxel culling
    #[allow(dead_code)] // Phase 7: Voxel culling
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> VisibilityResult {
        let visible = self.frustum.is_sphere_visible(center, radius);
        let distance = (center - self.camera_pos).length();
        let lod = if visible {
            self.config.get_level(distance)
        } else {
            LodLevel::Culled
        };

        VisibilityResult {
            visible: visible && lod != LodLevel::Culled,
            lod,
            distance,
        }
    }

    /// Test visibility and LOD for an AABB
    /// Phase 7: AABB-based chunk culling
    #[allow(dead_code)] // Phase 7: Chunk culling
    pub fn test_aabb(&self, min: Vec3, max: Vec3) -> VisibilityResult {
        let visible = self.frustum.is_aabb_visible(min, max);
        let center = (min + max) * 0.5;
        let distance = (center - self.camera_pos).length();
        let lod = if visible {
            self.config.get_level(distance)
        } else {
            LodLevel::Culled
        };

        VisibilityResult {
            visible: visible && lod != LodLevel::Culled,
            lod,
            distance,
        }
    }

    /// Get current budget stress level for telemetry
    pub fn stress_level(&self) -> f32 {
        self.budget.utilization()
    }
}

impl Default for LodCuller {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_levels() {
        let config = LodConfig::default();
        assert_eq!(config.get_level(1.0), LodLevel::High);
        assert_eq!(config.get_level(3.0), LodLevel::Medium);
        assert_eq!(config.get_level(10.0), LodLevel::Low);
        assert_eq!(config.get_level(20.0), LodLevel::Culled);
    }

    #[test]
    fn test_instance_budget() {
        let mut budget = InstanceBudget {
            max_particles: 10_000, // Set explicit max for test
            ..InstanceBudget::default()
        };
        
        // Request within budget
        let granted = budget.allocate_particles(5000);
        assert_eq!(granted, 5000);
        
        // Request exceeding remaining budget
        let granted = budget.allocate_particles(8000);
        assert_eq!(granted, 5000); // Only 5000 remaining
        
        // Reset and verify
        budget.reset();
        let granted = budget.allocate_particles(1000);
        assert_eq!(granted, 1000);
    }

    #[test]
    fn test_frustum_sphere() {
        let vp = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let frustum = Frustum::from_view_projection(vp);
        
        // Sphere in front of camera should be visible
        assert!(frustum.is_sphere_visible(Vec3::new(0.0, 0.0, -5.0), 1.0));
        
        // Sphere behind camera should be culled
        assert!(!frustum.is_sphere_visible(Vec3::new(0.0, 0.0, 5.0), 1.0));
    }
}

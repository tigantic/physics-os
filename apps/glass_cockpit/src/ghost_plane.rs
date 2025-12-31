//! Phase 3C-5: Ghost Plane Renderer
//!
//! Renders a semi-transparent aircraft avatar showing the predicted
//! position along the optimized trajectory (the "Digital Twin").
//!
//! The ghost plane shows where the vehicle should be at time T+5 seconds,
//! giving the pilot/AI a visual target to fly toward.

use glam::{Vec3, Quat, Mat4};
use hyper_bridge::trajectory::{TrajectoryData, Waypoint};

/// Ghost plane configuration
#[derive(Debug, Clone, Copy)]
pub struct GhostPlaneConfig {
    /// Time offset ahead of current position (seconds)
    pub look_ahead_seconds: f32,
    /// Ghost plane opacity (0.0 - 1.0)
    pub opacity: f32,
    /// Ghost plane scale relative to real aircraft
    pub scale: f32,
    /// Color tint (RGB)
    pub color: [f32; 3],
    /// Enable glow effect
    pub glow: bool,
    /// Glow intensity
    pub glow_intensity: f32,
}

impl Default for GhostPlaneConfig {
    fn default() -> Self {
        Self {
            look_ahead_seconds: 5.0,
            opacity: 0.5,
            scale: 1.0,
            color: [0.0, 1.0, 0.8], // Cyan/teal
            glow: true,
            glow_intensity: 0.3,
        }
    }
}

/// Ghost plane state for rendering
#[derive(Debug, Clone)]
pub struct GhostPlane {
    /// World position of ghost plane
    pub position: Vec3,
    /// Rotation quaternion (orientation)
    pub rotation: Quat,
    /// Model matrix for rendering
    pub model_matrix: Mat4,
    /// Velocity vector (for trail effects)
    pub velocity: Vec3,
    /// Is the ghost plane valid (on trajectory)
    pub valid: bool,
    /// Distance from current position to ghost
    pub distance_to_current: f32,
    /// Time until reaching ghost position
    pub eta_seconds: f32,
}

impl Default for GhostPlane {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            model_matrix: Mat4::IDENTITY,
            velocity: Vec3::ZERO,
            valid: false,
            distance_to_current: 0.0,
            eta_seconds: 0.0,
        }
    }
}

impl GhostPlane {
    /// Update ghost plane from trajectory data
    ///
    /// # Arguments
    /// * `trajectory` - Current optimized trajectory
    /// * `current_time` - Current simulation time (seconds from trajectory start)
    /// * `current_position` - Current aircraft position
    /// * `config` - Ghost plane configuration
    pub fn update(
        &mut self,
        trajectory: &TrajectoryData,
        current_time: f32,
        current_position: Vec3,
        config: &GhostPlaneConfig,
    ) {
        // Get waypoint at look-ahead time
        let look_ahead_time = current_time + config.look_ahead_seconds;
        
        if let Some(wp) = trajectory.waypoint_at_time(look_ahead_time) {
            self.position = wp.to_vec3();
            self.valid = true;
            self.eta_seconds = config.look_ahead_seconds;
            self.distance_to_current = (self.position - current_position).length();
            
            // Compute orientation from trajectory tangent
            if let Some(wp_next) = trajectory.waypoint_at_time(look_ahead_time + 0.1) {
                let tangent = (wp_next.to_vec3() - self.position).normalize_or_zero();
                self.velocity = tangent;
                
                // Build rotation from tangent (look_at style)
                if tangent.length_squared() > 0.01 {
                    let world_up = Vec3::Y;
                    let right = tangent.cross(world_up).normalize_or_zero();
                    let up = right.cross(tangent).normalize_or_zero();
                    
                    // Rotation matrix → quaternion
                    let rot_mat = Mat4::from_cols(
                        right.extend(0.0),
                        up.extend(0.0),
                        (-tangent).extend(0.0), // -Z is forward
                        Vec3::ZERO.extend(1.0),
                    );
                    self.rotation = Quat::from_mat4(&rot_mat);
                }
            }
            
            // Build model matrix: translate, rotate, scale
            self.model_matrix = Mat4::from_scale_rotation_translation(
                Vec3::splat(config.scale),
                self.rotation,
                self.position,
            );
        } else {
            self.valid = false;
        }
    }
    
    /// Get the ghost plane color with opacity applied
    pub fn get_color(&self, config: &GhostPlaneConfig) -> [f32; 4] {
        [
            config.color[0],
            config.color[1],
            config.color[2],
            config.opacity,
        ]
    }
}

/// Simple aircraft mesh vertices (low-poly for ghost effect)
///
/// Returns (positions, normals, indices) for a basic delta-wing shape
pub fn create_ghost_aircraft_mesh() -> (Vec<f32>, Vec<f32>, Vec<u32>) {
    // Very simple delta wing shape pointing in -Z direction
    // Scale: ~1 unit wingspan
    
    let positions: Vec<f32> = vec![
        // Fuselage
         0.0,  0.0, -0.5,   // Nose (0)
         0.0,  0.05, 0.0,   // Top mid (1)
         0.0, -0.05, 0.0,   // Bottom mid (2)
         0.0,  0.0,  0.5,   // Tail (3)
        
        // Wings
        -0.5,  0.0,  0.3,   // Left wing tip (4)
         0.5,  0.0,  0.3,   // Right wing tip (5)
        
        // Vertical stabilizer
         0.0,  0.2,  0.4,   // Fin top (6)
    ];
    
    let normals: Vec<f32> = vec![
        // Simplified normals (point outward from surface)
         0.0,  0.0, -1.0,   // Nose
         0.0,  1.0,  0.0,   // Top
         0.0, -1.0,  0.0,   // Bottom
         0.0,  0.0,  1.0,   // Tail
        -0.5,  0.5,  0.0,   // Left wing
         0.5,  0.5,  0.0,   // Right wing
         0.0,  1.0,  0.0,   // Fin
    ];
    
    let indices: Vec<u32> = vec![
        // Top surface
        0, 1, 4,    // Left top front
        0, 5, 1,    // Right top front
        1, 4, 3,    // Left top rear
        1, 3, 5,    // Right top rear
        
        // Bottom surface
        0, 4, 2,    // Left bottom front
        0, 2, 5,    // Right bottom front
        2, 3, 4,    // Left bottom rear
        2, 5, 3,    // Right bottom rear
        
        // Vertical fin
        3, 1, 6,    // Fin left
        3, 6, 1,    // Fin right (backface)
    ];
    
    (positions, normals, indices)
}

/// Trail effect data for ghost plane
#[derive(Debug, Clone)]
pub struct GhostTrail {
    /// Trail points (most recent first)
    pub points: Vec<Vec3>,
    /// Maximum trail length
    pub max_points: usize,
    /// Trail fade (0 = invisible, 1 = full)
    pub fade: f32,
}

impl GhostTrail {
    pub fn new(max_points: usize) -> Self {
        Self {
            points: Vec::with_capacity(max_points),
            max_points,
            fade: 1.0,
        }
    }
    
    /// Add a point to the trail
    pub fn push(&mut self, point: Vec3) {
        self.points.insert(0, point);
        if self.points.len() > self.max_points {
            self.points.pop();
        }
    }
    
    /// Get trail vertices with alpha gradient
    pub fn get_line_vertices(&self) -> (Vec<f32>, Vec<f32>) {
        let mut positions = Vec::with_capacity(self.points.len() * 3);
        let mut colors = Vec::with_capacity(self.points.len() * 4);
        
        for (i, point) in self.points.iter().enumerate() {
            let alpha = self.fade * (1.0 - i as f32 / self.max_points as f32);
            
            positions.extend_from_slice(&[point.x, point.y, point.z]);
            colors.extend_from_slice(&[0.0, 1.0, 0.8, alpha]); // Cyan fade
        }
        
        (positions, colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ghost_plane_update() {
        let waypoints = vec![
            Waypoint::new(0.0, 0.0, 0.0, 0.0),
            Waypoint::new(1.0, 1.0, 100.0, 10.0),
            Waypoint::new(2.0, 2.0, 200.0, 20.0),
        ];
        
        let trajectory = TrajectoryData::new(waypoints);
        let config = GhostPlaneConfig::default();
        
        let mut ghost = GhostPlane::default();
        ghost.update(&trajectory, 0.0, Vec3::ZERO, &config);
        
        assert!(ghost.valid);
        assert!(ghost.eta_seconds > 0.0);
    }
    
    #[test]
    fn test_aircraft_mesh() {
        let (positions, normals, indices) = create_ghost_aircraft_mesh();
        
        assert!(!positions.is_empty());
        assert!(!normals.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(positions.len(), normals.len()); // Same vertex count
    }
}

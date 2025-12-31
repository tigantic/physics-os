/*!
 * Camera and View Controls (Physics Enabled)
 * 
 * "Heavy" Orbit camera with mass, inertia, and exponential decay friction.
 * Constitutional Compliance: Doctrine 6 (Physics-based interaction)
 * 
 * Phase 8: RTE (Relative-To-Eye) precision pipeline
 * Phase 9: Inertial physics for "God's Eye" satellite feel
 */
#![allow(dead_code)]  // Physics methods and RTE pipeline ready for Phase 9

use glam::{Mat4, Vec3, DVec3};

/// Split-double precision for GPU upload (prevents planetary jitter)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplitPosition {
    pub high: [f32; 3],
    pub _pad0: f32,
    pub low: [f32; 3],
    pub _pad1: f32,
}

impl SplitPosition {
    /// Split f64 position into high/low f32 pairs for RTE pipeline
    pub fn from_f64(pos: DVec3) -> Self {
        // High = truncated to f32 precision
        let high = pos.as_vec3();
        // Low = remainder (what was lost in truncation)
        let low = (pos - high.as_dvec3()).as_vec3();
        Self {
            high: high.to_array(),
            _pad0: 0.0,
            low: low.to_array(),
            _pad1: 0.0,
        }
    }
}

pub struct Camera {
    // Spatial State
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    
    // Orbit Parameters (Spherical)
    pub distance: f32,
    pub azimuth: f32,   // Horizontal (Radians) - replaces yaw
    pub polar: f32,     // Vertical (Radians) - replaces pitch
    
    // Physics State (Momentum)
    vel_azimuth: f32,
    vel_polar: f32,
    vel_distance: f32,
    vel_pan: Vec3,
    
    // Tuning (The "Feel")
    pub mass: f32,          // Higher = Heavier feel (harder to start/stop)
    pub friction: f32,      // Lower = Glides longer
    pub sensitivity: f32,   // Mouse input multiplier
    
    // Projection
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    
    // High-precision position for RTE pipeline (f64)
    pub position_f64: DVec3,
    pub target_f64: DVec3,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        let mut cam = Self {
            position: Vec3::ZERO,
            target: Vec3::ZERO,
            up: Vec3::Y,
            
            distance: 15.0,
            azimuth: std::f32::consts::PI * 0.25,
            polar: std::f32::consts::PI * 0.3,
            
            // Initial Velocity (Stationary)
            vel_azimuth: 0.0,
            vel_polar: 0.0,
            vel_distance: 0.0,
            vel_pan: Vec3::ZERO,
            
            // [TUNING] HEAVY SATELLITE CONFIG
            mass: 5.0,
            friction: 2.0,      // Low friction = space glide
            sensitivity: 0.08,  // Sensitive input to overcome mass
            
            fov: 35.0_f32.to_radians(), // Telephoto (cinematic look)
            aspect,
            near: 0.1,
            far: 10000.0,
            
            // RTE precision
            position_f64: DVec3::ZERO,
            target_f64: DVec3::ZERO,
        };
        
        cam.update_position();
        cam
    }
    
    /// Physics Step - Call this EVERY FRAME with delta_time
    pub fn update(&mut self, dt: f32) {
        // 1. Apply Momentum
        self.azimuth += self.vel_azimuth * dt;
        self.polar += self.vel_polar * dt;
        self.distance += self.vel_distance * dt;
        
        // 2. Apply Friction (Exponential Decay for FPS independence)
        let drag = (-self.friction * dt).exp();
        self.vel_azimuth *= drag;
        self.vel_polar *= drag;
        self.vel_distance *= drag;
        self.vel_pan *= drag;
        
        // 3. Stop micro-drifting
        if self.vel_azimuth.abs() < 0.001 { self.vel_azimuth = 0.0; }
        if self.vel_polar.abs() < 0.001 { self.vel_polar = 0.0; }
        if self.vel_distance.abs() < 0.01 { self.vel_distance = 0.0; }
        if self.vel_pan.length() < 0.0001 { self.vel_pan = Vec3::ZERO; }
        
        // 4. Apply pan velocity
        if self.vel_pan.length() > 0.0001 {
            self.target += self.vel_pan * dt;
            self.target_f64 += self.vel_pan.as_dvec3() * dt as f64;
        }

        // 5. Constraints
        self.polar = self.polar.clamp(0.05, std::f32::consts::PI - 0.05);
        self.distance = self.distance.clamp(1.2, 500.0);
        
        self.update_position();
    }
    
    /// Legacy alias for update() - maintains API compatibility
    pub fn update_physics(&mut self, dt: f32) {
        self.update(dt);
    }
    
    /// Input: Mouse Drag (Adds force, doesn't set position)
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        // F = ma -> a = F/m
        // We apply instantaneous acceleration based on mouse flick
        self.vel_azimuth -= (delta_x * self.sensitivity) / self.mass;
        self.vel_polar -= (delta_y * self.sensitivity) / self.mass;
    }
    
    /// Input: Scroll Wheel
    pub fn zoom(&mut self, delta: f32) {
        // Zoom speed scales with distance (faster when far away)
        let zoom_force = delta * (self.distance * 0.5);
        self.vel_distance -= zoom_force / self.mass;
    }
    
    /// Pan camera target (Screen space) - with momentum
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let right = (self.target - self.position).cross(self.up).normalize();
        let up = right.cross(self.target - self.position).normalize();
        
        let pan_speed = self.distance * 0.001 * self.sensitivity;
        let pan_force = (right * delta_x + up * delta_y) * pan_speed / self.mass;
        self.vel_pan += pan_force;
    }
    
    /// Pan camera instantly (no momentum)
    pub fn pan_instant(&mut self, delta_x: f32, delta_y: f32) {
        let right = (self.target - self.position).cross(self.up).normalize();
        let up = right.cross(self.target - self.position).normalize();
        
        let pan_speed = self.distance * 0.001;
        self.target += right * delta_x * pan_speed;
        self.target += up * delta_y * pan_speed;
        self.target_f64 += (right * delta_x * pan_speed).as_dvec3();
        self.target_f64 += (up * delta_y * pan_speed).as_dvec3();
        self.update_position();
    }
    
    /// Kill all momentum (emergency stop)
    pub fn stop(&mut self) {
        self.vel_azimuth = 0.0;
        self.vel_polar = 0.0;
        self.vel_distance = 0.0;
        self.vel_pan = Vec3::ZERO;
    }
    
    /// Called when mouse drag starts (no-op, physics handles momentum)
    pub fn start_drag(&mut self) {
        // Optional: could kill velocity here for instant response
        // self.stop();
    }
    
    /// Called when mouse drag ends (no-op, inertia continues)
    pub fn stop_drag(&mut self) {
        // Momentum continues naturally via friction decay
    }

    pub fn update_position(&mut self) {
        let x = self.distance * self.azimuth.sin() * self.polar.sin();
        let y = self.distance * self.polar.cos();
        let z = self.distance * self.azimuth.cos() * self.polar.sin();
        
        let offset = Vec3::new(x, y, z);
        self.position = self.target + offset;
        
        // Maintain f64 precision for RTE
        self.position_f64 = self.target_f64 + offset.as_dvec3();
    }
    
    /// Get split-precision camera position for GPU (RTE pipeline)
    pub fn split_position(&self) -> SplitPosition {
        SplitPosition::from_f64(self.position_f64)
    }
    
    // Standard Matrices
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }
    
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }
    
    pub fn view_proj_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
    
    /// Get inverse view-projection matrix for unprojection
    pub fn inv_view_proj_matrix(&self) -> Mat4 {
        self.view_proj_matrix().inverse()
    }
    
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
    
    // Legacy getters for compatibility
    pub fn yaw(&self) -> f32 { self.azimuth }
    pub fn pitch(&self) -> f32 { self.polar }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_camera_initialization() {
        let camera = Camera::new(16.0 / 9.0);
        assert!(camera.position.length() > 0.0);
        assert_eq!(camera.target, Vec3::ZERO);
    }
    
    #[test]
    fn test_physics_decay() {
        let mut camera = Camera::new(1.0);
        camera.orbit(1.0, 0.0);
        let initial_vel = camera.vel_azimuth;
        
        // After physics step, velocity should decay
        camera.update(0.016); // ~60fps
        assert!(camera.vel_azimuth.abs() < initial_vel.abs());
    }
    
    #[test]
    fn test_zoom_momentum() {
        let mut camera = Camera::new(1.0);
        let initial_distance = camera.distance;
        
        camera.zoom(1.0);
        camera.update(0.1);
        
        // Distance should have changed due to velocity
        assert_ne!(camera.distance, initial_distance);
    }
}

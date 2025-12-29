/*!
 * Camera and View Controls
 * 
 * Orbit camera for 3D grid navigation with mouse/keyboard controls.
 */

use glam::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    
    // Orbit parameters
    pub distance: f32,
    pub yaw: f32,   // Rotation around Y axis (radians)
    pub pitch: f32, // Rotation around X axis (radians)
    
    // Projection
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        let mut camera = Self {
            position: Vec3::new(10.0, 8.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            
            distance: 15.0,
            yaw: std::f32::consts::PI * 0.25,   // 45 degrees
            pitch: std::f32::consts::PI * 0.3,  // ~54 degrees
            
            fov: 60.0_f32.to_radians(),
            aspect,
            near: 0.1,
            far: 1000.0,
        };
        
        camera.update_position();
        camera
    }
    
    /// Update camera position from orbit parameters
    pub fn update_position(&mut self) {
        let x = self.distance * self.yaw.cos() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.sin() * self.pitch.cos();
        
        self.position = self.target + Vec3::new(x, y, z);
    }
    
    /// Get view matrix
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }
    
    /// Get projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }
    
    /// Get combined view-projection matrix
    pub fn view_proj_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
    
    /// Get inverse view-projection matrix for unprojection
    pub fn inv_view_proj_matrix(&self) -> Mat4 {
        self.view_proj_matrix().inverse()
    }
    
    /// Orbit camera by delta angles
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch += delta_pitch;
        
        // Clamp pitch to avoid gimbal lock
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        
        self.update_position();
    }
    
    /// Zoom camera by delta distance
    pub fn zoom(&mut self, delta: f32) {
        self.distance += delta;
        self.distance = self.distance.clamp(2.0, 200.0);
        self.update_position();
    }
    
    /// Pan camera in screen space
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let right = (self.target - self.position)
            .cross(self.up)
            .normalize();
        let up = right.cross(self.target - self.position).normalize();
        
        let pan_speed = self.distance * 0.001;
        self.target += right * delta_x * pan_speed;
        self.target += up * delta_y * pan_speed;
        self.update_position();
    }
    
    /// Update aspect ratio (on window resize)
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
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
    fn test_orbit() {
        let mut camera = Camera::new(1.0);
        let initial_yaw = camera.yaw;
        
        camera.orbit(0.1, 0.0);
        assert!((camera.yaw - initial_yaw - 0.1).abs() < 0.001);
    }
    
    #[test]
    fn test_zoom() {
        let mut camera = Camera::new(1.0);
        let initial_distance = camera.distance;
        
        camera.zoom(-2.0);
        assert_eq!(camera.distance, initial_distance - 2.0);
        
        // Test clamping
        camera.zoom(-1000.0);
        assert!(camera.distance >= 2.0);
    }
}

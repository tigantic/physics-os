//! Camera System for Globe Viewing
//!
//! Orbital camera that rotates around the globe.

use bytemuck::{Pod, Zeroable};

/// Camera uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _padding: f32,
}

/// Orbital camera controller
pub struct Camera {
    /// Distance from center
    pub radius: f32,
    /// Horizontal angle (radians)
    pub theta: f32,
    /// Vertical angle (radians) - clamped to avoid gimbal lock
    pub phi: f32,
    /// Field of view (radians)
    pub fov: f32,
    /// Aspect ratio (width/height)
    pub aspect: f32,
    /// Near clip plane
    pub near: f32,
    /// Far clip plane
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            radius: 3.0,
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_4, // 45 degrees
            fov: std::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Get camera position in world space
    pub fn position(&self) -> [f32; 3] {
        let sin_phi = self.phi.sin();
        let cos_phi = self.phi.cos();
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();
        
        [
            self.radius * sin_phi * cos_theta,
            self.radius * cos_phi,
            self.radius * sin_phi * sin_theta,
        ]
    }
    
    /// Build view-projection matrix
    pub fn build_view_proj(&self) -> [[f32; 4]; 4] {
        let pos = self.position();
        
        // View matrix: look at origin
        let view = look_at(pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        
        // Perspective projection
        let proj = perspective(self.fov, self.aspect, self.near, self.far);
        
        // Combine
        mat4_mul(&proj, &view)
    }
    
    /// Get uniform data for GPU
    pub fn uniform(&self) -> CameraUniform {
        CameraUniform {
            view_proj: self.build_view_proj(),
            camera_pos: self.position(),
            _padding: 0.0,
        }
    }
    
    /// Rotate camera
    pub fn rotate(&mut self, d_theta: f32, d_phi: f32) {
        self.theta += d_theta;
        self.phi = (self.phi + d_phi).clamp(0.1, std::f32::consts::PI - 0.1);
    }
    
    /// Zoom camera
    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius + delta).clamp(1.5, 10.0);
    }
    
    /// Update aspect ratio
    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.aspect = width / height;
    }
}

// Matrix math utilities (avoiding external dependencies)

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize(sub(target, eye));
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    let nf = 1.0 / (near - far);
    
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[k][j] * b[i][k];
            }
        }
    }
    result
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0001 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_camera_position() {
        let cam = Camera::default();
        let pos = cam.position();
        
        // Should be at some distance from origin
        let dist = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!((dist - cam.radius).abs() < 0.001);
    }
}

// Phase 4: Globe Geometry
// Icosphere mesh generation with adaptive subdivision and RTE coordinates
// Constitutional compliance: Doctrine 1 (procedural generation), Doctrine 3 (GPU-first)

use glam::{Vec3, Mat4};
use std::f32::consts::PI;

/// Globe rendering configuration
#[derive(Debug, Clone)]
pub struct GlobeConfig {
    /// Globe radius (normalized to 1.0 for rendering)
    pub radius: f64,
    /// Subdivision level (0-8, controls mesh density)
    pub subdivision_level: u32,
}

impl Default for GlobeConfig {
    fn default() -> Self {
        Self {
            radius: 1.0, // Normalized for camera math
            subdivision_level: 5, // ~10k vertices
        }
    }
}

/// Icosphere vertex with geodetic coordinates
#[derive(Debug, Clone, Copy)]
pub struct GlobeVertex {
    /// Position in ECEF coordinates (meters, double precision simulated)
    pub position: Vec3,
    /// Latitude in radians (-π/2 to π/2)
    pub lat: f32,
    /// Longitude in radians (-π to π)
    pub lon: f32,
    /// Normalized normal vector
    pub normal: Vec3,
    /// UV coordinates for texture mapping (0-1)
    pub uv: [f32; 2],
}

/// Icosphere mesh for globe rendering
pub struct Icosphere {
    /// Vertex buffer (ECEF coordinates)
    pub vertices: Vec<GlobeVertex>,
    /// Index buffer (triangles)
    pub indices: Vec<u32>,
    /// Configuration
    pub config: GlobeConfig,
}

impl Icosphere {
    /// Create new icosphere with given subdivision level
    pub fn new(config: GlobeConfig) -> Self {
        let mut icosphere = Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            config,
        };
        
        icosphere.generate_base_icosahedron();
        icosphere.subdivide();
        icosphere.compute_geodetic_coords();
        
        icosphere
    }
    
    /// Generate base icosahedron (20 triangular faces)
    fn generate_base_icosahedron(&mut self) {
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0; // Golden ratio
        let a = 1.0;
        let b = 1.0 / phi;
        
        // 12 vertices of icosahedron (normalized)
        let base_vertices = [
            Vec3::new(0.0, b, -a), Vec3::new(b, a, 0.0),  Vec3::new(-b, a, 0.0),
            Vec3::new(0.0, b, a),  Vec3::new(0.0, -b, a), Vec3::new(-a, 0.0, b),
            Vec3::new(0.0, -b, -a), Vec3::new(a, 0.0, -b), Vec3::new(a, 0.0, b),
            Vec3::new(-a, 0.0, -b), Vec3::new(b, -a, 0.0), Vec3::new(-b, -a, 0.0),
        ];
        
        self.vertices = base_vertices
            .iter()
            .map(|&pos| {
                let normalized = pos.normalize();
                GlobeVertex {
                    position: normalized * self.config.radius as f32,
                    lat: 0.0, // Computed later
                    lon: 0.0,
                    normal: normalized,
                    uv: [0.0, 0.0],
                }
            })
            .collect();
        
        // 20 triangular faces
        self.indices = vec![
            0, 1, 2,   1, 0, 7,   1, 7, 8,   1, 8, 3,   1, 3, 2,
            2, 3, 5,   2, 5, 9,   2, 9, 0,   0, 9, 6,   0, 6, 7,
            7, 6, 10,  7, 10, 8,  8, 10, 4,  8, 4, 3,   3, 4, 5,
            5, 4, 11,  5, 11, 9,  9, 11, 6,  6, 11, 10, 10, 11, 4,
        ];
    }
    
    /// Subdivide each triangle recursively
    fn subdivide(&mut self) {
        for _ in 0..self.config.subdivision_level {
            let mut new_indices = Vec::new();
            let old_indices = self.indices.clone(); // Clone to avoid borrow conflict
            
            // Process each triangle
            for chunk in old_indices.chunks(3) {
                let v0 = chunk[0];
                let v1 = chunk[1];
                let v2 = chunk[2];
                
                // Compute midpoint vertices
                let m0 = self.add_midpoint(v0, v1);
                let m1 = self.add_midpoint(v1, v2);
                let m2 = self.add_midpoint(v2, v0);
                
                // Create 4 sub-triangles
                new_indices.extend_from_slice(&[v0, m0, m2]);
                new_indices.extend_from_slice(&[v1, m1, m0]);
                new_indices.extend_from_slice(&[v2, m2, m1]);
                new_indices.extend_from_slice(&[m0, m1, m2]);
            }
            
            self.indices = new_indices;
        }
    }
    
    /// Add midpoint between two vertices (or reuse existing)
    fn add_midpoint(&mut self, v0: u32, v1: u32) -> u32 {
        let pos0 = self.vertices[v0 as usize].position;
        let pos1 = self.vertices[v1 as usize].position;
        
        // Compute midpoint on sphere surface
        let mid = ((pos0 + pos1) * 0.5).normalize();
        let scaled_mid = mid * self.config.radius as f32;
        
        // Check if this midpoint already exists (simple linear search, could optimize)
        for (i, vertex) in self.vertices.iter().enumerate() {
            if (vertex.position - scaled_mid).length() < 0.001 {
                return i as u32;
            }
        }
        
        // Add new vertex
        let idx = self.vertices.len() as u32;
        self.vertices.push(GlobeVertex {
            position: scaled_mid,
            lat: 0.0, // Computed later
            lon: 0.0,
            normal: mid,
            uv: [0.0, 0.0],
        });
        
        idx
    }
    
    /// Compute geodetic coordinates (lat/lon) and UV for each vertex
    fn compute_geodetic_coords(&mut self) {
        for vertex in &mut self.vertices {
            let pos = vertex.position;
            
            // Compute latitude and longitude from ECEF position
            let r = pos.length();
            vertex.lat = (pos.y / r).asin();
            vertex.lon = pos.z.atan2(pos.x);
            
            // UV mapping (equirectangular projection)
            vertex.uv[0] = (vertex.lon + PI) / (2.0 * PI); // 0-1
            vertex.uv[1] = (vertex.lat + PI / 2.0) / PI;   // 0-1
        }
    }
    
    // Phase 4-5 scaffolding: RTE coordinate transform for camera integration
    #[allow(dead_code)]
    /// Transform vertices to Relative-To-Eye (RTE) coordinates to prevent jitter
    pub fn to_rte_vertices(&self, camera_pos: Vec3) -> Vec<Vec3> {
        self.vertices
            .iter()
            .map(|v| v.position - camera_pos)
            .collect()
    }
    
    /// Get vertex count
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    
    /// Get triangle count
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Camera controller for globe navigation (Google Earth style orbit camera)
pub struct GlobeCamera {
    /// Camera position in normalized coordinates (globe radius = 1.0)
    pub position: Vec3,
    /// Orbit angles: (longitude, latitude) in radians
    pub orbit_angles: (f32, f32),
    /// Distance from globe center (1.0 = surface)
    pub distance: f32,
    /// Target distance for smooth zoom
    pub target_distance: f32,
    /// Target orbit angles for smooth pan
    pub target_orbit: (f32, f32),
    /// Zoom level (legacy, derived from distance)
    pub zoom: f32,
    /// Interpolation speed (0-1, higher = faster)
    pub lerp_speed: f32,
}

impl GlobeCamera {
    /// Create new camera orbiting globe (Google Earth style)
    pub fn new() -> Self {
        let distance = 3.0; // 3x globe radius = nice view
        let orbit = (0.0_f32, 0.3_f32); // Slight tilt to see Earth from above
        let pos = Self::orbit_to_position(orbit.0, orbit.1, distance);
        Self {
            position: pos,
            orbit_angles: orbit,
            distance,
            target_distance: distance,
            target_orbit: orbit,
            zoom: distance,
            lerp_speed: 5.0,
        }
    }
    
    /// Convert orbit angles to camera position
    fn orbit_to_position(lon: f32, lat: f32, dist: f32) -> Vec3 {
        let lat_clamped = lat.clamp(-PI * 0.49, PI * 0.49); // Prevent gimbal lock
        Vec3::new(
            dist * lat_clamped.cos() * lon.sin(),
            dist * lat_clamped.sin(),
            dist * lat_clamped.cos() * lon.cos(),
        )
    }
    
    /// Update camera position with smooth interpolation
    pub fn update(&mut self, delta_time: f32) {
        let t = (self.lerp_speed * delta_time).min(1.0);
        
        // Interpolate orbit angles
        self.orbit_angles.0 += (self.target_orbit.0 - self.orbit_angles.0) * t;
        self.orbit_angles.1 += (self.target_orbit.1 - self.orbit_angles.1) * t;
        
        // Interpolate distance
        self.distance += (self.target_distance - self.distance) * t;
        self.zoom = self.distance;
        
        // Update position from orbit
        self.position = Self::orbit_to_position(self.orbit_angles.0, self.orbit_angles.1, self.distance);
    }
    
    /// Pan camera by screen-space delta (rotates around globe)
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        // Rotate around globe (Google Earth style)
        self.target_orbit.0 += delta_x * 2.0;  // Longitude
        self.target_orbit.1 += delta_y * 2.0;  // Latitude
        // Clamp latitude to avoid flipping
        self.target_orbit.1 = self.target_orbit.1.clamp(-PI * 0.49, PI * 0.49);
    }
    
    /// Zoom camera (logarithmic scaling, can zoom very close)
    pub fn zoom(&mut self, delta: f32) {
        // Zoom closer with scroll up, further with scroll down
        let zoom_speed = 0.15;
        self.target_distance = (self.target_distance * (1.0 - delta * zoom_speed)).clamp(1.05, 20.0);
    }
    
    /// Get view matrix
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.position;
        let center = Vec3::ZERO; // Always look at globe center
        let up = Vec3::Y;
        Mat4::look_at_rh(eye, center, up)
    }
    
    /// Get projection matrix (adjusted for close-up viewing)
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let near = (self.distance - 1.0).max(0.001); // Near plane just outside globe
        let far = self.distance + 10.0;
        Mat4::perspective_rh(45.0_f32.to_radians(), aspect_ratio, near, far)
    }
}

impl Default for GlobeCamera {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_icosphere_generation() {
        let config = GlobeConfig {
            subdivision_level: 2,
            ..Default::default()
        };
        let icosphere = Icosphere::new(config);
        
        // Base icosahedron has 12 vertices
        // Each subdivision quadruples triangle count
        // Subdivision 0: 12 vertices, 20 triangles
        // Subdivision 1: ~42 vertices, 80 triangles
        // Subdivision 2: ~162 vertices, 320 triangles
        assert!(icosphere.vertex_count() > 100);
        assert!(icosphere.triangle_count() > 200);
    }
    
    #[test]
    fn test_geodetic_coords() {
        let config = GlobeConfig {
            subdivision_level: 0,
            ..Default::default()
        };
        let icosphere = Icosphere::new(config);
        
        // All vertices should have valid lat/lon
        for vertex in &icosphere.vertices {
            assert!(vertex.lat >= -PI / 2.0 && vertex.lat <= PI / 2.0);
            assert!(vertex.lon >= -PI && vertex.lon <= PI);
            assert!(vertex.uv[0] >= 0.0 && vertex.uv[0] <= 1.0);
            assert!(vertex.uv[1] >= 0.0 && vertex.uv[1] <= 1.0);
        }
    }
    
    #[test]
    fn test_rte_transform() {
        let config = GlobeConfig::default();
        let icosphere = Icosphere::new(config);
        let camera_pos = Vec3::new(0.0, 0.0, 10_000_000.0);
        
        let rte_vertices = icosphere.to_rte_vertices(camera_pos);
        
        // RTE vertices should be camera-relative
        assert_eq!(rte_vertices.len(), icosphere.vertex_count());
        
        // Check that vertices are shifted relative to camera
        let original = icosphere.vertices[0].position;
        let rte = rte_vertices[0];
        assert!((rte - (original - camera_pos)).length() < 0.1);
    }
}

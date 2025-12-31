// Phase 4: Globe Geometry
// Icosphere mesh generation with adaptive subdivision and RTE coordinates
// Constitutional compliance: Doctrine 1 (procedural generation), Doctrine 3 (GPU-first)
#![allow(dead_code)]  // Camera methods and config fields ready for Phase 8 integration

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
            subdivision_level: 6, // ~40k vertices - smoother sphere
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
    /// Handles seam duplication and pole singularities
    fn compute_geodetic_coords(&mut self) {
        // First pass: compute lat/lon for all vertices
        for vertex in &mut self.vertices {
            let pos = vertex.position;
            let r = pos.length();
            vertex.lat = (pos.y / r).asin();
            vertex.lon = pos.z.atan2(pos.x);
        }
        
        // Second pass: fix triangles that cross the antimeridian seam
        // and duplicate vertices as needed
        self.fix_seam_triangles();
        
        // Third pass: compute UV coordinates
        for vertex in &mut self.vertices {
            // UV mapping (equirectangular projection)
            vertex.uv[0] = (vertex.lon + PI) / (2.0 * PI); // 0-1
            vertex.uv[1] = (vertex.lat + PI / 2.0) / PI;   // 0-1
        }
    }
    
    /// Fix triangles that cross the antimeridian (±180° longitude)
    /// by duplicating vertices and adjusting UV coordinates
    fn fix_seam_triangles(&mut self) {
        let mut new_indices = Vec::new();
        let _original_vertex_count = self.vertices.len();
        
        for chunk in self.indices.chunks(3) {
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;
            
            let lon0 = self.vertices[i0].lon;
            let lon1 = self.vertices[i1].lon;
            let lon2 = self.vertices[i2].lon;
            
            // Check if triangle crosses the antimeridian
            // (large longitude difference indicates wrap-around)
            let d01 = (lon0 - lon1).abs();
            let d12 = (lon1 - lon2).abs();
            let d20 = (lon2 - lon0).abs();
            
            let crosses_seam = d01 > PI || d12 > PI || d20 > PI;
            
            if crosses_seam {
                // Duplicate vertices with adjusted longitude for seam-crossing triangles
                let mut new_v0 = self.vertices[i0];
                let mut new_v1 = self.vertices[i1];
                let mut new_v2 = self.vertices[i2];
                
                // Shift negative longitudes to positive side (+2π)
                if new_v0.lon < 0.0 { new_v0.lon += 2.0 * PI; }
                if new_v1.lon < 0.0 { new_v1.lon += 2.0 * PI; }
                if new_v2.lon < 0.0 { new_v2.lon += 2.0 * PI; }
                
                let new_i0 = self.vertices.len() as u32;
                self.vertices.push(new_v0);
                let new_i1 = self.vertices.len() as u32;
                self.vertices.push(new_v1);
                let new_i2 = self.vertices.len() as u32;
                self.vertices.push(new_v2);
                
                new_indices.push(new_i0);
                new_indices.push(new_i1);
                new_indices.push(new_i2);
            } else {
                new_indices.push(chunk[0]);
                new_indices.push(chunk[1]);
                new_indices.push(chunk[2]);
            }
        }
        
        self.indices = new_indices;
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
/// Uses spherical coordinates: azimuth (horizontal) + polar (vertical)
/// Earth is FIXED at origin, camera ORBITS around it
/// Features: Momentum/inertia for "throw and glide" feel
pub struct GlobeCamera {
    /// Camera position in normalized coordinates (globe radius = 1.0)
    pub position: Vec3,
    /// Azimuth angle (horizontal orbit, radians) - 0 = looking from +Z
    pub azimuth: f32,
    /// Polar angle (vertical orbit, radians) - PI/2 = equator, 0 = north pole
    pub polar: f32,
    /// Distance from globe center (1.0 = surface)
    pub radius: f32,
    /// Target distance for smooth zoom
    pub target_radius: f32,
    /// Zoom level (legacy, derived from distance)
    pub zoom: f32,
    
    // ═══ INERTIA PHYSICS (Google Earth feel) ═══
    /// Angular velocity for azimuth (radians per second)
    pub velocity_azimuth: f32,
    /// Angular velocity for polar (radians per second)  
    pub velocity_polar: f32,
    /// Drag factor - how fast it slows down (0.95 = smooth glide)
    pub drag_factor: f32,
    /// Whether user is currently dragging (no friction during drag)
    pub is_dragging: bool,
}

impl GlobeCamera {
    /// Create new camera orbiting globe (Google Earth style with inertia)
    pub fn new() -> Self {
        let radius = 5.0; // Pulled back for telephoto lens (narrower FOV)
        let azimuth = 0.0_f32;
        let polar = 1.4_f32; // ~80° from pole = nice Earth view
        let pos = Self::spherical_to_cartesian(azimuth, polar, radius);
        Self {
            position: pos,
            azimuth,
            polar,
            radius,
            target_radius: radius,
            zoom: radius,
            // Inertia physics
            velocity_azimuth: 0.0,
            velocity_polar: 0.0,
            drag_factor: 0.92, // Smooth glide (lower = more friction)
            is_dragging: false,
        }
    }
    
    /// Convert spherical coordinates (azimuth, polar, radius) to Cartesian
    /// azimuth: horizontal angle (0 = +Z axis, increases counter-clockwise when viewed from above)
    /// polar: vertical angle (0 = +Y pole, PI = -Y pole, PI/2 = equator)
    fn spherical_to_cartesian(azimuth: f32, polar: f32, radius: f32) -> Vec3 {
        Vec3::new(
            radius * polar.sin() * azimuth.sin(),
            radius * polar.cos(),
            radius * polar.sin() * azimuth.cos(),
        )
    }
    
    /// Update camera with physics-based momentum (call every frame)
    pub fn update(&mut self, delta_time: f32) {
        // [TUNING] INERTIA:
        // 0.80 = Very responsive, minimal drift
        // 0.85 = Responsive, slight drift  
        // 0.92 = Too slippery
        let inertia = 0.85;

        // Apply velocity to position
        self.azimuth += self.velocity_azimuth * delta_time;
        self.polar += self.velocity_polar * delta_time;
        
        // Apply friction
        if !self.is_dragging {
            self.velocity_azimuth *= inertia;
            self.velocity_polar *= inertia;
            
            // Higher threshold - stops sooner
            if self.velocity_azimuth.abs() < 0.005 { self.velocity_azimuth = 0.0; }
            if self.velocity_polar.abs() < 0.005 { self.velocity_polar = 0.0; }
        }
        
        // Clamp polar (prevents flipping)
        self.polar = self.polar.clamp(0.05, std::f32::consts::PI - 0.05);
        
        // Smooth zoom
        let zoom_lerp = (5.0 * delta_time).min(1.0);
        self.radius += (self.target_radius - self.radius) * zoom_lerp;
        self.zoom = self.radius;
        
        self.position = Self::spherical_to_cartesian(self.azimuth, self.polar, self.radius);
    }

    /// Pan camera - NATURAL DIRECTION
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        // [TUNING] SENSITIVITY: Moderate for controllable response
        let sensitivity = 0.5; 
        
        // Natural direction (not inverted)
        self.velocity_azimuth += delta_x * sensitivity * 0.01;
        self.velocity_polar += delta_y * sensitivity * 0.01;
    }
    
    /// Orbit camera (alias for pan - adds angular velocity)
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.pan(delta_x, delta_y);
    }
    
    /// Start dragging (disable friction during drag)
    pub fn start_drag(&mut self) {
        self.is_dragging = true;
        // Don't zero velocity - preserve momentum direction
    }
    
    /// Stop dragging (enable friction for glide)
    pub fn stop_drag(&mut self) {
        self.is_dragging = false;
    }
    
    /// Zoom camera (proportional speed, can zoom very close)
    pub fn zoom(&mut self, delta: f32) {
        let zoom_speed = self.radius * 0.1; // Proportional to distance
        self.target_radius -= delta * zoom_speed;
        self.target_radius = self.target_radius.clamp(1.15, 20.0); // Don't clip into Earth
    }
    
    /// Get view matrix (always looking at origin)
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, Vec3::ZERO, Vec3::Y)
    }
    
    /// Get projection matrix - TELEPHOTO LENS for technical instrument look
    /// Narrow FOV (25°) eliminates fisheye distortion - "God's Eye" view
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        // TELEPHOTO: 25° FOV eliminates edge distortion (vs 45-60° wide angle)
        // This creates the flat, technical, orthographic-like feel of Google Earth
        let fov = 25.0_f32.to_radians();
        let near = 0.01;
        let far = 100.0;
        Mat4::perspective_rh(fov, aspect_ratio, near, far)
    }
    
    // Legacy accessor for orbit_angles compatibility
    #[allow(dead_code)]
    pub fn orbit_angles(&self) -> (f32, f32) {
        (self.azimuth, self.polar - std::f32::consts::FRAC_PI_2)
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
        
        // Verify all vertices have valid latitude
        let epsilon = 1e-6;
        for vertex in &icosphere.vertices {
            assert!(vertex.lat >= -PI / 2.0 - epsilon && vertex.lat <= PI / 2.0 + epsilon,
                    "lat {} out of range", vertex.lat);
        }
        
        // Verify we have vertices (non-empty mesh)
        assert!(!icosphere.vertices.is_empty());
        assert!(!icosphere.indices.is_empty());
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

//! Phase 3C-4: Tube Geometry Generator
//!
//! Generates 3D tunnel/tube meshes from trajectory waypoints for the
//! "Tunnel in the Sky" guidance visualization.
//!
//! The tube represents the safe flight corridor - staying within the tube
//! ensures the vehicle remains on the optimized trajectory through the
//! atmospheric hazard field.

use glam::{Vec3, Quat, Mat4};
use std::f32::consts::{PI, TAU};

/// Tube mesh data ready for GPU upload
#[derive(Debug, Clone)]
pub struct TubeMesh {
    /// Vertex positions (3 floats per vertex)
    pub positions: Vec<f32>,
    /// Vertex normals (3 floats per vertex)
    pub normals: Vec<f32>,
    /// Texture coordinates (2 floats per vertex)
    pub uvs: Vec<f32>,
    /// Triangle indices
    pub indices: Vec<u32>,
    /// Number of vertices
    pub vertex_count: u32,
    /// Number of triangles
    pub triangle_count: u32,
}

impl TubeMesh {
    /// Create empty mesh
    pub fn empty() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            vertex_count: 0,
            triangle_count: 0,
        }
    }
    
    /// Get interleaved vertex data [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, u, v]
    pub fn interleaved(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.vertex_count as usize * 8);
        
        for i in 0..self.vertex_count as usize {
            // Position
            data.push(self.positions[i * 3]);
            data.push(self.positions[i * 3 + 1]);
            data.push(self.positions[i * 3 + 2]);
            // Normal
            data.push(self.normals[i * 3]);
            data.push(self.normals[i * 3 + 1]);
            data.push(self.normals[i * 3 + 2]);
            // UV
            data.push(self.uvs[i * 2]);
            data.push(self.uvs[i * 2 + 1]);
        }
        
        data
    }
}

/// Tube geometry configuration
#[derive(Debug, Clone, Copy)]
pub struct TubeConfig {
    /// Tube radius in world units
    pub radius: f32,
    /// Number of segments around the tube circumference
    pub radial_segments: u32,
    /// Generate end caps
    pub caps: bool,
    /// UV tiling along length
    pub uv_tile_length: f32,
}

impl Default for TubeConfig {
    fn default() -> Self {
        Self {
            radius: 0.5,
            radial_segments: 16,
            caps: false,
            uv_tile_length: 1.0,
        }
    }
}

/// Generate a tube mesh following a path of waypoints
///
/// Creates a cylindrical mesh that smoothly follows the given path,
/// suitable for rendering as a "tunnel in the sky" guidance corridor.
///
/// # Arguments
/// * `path` - Slice of Vec3 waypoints defining the centerline
/// * `config` - Tube geometry configuration
///
/// # Returns
/// * `TubeMesh` with positions, normals, UVs, and indices
pub fn generate_tube_mesh(path: &[Vec3], config: TubeConfig) -> TubeMesh {
    if path.len() < 2 {
        return TubeMesh::empty();
    }
    
    let n_points = path.len();
    let segments = config.radial_segments as usize;
    
    // Pre-allocate
    let verts_per_ring = segments + 1; // +1 for UV seam
    let total_verts = n_points * verts_per_ring;
    let total_tris = (n_points - 1) * segments * 2;
    
    let mut positions = Vec::with_capacity(total_verts * 3);
    let mut normals = Vec::with_capacity(total_verts * 3);
    let mut uvs = Vec::with_capacity(total_verts * 2);
    let mut indices = Vec::with_capacity(total_tris * 3);
    
    // Compute frames along path (parallel transport)
    let frames = compute_parallel_transport_frames(path);
    
    // Cumulative arc length for UV mapping
    let mut arc_length = 0.0f32;
    let mut arc_lengths = vec![0.0f32; n_points];
    for i in 1..n_points {
        arc_length += (path[i] - path[i - 1]).length();
        arc_lengths[i] = arc_length;
    }
    let total_length = arc_length.max(0.001);
    
    // Generate vertices for each ring
    for i in 0..n_points {
        let (tangent, normal, binormal) = frames[i];
        let center = path[i];
        let v = arc_lengths[i] / config.uv_tile_length;
        
        for s in 0..=segments {
            let theta = (s as f32 / segments as f32) * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            
            // Position on ring
            let offset = (normal * cos_t + binormal * sin_t) * config.radius;
            let pos = center + offset;
            
            // Normal points outward from center
            let norm = offset.normalize();
            
            // UV: u wraps around, v follows arc length
            let u = s as f32 / segments as f32;
            
            positions.extend_from_slice(&[pos.x, pos.y, pos.z]);
            normals.extend_from_slice(&[norm.x, norm.y, norm.z]);
            uvs.extend_from_slice(&[u, v]);
        }
    }
    
    // Generate indices (triangle strip between adjacent rings)
    for i in 0..(n_points - 1) {
        let ring_start = (i * verts_per_ring) as u32;
        let next_ring_start = ((i + 1) * verts_per_ring) as u32;
        
        for s in 0..segments as u32 {
            let a = ring_start + s;
            let b = ring_start + s + 1;
            let c = next_ring_start + s;
            let d = next_ring_start + s + 1;
            
            // Two triangles per quad
            indices.extend_from_slice(&[a, c, b]);
            indices.extend_from_slice(&[b, c, d]);
        }
    }
    
    TubeMesh {
        positions,
        normals,
        uvs,
        indices,
        vertex_count: total_verts as u32,
        triangle_count: total_tris as u32,
    }
}

/// Compute parallel transport frames along a path
///
/// Returns (tangent, normal, binormal) for each point, maintaining
/// a smooth rotation-minimizing frame.
fn compute_parallel_transport_frames(path: &[Vec3]) -> Vec<(Vec3, Vec3, Vec3)> {
    let n = path.len();
    let mut frames = Vec::with_capacity(n);
    
    if n < 2 {
        return vec![(Vec3::Z, Vec3::X, Vec3::Y)];
    }
    
    // First frame: use world up to define initial orientation
    let first_tangent = (path[1] - path[0]).normalize_or_zero();
    let world_up = if first_tangent.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let first_binormal = first_tangent.cross(world_up).normalize_or_zero();
    let first_normal = first_binormal.cross(first_tangent).normalize_or_zero();
    
    frames.push((first_tangent, first_normal, first_binormal));
    
    // Propagate frames along path using parallel transport
    for i in 1..n {
        let prev_tangent = frames[i - 1].0;
        let prev_normal = frames[i - 1].1;
        let prev_binormal = frames[i - 1].2;
        
        // Current tangent
        let tangent = if i < n - 1 {
            (path[i + 1] - path[i - 1]).normalize_or_zero()
        } else {
            (path[i] - path[i - 1]).normalize_or_zero()
        };
        
        // Rotation from previous to current tangent
        let axis = prev_tangent.cross(tangent);
        let cos_angle = prev_tangent.dot(tangent).clamp(-1.0, 1.0);
        
        let (normal, binormal) = if axis.length_squared() > 1e-10 {
            let angle = cos_angle.acos();
            let rotation = Quat::from_axis_angle(axis.normalize(), angle);
            
            let normal = rotation * prev_normal;
            let binormal = rotation * prev_binormal;
            (normal, binormal)
        } else if cos_angle < 0.0 {
            // 180 degree flip
            (-prev_normal, -prev_binormal)
        } else {
            // No rotation needed
            (prev_normal, prev_binormal)
        };
        
        frames.push((tangent, normal.normalize_or_zero(), binormal.normalize_or_zero()));
    }
    
    frames
}

/// Generate a wireframe tube (lines only, no filled triangles)
///
/// Useful for overlay rendering without depth testing.
pub fn generate_tube_wireframe(path: &[Vec3], config: TubeConfig) -> (Vec<f32>, Vec<u32>) {
    if path.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    
    let n_points = path.len();
    let segments = config.radial_segments as usize;
    
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    
    let frames = compute_parallel_transport_frames(path);
    
    // Generate ring vertices
    for i in 0..n_points {
        let (_, normal, binormal) = frames[i];
        let center = path[i];
        
        for s in 0..segments {
            let theta = (s as f32 / segments as f32) * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            
            let offset = (normal * cos_t + binormal * sin_t) * config.radius;
            let pos = center + offset;
            
            positions.extend_from_slice(&[pos.x, pos.y, pos.z]);
        }
    }
    
    // Ring edges (circumference)
    for i in 0..n_points {
        let ring_start = (i * segments) as u32;
        for s in 0..segments as u32 {
            let a = ring_start + s;
            let b = ring_start + (s + 1) % segments as u32;
            indices.push(a);
            indices.push(b);
        }
    }
    
    // Longitudinal edges (along path)
    for i in 0..(n_points - 1) {
        let ring_start = (i * segments) as u32;
        let next_ring_start = ((i + 1) * segments) as u32;
        
        // Only add every 4th line to reduce clutter
        for s in (0..segments).step_by(4) {
            let a = ring_start + s as u32;
            let b = next_ring_start + s as u32;
            indices.push(a);
            indices.push(b);
        }
    }
    
    (positions, indices)
}

/// Generate a gradient-colored tube for hazard visualization
///
/// Colors can represent cost/danger along the path.
pub fn generate_tube_with_colors(
    path: &[Vec3],
    costs: &[f32],
    config: TubeConfig,
) -> (TubeMesh, Vec<f32>) {
    let mesh = generate_tube_mesh(path, config);
    
    // Generate per-vertex colors based on path costs
    let n_points = path.len();
    let verts_per_ring = config.radial_segments as usize + 1;
    
    let mut colors = Vec::with_capacity(mesh.vertex_count as usize * 4);
    
    for i in 0..n_points {
        let cost = costs.get(i).copied().unwrap_or(0.0);
        let t = cost.clamp(0.0, 1.0);
        
        // Green (safe) -> Yellow -> Red (danger)
        let (r, g, b) = if t < 0.5 {
            (t * 2.0, 1.0, 0.0)
        } else {
            (1.0, 2.0 - t * 2.0, 0.0)
        };
        
        for _ in 0..verts_per_ring {
            colors.extend_from_slice(&[r, g, b, 0.7]); // 70% opacity
        }
    }
    
    (mesh, colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_tube() {
        let path = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        
        let config = TubeConfig {
            radius: 0.5,
            radial_segments: 8,
            ..Default::default()
        };
        
        let mesh = generate_tube_mesh(&path, config);
        
        // 3 rings × 9 verts = 27 vertices
        assert_eq!(mesh.vertex_count, 27);
        // 2 segments × 8 quads × 2 tris = 32 triangles
        assert_eq!(mesh.triangle_count, 32);
        assert_eq!(mesh.indices.len(), 96); // 32 tris × 3
    }
    
    #[test]
    fn test_curved_tube() {
        let path = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 0.0, 2.0),
            Vec3::new(3.0, -1.0, 3.0),
        ];
        
        let mesh = generate_tube_mesh(&path, TubeConfig::default());
        
        assert!(mesh.vertex_count > 0);
        assert!(mesh.triangle_count > 0);
        
        // Check normals are unit length
        for i in 0..mesh.vertex_count as usize {
            let nx = mesh.normals[i * 3];
            let ny = mesh.normals[i * 3 + 1];
            let nz = mesh.normals[i * 3 + 2];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!((len - 1.0).abs() < 0.01, "Normal not unit length: {}", len);
        }
    }
    
    #[test]
    fn test_empty_path() {
        let mesh = generate_tube_mesh(&[], TubeConfig::default());
        assert_eq!(mesh.vertex_count, 0);
        
        let mesh = generate_tube_mesh(&[Vec3::ZERO], TubeConfig::default());
        assert_eq!(mesh.vertex_count, 0);
    }
}

// Phase 8: Dynamic Quadtree Globe
// Replaces static Icosphere with adaptive LOD terrain chunks
// Constitutional compliance: Doctrine 1 (procedural), Doctrine 3 (GPU-first)
#![allow(dead_code)] // Camera and config structs ready for integration

use glam::{Vec3, DVec3, Mat4};
use crate::tile_fetcher::TileCoord;
use crate::lod::LodConfig;

/// Vertex format matching shaders/globe.wgsl
/// Phase 3: Added tile_layer for texture array indexing
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobeVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub lat_lon: [f32; 2],
    /// Texture array layer index (-1 = no texture, use procedural)
    pub tile_layer: f32,
    /// Padding to maintain 16-byte alignment
    pub _padding: [f32; 3],
}

/// A chunk of the planetary terrain
pub struct GlobeChunk {
    pub tile_coord: TileCoord,
    pub face: u8,
    pub bounds: BoundingBox,
    pub children: Option<Box<[GlobeChunk; 4]>>,
    pub vertices: Vec<GlobeVertex>,
    pub indices: Vec<u32>,
    pub is_leaf: bool,
    pub texture_layer: i32, // -1 if waiting for texture
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub center: DVec3,
    pub radius: f64,
}

impl BoundingBox {
    fn from_chunk(globe_radius: f64, coord: TileCoord, face: u8) -> Self {
        let zoom_factor = 1.0 / (1u32 << coord.z) as f64;
        let chunk_radius = globe_radius * 2.0 * zoom_factor;
        
        // Calculate center based on tile coordinates and face
        let n = (1u32 << coord.z) as f64;
        let u = (coord.x as f64 + 0.5) / n * 2.0 - 1.0;
        let v = 1.0 - (coord.y as f64 + 0.5) / n * 2.0;
        
        let cube_center = match face {
            0 => DVec3::new(1.0, v, -u),   // +X
            1 => DVec3::new(-1.0, v, u),   // -X
            2 => DVec3::new(u, 1.0, -v),   // +Y
            3 => DVec3::new(u, -1.0, v),   // -Y
            4 => DVec3::new(u, v, 1.0),    // +Z
            5 => DVec3::new(-u, v, -1.0),  // -Z
            _ => DVec3::X,
        };
        
        let sphere_center = cube_center.normalize() * globe_radius;
        
        Self {
            center: sphere_center,
            radius: chunk_radius,
        }
    }
    
    pub fn distance_to(&self, point: Vec3) -> f32 {
        let p_d = DVec3::new(point.x as f64, point.y as f64, point.z as f64);
        ((self.center - p_d).length() - self.radius).max(0.0) as f32
    }
}

/// Dynamic Quadtree Globe Engine
/// Replaces static Icosphere with adaptive LOD terrain
pub struct QuadTreeGlobe {
    pub radius: f64,
    pub roots: Vec<GlobeChunk>,
    pub lod_config: LodConfig,
    /// Maximum subdivision depth (prevents infinite recursion)
    pub max_depth: u8,
    /// Grid resolution per chunk (higher = smoother but more vertices)
    pub grid_size: u32,
}

impl QuadTreeGlobe {
    pub fn new(radius: f64) -> Self {
        let mut globe = Self {
            radius,
            roots: Vec::with_capacity(6),
            lod_config: LodConfig::globe_scale(),
            max_depth: 12,
            grid_size: 16,
        };
        globe.generate_roots();
        globe
    }

    /// Generate the 6 root faces of the cube-sphere
    fn generate_roots(&mut self) {
        for face in 0..6u8 {
            self.roots.push(self.create_chunk(TileCoord { z: 0, x: 0, y: 0 }, face));
        }
    }

    fn create_chunk(&self, coord: TileCoord, face: u8) -> GlobeChunk {
        let (vertices, indices) = self.generate_mesh(coord, face);
        let bbox = BoundingBox::from_chunk(self.radius, coord, face);
        
        GlobeChunk {
            tile_coord: coord,
            face,
            bounds: bbox,
            children: None,
            vertices,
            indices,
            is_leaf: true,
            texture_layer: -1,
        }
    }

    /// Update LOD based on camera position
    pub fn update(&mut self, camera_pos: Vec3) {
        // We need to collect face indices first to avoid borrow issues
        for i in 0..self.roots.len() {
            let face = self.roots[i].face;
            self.update_chunk_recursive(i, camera_pos, face);
        }
    }
    
    fn update_chunk_recursive(&mut self, root_idx: usize, camera_pos: Vec3, face: u8) {
        // Get the root chunk and process it
        let root = &mut self.roots[root_idx];
        Self::update_chunk_inner(root, camera_pos, face, self.radius, self.max_depth, self.grid_size);
    }
    
    fn update_chunk_inner(
        chunk: &mut GlobeChunk, 
        camera_pos: Vec3, 
        face: u8,
        radius: f64,
        max_depth: u8,
        grid_size: u32,
    ) {
        let dist = chunk.bounds.distance_to(camera_pos);
        
        // Split logic: Distance < Threshold / ZoomLevel
        let split_factor = 4.0;
        let split_dist = (radius as f32 * split_factor) / (1u32 << chunk.tile_coord.z) as f32;
        
        if dist < split_dist && chunk.tile_coord.z < max_depth {
            if chunk.children.is_none() {
                Self::split_chunk(chunk, face, radius, grid_size);
            }
            
            // Recurse to children
            if let Some(children) = &mut chunk.children {
                for child in children.iter_mut() {
                    Self::update_chunk_inner(child, camera_pos, face, radius, max_depth, grid_size);
                }
            }
        } else if dist > split_dist * 1.2 {
            // Merge logic with hysteresis
            if chunk.children.is_some() {
                chunk.children = None;
                chunk.is_leaf = true;
                // Regenerate mesh if needed
                if chunk.vertices.is_empty() {
                    let (v, i) = Self::generate_mesh_static(radius, grid_size, chunk.tile_coord, face);
                    chunk.vertices = v;
                    chunk.indices = i;
                }
            }
        }
    }
    
    fn split_chunk(chunk: &mut GlobeChunk, face: u8, radius: f64, grid_size: u32) {
        let z = chunk.tile_coord.z + 1;
        let x = chunk.tile_coord.x * 2;
        let y = chunk.tile_coord.y * 2;
        
        let create = |coord: TileCoord| -> GlobeChunk {
            let (vertices, indices) = Self::generate_mesh_static(radius, grid_size, coord, face);
            let bbox = BoundingBox::from_chunk(radius, coord, face);
            GlobeChunk {
                tile_coord: coord,
                face,
                bounds: bbox,
                children: None,
                vertices,
                indices,
                is_leaf: true,
                texture_layer: -1,
            }
        };
        
        let children = [
            create(TileCoord { z, x, y }),
            create(TileCoord { z, x: x+1, y }),
            create(TileCoord { z, x, y: y+1 }),
            create(TileCoord { z, x: x+1, y: y+1 }),
        ];
        
        chunk.children = Some(Box::new(children));
        chunk.is_leaf = false;
    }

    /// Generate mesh for a chunk (Cube -> Sphere projection)
    fn generate_mesh(&self, coord: TileCoord, face: u8) -> (Vec<GlobeVertex>, Vec<u32>) {
        Self::generate_mesh_static(self.radius, self.grid_size, coord, face)
    }
    
    fn generate_mesh_static(radius: f64, grid_size: u32, coord: TileCoord, face: u8) -> (Vec<GlobeVertex>, Vec<u32>) {
        let mut vertices = Vec::with_capacity(((grid_size + 1) * (grid_size + 1)) as usize);
        let mut indices = Vec::with_capacity((grid_size * grid_size * 6) as usize);
        
        let step = 1.0 / grid_size as f64;
        let n = (1u32 << coord.z) as f64;
        
        let u_start = coord.x as f64 / n;
        let v_start = coord.y as f64 / n;
        let scale = 1.0 / n;

        for iy in 0..=grid_size {
            for ix in 0..=grid_size {
                let u_local = ix as f64 * step;
                let v_local = iy as f64 * step;
                
                let u_face = u_start + u_local * scale;
                let v_face = v_start + v_local * scale;
                
                // Map 0..1 to -1..1 range
                let px = u_face * 2.0 - 1.0;
                let py = 1.0 - v_face * 2.0;
                
                // Cube face projection
                let cube_point = match face {
                    0 => DVec3::new(1.0, py, -px),
                    1 => DVec3::new(-1.0, py, px),
                    2 => DVec3::new(px, 1.0, -py),
                    3 => DVec3::new(px, -1.0, py),
                    4 => DVec3::new(px, py, 1.0),
                    5 => DVec3::new(-px, py, -1.0),
                    _ => DVec3::ZERO,
                };
                
                // Spherify (normalize and scale)
                let sphere_point = cube_point.normalize();
                let position = sphere_point * radius;
                
                // Compute lat/lon
                let lat = sphere_point.y.asin();
                let lon = sphere_point.z.atan2(sphere_point.x);
                
                vertices.push(GlobeVertex {
                    position: [position.x as f32, position.y as f32, position.z as f32],
                    normal: [sphere_point.x as f32, sphere_point.y as f32, sphere_point.z as f32],
                    uv: [u_local as f32, v_local as f32],
                    lat_lon: [lat as f32, lon as f32],
                    tile_layer: -1.0, // Will be set when texture arrives
                    _padding: [0.0, 0.0, 0.0],
                });
            }
        }
        
        // Generate indices (two triangles per grid cell)
        for iy in 0..grid_size {
            for ix in 0..grid_size {
                let i0 = iy * (grid_size + 1) + ix;
                let i1 = i0 + 1;
                let i2 = (iy + 1) * (grid_size + 1) + ix;
                let i3 = i2 + 1;
                
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }
        
        (vertices, indices)
    }
    
    /// Collect visible leaf chunks for rendering
    pub fn get_render_chunks(&self) -> Vec<&GlobeChunk> {
        let mut chunks = Vec::new();
        for root in &self.roots {
            Self::collect_leaves(root, &mut chunks);
        }
        chunks
    }
    
    fn collect_leaves<'a>(chunk: &'a GlobeChunk, list: &mut Vec<&'a GlobeChunk>) {
        if chunk.is_leaf {
            list.push(chunk);
        } else if let Some(children) = &chunk.children {
            for child in children.iter() {
                Self::collect_leaves(child, list);
            }
        }
    }
    
    /// Get total vertex count for debugging
    pub fn total_vertices(&self) -> usize {
        let mut count = 0;
        for root in &self.roots {
            count += Self::count_vertices(root);
        }
        count
    }
    
    fn count_vertices(chunk: &GlobeChunk) -> usize {
        if chunk.is_leaf {
            chunk.vertices.len()
        } else if let Some(children) = &chunk.children {
            children.iter().map(Self::count_vertices).sum()
        } else {
            0
        }
    }
    
    /// Get total chunk count for debugging
    pub fn chunk_count(&self) -> usize {
        let mut count = 0;
        for root in &self.roots {
            count += Self::count_chunks(root);
        }
        count
    }
    
    fn count_chunks(chunk: &GlobeChunk) -> usize {
        if chunk.is_leaf {
            1
        } else if let Some(children) = &chunk.children {
            children.iter().map(Self::count_chunks).sum()
        } else {
            1
        }
    }
    
    /// Apply texture layer from TileTextureArray to all visible chunks
    /// Called after tile_array.update() to propagate loaded texture indices
    pub fn apply_texture_layer(&mut self, coord: TileCoord, layer: i32) {
        for root in &mut self.roots {
            Self::apply_layer_recursive(root, coord, layer);
        }
    }
    
    fn apply_layer_recursive(chunk: &mut GlobeChunk, coord: TileCoord, layer: i32) {
        if chunk.is_leaf {
            if chunk.tile_coord == coord && chunk.texture_layer != layer {
                chunk.texture_layer = layer;
                // Update all vertices with this layer
                for v in &mut chunk.vertices {
                    v.tile_layer = layer as f32;
                }
            }
        } else if let Some(children) = &mut chunk.children {
            for child in children.iter_mut() {
                Self::apply_layer_recursive(child, coord, layer);
            }
        }
    }
    
    /// Get mutable render chunks for texture layer updates
    pub fn get_render_chunks_mut(&mut self) -> Vec<&mut GlobeChunk> {
        let mut chunks = Vec::new();
        for root in &mut self.roots {
            Self::collect_leaves_mut(root, &mut chunks);
        }
        chunks
    }
    
    fn collect_leaves_mut<'a>(chunk: &'a mut GlobeChunk, list: &mut Vec<&'a mut GlobeChunk>) {
        if chunk.is_leaf {
            list.push(chunk);
        } else if let Some(children) = &mut chunk.children {
            for child in children.iter_mut() {
                Self::collect_leaves_mut(child, list);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LEGACY COMPATIBILITY LAYER
// Keep GlobeCamera from original globe.rs (it still works with quadtree)
// ═══════════════════════════════════════════════════════════════════════

/// Camera controller for globe navigation (Google Earth style orbit camera)
pub struct GlobeCamera {
    pub position: Vec3,
    pub azimuth: f32,
    pub polar: f32,
    pub radius: f32,
    pub target_radius: f32,
    pub zoom: f32,
    pub velocity_azimuth: f32,
    pub velocity_polar: f32,
    pub drag_factor: f32,
    pub is_dragging: bool,
}

impl GlobeCamera {
    pub fn new() -> Self {
        let radius = 5.0;
        let azimuth = 0.0_f32;
        let polar = 1.4_f32;
        let pos = Self::spherical_to_cartesian(azimuth, polar, radius);
        Self {
            position: pos,
            azimuth,
            polar,
            radius,
            target_radius: radius,
            zoom: radius,
            velocity_azimuth: 0.0,
            velocity_polar: 0.0,
            drag_factor: 0.92,
            is_dragging: false,
        }
    }
    
    fn spherical_to_cartesian(azimuth: f32, polar: f32, radius: f32) -> Vec3 {
        Vec3::new(
            radius * polar.sin() * azimuth.sin(),
            radius * polar.cos(),
            radius * polar.sin() * azimuth.cos(),
        )
    }
    
    pub fn update(&mut self, delta_time: f32) {
        let inertia = 0.85;

        self.azimuth += self.velocity_azimuth * delta_time;
        self.polar += self.velocity_polar * delta_time;
        
        if !self.is_dragging {
            self.velocity_azimuth *= inertia;
            self.velocity_polar *= inertia;
            
            if self.velocity_azimuth.abs() < 0.005 { self.velocity_azimuth = 0.0; }
            if self.velocity_polar.abs() < 0.005 { self.velocity_polar = 0.0; }
        }
        
        self.polar = self.polar.clamp(0.05, std::f32::consts::PI - 0.05);
        
        let zoom_lerp = (5.0 * delta_time).min(1.0);
        self.radius += (self.target_radius - self.radius) * zoom_lerp;
        self.zoom = self.radius;
        
        self.position = Self::spherical_to_cartesian(self.azimuth, self.polar, self.radius);
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.5;
        self.velocity_azimuth += delta_x * sensitivity * 0.01;
        self.velocity_polar += delta_y * sensitivity * 0.01;
    }
    
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.pan(delta_x, delta_y);
    }
    
    pub fn start_drag(&mut self) {
        self.is_dragging = true;
    }
    
    pub fn stop_drag(&mut self) {
        self.is_dragging = false;
    }
    
    pub fn zoom(&mut self, delta: f32) {
        let zoom_speed = self.radius * 0.1;
        self.target_radius -= delta * zoom_speed;
        self.target_radius = self.target_radius.clamp(1.15, 20.0);
    }
    
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, Vec3::ZERO, Vec3::Y)
    }
    
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let fov = 25.0_f32.to_radians();
        let near = 0.01;
        let far = 100.0;
        Mat4::perspective_rh(fov, aspect_ratio, near, far)
    }
}

impl Default for GlobeCamera {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LEGACY ICOSPHERE (Keep for fallback/comparison)
// ═══════════════════════════════════════════════════════════════════════

/// Globe rendering configuration
#[derive(Debug, Clone)]
pub struct GlobeConfig {
    pub radius: f64,
    pub subdivision_level: u32,
}

impl Default for GlobeConfig {
    fn default() -> Self {
        Self {
            radius: 1.0,
            subdivision_level: 6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quadtree_generation() {
        let globe = QuadTreeGlobe::new(1.0);
        assert_eq!(globe.roots.len(), 6);
        
        // Each root should have vertices
        for root in &globe.roots {
            assert!(!root.vertices.is_empty());
            assert!(!root.indices.is_empty());
        }
    }
    
    #[test]
    fn test_chunk_split() {
        let mut globe = QuadTreeGlobe::new(1.0);
        
        // Camera very close to one face should cause splits
        let close_camera = Vec3::new(1.5, 0.0, 0.0);
        globe.update(close_camera);
        
        // Should have more chunks now
        let chunks = globe.get_render_chunks();
        assert!(chunks.len() > 6);
    }
}

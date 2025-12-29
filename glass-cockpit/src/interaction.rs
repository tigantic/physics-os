/*!
 * Interaction Module - Raycasting for 3D Point Cloud Selection
 * 
 * Converts 2D mouse position → 3D ray → voxel intersection
 * No physics engine bloat - pure vector math at 165Hz
 * 
 * Constitutional: Article V GPU mandate, zero Python dependency
 */

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use winit::dpi::PhysicalPosition;

/// Ray in 3D space
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction: direction.normalize() }
    }
    
    /// Get point along ray at distance t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Interaction state for mouse-based selection
pub struct InteractionState {
    /// Current mouse position (screen space)
    pub mouse_pos: Option<PhysicalPosition<f64>>,
    /// ID of currently hovered node (if any)
    pub hovered_node_id: Option<u32>,
    /// ID of locked/selected node
    pub locked_node_id: Option<u32>,
    /// Is selection locked (click-held)?
    pub is_locked: bool,
    /// Window dimensions for unprojection
    pub window_size: (u32, u32),
}

impl InteractionState {
    pub fn new(window_size: (u32, u32)) -> Self {
        Self {
            mouse_pos: None,
            hovered_node_id: None,
            locked_node_id: None,
            is_locked: false,
            window_size,
        }
    }
    
    /// Update mouse position
    pub fn set_mouse_pos(&mut self, pos: PhysicalPosition<f64>) {
        self.mouse_pos = Some(pos);
    }
    
    /// Clear mouse position (cursor left window)
    pub fn clear_mouse_pos(&mut self) {
        self.mouse_pos = None;
        if !self.is_locked {
            self.hovered_node_id = None;
        }
    }
    
    /// Lock current hover as selection
    pub fn lock_selection(&mut self) {
        if let Some(id) = self.hovered_node_id {
            self.locked_node_id = Some(id);
            self.is_locked = true;
        }
    }
    
    /// Update window size (for resize events)
    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.window_size = new_size;
    }
    
    /// Release selection lock
    pub fn unlock_selection(&mut self) {
        self.is_locked = false;
        // Keep locked_node_id until something else is selected
    }
    
    /// Convert 2D mouse position to 3D ray
    /// 
    /// Uses inverse view-projection matrix to unproject screen coords
    pub fn cast_ray(
        &self,
        view_proj_matrix: &Mat4,
        camera_pos: Vec3,
    ) -> Option<Ray> {
        let pos = self.mouse_pos?;
        let (x, y) = (pos.x, pos.y);
        let width = self.window_size.0 as f32;
        let height = self.window_size.1 as f32;
        
        // 1. Convert to Normalized Device Coordinates (NDC)
        // X: -1 (left) to +1 (right)
        // Y: +1 (top) to -1 (bottom) - flipped for wgpu
        let ndc_x = (2.0 * x as f32) / width - 1.0;
        let ndc_y = 1.0 - (2.0 * y as f32) / height;
        
        // 2. Unproject near and far points
        let inv_view_proj = view_proj_matrix.inverse();
        
        // Near plane (z = -1 in NDC)
        let near_clip = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let near_world = inv_view_proj * near_clip;
        let near_pos = near_world.xyz() / near_world.w;
        
        // Far plane (z = 1 in NDC)
        let far_clip = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let far_world = inv_view_proj * far_clip;
        let far_pos = far_world.xyz() / far_world.w;
        
        // 3. Compute ray direction
        let direction = (far_pos - near_pos).normalize();
        
        Some(Ray {
            origin: camera_pos,
            direction,
        })
    }
}

/// Sphere intersection test for point cloud voxels
/// 
/// Returns distance to intersection, or None if miss
pub fn intersect_sphere(ray: &Ray, center: Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin - center;
    let a = ray.direction.dot(ray.direction);
    let b = 2.0 * oc.dot(ray.direction);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if discriminant < 0.0 {
        None
    } else {
        let t = (-b - discriminant.sqrt()) / (2.0 * a);
        if t > 0.0 {
            Some(t)
        } else {
            // Behind camera
            None
        }
    }
}

/// Axis-Aligned Bounding Box intersection
/// 
/// Fast rejection test before detailed sphere check
pub fn intersect_aabb(ray: &Ray, min: Vec3, max: Vec3) -> Option<f32> {
    let inv_dir = Vec3::new(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z,
    );
    
    let t1 = (min.x - ray.origin.x) * inv_dir.x;
    let t2 = (max.x - ray.origin.x) * inv_dir.x;
    let t3 = (min.y - ray.origin.y) * inv_dir.y;
    let t4 = (max.y - ray.origin.y) * inv_dir.y;
    let t5 = (min.z - ray.origin.z) * inv_dir.z;
    let t6 = (max.z - ray.origin.z) * inv_dir.z;
    
    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));
    
    if tmax < 0.0 || tmin > tmax {
        None
    } else {
        Some(tmin.max(0.0))
    }
}

/// Hit result from raycasting
#[derive(Clone, Copy, Debug)]
pub struct HitResult {
    pub node_id: u32,
    pub distance: f32,
    pub world_pos: Vec3,
    pub normal: Vec3,
}

/// Spatial hash for fast ray-node intersection
/// 
/// Divides space into buckets to avoid O(n) checks
pub struct SpatialHash {
    cell_size: f32,
    // In a full impl, this would be a HashMap<(i32,i32,i32), Vec<u32>>
    // For now, we do simple bounds checking
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self { cell_size }
    }
    
    /// Get cell coordinates for a position
    pub fn cell_coords(&self, pos: Vec3) -> (i32, i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }
}

/// Node data for intersection testing
#[derive(Clone, Copy)]
pub struct InteractableNode {
    pub id: u32,
    pub position: Vec3,
    pub radius: f32,
}

/// Cast ray against a set of nodes, return closest hit
pub fn raycast_nodes(ray: &Ray, nodes: &[InteractableNode]) -> Option<HitResult> {
    let mut closest_dist = f32::MAX;
    let mut closest_hit: Option<HitResult> = None;
    
    for node in nodes {
        // Hitbox slightly larger than visual for Fitts's Law usability
        let hit_radius = node.radius * 1.5;
        
        if let Some(dist) = intersect_sphere(ray, node.position, hit_radius) {
            if dist < closest_dist {
                closest_dist = dist;
                let hit_pos = ray.at(dist);
                let normal = (hit_pos - node.position).normalize();
                
                closest_hit = Some(HitResult {
                    node_id: node.id,
                    distance: dist,
                    world_pos: hit_pos,
                    normal,
                });
            }
        }
    }
    
    closest_hit
}

/// Convert 3D world position (on globe surface) to lat/lon in radians
/// 
/// Phase 8: Used for probe hover detection
/// Assumes globe at origin with given radius
pub fn world_to_geo(pos: Vec3, _radius: f32) -> (f32, f32) {
    // Normalize position to unit sphere
    let n = pos.normalize();
    
    // Latitude: arcsin(y) - range [-π/2, π/2]
    let lat = n.y.asin();
    
    // Longitude: atan2(z, x) - range [-π, π]
    let lon = n.z.atan2(n.x);
    
    (lat, lon)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sphere_intersection() {
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let center = Vec3::new(0.0, 0.0, 5.0);
        let radius = 1.0;
        
        let hit = intersect_sphere(&ray, center, radius);
        assert!(hit.is_some());
        assert!((hit.unwrap() - 4.0).abs() < 0.001);
    }
    
    #[test]
    fn test_sphere_miss() {
        let ray = Ray::new(Vec3::ZERO, Vec3::Z);
        let center = Vec3::new(10.0, 0.0, 5.0); // Off to the side
        let radius = 1.0;
        
        let hit = intersect_sphere(&ray, center, radius);
        assert!(hit.is_none());
    }
}

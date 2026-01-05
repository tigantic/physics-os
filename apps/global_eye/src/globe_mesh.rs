//! Globe Mesh Generation - UV Sphere for Weather Visualization
//!
//! Generates a sphere mesh with UV coordinates for texture mapping.

use bytemuck::{Pod, Zeroable};

/// Vertex format for globe rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GlobeVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl GlobeVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
    ];
    
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlobeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Globe mesh data
pub struct GlobeMesh {
    pub vertices: Vec<GlobeVertex>,
    pub indices: Vec<u32>,
}

impl GlobeMesh {
    /// Generate a UV sphere
    ///
    /// # Arguments
    /// * `radius` - Sphere radius
    /// * `lat_segments` - Number of latitude segments (rings)
    /// * `lon_segments` - Number of longitude segments (slices)
    pub fn new(radius: f32, lat_segments: u32, lon_segments: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        // Generate vertices
        for lat in 0..=lat_segments {
            let theta = std::f32::consts::PI * lat as f32 / lat_segments as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            
            for lon in 0..=lon_segments {
                let phi = 2.0 * std::f32::consts::PI * lon as f32 / lon_segments as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                
                // Position on unit sphere
                let x = sin_theta * cos_phi;
                let y = cos_theta;
                let z = sin_theta * sin_phi;
                
                // UV coordinates
                // U: 0 at lon=0, 1 at lon=lon_segments
                // V: 0 at north pole, 1 at south pole
                let u = lon as f32 / lon_segments as f32;
                let v = lat as f32 / lat_segments as f32;
                
                vertices.push(GlobeVertex {
                    position: [x * radius, y * radius, z * radius],
                    normal: [x, y, z],
                    uv: [u, v],
                });
            }
        }
        
        // Generate indices
        for lat in 0..lat_segments {
            for lon in 0..lon_segments {
                let first = lat * (lon_segments + 1) + lon;
                let second = first + lon_segments + 1;
                
                // Two triangles per quad
                indices.push(first);
                indices.push(second);
                indices.push(first + 1);
                
                indices.push(second);
                indices.push(second + 1);
                indices.push(first + 1);
            }
        }
        
        Self { vertices, indices }
    }
    
    /// Create GPU buffers
    pub fn create_buffers(&self, device: &wgpu::Device) -> (wgpu::Buffer, wgpu::Buffer) {
        use wgpu::util::DeviceExt;
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globe Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globe Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        (vertex_buffer, index_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sphere_generation() {
        let mesh = GlobeMesh::new(1.0, 16, 32);
        
        // Should have (lat_segments + 1) * (lon_segments + 1) vertices
        assert_eq!(mesh.vertices.len(), 17 * 33);
        
        // Should have lat_segments * lon_segments * 2 triangles * 3 indices
        assert_eq!(mesh.indices.len(), 16 * 32 * 6);
    }
}
